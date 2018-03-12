#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_eigen.h>

#define vlen(v) (v)->size
#define mrow(m) (m)->size1
#define mcol(m) (m)->size2

typedef struct {
  /* Parameter struct to be passed between function calls */
  int k;                    /* Number of mixture components */
  int d;                    /* Number of data dimensions */

  int i;                    /* Number of iterations */
  int r;                    /* Number of restarts */
  double ll;                /* Number of current log-likelihood */
  gsl_vector *pi;           /* Vector of mixture proportions */
  gsl_matrix *mu;           /* k-by-d matrix of means */
  gsl_matrix **cov;         /* Pointer to k d-by-d covariance matrices */

  gsl_matrix *X;            /* n-by-d data matrix */

} dpar_t;

int printm(gsl_matrix *m, FILE *out)
/* Print a matrix to a file or stdout (if out==NULL) */
{

  if (!out)
    out = stdout;

  int i,j;
  for (i = 0; i < mrow(m); ++i) {
    fprintf(out, "%f", gsl_matrix_get(m, i, 0));
    for (j = 1; j < mcol(m); ++j) {
      fprintf(out, "\t%f", gsl_matrix_get(m, i, j));
    }
    fprintf(out, "\n");
  }
  return 0;
}

int get_dims(FILE *fp, int *r, int *c)
/* Get dimensions of input data */
{
  size_t n = 0;
  char *buf = NULL;

  while(getline(&buf, &n, fp) != -1) {
    if (*r == 0) {
      strtok(buf, " \t");
      *c += 1;
      while(strtok(0, " \t")) {
        *c += 1;
      }
    }
    *r += 1;
  }
  free(buf);
  fseek(fp, 0, SEEK_SET);
  return 0;
}

int read_in(FILE *fp, gsl_matrix *X)
/* Read input data into gsl_matrix */
{
  size_t n = 0;
  char *buf = NULL;

  int i = 0;
  char *token;
  while(getline(&buf, &n, fp) != -1) {
    int j = 0;
    token = strtok(buf, " \t");
    gsl_matrix_set(X, i, j, atof(token));
    ++j;

    while((token = strtok(0, " \t"))) {
      gsl_matrix_set(X, i, j, atof(token));
      ++j;
    }
    ++i;
  }

  free(buf);

  return 0;
}

double vsum(gsl_vector *v)
/* Sum a vector */
{
  double sum = 0;
  int i;
  for(i = 0; i < vlen(v); ++i) {
    sum += gsl_vector_get(v, i);
  }
  return sum;
}

int vnorm(gsl_vector *v)
/* Normalize a vector */
{
  double sum = vsum(v);

  gsl_vector_scale(v, 1/sum);

  return 0;
}

int sim_mvn(FILE *fp, int s, int k, int d)
/* Simulate s data points from a multivariate normal
   mixture with k mixture components in d dimensions */
{

  int i,j,n;
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, time(NULL)); // RNG seed

  gsl_vector *pi = gsl_vector_alloc(k);
  for (i = 0; i < k; ++i) {
    gsl_vector_set(pi, i, gsl_rng_uniform_pos(r));
  }
  vnorm(pi);

  for (i = 0; i < k; ++i) {
    gsl_vector *mu = gsl_vector_alloc(d);
    gsl_matrix *cov = gsl_matrix_alloc(d, d);

    gsl_matrix_set_all(cov, gsl_rng_uniform_pos(r));
    gsl_vector_view v = gsl_matrix_diagonal(cov);
    for (j = 0; j < d; ++j) {
      gsl_vector_set(mu, j, gsl_rng_uniform_int(r, 19) + 1);
      gsl_vector_set(&v.vector, j, gsl_rng_uniform_int(r, 3) + 1);
    }

    // Print simulation parameters
    printf("%f\n-\n", gsl_vector_get(pi, i));
    printf("%f", gsl_vector_get(mu, 0));
    for (j = 1; j < d; ++j) {
      printf("\t%f", gsl_vector_get(mu, j));
    }
    printf("\n");
    //    printf("-\n");
    //    printm(cov, NULL);
    printf("-----------\n");

    // Simulate and write to file
    gsl_linalg_cholesky_decomp1(cov);
    gsl_vector *res = gsl_vector_alloc(d);
    for (n = 0; n < s*gsl_vector_get(pi, i); ++n) {
      gsl_ran_multivariate_gaussian(r, mu, cov, res);
      fprintf(fp, "%f", gsl_vector_get(res, 0));
      for (j = 1; j < d; ++j) {
        fprintf(fp, "\t%f", gsl_vector_get(res, j));
      }
      fprintf(fp, "\n");
    }

    gsl_vector_free(res);
    gsl_vector_free(mu);
    gsl_matrix_free(cov);
  }

  gsl_vector_free(pi);
  gsl_rng_free(r);
  return 0;
}

int init_mu(dpar_t *par)
/* Initialize the mean matrix as follows:
   - First, the dimension with the highest variance is selected
   - The selected dimension is sorted and a sorted permutation is generated
   - Successively, the permutation is applied to each dimension
   - The permuted dimension is split into k partitions
   - The mean and standard deviation of each partition is calculated
   - The mean is initialized based on a Gaussian with mean and standard
     deviation of the partition
 */
{
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, time(NULL));

  gsl_vector *var = gsl_vector_calloc(par->d);
  int i;
  for (i = 0; i < par->d; ++i) {
    gsl_vector_view d = gsl_matrix_column(par->X, i);
    gsl_vector_set(var, i, gsl_stats_variance((&d.vector)->data, (&d.vector)->stride, (&d.vector)->size));
  }
  int s = gsl_vector_max_index(var);
  gsl_vector_free(var);

  gsl_vector_view h = gsl_matrix_column(par->X, s);
  gsl_permutation *perm = gsl_permutation_alloc(vlen(&h.vector));
  gsl_sort_vector_index(perm, &h.vector);


  for (i = 0; i < par->d; ++i) {

    gsl_vector_view d = gsl_matrix_column(par->X, i);
    gsl_vector *x = gsl_vector_alloc(vlen(&d.vector));
    gsl_vector_memcpy(x, &d.vector);

    gsl_permute_vector(perm, x);

    gsl_vector_view m = gsl_matrix_column(par->mu, i);

    int j;
    for(j = 0; j < vlen(&m.vector); ++j) {
      gsl_vector_view s = gsl_vector_subvector(x, j * (vlen(x)/vlen(&m.vector)), vlen(x)/vlen(&m.vector));
      double mean = gsl_stats_mean((&s.vector)->data, (&s.vector)->stride, (&s.vector)->size);
      double sd   = gsl_stats_sd_m((&s.vector)->data, (&s.vector)->stride, (&s.vector)->size, mean);

      gsl_vector_set(&m.vector, j, mean + gsl_ran_gaussian(r, sd));
    }

    gsl_vector_free(x);
  }

  gsl_rng_free(r);
  gsl_permutation_free(perm);
  return 0;
}

int init_covar(gsl_matrix *m)
/* Initialize each covariance matrix to values
   0.1 at off-diagonal and 2.1 on the diagonal */
{
  gsl_matrix_set_all(m, 0.1);

  gsl_vector_view s = gsl_matrix_diagonal(m);
  gsl_vector_add_constant(&s.vector, 2.0);

  return 0;
}

int setup_dpar(dpar_t *par, gsl_matrix *X, int c, int k)
/* Set up the dpar_t struct for initialization */
{

  par->k = k;
  par->d = c;
  par->X = X;

  par->ll = 0.0;
  par->i = 0;
  par->r = 0;

  par->pi = gsl_vector_calloc(k);
  par->mu = gsl_matrix_calloc(k, c);

  par->cov = malloc(k * sizeof *par->cov);
  int i;
  for(i = 0; i < k; ++i) {
    par->cov[i] = gsl_matrix_calloc(c, c);
  }

  return 0;
}

int init_dpar(dpar_t *par)
/* Initialize the dpar_t struct */
{

  gsl_vector_set_all(par->pi, 1.0);
  vnorm(par->pi);

  init_mu(par);

  int z;
  for (z = 0; z < par->k; ++z) {
    init_covar(par->cov[z]);
  }

  return 0;
}

int free_dpar(dpar_t *par)
/* Free the dpar_t struct */
{

  gsl_matrix_free(par->X);
  gsl_vector_free(par->pi);
  gsl_matrix_free(par->mu);

  int i;
  for(i = 0; i < par->k; ++i) {
    gsl_matrix_free(par->cov[i]);
  }
  free(par->cov);

  return 0;
}

int pos_def(gsl_matrix *M)
/* Test for positive definite matrix - returns -1 on failure, 0 on success
   For this, the eigenvalues of the covariance matrices are tested to be no
   smaller than some threshold close to zero. */
{
  int i;
  int ret = 0;
  gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(mrow(M));
  gsl_vector *eigen = gsl_vector_alloc(mrow(M));
  gsl_matrix *Cov = gsl_matrix_alloc(mrow(M), mcol(M));
  gsl_matrix_memcpy(Cov, M);

  gsl_eigen_symm(Cov, eigen, w);

  for(i = 0; i < mrow(M); ++i) {
    if(gsl_vector_get(eigen, i) <= 1e-10) {
      ret = -1;
    }
  }

  gsl_vector_free(eigen);
  gsl_matrix_free(Cov);
  gsl_eigen_symm_free(w);
  return ret;
}

gsl_matrix **get_decomp(dpar_t *par)
/* Get cholesky decompositions of covariance matrices
   needed for multivariate normal density in GSL */
{
  gsl_matrix **L;
  L = malloc(par->k * sizeof *L);
  int z;
  for (z = 0; z < par->k; ++z) {
    L[z] = gsl_matrix_calloc(par->d, par->d);
    gsl_matrix_memcpy(L[z], par->cov[z]);
    gsl_linalg_cholesky_decomp1(L[z]);
  }

  return L;
}

int free_matarr(gsl_matrix **L, int k)
/* Free matrix array from cholesky decompositions of covariance matrices */
{
  int z;
  for (z = 0; z < k; ++z) {
    gsl_matrix_free(L[z]);
  }
  free(L);

  return 0;
}

int tsubsq(gsl_matrix *res, const gsl_vector *d, const gsl_vector *m)
/* Calculate (X[i,] - mu[j,]) %*% t(X[i,] - mu[j,]) needed during
   covariance calculation. Name stands for 'transpose substraction squared' */
{
  gsl_matrix *M = gsl_matrix_calloc(mrow(res), 1);

  gsl_vector_view s = gsl_matrix_column(M, 0);
  gsl_vector_memcpy(&s.vector, d);
  gsl_vector_sub(&s.vector, m);

  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
                 M, M, 0.0, res);

  gsl_matrix_free(M);
  return 0;
}

double get_det(gsl_matrix *m)
/* Get the determinant of a matrix using LU decomposition */
{
  double det;
  int signum;
  gsl_permutation *p = gsl_permutation_alloc(mrow(m));
  gsl_matrix *tmp = gsl_matrix_alloc(mrow(m), mcol(m));
  gsl_matrix_memcpy(tmp, m);

  gsl_linalg_LU_decomp(tmp, p, &signum);
  det = gsl_linalg_LU_det(tmp, signum);

  gsl_permutation_free(p);
  gsl_matrix_free(tmp);

  return det;
}

int get_E(dpar_t *par, gsl_matrix *E)
/* Calculate the 'E'-Matrix, which is the datastructure from the
   E step of the EM. It's a n-by-k matrix, where each row corresponds
   to the probabilities of cluster assignment of the datapoint in
   that row. */
{

  gsl_matrix **L = get_decomp(par);
  gsl_vector *w = gsl_vector_calloc(par->d);

  int i,j;
  for (i = 0; i < mrow(E); ++i) {
    for (j = 0; j < mcol(E); ++j) {

      gsl_vector_view d = gsl_matrix_row(par->X, i);
      gsl_vector_view m = gsl_matrix_row(par->mu, j);
      double res;

      gsl_ran_multivariate_gaussian_pdf(&d.vector, &m.vector, L[j], &res, w);

      gsl_matrix_set(E, i, j, gsl_vector_get(par->pi, j) * res);
    }
    gsl_vector_view s = gsl_matrix_row(E, i);
    vnorm(&s.vector);
  }

  gsl_vector_free(w);
  free_matarr(L, par->k);

  return 0;
}

int iter_em(dpar_t *par)
/* One iteration of the EM. During the M-step, the values
   for mixture proportions, means and covarances are updated. */
{
  int i,j;
  int ret = 0;

  //// E-step of the EM
  gsl_matrix *E = gsl_matrix_calloc(mrow(par->X), par->k);
  get_E(par, E);


  //// M-step of the EM
  // update mixture proportion and store colSums(E)
  gsl_vector *cS = gsl_vector_calloc(mcol(E));
  for (j = 0; j < mcol(E); ++j) {
    gsl_vector_view c = gsl_matrix_column(E, j);
    double sum = vsum(&c.vector);

    gsl_vector_set(cS, j, sum);

    double mix = sum / mrow(E);
    if (gsl_isnan(mix)) {
      /* Some starting values lead to empty clusters, which generate nan values.
         In that case, we set ret to -1 to restart the EM with new starting values. */
      ret = -1;
    }
    gsl_vector_set(par->pi, j, mix);
  }

  // update means
  gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                 1.0, E, par->X,
                 0.0, par->mu);

  for (j = 0; j < mrow(par->mu); ++j) {
    gsl_vector_view r = gsl_matrix_row(par->mu, j);
    gsl_vector_scale(&r.vector, 1.0/gsl_vector_get(cS, j));
  }

  gsl_vector_free(cS);

  /* Update covariance structure
     This follows the EVV model from mclust5, described in
     -    Scrucca et al., The R journal 8.1 (2016)    - and
     - Celeux and Covaert, Pattern Recognition (1995) -
   */
  // calculate Wk
  gsl_matrix **W;
  W = malloc(par->k * sizeof *W);
  for (j = 0; j < par->k; ++j) {
    W[j] = gsl_matrix_calloc(par->d, par->d);
  }
  gsl_matrix *tmp = gsl_matrix_calloc(par->d, par->d);

  for (i = 0; i < mrow(par->X); ++i) {
    for (j = 0; j < mcol(E); ++j) {
      gsl_vector_view d = gsl_matrix_row(par->X, i);
      gsl_vector_view m = gsl_matrix_row(par->mu, j);

      tsubsq(tmp, &d.vector, &m.vector);
      gsl_matrix_scale(tmp, gsl_matrix_get(E, i, j));

      gsl_matrix_add(W[j], tmp);
    }
  }

  gsl_matrix_free(tmp);
  gsl_matrix_free(E);

  gsl_vector *wd = gsl_vector_alloc(par->k);

  // calculate determinant and scale Wk to Ck
  for (j = 0; j < par->k; ++j) {
    double den = pow(get_det(W[j]), 1.0/par->d);
    gsl_vector_set(wd, j, den);
    gsl_matrix_scale(W[j], 1.0/den);
  }

  // calculate lambda (same for all mixture components)
  double l = vsum(wd) / mrow(par->X);

  // lambda * Ck
  for (j = 0; j < par->k; ++j) {
    gsl_matrix_scale(W[j], l);
    gsl_matrix_memcpy(par->cov[j], W[j]);

    if(pos_def(par->cov[j]) == -1) {
      /* Some starting values lead to positive semi-definite covariance matrices.
         In that case, we set ret to -1 to restart the EM with new starting values. */
      ret = -1;
    }
  }

  gsl_vector_free(wd);
  free_matarr(W, par->k);
  return ret;
}

double calcll(dpar_t *par)
/* Calculate current log-likelihood */
{
  double ll = 0.0;
  double tmp;

  gsl_matrix **L = get_decomp(par);
  gsl_vector *w = gsl_vector_calloc(par->d);

  int i,j;
  for (i = 0; i < mrow(par->X); ++i) {
    double tsum = 0.0;
    for (j = 0; j < vlen(par->pi); ++j) {

      gsl_vector_view d = gsl_matrix_row(par->X, i);
      gsl_vector_view m = gsl_matrix_row(par->mu, j);

      gsl_ran_multivariate_gaussian_pdf(&d.vector, &m.vector, L[j], &tmp, w);
      tsum += gsl_vector_get(par->pi, j) * tmp;
    }
    ll += tsum > DBL_MIN ? log(tsum) : log(DBL_MIN);
  }

  gsl_vector_free(w);
  free_matarr(L, par->k);

  return ll;
}

int get_df(dpar_t *par)
/* Get the number of freely estimated parameters for BIC calculation
   The number is the sum of:
       k - 1, since the mixture proportions sum to one
       k * d, since the matrix of means is estimated freely
       k * nCovar, there are k covariance matrices with nCovar free parameters
           nCovar is the number of diagonal elements plus the number of
           one half of the off-diagonal elements, since matrix is symmetric.
           This is calulated as d(d+1)/2.
           Additionally, (k - 1) has to be substracted as all covariance matrices
           share the same lambda in the EVV model. */
{
  int df;

  df = (par->k - 1) + (par->d * par->k) + par->k * ((par->d * (par->d + 1))/2) - (par->k - 1);

  return df;
}

double get_BIC(dpar_t *par)
/* Calculate Bayesian Information Criterion */
{
  double BIC;

  BIC = log((double)mrow(par->X)) * get_df(par) - 2 * par->ll;

  return BIC;
}

int print_par(dpar_t *par, int flag, FILE *out)
/* Print the parameter struct, either to a File or to
   stdout (if out==NULL). The 'flag' parameter controls
   whether the covariance matrices are printed as well. */
{

  if (!out)
    out = stdout;

  fprintf(out, "    iter: %d  \t  ll: %f  \t  n: %d\n", par->i, par->ll, (int)mrow(par->X));
  fprintf(out, "restarts: %d  \t BIC: %f  \t df: %d\n", par->r, get_BIC(par), get_df(par));
  fprintf(out, "proportions:\n");
  gsl_vector_fprintf(stdout, par->pi, "%f");
  fprintf(out, "means:\n");
  printm(par->mu, out);

  if (flag) {
    fprintf(out, "covariance matrices:");
    int z;
    for (z = 0; z < par->k; ++z) {
      fprintf(out, "\n");
      printm(par->cov[z], out);
    }
  }
  fprintf(out, "----------------\n");

  return 0;
}

int run_em(dpar_t *par, double delta, int verbose)
/* The driver of the EM. Checks convergence, restarts
   if neccessary, and prints during iterations depending
   on verbosity. */
{

  int ret = 0;
  double ll = calcll(par);
  double last = ll;
  double d = 10000.00;

  while(d > delta) {

    ret = iter_em(par);
    if (ret == -1) {
      int i = par->i;
      int r = par->r;
      init_dpar(par);
      par->i = i;
      par->r = r + 1;
    }

    ll = calcll(par);
    d = fabs(last - ll);
    last = ll;

    par->ll = ll;
    ++par->i;

    if (verbose == 1 && par->i % 10 == 0) {
      printf("    iter: %d  \t  ll: %f  \t  n: %d\n", par->i, par->ll, (int)mrow(par->X));
      printf("restarts: %d  \t BIC: %f  \t df: %d\n\n", par->r, get_BIC(par), get_df(par));
    }
    else if (verbose == 2) {
    print_par(par, 1, NULL);
    }

  }

  return 0;
}

int get_assign(dpar_t *par, gsl_matrix *E, FILE *out)
/* Get the cluster assignments for all data points, as well as
   their uncertainty. */
{
  int i;

  if (!out)
    out = stdout;

  for (i = 0; i < mrow(E); ++i) {
    gsl_vector_view s = gsl_matrix_row(E, i);
    fprintf(out, "%d\t%f\n",
            1 + (int)gsl_vector_max_index(&s.vector),
            1.0 - gsl_vector_max(&s.vector));
  }

  return 0;
}

int usage(int r, char **argv)
{
  printf("\nUsage: %s [options] file.tsv\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-k\tNumber of clusters (default: k = 2)\n\n");
  printf("\t-a\tFile name for cluster assignment results (optional)\n\n");
  printf("\t-s\tSimulate -s samples from a -d dimensional mixture of -k clusters (triggers simulation over EM)\n");
  printf("\t-d\tNumber of dimensions for simulation (only useful with -s)\n\n");
  printf("\t-v\tSet verbosity - {0, 1, 2} (default 0)\n\n");

  return r;
}

int main(int argc, char **argv)
{

  int elem;
  int k = 2;
  int verbose = 0;
  int s = 0;
  int d = 0;
  char *assfn = NULL;

  while (( elem = getopt(argc, argv, "k:s:d:a:v:") ) >= 0) {
    switch(elem) {
    case 'k': k = atoi(optarg); break;
    case 's': s = atoi(optarg); break;
    case 'd': d = atoi(optarg); break;
    case 'a': assfn = optarg; break;
    case 'v': verbose = atoi(optarg); break;
    }
  }

  if (argc - optind != 1)
    return usage(1, argv);

  if (s) {

    if (s < k || !d)
      return 5;

    FILE *fp = fopen(argv[optind], "w");
    if (!fp)
      return 2;

    sim_mvn(fp, s, k, d);

    fclose(fp);
    return 0;
  }

  FILE *fp = fopen(argv[optind], "r");
  if (!fp)
    return 2;

  int ret;
  int r = 0;
  int c = 0;
  get_dims(fp, &r, &c);

  gsl_matrix *X = NULL;
  X = gsl_matrix_calloc(r, c);

  if (X == NULL)
    return 3;

  ret = read_in(fp, X);
  //  gsl_matrix_fscanf(fp, X); // Alternative from API
  if (ret != 0)
    return 4;
  fclose(fp);

  dpar_t par;

  setup_dpar(&par, X, c, k);

  init_dpar(&par);

  //  print_par(&par, 0, NULL);

  run_em(&par, 1e-4, verbose);

  print_par(&par, 0, NULL);

  if (assfn) {
    FILE *assf = fopen(assfn, "w");
    gsl_matrix *E = gsl_matrix_calloc(mrow(par.X), par.k);
    get_E(&par, E);
    get_assign(&par, E, assf);
    gsl_matrix_free(E);
    fclose(assf);
  }

  free_dpar(&par);

  return 0;
}
