#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef M
# define M 512
#endif
#ifndef N
# define N 512
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE nrm;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[M][N];
DATA_TYPE R[M][N];
DATA_TYPE Q[M][N];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(M * sizeof(DATA_TYPE*));
DATA_TYPE** R = (DATA_TYPE**)malloc(M * sizeof(DATA_TYPE*));
DATA_TYPE** Q = (DATA_TYPE**)malloc(M * sizeof(DATA_TYPE*));
{
  int i;
  for (i = 0; i < M; ++i)
    {
      A[i] = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
      R[i] = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
      Q[i] = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    }
}
#endif

inline
void init_array()
{
  int i, j;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      A[i][j] = ((DATA_TYPE) i*j) / M;
}

/* Define the live-out variables. Code is not executed unless
   POLYBENCH_DUMP_ARRAYS is defined. */
inline
void print_array(int argc, char** argv)
{
  int i, j;
#ifndef POLYBENCH_DUMP_ARRAYS
  if (argc > 42 && ! strcmp(argv[0], ""))
#endif
    {
      for (i = 0; i < M; i++)
	for (j = 0; j < N; j++) {
	  fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	  if ((i * M + j) % 80 == 20) fprintf(stderr, "\n");
	}
      fprintf(stderr, "\n");
    }
}

#ifndef SCOP_PARAM
void scop_func() {
  long m = M;
  long n = N;
#else
void scop_func(long m, long n) {
#endif
  long i, j, k;

#pragma scop
#pragma live-out A

  for (k = 0; k < n; k++)
  {
    nrm = 0;
    for (i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = sqrt(nrm);
    for (i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < n; j++)
    {
      R[k][j] = 0;
      for (i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }

#pragma endscop
}

int main(int argc, char** argv)
{
  int i, j, k;
  int m = M;
  int n = N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#ifndef SCOP_PARAM
  scop_func();
#else
  scop_func(m, n);
#endif

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
