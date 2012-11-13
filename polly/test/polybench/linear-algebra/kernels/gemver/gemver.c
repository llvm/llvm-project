#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef N
# define N 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE alpha;
DATA_TYPE beta;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[N][N];
DATA_TYPE B[N][N];
DATA_TYPE x[N];
DATA_TYPE u1[N];
DATA_TYPE u2[N];
DATA_TYPE v2[N];
DATA_TYPE v1[N];
DATA_TYPE w[N];
DATA_TYPE y[N];
DATA_TYPE z[N];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* u1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* u2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* v1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* v2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* w = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* y = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* z = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < N; ++i)
    {
      A[i] = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
      B[i] = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    }
}
#endif

inline
void init_array()
{
  int i, j;

  alpha = 43532;
  beta = 12313;
  for (i = 0; i < N; i++)
    {
      u1[i] = i;
      u2[i] = (i+1)/N/2.0;
      v1[i] = (i+1)/N/4.0;
      v2[i] = (i+1)/N/6.0;
      y[i] = (i+1)/N/8.0;
      z[i] = (i+1)/N/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < N; j++)
	A[i][j] = ((DATA_TYPE) i*j) / N;
    }
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
      for (i = 0; i < N; i++) {
	fprintf(stderr, "%0.2lf ", w[i]);
	if (i%80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}

#ifndef SCOP_PARAM
void scop_func() {
  long n = N;
#else
void scop_func(long n) {
#endif
  long i, j;

#pragma scop
#pragma live-out w

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}

int main(int argc, char** argv)
{
  int i, j;
  int n = N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#ifndef SCOP_PARAM
  scop_func();
#else
  scop_func(n);
#endif
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
