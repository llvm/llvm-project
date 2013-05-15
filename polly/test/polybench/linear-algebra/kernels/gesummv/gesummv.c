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
DATA_TYPE y[N];
DATA_TYPE tmp[N];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* y = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* tmp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
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
      x[i] = ((DATA_TYPE) i) / N;
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
	fprintf(stderr, "%0.2lf ", y[i]);
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
#pragma live-out y

  for (i = 0; i < n; i++)
  {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < n; j++)
    {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }

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
