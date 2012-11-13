#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef NI
# define NI 512
#endif
#ifndef NJ
# define NJ 512
#endif
#ifndef NK
# define NK 512
#endif
#ifndef NL
# define NL 512
#endif


/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE alpha1;
DATA_TYPE beta1;
DATA_TYPE alpha2;
DATA_TYPE beta2;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE C[NI][NJ];
DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];
DATA_TYPE D[NJ][NL];
DATA_TYPE E[NI][NL];
#else
DATA_TYPE** C = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** A = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(NK * sizeof(DATA_TYPE*));
DATA_TYPE** D = (DATA_TYPE**)malloc(NJ * sizeof(DATA_TYPE*));
DATA_TYPE** E = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
{
  int i;
  for (i = 0; i < NI; ++i)
    {
      C[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
      A[i] = (DATA_TYPE*)malloc(NK * sizeof(DATA_TYPE));
      E[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
    }
  for (i = 0; i < NK; ++i)
    B[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
  for (i = 0; i < NJ; ++i)
    D[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
}
#endif


inline
void init_array()
{
  int i, j;

  alpha1 = 32412;
  beta1 = 2123;
  alpha2 = 132412;
  beta2 = 92123;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i][j] = ((DATA_TYPE) i*j)/NI;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i][j] = ((DATA_TYPE) i*j + 1)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++)
      D[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      E[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
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
      for (i = 0; i < NI; i++) {
	for (j = 0; j < NL; j++) {
	  fprintf(stderr, "%0.2lf ", E[i][j]);
	  if ((i * NI + j) % 80 == 20) fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
      }
    }
}

#ifndef SCOP_PARAM
void scop_func() {
  long ni = NI;
  long nj = NJ;
  long nk = NK;
  long nl = NL;
#else
void scop_func(long ni, long nj, long nk, long nl) {
#endif
  long i, j, k;
#pragma scop
#pragma live-out E

  /* E := A*B*C */
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      C[i][j] = 0;
      for (k = 0; k < nk; ++k)
        C[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
    {
      E[i][j] = 0;
      for (k = 0; k < nj; ++k)
        E[i][j] += C[i][k] * D[k][j];
    }


#pragma endscop
}

int main(int argc, char** argv)
{
  int i, j, k;
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#ifndef SCOP_PARAM
  scop_func();
#else
  scop_func(ni, nj, nk, nl);
#endif

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
