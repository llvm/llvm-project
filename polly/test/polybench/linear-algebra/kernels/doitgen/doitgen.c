#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef NR
# define NR 128
#endif
#ifndef NQ
# define NQ 128
#endif
#ifndef NP
# define NP 128
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[NR][NQ][NP];
DATA_TYPE sum[NR][NQ][NP];
DATA_TYPE C4[NP][NP];
#else
DATA_TYPE*** A = (DATA_TYPE***)malloc(NR * sizeof(DATA_TYPE**));
DATA_TYPE*** sum = (DATA_TYPE***)malloc(NR * sizeof(DATA_TYPE**));
DATA_TYPE** C4 = (DATA_TYPE**)malloc(NP * sizeof(DATA_TYPE*));
{
  int i, j;
  for (i = 0; i < NR; ++i)
    {
      A[i] = (DATA_TYPE**)malloc(NQ * sizeof(DATA_TYPE*));
      sum[i] = (DATA_TYPE**)malloc(NQ * sizeof(DATA_TYPE*));
      for (i = 0; i < NP; ++i)
	{
	  A[i][j] = (DATA_TYPE*)malloc(NP * sizeof(DATA_TYPE));
	  sum[i][j] = (DATA_TYPE*)malloc(NP * sizeof(DATA_TYPE));
	}
    }
  for (i = 0; i < NP; ++i)
    C4[i] = (DATA_TYPE*)malloc(NP * sizeof(DATA_TYPE));
}
#endif

inline
void init_array()
{
  int i, j, k;

  for (i = 0; i < NR; i++)
    for (j = 0; j < NQ; j++)
      for (k = 0; k < NP; k++)
	A[i][j][k] = ((DATA_TYPE) i*j + k) / NP;
  for (i = 0; i < NP; i++)
    for (j = 0; j < NP; j++)
      C4[i][j] = ((DATA_TYPE) i*j) / NP;
}


/* Define the live-out variables. Code is not executed unless
   POLYBENCH_DUMP_ARRAYS is defined. */
inline
void print_array(int argc, char** argv)
{
  int i, j, k;
#ifndef POLYBENCH_DUMP_ARRAYS
  if (argc > 42 && ! strcmp(argv[0], ""))
#endif
    {
      for (i = 0; i < NR; i++)
	for (j = 0; j < NQ; j++)
	  for (k = 0; k < NP; k++) {
	    fprintf(stderr, "%0.2lf ", A[i][j][k]);
	    if ((i * NR + j * NQ + k)% 80 == 20) fprintf(stderr, "\n");
	  }
      fprintf(stderr, "\n");
    }
}

#ifndef SCOP_PARAM
void scop_func() {
  long nr = NR;
  long nq = NQ;
  long np = NP;
#else
void scop_func(long nr, long nq, long np) {
#endif

  long r, q, p, s;
#pragma scop
#pragma live-out A

  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++)  {
      for (p = 0; p < np; p++)  {
        sum[r][q][p] = 0;
        for (s = 0; s < np; s++)
          sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < np; p++)
        A[r][q][p] = sum[r][q][p];
    }


#pragma endscop
}

int main(int argc, char** argv)
{
  int r, q, p, s;
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#ifndef SCOP_PARAM
  scop_func();
#else
  scop_func(nr, nq, np);
#endif


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
