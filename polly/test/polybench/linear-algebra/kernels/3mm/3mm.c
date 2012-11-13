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
#ifndef NM
# define NM 512
#endif


/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];
DATA_TYPE C[NJ][NM];
DATA_TYPE D[NM][NL];
DATA_TYPE E[NI][NJ];
DATA_TYPE F[NJ][NL];
DATA_TYPE G[NI][NL];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(NK * sizeof(DATA_TYPE*));
DATA_TYPE** C = (DATA_TYPE**)malloc(NJ * sizeof(DATA_TYPE*));
DATA_TYPE** D = (DATA_TYPE**)malloc(NM * sizeof(DATA_TYPE*));
DATA_TYPE** E = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** F = (DATA_TYPE**)malloc(NJ * sizeof(DATA_TYPE*));
DATA_TYPE** G = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
{
  int i;
  for (i = 0; i < NI; ++i)
    {
      A[i] = (DATA_TYPE*)malloc(NK * sizeof(DATA_TYPE));
      E[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
      G[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
    }
  for (i = 0; i < NK; ++i)
    B[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
  for (i = 0; i < NJ; ++i)
    {
      C[i] = (DATA_TYPE*)malloc(NM * sizeof(DATA_TYPE));
      F[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
    }
  for (i = 0; i < NM; ++i)
    D[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
}
#endif


inline
void init_array()
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i][j] = ((DATA_TYPE) i*j)/NI;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i][j] = ((DATA_TYPE) i*j + 1)/NJ;
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NM; j++)
      C[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NM; i++)
    for (j = 0; j < NL; j++)
      D[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      E[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++)
      F[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      G[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
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
	  fprintf(stderr, "%0.2lf ", G[i][j]);
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
  long nm = NM;
#else
void scop_func(long ni, long nj, long nk, long nl, long nm) {
#endif
  long i, j, k;

#pragma scop
#pragma live-out G

  /*   /\* E := A*B *\/ */
  /*   for (i = 0; i < ni; i++) */
  /*     for (j = 0; j < nj; j++) */
  /*       { */
  /* 	E[i][j] = 0; */
  /* 	for (k = 0; k < nk; ++k) */
  /* 	  E[i][j] += A[i][k] * B[k][j]; */
  /*       } */

  /*   /\* F := C*D *\/ */
  /*   for (i = 0; i < nj; i++) */
  /*     for (j = 0; j < nl; j++) */
  /*       { */
  /* 	F[i][j] = 0; */
  /* 	for (k = 0; k < nm; ++k) */
  /* 	  F[i][j] += C[i][k] * D[k][j]; */
  /*       } */
  /*   /\* G := E*F *\/ */
  /*   for (i = 0; i < ni; i++) */
  /*     for (j = 0; j < nl; j++) */
  /*       { */
  /* 	G[i][j] = 0; */
  /* 	for (k = 0; k < nj; ++k) */
  /* 	  G[i][j] += E[i][k] * F[k][j]; */
  /*       } */

  /// FIXME: Remove some parameters, CLooG-ISL crashes...

  /* E := A*B */
  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++)
    {
      E[i][j] = 0;
      for (k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }

    /* F := C*D */
    for (i = 0; i < ni; i++)
      for (j = 0; j < ni; j++)
      {
        F[i][j] = 0;
        for (k = 0; k < nk; ++k)
          F[i][j] += C[i][k] * D[k][j];
      }
      /* G := E*F */
      for (i = 0; i < ni; i++)
        for (j = 0; j < ni; j++)
        {
          G[i][j] = 0;
          for (k = 0; k < nk; ++k)
            G[i][j] += E[i][k] * F[k][j];
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
  int nm = NM;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#ifndef SCOP_PARAM
  scop_func();
#else
  scop_func(ni, nj, nk, nl, nm);
#endif
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
