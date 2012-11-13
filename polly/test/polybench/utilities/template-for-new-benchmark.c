#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef NX
# define NX 8000
#endif
#ifnef NY
# define NY 8000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[nx][ny];
DATA_TYPE x[ny];
DATA_TYPE y[ny];
DATA_TYPE tmp[nx];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(nx * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc(ny * sizeof(DATA_TYPE));
DATA_TYPE* y = (DATA_TYPE*)malloc(ny * sizeof(DATA_TYPE));
DATA_TYPE* tmp = (DATA_TYPE*)malloc(nx * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < nx; ++i)
    A[i] = (DATA_TYPE*)malloc(ny * sizeof(DATA_TYPE));
}
#endif

inline
void init_array()
{
  int i, j;

  for (i = 0; i < nx; i++)
    {
      x[i] = i * M_PI;
      for (j = 0; j < ny; j++)
	A[i][j] = ((DATA_TYPE) i*j) / nx;
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
      for (i = 0; i < nx; i++) {
	fprintf(stderr, "%0.2lf ", y[i]);
	if (i%80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int i, j;
  int nx = NX;
  int ny = NY;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#pragma scop
#pragma live-out




#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
