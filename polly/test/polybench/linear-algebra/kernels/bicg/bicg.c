#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"

/* Default problem size. */
#ifndef NX
# define NX 8000
#endif
#ifndef NY
# define NY 8000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[NX][NY];
DATA_TYPE r[NX];
DATA_TYPE s[NX];
DATA_TYPE p[NX];
DATA_TYPE q[NX];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(NX * sizeof(DATA_TYPE*));
DATA_TYPE* r = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
DATA_TYPE* s = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
DATA_TYPE* p = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
DATA_TYPE* q = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < NX; ++i)
    A[i] = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
}
#endif

inline
void init_array()
{
    int i, j;

    for (i = 0; i < NX; i++) {
        r[i] = i * M_PI;
        p[i] = i * M_PI;
        for (j = 0; j < NY; j++) {
            A[i][j] = ((DATA_TYPE) i*j)/NX;
        }
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
	for (i = 0; i < NX; i++) {
	  fprintf(stderr, "%0.2lf ", s[i]);
	  fprintf(stderr, "%0.2lf ", q[i]);
	  if (i%80 == 20) fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
      }
}

#ifndef SCOP_PARAM
void scop_func() {
  long nx = NX;
  long ny = NY;
#else
void scop_func(long nx, long ny) {
#endif
  long i, j;

#pragma scop
#pragma live-out s, q

  for (i = 0; i < ny; i++)
    s[i] = 0;

  for (i = 0; i < nx; i++)
  {
    q[i] = 0;
    for (j = 0; j < ny; j++)
    {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }

#pragma endscop
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
#ifndef SCOP_PARAM
    scop_func();
#else
    scop_func(nx, ny);
#endif

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    print_array(argc, argv);

    return 0;
}
