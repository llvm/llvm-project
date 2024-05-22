// RUN: %libomptarget-compileopt-run-and-check-generic
//
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO

#include <math.h>
#include <omp.h>
#include <ompx.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int N = 1 << 29;
  if (argc > 1)
    N = atoi(argv[1]);
  float a = 2.f;

  float *X = (float *)malloc(sizeof(*X) * N);
  float *Y = (float *)malloc(sizeof(*X) * N);

  for (int i = 0; i < N; i++) {
    X[i] = 1.0f;
    Y[i] = 2.0f;
  }

  int TL = 256;
  int NT = (N + TL - 1) / TL;

#pragma omp target data map(to : X [0:N]) map(Y [0:N])
#pragma omp target teams num_teams(NT) thread_limit(TL)
  {
#pragma omp parallel
    {
      int tid = ompx_thread_id_x();
      int bid = ompx_block_id_x();
      int tdim = ompx_block_dim_x();
      int gid = tid + bid * tdim;
      if (gid < N)
        Y[gid] = a * X[gid] + Y[gid];
    }
  }

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(Y[i] - 4.0f));
    if (maxError) {
      printf("%i %f %f\n", i, maxError, Y[i]);
      break;
    }
  }
  // CHECK: Max error: 0.00
  printf("Max error: %f\n", maxError);

  return 0;
}
