// RUN: %libomptarget-compileopt-run-and-check-generic
//
// REQUIRES: gpu

#include <omp.h>
#include <ompx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct info {
  int tid, bid, tdim;
};

int main(int argc, char **argv) {
  int N = 1 << 20;
  if (argc > 1)
    N = atoi(argv[1]);

  struct info *X = (struct info *)malloc(sizeof(*X) * N);
  memset(X, '0', sizeof(*X) * N);

  int TL = 256;
  int NT = (N + TL - 1) / TL;

#pragma omp target data map(tofrom : X [0:N])
#pragma omp target teams num_teams(NT) thread_limit(TL)
  {
#pragma omp parallel
    {
      int tid = ompx_thread_id_x();
      int bid = ompx_block_id_x();
      int tdim = ompx_block_dim_x();
      int gid = tid + bid * tdim;
      if (gid < N) {
        X[gid].tid = tid;
        X[gid].bid = bid;
        X[gid].tdim = tdim;
      };
    }
  }

  int tid = 0, bid = 0, tdim = 256;
  for (int i = 0; i < N; i++) {
    if (X[i].tid != tid || X[i].bid != bid || X[i].tdim != tdim) {
      printf("%i: %i vs %i, %i vs %i, %i vs %i\n", i, X[i].tid, tid, X[i].bid,
             bid, X[i].tdim, tdim);
      return 1;
    }
    tid++;
    if (tid == tdim) {
      tid = 0;
      bid++;
    }
  }

  // CHECK: OK
  printf("OK");
  return 0;
}
