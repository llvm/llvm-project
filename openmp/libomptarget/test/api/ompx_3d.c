// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <ompx.h>
#include <stdio.h>

void foo(int device) {
  int tid = 0, bid = 0, bdim = 0;
#pragma omp target teams distribute parallel for map(from                      \
                                                     : tid, bid, bdim)         \
    device(device) thread_limit(2) num_teams(5)
  for (int i = 0; i < 1000; ++i) {
    if (i == 42) {
      tid = ompx_block_dim_x();
      bid = ompx_block_id_x();
      bdim = ompx_grid_dim_x();
    }
  }
  // CHECK: tid: 2, bid: 1, bdim: 5
  // CHECK: tid: 2, bid: 0, bdim: 1
  printf("tid: %i, bid: %i, bdim: %i\n", tid, bid, bdim);
}

int isGPU() { return 0; }
#pragma omp declare variant(isGPU) match(device = {arch(gpu)})
int isGPUvariant() { return 1; }

int defaultIsGPU() {
  int r = 0;
#pragma omp target map(from : r)
  r = isGPU();
  return r;
}

int main() {
  if (defaultIsGPU())
    foo(omp_get_default_device());
  else
    printf("tid: 2, bid: 1, bdim: 5\n");
  foo(omp_get_initial_device());
}
