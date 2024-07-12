// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: gpu

#include <assert.h>
#include <ompx.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  const int num_blocks = 64;
  const int block_size = 64;
  const int N = num_blocks * block_size;
  int *data = (int *)malloc(N * sizeof(int));

  // CHECK: "PluginInterface" device 0 info: Launching kernel __omp_offloading_{{.*}} with 64 blocks and 64 threads in SPMD mode

#pragma omp target teams ompx_bare num_teams(num_blocks) thread_limit(block_size) map(from: data[0:N])
  {
    int bid = ompx_block_id_x();
    int bdim = ompx_block_dim_x();
    int tid = ompx_thread_id_x();
    int idx = bid * bdim + tid;
    data[idx] = idx;
  }

  for (int i = 0; i < N; ++i)
    assert(data[i] == i);

  // CHECK: PASS
  printf("PASS\n");

  return 0;
}
