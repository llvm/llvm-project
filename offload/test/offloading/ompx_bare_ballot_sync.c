// RUN: %libomptarget-compilexx-run-and-check-generic
//
// REQUIRES: gpu

#include <assert.h>
#include <ompx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#pragma omp begin declare variant match(device = {arch(amdgcn)})
unsigned get_warp_size() { return __builtin_amdgcn_wavefrontsize(); }
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {arch(nvptx64)})
unsigned get_warp_size() { return __nvvm_read_ptx_sreg_warpsize(); }
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
unsigned get_warp_size() { return 1; }
#pragma omp end declare variant

int main(int argc, char *argv[]) {
  const int num_blocks = 1;
  const int block_size = 256;
  const int N = num_blocks * block_size;
  int *res = (int *)malloc(N * sizeof(int));

#pragma omp target teams ompx_bare num_teams(num_blocks)                       \
    thread_limit(block_size) map(from : res[0 : N])
  {
    int tid = ompx_thread_id_x();
    uint64_t mask = ompx_ballot_sync(~0LU, tid & 0x1);
    if (get_warp_size() == 64)
      res[tid] = mask == 0xaaaaaaaaaaaaaaaa;
    else
      res[tid] = mask == 0xaaaaaaaa;
  }

  for (int i = 0; i < N; ++i)
    assert(res[i]);

  // CHECK: PASS
  printf("PASS\n");

  free(res);

  return 0;
}
