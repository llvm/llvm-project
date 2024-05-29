// RUN: %libomptarget-compilexx-run-and-check-generic
//
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#if defined __AMDGCN_WAVEFRONT_SIZE && __AMDGCN_WAVEFRONT_SIZE == 64
#define MASK 0xaaaaaaaaaaaaaaaa
#else
#define MASK 0xaaaaaaaa
#endif

#include <assert.h>
#include <ompx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  const int num_blocks = 1;
  const int block_size = 256;
  const int N = num_blocks * block_size;
  uint64_t *data = (uint64_t *)malloc(N * sizeof(uint64_t));

  for (int i = 0; i < N; ++i)
    data[i] = i & 0x1;

#pragma omp target teams ompx_bare num_teams(num_blocks) thread_limit(block_size) map(tofrom: data[0:N])
  {
    int tid = ompx_thread_id_x();
    uint64_t mask = ompx_ballot_sync(~0U, data[tid]);
    data[tid] += mask;
  }

  for (int i = 0; i < N; ++i)
    assert(data[i] == ((i & 0x1) + MASK));

  // CHECK: PASS
  printf("PASS\n");

  return 0;
}
