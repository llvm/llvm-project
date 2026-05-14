// End-to-end test for the divergent VALU expansion of the AMDGPU
// s_bitreplicate intrinsic (lowerBitReplicateToVALU in GlobalISel
// RegBankLegalize).
//
// RUN: %libomptarget-compile-generic \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mllvm=-global-isel \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -mllvm=-new-reg-bank-select
// RUN: %libomptarget-run-generic | %fcheck-generic
//
// REQUIRES: amdgpu

#include <assert.h>
#include <ompx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Host-side reference: replicate each bit of x into two adjacent output bits.
static uint64_t bitreplicate_ref(uint32_t x) {
  uint64_t r = 0;
  for (int i = 0; i < 32; ++i)
    if (x & (1u << i))
      r |= ((uint64_t)0x3) << (2 * i);
  return r;
}

// Call llvm.amdgcn.s.bitreplicate directly via __asm__ symbol rename so we
// don't need a clang builtin. Guard the host side with declare variant.
#pragma omp begin declare variant match(device = {arch(amdgcn)})
uint64_t
__amdgcn_s_bitreplicate(uint32_t) __asm__("llvm.amdgcn.s.bitreplicate");
static uint64_t s_bitreplicate(uint32_t x) {
  return __amdgcn_s_bitreplicate(x);
}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(cpu)})
static uint64_t s_bitreplicate(uint32_t x) { return bitreplicate_ref(x); }
#pragma omp end declare variant

int main(int argc, char *argv[]) {
  const int num_blocks = 1;
  const int block_size = 256;
  const int N = num_blocks * block_size;
  uint64_t *res = (uint64_t *)malloc(N * sizeof(uint64_t));

#pragma omp target teams ompx_bare num_teams(num_blocks)                       \
    thread_limit(block_size) map(from : res[0 : N])
  {
    int bid = ompx_block_id_x();
    int bdim = ompx_block_dim_x();
    int tid = ompx_thread_id_x();
    int idx = bid * bdim + tid;
    res[idx] = s_bitreplicate((uint32_t)idx + 1);
  }
  for (int i = 0; i < N; ++i)
    assert(res[i] == bitreplicate_ref((uint32_t)i + 1));

  // CHECK: PASS
  printf("PASS\n");

  free(res);
  return 0;
}
