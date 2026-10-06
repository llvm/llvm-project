// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: amdgpu

// Regression test for a cross-team reduction bug where the final xteam
// reduction used a full-wave reduction path even though the kernel was launched
// with fewer threads than the device wave size.  On wave64 AMDGPU targets,
// thread_limit(32) creates a partial wave.  The 63-team case uses the
// single-thread final reduction path; the 64-team case crosses the wave-size
// boundary and must still ignore inactive lanes.

#include <limits.h>
#include <stdio.h>

#define THREAD_LIMIT 32
#define TEAMS_BELOW_WAVE_SIZE_BOUNDARY 63
#define TEAMS_AT_WAVE_SIZE_BOUNDARY 64
#define NUM_ITERS (THREAD_LIMIT * TEAMS_BELOW_WAVE_SIZE_BOUNDARY + 1)
#define EXPECTED_MIN UINT_MAX

static unsigned reduce_min(int teams) {
  unsigned min_val = UINT_MAX;

#pragma omp target teams distribute parallel for num_teams(teams)              \
    thread_limit(THREAD_LIMIT) reduction(min : min_val)
  for (int i = 0; i < NUM_ITERS; ++i) {
    unsigned val = EXPECTED_MIN;
    if (val < min_val)
      min_val = val;
  }

  return min_val;
}

int main(void) {
  unsigned min63 = reduce_min(TEAMS_BELOW_WAVE_SIZE_BOUNDARY);
  unsigned min64 = reduce_min(TEAMS_AT_WAVE_SIZE_BOUNDARY);

  // CHECK: Launching kernel {{.*}} with [63,1,1] blocks and [32,1,1] threads
  // CHECK-SAME: in SPMD mode
  // CHECK: Launching kernel {{.*}} with [64,1,1] blocks and [32,1,1] threads
  // CHECK-SAME: in SPMD mode
  // CHECK: min63 = 0xffffffff
  // CHECK: min64 = 0xffffffff
  printf("min63 = %#x\n", min63);
  printf("min64 = %#x\n", min64);

  return min63 == EXPECTED_MIN && min64 == EXPECTED_MIN ? 0 : 1;
}
