// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// REQUIRES: gpu
// UNSUPPORTED: intelgpu

// Regression test for a cross-team reduction bug where the final xteam
// reduction used a full-wave reduction path even though the kernel was launched
// with fewer threads than the device wave size.  On wave64 AMDGPU targets,
// thread_limit(32) creates a partial wave.  The 63-team case uses the
// single-thread final reduction path; the 64-team case crosses the wave-size
// boundary and must still ignore inactive lanes.

#include <limits.h>
#include <stdio.h>

static unsigned reduce_min(int teams, int seed) {
  unsigned min_val = UINT_MAX;

#pragma omp target teams distribute parallel for num_teams(teams)              \
    thread_limit(32) map(to : seed) reduction(min : min_val)
  for (int i = 0; i < 2017; ++i) {
    unsigned val = 0xdeadbeefU + ((i + seed) & 1);
    if (val < min_val)
      min_val = val;
  }

  return min_val;
}

int main(int argc, char **argv) {
  unsigned min63 = reduce_min(63, argc);
  unsigned min64 = reduce_min(64, argc);

  // CHECK: min63 = 0xdeadbeef
  // CHECK: min64 = 0xdeadbeef
  printf("min63 = %#x\n", min63);
  printf("min64 = %#x\n", min64);

  return min63 == 0xdeadbeefU && min64 == 0xdeadbeefU ? 0 : 1;
}
