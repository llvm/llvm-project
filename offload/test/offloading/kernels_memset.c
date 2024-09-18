// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: gpu

#include <omp.h>

int main(int argc, char *argv[]) {
  const int num_blocks = 64;
  const int block_size = 256;
  const int N = num_blocks * block_size;
  int *data =
      (int *)omp_target_alloc(N * sizeof(int), omp_get_default_device());

  // clang-format off
  // CHECK: Launching kernel __memset_zero with 256 blocks and 256 threads in SPMD mode
  // CHECK: Launching kernel __omp_offloading{{.*}} with 64 blocks and 256 threads in SPMD mode
  omp_target_memset(data, '\0', N * sizeof(int), omp_get_default_device());
  // clang-format on

#pragma omp target teams num_teams(num_blocks) thread_limit(block_size)
  {
#pragma omp parallel
    if (data[omp_get_team_num() * omp_get_num_threads() +
             omp_get_thread_num()] != 0)
      __builtin_trap();
  }

  // clang-format off
  // CHECK: Launching kernel __memset_ones with 256 blocks and 256 threads in SPMD mode
  // CHECK: Launching kernel __omp_offloading{{.*}} with 64 blocks and 256 threads in SPMD mode
  omp_target_memset(data, ~0, N * sizeof(int), omp_get_default_device());
  // clang-format on

#pragma omp target teams num_teams(num_blocks) thread_limit(block_size)
  {
#pragma omp parallel
    if (data[omp_get_team_num() * omp_get_num_threads() +
             omp_get_thread_num()] != ~0)
      __builtin_trap();
  }

  // clang-format off
  // CHECK: Launching kernel __memset with 256 blocks and 256 threads in SPMD mode
  // CHECK: Launching kernel __omp_offloading{{.*}} with 256 blocks and 256 threads in SPMD mode
  omp_target_memset(data, '$', N * sizeof(int), omp_get_default_device());
  // clang-format on

  char *cdata = (char *)data;
#pragma omp target teams num_teams(num_blocks * sizeof(int))                   \
    thread_limit(block_size)
  {
#pragma omp parallel
    if (cdata[omp_get_team_num() * omp_get_num_threads() +
              omp_get_thread_num()] != '$')
      __builtin_trap();
  }

  return 0;
}
