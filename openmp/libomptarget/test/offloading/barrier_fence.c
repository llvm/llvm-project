// RUN: %libomptarget-compile-generic -fopenmp-offload-mandatory -O3
// RUN: %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-offload-mandatory -O3
// RUN: %libomptarget-run-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

struct IdentTy;
void __kmpc_barrier_simple_spmd(struct IdentTy *Loc, int32_t TId);
void __kmpc_barrier_simple_generic(struct IdentTy *Loc, int32_t TId);

#pragma omp begin declare target device_type(nohost)
static int A[512] __attribute__((address_space(3), loader_uninitialized));
static int B[512 * 32] __attribute__((loader_uninitialized));
#pragma omp end declare target

int main() {
  printf("Testing simple spmd barrier\n");
  for (int r = 0; r < 50; r++) {
#pragma omp target teams distribute thread_limit(512) num_teams(440)
    for (int j = 0; j < 512 * 32; ++j) {
#pragma omp parallel firstprivate(j)
      {
        int TId = omp_get_thread_num();
        int TeamId = omp_get_team_num();
        int NT = omp_get_num_threads();
        // Sequential
        for (int i = 0; i < NT; ++i) {
          // Test shared memory globals
          if (TId == i)
            A[i] = i + j;
          __kmpc_barrier_simple_spmd(0, TId);
          if (A[i] != i + j)
            __builtin_trap();
          __kmpc_barrier_simple_spmd(0, TId);
          // Test generic globals
          if (TId == i)
            B[TeamId] = i;
          __kmpc_barrier_simple_spmd(0, TId);
          if (B[TeamId] != i)
            __builtin_trap();
          __kmpc_barrier_simple_spmd(0, TId);
        }
      }
    }
  }

  printf("Testing simple generic barrier\n");
  for (int r = 0; r < 50; r++) {
#pragma omp target teams distribute thread_limit(512) num_teams(440)
    for (int j = 0; j < 512 * 32; ++j) {
#pragma omp parallel firstprivate(j)
      {
        int TId = omp_get_thread_num();
        int TeamId = omp_get_team_num();
        int NT = omp_get_num_threads();
        // Sequential
        for (int i = 0; i < NT; ++i) {
          if (TId == i)
            A[i] = i + j;
          __kmpc_barrier_simple_generic(0, TId);
          if (A[i] != i + j)
            __builtin_trap();
          __kmpc_barrier_simple_generic(0, TId);
          if (TId == i)
            B[TeamId] = i;
          __kmpc_barrier_simple_generic(0, TId);
          if (B[TeamId] != i)
            __builtin_trap();
          __kmpc_barrier_simple_generic(0, TId);
        }
      }
    }
  }
  return 0;
}
