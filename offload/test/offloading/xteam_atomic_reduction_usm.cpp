// Validate the atomic cross-team reduction fast path under unified shared
// memory.
//
// RUN: %libomptarget-compilexx-generic -fopenmp-force-usm \
// RUN:   -fopenmp-target-fast-reduction && env HSA_XNACK=1 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileoptxx-generic -fopenmp-force-usm \
// RUN:   -fopenmp-target-fast-reduction && env HSA_XNACK=1 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
//
// REQUIRES: amdgpu, unified_shared_memory

#include <cassert>
#include <climits>
#include <cstdio>
#include <omp.h>

int main() {
  const int N = 1 << 16;

  long *sum = (long *)omp_alloc(sizeof(long), llvm_omp_target_shared_mem_alloc);
  *sum = 7;
#pragma omp target teams distribute parallel for reduction(+ : sum[0])
  for (int i = 0; i < N; ++i)
    sum[0] += i;
  assert(sum[0] == 7 + (long)(N - 1) * N / 2 && "atomic + reduction incorrect");
  omp_free(sum, llvm_omp_target_shared_mem_alloc);

  double *fsum =
      (double *)omp_alloc(sizeof(double), llvm_omp_target_shared_mem_alloc);
  *fsum = 0.5;
#pragma omp target teams distribute parallel for reduction(+ : fsum[0])
  for (int i = 0; i < N; ++i)
    fsum[0] += 1.0;
  assert(fsum[0] == 0.5 + (double)N && "atomic fp + reduction incorrect");
  omp_free(fsum, llvm_omp_target_shared_mem_alloc);

  int *mx = (int *)omp_alloc(sizeof(int), llvm_omp_target_shared_mem_alloc);
  int *mn = (int *)omp_alloc(sizeof(int), llvm_omp_target_shared_mem_alloc);
  *mx = INT_MIN;
  *mn = INT_MAX;
#pragma omp target teams distribute parallel for reduction(max : mx[0])        \
    reduction(min : mn[0])
  for (int i = 0; i < N; ++i) {
    mx[0] = i > mx[0] ? i : mx[0];
    mn[0] = i < mn[0] ? i : mn[0];
  }
  assert(mx[0] == N - 1 && mn[0] == 0 && "atomic min/max reduction incorrect");
  omp_free(mx, llvm_omp_target_shared_mem_alloc);
  omp_free(mn, llvm_omp_target_shared_mem_alloc);

  unsigned *bits =
      (unsigned *)omp_alloc(sizeof(unsigned), llvm_omp_target_shared_mem_alloc);
  *bits = 0;
#pragma omp target teams distribute parallel for reduction(| : bits[0])
  for (int i = 0; i < N; ++i)
    bits[0] |= (unsigned)i;
  assert(bits[0] == (unsigned)(N - 1) && "atomic | reduction incorrect");
  omp_free(bits, llvm_omp_target_shared_mem_alloc);

  printf("SUCCESS\n");
  // CHECK: SUCCESS
  return 0;
}
