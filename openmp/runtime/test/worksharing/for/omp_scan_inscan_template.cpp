// RUN: %libomp-cxx-compile-and-run | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/191549
// Inscan reductions with non-dependent types inside function templates were
// rejected by Clang because the DSA stack was not populated in dependent
// contexts.  This test verifies runtime correctness of both exclusive and
// inclusive scans in template functions.

#include <cstdlib>
#include <numeric>
#include <stdio.h>

#define N 100

template <typename T> bool test_exclusive(T *a, const T *b) {
  int sum = 0;
#pragma omp parallel for reduction(inscan, + : sum)
  for (int i = 0; i < N; i++) {
    a[i] = sum;
#pragma omp scan exclusive(sum)
    sum += b[i];
  }

  for (int i = 0, prefix = 0; i < N; i++) {
    if (a[i] != prefix)
      return false;
    prefix += b[i];
  }
  return true;
}

template <typename T> bool test_inclusive(T *a, const T *b) {
  int sum = 0;
#pragma omp parallel for reduction(inscan, + : sum)
  for (int i = 0; i < N; i++) {
    sum += b[i];
#pragma omp scan inclusive(sum)
    a[i] = sum;
  }

  for (int i = 0, prefix = 0; i < N; i++) {
    prefix += b[i];
    if (a[i] != prefix)
      return false;
  }
  return true;
}

int main() {
  bool success = true;

  int a[N], b[N];
  std::iota(b, b + N, 1);

  if (test_exclusive<int>(a, b)) {
    printf("exclusive: PASS\n");
  } else {
    printf("exclusive: FAIL\n");
    success = false;
  }

  if (test_inclusive<int>(a, b)) {
    printf("inclusive: PASS\n");
  } else {
    printf("inclusive: FAIL\n");
    success = false;
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

// CHECK: exclusive: PASS
// CHECK: inclusive: PASS
