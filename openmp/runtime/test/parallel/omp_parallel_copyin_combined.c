// RUN: %libomp-compile-and-run
#include "omp_testsuite.h"

#define N 100

int x1, x2, x3, x4, x5;
#pragma omp threadprivate(x1, x2, x3, x4, x5)

int test_omp_parallel_copyin() {
  int a[N];
  x1 = 1;

#pragma omp parallel copyin(x1)
#pragma omp for
  for (int i = 0; i < N; i++)
    a[i] = i + x1;

  int sum = 0;

  for (int i = 0; i < N; i++)
    sum += a[i];

  return (sum == ((99 + 2 * x1) * 100) / 2);
}

int test_omp_parallel_for_copyin() {
  int a[N];
  x2 = 2;

#pragma omp parallel for copyin(x2)
  for (int i = 0; i < N; i++)
    a[i] = i + x2;

  int sum = 0;

  for (int i = 0; i < N; i++)
    sum += a[i];

  return (sum == ((99 + 2 * x2) * 100) / 2);
}

int test_omp_parallel_for_simd_copyin() {
  int a[N];
  x3 = 3;

#pragma omp parallel for simd copyin(x3)
  for (int i = 0; i < N; i++)
    a[i] = i + x3;

  int sum = 0;

  for (int i = 0; i < N; i++)
    sum += a[i];

  return (sum == ((99 + 2 * x3) * 100) / 2);
}

int test_omp_parallel_sections_copyin() {
  int a = 0;
  int b = 0;
  x4 = 4;

#pragma omp parallel sections copyin(x4)
  {
#pragma omp section
    { a = x4; }

#pragma omp section
    { b = x4; }
  }

  return (a + b == x4 * 2);
}

int test_omp_parallel_master_copyin() {
  int a[N];
  x5 = 5;

#pragma omp parallel master copyin(x5)
  for (int i = 0; i < N; i++)
    a[i] = i + x5;

  int sum = 0;

  for (int i = 0; i < N; i++)
    sum += a[i];

  return (sum == ((99 + 2 * x5) * 100) / 2);
}

int main() {
  int num_failed = 0;

  if (!test_omp_parallel_copyin())
    num_failed++;

  if (!test_omp_parallel_for_copyin())
    num_failed++;

  if (!test_omp_parallel_for_simd_copyin())
    num_failed++;

  if (!test_omp_parallel_sections_copyin())
    num_failed++;

  if (!test_omp_parallel_master_copyin())
    num_failed++;

  return num_failed;
}
