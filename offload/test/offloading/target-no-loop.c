// clang-format off
// C counterpart of fortran/target-no-loop.f90.

// RUN: %libomptarget-compile-generic -O3 -fopenmp-assume-threads-oversubscription -fopenmp-assume-teams-oversubscription -fopenmp-enable-irbuilder
// RUN: env LIBOMPTARGET_INFO=16 OMP_NUM_TEAMS=16 OMP_TEAMS_THREAD_LIMIT=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
// REQUIRES: gpu
// XFAIL: intelgpu

#include <stdio.h>

static int check_errors(int *array) {
  int errors = 0;
  for (int i = 0; i < 1024; ++i)
    if (array[i] != i + 1)
      ++errors;
  return errors;
}

int main(void) {
  int array[1024];
  int errors = 0;
  int red;

  for (int i = 0; i < 1024; ++i)
    array[i] = 1;

  // No-loop kernel
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
  errors += check_errors(array);

  // SPMD kernel (num_teams clause blocks promotion to no-loop)
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
#pragma omp target teams distribute parallel for num_teams(3)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
  errors += check_errors(array);

  // No-loop kernel
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
#pragma omp target teams distribute parallel for num_threads(64)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
  errors += check_errors(array);

  // SPMD kernel
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
#pragma omp target parallel for
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
  errors += check_errors(array);

  // Generic kernel
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
#pragma omp target teams distribute
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
  errors += check_errors(array);

  // SPMD kernel (reduction clause blocks promotion to no-loop)
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
  red = 0;
#pragma omp target teams distribute parallel for reduction(+ : red)
  for (int i = 0; i < 1024; ++i)
    red += array[i];
  if (red != 1024)
    ++errors;

  // No-loop kernel with a lastprivate variable (final value from thread executing last iteration)
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;

  int last = -1;
#pragma omp target teams distribute parallel for lastprivate(last)
  for (int i = 0; i < 1024; ++i) {
    array[i] = i + 1;
    last = i;
  }
  errors += check_errors(array);
#ifdef ASSERT_LASTPRIVATE
  if (last != 1023)
    ++errors;
#endif

  // No-loop kernel with a lastprivate loop counter (final value from evaluation at distribution level)
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
  int iv = -1;
#pragma omp target teams distribute parallel for lastprivate(iv)
  for (iv = 0; iv < 1024; ++iv)
    array[iv] = iv + 1;
  errors += check_errors(array);
#ifdef ASSERT_LASTPRIVATE
  if (iv != 1024)
    ++errors;
#endif

  // No-loop simd kernel with a lastprivate loop counter
  for (int i = 0; i < 1024; ++i)
    array[i] = 1;
  int simd_iv = -1;
#pragma omp target teams distribute parallel for simd lastprivate(simd_iv)
  for (simd_iv = 0; simd_iv < 1024; ++simd_iv)
    array[simd_iv] = simd_iv + 1;
  errors += check_errors(array);
#ifdef ASSERT_LASTPRIVATE
  if (simd_iv != 1024)
    ++errors;
#endif

  printf("number of errors: %d\n", errors);
  return 0;
}

// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
// CHECK:  info: #Args: 2 Teams x Thrds:   64x  16
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
// CHECK:  info: #Args: 2 Teams x Thrds:   3x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
// CHECK:  info: #Args: 2 Teams x Thrds:   64x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
// CHECK:  info: #Args: 2 Teams x Thrds:   1x  16
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} Generic-SPMD mode
// CHECK:  info: #Args: 2 Teams x Thrds:   16x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
// CHECK:  info: #Args: 3 Teams x Thrds:   16x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
// CHECK:  info: #Args: 3 Teams x Thrds:   64x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
// CHECK:  info: #Args: 3 Teams x Thrds:   64x  16 {{.*}}
// CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
// CHECK:  info: #Args: 3 Teams x Thrds:   64x  16 {{.*}}
// CHECK:  number of errors: 0
