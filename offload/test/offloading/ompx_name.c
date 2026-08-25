// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: gpu

#include <stdio.h>

int main() {
  int result = 0;

// CHECK: PluginInterface device {{[0-9]}} info: Launching kernel
// CHECK-SAME: my_custom_kernel
#pragma omp target ompx_name("my_custom_kernel") map(from : result)
  {
    result = 42;
  }

  if (result != 42) {
    printf("FAIL: result = %d\n", result);
    return 1;
  }

  result = 0;

// CHECK: PluginInterface device {{[0-9]}} info: Launching kernel
// CHECK-SAME: parallel_kernel_name
#pragma omp target parallel ompx_name("parallel_kernel_name")                  \
    map(tofrom : result)
  {
#pragma omp atomic
    result++;
  }

  if (result == 0) {
    printf("FAIL: parallel result = %d\n", result);
    return 1;
  }

  result = 0;

// CHECK: PluginInterface device {{[0-9]}} info: Launching kernel
// CHECK-SAME: teams_kernel_name
#pragma omp target teams ompx_name("teams_kernel_name") map(tofrom : result)
  {
#pragma omp atomic
    result++;
  }

  if (result == 0) {
    printf("FAIL: teams result = %d\n", result);
    return 1;
  }

  int data[100];
  for (int i = 0; i < 100; i++)
    data[i] = 0;

// CHECK: PluginInterface device {{[0-9]}} info: Launching kernel
// CHECK-SAME: parallel_for_kernel
#pragma omp target parallel for ompx_name("parallel_for_kernel")               \
    map(tofrom : data[0 : 100])
  for (int i = 0; i < 100; i++) {
    data[i] = i;
  }

  for (int i = 0; i < 100; i++) {
    if (data[i] != i) {
      printf("FAIL: data[%d] = %d\n", i, data[i]);
      return 1;
    }
  }

  // CHECK: PASS
  printf("PASS\n");

  return 0;
}
