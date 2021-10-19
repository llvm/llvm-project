// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// CHECK: Libomptarget message: explicit extension not allowed: host address specified is 0x{{.*}} (8 bytes), but device allocation maps to host at 0x{{.*}} (8 bytes)
// CHECK: Libomptarget error: Call to getTargetPointer returned null pointer (device failure or illegal mapping).
// CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory

int main() {
  int arr[4] = {0, 1, 2, 3};
#pragma omp target data map(alloc: arr[0:2])
#pragma omp target data map(alloc: arr[1:2])
  ;
  return 0;
}
