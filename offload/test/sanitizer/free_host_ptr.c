// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu

#include <omp.h>

int main(void) {
  int X;
  omp_target_free(&X, 0);
}

// CHECK: Failure to deallocate device pointer 0x{{.*}}: pointer is not a known allocation in this context
// CHECK: Failed to deallocate device ptr
