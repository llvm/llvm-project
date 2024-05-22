// RUN: %libomptarget-compile-generic && %libomptarget-run-fail-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefix=CHECK

// REQUIRES: libc

// UNSUPPORTED: powerpc64-ibm-linux-gnu
// UNSUPPORTED: powerpc64-ibm-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <assert.h>

int main() {
  // CHECK: Assertion failed: '0 && "Trivial failure"' in function: 'int main()'
  // CHECK-NOT: Assertion failed:
#pragma omp target
#pragma omp parallel
  { assert(0 && "Trivial failure"); }
}
