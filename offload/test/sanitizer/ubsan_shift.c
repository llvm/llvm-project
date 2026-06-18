// clang-format off
// RUN: %libomptarget-compile-generic -g -Xarch_device -fsanitize=undefined -Xarch_device -fsanitize-minimal-runtime
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK

// REQUIRES: libc
// REQUIRES: gpu

#include <omp.h>

int main(void) {
#pragma omp target teams num_teams(1) thread_limit(2)
#pragma omp parallel num_threads(2)
  {
    volatile int shift = 33;
    volatile int x = 1;
    x <<= shift; // shift exponent >= width: undefined behavior
    (void)x;
  }
}

// CHECK: {{.*}}ubsan_shift.c:[[#@LINE-9]]:{{[0-9]+}}: runtime error: shift-out-of-bounds
// CHECK-NEXT: on GPU thread: block (0,0,0) thread ({{[0-9]+}},0,0) lane {{[0-9]+}}
// CHECK-NEXT: #0 0x{{[0-9a-f]+}} in {{.*}}ubsan_shift.c:[[#@LINE-11]]:
// CHECK: SUMMARY: UndefinedBehaviorSanitizer: shift-out-of-bounds {{.*}}ubsan_shift.c:[[#@LINE-12]]:
