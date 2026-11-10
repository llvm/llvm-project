// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %clang-generic --offload-device-only -S -emit-llvm %s -o - \
// RUN:   | %fcheck-generic --check-prefix=IR
// RUN: %clang-generic -munsafe-fp-atomics --offload-device-only -S -emit-llvm \
// RUN:   %s -o - | %fcheck-generic --check-prefix=UNSAFE
// REQUIRES: amdgpu

// Check that the fp atomics hints from omp.h select the denormal mode of the
// generated atomics.

#include <omp.h>
#include <stdio.h>

#define N 4096

#pragma omp declare target
// IR-LABEL: define {{.*}}@addNoHint(
// IR: atomicrmw fadd
// IR-NOT: !atomic.ignore.denormal.mode
// IR: ret void
// UNSAFE-LABEL: define {{.*}}@addNoHint(
// UNSAFE: atomicrmw fadd {{.*}}!atomic.ignore.denormal.mode
void addNoHint(float *X) {
#pragma omp atomic
  *X += 1.0f;
}

// IR-LABEL: define {{.*}}@addFast(
// IR: atomicrmw fadd {{.*}}!atomic.ignore.denormal.mode
// UNSAFE-LABEL: define {{.*}}@addFast(
// UNSAFE: atomicrmw fadd {{.*}}!atomic.ignore.denormal.mode
void addFast(float *X) {
#pragma omp atomic hint(ompx_fast_fp_atomics)
  *X += 1.0f;
}

// IR-LABEL: define {{.*}}@addSafe(
// IR: atomicrmw fadd
// IR-NOT: !atomic.ignore.denormal.mode
// IR: ret void
// UNSAFE-LABEL: define {{.*}}@addSafe(
// UNSAFE: atomicrmw fadd
// UNSAFE-NOT: !atomic.ignore.denormal.mode
// UNSAFE: ret void
void addSafe(float *X) {
#pragma omp atomic hint(ompx_safe_fp_atomics)
  *X += 1.0f;
}
#pragma omp end declare target

int main() {
  float NoHint = 0.0f, Fast = 0.0f, Safe = 0.0f;
#pragma omp target teams distribute parallel for map(tofrom : NoHint, Fast,    \
                                                         Safe)
  for (int I = 0; I < N; ++I) {
    addNoHint(&NoHint);
    addFast(&Fast);
    addSafe(&Safe);
  }
  // CHECK: no hint: 4096
  // CHECK: fast: 4096
  // CHECK: safe: 4096
  printf("no hint: %d\n", (int)NoHint);
  printf("fast: %d\n", (int)Fast);
  printf("safe: %d\n", (int)Safe);
  return 0;
}
