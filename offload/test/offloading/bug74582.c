// RUN: %libomptarget-compile-generic && %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic && %libomptarget-run-generic
// https://github.com/llvm/llvm-project/issues/182119
// UNSUPPORTED: intelgpu

// Verify we do not read bits in the image that are not there (nobits section).

#pragma omp begin declare target
char BigUninitializedBuffer[4096 * 64] __attribute__((loader_uninitialized));
#pragma omp end declare target

int main() {
#pragma omp target
  {}
}
