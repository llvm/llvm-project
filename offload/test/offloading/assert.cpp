// RUN: %libomptarget-compilexx-generic && %libomptarget-run-fail-generic
// RUN: %libomptarget-compileoptxx-generic && %libomptarget-run-fail-generic
// https://github.com/llvm/llvm-project/issues/182119
// UNSUPPORTED: intelgpu

int main(int argc, char *argv[]) {
#pragma omp target
  { __builtin_trap(); }

  return 0;
}
