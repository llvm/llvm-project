// RUN: %libomptarget-compilexx-generic && %libomptarget-run-fail-generic
// RUN: %libomptarget-compileoptxx-generic && %libomptarget-run-fail-generic

int main(int argc, char *argv[]) {
#pragma omp target
  { __builtin_trap(); }

  return 0;
}
