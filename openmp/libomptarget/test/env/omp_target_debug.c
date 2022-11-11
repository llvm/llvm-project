// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=NDEBUG
// REQUIRES: libomptarget-debug

int main(void) {
#pragma omp target
  {}
  return 0;
}

// DEBUG: Libomptarget
// NDEBUG-NOT: Libomptarget
// NDEBUG-NOT: Target
