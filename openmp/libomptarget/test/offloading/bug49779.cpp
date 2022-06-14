// RUN: %libomptarget-compilexx-generic && \
// RUN:   env LIBOMPTARGET_STACK_SIZE=2048 %libomptarget-run-generic

// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: amdgcn-amd-amdhsa-oldDriver
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO

#include <cassert>
#include <iostream>

void work(int *C) {
#pragma omp atomic
  ++(*C);
}

void use(int *C) {
#pragma omp parallel num_threads(2)
  work(C);
}

int main() {
  int C = 0;
#pragma omp target map(C)
  {
    use(&C);
#pragma omp parallel num_threads(2)
    use(&C);
  }

  assert(C >= 2 && C <= 6);

  std::cout << "PASS\n";

  return 0;
}

// CHECK: PASS
