// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

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
