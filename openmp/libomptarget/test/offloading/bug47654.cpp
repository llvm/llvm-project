// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
  int i = 0, j = 0;

#pragma omp target map(tofrom : i, j) nowait
  {
    i = 1;
    j = 2;
  }

#pragma omp taskwait

  assert(i == 1);
  assert(j == 2);

  std::cout << "PASS\n";

  return 0;
}

// CHECK: PASS
