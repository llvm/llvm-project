// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic

// FIXME: This is a bug in host offload, this should run fine.
// REQUIRES: gpu

#include <iostream>
#include <vector>

#define N 8

int main() {
  std::vector<int> avec(N);
  int *a = avec.data();
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    a[i] = 0;
#pragma omp target teams distribute parallel for reduction(+ : a[i])
    for (int j = 0; j < N; j++)
      a[i] += 1;
  }

  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  // CHECK: 8
  for (int i = 0; i < N; i++)
    std::cout << a[i] << std::endl;
}
