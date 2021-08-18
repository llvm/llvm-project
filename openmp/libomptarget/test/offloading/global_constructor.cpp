// RUN: %libomptarget-compilexx-generic && %libomptarget-run-generic | %fcheck-generic

#include <cmath>
#include <cstdio>

const double Host = log(2.0) / log(2.0);
#pragma omp declare target
const double Device = log(2.0) / log(2.0);
#pragma omp end declare target

int main() {
  double X;
#pragma omp target map(from : X)
  { X = Device; }

  // CHECK: PASS
  if (X == Host)
    printf("PASS\n");
}
