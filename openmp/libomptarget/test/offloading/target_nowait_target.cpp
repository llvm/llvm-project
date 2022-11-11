// RUN: %libomptarget-compilexx-and-run-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

#include <cassert>

int main(int argc, char *argv[]) {
  int data[1024];
  int sum = 0;

  for (int i = 0; i < 1024; ++i)
    data[i] = i;

#pragma omp target map(tofrom : sum) map(to : data) depend(inout : data[0])    \
    nowait
  {
    for (int i = 0; i < 1024; ++i) {
      sum += data[i];
    }
  }

#pragma omp target map(tofrom : sum) map(to : data) depend(inout : data[0])
  {
    for (int i = 0; i < 1024; ++i) {
      sum += data[i];
    }
  }

  assert(sum == 1023 * 1024);

  return 0;
}
