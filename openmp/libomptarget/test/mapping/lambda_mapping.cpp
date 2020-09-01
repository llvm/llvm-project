// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <iostream>

template <typename LOOP_BODY>
inline void forall(int Begin, int End, LOOP_BODY LoopBody) {
#pragma omp target parallel for schedule(static)
  for (int I = Begin; I < End; ++I) {
    LoopBody(I);
  }
}

#define N (1000)

//
// Demonstration of the RAJA abstraction using lambdas
// Requires data mapping onto the target section
//
int main() {
  double A[N], B[N], C[N];

  for (int I = 0; I < N; I++) {
    A[I] = I + 1;
    B[I] = -I;
    C[I] = -9;
  }

#pragma omp target data map(tofrom : C [0:N]) map(to : A [0:N], B [0:N])
  {
    forall(0, N, [&](int I) { C[I] += A[I] + B[I]; });
  }

  int Fail = 0;
  for (int I = 0; I < N; I++) {
    if (C[I] != -8) {
      std::cout << "Failed at " << I << " with val " << C[I] << std::endl;
      Fail = 1;
    }
  }

  // CHECK: Succeeded
  if (Fail) {
    std::cout << "Failed" << std::endl;
  } else {
    std::cout << "Succeeded" << std::endl;
  }

  return 0;
}
