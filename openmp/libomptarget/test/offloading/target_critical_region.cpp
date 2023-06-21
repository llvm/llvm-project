// RUN: %libomptarget-compilexx-run-and-check-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

#define N 1000000

int A[N];
int main() {
  for (int i = 0; i < N; i++)
    A[i] = 1;

  int sum[1];
  sum[0] = 0;

#pragma omp target teams distribute parallel for num_teams(256)                \
    schedule(static, 1) map(to                                                 \
                            : A[:N]) map(tofrom                                \
                                         : sum[:1])
  {
    for (int i = 0; i < N; i++) {
#pragma omp critical
      { sum[0] += A[i]; }
    }
  }

  // CHECK: SUM = 1000000
  printf("SUM = %d\n", sum[0]);

  return 0;
}
