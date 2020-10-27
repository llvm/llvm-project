// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu && env OMP_MAX_ACTIVE_LEVELS=2 %libomptarget-run-aarch64-unknown-linux-gnu | %fcheck-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu && env OMP_MAX_ACTIVE_LEVELS=2 %libomptarget-run-powerpc64-ibm-linux-gnu | %fcheck-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu && env OMP_MAX_ACTIVE_LEVELS=2 %libomptarget-run-powerpc64le-ibm-linux-gnu | %fcheck-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu && env OMP_MAX_ACTIVE_LEVELS=2 %libomptarget-run-x86_64-pc-linux-gnu | %fcheck-x86_64-pc-linux-gnu -allow-empty
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env OMP_MAX_ACTIVE_LEVELS=2 %libomptarget-run-nvptx64-nvidia-cuda | %fcheck-nvptx64-nvidia-cuda -allow-empty
#include <assert.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  const int num_threads = 64, N = 128;
  int array[num_threads] = {0};

#pragma omp parallel for
  for (int i = 0; i < num_threads; ++i) {
    int tmp[N];

    for (int j = 0; j < N; ++j) {
      tmp[j] = i;
    }

#pragma omp target teams distribute parallel for map(tofrom : tmp)
    for (int j = 0; j < N; ++j) {
      tmp[j] += j;
    }

    for (int j = 0; j < N; ++j) {
      array[i] += tmp[j];
    }
  }

  // Verify
  for (int i = 0; i < num_threads; ++i) {
    const int ref = (0 + N - 1) * N / 2 + i * N;
    assert(array[i] == ref);
  }

  printf("PASS\n");

  return 0;
}

// CHECK: PASS
