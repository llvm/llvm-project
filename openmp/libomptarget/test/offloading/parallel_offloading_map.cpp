// RUN: %libomptarget-compilexx-run-and-check-generic

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-oldDriver

#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
  constexpr const int num_threads = 64, N = 128;
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

  std::cout << "PASS\n";

  return 0;
}

// CHECK: PASS
