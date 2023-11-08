// RUN: %libomptarget-compilexx-and-run-generic
// RUN: %libomptarget-compileoptxx-and-run-generic

// FIXME: This is a bug in host offload, this should run fine.
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// This test validates that the OpenMP target reductions to find a maximum work
// as indended for a few common data types.

#include <algorithm>
#include <cassert>
#include <vector>

template <class Tp> void test_max_reduction() {
  const int length = 1000;
  const int nmaximas = 8;
  std::vector<Tp> a(length, (Tp)3);
  const int step = length / nmaximas;
  for (int i = 0; i < nmaximas; i++) {
    a[i * step] += (Tp)1;
  }
  for (int i = nmaximas - 1; i >= 0; i--) {
    int idx = 0;
    Tp *b = a.data();
#pragma omp target teams distribute parallel for reduction(max : idx)          \
    map(always, to : b[0 : length])
    for (int j = 1; j < length; j++) {
      if (b[j] > b[j - 1]) {
        idx = std::max(idx, j);
      }
    }
    assert(idx == i * step &&
           "#pragma omp target teams distribute parallel for "
           "reduction(max:<identifier list>) does not work as intended.");
    a[idx] -= (Tp)1;
  }
}

int main() {
  test_max_reduction<float>();
  test_max_reduction<double>();
  test_max_reduction<int>();
  test_max_reduction<unsigned int>();
  test_max_reduction<long>();
  return 0;
}
