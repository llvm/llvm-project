// RUN: %libomptarget-compilexx-and-run-generic
// RUN: %libomptarget-compileoptxx-and-run-generic

// FIXME: This is a bug in host offload, this should run fine.
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// This test validates that the OpenMP target reductions to find a minimum work
// as indended for a few common data types.

#include <algorithm>
#include <cassert>
#include <vector>

template <class Tp> void test_min_reduction() {
  const int length = 1000;
  const int nminimas = 8;
  std::vector<Tp> a(length, (Tp)3);
  const int step = length / nminimas;
  for (int i = 0; i < nminimas; i++) {
    a[i * step] -= (Tp)1;
  }
  for (int i = 0; i < nminimas; i++) {
    int idx = a.size();
    Tp *b = a.data();
#pragma omp target teams distribute parallel for reduction(min : idx)          \
    map(always, to : b[0 : length])
    for (int j = 0; j < length - 1; j++) {
      if (b[j] < b[j + 1]) {
        idx = std::min(idx, j);
      }
    }
    assert(idx == i * step &&
           "#pragma omp target teams distribute parallel for "
           "reduction(min:<identifier list>) does not work as intended.");
    a[idx] += (Tp)1;
  }
}

int main() {
  test_min_reduction<float>();
  test_min_reduction<double>();
  test_min_reduction<int>();
  test_min_reduction<unsigned int>();
  test_min_reduction<long>();
  return 0;
}
