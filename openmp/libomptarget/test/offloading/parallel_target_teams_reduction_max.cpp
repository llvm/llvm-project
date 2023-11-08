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
#include <limits>
#include <vector>

template <class Tp> void test_max_idx_reduction() {
  const Tp length = 1000;
  const Tp nmaximas = 8;
  std::vector<float> a(length, 3.0f);
  const Tp step = length / nmaximas;
  for (Tp i = 0; i < nmaximas; i++) {
    a[i * step] += 1.0f;
  }
  for (Tp i = nmaximas; i > 0; i--) {
    Tp idx = 0;
    float *b = a.data();
#pragma omp target teams distribute parallel for reduction(max : idx)          \
    map(always, to : b[0 : length])
    for (Tp j = 0; j < length - 1; j++) {
      if (b[j] > b[j + 1]) {
        idx = std::max(idx, j);
      }
    }
    assert(idx == (i - 1) * step &&
           "#pragma omp target teams distribute parallel for "
           "reduction(max:<identifier list>) does not work as intended.");
    a[idx] -= 1.0f;
  }
}

template <class Tp> void test_max_val_reduction() {
  const int length = 1000;
  const int half = length / 2;
  std::vector<Tp> a(length, (Tp)3);
  a[half] += (Tp)1;
  Tp max_val = std::numeric_limits<Tp>::lowest();
  Tp *b = a.data();
#pragma omp target teams distribute parallel for reduction(max : max_val)      \
    map(always, to : b[0 : length])
  for (int i = 0; i < length; i++) {
    max_val = std::max(max_val, b[i]);
  }
  assert(std::abs(((double)a[half + 1]) - ((double)max_val) + 1.0) < 1e-6 &&
         "#pragma omp target teams distribute parallel for "
         "reduction(max:<identifier list>) does not work as intended.");
}

int main() {
  // Reducing over indices
  test_max_idx_reduction<int>();
  test_max_idx_reduction<unsigned int>();
  test_max_idx_reduction<long>();

  // Reducing over values
  test_max_val_reduction<int>();
  test_max_val_reduction<unsigned int>();
  test_max_val_reduction<long>();
  test_max_val_reduction<float>();
  test_max_val_reduction<double>();
  return 0;
}
