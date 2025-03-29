// RUN: %libomptarget-compilexx-and-run-generic
// RUN: %libomptarget-compileoptxx-and-run-generic

// FIXME: This is a bug in host offload, this should run fine.
// REQUIRES: gpu

// This test validates that the OpenMP target reductions to find a minimum work
// as intended for a few common data types.

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

template <class Tp> void test_min_idx_reduction() {
  const Tp length = 1000;
  const Tp nminimas = 8;
  std::vector<float> a(length, 3.0f);
  const Tp step = length / nminimas;
  for (Tp i = 0; i < nminimas; i++) {
    a[i * step] -= 1.0f;
  }
  for (Tp i = 0; i < nminimas; i++) {
    Tp idx = a.size();
    float *b = a.data();
#pragma omp target teams distribute parallel for reduction(min : idx)          \
    map(always, to : b[0 : length])
    for (Tp j = 0; j < length - 1; j++) {
      if (b[j] < b[j + 1]) {
        idx = std::min(idx, j);
      }
    }
    assert(idx == i * step &&
           "#pragma omp target teams distribute parallel for "
           "reduction(min:<identifier list>) does not work as intended.");
    a[idx] += 1.0f;
  }
}

template <class Tp> void test_min_val_reduction() {
  const int length = 1000;
  const int half = length / 2;
  std::vector<Tp> a(length, (Tp)3);
  a[half] -= (Tp)1;
  Tp min_val = std::numeric_limits<Tp>::max();
  Tp *b = a.data();
#pragma omp target teams distribute parallel for reduction(min : min_val)      \
    map(always, to : b[0 : length])
  for (int i = 0; i < length; i++) {
    min_val = std::min(min_val, b[i]);
  }
  assert(std::abs(((double)a[half + 1]) - ((double)min_val) - 1.0) < 1e-6 &&
         "#pragma omp target teams distribute parallel for "
         "reduction(min:<identifier list>) does not work as intended.");
}

int main() {
  // Reducing over indices
  test_min_idx_reduction<int>();
  test_min_idx_reduction<unsigned int>();
  test_min_idx_reduction<long>();

  // Reducing over values
  test_min_val_reduction<int>();
  test_min_val_reduction<unsigned int>();
  test_min_val_reduction<long>();
  test_min_val_reduction<float>();
  test_min_val_reduction<double>();
  return 0;
}
