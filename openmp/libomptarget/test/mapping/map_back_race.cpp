// RUN: %libomptarget-compilexx-and-run-generic

// Taken from https://github.com/llvm/llvm-project/issues/54216


// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-oldDriver

#include <algorithm>
#include <cstdlib>
#include <iostream>

bool almost_equal(float x, float gold, float rel_tol = 1e-09,
                  float abs_tol = 0.0) {
  return std::abs(x - gold) <=
         std::max(rel_tol * std::max(std::abs(x), std::abs(gold)), abs_tol);
}
void test_parallel_for__target() {
  const int N0{32768};
  const float expected_value{N0};
  float counter_N0{};
#pragma omp parallel for
  for (int i0 = 0; i0 < N0; i0++) {
#pragma omp target map(tofrom : counter_N0)
    {
#pragma omp atomic update
      counter_N0 = counter_N0 + 1.;
    }
  }
  if (!almost_equal(counter_N0, expected_value, 0.01)) {
    std::cerr << "Expected: " << expected_value << " Got: " << counter_N0
              << std::endl;
    std::exit(112);
  }
}
int main() { test_parallel_for__target(); }
