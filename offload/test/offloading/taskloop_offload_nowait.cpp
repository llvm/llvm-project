// RUN: %libomptarget-compilexx-and-run-generic

// REQUIRES: gpu

#include <cmath>
#include <cstdlib>
#include <iostream>

bool almost_equal(float x, float gold, float tol) {
  if (std::signbit(x) != std::signbit(gold))
    x = std::abs(gold) - std::abs(x);

  return std::abs(gold) * (1 - tol) <= std::abs(x) &&
         std::abs(x) <= std::abs(gold) * (1 + tol);
}

int main(int argc, char *argv[]) {
  constexpr const int N0{2};
  constexpr const int N1{182};
  constexpr const float expected_value{N0 * N1};
  float counter_N0{};

#pragma omp target data map(tofrom : counter_N0)
  {
#pragma omp taskloop shared(counter_N0)
    for (int i0 = 0; i0 < N0; i0++) {
#pragma omp target teams distribute parallel for map(tofrom : counter_N0) nowait
      for (int i1 = 0; i1 < N1; i1++) {
#pragma omp atomic update
        counter_N0 = counter_N0 + 1.;
      }
    }
  }

  if (!almost_equal(counter_N0, expected_value, 0.1)) {
    std::cerr << "Expected: " << expected_value << " Got: " << counter_N0
              << '\n';
    return -1;
  }

  return 0;
}
