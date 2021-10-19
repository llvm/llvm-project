// RUN: %libomp-cxx-compile-and-run

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

int main(int argc, char *argv[]) {
  const int N = std::min(std::max(std::max(32, 4 * omp_get_max_threads()),
                                  4 * omp_get_num_procs()),
                         std::numeric_limits<int>::max());

  std::vector<int> data(N);

#pragma omp parallel for num_threads(N)
  for (unsigned i = 0; i < N; ++i) {
    data[i] = i;
  }

#pragma omp parallel for num_threads(N + 1)
  for (unsigned i = 0; i < N; ++i) {
    data[i] += i;
  }

  for (unsigned i = 0; i < N; ++i) {
    assert(data[i] == 2 * i);
  }

  return 0;
}
