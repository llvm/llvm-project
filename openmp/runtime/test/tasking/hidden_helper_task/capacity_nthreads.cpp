// RUN: %libomp-cxx-compile-and-run

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

// AIX runs out of resource in 32-bit if 4*omp_get_max_threads() is more
// than 64 threads with the default stacksize.
#if defined(_AIX) && !__LP64__
#define MAX_THREADS 64
#endif

int main(int argc, char *argv[]) {
  int N = std::min(std::max(std::max(32, 4 * omp_get_max_threads()),
                            4 * omp_get_num_procs()),
                   std::numeric_limits<int>::max());

#if defined(_AIX) && !__LP64__
  if (N > MAX_THREADS)
    N = MAX_THREADS;
#endif

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
