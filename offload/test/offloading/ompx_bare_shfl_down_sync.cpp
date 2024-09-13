// RUN: %libomptarget-compilexx-run-and-check-generic
//
// REQUIRES: gpu

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ompx.h>
#include <type_traits>

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
bool equal(T LHS, T RHS) {
  return LHS == RHS;
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
bool equal(T LHS, T RHS) {
  return __builtin_fabs(LHS - RHS) < std::numeric_limits<T>::epsilon();
}

template <typename T> void test() {
  constexpr const int num_blocks = 1;
  constexpr const int block_size = 256;
  constexpr const int N = num_blocks * block_size;
  int *res = new int[N];

#pragma omp target teams ompx_bare num_teams(num_blocks) thread_limit(block_size) \
        map(from: res[0:N])
  {
    int tid = ompx_thread_id_x();
    T val = ompx::shfl_down_sync(~0U, static_cast<T>(tid), 1);
#ifdef __AMDGCN_WAVEFRONT_SIZE
    int warp_size = __AMDGCN_WAVEFRONT_SIZE;
#else
    int warp_size = 32;
#endif
    if ((tid & (warp_size - 1)) != warp_size - 1)
      res[tid] = equal(val, static_cast<T>(tid + 1));
    else
      res[tid] = equal(val, static_cast<T>(tid));
  }

  for (int i = 0; i < N; ++i)
    assert(res[i]);

  delete[] res;
}

int main(int argc, char *argv[]) {
  test<int32_t>();
  test<int64_t>();
  test<float>();
  test<double>();
  // CHECK: PASS
  printf("PASS\n");

  return 0;
}
