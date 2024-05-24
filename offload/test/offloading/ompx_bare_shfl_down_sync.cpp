// RUN: %libomptarget-compilexx-run-and-check-generic
//
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#ifdef __AMDGCN_WAVEFRONT_SIZE
#define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE
#else
#define WARP_SIZE 32
#endif

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
  return std::abs(LHS - RHS) < std::numeric_limits<T>::epsilon();
}

template <typename T> void test() {
  constexpr const int num_blocks = 1;
  constexpr const int block_size = 256;
  constexpr const int N = num_blocks * block_size;
  T *data = new T[N];

  for (int i = 0; i < N; ++i)
    data[i] = i;

#pragma omp target teams ompx_bare num_teams(num_blocks)                       \
    thread_limit(block_size) map(tofrom : data[0 : N])
  {
    int tid = ompx_thread_id_x();
    data[tid] = ompx::shfl_down_sync(~0U, data[tid], 1);
  }

  for (int i = N - 1; i > 0; i -= WARP_SIZE)
    for (int j = i; j > i - WARP_SIZE; --j)
      assert(equal(data[i], data[i - 1]));

  delete[] data;
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
