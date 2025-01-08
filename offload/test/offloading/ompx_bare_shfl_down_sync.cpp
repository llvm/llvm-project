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

#pragma omp begin declare variant match(device = {arch(amdgcn)})
unsigned get_warp_size() { return __builtin_amdgcn_wavefrontsize(); }
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {arch(nvptx64)})
unsigned get_warp_size() { return __nvvm_read_ptx_sreg_warpsize(); }
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
unsigned get_warp_size() { return 1; }
#pragma omp end declare variant

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
    int warp_size = get_warp_size();
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
