//===----- test_xteamr.cpp - Test for Xteamr DeviceRTL functions ---C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// performance and functional tests for Xteamr reduction helper functions in
// libomptarget/DeviceRTL/Xteamr.cpp
//
// RUN: %libomptarget-compileoptxx-run-and-check-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda
// CHECK: ALL TESTS PASSED
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

#include "test_xteamr.h"

#ifndef _ARRAY_SIZE
#define _ARRAY_SIZE 33554432
#endif
const uint64_t ARRAY_SIZE = _ARRAY_SIZE;
unsigned int repeat_num_times = 12;
unsigned int ignore_times =
    2; // ignore this many timings first

// If we know at compile time that we have 0 index with 1 stride,
// then generate an optimized _BIG_JUMP_LOOP.
// This test case has index 0 and stride 1, so we set this here.
#define __OPTIMIZE_INDEX0_STRIDE1

//  Extern Xteamr functions are designed for 1024, 512, and 256 thread blocks.
//  The default here is 512.

#ifndef _XTEAM_NUM_THREADS
#define _XTEAM_NUM_THREADS 512
#endif
#ifndef _XTEAM_NUM_TEAMS
#define _XTEAM_NUM_TEAMS 80
#endif

#if _XTEAM_NUM_THREADS == 1024
#define _SUM_OVERLOAD_64_FCT _overload_to_extern_sum_16x64
#define _SUM_OVERLOAD_32_FCT _overload_to_extern_sum_32x32
#define _MAX_OVERLOAD_64_FCT _overload_to_extern_max_16x64
#define _MAX_OVERLOAD_32_FCT _overload_to_extern_max_32x32
#define _MIN_OVERLOAD_64_FCT _overload_to_extern_min_16x64
#define _MIN_OVERLOAD_32_FCT _overload_to_extern_min_32x32
#elif _XTEAM_NUM_THREADS == 512
#define _SUM_OVERLOAD_64_FCT _overload_to_extern_sum_8x64
#define _SUM_OVERLOAD_32_FCT _overload_to_extern_sum_16x32
#define _MAX_OVERLOAD_64_FCT _overload_to_extern_max_8x64
#define _MAX_OVERLOAD_32_FCT _overload_to_extern_max_16x32
#define _MIN_OVERLOAD_64_FCT _overload_to_extern_min_8x64
#define _MIN_OVERLOAD_32_FCT _overload_to_extern_min_16x32
#elif _XTEAM_NUM_THREADS == 256
#define _SUM_OVERLOAD_64_FCT _overload_to_extern_sum_4x64
#define _SUM_OVERLOAD_32_FCT _overload_to_extern_sum_8x32
#define _MAX_OVERLOAD_64_FCT _overload_to_extern_max_4x64
#define _MAX_OVERLOAD_32_FCT _overload_to_extern_max_8x32
#define _MIN_OVERLOAD_64_FCT _overload_to_extern_min_4x64
#define _MIN_OVERLOAD_32_FCT _overload_to_extern_min_8x32
#elif _XTEAM_NUM_THREADS == 128
#define _SUM_OVERLOAD_64_FCT _overload_to_extern_sum_2x64
#define _SUM_OVERLOAD_32_FCT _overload_to_extern_sum_4x32
#define _MAX_OVERLOAD_64_FCT _overload_to_extern_max_2x64
#define _MAX_OVERLOAD_32_FCT _overload_to_extern_max_4x32
#define _MIN_OVERLOAD_64_FCT _overload_to_extern_min_2x64
#define _MIN_OVERLOAD_32_FCT _overload_to_extern_min_4x32
#elif _XTEAM_NUM_THREADS == 64
#define _SUM_OVERLOAD_64_FCT _overload_to_extern_sum_1x64
#define _SUM_OVERLOAD_32_FCT _overload_to_extern_sum_2x32
#define _MAX_OVERLOAD_64_FCT _overload_to_extern_max_1x64
#define _MAX_OVERLOAD_32_FCT _overload_to_extern_max_2x32
#define _MIN_OVERLOAD_64_FCT _overload_to_extern_min_1x64
#define _MIN_OVERLOAD_32_FCT _overload_to_extern_min_2x32
#else
#error Invalid value for _XTEAM_NUM_THREADS. Must be 1024, 512, 256, 128, or 64
#endif

// Question to Dhruva, should the limiter include the stride?
#if defined(__NVPTX__) && _XTEAM_NUM_THREADS == 1024
       // Cuda may restrict max threads when requesting 1024, so the bigjump
// on the inner loop depends on the actual limited number of threads
// determined by omp_get_num_threads(). It also requires we only call
// the helper reducer function when k is in this range. Lastly, the
// helper function must clear (set to rnv) unused xwave values
// before the optimized (unrolled) xwave reduction loop. See Xteamr.cpp.
// These three things kill performance on nvidia when requested thread=1024.
// So codegen max request of 512 threads (16x32) for nvidia GPUs.
#define _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(nteams)                            \
  if (k < (nteams * omp_get_num_threads()))
#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * omp_get_num_threads())
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * omp_get_num_threads() * stride))
#endif
#else
       // Assume AMDGPU or NVIDIA=512|256 always gets requested number of
       // threads.
// So no conditional needed to limit reductions.
#define _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(nteams)

//  Format of BIG_JUMP_LOOP depends on if we optimize for 0 index 1 stride
#if _XTEAM_NUM_THREADS == 1024

#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * 1024)
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * 1024 * stride))
#endif

#elif _XTEAM_NUM_THREADS == 512

#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * 512)
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * 512 * stride))
#endif

#elif _XTEAM_NUM_THREADS == 256 

#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * 256)
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * 256* stride))
#endif

#elif _XTEAM_NUM_THREADS == 128

#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * 128)
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * 128* stride))
#endif

#else

#ifdef __OPTIMIZE_INDEX0_STRIDE1
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = k; i < size; i += nteams * 64)
#else
#define _BIG_JUMP_LOOP(nteams, size, stride, offset)                           \
  for (int64_t i = ((k * stride) + offset); i < size;                          \
       i += (nteams * 64 * stride))
#endif

#endif // end  if _XTEAM_NUM_THREADS == 1024, elif,elif ..  else
#endif // if defined(__NVPTX__) && _XTEAM_NUM_THREADS == 1024 else

unsigned int test_run_rc = 0;

template <typename T, bool> void run_tests(const uint64_t);
template <typename TC, typename T> void run_tests_complex(const uint64_t);

int main(int argc, char *argv[]) {
  std::cout << std::endl
            << "TEST DOUBLE " << _XTEAM_NUM_THREADS << " THREADS" << std::endl;
  run_tests<double, false>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST FLOAT " << _XTEAM_NUM_THREADS << " THREADS" << std::endl;
  run_tests<float, false>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST INT " << _XTEAM_NUM_THREADS << " THREADS" << std::endl;
  run_tests<int, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST UNSIGNED INT " << _XTEAM_NUM_THREADS << " THREADS"
            << std::endl;
  run_tests<unsigned, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST LONG " << _XTEAM_NUM_THREADS << " THREADS " << std::endl;
  run_tests<long, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST UNSIGNED LONG " << _XTEAM_NUM_THREADS << " THREADS"
            << std::endl;
  run_tests<unsigned long, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST DOUBLE COMPLEX " << _XTEAM_NUM_THREADS << " THREADS"
            << std::endl;
  run_tests_complex<double _Complex, double>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST FLOAT COMPLEX " << _XTEAM_NUM_THREADS << " THREADS"
            << std::endl;
  run_tests_complex<float _Complex, float>(ARRAY_SIZE);
  if (test_run_rc == 0)
    printf("ALL TESTS PASSED\n");
  return test_run_rc;
}

template <typename T> T omp_dot(T *a, T *b, uint64_t array_size) {
  T sum = 0.0;
#pragma omp target teams distribute parallel for map(tofrom: sum) reduction(+:sum)
  for (int64_t i = 0; i < array_size; i++)
    sum += a[i] * b[i];
  return sum;
}

template <typename T> T omp_max(T *c, uint64_t array_size) {
  T maxval = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : maxval)                 \
    reduction(max                                                              \
              : maxval)
  for (int64_t i = 0; i < array_size; i++)
    maxval = (c[i] > maxval) ? c[i] : maxval;
  return maxval;
}

template <typename T> T omp_min(T *c, uint64_t array_size) {
  T minval = std::numeric_limits<T>::max();
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : minval)                 \
    reduction(min                                                              \
              : minval)
  for (int64_t i = 0; i < array_size; i++) {
    minval = (c[i] < minval) ? c[i] : minval;
  }
  return minval;
}

template <typename T> T sim_dot(T *a, T *b, int warp_size) {
  T sum = T(0);
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    const int64_t stride = 1; // stride to process input vectors
    const int64_t offset = 0; // Offset to initial index of input vectors
    const int64_t size = _ARRAY_SIZE; // Size of input vector
    const T rnv = T(0);               // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc0;
  static int64_t num_teams0 = 0;
  if (!num_teams0) {
    // num_teams0    = ompx_get_device_num_units(devid);
    num_teams0 = _XTEAM_NUM_TEAMS;
    lc0.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc0.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams0, devid);
    omp_target_memcpy(lc0.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
  }

  if (warp_size == 64) {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : sum) map(to                          \
                                                   : lc0)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val0 = lc0.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc0.size, lc0.stride, lc0.offset)
      val0 += a[i] * b[i];
      _SUM_OVERLOAD_64_FCT(val0, &sum, lc0.team_vals, lc0.td_ptr, lc0.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  } else {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : sum) map(to                          \
                                                   : lc0)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val0 = lc0.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc0.size, lc0.stride, lc0.offset)
      val0 += a[i] * b[i];
      _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(_XTEAM_NUM_TEAMS)
      _SUM_OVERLOAD_32_FCT(val0, &sum, lc0.team_vals, lc0.td_ptr, lc0.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  }
  return sum;
}

template <typename T> T sim_max(T *c, int warp_size) {
  T retval = std::numeric_limits<T>::lowest();
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;                 // Atomic counter accessed on device
    uint32_t reserved;                // reserved
    const int64_t stride = 1;         // stride to process input vectors
    const int64_t offset = 0;         // Offset to index of input vectors
    const int64_t size = _ARRAY_SIZE; // Size of input vectors
    T rnv;                            // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc1;
  static int64_t num_teams1 = 0;
  if (!num_teams1) {
    // num_teams1    = ompx_get_device_num_units(devid);
    num_teams1 = _XTEAM_NUM_TEAMS;
    lc1.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc1.rnv = std::numeric_limits<T>::lowest();
    lc1.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams1, devid);
    omp_target_memcpy(lc1.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
  }
  if (warp_size == 64) {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : retval) map(to                       \
                                                      : lc1)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val1 = lc1.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc1.size, lc1.stride, lc1.offset)
      val1 = (c[i] > val1) ? c[i] : val1;
      _MAX_OVERLOAD_64_FCT(val1, &retval, lc1.team_vals, lc1.td_ptr, lc1.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  } else {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : retval) map(to                       \
                                                      : lc1)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val1 = lc1.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc1.size, lc1.stride, lc1.offset)
      val1 = (c[i] > val1) ? c[i] : val1;
      _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(_XTEAM_NUM_TEAMS)
      _MAX_OVERLOAD_32_FCT(val1, &retval, lc1.team_vals, lc1.td_ptr, lc1.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  }
  return retval;
}

template <typename T> T sim_min(T *c, int warp_size) {
  T retval = std::numeric_limits<T>::max();
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    const int64_t stride = 1; // stride to process input vectors
    const int64_t offset = 0; // Offset to initial index of input vectors
    const int64_t size = _ARRAY_SIZE; // Size of input vectors
    T rnv;                            // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc2;
  static int64_t num_teams2 = 0;
  if (!num_teams2) {
    // num_teams2    = ompx_get_device_num_units(devid);
    num_teams2 = _XTEAM_NUM_TEAMS;
    lc2.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc2.rnv = std::numeric_limits<T>::max();
    lc2.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams2, devid);
    omp_target_memcpy(lc2.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
  }
  if (warp_size == 64) {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : retval) map(to                       \
                                                      : lc2)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val2 = lc2.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc2.size, lc2.stride, lc2.offset)
      val2 = (c[i] < val2) ? c[i] : val2;
      _MIN_OVERLOAD_64_FCT(val2, &retval, lc2.team_vals, lc2.td_ptr, lc2.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  } else {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : retval) map(to                       \
                                                      : lc2)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val2 = lc2.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc2.size, lc2.stride, lc2.offset)
      val2 = (c[i] < val2) ? c[i] : val2;
      _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(_XTEAM_NUM_TEAMS)
      _MIN_OVERLOAD_32_FCT(val2, &retval, lc2.team_vals, lc2.td_ptr, lc2.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  }
  return retval;
}

template <typename T, bool DATA_TYPE_IS_INT>
void _check_val(T computed_val, T gold_val, const char *msg) {
  double ETOL = 0.0000001;
  if (DATA_TYPE_IS_INT) {
    if (computed_val != gold_val) {
      std::cerr << msg << " FAIL "
                << "Integar Value was " << computed_val << " but should be "
                << gold_val << std::endl;
      test_run_rc = 1;
    }
  } else {
    double dcomputed_val = (double)computed_val;
    double dvalgold = (double)gold_val;
    double ompErrSum = abs((dcomputed_val - dvalgold) / dvalgold);
    if (ompErrSum > ETOL) {
      std::cerr << msg << " FAIL " << ompErrSum << " tol:" << ETOL << std::endl
                << std::setprecision(15) << "Value was " << computed_val
                << " but should be " << gold_val << std::endl;
      test_run_rc = 1;
    }
  }
}

#define ALIGNMENT (2 * 1024 * 1024)

template <typename T, bool DATA_TYPE_IS_INT>
void run_tests(uint64_t array_size) {

  // FIXME: How do we get warpsize of a device from host?
  int warp_size = 64;
#pragma omp target map(tofrom : warp_size)
  warp_size = __kmpc_get_warp_size();

  //  Align on 2M boundaries
  T *a = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *b = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *c = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
#pragma omp target enter data map(alloc                                        \
                                  : a [0:array_size], b [0:array_size],        \
                                    c [0:array_size])
#pragma omp target teams distribute parallel for
  for (int64_t i = 0; i < array_size; i++) {
    a[i] = 2;
    b[i] = 3;
    c[i] = (i + 1);
  }

  std::cout << "Running kernels " << repeat_num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_times << "  runs "
            << std::endl;

  double ETOL = 0.0000001;
  if (DATA_TYPE_IS_INT) {
    std::cout << "Integer Size: " << sizeof(T) << std::endl;
  } else {
    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;
  }

  std::cout << "Warp size:" << warp_size << std::endl;
  // int num_teams = ompx_get_device_num_units(omp_get_default_device());
  int num_teams = _XTEAM_NUM_TEAMS;
  std::cout << "Array elements: " << array_size << std::endl;
  std::cout << "Array size:     " << ((array_size * sizeof(T)) / (1024 * 1024))
            << " MB" << std::endl;

  T goldDot = (T)6 * (T)array_size;
  T goldMax = (T)array_size;
  T goldMin = (T)1;

  double goldDot_d = (double)goldDot;
  double goldMax_d = (double)goldMax;
  double goldMin_d = (double)goldMin;

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Timing loop
  for (unsigned int k = 0; k < repeat_num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T omp_sum = omp_dot<T>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_sum, goldDot, "omp_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_sum = sim_dot<T>(a, b, warp_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_sum, goldDot, "sim_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_max_val = omp_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_max_val, goldMax, "omp_max");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_max_val = sim_max<T>(c, warp_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_max_val, goldMax, "sim_max");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_min_val = omp_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_min_val, goldMin, "omp_min");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_min_val = sim_min<T>(c, warp_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_min_val, goldMin, "sim_min");

  } // end Timing loop

  // Display timing results
  std::cout << std::left << std::setw(12) << "Function" << std::left
            << std::setw(12) << "Best-MB/sec" << std::left << std::setw(12)
            << " Min (sec)" << std::left << std::setw(12) << "   Max"
            << std::left << std::setw(12) << "Average" << std::left
            << std::setw(12) << "Avg-MB/sec" << std::endl;

  std::cout << std::fixed;

  std::string labels[6] = {"ompdot", "simdot", "ompmax",
                           "simmax", "ompmin", "simmin"};
  size_t sizes[6] = {2 * sizeof(T) * array_size, 2 * sizeof(T) * array_size,
                     1 * sizeof(T) * array_size, 1 * sizeof(T) * array_size,
                     1 * sizeof(T) * array_size, 1 * sizeof(T) * array_size};

  for (int i = 0; i < 6; i++) {
    // Get min/max; ignore the first couple results
    auto minmax = std::minmax_element(timings[i].begin() + ignore_times,
                                      timings[i].end());
    // Calculate average; ignore ignore_times
    double average = std::accumulate(timings[i].begin() + ignore_times,
                                     timings[i].end(), 0.0) /
                     (double)(repeat_num_times - ignore_times);
    printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n",
           labels[i].c_str(), 1.0E-6 * sizes[i] / (*minmax.first),
           (double)*minmax.first, (double)*minmax.second, (double)average,
           1.0E-6 * sizes[i] / (average));
  }
#pragma omp target exit data map(release                                       \
                                 : a [0:array_size], b [0:array_size],         \
                                   c [0:array_size])
  free(a);
  free(b);
  free(c);
}

template <typename TC, typename T>
void _check_val_complex(TC computed_val_complex, TC gold_val_complex,
                        const char *msg) {
  T gold_val_r = __real__(gold_val_complex);
  T computed_val_r = __real__(computed_val_complex);
  T gold_val_i = __imag__(gold_val_complex);
  T computed_val_i = __imag__(computed_val_complex);
  double ETOL = 0.0000001;
  double computed_val_r_d = (double)computed_val_r;
  double valgold_r_d = (double)gold_val_r;
  double ompErrSum_r = abs((computed_val_r_d - valgold_r_d) / valgold_r_d);
  double computed_val_i_d = (double)computed_val_i;
  double valgold_i_d = (double)gold_val_i;
  double ompErrSum_i = abs((computed_val_i_d - valgold_i_d) / valgold_i_d);
  if ((ompErrSum_r > ETOL) || (ompErrSum_i > ETOL)) {
    std::cerr << msg << " FAIL " << ompErrSum_r << " tol:" << ETOL << std::endl
              << std::setprecision(15) << "Value was (" << computed_val_r
              << " + " << computed_val_i << " i )" << std::endl
              << " but should be (" << gold_val_r << " + " << gold_val_i
              << "i) " << std::endl;
    test_run_rc = 1;
  }
}

template <typename TC> TC omp_dot_complex(TC *a, TC *b, uint64_t array_size) {
  TC dot;
  __real__(dot) = 0.0;
  __imag__(dot) = 0.0;
#pragma omp target teams distribute parallel for map(tofrom: dot) reduction(+:dot)
  for (int64_t i = 0; i < array_size; i++)
    dot += a[i] * b[i];
  return dot;
}

template <typename T> T sim_dot_complex(T *a, T *b, int warp_size) {
  int devid = 0;
  T zero_c;
  __real__(zero_c) = 0.0;
  __imag__(zero_c) = 0.0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    const int64_t stride = 1; // stride to process input vectors
    const int64_t offset = 0; // Offset to initial index of input vectors
    const int64_t size = _ARRAY_SIZE; // Size of input vectors
    T rnv;                            // reduction null value
    T *team_vals;                     // array of global team values
  };
  T sum = zero_c;
  uint32_t zero = 0;
  static loop_ctl_t lc3;
  static int64_t num_teams3 = 0;
  if (!num_teams3) {
    // num_teams3    = ompx_get_device_num_units(devid);
    num_teams3 = _XTEAM_NUM_TEAMS;
    lc3.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc3.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams3, devid);
    lc3.rnv = zero_c;
    omp_target_memcpy(lc3.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
  }

  if (warp_size == 64) {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : sum) map(to                          \
                                                   : lc3)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val3 = lc3.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc3.size, lc3.stride, lc3.offset)
      val3 += a[i] * b[i];
      _SUM_OVERLOAD_64_FCT(val3, &sum, lc3.team_vals, lc3.td_ptr, lc3.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  } else {
#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : sum) map(to                          \
                                                   : lc3)
    for (uint64_t k = 0; k < (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS); k++) {
      T val3 = lc3.rnv;
      _BIG_JUMP_LOOP(_XTEAM_NUM_TEAMS, lc3.size, lc3.stride, lc3.offset)
      val3 += a[i] * b[i];
      _LIMIT_JUMP_TO_CUDA_REDUCED_THREADS(_XTEAM_NUM_TEAMS)
      _SUM_OVERLOAD_32_FCT(val3, &sum, lc3.team_vals, lc3.td_ptr, lc3.rnv, k,
                           _XTEAM_NUM_TEAMS);
    }
  }
  return sum;
}

template <typename TC, typename T>
void run_tests_complex(const uint64_t array_size) {

  // FIXME: How do we get warpsize of a device from host?
  int warp_size = 64;
#pragma omp target map(tofrom : warp_size)
  warp_size = __kmpc_get_warp_size();

  TC *a = (TC *)aligned_alloc(ALIGNMENT, sizeof(TC) * array_size);
  TC *b = (TC *)aligned_alloc(ALIGNMENT, sizeof(TC) * array_size);

#pragma omp target enter data map(alloc : a [0:array_size], b [0:array_size])
  TC startA;
  __real__(startA) = 1.0;
  __imag__(startA) = 1.0;
  TC startB;
  __real__(startB) = 1.0;
  __imag__(startB) = -1.0;

#pragma omp target teams distribute parallel for
  for (int64_t i = 0; i < array_size; i++) {
    a[i] = startA;
    b[i] = startB;
    // a[i] * b[i] = 2 + 0i
  }

  std::cout << "Running kernels " << repeat_num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_times << "  runs "
            << std::endl;

  double ETOL = 0.0000001;
  if (sizeof(TC) == sizeof(float _Complex))
    std::cout << "Precision: float _Complex" << std::endl;
  else
    std::cout << "Precision: double _Complex" << std::endl;

  std::cout << "Warp size:" << warp_size << std::endl;
  std::cout << "Array elements: " << array_size << std::endl;
  std::cout << "Array size:     " << ((array_size * sizeof(TC)) / (1024 * 1024))
            << " MB" << std::endl;

  T goldDotr = T(2) * (T)array_size;
  T goldDoti = T(0);

  TC goldDot;
  __real__(goldDot) = goldDotr;
  __imag__(goldDot) = goldDoti;

  // List of times
  std::vector<std::vector<double>> timings(2);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // timing loop
  for (unsigned int k = 0; k < repeat_num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    TC omp_sum = omp_dot_complex<TC>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val_complex<TC, T>(omp_sum, goldDot, "omp_dot");

    t1 = std::chrono::high_resolution_clock::now();
    TC sim_sum = sim_dot_complex<TC>(a, b, warp_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val_complex<TC, T>(sim_sum, goldDot, "sim_dot");

  } // end timing loop

  // Display timing results
  std::cout << std::left << std::setw(12) << "Function" << std::left
            << std::setw(12) << "Best-MB/sec" << std::left << std::setw(12)
            << " Min (sec)" << std::left << std::setw(12) << "   Max"
            << std::left << std::setw(12) << "Average" << std::left
            << std::setw(12) << "Avg-MB/sec" << std::endl;

  std::cout << std::fixed;

  std::string labels[2] = {"ompdot", "simdot"};
  size_t sizes[2] = {2 * sizeof(TC) * array_size, 2 * sizeof(TC) * array_size};

  for (int i = 0; i < 2; i++) {
    // Get min/max; ignore the first couple results
    auto minmax = std::minmax_element(timings[i].begin() + ignore_times,
                                      timings[i].end());

    // Calculate average; ignore ignore_times
    double average = std::accumulate(timings[i].begin() + ignore_times,
                                     timings[i].end(), 0.0) /
                     (double)(repeat_num_times - ignore_times);

    printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n",
           labels[i].c_str(), 1.0E-6 * sizes[i] / (*minmax.first),
           (double)*minmax.first, (double)*minmax.second, (double)average,
           1.0E-6 * sizes[i] / (average));
  }
#pragma omp target exit data map(release : a [0:array_size], b [0:array_size])
  free(a);
  free(b);
}
