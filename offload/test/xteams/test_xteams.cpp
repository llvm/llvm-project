//===----- test_xteams.cpp - Test for Xteams DeviceRTL functions ---C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// performance and functional tests for Xteams scan helper functions in
// libomptarget/DeviceRTL/Xteams.cpp
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

#include "test_xteams.h"

#ifndef _ARRAY_SIZE
#define _ARRAY_SIZE 33554432
#endif
const uint64_t ARRAY_SIZE = _ARRAY_SIZE;
unsigned int repeat_num_times = 12;
unsigned int ignore_times = 2; // ignore this many timings first

#define ALIGNMENT (128)

//  Extern Xteams functions are designed for 1024, 512, 256 and 128 team sizes.
//  The default here is 512.

// Represents the Team Size
#ifndef _XTEAM_NUM_THREADS
#define _XTEAM_NUM_THREADS 512
#endif

// Represents the Number of Teams
#ifndef _XTEAM_NUM_TEAMS
#define _XTEAM_NUM_TEAMS 4
#endif

// Represents the total of threads in the Grid
#define _XTEAM_TOTAL_NUM_THREADS (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS)

#if _XTEAM_NUM_THREADS == 1024
#define _SUM_OVERLOAD_64_SCAN _overload_to_extern_scan_sum_16x64
#define _MAX_OVERLOAD_64_SCAN _overload_to_extern_scan_max_16x64
#define _MIN_OVERLOAD_64_SCAN _overload_to_extern_scan_min_16x64
#define _SUM_OVERLOAD_32_SCAN _overload_to_extern_scan_sum_32x32
#define _MAX_OVERLOAD_32_SCAN _overload_to_extern_scan_max_32x32
#define _MIN_OVERLOAD_32_SCAN _overload_to_extern_scan_min_32x32
#elif _XTEAM_NUM_THREADS == 512
#define _SUM_OVERLOAD_64_SCAN _overload_to_extern_scan_sum_8x64
#define _MAX_OVERLOAD_64_SCAN _overload_to_extern_scan_max_8x64
#define _MIN_OVERLOAD_64_SCAN _overload_to_extern_scan_min_8x64
#define _SUM_OVERLOAD_32_SCAN _overload_to_extern_scan_sum_16x32
#define _MAX_OVERLOAD_32_SCAN _overload_to_extern_scan_max_16x32
#define _MIN_OVERLOAD_32_SCAN _overload_to_extern_scan_min_16x32
#elif _XTEAM_NUM_THREADS == 256
#define _SUM_OVERLOAD_64_SCAN _overload_to_extern_scan_sum_4x64
#define _MAX_OVERLOAD_64_SCAN _overload_to_extern_scan_max_4x64
#define _MIN_OVERLOAD_64_SCAN _overload_to_extern_scan_min_4x64
#define _SUM_OVERLOAD_32_SCAN _overload_to_extern_scan_sum_8x32
#define _MAX_OVERLOAD_32_SCAN _overload_to_extern_scan_max_8x32
#define _MIN_OVERLOAD_32_SCAN _overload_to_extern_scan_min_8x32
#elif _XTEAM_NUM_THREADS == 128
#define _SUM_OVERLOAD_64_SCAN _overload_to_extern_scan_sum_2x64
#define _MAX_OVERLOAD_64_SCAN _overload_to_extern_scan_max_2x64
#define _MIN_OVERLOAD_64_SCAN _overload_to_extern_scan_min_2x64
#define _SUM_OVERLOAD_32_SCAN _overload_to_extern_scan_sum_4x32
#define _MAX_OVERLOAD_32_SCAN _overload_to_extern_scan_max_4x32
#define _MIN_OVERLOAD_32_SCAN _overload_to_extern_scan_min_4x32
#else
#error Invalid value for _XTEAM_NUM_THREADS.  Must be 1024, 512, 256 or 128
#endif

unsigned int test_run_rc = 0;

template <typename T, bool> void run_tests(const uint64_t);

int main(int argc, char *argv[]) {
  std::cout << std::endl
            << "TEST INT " << _XTEAM_NUM_THREADS << " THREADS" << std::endl;
  run_tests<int, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST UNSIGNED INT " << _XTEAM_NUM_THREADS << " THREADS"  
            << std::endl;
  run_tests<unsigned, true>(ARRAY_SIZE);
  if (test_run_rc == 0)
    printf("ALL TESTS PASSED\n");
  return test_run_rc;
}

// FIXME: Template function for omp_dot doesn't compile. Therefore pragmas are commented.
// Therefore `omp_dot` essentially represents sequential execution on host.
template <typename T> T* omp_dot(T *a, T *b, uint64_t array_size) {
  T* dot_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T sum = 0;
  // #pragma omp parallel for reduction(inscan, +:sum)
  for (int64_t i = 0; i < array_size; i++ ) {
    sum += a[i] * b[i];
    // #pragma omp scan inclusive(sum)
    dot_arr[i] = sum;
  }
  return dot_arr;
}

// FIXME: Template function for omp_max doesn't compile. Therefore pragmas are commented.
// Therefore `omp_max` essentially represents sequential execution on host.
template <typename T> T* omp_max(T *a, uint64_t array_size) {
  T* max_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T max_val = std::numeric_limits<T>::lowest();
  // #pragma omp parallel for reduction(inscan, max:max_val)
  for (uint64_t i = 0; i < array_size; i++ ) {
    max_val = std::max(a[i], max_val);
    // #pragma omp scan inclusive(max_val)
    max_arr[i] = max_val;
  }
  return max_arr;
}

// FIXME: Template function for omp_min doesn't compile. Therefore pragmas are commented.
// Therefore `omp_min` essentially represents sequential execution on host.
template <typename T> T* omp_min(T *a, uint64_t array_size) {
  T* min_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T min_val = std::numeric_limits<T>::max();
  // #pragma omp parallel for reduction(inscan, min:min_val)
  for (uint64_t i = 0; i < array_size; i++ ) {
    min_val = std::min(a[i], min_val);
    // #pragma omp scan inclusive(min_val)
    min_arr[i] = min_val;
  }
  return min_arr;
}

// Simulates the reduction operator `+` for a scan operation by making use of
// the `scan` directive of OpenMP. The dot product of a[] and b[] are computed
// and the result is verified along with an output containting time taken and
// bandwidth calculated.
template <typename T> T* sim_dot(T *a, T *b, int warp_size, uint64_t array_size) {
  T *dot = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size); // the output array
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    T* prev_reduction;        // Reduced value from the kernel launch of the prev iteration
    uint64_t stride = 1;      // stride to process input vectors
    const uint64_t offset = 0;        // Offset to initial index of input vectors
    uint64_t size;                    // Size of input vector
    const T rnv = T(0);               // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc0;
  lc0.size = array_size;
  static int64_t num_teams0 = 0;
  if (!num_teams0) {
    // num_teams0    = ompx_get_device_num_units(devid);
    num_teams0 = _XTEAM_NUM_TEAMS;
    lc0.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc0.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams0, devid);
    lc0.prev_reduction = (T*) omp_target_alloc(sizeof(T), devid);
    omp_target_memcpy(lc0.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
    omp_target_memcpy(lc0.prev_reduction, &lc0.rnv, sizeof(T), 0, 0, devid,
                      omp_get_initial_device());
  }

  // shared storage across all threads for double buffering to work in the First Kernel
  T* storage = (T *)omp_target_alloc(sizeof(T) * (2*_XTEAM_TOTAL_NUM_THREADS + 1), devid);
  #pragma omp target data map(tofrom: dot[0:array_size]) map(tofrom: lc0, storage)
  {
    // First Kernel: Computes the Intra Team Scan and calculates the scan of the
    // Team level values into the lc0.team_vals[] array.
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      lc0.stride = array_size / _XTEAM_TOTAL_NUM_THREADS;

      // compute scan serially per thread instead of launching multiple
      // kernels sequentially
      // FIXME: Replace T(0) with `lc0.rnv` to make it generic to any rnv
      T val0 = T(0); 
      for(uint64_t i = 0; 
          i < lc0.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc0.stride+i < array_size));
          i++) {
        val0 += a[k*lc0.stride+i] * b[k*lc0.stride+i];
        dot[k*lc0.stride+i] = val0;
      }
      storage[k] = val0; // Reduction is performed on this segment level value: val0
      if (warp_size == 64) // for amdgpu
        _SUM_OVERLOAD_64_SCAN(val0, storage, dot, lc0.team_vals, lc0.td_ptr, lc0.rnv, 
                            k, _XTEAM_NUM_TEAMS);
      else  // for nvptx machines
        _SUM_OVERLOAD_32_SCAN(val0, storage, dot, lc0.team_vals, lc0.td_ptr, lc0.rnv, 
                            k, _XTEAM_NUM_TEAMS);
    }

    // Second Kernel: Distributes the results of Scan computed at both the team
    // level as well as the segment level to the corresponding teams and segments
    // in their respective contexts. 
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      const uint32_t omp_team_num = k / _XTEAM_NUM_THREADS;    // team ID

      // team ID of previous stride
      const uint32_t prev_stride_team_num = (k-1) / _XTEAM_NUM_THREADS;    
      
      // team level scan of previous team
      const T prev_team_result = omp_team_num 
                                ? lc0.team_vals[omp_team_num - 1] 
                                : lc0.rnv; 

      // result of previous stride in first level scan                          
      const T prev_stride_result = (k && (omp_team_num == prev_stride_team_num)) 
                                  ? storage[k-1] 
                                  : lc0.rnv ;    

      // redistribution of the scanned result back to output array `dot`
      for(uint64_t i = 0; 
          i < lc0.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc0.stride+i < array_size));
          i++) {
        dot[k*lc0.stride+i] += (prev_team_result + prev_stride_result);
      }
    }
  }
  return dot;
}


template <typename T> T* sim_max(T *c, int warp_size, uint64_t array_size) {
  T *scanned_max = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size); // the output array
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    T* prev_reduction;        // Reduced value from the kernel launch of the prev iteration
    uint64_t stride = 1; // stride to process input vectors
    const uint64_t offset = 0; // Offset to initial index of input vectors
    uint64_t size; // Size of input vector
    const T rnv = std::numeric_limits<T>::lowest();         // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc1;
  lc1.size = array_size;
  static int64_t num_teams1 = 0;
  if (!num_teams1) {
    // num_teams1    = ompx_get_device_num_units(devid);
    num_teams1 = _XTEAM_NUM_TEAMS;
    lc1.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc1.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams1, devid);
    lc1.prev_reduction = (T*) omp_target_alloc(sizeof(T), devid);
    omp_target_memcpy(lc1.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
    omp_target_memcpy(lc1.prev_reduction, &lc1.rnv, sizeof(T), 0, 0, devid,
                      omp_get_initial_device());
  }

  // shared storage across all threads for double buffering to work in the First Kernel
  T* storage = (T *)omp_target_alloc(sizeof(T) * (2*_XTEAM_TOTAL_NUM_THREADS + 1), devid);
  #pragma omp target data map(tofrom: scanned_max[0:array_size]) map(tofrom: lc1, storage)
  {
    // First Kernel: Computes the Intra Team Scan and calculates the scan of the
    // Team level values into the lc1.team_vals[] array.
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      lc1.stride = array_size / _XTEAM_TOTAL_NUM_THREADS;

      // compute scan serially per thread instead of launching multiple
      // kernels sequentially
      T val0 = std::numeric_limits<T>::lowest(); 
      for(uint64_t i = 0; 
          i < lc1.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc1.stride+i < array_size));
          i++) {
        val0 = std::max(val0, c[k*lc1.stride+i]);
        scanned_max[k*lc1.stride+i] = val0;
      }
      storage[k] = val0; // Reduction is performed on this segment level value: val0
      if (warp_size == 64) 
        _MAX_OVERLOAD_64_SCAN(val0, storage, scanned_max, lc1.team_vals, lc1.td_ptr, lc1.rnv, 
                            k, _XTEAM_NUM_TEAMS);
      else  // for nvptx machines
        _MAX_OVERLOAD_32_SCAN(val0, storage, scanned_max, lc1.team_vals, lc1.td_ptr, lc1.rnv, 
                            k, _XTEAM_NUM_TEAMS);
    }

    // Second Kernel: Distributes the results of Scan computed at both the team
    // level as well as the segment level to the corresponding teams and segments
    // in their respective contexts. 
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      const uint32_t omp_team_num = k / _XTEAM_NUM_THREADS;   // team ID

      // team ID of previous stride      
      const uint32_t prev_stride_team_num = (k-1) / _XTEAM_NUM_THREADS;    
      
      // team level scan of previous team
      const T prev_team_result = omp_team_num 
                                ? lc1.team_vals[omp_team_num - 1] 
                                : lc1.rnv; 
      
      // result of previous stride in first level scan                                
      const T prev_stride_result = (k && (omp_team_num == prev_stride_team_num)) 
                                  ? storage[k-1] 
                                  : lc1.rnv ;    

      // redistribution of the scanned result back to output array `scanned_max`
      for(uint64_t i = 0; 
          i < lc1.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc1.stride+i < array_size));
          i++) {
        scanned_max[k*lc1.stride+i] = std::max(scanned_max[k*lc1.stride+i], 
                                      std::max(prev_team_result, prev_stride_result));
      }
    }
  }
  return scanned_max;
}


template <typename T> T* sim_min(T *c, int warp_size, uint64_t array_size) {
  T* scanned_min = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size); // the output array
  int devid = 0;
  struct loop_ctl_t {
    uint32_t *td_ptr;         // Atomic counter accessed on device
    uint32_t reserved;        // reserved
    T* prev_reduction;        // Reduced value from the kernel launch of the prev iteration
    uint64_t stride = 1; // stride to process input vectors
    const uint64_t offset = 0; // Offset to initial index of input vectors
    uint64_t size; // Size of input vector
    const T rnv = std::numeric_limits<T>::max();         // reduction null value
    T *team_vals;                     // array of global team values
  };
  static uint32_t zero = 0;
  static loop_ctl_t lc2;
  static int64_t num_teams2 = 0;
  if (!num_teams2) {
    // num_teams2    = ompx_get_device_num_units(devid);
    num_teams2 = _XTEAM_NUM_TEAMS;
    lc2.td_ptr = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    lc2.team_vals = (T *)omp_target_alloc(sizeof(T) * num_teams2, devid);
    lc2.prev_reduction = (T*) omp_target_alloc(sizeof(T), devid);
    omp_target_memcpy(lc2.td_ptr, &zero, sizeof(uint32_t), 0, 0, devid,
                      omp_get_initial_device());
    omp_target_memcpy(lc2.prev_reduction, &lc2.rnv, sizeof(T), 0, 0, devid,
                      omp_get_initial_device());
  }

  // shared storage across all threads for double buffering to work in the First Kernel
  T* storage = (T *)omp_target_alloc(sizeof(T) * (2*_XTEAM_TOTAL_NUM_THREADS + 1), devid);
  #pragma omp target data map(tofrom: scanned_min[0:array_size]) map(tofrom: lc2, storage)
  {
    // First Kernel: Computes the Intra Team Scan and calculates the scan of the
    // Team level values into the lc2.team_vals[] array.
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      lc2.stride = array_size / _XTEAM_TOTAL_NUM_THREADS;

      // compute scan serially per thread instead of launching multiple
      // kernels sequentially
      T val0 = std::numeric_limits<T>::max(); 
      for(uint64_t i = 0; 
          i < lc2.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc2.stride+i < array_size));
          i++) {
        val0 = std::min(val0, c[k*lc2.stride+i]);
        scanned_min[k*lc2.stride+i] = val0;
      }
      storage[k] = val0; // Reduction is performed on this segment level value: val0
      if (warp_size == 64) 
        _MIN_OVERLOAD_64_SCAN(val0, storage, scanned_min, lc2.team_vals, lc2.td_ptr, lc2.rnv, 
                            k, _XTEAM_NUM_TEAMS);
      else  // for nvptx machines
        _MIN_OVERLOAD_32_SCAN(val0, storage, scanned_min, lc2.team_vals, lc2.td_ptr, lc2.rnv, 
                            k, _XTEAM_NUM_TEAMS);
    }

    // Second Kernel: Distributes the results of Scan computed at both the team
    // level as well as the segment level to the corresponding teams and segments
    // in their respective contexts. 
    #pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS) \
                                          num_threads(_XTEAM_NUM_THREADS) 
    for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
      // Every thread processes one segment of `stride` size
      const uint32_t omp_team_num = k / _XTEAM_NUM_THREADS;    // team ID

      // team ID of previous stride      
      const uint32_t prev_stride_team_num = (k-1) / _XTEAM_NUM_THREADS;    
      
      // team level scan of previous team
      const T prev_team_result = omp_team_num 
                                ? lc2.team_vals[omp_team_num - 1] 
                                : lc2.rnv; 

      // result of previous stride in first level scan                                
      const T prev_stride_result = (k && (omp_team_num == prev_stride_team_num)) 
                                  ? storage[k-1] 
                                  : lc2.rnv ;    

      // redistribution of the scanned result back to output array `scanned_min`
      for(uint64_t i = 0; 
          i < lc2.stride || ((k == _XTEAM_TOTAL_NUM_THREADS - 1) 
          && (k*lc2.stride+i < array_size));
          i++) {
        scanned_min[k*lc2.stride+i] = std::min(scanned_min[k*lc2.stride+i], 
                                      std::min(prev_team_result, prev_stride_result));
      }
    }
  }
  return scanned_min;
}


// Sets test_run_rc if the computed_val[] is not same as the gold_val[]
template <typename T, bool DATA_TYPE_IS_INT>
void _check_val(T* computed_val, T* gold_val, const char *msg, uint64_t array_size) {
  double ETOL = 0.0000001; // Error Tolerance
  for(int i = 0; i < array_size; i++) {
    if (DATA_TYPE_IS_INT) {
      if (computed_val[i] != gold_val[i]) {
        std::cerr << msg << " FAIL at: " << i << ": Integer Value was " << 
                computed_val[i] << " but should be " << gold_val[i] << 
                ", type: " << typeid(T).name() << std::endl;
        test_run_rc = 1;
        break;
      }
    } else {
      double dcomputed_val = (double)computed_val[i];
      double dvalgold = (double)gold_val[i];
      double ompErrSum = abs((dcomputed_val - dvalgold) / dvalgold);
      if (ompErrSum > ETOL) {
        std::cerr << msg << " FAIL at: " << i << " tol:" << ETOL << std::endl
                  << std::setprecision(15) << ". Value was " << computed_val[i]
                  << " but should be " << gold_val[i] << ", type: " << typeid(T).name()
                  << std::endl;
        test_run_rc = 1;
        break;
      }
    }
  }
}


// Serially compute the correct scanned dot product output
template <typename T>
T* getGoldDot(T* a, T* b, uint64_t array_size) {
  T *goldDot = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for(uint64_t i = 0; i < array_size; i++) 
    goldDot[i] = i ? goldDot[i-1] + a[i]*b[i] : a[i]*b[i];
  return goldDot;
}

// Serially compute the correct scanned max output
template <typename T>
T* getGoldMax(T* a, uint64_t array_size) {
  T *goldMax = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for(uint64_t i = 0; i < array_size; i++) 
    goldMax[i] = i ? std::max(goldMax[i-1], a[i]) : a[i];
  return goldMax;
}

// Serially compute the correct scanned min output
template <typename T>
T* getGoldMin(T* a, uint64_t array_size) {
  T *goldMin = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for(uint64_t i = 0; i < array_size; i++) 
    goldMin[i] = i ? std::min(goldMin[i-1], a[i]) : a[i];
  return goldMin;
}

// Templated test launcher for array input of any datatype and size
template <typename T, bool DATA_TYPE_IS_INT>
void run_tests(uint64_t array_size) {
  int warp_size = 64;
  #pragma omp target map(tofrom : warp_size)
    warp_size = __kmpc_get_warp_size();

  srand(time(0));
  T *a = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *b = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *c = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for (int64_t i = 0; i < array_size; i++) {
    a[i] = T(2);
    b[i] = T(3);
    c[i] = rand() % (int)1e5;
  }
#pragma omp target enter data map(to: a[0:array_size], b[0:array_size], \
                                      c[0:array_size])

  std::cout << "Running kernels " << repeat_num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_times << "  runs "
            << std::endl;
  std::cout << "Integer Size: " << sizeof(T) << std::endl;
  std::cout << "Warp size:" << warp_size << std::endl;
  int num_teams = _XTEAM_NUM_TEAMS;
  std::cout << "Array elements: " << array_size << std::endl;
  std::cout << "Array size:     " << (double(array_size * sizeof(T)) / (1024 * 1024))
            << " MB" << std::endl;

  T* goldDot = getGoldDot(a, b, array_size);
  T* goldMax = getGoldMax(c, array_size);
  T* goldMin = getGoldMin(c, array_size);

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Timing loop
  for (unsigned int k = 0; k < repeat_num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T * omp_dot_arr = omp_dot(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_dot_arr, goldDot, "omp_dot", array_size);
    free(omp_dot_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T* sim_dot_arr = sim_dot<T>(a, b, warp_size, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_dot_arr, goldDot, "sim_dot", array_size);
    free(sim_dot_arr);
 
    t1 = std::chrono::high_resolution_clock::now();
    T* omp_max_arr = omp_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_max_arr, goldMax, "omp_max", array_size);
    free(omp_max_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T* sim_max_arr = sim_max<T>(c, warp_size, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_max_arr, goldMax, "sim_max", array_size);
    free(sim_max_arr);
    
    t1 = std::chrono::high_resolution_clock::now();
    T* omp_min_arr = omp_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_min_arr, goldMin, "omp_min", array_size);
    free(omp_min_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T* sim_min_arr = sim_min<T>(c, warp_size, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_min_arr, goldMin, "sim_min", array_size);
    free(sim_min_arr);
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

#pragma omp target exit data map(release: a[0:array_size], b[0:array_size], \
                                          c[0:array_size])
  free(goldDot);
  free(goldMax);
  free(goldMin);
  free(a);
  free(b);
  free(c);
}
