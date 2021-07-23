//===--- amdgpu/impl/internal.h ----------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SRC_RUNTIME_INCLUDE_INTERNAL_H_
#define SRC_RUNTIME_INCLUDE_INTERNAL_H_
#include <inttypes.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "hsa_api.h"

#include "atmi.h"
#include "atmi_runtime.h"
#include "rt.h"

#define MAX_NUM_KERNELS (1024 * 16)

typedef struct atmi_implicit_args_s {
  unsigned long offset_x;
  unsigned long offset_y;
  unsigned long offset_z;
  unsigned long hostcall_ptr;
  char num_gpu_queues;
  unsigned long gpu_queue_ptr;
  char num_cpu_queues;
  unsigned long cpu_worker_signals;
  unsigned long cpu_queue_ptr;
  unsigned long kernarg_template_ptr;
} atmi_implicit_args_t;

extern "C" {

#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...)                                                  \
  if (core::Runtime::getInstance().getDebugMode()) {                           \
    fprintf(stderr, "[%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__);        \
  }
#else
#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
  } while (false)
#endif

#ifndef HSA_RUNTIME_INC_HSA_H_
typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;
#endif

}

/* ---------------------------------------------------------------------------------
 * Simulated CPU Data Structures and API
 * ---------------------------------------------------------------------------------
 */

#define ATMI_WAIT_STATE HSA_WAIT_STATE_BLOCKED

// ---------------------- Kernel Start -------------
typedef struct atl_kernel_info_s {
  uint64_t kernel_object;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  uint32_t sgpr_count;
  uint32_t vgpr_count;
  uint32_t sgpr_spill_count;
  uint32_t vgpr_spill_count;
  uint32_t kernel_segment_size;
  uint32_t num_args;
  std::vector<uint64_t> arg_alignments;
  std::vector<uint64_t> arg_offsets;
  std::vector<uint64_t> arg_sizes;
} atl_kernel_info_t;

typedef struct atl_symbol_info_s {
  uint64_t addr;
  uint32_t size;
} atl_symbol_info_t;

// ---------------------- Kernel End -------------

namespace core {
class TaskgroupImpl;
class TaskImpl;
class Kernel;
class KernelImpl;
} // namespace core

struct SignalPoolT {
  SignalPoolT() {}
  SignalPoolT(const SignalPoolT &) = delete;
  SignalPoolT(SignalPoolT &&) = delete;
  ~SignalPoolT() {
    size_t N = state.size();
    for (size_t i = 0; i < N; i++) {
      hsa_signal_t signal = state.front();
      state.pop();
      hsa_status_t rc = hsa_signal_destroy(signal);
      if (rc != HSA_STATUS_SUCCESS) {
        DEBUG_PRINT("Signal pool destruction failed\n");
      }
    }
  }
  size_t size() {
    lock l(&mutex);
    return state.size();
  }
  void push(hsa_signal_t s) {
    lock l(&mutex);
    state.push(s);
  }
  hsa_signal_t pop(void) {
    lock l(&mutex);
    if (!state.empty()) {
      hsa_signal_t res = state.front();
      state.pop();
      return res;
    }

    // Pool empty, attempt to create another signal
    hsa_signal_t new_signal;
    hsa_status_t err = hsa_signal_create(0, 0, NULL, &new_signal);
    if (err == HSA_STATUS_SUCCESS) {
      return new_signal;
    }

    // Fail
    return {0};
  }

private:
  static pthread_mutex_t mutex;
  std::queue<hsa_signal_t> state;
  struct lock {
    lock(pthread_mutex_t *m) : m(m) { pthread_mutex_lock(m); }
    ~lock() { pthread_mutex_unlock(m); }
    pthread_mutex_t *m;
  };
};

namespace core {
hsa_status_t atl_init_gpu_context();

hsa_status_t init_hsa();
hsa_status_t finalize_hsa();
/*
 * Generic utils
 */
template <typename T> inline T alignDown(T value, size_t alignment) {
  return (T)(value & ~(alignment - 1));
}

template <typename T> inline T *alignDown(T *value, size_t alignment) {
  return reinterpret_cast<T *>(alignDown((intptr_t)value, alignment));
}

template <typename T> inline T alignUp(T value, size_t alignment) {
  return alignDown((T)(value + alignment - 1), alignment);
}

template <typename T> inline T *alignUp(T *value, size_t alignment) {
  return reinterpret_cast<T *>(
      alignDown((intptr_t)(value + alignment - 1), alignment));
}

extern bool atl_is_atmi_initialized();

bool handle_group_signal(hsa_signal_value_t value, void *arg);

hsa_status_t allow_access_to_all_gpu_agents(void *ptr);
} // namespace core

const char *get_error_string(hsa_status_t err);

#endif // SRC_RUNTIME_INCLUDE_INTERNAL_H_
