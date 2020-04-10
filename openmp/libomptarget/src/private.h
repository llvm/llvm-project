//===---------- private.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private function declarations and helper macros for debugging output.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_PRIVATE_H
#define _OMPTARGET_PRIVATE_H

#include <omptarget.h>

#include <cstdint>

extern int target_data_begin(DeviceTy &Device, int32_t arg_num,
                             void **args_base, void **args, int64_t *arg_sizes,
                             int64_t *arg_types,
                             __tgt_async_info *async_info_ptr);

extern int target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
                           void **args, int64_t *arg_sizes, int64_t *arg_types,
                           __tgt_async_info *async_info_ptr);

extern int target_data_update(DeviceTy &Device, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types);

extern int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct);

extern int CheckDeviceAndCtors(int64_t device_id);

// enum for OMP_TARGET_OFFLOAD; keep in sync with kmp.h definition
enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};
typedef enum kmp_target_offload_kind kmp_target_offload_kind_t;
extern kmp_target_offload_kind_t TargetOffloadPolicy;

// This structure stores information of a mapped memory region.
struct MapComponentInfoTy {
  void *Base;
  void *Begin;
  int64_t Size;
  int64_t Type;
  MapComponentInfoTy() = default;
  MapComponentInfoTy(void *Base, void *Begin, int64_t Size, int64_t Type)
      : Base(Base), Begin(Begin), Size(Size), Type(Type) {}
};

// This structure stores all components of a user-defined mapper. The number of
// components are dynamically decided, so we utilize C++ STL vector
// implementation here.
struct MapperComponentsTy {
  std::vector<MapComponentInfoTy> Components;
};

////////////////////////////////////////////////////////////////////////////////
// implementation for fatal messages
////////////////////////////////////////////////////////////////////////////////

#define FATAL_MESSAGE0(_num, _str)                                    \
  do {                                                                \
    fprintf(stderr, "Libomptarget fatal error %d: %s\n", _num, _str); \
    exit(1);                                                          \
  } while (0)

#define FATAL_MESSAGE(_num, _str, ...)                              \
  do {                                                              \
    fprintf(stderr, "Libomptarget fatal error %d:" _str "\n", _num, \
            __VA_ARGS__);                                           \
    exit(1);                                                        \
  } while (0)

// Implemented in libomp, they are called from within __tgt_* functions.
#ifdef __cplusplus
extern "C" {
#endif
// functions that extract info from libomp; keep in sync
int omp_get_default_device(void) __attribute__((weak));
int32_t __kmpc_omp_taskwait(void *loc_ref, int32_t gtid) __attribute__((weak));
int32_t __kmpc_global_thread_num(void *) __attribute__((weak));
int __kmpc_get_target_offload(void) __attribute__((weak));
#ifdef __cplusplus
}
#endif

#ifdef OMPTARGET_DEBUG
extern int DebugLevel;

#define DP(...) \
  do { \
    if (DebugLevel > 0) { \
      DEBUGP("Libomptarget", __VA_ARGS__); \
    } \
  } while (false)
#else // OMPTARGET_DEBUG
#define DP(...) {}
#endif // OMPTARGET_DEBUG

#endif
