//===-------- interface.cpp - Target independent OpenMP target RTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <cstdlib>
#include <mutex>

// Store target policy (disabled, mandatory, default)
kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
std::mutex TargetOffloadMtx;

////////////////////////////////////////////////////////////////////////////////
/// manage the success or failure of a target construct

static void HandleDefaultTargetOffload() {
  TargetOffloadMtx.lock();
  if (TargetOffloadPolicy == tgt_default) {
    if (omp_get_num_devices() > 0) {
      DP("Default TARGET OFFLOAD policy is now mandatory "
         "(devices were found)\n");
      TargetOffloadPolicy = tgt_mandatory;
    } else {
      DP("Default TARGET OFFLOAD policy is now disabled "
         "(no devices were found)\n");
      TargetOffloadPolicy = tgt_disabled;
    }
  }
  TargetOffloadMtx.unlock();
}

static int IsOffloadDisabled() {
  if (TargetOffloadPolicy == tgt_default) HandleDefaultTargetOffload();
  return TargetOffloadPolicy == tgt_disabled;
}

static void HandleTargetOutcome(bool success) {
  switch (TargetOffloadPolicy) {
    case tgt_disabled:
      if (success) {
        FATAL_MESSAGE0(1, "expected no offloading while offloading is disabled");
      }
      break;
    case tgt_default:
      FATAL_MESSAGE0(1, "default offloading policy must be switched to "
                        "mandatory or disabled");
      break;
    case tgt_mandatory:
      if (!success) {
        FATAL_MESSAGE0(1, "failure of target construct while offloading is mandatory");
      }
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t flags) {
  RTLs->RegisterRequires(flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  RTLs->RegisterLib(desc);
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  RTLs->UnregisterLib(desc);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_begin_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_begin_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN void __tgt_target_data_begin_mapper(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    void **arg_mappers) {
  if (IsOffloadDisabled()) return;

  DP("Entering data begin region for device %" PRId64 " with %d mappings\n",
      device_id, arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %" PRId64 "\n", device_id);
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DeviceTy &Device = Devices[device_id];

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 "\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i]);
  }
#endif

  int rc = targetDataBegin(Device, arg_num, args_base, args, arg_sizes,
                           arg_types, arg_mappers, nullptr);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_begin_nowait_mapper(int64_t device_id,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_begin_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, arg_mappers);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_end_mapper(device_id, arg_num, args_base, args, arg_sizes,
      arg_types, nullptr);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_end_mapper(device_id, arg_num, args_base, args, arg_sizes,
      arg_types, nullptr);
}

EXTERN void __tgt_target_data_end_mapper(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    void **arg_mappers) {
  if (IsOffloadDisabled()) return;
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  RTLsMtx->lock();
  size_t Devices_size = Devices.size();
  RTLsMtx->unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %" PRId64 " does not have a matching RTL.\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("Uninit device: ignore");
    HandleTargetOutcome(false);
    return;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(args_base[i]), DPxPTR(args[i]),
        arg_sizes[i], arg_types[i]);
  }
#endif

  int rc = targetDataEnd(Device, arg_num, args_base, args, arg_sizes, arg_types,
                         arg_mappers, nullptr);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_end_nowait_mapper(int64_t device_id,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_end_mapper(device_id, arg_num, args_base, args, arg_sizes,
      arg_types, arg_mappers);
}

EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_update_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN void __tgt_target_data_update_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_update_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN void __tgt_target_data_update_mapper(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    void **arg_mappers) {
  if (IsOffloadDisabled()) return;
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DeviceTy& Device = Devices[device_id];
  int rc = target_data_update(Device, arg_num, args_base,
      args, arg_sizes, arg_types, arg_mappers);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
}

EXTERN void __tgt_target_data_update_nowait_mapper(int64_t device_id,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_update_mapper(device_id, arg_num, args_base, args,
      arg_sizes, arg_types, arg_mappers);
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  return __tgt_target_mapper(device_id, host_ptr, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_mapper(device_id, host_ptr, arg_num, args_base, args,
      arg_sizes, arg_types, nullptr);
}

EXTERN int __tgt_target_mapper(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers) {
  if (IsOffloadDisabled()) return OFFLOAD_FAIL;
  DP("Entering target region with entry point " DPxMOD " and device Id %"
      PRId64 "\n", DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return OFFLOAD_FAIL;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(args_base[i]), DPxPTR(args[i]),
        arg_sizes[i], arg_types[i]);
  }
#endif

  int rc = target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
      arg_types, arg_mappers, 0, 0, false /*team*/);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);
  return rc;
}

EXTERN int __tgt_target_nowait_mapper(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_mapper(device_id, host_ptr, arg_num, args_base, args,
      arg_sizes, arg_types, arg_mappers);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  return __tgt_target_teams_mapper(device_id, host_ptr, arg_num, args_base,
      args, arg_sizes, arg_types, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_teams_mapper(device_id, host_ptr, arg_num, args_base,
      args, arg_sizes, arg_types, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_mapper(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t team_num, int32_t thread_limit) {
  if (IsOffloadDisabled()) return OFFLOAD_FAIL;
  DP("Entering target region with entry point " DPxMOD " and device Id %"
      PRId64 "\n", DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return OFFLOAD_FAIL;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(args_base[i]), DPxPTR(args[i]),
        arg_sizes[i], arg_types[i]);
  }
#endif

  int rc = target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
      arg_types, arg_mappers, team_num, thread_limit, true /*team*/);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS);

  return rc;
}

EXTERN int __tgt_target_teams_nowait_mapper(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, void **arg_mappers, int32_t team_num,
    int32_t thread_limit, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_teams_mapper(device_id, host_ptr, arg_num, args_base,
      args, arg_sizes, arg_types, arg_mappers, team_num, thread_limit);
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *rt_mapper_handle) {
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  int64_t size = MapperComponentsPtr->Components.size();
  DP("__tgt_mapper_num_components(Handle=" DPxMOD ") returns %" PRId64 "\n",
     DPxPTR(rt_mapper_handle), size);
  return size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *rt_mapper_handle, void *base,
                                        void *begin, int64_t size,
                                        int64_t type) {
  DP("__tgt_push_mapper_component(Handle=" DPxMOD
     ") adds an entry (Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
     ", Type=0x%" PRIx64 ").\n",
     DPxPTR(rt_mapper_handle), DPxPTR(base), DPxPTR(begin), size, type);
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(base, begin, size, type));
}

EXTERN void __kmpc_push_target_tripcount(int64_t device_id,
    uint64_t loop_tripcount) {
  if (IsOffloadDisabled())
    return;

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false);
    return;
  }

  DP("__kmpc_push_target_tripcount(%" PRId64 ", %" PRIu64 ")\n", device_id,
      loop_tripcount);
  TblMapMtx->lock();
  Devices[device_id].LoopTripCnt.emplace(__kmpc_global_thread_num(NULL),
                                         loop_tripcount);
  TblMapMtx->unlock();
}
