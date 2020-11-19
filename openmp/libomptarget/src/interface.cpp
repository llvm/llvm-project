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

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>

////////////////////////////////////////////////////////////////////////////////
/// manage the success or failure of a target construct
static void HandleDefaultTargetOffload() {
  PM->TargetOffloadMtx.lock();
  if (PM->TargetOffloadPolicy == tgt_default) {
    if (omp_get_num_devices() > 0) {
      DP("Default TARGET OFFLOAD policy is now mandatory "
         "(devices were found)\n");
      PM->TargetOffloadPolicy = tgt_mandatory;
    } else {
      DP("Default TARGET OFFLOAD policy is now disabled "
         "(no devices were found)\n");
      PM->TargetOffloadPolicy = tgt_disabled;
    }
  }
  PM->TargetOffloadMtx.unlock();
}

static int IsOffloadDisabled() {
  if (PM->TargetOffloadPolicy == tgt_default)
    HandleDefaultTargetOffload();
  return PM->TargetOffloadPolicy == tgt_disabled;
}

static void HandleTargetOutcome(bool success, ident_t *loc = nullptr) {
  switch (PM->TargetOffloadPolicy) {
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
      if (getInfoLevel() > 1)
        for (const auto &Device : PM->Devices)
          dumpTargetPointerMappings(Device);
      else
        FAILURE_MESSAGE("run with env LIBOMPTARGET_INFO>1 to dump host-target "
                        "pointer maps\n");

      SourceInfo info(loc);
      if (info.isAvailible())
        fprintf(stderr, "%s:%d:%d: ", info.getFilename(), info.getLine(),
                info.getColumn());
      else
        FAILURE_MESSAGE(
            "Build with debug information to provide more information");
      FATAL_MESSAGE0(
          1, "failure of target construct while offloading is mandatory");
    }
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t flags) {
  PM->RTLs.RegisterRequires(flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  PM->RTLs.RegisterLib(desc);
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  PM->RTLs.UnregisterLib(desc);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_begin_mapper(nullptr, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_begin_mapper(nullptr, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_mapper(ident_t *loc, int64_t device_id,
                                           int32_t arg_num, void **args_base,
                                           void **args, int64_t *arg_sizes,
                                           int64_t *arg_types,
                                           map_var_info_t *arg_names,
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
    HandleTargetOutcome(false, loc);
    return;
  }

  DeviceTy &Device = PM->Devices[device_id];

#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "(null)");
  }
#endif

  int rc = targetDataBegin(Device, arg_num, args_base, args, arg_sizes,
                           arg_types, arg_names, arg_mappers, nullptr);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);
}

EXTERN void __tgt_target_data_begin_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(loc, __kmpc_global_thread_num(loc));

  __tgt_target_data_begin_mapper(loc, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, arg_names, arg_mappers);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_end_mapper(nullptr, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_end_mapper(nullptr, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_mapper(ident_t *loc, int64_t device_id,
                                         int32_t arg_num, void **args_base,
                                         void **args, int64_t *arg_sizes,
                                         int64_t *arg_types,
                                         map_var_info_t *arg_names,
                                         void **arg_mappers) {
  if (IsOffloadDisabled()) return;
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  PM->RTLsMtx.lock();
  size_t DevicesSize = PM->Devices.size();
  PM->RTLsMtx.unlock();
  if (DevicesSize <= (size_t)device_id) {
    DP("Device ID  %" PRId64 " does not have a matching RTL.\n", device_id);
    HandleTargetOutcome(false, loc);
    return;
  }

  DeviceTy &Device = PM->Devices[device_id];
  if (!Device.IsInit) {
    DP("Uninit device: ignore");
    HandleTargetOutcome(false, loc);
    return;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "(null)");
  }
#endif

  int rc = targetDataEnd(Device, arg_num, args_base, args, arg_sizes, arg_types,
                         arg_names, arg_mappers, nullptr);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);
}

EXTERN void __tgt_target_data_end_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(loc, __kmpc_global_thread_num(loc));

  __tgt_target_data_end_mapper(loc, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  __tgt_target_data_update_mapper(nullptr, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  __tgt_target_data_update_mapper(nullptr, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_mapper(ident_t *loc, int64_t device_id,
                                            int32_t arg_num, void **args_base,
                                            void **args, int64_t *arg_sizes,
                                            int64_t *arg_types,
                                            map_var_info_t *arg_names,
                                            void **arg_mappers) {
  if (IsOffloadDisabled()) return;
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false, loc);
    return;
  }

  DeviceTy &Device = PM->Devices[device_id];
  int rc = targetDataUpdate(Device, arg_num, args_base, args, arg_sizes,
                            arg_types, arg_names, arg_mappers);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);
}

EXTERN void __tgt_target_data_update_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(loc, __kmpc_global_thread_num(loc));

  __tgt_target_data_update_mapper(loc, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  return __tgt_target_mapper(nullptr, device_id, host_ptr, arg_num, args_base,
                             args, arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_mapper(nullptr, device_id, host_ptr, arg_num, args_base,
                             args, arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN int __tgt_target_mapper(ident_t *loc, int64_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int64_t *arg_types,
                               map_var_info_t *arg_names, void **arg_mappers) {
  if (IsOffloadDisabled()) return OFFLOAD_FAIL;
  DP("Entering target region with entry point " DPxMOD " and device Id %"
      PRId64 "\n", DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    REPORT("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false, loc);
    return OFFLOAD_FAIL;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "(null)");
  }
#endif

  int rc = target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                  arg_types, arg_names, arg_mappers, 0, 0, false /*team*/);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);
  return rc;
}

EXTERN int __tgt_target_nowait_mapper(
    ident_t *loc, int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(loc, __kmpc_global_thread_num(loc));

  return __tgt_target_mapper(loc, device_id, host_ptr, arg_num, args_base, args,
                             arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  return __tgt_target_teams_mapper(nullptr, device_id, host_ptr, arg_num,
                                   args_base, args, arg_sizes, arg_types,
                                   nullptr, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, __kmpc_global_thread_num(NULL));

  return __tgt_target_teams_mapper(nullptr, device_id, host_ptr, arg_num,
                                   args_base, args, arg_sizes, arg_types,
                                   nullptr, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_mapper(ident_t *loc, int64_t device_id,
                                     void *host_ptr, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int64_t *arg_types,
                                     map_var_info_t *arg_names,
                                     void **arg_mappers, int32_t team_num,
                                     int32_t thread_limit) {
  if (IsOffloadDisabled()) return OFFLOAD_FAIL;
  DP("Entering target region with entry point " DPxMOD " and device Id %"
      PRId64 "\n", DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    REPORT("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false, loc);
    return OFFLOAD_FAIL;
  }

#ifdef OMPTARGET_DEBUG
  for (int i=0; i<arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "(null)");
  }
#endif

  int rc = target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                  arg_types, arg_names, arg_mappers, team_num, thread_limit,
                  true /*team*/);
  HandleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  return rc;
}

EXTERN int __tgt_target_teams_nowait_mapper(
    ident_t *loc, int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t team_num,
    int32_t thread_limit, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(loc, __kmpc_global_thread_num(loc));

  return __tgt_target_teams_mapper(loc, device_id, host_ptr, arg_num, args_base,
                                   args, arg_sizes, arg_types, arg_names,
                                   arg_mappers, team_num, thread_limit);
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

EXTERN void __kmpc_push_target_tripcount(ident_t *loc, int64_t device_id,
                                         uint64_t loop_tripcount) {
  if (IsOffloadDisabled())
    return;

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %" PRId64 " ready\n", device_id);
    HandleTargetOutcome(false, loc);
    return;
  }

  DP("__kmpc_push_target_tripcount(%" PRId64 ", %" PRIu64 ")\n", device_id,
      loop_tripcount);
  PM->TblMapMtx.lock();
  PM->Devices[device_id].LoopTripCnt.emplace(__kmpc_global_thread_num(NULL),
                                             loop_tripcount);
  PM->TblMapMtx.unlock();
}
