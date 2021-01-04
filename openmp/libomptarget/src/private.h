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

#include <Debug.h>
#include <SourceInfo.h>
#include <omptarget.h>

#include <cstdint>

extern int targetDataBegin(DeviceTy &Device, int32_t arg_num, void **args_base,
                           void **args, int64_t *arg_sizes, int64_t *arg_types,
                           map_var_info_t *arg_names, void **arg_mappers,
                           __tgt_async_info *async_info_ptr);

extern int targetDataEnd(DeviceTy &Device, int32_t ArgNum, void **ArgBases,
                         void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
                         map_var_info_t *arg_names, void **ArgMappers,
                         __tgt_async_info *AsyncInfo);

extern int targetDataUpdate(DeviceTy &Device, int32_t arg_num, void **args_base,
                            void **args, int64_t *arg_sizes, int64_t *arg_types,
                            map_var_info_t *arg_names, void **arg_mappers,
                            __tgt_async_info *async_info_ptr = nullptr);

extern int target(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *arg_names,
                  void **ArgMappers, int32_t TeamNum, int32_t ThreadLimit,
                  int IsTeamConstruct);

extern int CheckDeviceAndCtors(int64_t device_id);

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
  int32_t size() { return Components.size(); }
};

// The mapper function pointer type. It follows the signature below:
// void .omp_mapper.<type_name>.<mapper_id>.(void *rt_mapper_handle,
//                                           void *base, void *begin,
//                                           size_t size, int64_t type);
typedef void (*MapperFuncPtrTy)(void *, void *, void *, int64_t, int64_t);

// Function pointer type for target_data_* functions (targetDataBegin,
// targetDataEnd and targetDataUpdate).
typedef int (*TargetDataFuncPtrTy)(DeviceTy &, int32_t, void **, void **,
                                   int64_t *, int64_t *, map_var_info_t *,
                                   void **, __tgt_async_info *);

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

#define TARGET_NAME Libomptarget
#define DEBUG_PREFIX GETNAME(TARGET_NAME)

////////////////////////////////////////////////////////////////////////////////
/// dump a table of all the host-target pointer pairs on failure
static inline void dumpTargetPointerMappings(const ident_t *Loc,
                                             const DeviceTy &Device) {
  if (Device.HostDataToTargetMap.empty())
    return;

  SourceInfo Kernel(Loc);
  INFO(Device.DeviceID,
       "OpenMP Host-Device pointer mappings after block at %s:%d:%d:\n",
       Kernel.getFilename(), Kernel.getLine(), Kernel.getColumn());
  INFO(Device.DeviceID, "%-18s %-18s %s %s %s\n", "Host Ptr", "Target Ptr",
       "Size (B)", "RefCount", "Declaration");
  for (const auto &HostTargetMap : Device.HostDataToTargetMap) {
    SourceInfo Info(HostTargetMap.HstPtrName);
    INFO(Device.DeviceID, DPxMOD " " DPxMOD " %-8lu %-8ld %s at %s:%d:%d\n",
         DPxPTR(HostTargetMap.HstPtrBegin), DPxPTR(HostTargetMap.TgtPtrBegin),
         (long unsigned)(HostTargetMap.HstPtrEnd - HostTargetMap.HstPtrBegin),
         HostTargetMap.getRefCount(), Info.getName(), Info.getFilename(),
         Info.getLine(), Info.getColumn());
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Print out the names and properties of the arguments to each kernel
static inline void
printKernelArguments(const ident_t *Loc, const int64_t DeviceId,
                     const int32_t ArgNum, const int64_t *ArgSizes,
                     const int64_t *ArgTypes, const map_var_info_t *ArgNames,
                     const char *RegionType) {
  SourceInfo info(Loc);
  INFO(DeviceId, "%s at %s:%d:%d with %d arguments:\n", RegionType,
       info.getFilename(), info.getLine(), info.getColumn(), ArgNum);

  for (int32_t i = 0; i < ArgNum; ++i) {
    const map_var_info_t varName = (ArgNames) ? ArgNames[i] : nullptr;
    const char *type = nullptr;
    const char *implicit =
        (ArgTypes[i] & OMP_TGT_MAPTYPE_IMPLICIT) ? "(implicit)" : "";
    if (ArgTypes[i] & OMP_TGT_MAPTYPE_TO && ArgTypes[i] & OMP_TGT_MAPTYPE_FROM)
      type = "tofrom";
    else if (ArgTypes[i] & OMP_TGT_MAPTYPE_TO)
      type = "to";
    else if (ArgTypes[i] & OMP_TGT_MAPTYPE_FROM)
      type = "from";
    else if (ArgTypes[i] & OMP_TGT_MAPTYPE_PRIVATE)
      type = "private";
    else if (ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL)
      type = "firstprivate";
    else if (ArgTypes[i] & OMP_TGT_MAPTYPE_TARGET_PARAM && ArgSizes[i] != 0)
      type = "alloc";
    else
      type = "use_address";

    INFO(DeviceId, "%s(%s)[%ld] %s\n", type,
         getNameFromMapping(varName).c_str(), ArgSizes[i], implicit);
  }
}

#ifdef OMPTARGET_PROFILE_ENABLED
#include "llvm/Support/TimeProfiler.h"
#define TIMESCOPE() llvm::TimeTraceScope TimeScope(__FUNCTION__)
#else
#define TIMESCOPE()
#endif

#endif
