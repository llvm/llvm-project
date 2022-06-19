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

#include "device.h"
#include <Debug.h>
#include <SourceInfo.h>
#include <omptarget.h>

#include <cstdint>

extern int targetDataBegin(ident_t *loc, DeviceTy &Device, int32_t arg_num,
                           void **args_base, void **args, int64_t *arg_sizes,
                           int64_t *arg_types, map_var_info_t *arg_names,
                           void **arg_mappers, AsyncInfoTy &AsyncInfo,
                           bool FromMapper = false);

extern int targetDataEnd(ident_t *loc, DeviceTy &Device, int32_t ArgNum,
                         void **ArgBases, void **Args, int64_t *ArgSizes,
                         int64_t *ArgTypes, map_var_info_t *arg_names,
                         void **ArgMappers, AsyncInfoTy &AsyncInfo,
                         bool FromMapper = false);

extern int targetDataUpdate(ident_t *loc, DeviceTy &Device, int32_t arg_num,
                            void **args_base, void **args, int64_t *arg_sizes,
                            int64_t *arg_types, map_var_info_t *arg_names,
                            void **arg_mappers, AsyncInfoTy &AsyncInfo,
                            bool FromMapper = false);

extern int target(ident_t *loc, DeviceTy &Device, void *HostPtr, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *arg_names,
                  void **ArgMappers, int32_t TeamNum, int32_t ThreadLimit,
                  int IsTeamConstruct, AsyncInfoTy &AsyncInfo);

extern void handleTargetOutcome(bool Success, ident_t *Loc);
extern bool checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc);
extern void *targetAllocExplicit(size_t size, int device_num, int kind,
                                 const char *name);

// This structure stores information of a mapped memory region.
struct MapComponentInfoTy {
  void *Base;
  void *Begin;
  int64_t Size;
  int64_t Type;
  void *Name;
  MapComponentInfoTy() = default;
  MapComponentInfoTy(void *Base, void *Begin, int64_t Size, int64_t Type,
                     void *Name)
      : Base(Base), Begin(Begin), Size(Size), Type(Type), Name(Name) {}
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
//                                           size_t size, int64_t type,
//                                           void * name);
typedef void (*MapperFuncPtrTy)(void *, void *, void *, int64_t, int64_t,
                                void *);

// Function pointer type for targetData* functions (targetDataBegin,
// targetDataEnd and targetDataUpdate).
typedef int (*TargetDataFuncPtrTy)(ident_t *, DeviceTy &, int32_t, void **,
                                   void **, int64_t *, int64_t *,
                                   map_var_info_t *, void **, AsyncInfoTy &,
                                   bool);

// Implemented in libomp, they are called from within __tgt_* functions.
#ifdef __cplusplus
extern "C" {
#endif
/*!
 * The ident structure that describes a source location.
 * The struct is identical to the one in the kmp.h file.
 * We maintain the same data structure for compatibility.
 */
typedef int kmp_int32;
typedef intptr_t kmp_intptr_t;
// Compiler sends us this info:
typedef struct kmp_depend_info {
  kmp_intptr_t base_addr;
  size_t len;
  struct {
    bool in : 1;
    bool out : 1;
    bool mtx : 1;
  } flags;
} kmp_depend_info_t;
// functions that extract info from libomp; keep in sync
int omp_get_default_device(void) __attribute__((weak));
int32_t __kmpc_global_thread_num(void *) __attribute__((weak));
int __kmpc_get_target_offload(void) __attribute__((weak));
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list)
    __attribute__((weak));
#ifdef __cplusplus
}
#endif

#define TARGET_NAME Libomptarget
#define DEBUG_PREFIX GETNAME(TARGET_NAME)

////////////////////////////////////////////////////////////////////////////////
/// dump a table of all the host-target pointer pairs on failure
static inline void dumpTargetPointerMappings(const ident_t *Loc,
                                             DeviceTy &Device) {
  DeviceTy::HDTTMapAccessorTy HDTTMap =
      Device.HostDataToTargetMap.getExclusiveAccessor();
  if (HDTTMap->empty())
    return;

  SourceInfo Kernel(Loc);
  INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
       "OpenMP Host-Device pointer mappings after block at %s:%d:%d:\n",
       Kernel.getFilename(), Kernel.getLine(), Kernel.getColumn());
  INFO(OMP_INFOTYPE_ALL, Device.DeviceID, "%-18s %-18s %s %s %s %s\n",
       "Host Ptr", "Target Ptr", "Size (B)", "DynRefCount", "HoldRefCount",
       "Declaration");
  for (const auto &It : *HDTTMap) {
    HostDataToTargetTy &HDTT = *It.HDTT;
    SourceInfo Info(HDTT.HstPtrName);
    INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
         DPxMOD " " DPxMOD " %-8" PRIuPTR " %-11s %-12s %s at %s:%d:%d\n",
         DPxPTR(HDTT.HstPtrBegin), DPxPTR(HDTT.TgtPtrBegin),
         HDTT.HstPtrEnd - HDTT.HstPtrBegin, HDTT.dynRefCountToStr().c_str(),
         HDTT.holdRefCountToStr().c_str(), Info.getName(), Info.getFilename(),
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
  INFO(OMP_INFOTYPE_ALL, DeviceId, "%s at %s:%d:%d with %d arguments:\n",
       RegionType, info.getFilename(), info.getLine(), info.getColumn(),
       ArgNum);

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
    else if (ArgSizes[i] != 0)
      type = "alloc";
    else
      type = "use_address";

    INFO(OMP_INFOTYPE_ALL, DeviceId, "%s(%s)[%" PRId64 "] %s\n", type,
         getNameFromMapping(varName).c_str(), ArgSizes[i], implicit);
  }
}

#ifdef OMPTARGET_PROFILE_ENABLED
#include "llvm/Support/TimeProfiler.h"
#define TIMESCOPE() llvm::TimeTraceScope TimeScope(__FUNCTION__)
#define TIMESCOPE_WITH_IDENT(IDENT)                                            \
  SourceInfo SI(IDENT);                                                        \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, SI.getProfileLocation())
#define TIMESCOPE_WITH_NAME_AND_IDENT(NAME, IDENT)                             \
  SourceInfo SI(IDENT);                                                        \
  llvm::TimeTraceScope TimeScope(NAME, SI.getProfileLocation())
#else
#define TIMESCOPE()
#define TIMESCOPE_WITH_IDENT(IDENT)
#define TIMESCOPE_WITH_NAME_AND_IDENT(NAME, IDENT)
#endif

#endif
