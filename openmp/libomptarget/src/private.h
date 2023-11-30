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

#include "Shared/Debug.h"
#include "Shared/SourceInfo.h"

#include "OpenMP/InternalTypes.h"

#include "device.h"
#include "omptarget.h"

#include <cstdint>

#define DI_DEP_TYPE_IN 11
#define DI_DEP_TYPE_OUT 12
#define DI_DEP_TYPE_INOUT 13

extern int targetDataBegin(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                           void **ArgsBase, void **Args, int64_t *ArgSizes,
                           int64_t *ArgTypes, map_var_info_t *ArgNames,
                           void **ArgMappers, AsyncInfoTy &AsyncInfo,
                           bool FromMapper = false);

extern int targetDataEnd(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                         void **ArgBases, void **Args, int64_t *ArgSizes,
                         int64_t *ArgTypes, map_var_info_t *ArgNames,
                         void **ArgMappers, AsyncInfoTy &AsyncInfo,
                         bool FromMapper = false);

extern int targetDataUpdate(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                            void **ArgsBase, void **Args, int64_t *ArgSizes,
                            int64_t *ArgTypes, map_var_info_t *ArgNames,
                            void **ArgMappers, AsyncInfoTy &AsyncInfo,
                            bool FromMapper = false);

extern int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
                  KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

extern int target_activate_rr(DeviceTy &Device, uint64_t MemorySize,
                              void *ReqAddr, bool isRecord, bool SaveOutput,
                              uint64_t &ReqPtrArgOffset);

extern int target_replay(ident_t *Loc, DeviceTy &Device, void *HostPtr,
                         void *DeviceMemory, int64_t DeviceMemorySize,
                         void **TgtArgs, ptrdiff_t *TgtOffsets, int32_t NumArgs,
                         int32_t NumTeams, int32_t ThreadLimit,
                         uint64_t LoopTripCount, AsyncInfoTy &AsyncInfo);

extern void handleTargetOutcome(bool Success, ident_t *Loc);
extern bool checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc);
extern void *targetLockExplicit(void *ptr, size_t size, int device_num,
                                const char *name);
extern void targetUnlockExplicit(void *ptr, int device_num, const char *name);
extern void *targetAllocExplicit(size_t Size, int DeviceNum, int Kind,
                                 const char *Name);
extern void targetFreeExplicit(void *DevicePtr, int DeviceNum, int Kind,
                               const char *Name);
extern void *targetLockExplicit(void *HostPtr, size_t Size, int DeviceNum,
                                const char *Name);
extern void targetUnlockExplicit(void *HostPtr, int DeviceNum,
                                 const char *Name);

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

int __kmpc_get_target_offload(void) __attribute__((weak));
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, int32_t gtid, int32_t flags,
                                  size_t sizeof_kmp_task_t,
                                  size_t sizeof_shareds,
                                  kmp_routine_entry_t task_entry)
    __attribute__((weak));

kmp_task_t *
__kmpc_omp_target_task_alloc(ident_t *loc_ref, int32_t gtid, int32_t flags,
                             size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                             kmp_routine_entry_t task_entry, int64_t device_id)
    __attribute__((weak));

int32_t __kmpc_omp_task_with_deps(ident_t *loc_ref, int32_t gtid,
                                  kmp_task_t *new_task, int32_t ndeps,
                                  kmp_depend_info_t *dep_list,
                                  int32_t ndeps_noalias,
                                  kmp_depend_info_t *noalias_dep_list)
    __attribute__((weak));

/**
 * The argument set that is passed from asynchronous memory copy to block
 * version of memory copy invoked in helper task
 */
struct TargetMemcpyArgsTy {
  /**
   * Common attribuutes
   */
  void *Dst;
  const void *Src;
  int DstDevice;
  int SrcDevice;

  /**
   * The flag that denotes single dimensional or rectangle dimensional copy
   */
  bool IsRectMemcpy;

  /**
   * Arguments for single dimensional copy
   */
  size_t Length;
  size_t DstOffset;
  size_t SrcOffset;

  /**
   * Arguments for rectangle dimensional copy
   */
  size_t ElementSize;
  int NumDims;
  const size_t *Volume;
  const size_t *DstOffsets;
  const size_t *SrcOffsets;
  const size_t *DstDimensions;
  const size_t *SrcDimensions;

  /**
   * Constructor for single dimensional copy
   */
  TargetMemcpyArgsTy(void *Dst, const void *Src, size_t Length,
                     size_t DstOffset, size_t SrcOffset, int DstDevice,
                     int SrcDevice)
      : Dst(Dst), Src(Src), DstDevice(DstDevice), SrcDevice(SrcDevice),
        IsRectMemcpy(false), Length(Length), DstOffset(DstOffset),
        SrcOffset(SrcOffset), ElementSize(0), NumDims(0), Volume(0),
        DstOffsets(0), SrcOffsets(0), DstDimensions(0), SrcDimensions(0){};

  /**
   * Constructor for rectangle dimensional copy
   */
  TargetMemcpyArgsTy(void *Dst, const void *Src, size_t ElementSize,
                     int NumDims, const size_t *Volume,
                     const size_t *DstOffsets, const size_t *SrcOffsets,
                     const size_t *DstDimensions, const size_t *SrcDimensions,
                     int DstDevice, int SrcDevice)
      : Dst(Dst), Src(Src), DstDevice(DstDevice), SrcDevice(SrcDevice),
        IsRectMemcpy(true), Length(0), DstOffset(0), SrcOffset(0),
        ElementSize(ElementSize), NumDims(NumDims), Volume(Volume),
        DstOffsets(DstOffsets), SrcOffsets(SrcOffsets),
        DstDimensions(DstDimensions), SrcDimensions(SrcDimensions){};
};

struct TargetMemsetArgsTy {
  // Common attributes of a memset operation
  void *Ptr;
  int C;
  size_t N;
  int DeviceNum;

  // no constructors defined, because this is a PoD
};

#ifdef __cplusplus
}
#endif

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
  SourceInfo Info(Loc);
  INFO(OMP_INFOTYPE_ALL, DeviceId, "%s at %s:%d:%d with %d arguments:\n",
       RegionType, Info.getFilename(), Info.getLine(), Info.getColumn(),
       ArgNum);

  for (int32_t I = 0; I < ArgNum; ++I) {
    const map_var_info_t VarName = (ArgNames) ? ArgNames[I] : nullptr;
    const char *Type = nullptr;
    const char *Implicit =
        (ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT) ? "(implicit)" : "";
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_TO && ArgTypes[I] & OMP_TGT_MAPTYPE_FROM)
      Type = "tofrom";
    else if (ArgTypes[I] & OMP_TGT_MAPTYPE_TO)
      Type = "to";
    else if (ArgTypes[I] & OMP_TGT_MAPTYPE_FROM)
      Type = "from";
    else if (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE)
      Type = "private";
    else if (ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL)
      Type = "firstprivate";
    else if (ArgSizes[I] != 0)
      Type = "alloc";
    else
      Type = "use_address";

    INFO(OMP_INFOTYPE_ALL, DeviceId, "%s(%s)[%" PRId64 "] %s\n", Type,
         getNameFromMapping(VarName).c_str(), ArgSizes[I], Implicit);
  }
}

#endif
