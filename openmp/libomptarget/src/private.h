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

extern int target_replay(ident_t *Loc, DeviceTy &Device, void *HostPtr,
                         void *DeviceMemory, int64_t DeviceMemorySize,
                         void **TgtArgs, ptrdiff_t *TgtOffsets, int32_t NumArgs,
                         int32_t NumTeams, int32_t ThreadLimit,
                         uint64_t LoopTripCount, AsyncInfoTy &AsyncInfo);

extern void handleTargetOutcome(bool Success, ident_t *Loc);
extern bool checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc);
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
  llvm::SmallVector<MapComponentInfoTy> Components;
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
typedef int64_t kmp_int64;
typedef intptr_t kmp_intptr_t;

typedef void *omp_depend_t;
struct kmp_task;
typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32, struct kmp_task *);
typedef struct kmp_task {
  void *shareds;
  kmp_routine_entry_t routine;
  kmp_int32 part_id;
} kmp_task_t;

typedef struct kmp_tasking_flags { /* Total struct must be exactly 32 bits */
  /* Compiler flags */             /* Total compiler flags must be 16 bits */
  unsigned tiedness : 1;           /* task is either tied (1) or untied (0) */
  unsigned final : 1;              /* task is final(1) so execute immediately */
  unsigned merged_if0 : 1; /* no __kmpc_task_{begin/complete}_if0 calls in if0
                              code path */
  unsigned destructors_thunk : 1; /* set if the compiler creates a thunk to
                                     invoke destructors from the runtime */
  unsigned proxy : 1; /* task is a proxy task (it will be executed outside the
                         context of the RTL) */
  unsigned priority_specified : 1; /* set if the compiler provides priority
                                      setting for the task */
  unsigned detachable : 1;         /* 1 == can detach */
  unsigned hidden_helper : 1;      /* 1 == hidden helper task */
  unsigned reserved : 8;           /* reserved for compiler use */

  /* Library flags */       /* Total library flags must be 16 bits */
  unsigned tasktype : 1;    /* task is either explicit(1) or implicit (0) */
  unsigned task_serial : 1; // task is executed immediately (1) or deferred (0)
  unsigned tasking_ser : 1; // all tasks in team are either executed immediately
  // (1) or may be deferred (0)
  unsigned team_serial : 1; // entire team is serial (1) [1 thread] or parallel
  // (0) [>= 2 threads]
  /* If either team_serial or tasking_ser is set, task team may be NULL */
  /* Task State Flags: */
  unsigned started : 1;    /* 1==started, 0==not started     */
  unsigned executing : 1;  /* 1==executing, 0==not executing */
  unsigned complete : 1;   /* 1==complete, 0==not complete   */
  unsigned freed : 1;      /* 1==freed, 0==allocated        */
  unsigned native : 1;     /* 1==gcc-compiled task, 0==intel */
  unsigned reserved31 : 7; /* reserved for library use */
} kmp_tasking_flags_t;

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
void **__kmpc_omp_get_target_async_handle_ptr(kmp_int32 gtid)
    __attribute__((weak));
bool __kmpc_omp_has_task_team(kmp_int32 gtid) __attribute__((weak));
kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
                                  kmp_int32 flags, size_t sizeof_kmp_task_t,
                                  size_t sizeof_shareds,
                                  kmp_routine_entry_t task_entry)
    __attribute__((weak));

kmp_task_t *
__kmpc_omp_target_task_alloc(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 flags,
                             size_t sizeof_kmp_task_t, size_t sizeof_shareds,
                             kmp_routine_entry_t task_entry,
                             kmp_int64 device_id) __attribute__((weak));

kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
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
// Invalid GTID as defined by libomp; keep in sync
#define KMP_GTID_DNE (-2)
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

// Wrapper for task stored async info objects.
class TaskAsyncInfoWrapperTy {
  const int ExecThreadID = KMP_GTID_DNE;
  AsyncInfoTy LocalAsyncInfo;
  AsyncInfoTy *AsyncInfo = &LocalAsyncInfo;
  void **TaskAsyncInfoPtr = nullptr;

public:
  TaskAsyncInfoWrapperTy(DeviceTy &Device)
      : ExecThreadID(__kmpc_global_thread_num(NULL)), LocalAsyncInfo(Device) {
    // If we failed to acquired the current global thread id, we cannot
    // re-enqueue the current task. Thus we should use the local blocking async
    // info.
    if (ExecThreadID == KMP_GTID_DNE)
      return;

    // Only tasks with an assigned task team can be re-enqueue and thus can
    // use the non-blocking synchronization scheme. Thus we should use the local
    // blocking async info, if we don´t have one.
    if (!__kmpc_omp_has_task_team(ExecThreadID))
      return;

    // Acquire a pointer to the AsyncInfo stored inside the current task being
    // executed.
    TaskAsyncInfoPtr = __kmpc_omp_get_target_async_handle_ptr(ExecThreadID);

    // If we cannot acquire such pointer, fallback to using the local blocking
    // async info.
    if (!TaskAsyncInfoPtr)
      return;

    // When creating a new task async info, the task handle must always be
    // invalid. We must never overwrite any task async handle and there should
    // never be any valid handle store inside the task at this point.
    assert((*TaskAsyncInfoPtr) == nullptr &&
           "Task async handle is not empty when dispatching new device "
           "operations. The handle was not cleared properly or "
           "__tgt_target_nowait_query should have been called!");

    // If no valid async handle is present, a new AsyncInfo will be allocated
    // and stored in the current task.
    AsyncInfo = new AsyncInfoTy(Device, AsyncInfoTy::SyncTy::NON_BLOCKING);
    *TaskAsyncInfoPtr = (void *)AsyncInfo;
  }

  ~TaskAsyncInfoWrapperTy() {
    // Local async info destruction is automatically handled by ~AsyncInfoTy.
    if (AsyncInfo == &LocalAsyncInfo)
      return;

    // If the are device operations still pending, return immediately without
    // deallocating the handle.
    if (!AsyncInfo->isDone())
      return;

    // Delete the handle and unset it from the OpenMP task data.
    delete AsyncInfo;
    *TaskAsyncInfoPtr = nullptr;
  }

  operator AsyncInfoTy &() { return *AsyncInfo; }
};

// Implement exponential backoff counting.
// Linearly increments until given maximum, exponentially decrements based on
// given backoff factor.
class ExponentialBackoff {
  int64_t Count = 0;
  const int64_t MaxCount = 0;
  const int64_t CountThreshold = 0;
  const float BackoffFactor = 0.0f;

public:
  ExponentialBackoff(int64_t MaxCount, int64_t CountThreshold,
                     float BackoffFactor)
      : MaxCount(MaxCount), CountThreshold(CountThreshold),
        BackoffFactor(BackoffFactor) {
    assert(MaxCount >= 0 &&
           "ExponentialBackoff: maximum count value should be non-negative");
    assert(CountThreshold >= 0 &&
           "ExponentialBackoff: count threshold value should be non-negative");
    assert(BackoffFactor >= 0 && BackoffFactor < 1 &&
           "ExponentialBackoff: backoff factor should be in [0, 1) interval");
  }

  void increment() { Count = std::min(Count + 1, MaxCount); }

  void decrement() { Count *= BackoffFactor; }

  bool isAboveThreshold() const { return Count > CountThreshold; }
};

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
