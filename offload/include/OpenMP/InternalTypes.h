//===-- OpenMP/InternalTypes.h -- Internal OpenMP Types ------------- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private type declarations and helper macros for OpenMP.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_INTERNAL_TYPES_H
#define OMPTARGET_OPENMP_INTERNAL_TYPES_H

#include <cstddef>
#include <cstdint>

extern "C" {

// Compiler sends us this info:
typedef struct kmp_depend_info {
  intptr_t base_addr;
  size_t len;
  struct {
    bool in : 1;
    bool out : 1;
    bool mtx : 1;
  } flags;
} kmp_depend_info_t;

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

struct kmp_task;
typedef int32_t (*kmp_routine_entry_t)(int32_t, struct kmp_task *);
typedef struct kmp_task {
  void *shareds;
  kmp_routine_entry_t routine;
  int32_t part_id;
} kmp_task_t;

int32_t __kmpc_global_thread_num(void *) __attribute__((weak));
bool __kmpc_omp_has_task_team(int32_t gtid) __attribute__((weak));
void **__kmpc_omp_get_target_async_handle_ptr(int32_t gtid)
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

} // extern "C"

#endif // OMPTARGET_OPENMP_INTERNAL_TYPES_H
