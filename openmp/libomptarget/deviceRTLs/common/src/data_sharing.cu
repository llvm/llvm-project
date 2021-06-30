//===----- data_sharing.cu - OpenMP GPU data sharing ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of data sharing environments
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/omptarget.h"
#include "target/shuffle.h"
#include "target_impl.h"

// Return true if this is the master thread.
INLINE static bool IsMasterThread(bool isSPMDExecutionMode) {
  return !isSPMDExecutionMode && GetMasterThreadID() == GetThreadIdInBlock();
}

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

static constexpr unsigned MinBytes = 8;

template <unsigned BytesPerThread, unsigned NThreads = MAX_THREADS_PER_TEAM>
struct alignas(32) ThreadStackTy {
  static constexpr unsigned MaxSize = NThreads * BytesPerThread;
  static constexpr unsigned NumThreads = NThreads;
  static constexpr unsigned NumWarps = (NThreads + WARPSIZE - 1) / WARPSIZE;
  static constexpr unsigned MaxSizePerWarp = MaxSize / NumWarps;

  unsigned char Data[MaxSize];
  char Sizes[MaxSize / MinBytes];
  char SizeUsage[NumWarps];
  char Usage[NumWarps];
};

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes * 8, 1> MainSharedStack;
#pragma omp allocate(MainSharedStack) allocator(omp_pteam_mem_alloc)

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes * 2,
                                              MAX_THREADS_PER_TEAM / 8>
    WorkerSharedStack;
#pragma omp allocate(WorkerSharedStack) allocator(omp_pteam_mem_alloc)

template <typename AllocTy>
static void *__kmpc_alloc_for_warp(AllocTy Alloc, unsigned Bytes,
                                   unsigned WarpBytes) {
  void *Ptr;
  __kmpc_impl_lanemask_t CurActive = __kmpc_impl_activemask();
  unsigned LeaderID = __kmpc_impl_ffs(CurActive) - 1;
  bool IsWarpLeader = (GetThreadIdInBlock() % WARPSIZE) == LeaderID;
  if (IsWarpLeader)
    Ptr = Alloc();
  // Get address from the first active lane.
  int *FP = (int *)&Ptr;
  FP[0] = __kmpc_impl_shfl_sync(CurActive, FP[0], LeaderID);
  if (sizeof(Ptr) == 8)
    FP[1] = __kmpc_impl_shfl_sync(CurActive, FP[1], LeaderID);
  return (void *)&((char *)(Ptr))[(GetLaneId() - LeaderID) * Bytes];
}

EXTERN void *__kmpc_alloc_shared(size_t Bytes) {
  Bytes = Bytes + (Bytes % MinBytes);
  if (IsMasterThread(__kmpc_is_spmd_exec_mode())) {
    // Main thread alone, use shared memory if space is available.
    if (MainSharedStack.Usage[0] + Bytes <= MainSharedStack.MaxSize) {
      void *Ptr = &MainSharedStack.Data[MainSharedStack.Usage[0]];
      MainSharedStack.Usage[0] += Bytes;
      MainSharedStack.Sizes[MainSharedStack.SizeUsage[0]++] = Bytes;
      return Ptr;
    }
  } else {
    int TID = GetThreadIdInBlock();
    int WID = GetWarpId();
    unsigned WarpBytes = Bytes * WARPSIZE;
    auto AllocSharedStack = [&]() {
      unsigned WarpOffset = WID * WorkerSharedStack.MaxSizePerWarp;
      void *Ptr =
          &WorkerSharedStack.Data[WarpOffset + WorkerSharedStack.Usage[WID]];
      WorkerSharedStack.Usage[WID] += WarpBytes;
      WorkerSharedStack.Sizes[WorkerSharedStack.SizeUsage[WID]++] = WarpBytes;
      return Ptr;
    };
    if (TID < WorkerSharedStack.NumThreads &&
        WorkerSharedStack.Usage[WID] + WarpBytes <=
            WorkerSharedStack.MaxSizePerWarp)
      return __kmpc_alloc_for_warp(AllocSharedStack, Bytes, WarpBytes);
  }
  // Fallback to malloc
  int TID = GetThreadIdInBlock();
  unsigned WarpBytes = Bytes * WARPSIZE;
  auto AllocGlobal = [&] {
    return SafeMalloc(WarpBytes, "AllocGlobalFallback");
  };
  return __kmpc_alloc_for_warp(AllocGlobal, Bytes, WarpBytes);
}

EXTERN void __kmpc_free_shared(void *Ptr) {
  __kmpc_impl_lanemask_t CurActive = __kmpc_impl_activemask();
  unsigned LeaderID = __kmpc_impl_ffs(CurActive) - 1;
  bool IsWarpLeader = (GetThreadIdInBlock() % WARPSIZE) == LeaderID;
  __kmpc_syncwarp(CurActive);
  if (IsWarpLeader) {
    if (Ptr >= &MainSharedStack.Data[0] &&
        Ptr < &MainSharedStack.Data[MainSharedStack.MaxSize]) {
      unsigned Bytes = MainSharedStack.Sizes[--MainSharedStack.SizeUsage[0]];
      MainSharedStack.Usage[0] -= Bytes;
      return;
    }
    if (Ptr >= &WorkerSharedStack.Data[0] &&
        Ptr < &WorkerSharedStack.Data[WorkerSharedStack.MaxSize]) {
      int WID = GetWarpId();
      unsigned Bytes =
          WorkerSharedStack.Sizes[--WorkerSharedStack.SizeUsage[WID]];
      WorkerSharedStack.Usage[WID] -= Bytes;
      return;
    }
    SafeFree(Ptr, "FreeGlobalFallback");
  }
}

EXTERN void __kmpc_data_sharing_init_stack() {
  for (unsigned i = 0; i < MainSharedStack.NumWarps; ++i) {
    MainSharedStack.SizeUsage[i] = 0;
    MainSharedStack.Usage[i] = 0;
  }
  for (unsigned i = 0; i < WorkerSharedStack.NumWarps; ++i) {
    WorkerSharedStack.SizeUsage[i] = 0;
    WorkerSharedStack.Usage[i] = 0;
  }
}

// Begin a data sharing context. Maintain a list of references to shared
// variables. This list of references to shared variables will be passed
// to one or more threads.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_begin_sharing_variables(void ***GlobalArgs, size_t nArgs) {
  omptarget_nvptx_globalArgs.EnsureSize(nArgs);
  *GlobalArgs = omptarget_nvptx_globalArgs.GetArgs();
}

// End a data sharing context. There is no need to have a list of refs
// to shared variables because the context in which those variables were
// shared has now ended. This should clean-up the list of references only
// without affecting the actual global storage of the variables.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_end_sharing_variables() {
  omptarget_nvptx_globalArgs.DeInit();
}

// This function will return a list of references to global variables. This
// is how the workers will get a reference to the globalized variable. The
// members of this list will be passed to the outlined parallel function
// preserving the order.
// Called by all workers.
EXTERN void __kmpc_get_shared_variables(void ***GlobalArgs) {
  *GlobalArgs = omptarget_nvptx_globalArgs.GetArgs();
}

// This function is used to init static memory manager. This manager is used to
// manage statically allocated global memory. This memory is allocated by the
// compiler and used to correctly implement globalization of the variables in
// target, teams and distribute regions.
EXTERN void __kmpc_get_team_static_memory(int16_t isSPMDExecutionMode,
                                          const void *buf, size_t size,
                                          int16_t is_shared,
                                          const void **frame) {
  if (is_shared) {
    *frame = buf;
    return;
  }
  if (isSPMDExecutionMode) {
    if (GetThreadIdInBlock() == 0) {
      *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
    }
    __kmpc_impl_syncthreads();
    return;
  }
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
  __kmpc_impl_threadfence();
}

EXTERN void __kmpc_restore_team_static_memory(int16_t isSPMDExecutionMode,
                                              int16_t is_shared) {
  if (is_shared)
    return;
  if (isSPMDExecutionMode) {
    __kmpc_impl_syncthreads();
    if (GetThreadIdInBlock() == 0) {
      omptarget_nvptx_simpleMemoryManager.Release();
    }
    return;
  }
  __kmpc_impl_threadfence();
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  omptarget_nvptx_simpleMemoryManager.Release();
}

#pragma omp end declare target
