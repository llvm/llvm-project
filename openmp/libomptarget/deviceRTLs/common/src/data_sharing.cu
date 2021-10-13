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

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

static constexpr unsigned MinBytes = 8;

static constexpr unsigned Alignment = 8;

/// External symbol to access dynamic shared memory.
extern unsigned char DynamicSharedBuffer[] __attribute__((aligned(Alignment)));
#pragma omp allocate(DynamicSharedBuffer) allocator(omp_pteam_mem_alloc)

EXTERN void *__kmpc_get_dynamic_shared() { return DynamicSharedBuffer; }

EXTERN void *llvm_omp_get_dynamic_shared() {
  return __kmpc_get_dynamic_shared();
}

template <unsigned BPerThread, unsigned NThreads = MAX_THREADS_PER_TEAM>
struct alignas(32) ThreadStackTy {
  static constexpr unsigned BytesPerThread = BPerThread;
  static constexpr unsigned NumThreads = NThreads;
  static constexpr unsigned NumWarps = (NThreads + WARPSIZE - 1) / WARPSIZE;

  unsigned char Data[NumThreads][BytesPerThread];
  unsigned char Usage[NumThreads];
};

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes * 8, 1> MainSharedStack;
#pragma omp allocate(MainSharedStack) allocator(omp_pteam_mem_alloc)

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes,
                                              MAX_THREADS_PER_TEAM / 4>
    WorkerSharedStack;
#pragma omp allocate(WorkerSharedStack) allocator(omp_pteam_mem_alloc)

EXTERN void *__kmpc_alloc_shared(size_t Bytes) {
  size_t AlignedBytes = Bytes + (Bytes % MinBytes);
  int TID = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(TID)) {
    // Main thread alone, use shared memory if space is available.
    if (MainSharedStack.Usage[0] + AlignedBytes <= MainSharedStack.BytesPerThread) {
      void *Ptr = &MainSharedStack.Data[0][MainSharedStack.Usage[0]];
      MainSharedStack.Usage[0] += AlignedBytes;
      return Ptr;
    }
  } else if (TID < WorkerSharedStack.NumThreads) {
    if (WorkerSharedStack.Usage[TID] + AlignedBytes <= WorkerSharedStack.BytesPerThread) {
      void *Ptr = &WorkerSharedStack.Data[TID][WorkerSharedStack.Usage[TID]];
      WorkerSharedStack.Usage[TID] += AlignedBytes;
      return Ptr;
    }
  }
  // Fallback to malloc
  return SafeMalloc(Bytes, "AllocGlobalFallback");
}

EXTERN void __kmpc_free_shared(void *Ptr, size_t Bytes) {
  size_t AlignedBytes = Bytes + (Bytes % MinBytes);
  int TID = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(TID)) {
    if (Ptr >= &MainSharedStack.Data[0][0] &&
        Ptr < &MainSharedStack.Data[MainSharedStack.NumThreads][0]) {
      MainSharedStack.Usage[0] -= AlignedBytes;
      return;
    }
  } else if (TID < WorkerSharedStack.NumThreads) {
    if (Ptr >= &WorkerSharedStack.Data[0][0] &&
        Ptr < &WorkerSharedStack.Data[WorkerSharedStack.NumThreads][0]) {
      int TID = __kmpc_get_hardware_thread_id_in_block();
      WorkerSharedStack.Usage[TID] -= AlignedBytes;
      return;
    }
  }
  SafeFree(Ptr, "FreeGlobalFallback");
}

EXTERN void __kmpc_data_sharing_init_stack() {
  for (unsigned i = 0; i < MainSharedStack.NumWarps; ++i)
    MainSharedStack.Usage[i] = 0;
  for (unsigned i = 0; i < WorkerSharedStack.NumThreads; ++i)
    WorkerSharedStack.Usage[i] = 0;
}

/// Allocate storage in shared memory to communicate arguments from the main
/// thread to the workers in generic mode. If we exceed
/// NUM_SHARED_VARIABLES_IN_SHARED_MEM we will malloc space for communication.
#define NUM_SHARED_VARIABLES_IN_SHARED_MEM 64

[[clang::loader_uninitialized]] static void
    *SharedMemVariableSharingSpace[NUM_SHARED_VARIABLES_IN_SHARED_MEM];
#pragma omp allocate(SharedMemVariableSharingSpace)                            \
    allocator(omp_pteam_mem_alloc)
[[clang::loader_uninitialized]] static void **SharedMemVariableSharingSpacePtr;
#pragma omp allocate(SharedMemVariableSharingSpacePtr)                         \
    allocator(omp_pteam_mem_alloc)

// Begin a data sharing context. Maintain a list of references to shared
// variables. This list of references to shared variables will be passed
// to one or more threads.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_begin_sharing_variables(void ***GlobalArgs, size_t nArgs) {
  if (nArgs <= NUM_SHARED_VARIABLES_IN_SHARED_MEM) {
    SharedMemVariableSharingSpacePtr = &SharedMemVariableSharingSpace[0];
  } else {
    SharedMemVariableSharingSpacePtr =
        (void **)SafeMalloc(nArgs * sizeof(void *), "new extended args");
  }
  *GlobalArgs = SharedMemVariableSharingSpacePtr;
}

// End a data sharing context. There is no need to have a list of refs
// to shared variables because the context in which those variables were
// shared has now ended. This should clean-up the list of references only
// without affecting the actual global storage of the variables.
// In L0 data sharing this is called by master thread.
// In L1 data sharing this is called by active warp master thread.
EXTERN void __kmpc_end_sharing_variables() {
  if (SharedMemVariableSharingSpacePtr != &SharedMemVariableSharingSpace[0])
    SafeFree(SharedMemVariableSharingSpacePtr, "new extended args");
}

// This function will return a list of references to global variables. This
// is how the workers will get a reference to the globalized variable. The
// members of this list will be passed to the outlined parallel function
// preserving the order.
// Called by all workers.
EXTERN void __kmpc_get_shared_variables(void ***GlobalArgs) {
  *GlobalArgs = SharedMemVariableSharingSpacePtr;
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
    if (__kmpc_get_hardware_thread_id_in_block() == 0) {
      *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
    }
    __kmpc_impl_syncthreads();
    return;
  }
  ASSERT0(LT_FUSSY,
          __kmpc_get_hardware_thread_id_in_block() == GetMasterThreadID(),
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
    if (__kmpc_get_hardware_thread_id_in_block() == 0) {
      omptarget_nvptx_simpleMemoryManager.Release();
    }
    return;
  }
  __kmpc_impl_threadfence();
  ASSERT0(LT_FUSSY,
          __kmpc_get_hardware_thread_id_in_block() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  omptarget_nvptx_simpleMemoryManager.Release();
}

#pragma omp end declare target
