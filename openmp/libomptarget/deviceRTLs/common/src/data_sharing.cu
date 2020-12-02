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
#include "common/omptarget.h"
#include "target_impl.h"

// Return true if this is the master thread.
INLINE static bool IsMasterThread(bool isSPMDExecutionMode) {
  return !isSPMDExecutionMode && GetMasterThreadID() == GetThreadIdInBlock();
}

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

INLINE static void data_sharing_init_stack_common() {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Runtime must be initialized.");
  omptarget_nvptx_TeamDescr *teamDescr =
      &omptarget_nvptx_threadPrivateContext->TeamContext();

  for (int WID = 0; WID < DS_Max_Warp_Number; WID++) {
    __kmpc_data_sharing_slot *RootS = teamDescr->GetPreallocatedSlotAddr(WID);
    DataSharingState.SlotPtr[WID] = RootS;
    DataSharingState.StackPtr[WID] = (void *)&RootS->Data[0];
  }
}

// Initialize data sharing data structure. This function needs to be called
// once at the beginning of a data sharing context (coincides with the kernel
// initialization). This function is called only by the MASTER thread of each
// team in non-SPMD mode.
EXTERN void __kmpc_data_sharing_init_stack() {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Runtime must be initialized.");
  // This function initializes the stack pointer with the pointer to the
  // statically allocated shared memory slots. The size of a shared memory
  // slot is pre-determined to be 256 bytes.
  data_sharing_init_stack_common();
  omptarget_nvptx_globalArgs.Init();
}

// Initialize data sharing data structure. This function needs to be called
// once at the beginning of a data sharing context (coincides with the kernel
// initialization). This function is called in SPMD mode only.
EXTERN void __kmpc_data_sharing_init_stack_spmd() {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Runtime must be initialized.");
  // This function initializes the stack pointer with the pointer to the
  // statically allocated shared memory slots. The size of a shared memory
  // slot is pre-determined to be 256 bytes.
  if (GetThreadIdInBlock() == 0)
    data_sharing_init_stack_common();

  __kmpc_impl_threadfence_block();
}

INLINE static void* data_sharing_push_stack_common(size_t PushSize) {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  // Only warp active master threads manage the stack.
  bool IsWarpMaster = (GetThreadIdInBlock() % WARPSIZE) == 0;

  // Add worst-case padding to DataSize so that future stack allocations are
  // correctly aligned.
  const size_t Alignment = 8;
  PushSize = (PushSize + (Alignment - 1)) / Alignment * Alignment;

  // Frame pointer must be visible to all workers in the same warp.
  const unsigned WID = GetWarpId();
  void *FrameP = 0;
  __kmpc_impl_lanemask_t CurActive = __kmpc_impl_activemask();

  if (IsWarpMaster) {
    // SlotP will point to either the shared memory slot or an existing
    // global memory slot.
    __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
    void *&StackP = DataSharingState.StackPtr[WID];

    // Check if we have room for the data in the current slot.
    const uintptr_t StartAddress = (uintptr_t)StackP;
    const uintptr_t EndAddress = (uintptr_t)SlotP->DataEnd;
    const uintptr_t RequestedEndAddress = StartAddress + (uintptr_t)PushSize;

    // If we requested more data than there is room for in the rest
    // of the slot then we need to either re-use the next slot, if one exists,
    // or create a new slot.
    if (EndAddress < RequestedEndAddress) {
      __kmpc_data_sharing_slot *NewSlot = 0;
      size_t NewSize = PushSize;

      // Allocate at least the default size for each type of slot.
      // Master is a special case and even though there is only one thread,
      // it can share more things with the workers. For uniformity, it uses
      // the full size of a worker warp slot.
      size_t DefaultSlotSize = DS_Worker_Warp_Slot_Size;
      if (DefaultSlotSize > NewSize)
        NewSize = DefaultSlotSize;
      NewSlot = (__kmpc_data_sharing_slot *) SafeMalloc(
          sizeof(__kmpc_data_sharing_slot) + NewSize,
          "Global memory slot allocation.");

      NewSlot->Next = 0;
      NewSlot->Prev = SlotP;
      NewSlot->PrevSlotStackPtr = StackP;
      NewSlot->DataEnd = &NewSlot->Data[0] + NewSize;

      // Make previous slot point to the newly allocated slot.
      SlotP->Next = NewSlot;
      // The current slot becomes the new slot.
      SlotP = NewSlot;
      // The stack pointer always points to the next free stack frame.
      StackP = &NewSlot->Data[0] + PushSize;
      // The frame pointer always points to the beginning of the frame.
      FrameP = DataSharingState.FramePtr[WID] = &NewSlot->Data[0];
    } else {
      // Add the data chunk to the current slot. The frame pointer is set to
      // point to the start of the new frame held in StackP.
      FrameP = DataSharingState.FramePtr[WID] = StackP;
      // Reset stack pointer to the requested address.
      StackP = (void *)RequestedEndAddress;
    }
  }
  // Get address from lane 0.
  int *FP = (int *)&FrameP;
  FP[0] = __kmpc_impl_shfl_sync(CurActive, FP[0], 0);
  if (sizeof(FrameP) == 8)
    FP[1] = __kmpc_impl_shfl_sync(CurActive, FP[1], 0);

  return FrameP;
}

EXTERN void *__kmpc_data_sharing_coalesced_push_stack(size_t DataSize,
                                                      int16_t UseSharedMemory) {
  return data_sharing_push_stack_common(DataSize);
}

// Called at the time of the kernel initialization. This is used to initilize
// the list of references to shared variables and to pre-allocate global storage
// for holding the globalized variables.
//
// By default the globalized variables are stored in global memory. If the
// UseSharedMemory is set to true, the runtime will attempt to use shared memory
// as long as the size requested fits the pre-allocated size.
EXTERN void *__kmpc_data_sharing_push_stack(size_t DataSize,
                                            int16_t UseSharedMemory) {
  // Compute the total memory footprint of the requested data.
  // The master thread requires a stack only for itself. A worker
  // thread (which at this point is a warp master) will require
  // space for the variables of each thread in the warp,
  // i.e. one DataSize chunk per warp lane.
  // TODO: change WARPSIZE to the number of active threads in the warp.
  size_t PushSize = (isRuntimeUninitialized() || IsMasterThread(isSPMDMode()))
                        ? DataSize
                        : WARPSIZE * DataSize;

  // Compute the start address of the frame of each thread in the warp.
  uintptr_t FrameStartAddress =
      (uintptr_t) data_sharing_push_stack_common(PushSize);
  FrameStartAddress += (uintptr_t) (GetLaneId() * DataSize);
  return (void *)FrameStartAddress;
}

// Pop the stack and free any memory which can be reclaimed.
//
// When the pop operation removes the last global memory slot,
// reclaim all outstanding global memory slots since it is
// likely we have reached the end of the kernel.
EXTERN void __kmpc_data_sharing_pop_stack(void *FrameStart) {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  __kmpc_impl_threadfence_block();

  if (GetThreadIdInBlock() % WARPSIZE == 0) {
    unsigned WID = GetWarpId();

    // Current slot
    __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];

    // Pointer to next available stack.
    void *&StackP = DataSharingState.StackPtr[WID];

    // Pop the frame.
    StackP = FrameStart;

    // If the current slot is empty, we need to free the slot after the
    // pop.
    bool SlotEmpty = (StackP == &SlotP->Data[0]);

    if (SlotEmpty && SlotP->Prev) {
      // Before removing the slot we need to reset StackP.
      StackP = SlotP->PrevSlotStackPtr;

      // Remove the slot.
      SlotP = SlotP->Prev;
      SafeFree(SlotP->Next, "Free slot.");
      SlotP->Next = 0;
    }
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

