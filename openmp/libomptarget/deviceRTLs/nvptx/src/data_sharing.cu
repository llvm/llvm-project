//===----- data_sharing.cu - NVPTX OpenMP debug utilities -------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of data sharing environments/
//
//===----------------------------------------------------------------------===//
#include "omptarget-nvptx.h"
#include <stdio.h>

// Warp ID in the CUDA block
INLINE static unsigned getWarpId() { return threadIdx.x / WARPSIZE; }
// Lane ID in the CUDA warp.
INLINE static unsigned getLaneId() { return threadIdx.x % WARPSIZE; }

// Return true if this is the first active thread in the warp.
INLINE static bool IsWarpMasterActiveThread() {
  unsigned long long Mask = __ACTIVEMASK();
  unsigned long long ShNum = WARPSIZE - (GetThreadIdInBlock() % WARPSIZE);
  unsigned long long Sh = Mask << ShNum;
  // Truncate Sh to the 32 lower bits
  return (unsigned)Sh == 0;
}
// Return true if this is the master thread.
INLINE static bool IsMasterThread(bool isSPMDExecutionMode) {
  return !isSPMDExecutionMode && GetMasterThreadID() == GetThreadIdInBlock();
}

/// Return the provided size aligned to the size of a pointer.
INLINE static size_t AlignVal(size_t Val) {
  const size_t Align = (size_t)sizeof(void *);
  if (Val & (Align - 1)) {
    Val += Align;
    Val &= ~(Align - 1);
  }
  return Val;
}

#define DSFLAG 0
#define DSFLAG_INIT 0
#define DSPRINT(_flag, _str, _args...)                                         \
  {                                                                            \
    if (_flag) {                                                               \
      /*printf("(%d,%d) -> " _str, blockIdx.x, threadIdx.x, _args);*/          \
    }                                                                          \
  }
#define DSPRINT0(_flag, _str)                                                  \
  {                                                                            \
    if (_flag) {                                                               \
      /*printf("(%d,%d) -> " _str, blockIdx.x, threadIdx.x);*/                 \
    }                                                                          \
  }

// Initialize the shared data structures. This is expected to be called for the
// master thread and warp masters. \param RootS: A pointer to the root of the
// data sharing stack. \param InitialDataSize: The initial size of the data in
// the slot.
EXTERN void
__kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot *rootS,
                                           size_t InitialDataSize) {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Runtime must be initialized.");
  DSPRINT0(DSFLAG_INIT,
           "Entering __kmpc_initialize_data_sharing_environment\n");

  unsigned WID = getWarpId();
  DSPRINT(DSFLAG_INIT, "Warp ID: %u\n", WID);

  omptarget_nvptx_TeamDescr *teamDescr =
      &omptarget_nvptx_threadPrivateContext->TeamContext();
  __kmpc_data_sharing_slot *RootS =
      teamDescr->RootS(WID, IsMasterThread(isSPMDMode()));

  DataSharingState.SlotPtr[WID] = RootS;
  DataSharingState.StackPtr[WID] = (void *)&RootS->Data[0];

  // We don't need to initialize the frame and active threads.

  DSPRINT(DSFLAG_INIT, "Initial data size: %08x \n", (unsigned)InitialDataSize);
  DSPRINT(DSFLAG_INIT, "Root slot at: %016llx \n", (unsigned long long)RootS);
  DSPRINT(DSFLAG_INIT, "Root slot data-end at: %016llx \n",
          (unsigned long long)RootS->DataEnd);
  DSPRINT(DSFLAG_INIT, "Root slot next at: %016llx \n",
          (unsigned long long)RootS->Next);
  DSPRINT(DSFLAG_INIT, "Shared slot ptr at: %016llx \n",
          (unsigned long long)DataSharingState.SlotPtr[WID]);
  DSPRINT(DSFLAG_INIT, "Shared stack ptr at: %016llx \n",
          (unsigned long long)DataSharingState.StackPtr[WID]);

  DSPRINT0(DSFLAG_INIT, "Exiting __kmpc_initialize_data_sharing_environment\n");
}

EXTERN void *__kmpc_data_sharing_environment_begin(
    __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack,
    void **SavedSharedFrame, int32_t *SavedActiveThreads,
    size_t SharingDataSize, size_t SharingDefaultDataSize,
    int16_t IsOMPRuntimeInitialized) {

  DSPRINT0(DSFLAG, "Entering __kmpc_data_sharing_environment_begin\n");

  // If the runtime has been elided, used __shared__ memory for master-worker
  // data sharing.
  if (!IsOMPRuntimeInitialized)
    return (void *)&DataSharingState;

  DSPRINT(DSFLAG, "Data Size %016llx\n", (unsigned long long)SharingDataSize);
  DSPRINT(DSFLAG, "Default Data Size %016llx\n",
          (unsigned long long)SharingDefaultDataSize);

  unsigned WID = getWarpId();
  unsigned CurActiveThreads = __ACTIVEMASK();

  __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
  void *&StackP = DataSharingState.StackPtr[WID];
  void * volatile &FrameP = DataSharingState.FramePtr[WID];
  int32_t &ActiveT = DataSharingState.ActiveThreads[WID];

  DSPRINT0(DSFLAG, "Save current slot/stack values.\n");
  // Save the current values.
  *SavedSharedSlot = SlotP;
  *SavedSharedStack = StackP;
  *SavedSharedFrame = FrameP;
  *SavedActiveThreads = ActiveT;

  DSPRINT(DSFLAG, "Warp ID: %u\n", WID);
  DSPRINT(DSFLAG, "Saved slot ptr at: %016llx \n", (unsigned long long)SlotP);
  DSPRINT(DSFLAG, "Saved stack ptr at: %016llx \n", (unsigned long long)StackP);
  DSPRINT(DSFLAG, "Saved frame ptr at: %016llx \n", (long long)FrameP);
  DSPRINT(DSFLAG, "Active threads: %08x \n", (unsigned)ActiveT);

  // Only the warp active master needs to grow the stack.
  if (IsWarpMasterActiveThread()) {
    // Save the current active threads.
    ActiveT = CurActiveThreads;

    // Make sure we use aligned sizes to avoid rematerialization of data.
    SharingDataSize = AlignVal(SharingDataSize);
    // FIXME: The default data size can be assumed to be aligned?
    SharingDefaultDataSize = AlignVal(SharingDefaultDataSize);

    // Check if we have room for the data in the current slot.
    const uintptr_t CurrentStartAddress = (uintptr_t)StackP;
    const uintptr_t CurrentEndAddress = (uintptr_t)SlotP->DataEnd;
    const uintptr_t RequiredEndAddress =
        CurrentStartAddress + (uintptr_t)SharingDataSize;

    DSPRINT(DSFLAG, "Data Size %016llx\n", (unsigned long long)SharingDataSize);
    DSPRINT(DSFLAG, "Default Data Size %016llx\n",
            (unsigned long long)SharingDefaultDataSize);
    DSPRINT(DSFLAG, "Current Start Address %016llx\n",
            (unsigned long long)CurrentStartAddress);
    DSPRINT(DSFLAG, "Current End Address %016llx\n",
            (unsigned long long)CurrentEndAddress);
    DSPRINT(DSFLAG, "Required End Address %016llx\n",
            (unsigned long long)RequiredEndAddress);
    DSPRINT(DSFLAG, "Active Threads %08x\n", (unsigned)ActiveT);

    // If we require a new slot, allocate it and initialize it (or attempt to
    // reuse one). Also, set the shared stack and slot pointers to the new
    // place. If we do not need to grow the stack, just adapt the stack and
    // frame pointers.
    if (CurrentEndAddress < RequiredEndAddress) {
      size_t NewSize = (SharingDataSize > SharingDefaultDataSize)
                           ? SharingDataSize
                           : SharingDefaultDataSize;
      __kmpc_data_sharing_slot *NewSlot = 0;

      // Attempt to reuse an existing slot.
      if (__kmpc_data_sharing_slot *ExistingSlot = SlotP->Next) {
        uintptr_t ExistingSlotSize = (uintptr_t)ExistingSlot->DataEnd -
                                     (uintptr_t)(&ExistingSlot->Data[0]);
        if (ExistingSlotSize >= NewSize) {
          DSPRINT(DSFLAG, "Reusing stack slot %016llx\n",
                  (unsigned long long)ExistingSlot);
          NewSlot = ExistingSlot;
        } else {
          DSPRINT(DSFLAG, "Cleaning up -failed reuse - %016llx\n",
                  (unsigned long long)SlotP->Next);
          free(ExistingSlot);
        }
      }

      if (!NewSlot) {
        NewSlot = (__kmpc_data_sharing_slot *)malloc(
            sizeof(__kmpc_data_sharing_slot) + NewSize);
        DSPRINT(DSFLAG, "New slot allocated %016llx (data size=%016llx)\n",
                (unsigned long long)NewSlot, NewSize);
      }

      NewSlot->Next = 0;
      NewSlot->DataEnd = &NewSlot->Data[NewSize];

      SlotP->Next = NewSlot;
      SlotP = NewSlot;
      StackP = &NewSlot->Data[SharingDataSize];
      FrameP = &NewSlot->Data[0];
    } else {

      // Clean up any old slot that we may still have. The slot producers, do
      // not eliminate them because that may be used to return data.
      if (SlotP->Next) {
        DSPRINT(DSFLAG, "Cleaning up - old not required - %016llx\n",
                (unsigned long long)SlotP->Next);
        free(SlotP->Next);
        SlotP->Next = 0;
      }

      FrameP = StackP;
      StackP = (void *)RequiredEndAddress;
    }
  }

  // FIXME: Need to see the impact of doing it here.
  __threadfence_block();

  DSPRINT0(DSFLAG, "Exiting __kmpc_data_sharing_environment_begin\n");

  // All the threads in this warp get the frame they should work with.
  return FrameP;
}

EXTERN void __kmpc_data_sharing_environment_end(
    __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack,
    void **SavedSharedFrame, int32_t *SavedActiveThreads,
    int32_t IsEntryPoint) {

  DSPRINT0(DSFLAG, "Entering __kmpc_data_sharing_environment_end\n");

  unsigned WID = getWarpId();

  if (IsEntryPoint) {
    if (IsWarpMasterActiveThread()) {
      DSPRINT0(DSFLAG, "Doing clean up\n");

      // The master thread cleans the saved slot, because this is an environment
      // only for the master.
      __kmpc_data_sharing_slot *S = IsMasterThread(isSPMDMode())
                                        ? *SavedSharedSlot
                                        : DataSharingState.SlotPtr[WID];

      if (S->Next) {
        free(S->Next);
        S->Next = 0;
      }
    }

    DSPRINT0(DSFLAG, "Exiting Exiting __kmpc_data_sharing_environment_end\n");
    return;
  }

  int32_t CurActive = __ACTIVEMASK();

  // Only the warp master can restore the stack and frame information, and only
  // if there are no other threads left behind in this environment (i.e. the
  // warp diverged and returns in different places). This only works if we
  // assume that threads will converge right after the call site that started
  // the environment.
  if (IsWarpMasterActiveThread()) {
    int32_t &ActiveT = DataSharingState.ActiveThreads[WID];

    DSPRINT0(DSFLAG, "Before restoring the stack\n");
    // Zero the bits in the mask. If it is still different from zero, then we
    // have other threads that will return after the current ones.
    ActiveT &= ~CurActive;

    DSPRINT(DSFLAG, "Active threads: %08x; New mask: %08x\n",
            (unsigned)CurActive, (unsigned)ActiveT);

    if (!ActiveT) {
      // No other active threads? Great, lets restore the stack.

      __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
      void *&StackP = DataSharingState.StackPtr[WID];
      void * volatile &FrameP = DataSharingState.FramePtr[WID];

      SlotP = *SavedSharedSlot;
      StackP = *SavedSharedStack;
      FrameP = *SavedSharedFrame;
      ActiveT = *SavedActiveThreads;

      DSPRINT(DSFLAG, "Restored slot ptr at: %016llx \n",
              (unsigned long long)SlotP);
      DSPRINT(DSFLAG, "Restored stack ptr at: %016llx \n",
              (unsigned long long)StackP);
      DSPRINT(DSFLAG, "Restored frame ptr at: %016llx \n",
              (unsigned long long)FrameP);
      DSPRINT(DSFLAG, "Active threads: %08x \n", (unsigned)ActiveT);
    }
  }

  // FIXME: Need to see the impact of doing it here.
  __threadfence_block();

  DSPRINT0(DSFLAG, "Exiting __kmpc_data_sharing_environment_end\n");
  return;
}

EXTERN void *
__kmpc_get_data_sharing_environment_frame(int32_t SourceThreadID,
                                          int16_t IsOMPRuntimeInitialized) {
  DSPRINT0(DSFLAG, "Entering __kmpc_get_data_sharing_environment_frame\n");

  // If the runtime has been elided, use __shared__ memory for master-worker
  // data sharing.  We're reusing the statically allocated data structure
  // that is used for standard data sharing.
  if (!IsOMPRuntimeInitialized)
    return (void *)&DataSharingState;

  // Get the frame used by the requested thread.

  unsigned SourceWID = SourceThreadID / WARPSIZE;

  DSPRINT(DSFLAG, "Source  warp: %u\n", SourceWID);

  void * volatile P = DataSharingState.FramePtr[SourceWID];
  DSPRINT0(DSFLAG, "Exiting __kmpc_get_data_sharing_environment_frame\n");
  return P;
}

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

INLINE static void data_sharing_init_stack_common() {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Runtime must be initialized.");
  omptarget_nvptx_TeamDescr *teamDescr =
      &omptarget_nvptx_threadPrivateContext->TeamContext();

  for (int WID = 0; WID < WARPSIZE; WID++) {
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
  if (threadIdx.x == 0)
    data_sharing_init_stack_common();

  __threadfence_block();
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
  const unsigned WID = getWarpId();
  void *FrameP = 0;
  int32_t CurActive = __ACTIVEMASK();

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
  ((int *)&FrameP)[0] = __SHFL_SYNC(CurActive, ((int *)&FrameP)[0], 0);
  if (sizeof(FrameP) == 8)
    ((int *)&FrameP)[1] = __SHFL_SYNC(CurActive, ((int *)&FrameP)[1], 0);

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
  FrameStartAddress += (uintptr_t) (getLaneId() * DataSize);
  return (void *)FrameStartAddress;
}

// Pop the stack and free any memory which can be reclaimed.
//
// When the pop operation removes the last global memory slot,
// reclaim all outstanding global memory slots since it is
// likely we have reached the end of the kernel.
EXTERN void __kmpc_data_sharing_pop_stack(void *FrameStart) {
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  __threadfence_block();

  if (GetThreadIdInBlock() % WARPSIZE == 0) {
    unsigned WID = getWarpId();

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
    // FIXME: use __syncthreads instead when the function copy is fixed in LLVM.
    __SYNCTHREADS();
    return;
  }
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
  __threadfence();
}

EXTERN void __kmpc_restore_team_static_memory(int16_t isSPMDExecutionMode,
                                              int16_t is_shared) {
  if (is_shared)
    return;
  if (isSPMDExecutionMode) {
    // FIXME: use __syncthreads instead when the function copy is fixed in LLVM.
    __SYNCTHREADS();
    if (GetThreadIdInBlock() == 0) {
      omptarget_nvptx_simpleMemoryManager.Release();
    }
    return;
  }
  __threadfence();
  ASSERT0(LT_FUSSY, GetThreadIdInBlock() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  omptarget_nvptx_simpleMemoryManager.Release();
}

