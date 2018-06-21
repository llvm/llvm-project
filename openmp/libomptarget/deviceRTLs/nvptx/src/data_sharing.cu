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

// Number of threads in the CUDA block.
__device__ static unsigned getNumThreads() { return blockDim.x; }
// Thread ID in the CUDA block
__device__ static unsigned getThreadId() { return threadIdx.x; }
// Warp ID in the CUDA block
__device__ static unsigned getWarpId() { return threadIdx.x / WARPSIZE; }
// Lane ID in the CUDA warp.
__device__ static unsigned getLaneId() { return threadIdx.x % WARPSIZE; }

// The CUDA thread ID of the master thread.
__device__ static unsigned getMasterThreadId() {
  unsigned Mask = WARPSIZE - 1;
  return (getNumThreads() - 1) & (~Mask);
}

// Find the active threads in the warp - return a mask whose n-th bit is set if
// the n-th thread in the warp is active.
__device__ static unsigned getActiveThreadsMask() {
  return __BALLOT_SYNC(0xFFFFFFFF, true);
}

// Return true if this is the first active thread in the warp.
__device__ static bool IsWarpMasterActiveThread() {
  unsigned long long Mask = getActiveThreadsMask();
  unsigned long long ShNum = WARPSIZE - (getThreadId() % WARPSIZE);
  unsigned long long Sh = Mask << ShNum;
  // Truncate Sh to the 32 lower bits
  return (unsigned)Sh == 0;
}
// Return true if this is the master thread.
__device__ static bool IsMasterThread() {
  return getMasterThreadId() == getThreadId();
}

/// Return the provided size aligned to the size of a pointer.
__device__ static size_t AlignVal(size_t Val) {
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

  DSPRINT0(DSFLAG_INIT,
           "Entering __kmpc_initialize_data_sharing_environment\n");

  unsigned WID = getWarpId();
  DSPRINT(DSFLAG_INIT, "Warp ID: %d\n", WID);

  omptarget_nvptx_TeamDescr *teamDescr =
      &omptarget_nvptx_threadPrivateContext->TeamContext();
  __kmpc_data_sharing_slot *RootS = teamDescr->RootS(WID, IsMasterThread());

  DataSharingState.SlotPtr[WID] = RootS;
  DataSharingState.StackPtr[WID] = (void *)&RootS->Data[0];

  // We don't need to initialize the frame and active threads.

  DSPRINT(DSFLAG_INIT, "Initial data size: %08x \n", InitialDataSize);
  DSPRINT(DSFLAG_INIT, "Root slot at: %016llx \n", (long long)RootS);
  DSPRINT(DSFLAG_INIT, "Root slot data-end at: %016llx \n",
          (long long)RootS->DataEnd);
  DSPRINT(DSFLAG_INIT, "Root slot next at: %016llx \n", (long long)RootS->Next);
  DSPRINT(DSFLAG_INIT, "Shared slot ptr at: %016llx \n",
          (long long)DataSharingState.SlotPtr[WID]);
  DSPRINT(DSFLAG_INIT, "Shared stack ptr at: %016llx \n",
          (long long)DataSharingState.StackPtr[WID]);

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

  DSPRINT(DSFLAG, "Data Size %016llx\n", SharingDataSize);
  DSPRINT(DSFLAG, "Default Data Size %016llx\n", SharingDefaultDataSize);

  unsigned WID = getWarpId();
  unsigned CurActiveThreads = getActiveThreadsMask();

  __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
  void *&StackP = DataSharingState.StackPtr[WID];
  void *&FrameP = DataSharingState.FramePtr[WID];
  int32_t &ActiveT = DataSharingState.ActiveThreads[WID];

  DSPRINT0(DSFLAG, "Save current slot/stack values.\n");
  // Save the current values.
  *SavedSharedSlot = SlotP;
  *SavedSharedStack = StackP;
  *SavedSharedFrame = FrameP;
  *SavedActiveThreads = ActiveT;

  DSPRINT(DSFLAG, "Warp ID: %d\n", WID);
  DSPRINT(DSFLAG, "Saved slot ptr at: %016llx \n", (long long)SlotP);
  DSPRINT(DSFLAG, "Saved stack ptr at: %016llx \n", (long long)StackP);
  DSPRINT(DSFLAG, "Saved frame ptr at: %016llx \n", (long long)FrameP);
  DSPRINT(DSFLAG, "Active threads: %08x \n", ActiveT);

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

    DSPRINT(DSFLAG, "Data Size %016llx\n", SharingDataSize);
    DSPRINT(DSFLAG, "Default Data Size %016llx\n", SharingDefaultDataSize);
    DSPRINT(DSFLAG, "Current Start Address %016llx\n", CurrentStartAddress);
    DSPRINT(DSFLAG, "Current End Address %016llx\n", CurrentEndAddress);
    DSPRINT(DSFLAG, "Required End Address %016llx\n", RequiredEndAddress);
    DSPRINT(DSFLAG, "Active Threads %08x\n", ActiveT);

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
                  (long long)ExistingSlot);
          NewSlot = ExistingSlot;
        } else {
          DSPRINT(DSFLAG, "Cleaning up -failed reuse - %016llx\n",
                  (long long)SlotP->Next);
          free(ExistingSlot);
        }
      }

      if (!NewSlot) {
        NewSlot = (__kmpc_data_sharing_slot *)malloc(
            sizeof(__kmpc_data_sharing_slot) + NewSize);
        DSPRINT(DSFLAG, "New slot allocated %016llx (data size=%016llx)\n",
                (long long)NewSlot, NewSize);
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
                (long long)SlotP->Next);
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
      __kmpc_data_sharing_slot *S =
          IsMasterThread() ? *SavedSharedSlot : DataSharingState.SlotPtr[WID];

      if (S->Next) {
        free(S->Next);
        S->Next = 0;
      }
    }

    DSPRINT0(DSFLAG, "Exiting Exiting __kmpc_data_sharing_environment_end\n");
    return;
  }

  int32_t CurActive = getActiveThreadsMask();

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

    DSPRINT(DSFLAG, "Active threads: %08x; New mask: %08x\n", CurActive,
            ActiveT);

    if (!ActiveT) {
      // No other active threads? Great, lets restore the stack.

      __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
      void *&StackP = DataSharingState.StackPtr[WID];
      void *&FrameP = DataSharingState.FramePtr[WID];

      SlotP = *SavedSharedSlot;
      StackP = *SavedSharedStack;
      FrameP = *SavedSharedFrame;
      ActiveT = *SavedActiveThreads;

      DSPRINT(DSFLAG, "Restored slot ptr at: %016llx \n", (long long)SlotP);
      DSPRINT(DSFLAG, "Restored stack ptr at: %016llx \n", (long long)StackP);
      DSPRINT(DSFLAG, "Restored frame ptr at: %016llx \n", (long long)FrameP);
      DSPRINT(DSFLAG, "Active threads: %08x \n", ActiveT);
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

  DSPRINT(DSFLAG, "Source  warp: %d\n", SourceWID);

  void *P = DataSharingState.FramePtr[SourceWID];
  DSPRINT0(DSFLAG, "Exiting __kmpc_get_data_sharing_environment_frame\n");
  return P;
}

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

// Initialize data sharing data structure. This function needs to be called
// once at the beginning of a data sharing context (coincides with the kernel
// initialization).
EXTERN void __kmpc_data_sharing_init_stack() {
  // This function initializes the stack pointer with the pointer to the
  // statically allocated shared memory slots. The size of a shared memory
  // slot is pre-determined to be 256 bytes.

  // Initialize the data sharing structures. This section should only be
  // executed by the warp active master threads.
  if (IsWarpMasterActiveThread()) {
    unsigned WID = getWarpId();
    omptarget_nvptx_TeamDescr *teamDescr =
        &omptarget_nvptx_threadPrivateContext->TeamContext();
    __kmpc_data_sharing_slot *RootS = teamDescr->RootS(WID, IsMasterThread());

    // If a valid address has been returned then proceed with the initalization.
    // Otherwise the initialization of the slot has already happened in a
    // previous call to this function.
    if (RootS) {
      DataSharingState.SlotPtr[WID] = RootS;
      DataSharingState.TailPtr[WID] = RootS;
      DataSharingState.StackPtr[WID] = (void *)&RootS->Data[0];
    }
  }

  // Currently we only support the sharing of variables between master and
  // workers. The list of references to shared variables exists only for
  // the master thread.
  if (IsMasterThread()) {
    // Initialize the list of references to arguments.
    omptarget_nvptx_globalArgs.Init();
  }
}

// Called at the time of the kernel initialization. This is used to initilize
// the list of references to shared variables and to pre-allocate global storage
// for holding the globalized variables.
//
// By default the globalized variables are stored in global memory. If the
// UseSharedMemory is set to true, the runtime will attempt to use shared memory
// as long as the size requested fits the pre-allocated size.
//
// Called by: master, TODO: call by workers
EXTERN void* __kmpc_data_sharing_push_stack(size_t DataSize,
    int16_t UseSharedMemory) {
  // Frame pointer must be visible to all workers in the same warp.
  unsigned WID = getWarpId();
  void *&FrameP = DataSharingState.FramePtr[WID];

  // Only warp active master threads manage the stack.
  if (IsWarpMasterActiveThread()) {
    // SlotP will point to either the shared memory slot or an existing
    // global memory slot.
    __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
    __kmpc_data_sharing_slot *&TailSlotP = DataSharingState.TailPtr[WID];
    void *&StackP = DataSharingState.StackPtr[WID];

    // Compute the total memory footprint of the requested data.
    // The master thread requires a stack only for itself. A worker
    // thread (which at this point is a warp master) will require
    // space for the variables of each thread in the warp,
    // i.e. one DataSize chunk per warp lane.
    // TODO: change WARPSIZE to the number of active threads in the warp.
    size_t PushSize = IsMasterThread() ? DataSize : WARPSIZE * DataSize;

    // Check if we have room for the data in the current slot.
    const uintptr_t StartAddress = (uintptr_t)StackP;
    const uintptr_t EndAddress = (uintptr_t)SlotP->DataEnd;
    const uintptr_t RequestedEndAddress = StartAddress + (uintptr_t)PushSize;

    // If we requested more data than there is room for in the rest
    // of the slot then we need to either re-use the next slot, if one exists,
    // or create a new slot.
    if (EndAddress < RequestedEndAddress) {
      size_t NewSize = PushSize;

      // The new or reused slot for holding the data being pushed.
      __kmpc_data_sharing_slot *NewSlot = 0;

      // Check if there is a next slot.
      if (__kmpc_data_sharing_slot *ExistingSlot = SlotP->Next) {
        // Attempt to reuse an existing slot provided the data fits in the slot.
        // The leftover data space will not be used.
        ptrdiff_t ExistingSlotSize = (uintptr_t)ExistingSlot->DataEnd -
                                     (uintptr_t)(&ExistingSlot->Data[0]);

        // Try to add the data in the next available slot. Search for a slot
        // with enough space.
        while (ExistingSlotSize < NewSize) {
          SlotP->Next = ExistingSlot->Next;
          SlotP->Next->Prev = ExistingSlot->Prev;
          free(ExistingSlot);
          ExistingSlot = SlotP->Next;
          if (!ExistingSlot)
            break;
          ExistingSlotSize = (uintptr_t)ExistingSlot->DataEnd -
                             (uintptr_t)(&ExistingSlot->Data[0]);
        }

        // Check if a slot has been found.
        if (ExistingSlotSize >= NewSize) {
          NewSlot = ExistingSlot;
          NewSlot->PrevSlotStackPtr = StackP;
        }
      }

      if (!NewSlot) {
        // Allocate at least the default size for each type of slot.
        size_t DefaultSlotSize =
            IsMasterThread() ? DS_Slot_Size : DS_Worker_Warp_Slot_Size;
        if (DefaultSlotSize > NewSize)
          NewSize = DefaultSlotSize;
        NewSlot = (__kmpc_data_sharing_slot *)malloc(
            sizeof(__kmpc_data_sharing_slot) + NewSize);
        NewSlot->Next = 0;
        NewSlot->Prev = SlotP;
        NewSlot->PrevSlotStackPtr = StackP;
        NewSlot->DataEnd = &NewSlot->Data[NewSize];

        // Newly allocated slots are also tail slots.
        TailSlotP = NewSlot;

        // Make previous slot point to the newly allocated slot.
        SlotP->Next = NewSlot;
      }

      // The current slot becomes the new slot.
      SlotP = NewSlot;
      // The stack pointer always points to the next free stack frame.
      StackP = &NewSlot->Data[PushSize];
      // The frame pointer always points to the beginning of the frame.
      FrameP = &NewSlot->Data[0];
    } else {
      // Add the data chunk to the current slot. The frame pointer is set to
      // point to the start of the new frame held in StackP.
      FrameP = StackP;
      // Reset stack pointer to the requested address.
      StackP = (void *)RequestedEndAddress;
    }
  }

  __threadfence_block();

  // Compute the start address of the frame of each thread in the warp.
  uintptr_t FrameStartAddress = (uintptr_t)FrameP;
  FrameStartAddress += (uintptr_t) (getLaneId() * DataSize);
  return (void *)FrameStartAddress;
}

// Pop the stack and free any memory which can be reclaimed.
//
// When the pop operation removes the last global memory slot,
// reclaim all outstanding global memory slots since it is
// likely we have reached the end of the kernel.
EXTERN void __kmpc_data_sharing_pop_stack(void *FrameStart) {
  if (IsWarpMasterActiveThread()) {
    unsigned WID = getWarpId();

    __kmpc_data_sharing_slot *&SlotP = DataSharingState.SlotPtr[WID];
    void *&StackP = DataSharingState.StackPtr[WID];

    // Pop current frame from slot.
    StackP = FrameStart;

    // If we try to pop the last frame of the current slot we need to
    // move to the previous slot if there is one.
    const uintptr_t StartAddress = (uintptr_t)FrameStart;
    if (StartAddress == (uintptr_t)&SlotP->Data[0]) {
      if (SlotP->Prev) {
        // The new stack pointer is the end of the data field of the
        // previous slot. This will allow the stack pointer to be
        // used in the computation of the remaining data space in
        // the current slot.
        StackP = SlotP->PrevSlotStackPtr;
        // Reset SlotP to previous slot.
        SlotP = SlotP->Prev;
      }

      // If this will "pop" the last global memory node then it is likely
      // that we are at the end of the data sharing region and we can
      // de-allocate any existing global memory slots.
      if (!SlotP->Prev) {
        __kmpc_data_sharing_slot *Tail = DataSharingState.TailPtr[WID];
        while(Tail->Prev) {
          Tail = Tail->Prev;
          free(Tail->Next);
        }
        Tail->Next=0;
      }
    }
  }

  __threadfence_block();
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
