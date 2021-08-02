//===--- omptarget.cu - OpenMP GPU initialization ---------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/omptarget.h"
#include "common/support.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext,
                             OMP_STATE_COUNT>
    omptarget_nvptx_device_State[MAX_SM];

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

static void __kmpc_generic_kernel_init() {
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n",
        OMPTARGET_NVPTX_VERSION);

  if (GetLaneId() == 0)
    parallelLevel[GetWarpId()] = 0;

  int threadIdInBlock = __kmpc_get_hardware_thread_id_in_block();
  if (threadIdInBlock != GetMasterThreadID())
    return;

  setExecutionParameters(Generic, RuntimeInitialized);
  ASSERT0(LT_FUSSY, threadIdInBlock == GetMasterThreadID(),
          "__kmpc_kernel_init() must be called by team master warp only!");
  PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");

  // Get a state object from the queue.
  int slot = __kmpc_impl_smid() % MAX_SM;
  usedSlotIdx = slot;
  omptarget_nvptx_threadPrivateContext =
      omptarget_nvptx_device_State[slot].Dequeue();

  // init thread private
  int threadId = 0;
  omptarget_nvptx_threadPrivateContext->InitThreadPrivateContext(threadId);

  // init team context
  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  currTeamDescr.InitTeamDescr();
  // this thread will start execution... has to update its task ICV
  // to point to the level zero task ICV. That ICV was init in
  // InitTeamDescr()
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTeamDescr.LevelZeroTaskDescr());

  // set number of threads and thread limit in team to started value
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
  nThreads = GetNumberOfWorkersInTeam();
  threadLimit = nThreads;

  __kmpc_data_sharing_init_stack();
  __kmpc_impl_target_init();
}

static void __kmpc_generic_kernel_deinit() {
  PRINT0(LD_IO, "call to __kmpc_kernel_deinit\n");
  // Enqueue omp state object for use by another team.
  int slot = usedSlotIdx;
  omptarget_nvptx_device_State[slot].Enqueue(
      omptarget_nvptx_threadPrivateContext);
  // Done with work.  Kill the workers.
  omptarget_nvptx_workFn = 0;
}

static void __kmpc_spmd_kernel_init(bool RequiresFullRuntime) {
  PRINT0(LD_IO, "call to __kmpc_spmd_kernel_init\n");

  setExecutionParameters(Spmd, RequiresFullRuntime ? RuntimeInitialized
                         : RuntimeUninitialized);
  int threadId = __kmpc_get_hardware_thread_id_in_block();
  if (threadId == 0) {
    usedSlotIdx = __kmpc_impl_smid() % MAX_SM;
  }

  if (GetLaneId() == 0) {
    parallelLevel[GetWarpId()] =
        1 + (__kmpc_get_hardware_num_threads_in_block() > 1
                 ? OMP_ACTIVE_PARALLEL_LEVEL
                 : 0);
  }

  __kmpc_data_sharing_init_stack();
  if (!RequiresFullRuntime)
    return;

  //
  // Team Context Initialization.
  //
  // In SPMD mode there is no master thread so use any cuda thread for team
  // context initialization.
  if (threadId == 0) {
    // Get a state object from the queue.
    omptarget_nvptx_threadPrivateContext =
        omptarget_nvptx_device_State[usedSlotIdx].Dequeue();

    omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
    omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
    // init team context
    currTeamDescr.InitTeamDescr();
  }
  __kmpc_impl_syncthreads();

  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();

  //
  // Initialize task descr for each thread.
  //
  omptarget_nvptx_TaskDescr *newTaskDescr =
      omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
  ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
  newTaskDescr->InitLevelOneTaskDescr(currTeamDescr.LevelZeroTaskDescr());
  // install new top descriptor
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);

  // init thread private from init value
  int ThreadLimit = GetNumberOfProcsInTeam(/* IsSPMD */ true);
  PRINT(LD_PAR,
        "thread will execute parallel region with id %d in a team of "
        "%d threads\n",
        (int)newTaskDescr->ThreadId(), (int)ThreadLimit);
}

static void __kmpc_spmd_kernel_deinit(bool RequiresFullRuntime) {
  // We're not going to pop the task descr stack of each thread since
  // there are no more parallel regions in SPMD mode.
  if (!RequiresFullRuntime)
    return;

  __kmpc_impl_syncthreads();
  int threadId = __kmpc_get_hardware_thread_id_in_block();
  if (threadId == 0) {
    // Enqueue omp state object for use by another team.
    int slot = usedSlotIdx;
    omptarget_nvptx_device_State[slot].Enqueue(
        omptarget_nvptx_threadPrivateContext);
  }
}

// Return true if the current target region is executed in SPMD mode.
EXTERN int8_t __kmpc_is_spmd_exec_mode() {
  return (execution_param & ModeMask) == Spmd;
}

EXTERN int8_t __kmpc_is_generic_main_thread(kmp_int32 Tid) {
  return !__kmpc_is_spmd_exec_mode() && __kmpc_is_generic_main_thread_id(Tid);
}

NOINLINE EXTERN int8_t __kmpc_is_generic_main_thread_id(kmp_int32 Tid) {
  return GetMasterThreadID() == Tid;
}

EXTERN bool __kmpc_kernel_parallel(void**WorkFn);

static void __kmpc_target_region_state_machine(ident_t *Ident) {

  int TId = __kmpc_get_hardware_thread_id_in_block();
  do {
    void* WorkFn = 0;

    // Wait for the signal that we have a new work function.
    __kmpc_barrier_simple_spmd(Ident, TId);


    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ((void(*)(uint32_t,uint32_t))WorkFn)(0, TId);
      __kmpc_kernel_end_parallel();
    }

    __kmpc_barrier_simple_spmd(Ident, TId);

  } while (true);
}

EXTERN
int32_t __kmpc_target_init(ident_t *Ident, bool IsSPMD,
                           bool UseGenericStateMachine,
                           bool RequiresFullRuntime) {
  int TId = __kmpc_get_hardware_thread_id_in_block();
  if (IsSPMD)
    __kmpc_spmd_kernel_init(RequiresFullRuntime);
  else
    __kmpc_generic_kernel_init();

   if (IsSPMD) {
    __kmpc_barrier_simple_spmd(Ident, TId);
     return -1;
   }

   if (TId == GetMasterThreadID())
     return -1;

  if (UseGenericStateMachine)
    __kmpc_target_region_state_machine(Ident);

  return TId;
}

EXTERN
void __kmpc_target_deinit(ident_t *Ident, bool IsSPMD,
                           bool RequiresFullRuntime) {
  if (IsSPMD)
    __kmpc_spmd_kernel_deinit(RequiresFullRuntime);
  else
    __kmpc_generic_kernel_deinit();
}


#pragma omp end declare target
