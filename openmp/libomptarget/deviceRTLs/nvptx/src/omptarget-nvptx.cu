//===--- omptarget-nvptx.cu - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern __device__
    omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
        omptarget_nvptx_device_State[MAX_SM];

extern __device__ __shared__
    omptarget_nvptx_ThreadPrivateContext *omptarget_nvptx_threadPrivateContext;

//
// The team master sets the outlined function and its arguments in these
// variables to communicate with the workers.  Since they are in shared memory,
// there is one copy of these variables for each kernel, instance, and team.
//
extern volatile __device__ __shared__ omptarget_nvptx_WorkFn
    omptarget_nvptx_workFn;
extern __device__ __shared__ uint32_t execution_param;

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

INLINE unsigned smid() {
  unsigned id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

EXTERN void __kmpc_kernel_init_params(void *Ptr) {
  PRINT(LD_IO, "call to __kmpc_kernel_init_params with version %f\n",
        OMPTARGET_NVPTX_VERSION);

  SetTeamsReductionScratchpadPtr(Ptr);
}

EXTERN void __kmpc_kernel_init(int ThreadLimit, int16_t RequiresOMPRuntime) {
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n",
        OMPTARGET_NVPTX_VERSION);

  if (!RequiresOMPRuntime) {
    // If OMP runtime is not required don't initialize OMP state.
    setExecutionParameters(Generic, RuntimeUninitialized);
    return;
  }
  setExecutionParameters(Generic, RuntimeInitialized);

  int threadIdInBlock = GetThreadIdInBlock();
  ASSERT0(LT_FUSSY, threadIdInBlock == GetMasterThreadID(),
          "__kmpc_kernel_init() must be called by team master warp only!");
  PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");

  // Get a state object from the queue.
  int slot = smid() % MAX_SM;
  omptarget_nvptx_threadPrivateContext =
      omptarget_nvptx_device_State[slot].Dequeue();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  omptarget_nvptx_threadPrivateContext->SetSourceQueue(slot);
#endif

  // init thread private
  int threadId = GetLogicalThreadIdInBlock();
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
  currTaskDescr->NThreads() = GetNumberOfWorkersInTeam();
  currTaskDescr->ThreadLimit() = ThreadLimit;
}

EXTERN void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized) {
  if (IsOMPRuntimeInitialized) {
    // Enqueue omp state object for use by another team.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    int slot = omptarget_nvptx_threadPrivateContext->GetSourceQueue();
#else
    int slot = smid() % MAX_SM;
#endif
    omptarget_nvptx_device_State[slot].Enqueue(
        omptarget_nvptx_threadPrivateContext);
  }
  // Done with work.  Kill the workers.
  omptarget_nvptx_workFn = 0;
}

EXTERN void __kmpc_spmd_kernel_init(int ThreadLimit, int16_t RequiresOMPRuntime,
                                    int16_t RequiresDataSharing) {
  PRINT0(LD_IO, "call to __kmpc_spmd_kernel_init\n");

  if (!RequiresOMPRuntime) {
    // If OMP runtime is not required don't initialize OMP state.
    setExecutionParameters(Spmd, RuntimeUninitialized);
    return;
  }
  setExecutionParameters(Spmd, RuntimeInitialized);

  //
  // Team Context Initialization.
  //
  // In SPMD mode there is no master thread so use any cuda thread for team
  // context initialization.
  int threadId = GetThreadIdInBlock();
  if (threadId == 0) {
    // Get a state object from the queue.
    int slot = smid() % MAX_SM;
    omptarget_nvptx_threadPrivateContext =
        omptarget_nvptx_device_State[slot].Dequeue();

    omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
    omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
    // init team context
    currTeamDescr.InitTeamDescr();
    // init counters (copy start to init)
    workDescr.CounterGroup().Reset();
  }
  __syncthreads();

  omptarget_nvptx_TeamDescr &currTeamDescr = getMyTeamDescriptor();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();

  //
  // Initialize task descr for each thread.
  //
  omptarget_nvptx_TaskDescr *newTaskDescr =
      omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
  ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
  newTaskDescr->InitLevelOneTaskDescr(ThreadLimit,
                                      currTeamDescr.LevelZeroTaskDescr());
  newTaskDescr->ThreadLimit() = ThreadLimit;
  // install new top descriptor
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);

  // init thread private from init value
  workDescr.CounterGroup().Init(
      omptarget_nvptx_threadPrivateContext->Priv(threadId));
  PRINT(LD_PAR,
        "thread will execute parallel region with id %d in a team of "
        "%d threads\n",
        newTaskDescr->ThreadId(), newTaskDescr->ThreadsInTeam());

  if (RequiresDataSharing && threadId % WARPSIZE == 0) {
    // Warp master innitializes data sharing environment.
    unsigned WID = threadId / WARPSIZE;
    __kmpc_data_sharing_slot *RootS = currTeamDescr.RootS(WID);
    DataSharingState.SlotPtr[WID] = RootS;
    DataSharingState.StackPtr[WID] = (void *)&RootS->Data[0];
  }
}

EXTERN void __kmpc_spmd_kernel_deinit() {
  // We're not going to pop the task descr stack of each thread since
  // there are no more parallel regions in SPMD mode.
  __syncthreads();
  int threadId = GetThreadIdInBlock();
  if (threadId == 0) {
    // Enqueue omp state object for use by another team.
    int slot = smid() % MAX_SM;
    omptarget_nvptx_device_State[slot].Enqueue(
        omptarget_nvptx_threadPrivateContext);
  }
}
