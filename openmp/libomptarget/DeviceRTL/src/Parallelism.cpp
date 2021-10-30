//===---- Parallelism.cpp - OpenMP GPU parallel implementation ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parallel implementation in the GPU. Here is the pattern:
//
//    while (not finished) {
//
//    if (master) {
//      sequential code, decide which par loop to do, or if finished
//     __kmpc_kernel_prepare_parallel() // exec by master only
//    }
//    syncthreads // A
//    __kmpc_kernel_parallel() // exec by all
//    if (this thread is included in the parallel) {
//      switch () for all parallel loops
//      __kmpc_kernel_end_parallel() // exec only by threads in parallel
//    }
//
//
//    The reason we don't exec end_parallel for the threads not included
//    in the parallel loop is that for each barrier in the parallel
//    region, these non-included threads will cycle through the
//    syncthread A. Thus they must preserve their current threadId that
//    is larger than thread in team.
//
//    To make a long story short...
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

#pragma omp declare target

namespace {

uint32_t determineNumberOfThreads(int32_t NumThreadsClause) {
  uint32_t NThreadsICV =
      NumThreadsClause != -1 ? NumThreadsClause : icv::NThreads;
  uint32_t NumThreads = mapping::getBlockSize();

  if (NThreadsICV != 0 && NThreadsICV < NumThreads)
    NumThreads = NThreadsICV;

  // Round down to a multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < mapping::getWarpSize())
    NumThreads = 1;
  else
    NumThreads = (NumThreads & ~((uint32_t)mapping::getWarpSize() - 1));

  return NumThreads;
}

// Invoke an outlined parallel function unwrapping arguments (up to 32).
void invokeMicrotask(int32_t global_tid, int32_t bound_tid, void *fn,
                     void **args, int64_t nargs) {
  DebugEntryRAII Entry(__FILE__, __LINE__, "<OpenMP Outlined Function>");
  switch (nargs) {
#include "generated_microtask_cases.gen"
  default:
    PRINT("Too many arguments in kmp_invoke_microtask, aborting execution.\n");
    __builtin_trap();
  }
}

} // namespace

extern "C" {

void __kmpc_parallel_51(IdentTy *ident, int32_t, int32_t if_expr,
                        int32_t num_threads, int proc_bind, void *fn,
                        void *wrapper_fn, void **args, int64_t nargs) {
  FunctionTracingRAII();

  uint32_t TId = mapping::getThreadIdInBlock();
  // Handle the serialized case first, same for SPMD/non-SPMD.
  if (OMP_UNLIKELY(!if_expr || icv::Level)) {
    state::enterDataEnvironment();
    ++icv::Level;
    invokeMicrotask(TId, 0, fn, args, nargs);
    state::exitDataEnvironment();
    return;
  }

  uint32_t NumThreads = determineNumberOfThreads(num_threads);
  if (mapping::isSPMDMode()) {
    // Avoid the race between the read of the `icv::Level` above and the write
    // below by synchronizing all threads here.
    synchronize::threadsAligned();
    {
      // Note that the order here is important. `icv::Level` has to be updated
      // last or the other updates will cause a thread specific state to be
      // created.
      state::ValueRAII ParallelTeamSizeRAII(state::ParallelTeamSize, NumThreads,
                                            1u, TId == 0);
      state::ValueRAII ActiveLevelRAII(icv::ActiveLevel, 1u, 0u, TId == 0);
      state::ValueRAII LevelRAII(icv::Level, 1u, 0u, TId == 0);

      // Synchronize all threads after the main thread (TId == 0) set up the
      // team state properly.
      synchronize::threadsAligned();

      ASSERT(state::ParallelTeamSize == NumThreads);
      ASSERT(icv::ActiveLevel == 1u);
      ASSERT(icv::Level == 1u);

      if (TId < NumThreads)
        invokeMicrotask(TId, 0, fn, args, nargs);

      // Synchronize all threads at the end of a parallel region.
      synchronize::threadsAligned();
    }

    // Synchronize all threads to make sure every thread exits the scope above;
    // otherwise the following assertions and the assumption in
    // __kmpc_target_deinit may not hold.
    synchronize::threadsAligned();

    ASSERT(state::ParallelTeamSize == 1u);
    ASSERT(icv::ActiveLevel == 0u);
    ASSERT(icv::Level == 0u);
    return;
  }

  // We do *not* create a new data environment because all threads in the team
  // that are active are now running this parallel region. They share the
  // TeamState, which has an increase level-var and potentially active-level
  // set, but they do not have individual ThreadStates yet. If they ever
  // modify the ICVs beyond this point a ThreadStates will be allocated.

  bool IsActiveParallelRegion = NumThreads > 1;
  if (!IsActiveParallelRegion) {
    state::ValueRAII LevelRAII(icv::Level, 1u, 0u, true);
    invokeMicrotask(TId, 0, fn, args, nargs);
    return;
  }

  void **GlobalArgs = nullptr;
  if (nargs) {
    __kmpc_begin_sharing_variables(&GlobalArgs, nargs);
#pragma unroll
    for (int I = 0; I < nargs; I++)
      GlobalArgs[I] = args[I];
  }

  {
    // Note that the order here is important. `icv::Level` has to be updated
    // last or the other updates will cause a thread specific state to be
    // created.
    state::ValueRAII ParallelTeamSizeRAII(state::ParallelTeamSize, NumThreads,
                                          1u, true);
    state::ValueRAII ParallelRegionFnRAII(state::ParallelRegionFn, wrapper_fn,
                                          (void *)nullptr, true);
    state::ValueRAII ActiveLevelRAII(icv::ActiveLevel, 1u, 0u, true);
    state::ValueRAII LevelRAII(icv::Level, 1u, 0u, true);

    // Master signals work to activate workers.
    synchronize::threads();
    // Master waits for workers to signal.
    synchronize::threads();
  }

  if (nargs)
    __kmpc_end_sharing_variables();
}

__attribute__((noinline)) bool
__kmpc_kernel_parallel(ParallelRegionFnTy *WorkFn) {
  FunctionTracingRAII();
  // Work function and arguments for L1 parallel region.
  *WorkFn = state::ParallelRegionFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn)
    return false;

  // Set to true for workers participating in the parallel region.
  uint32_t TId = mapping::getThreadIdInBlock();
  bool ThreadIsActive = TId < state::ParallelTeamSize;
  return ThreadIsActive;
}

__attribute__((noinline)) void __kmpc_kernel_end_parallel() {
  FunctionTracingRAII();
  // In case we have modified an ICV for this thread before a ThreadState was
  // created. We drop it now to not contaminate the next parallel region.
  ASSERT(!mapping::isSPMDMode());
  uint32_t TId = mapping::getThreadIdInBlock();
  state::resetStateForThread(TId);
  ASSERT(!mapping::isSPMDMode());
}

uint16_t __kmpc_parallel_level(IdentTy *, uint32_t) {
  FunctionTracingRAII();
  return omp_get_level();
}

int32_t __kmpc_global_thread_num(IdentTy *) {
  FunctionTracingRAII();
  return omp_get_thread_num();
}

void __kmpc_push_num_threads(IdentTy *, int32_t, int32_t NumThreads) {
  FunctionTracingRAII();
  icv::NThreads = NumThreads;
}

void __kmpc_push_num_teams(IdentTy *loc, int32_t tid, int32_t num_teams,
                           int32_t thread_limit) {
  FunctionTracingRAII();
}

void __kmpc_push_proc_bind(IdentTy *loc, uint32_t tid, int proc_bind) {
  FunctionTracingRAII();
}
}

#pragma omp end declare target
