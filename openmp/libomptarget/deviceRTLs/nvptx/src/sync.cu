//===------------ sync.h - NVPTX OpenMP synchronizations --------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include all synchronization.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// KMP Ordered calls
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_ordered(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_ordered\n");
}

EXTERN void __kmpc_end_ordered(kmp_Ident *loc, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_end_ordered\n");
}

////////////////////////////////////////////////////////////////////////////////
// KMP Barriers
////////////////////////////////////////////////////////////////////////////////

// a team is a block: we can use CUDA native synchronization mechanism
// FIXME: what if not all threads (warps) participate to the barrier?
// We may need to implement it differently

EXTERN int32_t __kmpc_cancel_barrier(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_IO, "call kmpc_cancel_barrier\n");
  __kmpc_barrier(loc_ref, tid);
  PRINT0(LD_SYNC, "completed kmpc_cancel_barrier\n");
  return 0;
}

EXTERN void __kmpc_barrier(kmp_Ident *loc_ref, int32_t tid) {
  if (checkRuntimeUninitialized(loc_ref)) {
    ASSERT0(LT_FUSSY, checkSPMDMode(loc_ref),
            "Expected SPMD mode with uninitialized runtime.");
    __kmpc_barrier_simple_spmd(loc_ref, tid);
  } else {
    tid = GetLogicalThreadIdInBlock(checkSPMDMode(loc_ref));
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(tid);
    int numberOfActiveOMPThreads = GetNumberOfOmpThreads(
        tid, checkSPMDMode(loc_ref), /*isRuntimeUninitialized=*/false);
    if (numberOfActiveOMPThreads > 1) {
      if (checkSPMDMode(loc_ref)) {
        __kmpc_barrier_simple_spmd(loc_ref, tid);
      } else {
        // The #threads parameter must be rounded up to the WARPSIZE.
        int threads =
            WARPSIZE * ((numberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

        PRINT(LD_SYNC,
              "call kmpc_barrier with %d omp threads, sync parameter %d\n",
              (int)numberOfActiveOMPThreads, (int)threads);
        // Barrier #1 is for synchronization among active threads.
        named_sync(L1_BARRIER, threads);
      }
    } // numberOfActiveOMPThreads > 1
    PRINT0(LD_SYNC, "completed kmpc_barrier\n");
  }
}

// Emit a simple barrier call in SPMD mode.  Assumes the caller is in an L0
// parallel region and that all worker threads participate.
EXTERN void __kmpc_barrier_simple_spmd(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_barrier_simple_spmd\n");
  // FIXME: use __syncthreads instead when the function copy is fixed in LLVM.
  __SYNCTHREADS();
  PRINT0(LD_SYNC, "completed kmpc_barrier_simple_spmd\n");
}

// Emit a simple barrier call in Generic mode.  Assumes the caller is in an L0
// parallel region and that all worker threads participate.
EXTERN void __kmpc_barrier_simple_generic(kmp_Ident *loc_ref, int32_t tid) {
  int numberOfActiveOMPThreads = GetNumberOfThreadsInBlock() - WARPSIZE;
  // The #threads parameter must be rounded up to the WARPSIZE.
  int threads =
      WARPSIZE * ((numberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

  PRINT(LD_SYNC,
        "call kmpc_barrier_simple_generic with %d omp threads, sync parameter "
        "%d\n",
        (int)numberOfActiveOMPThreads, (int)threads);
  // Barrier #1 is for synchronization among active threads.
  named_sync(L1_BARRIER, threads);
  PRINT0(LD_SYNC, "completed kmpc_barrier_simple_generic\n");
}

////////////////////////////////////////////////////////////////////////////////
// KMP MASTER
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_master(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_master\n");
  return IsTeamMaster(global_tid);
}

EXTERN void __kmpc_end_master(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_end_master\n");
  ASSERT0(LT_FUSSY, IsTeamMaster(global_tid), "expected only master here");
}

////////////////////////////////////////////////////////////////////////////////
// KMP SINGLE
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_single(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_single\n");
  // decide to implement single with master; master get the single
  return IsTeamMaster(global_tid);
}

EXTERN void __kmpc_end_single(kmp_Ident *loc, int32_t global_tid) {
  PRINT0(LD_IO, "call kmpc_end_single\n");
  // decide to implement single with master: master get the single
  ASSERT0(LT_FUSSY, IsTeamMaster(global_tid), "expected only master here");
  // sync barrier is explicitely called... so that is not a problem
}

////////////////////////////////////////////////////////////////////////////////
// Flush
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_flush(kmp_Ident *loc) {
  PRINT0(LD_IO, "call kmpc_flush\n");
  __threadfence_system();
}

////////////////////////////////////////////////////////////////////////////////
// Vote
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_warp_active_thread_mask() {
  PRINT0(LD_IO, "call __kmpc_warp_active_thread_mask\n");
  return __ACTIVEMASK();
}
