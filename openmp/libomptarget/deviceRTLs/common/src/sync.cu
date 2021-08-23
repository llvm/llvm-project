//===------------ sync.cu - GPU OpenMP synchronizations ---------- CUDA -*-===//
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
#pragma omp declare target

#include "common/omptarget.h"
#include "target_impl.h"

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
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, __kmpc_is_spmd_exec_mode(),
            "Expected SPMD mode with uninitialized runtime.");
    __kmpc_barrier_simple_spmd(loc_ref, tid);
  } else {
    tid = GetLogicalThreadIdInBlock();
    int numberOfActiveOMPThreads =
        GetNumberOfOmpThreads(__kmpc_is_spmd_exec_mode());
    if (numberOfActiveOMPThreads > 1) {
      if (__kmpc_is_spmd_exec_mode()) {
        __kmpc_barrier_simple_spmd(loc_ref, tid);
      } else {
        // The #threads parameter must be rounded up to the WARPSIZE.
        int threads =
            WARPSIZE * ((numberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

        PRINT(LD_SYNC,
              "call kmpc_barrier with %d omp threads, sync parameter %d\n",
              (int)numberOfActiveOMPThreads, (int)threads);
        __kmpc_impl_named_sync(threads);
      }
    } else {
      // Still need to flush the memory per the standard.
      __kmpc_flush(loc_ref);
    } // numberOfActiveOMPThreads > 1
    PRINT0(LD_SYNC, "completed kmpc_barrier\n");
  }
}

// Emit a simple barrier call in SPMD mode.  Assumes the caller is in an L0
// parallel region and that all worker threads participate.
EXTERN void __kmpc_barrier_simple_spmd(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_barrier_simple_spmd\n");
  __kmpc_impl_syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_barrier_simple_spmd\n");
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
  // sync barrier is explicitly called... so that is not a problem
}

////////////////////////////////////////////////////////////////////////////////
// Flush
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_flush(kmp_Ident *loc) {
  PRINT0(LD_IO, "call kmpc_flush\n");
  __kmpc_impl_threadfence();
}

////////////////////////////////////////////////////////////////////////////////
// Vote
////////////////////////////////////////////////////////////////////////////////

EXTERN uint64_t __kmpc_warp_active_thread_mask(void) {
  PRINT0(LD_IO, "call __kmpc_warp_active_thread_mask\n");
  return __kmpc_impl_activemask();
}

////////////////////////////////////////////////////////////////////////////////
// Syncwarp
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_syncwarp(uint64_t Mask) {
  PRINT0(LD_IO, "call __kmpc_syncwarp\n");
  __kmpc_impl_syncwarp(Mask);
}

#pragma omp end declare target
