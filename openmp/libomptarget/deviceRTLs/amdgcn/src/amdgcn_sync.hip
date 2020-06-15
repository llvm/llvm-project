//===--- amdgcn_sync.hip - GPU OpenMP synchronization extensions -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements amdgcn extensions to sync
//
//===----------------------------------------------------------------------===//

#include "common/omptarget.h"
#include "target_impl.h"

///////////////////////////////////////////////////////////////////////////////
/// The following five functions are used to coordinate barriers between
/// the master and worker warps in a generic kernel.  The amdgcn architecture
/// does not have a partial barrier so this coordination needs to be
/// recover if a user or implicit barrier is encountered in the sequential
/// part of the master warp or the parallel part of the worker warps.
///
/// __kmpc_amd_worker_start has a barrier to prevent worker warps
/// from starting till the master warp sets the workFn.  This function sets
/// workers_active to true after the barrier to keep the master warp
/// at its barrier in case a worker encounters an explicit or implicit
/// barrier.
EXTERN void __kmpc_amd_worker_start(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_worker_start\n");
  omptarget_workers_active = false;
  __kmpc_impl_syncthreads();
  while(omptarget_master_active) __kmpc_impl_syncthreads();
  omptarget_workers_active = true;
  PRINT0(LD_SYNC, "completed kmpc_amd_worker_start\n");
}

/// __kmpc_amd_worker_end sets workers_active to false and then
/// issues a barrier to release the master warp to terminate or get
/// the next subregion of work to process.
EXTERN void __kmpc_amd_worker_end(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_worker_end\n");
  omptarget_workers_active = false;
  __kmpc_impl_syncthreads(); // to sync with 2nd barrier in master_end
  PRINT0(LD_SYNC, "completed kmpc_amd_worker_end\n");
}

/// __kmpc_amd_master_start is executed first by the master.  It sets
/// master_active to true to keep the worker warps at its first barrier
/// in case the sequential part of the target region encounters any
/// barrier, explicit or otherwise.
EXTERN void __kmpc_amd_master_start(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_master_start\n");
  omptarget_master_active = true;
  PRINT0(LD_SYNC, "completed kmpc_amd_master_start\n");
}

/// __kmpc_amd_master_end sets master_active to false and then enters
/// the double barrier. The first releases the worker warps. The 2nd
/// barrier holds the master warp until the workers are done. If the worker
/// warps encounter a user barrier (implicitly or explicitly), the master
/// warp needs to loop at the barrier until it knows the worker is
/// really done by testing omptarget_workers_active.
EXTERN void __kmpc_amd_master_end(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_master_end\n");
  omptarget_master_active = false;
  __kmpc_impl_syncthreads();
  __kmpc_impl_syncthreads();
  while(omptarget_workers_active) __kmpc_impl_syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_amd_master_end\n");
}

EXTERN void __kmpc_amd_master_terminate(kmp_Ident *loc_ref, int32_t tid) {
  PRINT0(LD_SYNC, "call kmpc_amd_master_terminate\n");
  omptarget_master_active = false;
  __kmpc_impl_syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_amd_master_terminate\n");
}

