//===--- amdgcn_libcall.cu - extensions to OpenMP GPU user calls -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP runtime functions that can be invoked by
// the user in an OpenMP region that are not yet available for other targets
//
//===----------------------------------------------------------------------===//

#include "common/omptarget.h"
#include "common/target_atomic.h"
#include "target_impl.h"

EXTERN int omp_ext_get_warp_id() {
  int rc = GetWarpId();
  PRINT(LD_IO, "call omp_ext_get_warp_id() returns %d\n", rc);
  return rc;
}

EXTERN int omp_ext_get_lane_id() {
  int rc = GetLaneId();
  PRINT(LD_IO, "call omp_ext_get_lane_id() returns %d\n", rc);
  return rc;
}

DEVICE static unsigned getMasterThreadId() {
  unsigned Mask = WARPSIZE - 1;
  return (GetNumberOfThreadsInBlock() - 1) & (~Mask);
}

EXTERN int omp_ext_get_smid() {
  int rc = __kmpc_impl_smid();
  PRINT(LD_IO, "call omp_ext_get_smid() returns %d\n", rc);
  return rc;
}

EXTERN int omp_ext_is_spmd_mode() {
  int rc = isSPMDMode();
  PRINT(LD_IO, "call omp_ext_is_spmd_mode() returns %d\n", rc);
  return rc;
}

EXTERN int omp_ext_get_master_thread_id() {
  int rc = getMasterThreadId();
  PRINT(LD_IO, "call omp_ext_get_master_thread_id() returns %d\n", rc);
  return rc;
}

#ifdef __AMDGCN__
EXTERN unsigned long long omp_ext_get_active_threads_mask() {
  unsigned long long rc = __kmpc_impl_activemask();
  PRINT(LD_IO, "call omp_ext_get_active_threads_mask() returns %llx\n", rc);
  return rc;
}
#else
EXTERN unsigned long long omp_ext_get_active_threads_mask() {
  unsigned rc = __kmpc_impl_activemask();
  PRINT(LD_IO, "call omp_ext_get_active_threads_mask() returns %x\n", rc);
  return (unsigned long long)rc;
}
#endif

