//===------- target_impl.hip - AMDGCN OpenMP GPU implementation --- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions
//
//===----------------------------------------------------------------------===//

#include "target_impl.h"

DEVICE double __kmpc_impl_get_wtick() { return ((double)1E-9); }

EXTERN uint64_t __clock64();
DEVICE double __kmpc_impl_get_wtime() {
  return ((double)1.0 / 745000000.0) * __clock64();
}

EXTERN uint64_t __ockl_get_local_size(uint32_t);
EXTERN uint64_t __ockl_get_num_groups(uint32_t);
DEVICE int GetNumberOfBlocksInKernel() { return __ockl_get_num_groups(0); }
DEVICE int GetNumberOfThreadsInBlock() { return __ockl_get_local_size(0); }
