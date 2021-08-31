//===------- target_impl.h - AMDGCN OpenMP GPU implementation ----- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations and definitions of target specific functions and constants
//
//===----------------------------------------------------------------------===//
#ifndef OMPTARGET_AMDGCN_TARGET_IMPL_H
#define OMPTARGET_AMDGCN_TARGET_IMPL_H

#ifndef __AMDGCN__
#error "amdgcn target_impl.h expects to be compiled under __AMDGCN__"
#endif

#include "amdgcn_interface.h"

#include <stddef.h>
#include <stdint.h>

// subset of inttypes.h
#define PRId64 "ld"
#define PRIu64 "lu"

typedef uint64_t __kmpc_impl_lanemask_t;

#define INLINE inline
#define NOINLINE __attribute__((noinline))
#define ALIGN(N) __attribute__((aligned(N)))
#define PLUGIN_ACCESSIBLE                                                      \
  __attribute__((used)) /* Don't discard values the plugin reads */            \
  __attribute__((visibility("default"))) /* Access via SHT_HASH */             \
  __attribute__((section(".data")))      /* Not .bss, can write before load */

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

INLINE constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>();
}

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The following def must match the absolute limit hardwired in the host RTL
// max number of threads per team
enum { MAX_THREADS_PER_TEAM = getGridValue().GV_Max_WG_Size };
enum { WARPSIZE = getGridValue().GV_Warp_Size };

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#define OMP_STATE_COUNT 32
#define MAX_SM 64

#define OMP_ACTIVE_PARALLEL_LEVEL 128

// Data sharing related quantities, need to match what is used in the compiler.
enum DATA_SHARING_SIZES {
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = getGridValue().GV_Slot_Size,
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = getGridValue().warpSlotSize(),
  // The maximum number of warps in use
  DS_Max_Warp_Number = getGridValue().maxWarpNumber(),
};

enum : __kmpc_impl_lanemask_t {
  __kmpc_impl_all_lanes = ~(__kmpc_impl_lanemask_t)0
};

// The return code of printf is not checked in the call sites in this library.
// A call to a function named printf currently hits some special case handling
// for opencl, which translates to calls that do not presently exist for openmp
// Therefore, for now, stub out printf while building this library.
#define printf(...)

#endif
