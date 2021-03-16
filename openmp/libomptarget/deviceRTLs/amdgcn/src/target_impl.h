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

#define INLINE inline
#define NOINLINE __attribute__((noinline))
#define ALIGN(N) __attribute__((aligned(N)))

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The following def must match the absolute limit hardwired in the host RTL
// max number of threads per team
#define MAX_THREADS_PER_TEAM 1024

#define WARPSIZE 64

// Maximum number of preallocated arguments to an outlined parallel/simd
// function. Anything more requires dynamic memory allocation.
#define MAX_SHARED_ARGS 20

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#define OMP_STATE_COUNT 32
#define MAX_SM 64

#define OMP_ACTIVE_PARALLEL_LEVEL 128

// Data sharing related quantities, need to match what is used in the compiler.
enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 960,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 256,
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = WARPSIZE * DS_Slot_Size,
  // The maximum number of warps in use
  DS_Max_Warp_Number = 16,
};

enum : __kmpc_impl_lanemask_t {
  __kmpc_impl_all_lanes = ~(__kmpc_impl_lanemask_t)0
};

EXTERN int printf(const char *, ...);

#endif
