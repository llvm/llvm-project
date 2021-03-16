//===------------ target_impl.h - NVPTX OpenMP GPU options ------- CUDA -*-===//
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
#ifndef _TARGET_IMPL_H_
#define _TARGET_IMPL_H_

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "nvptx_interface.h"

#define INLINE inline __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))
#define ALIGN(N) __attribute__((aligned(N)))

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The following def must match the absolute limit hardwired in the host RTL
// max number of threads per team
#define MAX_THREADS_PER_TEAM 1024

#define WARPSIZE 32

// Maximum number of preallocated arguments to an outlined parallel/simd
// function. Anything more requires dynamic memory allocation.
#define MAX_SHARED_ARGS 20

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#if __CUDA_ARCH__ >= 600
#define OMP_STATE_COUNT 32
#else
#define OMP_STATE_COUNT 16
#endif

#if !defined(MAX_SM)
#if __CUDA_ARCH__ >= 900
#error unsupported compute capability, define MAX_SM via LIBOMPTARGET_NVPTX_MAX_SM cmake option
#elif __CUDA_ARCH__ >= 800
// GA100 design has a maxinum of 128 SMs but A100 product only has 108 SMs
// GA102 design has a maxinum of 84 SMs
#define MAX_SM 108
#elif __CUDA_ARCH__ >= 700
#define MAX_SM 84
#elif __CUDA_ARCH__ >= 600
#define MAX_SM 56
#else
#define MAX_SM 16
#endif
#endif

#define OMP_ACTIVE_PARALLEL_LEVEL 128

// Data sharing related quantities, need to match what is used in the compiler.
enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 992,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 256,
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = WARPSIZE * DS_Slot_Size,
  // The maximum number of warps in use
  DS_Max_Warp_Number = 32,
  // The size of the preallocated shared memory buffer per team
  DS_Shared_Memory_Size = 128,
};

enum : __kmpc_impl_lanemask_t {
  __kmpc_impl_all_lanes = ~(__kmpc_impl_lanemask_t)0
};

#endif
