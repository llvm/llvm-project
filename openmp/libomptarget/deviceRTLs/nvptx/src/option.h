//===------------ option.h - NVPTX OpenMP GPU options ------------ CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// GPU default options
//
//===----------------------------------------------------------------------===//
#ifndef _OPTION_H_
#define _OPTION_H_

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The following def must match the absolute limit hardwired in the host RTL
// max number of threads per team
#define MAX_THREADS_PER_TEAM 1024

#define WARPSIZE 32

// The named barrier for active parallel threads of a team in an L1 parallel
// region to synchronize with each other.
#define L1_BARRIER (1)

// Maximum number of preallocated arguments to an outlined parallel/simd function.
// Anything more requires dynamic memory allocation.
#define MAX_SHARED_ARGS 20

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#if __CUDA_ARCH__ >= 600
#define OMP_STATE_COUNT 32
#define MAX_SM 56
#else
#define OMP_STATE_COUNT 16
#define MAX_SM 16
#endif

////////////////////////////////////////////////////////////////////////////////
// algo options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// data options
////////////////////////////////////////////////////////////////////////////////

// decide if counters are 32 or 64 bit
#define Counter unsigned long long

////////////////////////////////////////////////////////////////////////////////
// misc options (by def everythig here is device)
////////////////////////////////////////////////////////////////////////////////

#define EXTERN extern "C" __device__
#define INLINE __inline__ __device__
#define NOINLINE __noinline__ __device__
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#endif
