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

#include <stdint.h>

#include "option.h"

INLINE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
}

INLINE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  uint64_t val;
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

typedef uint32_t __kmpc_impl_lanemask_t;

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(res));
  return res;
}

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(res));
  return res;
}

INLINE uint32_t __kmpc_impl_ffs(uint32_t x) { return __ffs(x); }

INLINE uint32_t __kmpc_impl_popc(uint32_t x) { return __popc(x); }

#ifndef CUDA_VERSION
#error CUDA_VERSION macro is undefined, something wrong with cuda.
#endif

// In Cuda 9.0, __ballot(1) from Cuda 8.0 is replaced with __activemask().

INLINE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return __ballot(1);
#endif
}

// In Cuda 9.0, the *_sync() version takes an extra argument 'mask'.

INLINE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t Mask, int32_t Var,
                                     int32_t SrcLane) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(Mask, Var, SrcLane);
#else
  return __shfl(Var, SrcLane);
#endif // CUDA_VERSION
}

INLINE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t Mask,
                                          int32_t Var, uint32_t Delta,
                                          int32_t Width) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(Mask, Var, Delta, Width);
#else
  return __shfl_down(Var, Delta, Width);
#endif // CUDA_VERSION
}

INLINE void __kmpc_impl_syncthreads() {
  // Use original __syncthreads if compiled by nvcc or clang >= 9.0.
#if !defined(__clang__) || __clang_major__ >= 9
  __syncthreads();
#else
  asm volatile("bar.sync %0;" : : "r"(0) : "memory");
#endif // __clang__
}

INLINE void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask) {
#if CUDA_VERSION >= 9000
  __syncwarp(Mask);
#else
  // In Cuda < 9.0 no need to sync threads in warps.
#endif // CUDA_VERSION
}

#endif
