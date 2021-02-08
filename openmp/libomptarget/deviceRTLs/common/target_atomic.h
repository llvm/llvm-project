//===---- target_atomic.h - OpenMP GPU target atomic functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations of atomic functions provided by each target
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_TARGET_ATOMIC_H
#define OMPTARGET_TARGET_ATOMIC_H

#include "target_impl.h"

template <typename T> INLINE T __kmpc_atomic_add(T *address, T val) {
#if defined(__AMDGCN__) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320 ) || !defined(__clang__)
  return atomicAdd(address, val);
#else
  //  Hack for k4000 without a working  64bit atomic add
  uint newval = (uint) val;
  uint* uintaddr = (uint *) address; // take little end for 32 bit atomic op
  return (T) atomicAdd(uintaddr, newval);
#endif
}

template <typename T> INLINE T __kmpc_atomic_inc(T *address, T val) {
  return atomicInc(address, val);
}

template <typename T> INLINE T __kmpc_atomic_max(T *address, T val) {
  return atomicMax(address, val);
}

template <typename T> INLINE T __kmpc_atomic_exchange(T *address, T val) {
  return atomicExch(address, val);
}

template <typename T> INLINE T __kmpc_atomic_cas(T *address, T compare, T val) {
  return atomicCAS(address, compare, val);
}

#endif
