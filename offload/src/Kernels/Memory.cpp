//===-- Kenrels/Memory.cpp - Memory related kernel definitions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#define LAUNCH_BOUNDS(MIN, MAX)                                                \
  __attribute__((launch_bounds(MAX), amdgpu_flat_work_group_size(MIN, MAX)))
#define INLINE [[clang::always_inline]] inline
#define KERNEL [[gnu::weak]] __global__
#define DEVICE __device__

extern "C" {
DEVICE int ompx_thread_id(int Dim);
DEVICE int ompx_block_id(int Dim);
DEVICE int ompx_block_dim(int Dim);
DEVICE int ompx_grid_dim(int Dim);
}

namespace {
INLINE
DEVICE void __memset_impl(char *Ptr, int ByteVal, size_t NumBytes) {
  int TId = ompx_thread_id(0);
  int BId = ompx_block_id(0);
  int BDim = ompx_block_dim(0);
  size_t GId = BId * BDim + TId;
  if (GId < NumBytes)
    Ptr[GId] = ByteVal;
}
} // namespace

extern "C" {
KERNEL void LAUNCH_BOUNDS(1, 256)
    __memset(char *Ptr, int ByteVal, size_t NumBytes) {
  __memset_impl(Ptr, ByteVal, NumBytes);
}

KERNEL void LAUNCH_BOUNDS(1, 256)
    __memset_zero(char *Ptr, int ByteVal, size_t NumBytes) {
  __memset_impl(Ptr, 0, NumBytes);
}

KERNEL void LAUNCH_BOUNDS(1, 256)
    __memset_ones(char *Ptr, int ByteVal, size_t NumBytes) {
  __memset_impl(Ptr, ~0, NumBytes);
}
}
