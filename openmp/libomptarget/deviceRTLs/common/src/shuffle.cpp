//===--- shuffle.cpp - Implementation of the external shuffle idiom API -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "target/shuffle.h"

#pragma omp declare target

static constexpr uint64_t AllLanes = -1;

int32_t __kmpc_shuffle_int32(int32_t val, int16_t delta, int16_t size) {
  return __kmpc_impl_shfl_down_sync(AllLanes, val, delta, size);
}

int64_t __kmpc_shuffle_int64(int64_t val, int16_t delta, int16_t size) {
  uint32_t lo, hi;
  __kmpc_impl_unpack(val, lo, hi);
  hi = __kmpc_impl_shfl_down_sync(AllLanes, hi, delta, size);
  lo = __kmpc_impl_shfl_down_sync(AllLanes, lo, delta, size);
  return __kmpc_impl_pack(lo, hi);
}

#pragma omp end declare target
