//===--------- Misc.cpp - OpenMP device misc interfaces ----------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "Types.h"

#include "Debug.h"

#pragma omp begin declare target device_type(nohost)

namespace ompx {
namespace impl {

double getWTick();

double getWTime();

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

double getWTick() {
  // The number of ticks per second for the AMDGPU clock varies by card and can
  // only be retrived by querying the driver. We rely on the device environment
  // to inform us what the proper frequency is.
  return 1.0 / config::getClockFrequency();
}

double getWTime() {
  uint64_t NumTicks = 0;
  if constexpr (__has_builtin(__builtin_amdgcn_s_sendmsg_rtnl))
    NumTicks = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
  else if constexpr (__has_builtin(__builtin_amdgcn_s_memrealtime))
    NumTicks = __builtin_amdgcn_s_memrealtime();
  else if constexpr (__has_builtin(__builtin_amdgcn_s_memtime))
    NumTicks = __builtin_amdgcn_s_memtime();

  return static_cast<double>(NumTicks) * getWTick();
}

#pragma omp end declare variant

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

double getWTick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

double getWTime() {
  unsigned long long nsecs;
  asm volatile("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return (double)nsecs * getWTick();
}

#pragma omp end declare variant

/// Lookup a device-side function using a host pointer /p HstPtr using the table
/// provided by the device plugin. The table is an ordered pair of host and
/// device pointers sorted on the value of the host pointer.
void *indirectCallLookup(void *HstPtr) {
  if (!HstPtr)
    return nullptr;

  struct IndirectCallTable {
    void *HstPtr;
    void *DevPtr;
  };
  IndirectCallTable *Table =
      reinterpret_cast<IndirectCallTable *>(config::getIndirectCallTablePtr());
  uint64_t TableSize = config::getIndirectCallTableSize();

  // If the table is empty we assume this is device pointer.
  if (!Table || !TableSize)
    return HstPtr;

  uint32_t Left = 0;
  uint32_t Right = TableSize;

  // If the pointer is definitely not contained in the table we exit early.
  if (HstPtr < Table[Left].HstPtr || HstPtr > Table[Right - 1].HstPtr)
    return HstPtr;

  while (Left != Right) {
    uint32_t Current = Left + (Right - Left) / 2;
    if (Table[Current].HstPtr == HstPtr)
      return Table[Current].DevPtr;

    if (HstPtr < Table[Current].HstPtr)
      Right = Current;
    else
      Left = Current;
  }

  // If we searched the whole table and found nothing this is a device pointer.
  return HstPtr;
}

} // namespace impl
} // namespace ompx

/// Interfaces
///
///{

extern "C" {
int32_t __kmpc_cancellationpoint(IdentTy *, int32_t, int32_t) { return 0; }

int32_t __kmpc_cancel(IdentTy *, int32_t, int32_t) { return 0; }

double omp_get_wtick(void) { return ompx::impl::getWTick(); }

double omp_get_wtime(void) { return ompx::impl::getWTime(); }

void *__llvm_omp_indirect_call_lookup(void *HstPtr) {
  return ompx::impl::indirectCallLookup(HstPtr);
}
}

///}
#pragma omp end declare target
