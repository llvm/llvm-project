//===------ State.cpp - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Shared/Environment.h"

#include "Allocator.h"
#include "Configuration.h"
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Mapping.h"
#include "Synchronization.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility(
      "protected")]] DeviceMemoryPoolTy __omp_rtl_device_memory_pool;
[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] DeviceMemoryPoolTrackingTy
    __omp_rtl_device_memory_pool_tracker;

/// Stateless bump allocator that uses the __omp_rtl_device_memory_pool
/// directly.
struct BumpAllocatorTy final {

  void *alloc(uint64_t Size) {
    Size = utils::roundUp(Size, uint64_t(allocator::ALIGNMENT));

    if (config::isDebugMode(DeviceDebugKind::AllocationTracker)) {
      atomic::add(&__omp_rtl_device_memory_pool_tracker.NumAllocations, 1,
                  atomic::seq_cst);
      atomic::add(&__omp_rtl_device_memory_pool_tracker.AllocationTotal, Size,
                  atomic::seq_cst);
      atomic::min(&__omp_rtl_device_memory_pool_tracker.AllocationMin, Size,
                  atomic::seq_cst);
      atomic::max(&__omp_rtl_device_memory_pool_tracker.AllocationMax, Size,
                  atomic::seq_cst);
    }

    uint64_t *Data =
        reinterpret_cast<uint64_t *>(&__omp_rtl_device_memory_pool.Ptr);
    uint64_t End =
        reinterpret_cast<uint64_t>(Data) + __omp_rtl_device_memory_pool.Size;

    uint64_t OldData = atomic::add(Data, Size, atomic::seq_cst);
    if (OldData + Size > End)
      __builtin_trap();

    return reinterpret_cast<void *>(OldData);
  }

  void free(void *) {}
};

BumpAllocatorTy BumpAllocator;

/// allocator namespace implementation
///
///{

void allocator::init(bool IsSPMD, KernelEnvironmentTy &KernelEnvironment) {
  // TODO: Check KernelEnvironment for an allocator choice as soon as we have
  // more than one.
}

void *allocator::alloc(uint64_t Size) { return BumpAllocator.alloc(Size); }

void allocator::free(void *Ptr) { BumpAllocator.free(Ptr); }

///}

#pragma omp end declare target
