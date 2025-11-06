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
using namespace allocator;

// Provide a default implementation of malloc / free for AMDGPU platforms built
// without 'libc' support.
extern "C" {
#if defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
[[gnu::weak]] void *malloc(size_t Size) { return allocator::alloc(Size); }
[[gnu::weak]] void free(void *Ptr) { allocator::free(Ptr); }
#else
[[gnu::leaf]] void *malloc(size_t Size);
[[gnu::leaf]] void free(void *Ptr);
#endif
}

static constexpr uint64_t MEMORY_SIZE = /* 1 MiB */ 1024 * 1024;
alignas(ALIGNMENT) static uint8_t Memory[MEMORY_SIZE] = {0};

// Fallback bump pointer interface for platforms without a functioning
// allocator.
struct BumpAllocatorTy final {
  uint64_t Offset = 0;

  void *alloc(uint64_t Size) {
    Size = utils::roundUp(Size, uint64_t(allocator::ALIGNMENT));

    uint64_t OldData = atomic::add(&Offset, Size, atomic::seq_cst);
    if (OldData + Size >= MEMORY_SIZE)
      __builtin_trap();

    return &Memory[OldData];
  }

  void free(void *) {}
};

BumpAllocatorTy BumpAllocator;

/// allocator namespace implementation
///
///{

void *allocator::alloc(uint64_t Size) {
#if defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
  return BumpAllocator.alloc(Size);
#else
  return ::malloc(Size);
#endif
}

void allocator::free(void *Ptr) {
#if defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
  BumpAllocator.free(Ptr);
#else
  ::free(Ptr);
#endif
}

///}
