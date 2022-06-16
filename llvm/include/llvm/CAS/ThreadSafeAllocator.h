//===- ThreadSafeAllocator.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_THREADSAFEALLOCATOR_H
#define LLVM_CAS_THREADSAFEALLOCATOR_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Allocator.h"
#include <atomic>

namespace llvm {
namespace cas {

/// Thread-safe allocator adaptor. Uses an unfair lock on the assumption that
/// contention here is extremely rare.
///
/// TODO: Using an unfair lock on every allocation can be quite expensive when
/// contention is high. Since this is mainly used for BumpPtrAllocator and
/// SpecificBumpPtrAllocator, it'd be better to have a specific thread-safe
/// BumpPtrAllocator implementation that only use a fair lock when allocating a
/// new stab but otherwise using atomic and be lock-free.
template <class AllocatorType> class ThreadSafeAllocator {
  struct LockGuard {
    LockGuard(std::atomic_flag &Flag) : Flag(Flag) {
      if (LLVM_UNLIKELY(Flag.test_and_set(std::memory_order_acquire)))
        while (Flag.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~LockGuard() { Flag.clear(std::memory_order_release); }
    std::atomic_flag &Flag;
  };

public:
  auto Allocate(size_t N = 1) {
    LockGuard Lock(Flag);
    return Alloc.Allocate(N);
  }

  auto Allocate(size_t Size, size_t Align) {
    LockGuard Lock(Flag);
    return Alloc.Allocate(Size, Align);
  }

  void applyLocked(llvm::function_ref<void(AllocatorType &Alloc)> Fn) {
    LockGuard Lock(Flag);
    Fn(Alloc);
  }

private:
  AllocatorType Alloc;
  std::atomic_flag Flag = ATOMIC_FLAG_INIT;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_THREADSAFEALLOCATOR_H
