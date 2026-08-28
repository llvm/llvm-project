//===--- Allocator.cpp - Simple memory allocation abstraction -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BumpPtrAllocator interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

namespace llvm {

namespace detail {

void printBumpPtrAllocatorStats(unsigned NumSlabs, size_t TotalMemory) {
  errs() << "\nNumber of memory regions: " << NumSlabs << '\n'
         << "Bytes allocated: " << TotalMemory << '\n'
         << " (includes alignment, etc)\n";
}

} // namespace detail

void PrintRecyclerStats(size_t Size,
                        size_t Align,
                        size_t FreeListSize) {
  errs() << "Recycler element size: " << Size << '\n'
         << "Recycler element alignment: " << Align << '\n'
         << "Number of elements free for recycling: " << FreeListSize << '\n';
}

namespace parallel::detail {
unsigned claimPerThreadAllocatorId() {
  static std::atomic<unsigned> Counter;
  return Counter.fetch_add(1, std::memory_order_relaxed);
}
} // namespace parallel::detail

} // namespace llvm
