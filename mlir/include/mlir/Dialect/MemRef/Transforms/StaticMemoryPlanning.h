//===- StaticMemoryPlanning.h - Static memory planning ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_STATICMEMORYPLANNING_H
#define MLIR_DIALECT_MEMREF_STATICMEMORYPLANNING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <stdint.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir {
class Operation;

namespace memoryplan {
enum class InplaceKind {
  ZERO_OFFSET, // this requires that the tensor share the same base
               // pointer of the replaced tensor
  FREE,        // the tensor can freely choose any offset on this tensor
};

struct MemoryTrace {
  // unique id of a buffer
  uintptr_t bufferId;
  // if > 0, size of the buffer allocation, if = 0, it is a deallocation trace
  std::size_t size;
  MemoryTrace(uintptr_t bufferId = 0, std::size_t size = 0)
      : bufferId{bufferId}, size{size} {}
};

using Traces = llvm::SmallVector<memoryplan::MemoryTrace, 8>;
using InplaceInfo = std::pair<uintptr_t, InplaceKind>;

using InplaceInfoMap =
    llvm::DenseMap<uintptr_t, llvm::SmallVector<InplaceInfo>>;

/**
 * Given a list of memory buffer alloc and free traces, try to use a large
 * buffer to hold all allocated memory, and statically allocate each memory
 * buffer from the large buffer for better memory reuse.
 * @param traces the list of memory alloc and free traces, sorted by event time.
 * @param alignment the alignment in number of elements
 * @param hotFirst use the hot buffer first, instead of using best fit in size
 * @param inplaceMap the map from the tensor to alloc into the candidate
 * tensors that can be inplace reused for it.
 * @param outSchedule the output schedule for each buffer: the location that
 * the buffer should be in the large buffer (as an offset in number of elements)
 * @param outInplaceSelection the output buffer id -> inplace buffer it reuses
 * @return the size of the large buffer, in number of elements
 * */
std::size_t scheduleMemoryAllocations(
    const Traces &traces, std::size_t alignment, bool hotFirst,
    const InplaceInfoMap &inplaceMap,
    std::unordered_map<uintptr_t, std::size_t> &outSchedule,
    std::unordered_map<uintptr_t, std::vector<uintptr_t>> &outInplaceSelection);

} // namespace memoryplan
} // namespace mlir

#endif
