//===- StaticMemoryPlanning.h - Memory planning algorithms ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure memory planning algorithms for static arena allocation. These operate
// on abstract allocation descriptors (size, alignment, lifetime) and produce
// byte offsets. They are independent of MLIR IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_STATICMEMORYPLANNING_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_STATICMEMORYPLANNING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
namespace bufferization {

/// Descriptor for a single allocation to be placed by the memory planner.
struct MemoryPlannerAlloc {
  int64_t sizeInBytes = 0;
  int64_t alignment = 1;
  int64_t timeStart = 0; // Operation index when allocation becomes live
  int64_t timeEnd = 0;   // Operation index when allocation is freed
};

/// Sequential packing without lifetime overlap. Each allocation is placed
/// immediately after the previous one (with alignment padding). Ignores
/// lifetimes entirely.
llvm::SmallVector<int64_t>
trivialMemoryPlanner(int64_t arenaAlignment,
                     llvm::ArrayRef<MemoryPlannerAlloc> allocs);

/// Best-fit packing with lifetime-aware gap reuse. Processes allocations in
/// time order and places each one in the smallest gap left by expired
/// allocations. Falls back to extending the arena if no gap fits.
llvm::SmallVector<int64_t>
bestFitMemoryPlanner(int64_t arenaAlignment,
                     llvm::ArrayRef<MemoryPlannerAlloc> allocs);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_STATICMEMORYPLANNING_H
