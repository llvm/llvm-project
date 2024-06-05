//===- MergeAlloc.h - The interfaces for merge alloc pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_MERGEALLOC_H
#define MLIR_DIALECT_MEMREF_MERGEALLOC_H

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
class BufferViewFlowAnalysis;
namespace memref {
FailureOr<MemoryTraceScopes>
tickBasedCollectMemoryTrace(Operation *root,
                            const mlir::BufferViewFlowAnalysis &aliasAnaly,
                            const MergeAllocationOptions &option);

FailureOr<MemorySchedule> tickBasedPlanMemory(Operation *op,
                                              const LifetimeTrace &tr,
                                              const MergeAllocationOptions &o);
LogicalResult tickBasedMutateAllocations(Operation *op, Operation *scope,
                                         const MemorySchedule &schedule,
                                         const MergeAllocationOptions &o);
} // namespace memref
} // namespace mlir

#endif
