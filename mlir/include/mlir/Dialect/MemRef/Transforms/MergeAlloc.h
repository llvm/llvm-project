//===- MergeAlloc.h - The interfaces for merge alloc pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_MERGEALLOC_H
#define MLIR_DIALECT_MEMREF_MERGEALLOC_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
class BufferViewFlowAnalysis;
namespace memref {
struct MergeAllocOptions;
// abstract base class for lifetime of different buffers. It should hold the
// lifetime informantion of buffers that are to be merged in the same allocation
// in an "allocation scope". TraceCollectorFunc decides which buffers are put
// into which "allocation scope".
class LifetimeTrace {
public:
  enum TraceKind { TK_TICK };
  virtual ~LifetimeTrace() = default;
  LifetimeTrace(TraceKind kind) : kind{kind} {}
  TraceKind getKind() const { return kind; }

private:
  TraceKind kind;
};

// top level memory trace info for multiple scopes. Each key-value is the
// traces and location for buffers in the same "allocation scope"
struct MemoryTraces {
  llvm::DenseMap<Operation *, std::unique_ptr<LifetimeTrace>> scopeToTraces;
  MemoryTraces() = default;
};

// the memory scheduling result for allocations in the same merged buffer.
// allocation => offset map. All Operation* in the map should be memref::AllocOp
// which are in the same LifetimeTrace.
struct MemorySchedule {
  size_t totalSize;
  llvm::DenseMap<Operation *, int64_t> allocToOffset;
  MemorySchedule() : totalSize{0} {}
};

using TraceCollectorFunc = llvm::function_ref<FailureOr<MemoryTraces>(
    Operation *, const BufferViewFlowAnalysis &, const MergeAllocOptions &)>;
using MemoryPlannerFunc = llvm::function_ref<FailureOr<MemorySchedule>(
    Operation *, const LifetimeTrace &, const MergeAllocOptions &)>;
using MemoryMergeMutatorFunc = llvm::function_ref<LogicalResult(
    Operation *toplevel, Operation *scope, const MemorySchedule &,
    const MergeAllocOptions &)>;

FailureOr<MemoryTraces>
tickBasedCollectMemoryTrace(Operation *root,
                            const mlir::BufferViewFlowAnalysis &aliasAnaly,
                            const MergeAllocOptions &option);

FailureOr<MemorySchedule> tickBasedPlanMemory(Operation *op,
                                              const LifetimeTrace &tr,
                                              const MergeAllocOptions &o);
LogicalResult tickBasedMutateAllocations(Operation *op, Operation *scope,
                                         const MemorySchedule &schedule,
                                         const MergeAllocOptions &o);
} // namespace memref
} // namespace mlir

#endif
