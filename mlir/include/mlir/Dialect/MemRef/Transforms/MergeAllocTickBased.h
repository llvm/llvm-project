//===- MergeAllocTickBased.h - Tick-based merge alloc interfaces *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_MERGEALLOCTICKBASED_H
#define MLIR_DIALECT_MEMREF_MERGEALLOCTICKBASED_H

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/StaticMemoryPlanning.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
class BufferViewFlowAnalysis;
class ViewLikeOpInterface;
namespace memref {

// Usually ticks should be non-negative numbers. There are two special ticks
// defined here.
namespace special_ticks {
// the memref is not accessed
static constexpr int64_t NO_ACCESS = -1;
// complex access happens on this memref
static constexpr int64_t COMPLEX_ACCESS = -2;
} // namespace special_ticks

// the collected tick [first, last] for a memref
struct Tick {
  // The tick when the buffer is allocated. allocTick is only used to stablize
  // the sorting results of the buffers when ticks of different buffers are the
  // same
  int64_t allocTick = special_ticks::NO_ACCESS;
  int64_t firstAccess = special_ticks::NO_ACCESS;
  int64_t lastAccess = special_ticks::NO_ACCESS;

  // access the memref at the tick, will update firstAccess and lastAccess based
  // on the tick
  void access(int64_t tick);
};

// A complex scope object is addition info for a RegionBranchOpInterface or
// LoopLikeOpInterface. It contains the scope itself, and the referenced alloc
// ops inside this scope. We use this object to track which buffers this scope
// accesses. These buffers must have overlapped lifetime
struct ComplexScope {
  Operation *scope;
  int64_t startTick;
  llvm::SmallPtrSet<Operation *, 8> operations;
  ComplexScope(Operation *scope, int64_t startTick)
      : scope{scope}, startTick{startTick} {}
};

// the top-level collected lifetime trace for merge-alloc pass
struct TickTraceResult : public LifetimeTrace {
  memoryplan::Traces traces;
  TickTraceResult() : LifetimeTrace{TK_TICK} {}
  static bool classof(const LifetimeTrace *S) {
    return S->getKind() == TK_TICK;
  }
};

// the internal states for TickCollecter analysis for a function
struct TickCollecterStates {
  // the alias analysis result for the function
  const mlir::BufferViewFlowAnalysis &aliasAnaly;
  const MergeAllocationOptions &opt;
  // the current tick for the current callback of walk(). It will be by default
  // incremented by 1 for each visited op
  int64_t curTick = 0;
  // the currently collected AllocOp -> [start, end] map
  llvm::DenseMap<Operation *, Tick> allocTicks;
  // the stack of ComplexScopes for the current visited position in the IR
  llvm::SmallVector<ComplexScope> complexScopeStack;
  TickCollecterStates(const mlir::BufferViewFlowAnalysis &aliasAnaly,
                      const MergeAllocationOptions &opt)
      : aliasAnaly{aliasAnaly}, opt{opt} {}
};

struct TickCollecter {
  TickCollecter() = default;
  virtual LogicalResult popScopeIfNecessary(TickCollecterStates *s,
                                            Operation *op) const;

  virtual void forwardTick(TickCollecterStates *s) const;

  virtual void accessValue(TickCollecterStates *s, Value v, bool complex) const;

  virtual void onMemrefViews(TickCollecterStates *s,
                             ViewLikeOpInterface op) const;

  virtual void onReturnOp(TickCollecterStates *s, Operation *op) const;

  virtual void onAllocOp(TickCollecterStates *s, Operation *op) const;

  virtual void onGeneralOp(TickCollecterStates *s, Operation *op) const;

  virtual void pushComplexScope(TickCollecterStates *s, Operation *op) const;

  // called when walk() runs outside of the scope
  LogicalResult onPopComplexScope(TickCollecterStates *s,
                                  int64_t endTick) const;

  // returns true of an allocation either is not defined in the scope, or the
  // allocation escapes from the scope
  bool needsResetTick(TickCollecterStates *s, Operation *scope,
                      Operation *allocation) const;

  virtual bool isMergeableAlloc(TickCollecterStates *s, Operation *op,
                                int64_t tick) const;

  // find the closest surrounding parent operation with AutomaticAllocationScope
  // trait, and is not scf.for
  virtual Operation *getAllocScope(TickCollecterStates *s, Operation *op) const;

  virtual FailureOr<size_t> getAllocSize(TickCollecterStates *s,
                                         Operation *op) const;

  virtual FailureOr<MemoryTraceScopes> getTrace(TickCollecterStates *s) const;

  virtual FailureOr<MemoryTraceScopes>
  operator()(Operation *root, const mlir::BufferViewFlowAnalysis &aliasAnaly,
             const MergeAllocationOptions &option) const;

  virtual ~TickCollecter() = default;
};

struct MergeAllocDefaultMutator {
  virtual Value buildAlloc(OpBuilder &build, Operation *scope, int64_t size,
                           int64_t alignment) const;
  virtual Value buildView(OpBuilder &build, Operation *scope,
                          Operation *origAllocOp, Value mergedAlloc,
                          int64_t byteOffset) const;
  virtual LogicalResult operator()(Operation *op, Operation *scope,
                                   const MemorySchedule &schedule,
                                   const MergeAllocationOptions &o) const;
  MergeAllocDefaultMutator() = default;
  virtual ~MergeAllocDefaultMutator() = default;
};

FailureOr<MemorySchedule> tickBasedPlanMemory(Operation *op,
                                              const LifetimeTrace &tr,
                                              const MergeAllocationOptions &o);

} // namespace memref
} // namespace mlir

#endif
