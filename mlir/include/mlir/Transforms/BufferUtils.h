//===- BufferUtils.h - Buffer optimization utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for passes optimizing code that has already
// been converted to buffers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_BUFFERUTILS_H
#define MLIR_TRANSFORMS_BUFFERUTILS_H

#include "mlir/Analysis/BufferAliasAnalysis.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// A simple analysis that detects allocation operations.
class BufferPlacementAllocs {
public:
  /// Represents a tuple of allocValue and deallocOperation.
  using AllocEntry = std::tuple<Value, Operation *>;

  /// Represents a list containing all alloc entries.
  using AllocEntryList = SmallVector<AllocEntry, 8>;

  /// Get the start operation to place the given alloc value within the
  /// specified placement block.
  static Operation *getStartOperation(Value allocValue, Block *placementBlock,
                                      const Liveness &liveness);

  /// Find an associated dealloc operation that is linked to the given
  /// allocation node (if any).
  static Operation *findDealloc(Value allocValue);

public:
  /// Initializes the internal list by discovering all supported allocation
  /// nodes.
  BufferPlacementAllocs(Operation *op);

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::const_iterator begin() const { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::const_iterator end() const { return allocs.end(); }

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::iterator begin() { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::iterator end() { return allocs.end(); }

  /// Registers a new allocation entry.
  void registerAlloc(const AllocEntry &entry) { allocs.push_back(entry); }

private:
  /// Searches for and registers all supported allocation entries.
  void build(Operation *op);

private:
  /// Maps allocation nodes to their associated blocks.
  AllocEntryList allocs;
};

/// The base class for all BufferPlacement transformations.
class BufferPlacementTransformationBase {
public:
  using ValueSetT = BufferAliasAnalysis::ValueSetT;

  /// Finds a common dominator for the given value while taking the positions
  /// of the values in the value set into account. It supports dominator and
  /// post-dominator analyses via template arguments.
  template <typename DominatorT>
  static Block *findCommonDominator(Value value, const ValueSetT &values,
                                    const DominatorT &doms) {
    // Start with the current block the value is defined in.
    Block *dom = value.getParentBlock();
    // Iterate over all aliases and their uses to find a safe placement block
    // according to the given dominator information.
    for (Value childValue : values) {
      for (Operation *user : childValue.getUsers()) {
        // Move upwards in the dominator tree to find an appropriate
        // dominator block that takes the current use into account.
        dom = doms.findNearestCommonDominator(dom, user->getBlock());
      }
      // Take values without any users into account.
      dom = doms.findNearestCommonDominator(dom, childValue.getParentBlock());
    }
    return dom;
  }

  /// Returns true if the given operation represents a loop by testing whether
  /// it implements the `LoopLikeOpInterface` or the `RegionBranchOpInterface`.
  /// In the case of a `RegionBranchOpInterface`, it checks all region-based
  /// control-flow edges for cycles.
  static bool isLoop(Operation *op);

  /// Constructs a new operation base using the given root operation.
  BufferPlacementTransformationBase(Operation *op);

protected:
  /// Alias information that can be updated during the insertion of copies.
  BufferAliasAnalysis aliases;

  /// Stores all internally managed allocations.
  BufferPlacementAllocs allocs;

  /// The underlying liveness analysis to compute fine grained information
  /// about alloc and dealloc positions.
  Liveness liveness;
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERUTILS_H
