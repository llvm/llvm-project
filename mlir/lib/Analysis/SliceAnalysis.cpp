//===- UseDefAnalysis.cpp - Analysis for Transitive UseDef chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Analysis functions specific to slicing in Function.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

///
/// Implements Analysis functions specific to slicing in Function.
///

using namespace mlir;

using llvm::SetVector;

static void getForwardSliceImpl(Operation *op,
                                SetVector<Operation *> *forwardSlice,
                                TransitiveFilter filter) {
  if (!op)
    return;

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (filter && !filter(op))
    return;

  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &blockOp : block)
        if (forwardSlice->count(&blockOp) == 0)
          getForwardSliceImpl(&blockOp, forwardSlice, filter);
  for (Value result : op->getResults()) {
    for (Operation *userOp : result.getUsers())
      if (forwardSlice->count(userOp) == 0)
        getForwardSliceImpl(userOp, forwardSlice, filter);
  }

  forwardSlice->insert(op);
}

void mlir::getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                           TransitiveFilter filter) {
  getForwardSliceImpl(op, forwardSlice, filter);
  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  forwardSlice->remove(op);

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  std::vector<Operation *> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

void mlir::getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                           TransitiveFilter filter) {
  for (Operation *user : root.getUsers())
    getForwardSliceImpl(user, forwardSlice, filter);

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  std::vector<Operation *> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

static void getBackwardSliceImpl(Operation *op,
                                 SetVector<Operation *> *backwardSlice,
                                 TransitiveFilter filter) {
  if (!op || op->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return;

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive backwardSlice in the current scope.
  if (filter && !filter(op))
    return;

  for (auto en : llvm::enumerate(op->getOperands())) {
    auto operand = en.value();
    if (auto *definingOp = operand.getDefiningOp()) {
      if (backwardSlice->count(definingOp) == 0)
        getBackwardSliceImpl(definingOp, backwardSlice, filter);
    } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      // TODO: determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they flow
      // into us. For now, just bail.
      assert(parentOp->getNumRegions() == 1 &&
             parentOp->getRegion(0).getBlocks().size() == 1);
      if (backwardSlice->count(parentOp) == 0)
        getBackwardSliceImpl(parentOp, backwardSlice, filter);
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }
  }

  backwardSlice->insert(op);
}

void mlir::getBackwardSlice(Operation *op,
                            SetVector<Operation *> *backwardSlice,
                            TransitiveFilter filter) {
  getBackwardSliceImpl(op, backwardSlice, filter);

  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  backwardSlice->remove(op);
}

void mlir::getBackwardSlice(Value root, SetVector<Operation *> *backwardSlice,
                            TransitiveFilter filter) {
  if (Operation *definingOp = root.getDefiningOp()) {
    getBackwardSlice(definingOp, backwardSlice, filter);
    return;
  }
  Operation *bbAargOwner = root.cast<BlockArgument>().getOwner()->getParentOp();
  getBackwardSlice(bbAargOwner, backwardSlice, filter);
}

SetVector<Operation *> mlir::getSlice(Operation *op,
                                      TransitiveFilter backwardFilter,
                                      TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    getBackwardSlice(currentOp, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return topologicalSort(slice);
}

namespace {
/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set)
      : toSort(set), topologicalCounts(), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;
};
} // namespace

static void DFSPostorder(Operation *current, DFSState *state) {
  for (Value result : current->getResults()) {
    for (Operation *op : result.getUsers())
      DFSPostorder(op, state);
  }
  bool inserted;
  using IterTy = decltype(state->seen.begin());
  IterTy iter;
  std::tie(iter, inserted) = state->seen.insert(current);
  if (inserted) {
    if (state->toSort.count(current) > 0) {
      state->topologicalCounts.push_back(current);
    }
  }
}

SetVector<Operation *>
mlir::topologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    DFSPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}
