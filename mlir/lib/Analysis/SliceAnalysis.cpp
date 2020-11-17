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
#include "mlir/IR/BuiltinDialect.h"
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
  if (!op) {
    return;
  }

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(op)) {
    return;
  }

  if (auto forOp = dyn_cast<AffineForOp>(op)) {
    for (Operation *userOp : forOp.getInductionVar().getUsers())
      if (forwardSlice->count(userOp) == 0)
        getForwardSliceImpl(userOp, forwardSlice, filter);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (Operation *userOp : forOp.getInductionVar().getUsers())
      if (forwardSlice->count(userOp) == 0)
        getForwardSliceImpl(userOp, forwardSlice, filter);
    for (Value result : forOp.getResults())
      for (Operation *userOp : result.getUsers())
        if (forwardSlice->count(userOp) == 0)
          getForwardSliceImpl(userOp, forwardSlice, filter);
  } else {
    assert(op->getNumRegions() == 0 && "unexpected generic op with regions");
    for (Value result : op->getResults()) {
      for (Operation *userOp : result.getUsers())
        if (forwardSlice->count(userOp) == 0)
          getForwardSliceImpl(userOp, forwardSlice, filter);
    }
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

static void getBackwardSliceImpl(Operation *op,
                                 SetVector<Operation *> *backwardSlice,
                                 TransitiveFilter filter) {
  if (!op)
    return;

  assert((op->getNumRegions() == 0 ||
          isa<AffineForOp, scf::ForOp, linalg::LinalgOp>(op)) &&
         "unexpected generic op with regions");

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(op)) {
    return;
  }

  for (auto en : llvm::enumerate(op->getOperands())) {
    auto operand = en.value();
    if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      if (auto affIv = getForInductionVarOwner(operand)) {
        auto *affOp = affIv.getOperation();
        if (backwardSlice->count(affOp) == 0)
          getBackwardSliceImpl(affOp, backwardSlice, filter);
      } else if (auto loopIv = scf::getForInductionVarOwner(operand)) {
        auto *loopOp = loopIv.getOperation();
        if (backwardSlice->count(loopOp) == 0)
          getBackwardSliceImpl(loopOp, backwardSlice, filter);
      } else if (blockArg.getOwner() !=
                 &op->getParentOfType<FuncOp>().getBody().front()) {
        op->emitError("unsupported CF for operand ") << en.index();
        llvm_unreachable("Unsupported control flow");
      }
      continue;
    }
    auto *op = operand.getDefiningOp();
    if (backwardSlice->count(op) == 0) {
      getBackwardSliceImpl(op, backwardSlice, filter);
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
