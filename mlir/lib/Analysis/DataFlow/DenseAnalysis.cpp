//===- DenseAnalysis.cpp - Dense data-flow analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AbstractDenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult AbstractDenseDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  visitOperation(top);
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      visitBlock(&block);
      for (Operation &op : block)
        if (failed(initialize(&op)))
          return failure();
    }
  }
  return success();
}

LogicalResult AbstractDenseDataFlowAnalysis::visit(ProgramPoint point) {
  if (auto *op = point.dyn_cast<Operation *>())
    visitOperation(op);
  else if (auto *block = point.dyn_cast<Block *>())
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractDenseDataFlowAnalysis::visitOperation(Operation *op) {
  // If the containing block is not executable, bail out.
  if (!getOrCreateFor<Executable>(op, op->getBlock())->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(op);

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    return visitRegionBranchOperation(op, branch, after);

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    const auto *predecessors = getOrCreateFor<PredecessorState>(op, call);
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown())
      return setToEntryState(after);
    for (Operation *predecessor : predecessors->getKnownPredecessors())
      join(after, *getLatticeFor(op, predecessor));
    return;
  }

  // Get the dense state before the execution of the op.
  const AbstractDenseLattice *before;
  if (Operation *prev = op->getPrevNode())
    before = getLatticeFor(op, prev);
  else
    before = getLatticeFor(op, op->getBlock());

  // Invoke the operation transfer function.
  visitOperationImpl(op, *before, after);
}

void AbstractDenseDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(block, block)->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(block);

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown())
        return setToEntryState(after);
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        // Get the dense lattice before the callsite.
        if (Operation *prev = callsite->getPrevNode())
          join(after, *getLatticeFor(block, prev));
        else
          join(after, *getLatticeFor(block, callsite->getBlock()));
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp()))
      return visitRegionBranchOperation(block, branch, after);

    // Otherwise, we can't reason about the data-flow.
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
    if (!getOrCreateFor<Executable>(
             block, getProgramPoint<CFGEdge>(predecessor, block))
             ->isLive())
      continue;

    // Merge in the state from the predecessor's terminator.
    join(after, *getLatticeFor(block, predecessor->getTerminator()));
  }
}

void AbstractDenseDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch,
    AbstractDenseLattice *after) {
  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      if (Operation *prev = op->getPrevNode())
        before = getLatticeFor(point, prev);
      else
        before = getLatticeFor(point, op->getBlock());

      // Otherwise, get the state after the terminator.
    } else {
      before = getLatticeFor(point, op);
    }
    join(after, *before);
  }
}

const AbstractDenseLattice *
AbstractDenseDataFlowAnalysis::getLatticeFor(ProgramPoint dependent,
                                             ProgramPoint point) {
  AbstractDenseLattice *state = getLattice(point);
  addDependency(state, dependent);
  return state;
}
