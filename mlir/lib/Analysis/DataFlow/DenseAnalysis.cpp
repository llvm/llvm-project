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
  processOperation(top);
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
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point))
    processOperation(op);
  else if (auto *block = llvm::dyn_cast_if_present<Block *>(point))
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractDenseDataFlowAnalysis::processOperation(Operation *op) {
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

//===----------------------------------------------------------------------===//
// AbstractDenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  processOperation(top);
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      visitBlock(&block);
      for (Operation &op : llvm::reverse(block)) {
        if (failed(initialize(&op)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult AbstractDenseBackwardDataFlowAnalysis::visit(ProgramPoint point) {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point))
    processOperation(op);
  else if (auto *block = llvm::dyn_cast_if_present<Block *>(point))
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractDenseBackwardDataFlowAnalysis::processOperation(Operation *op) {
  // If the containing block is not executable, bail out.
  if (!getOrCreateFor<Executable>(op, op->getBlock())->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseLattice *before = getLattice(op);

  // If the op implements region control flow, then the interface specifies the
  // control function.
  // TODO: this is not always true, e.g. linalg.generic, but is implement this
  // way for consistency with the dense forward analysis.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    return visitRegionBranchOperation(op, branch, std::nullopt, before);

  // If the op is a call-like, do inter-procedural data flow as follows:
  //
  //   - find the callable (resolve via the symbol table),
  //   - get the entry block of the callable region,
  //   - take the state before the first operation if present or at block end
  //   otherwise,
  //   - meet that state with the state before the call-like op.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    Operation *callee = call.resolveCallable(&symbolTable);
    if (auto callable = dyn_cast<CallableOpInterface>(callee)) {
      Region *region = callable.getCallableRegion();
      if (region && !region->empty()) {
        Block *entryBlock = &region->front();
        if (entryBlock->empty())
          meet(before, *getLatticeFor(op, entryBlock));
        else
          meet(before, *getLatticeFor(op, &entryBlock->front()));
      } else {
        setToExitState(before);
      }
    } else {
      setToExitState(before);
    }
    return;
  }

  // Get the dense state after execution of this op.
  const AbstractDenseLattice *after;
  if (Operation *next = op->getNextNode())
    after = getLatticeFor(op, next);
  else
    after = getLatticeFor(op, op->getBlock());

  // Invoke the operation transfer function.
  visitOperationImpl(op, *after, before);
}

void AbstractDenseBackwardDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(block, block)->isLive())
    return;

  AbstractDenseLattice *before = getLattice(block);

  // We need "exit" blocks, i.e. the blocks that may return control to the
  // parent operation.
  auto isExitBlock = [](Block *b) {
    // Treat empty and terminator-less blocks as exit blocks.
    if (b->empty() || !b->back().mightHaveTrait<OpTrait::IsTerminator>())
      return true;

    // There may be a weird case where a terminator may be transferring control
    // either to the parent or to another block, so exit blocks and successors
    // are not mutually exclusive.
    Operation *terminator = b->getTerminator();
    return terminator && (terminator->hasTrait<OpTrait::ReturnLike>() ||
                          isa<RegionBranchTerminatorOpInterface>(terminator));
  };
  if (isExitBlock(block)) {
    // If this block is exiting from a callable, the successors of exiting from
    // a callable are the successors of all call sites. And the call sites
    // themselves are predecessors of the callable.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all call sites are known, conservative mark all lattices as
      // having reached their pessimistic fix points.
      if (!callsites->allPredecessorsKnown())
        return setToExitState(before);

      for (Operation *callsite : callsites->getKnownPredecessors()) {
        if (Operation *next = callsite->getNextNode())
          meet(before, *getLatticeFor(block, next));
        else
          meet(before, *getLatticeFor(block, callsite->getBlock()));
      }
      return;
    }

    // If this block is exiting from an operation with region-based control
    // flow, follow that flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      visitRegionBranchOperation(block, branch,
                                 block->getParent()->getRegionNumber(), before);
      return;
    }

    // Cannot reason about successors of an exit block, set the pessimistic
    // fixpoint.
    return setToExitState(before);
  }

  // Meet the state with the state before block's successors.
  for (Block *successor : block->getSuccessors()) {
    if (!getOrCreateFor<Executable>(block,
                                    getProgramPoint<CFGEdge>(block, successor))
             ->isLive())
      continue;

    // Merge in the state from the successor: either the first operation, or the
    // block itself when empty.
    if (successor->empty())
      meet(before, *getLatticeFor(block, successor));
    else
      meet(before, *getLatticeFor(block, &successor->front()));
  }
}

void AbstractDenseBackwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch,
    std::optional<unsigned> regionNo, AbstractDenseLattice *before) {

  // The successors of the operation may be either the first operation of the
  // entry block of each possible successor region, or the next operation when
  // the branch is a successor of itself.
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(regionNo, successors);
  for (const RegionSuccessor &successor : successors) {
    const AbstractDenseLattice *after;
    if (successor.isParent() || successor.getSuccessor()->empty()) {
      if (Operation *next = branch->getNextNode())
        after = getLatticeFor(point, next);
      else
        after = getLatticeFor(point, branch->getBlock());
    } else {
      Region *successorRegion = successor.getSuccessor();
      assert(!successorRegion->empty() && "unexpected empty successor region");
      Block *successorBlock = &successorRegion->front();

      if (!getOrCreateFor<Executable>(point, successorBlock)->isLive())
        continue;

      if (successorBlock->empty())
        after = getLatticeFor(point, successorBlock);
      else
        after = getLatticeFor(point, &successorBlock->front());
    }
    meet(before, *after);
  }
}

const AbstractDenseLattice *
AbstractDenseBackwardDataFlowAnalysis::getLatticeFor(ProgramPoint dependent,
                                                     ProgramPoint point) {
  AbstractDenseLattice *state = getLattice(point);
  addDependency(state, dependent);
  return state;
}
