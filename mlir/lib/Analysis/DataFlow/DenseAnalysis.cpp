//===- DenseAnalysis.cpp - Dense data-flow analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AbstractDenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult AbstractDenseForwardDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  if (failed(processOperation(top)))
    return failure();

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

LogicalResult AbstractDenseForwardDataFlowAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart())
    return processOperation(point->getPrevOp());
  visitBlock(point->getBlock());
  return success();
}

void AbstractDenseForwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &before,
    AbstractDenseLattice *after) {
  // Allow for customizing the behavior of calls to external symbols, including
  // when the analysis is explicitly marked as non-interprocedural.
  auto callable =
      dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!getSolverConfig().isInterprocedural() ||
      (callable && !callable.getCallableRegion())) {
    return visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExternalCallee, before, after);
  }

  const auto *predecessors = getOrCreateFor<PredecessorState>(
      getProgramPointAfter(call.getOperation()), getProgramPointAfter(call));
  // Otherwise, if not all return sites are known, then conservatively assume we
  // can't reason about the data-flow.
  if (!predecessors->allPredecessorsKnown())
    return setToEntryState(after);

  for (Operation *predecessor : predecessors->getKnownPredecessors()) {
    // Get the lattices at callee return:
    //
    //   func.func @callee() {
    //     ...
    //     return  // predecessor
    //     // latticeAtCalleeReturn
    //   }
    //   func.func @caller() {
    //     ...
    //     call @callee
    //     // latticeAfterCall
    //     ...
    //   }
    AbstractDenseLattice *latticeAfterCall = after;
    const AbstractDenseLattice *latticeAtCalleeReturn =
        getLatticeFor(getProgramPointAfter(call.getOperation()),
                      getProgramPointAfter(predecessor));
    visitCallControlFlowTransfer(call, CallControlFlowAction::ExitCallee,
                                 *latticeAtCalleeReturn, latticeAfterCall);
  }
}

LogicalResult
AbstractDenseForwardDataFlowAnalysis::processOperation(Operation *op) {
  ProgramPoint *point = getProgramPointAfter(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<Executable>(point, getProgramPointBefore(op->getBlock()))
           ->isLive())
    return success();

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);

  // Get the dense state before the execution of the op.
  const AbstractDenseLattice *before =
      getLatticeFor(point, getProgramPointBefore(op));

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionBranchOperation(point, branch, after);
    return success();
  }

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    visitCallOperation(call, *before, after);
    return success();
  }

  // Invoke the operation transfer function.
  return visitOperationImpl(op, *before, after);
}

void AbstractDenseForwardDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  ProgramPoint *point = getProgramPointBefore(block);
  if (!getOrCreateFor<Executable>(point, point)->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(
          point, getProgramPointAfter(callable));
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints. Do the same if
      // interprocedural analysis is not enabled.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural())
        return setToEntryState(after);
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        // Get the dense lattice before the callsite.
        const AbstractDenseLattice *before;
        before = getLatticeFor(point, getProgramPointBefore(callsite));

        visitCallControlFlowTransfer(cast<CallOpInterface>(callsite),
                                     CallControlFlowAction::EnterCallee,
                                     *before, after);
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp()))
      return visitRegionBranchOperation(point, branch, after);

    // Otherwise, we can't reason about the data-flow.
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
    if (!getOrCreateFor<Executable>(
             point, getLatticeAnchor<CFGEdge>(predecessor, block))
             ->isLive())
      continue;

    // Merge in the state from the predecessor's terminator.
    join(after, *getLatticeFor(
                    point, getProgramPointAfter(predecessor->getTerminator())));
  }
}

void AbstractDenseForwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint *point, RegionBranchOpInterface branch,
    AbstractDenseLattice *after) {
  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      before = getLatticeFor(point, getProgramPointBefore(op));
      // Otherwise, get the state after the terminator.
    } else {
      before = getLatticeFor(point, getProgramPointAfter(op));
    }

    // This function is called in two cases:
    //   1. when visiting the block (point = block start);
    //   2. when visiting the parent operation (point = iter after parent op).
    // In both cases, we are looking for predecessor operations of the point,
    //   1. predecessor may be the terminator of another block from another
    //   region (assuming that the block does belong to another region via an
    //   assertion) or the parent (when parent can transfer control to this
    //   region);
    //   2. predecessor may be the terminator of a block that exits the
    //   region (when region transfers control to the parent) or the operation
    //   before the parent.
    // In the latter case, just perform the join as it isn't the control flow
    // affected by the region.
    std::optional<unsigned> regionFrom =
        op == branch ? std::optional<unsigned>()
                     : op->getBlock()->getParent()->getRegionNumber();
    if (point->isBlockStart()) {
      unsigned regionTo = point->getBlock()->getParent()->getRegionNumber();
      visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo,
                                           *before, after);
    } else {
      assert(point->getPrevOp() == branch &&
             "expected to be visiting the branch itself");
      // Only need to call the arc transfer when the predecessor is the region
      // or the op itself, not the previous op.
      if (op->getParentOp() == branch || op == branch) {
        visitRegionBranchControlFlowTransfer(
            branch, regionFrom, /*regionTo=*/std::nullopt, *before, after);
      } else {
        join(after, *before);
      }
    }
  }
}

const AbstractDenseLattice *
AbstractDenseForwardDataFlowAnalysis::getLatticeFor(ProgramPoint *dependent,
                                                    LatticeAnchor anchor) {
  AbstractDenseLattice *state = getLattice(anchor);
  addDependency(state, dependent);
  return state;
}

//===----------------------------------------------------------------------===//
// AbstractDenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  if (failed(processOperation(top)))
    return failure();

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

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockEnd())
    return processOperation(point->getNextOp());
  visitBlock(point->getBlock());
  return success();
}

void AbstractDenseBackwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &after,
    AbstractDenseLattice *before) {
  // Find the callee.
  Operation *callee = call.resolveCallableInTable(&symbolTable);

  auto callable = dyn_cast_or_null<CallableOpInterface>(callee);
  // No region means the callee is only declared in this module.
  // If that is the case or if the solver is not interprocedural,
  // let the hook handle it.
  if (!getSolverConfig().isInterprocedural() ||
      (callable && (!callable.getCallableRegion() ||
                    callable.getCallableRegion()->empty()))) {
    return visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExternalCallee, after, before);
  }

  if (!callable)
    return setToExitState(before);

  Region *region = callable.getCallableRegion();

  // Call-level control flow specifies the data flow here.
  //
  //   func.func @callee() {
  //     ^calleeEntryBlock:
  //     // latticeAtCalleeEntry
  //     ...
  //   }
  //   func.func @caller() {
  //     ...
  //     // latticeBeforeCall
  //     call @callee
  //     ...
  //   }
  Block *calleeEntryBlock = &region->front();
  ProgramPoint *calleeEntry = getProgramPointBefore(calleeEntryBlock);
  const AbstractDenseLattice &latticeAtCalleeEntry =
      *getLatticeFor(getProgramPointBefore(call.getOperation()), calleeEntry);
  AbstractDenseLattice *latticeBeforeCall = before;
  visitCallControlFlowTransfer(call, CallControlFlowAction::EnterCallee,
                               latticeAtCalleeEntry, latticeBeforeCall);
}

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::processOperation(Operation *op) {
  ProgramPoint *point = getProgramPointBefore(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<Executable>(point, getProgramPointBefore(op->getBlock()))
           ->isLive())
    return success();

  // Get the dense lattice to update.
  AbstractDenseLattice *before = getLattice(point);

  // Get the dense state after execution of this op.
  const AbstractDenseLattice *after =
      getLatticeFor(point, getProgramPointAfter(op));

  // Special cases where control flow may dictate data flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionBranchOperation(point, branch, RegionBranchPoint::parent(),
                               before);
    return success();
  }
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    visitCallOperation(call, *after, before);
    return success();
  }

  // Invoke the operation transfer function.
  return visitOperationImpl(op, *after, before);
}

void AbstractDenseBackwardDataFlowAnalysis::visitBlock(Block *block) {
  ProgramPoint *point = getProgramPointAfter(block);
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(point, getProgramPointBefore(block))
           ->isLive())
    return;

  AbstractDenseLattice *before = getLattice(point);

  // We need "exit" blocks, i.e. the blocks that may return control to the
  // parent operation.
  auto isExitBlock = [](Block *b) {
    // Treat empty and terminator-less blocks as exit blocks.
    if (b->empty() || !b->back().mightHaveTrait<OpTrait::IsTerminator>())
      return true;

    // There may be a weird case where a terminator may be transferring control
    // either to the parent or to another block, so exit blocks and successors
    // are not mutually exclusive.
    return isa_and_nonnull<RegionBranchTerminatorOpInterface>(
        b->getTerminator());
  };
  if (isExitBlock(block)) {
    // If this block is exiting from a callable, the successors of exiting from
    // a callable are the successors of all call sites. And the call sites
    // themselves are predecessors of the callable.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(
          point, getProgramPointAfter(callable));
      // If not all call sites are known, conservative mark all lattices as
      // having reached their pessimistic fix points.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural()) {
        return setToExitState(before);
      }

      for (Operation *callsite : callsites->getKnownPredecessors()) {
        const AbstractDenseLattice *after =
            getLatticeFor(point, getProgramPointAfter(callsite));
        visitCallControlFlowTransfer(cast<CallOpInterface>(callsite),
                                     CallControlFlowAction::ExitCallee, *after,
                                     before);
      }
      return;
    }

    // If this block is exiting from an operation with region-based control
    // flow, propagate the lattice back along the control flow edge.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      visitRegionBranchOperation(point, branch, block->getParent(), before);
      return;
    }

    // Cannot reason about successors of an exit block, set the pessimistic
    // fixpoint.
    return setToExitState(before);
  }

  // Meet the state with the state before block's successors.
  for (Block *successor : block->getSuccessors()) {
    if (!getOrCreateFor<Executable>(point,
                                    getLatticeAnchor<CFGEdge>(block, successor))
             ->isLive())
      continue;

    // Merge in the state from the successor: either the first operation, or the
    // block itself when empty.
    meet(before, *getLatticeFor(point, getProgramPointBefore(successor)));
  }
}

void AbstractDenseBackwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint branchPoint, AbstractDenseLattice *before) {

  // The successors of the operation may be either the first operation of the
  // entry block of each possible successor region, or the next operation when
  // the branch is a successor of itself.
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(branchPoint, successors);
  for (const RegionSuccessor &successor : successors) {
    const AbstractDenseLattice *after;
    if (successor.isParent() || successor.getSuccessor()->empty()) {
      after = getLatticeFor(point, getProgramPointAfter(branch));
    } else {
      Region *successorRegion = successor.getSuccessor();
      assert(!successorRegion->empty() && "unexpected empty successor region");
      Block *successorBlock = &successorRegion->front();

      if (!getOrCreateFor<Executable>(point,
                                      getProgramPointBefore(successorBlock))
               ->isLive())
        continue;

      after = getLatticeFor(point, getProgramPointBefore(successorBlock));
    }

    visitRegionBranchControlFlowTransfer(branch, branchPoint, successor, *after,
                                         before);
  }
}

const AbstractDenseLattice *
AbstractDenseBackwardDataFlowAnalysis::getLatticeFor(ProgramPoint *dependent,
                                                     LatticeAnchor anchor) {
  AbstractDenseLattice *state = getLattice(anchor);
  addDependency(state, dependent);
  return state;
}
