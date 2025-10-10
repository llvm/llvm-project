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
#include "llvm/Support/DebugLog.h"
#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;

#define DEBUG_TYPE "dense-analysis"

//===----------------------------------------------------------------------===//
// AbstractDenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

void AbstractDenseForwardDataFlowAnalysis::initializeEquivalentLatticeAnchor(
    Operation *top) {
  LDBG() << "initializeEquivalentLatticeAnchor: "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  top->walk([&](Operation *op) {
    if (isa<RegionBranchOpInterface, CallOpInterface>(op)) {
      LDBG() << "  Skipping "
             << OpWithFlags(op, OpPrintingFlags().skipRegions())
             << " (region branch or call)";
      return;
    }
    LDBG() << "  Building equivalent lattice anchor for "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    buildOperationEquivalentLatticeAnchor(op);
  });
}

LogicalResult AbstractDenseForwardDataFlowAnalysis::initialize(Operation *top) {
  LDBG() << "initialize (forward): "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  // Visit every operation and block.
  if (failed(processOperation(top))) {
    LDBG() << "  Failed to process top-level operation";
    return failure();
  }

  for (Region &region : top->getRegions()) {
    LDBG() << "  Processing region with " << region.getBlocks().size()
           << " blocks";
    for (Block &block : region) {
      LDBG() << "    Processing block with " << block.getOperations().size()
             << " operations";
      visitBlock(&block);
      for (Operation &op : block) {
        LDBG() << "      Initializing operation: "
               << OpWithFlags(&op, OpPrintingFlags().skipRegions());
        if (failed(initialize(&op))) {
          LDBG() << "      Failed to initialize operation";
          return failure();
        }
      }
    }
  }
  LDBG() << "  Forward initialization completed successfully";
  return success();
}

LogicalResult AbstractDenseForwardDataFlowAnalysis::visit(ProgramPoint *point) {
  LDBG() << "visit (forward): " << *point;
  if (!point->isBlockStart()) {
    LDBG() << "  Processing operation: "
           << OpWithFlags(point->getPrevOp(), OpPrintingFlags().skipRegions());
    return processOperation(point->getPrevOp());
  }
  LDBG() << "  Visiting block: " << point->getBlock();
  visitBlock(point->getBlock());
  return success();
}

void AbstractDenseForwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &before,
    AbstractDenseLattice *after) {
  LDBG() << "visitCallOperation (forward): "
         << OpWithFlags(call.getOperation(), OpPrintingFlags().skipRegions());
  LDBG() << "  before state: " << before;
  LDBG() << "  after state: " << *after;

  // Allow for customizing the behavior of calls to external symbols, including
  // when the analysis is explicitly marked as non-interprocedural.
  auto isExternalCallable = [&]() {
    auto callable =
        dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    return callable && !callable.getCallableRegion();
  };
  if (!getSolverConfig().isInterprocedural() || isExternalCallable()) {
    LDBG() << "  Handling as external callee (non-interprocedural or external)";
    return visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExternalCallee, before, after);
  }

  const auto *predecessors = getOrCreateFor<PredecessorState>(
      getProgramPointAfter(call.getOperation()), getProgramPointAfter(call));
  // Otherwise, if not all return sites are known, then conservatively assume we
  // can't reason about the data-flow.
  if (!predecessors->allPredecessorsKnown()) {
    LDBG() << "  Not all predecessors known, setting to entry state";
    return setToEntryState(after);
  }

  LDBG() << "  Processing " << predecessors->getKnownPredecessors().size()
         << " known predecessors";
  for (Operation *predecessor : predecessors->getKnownPredecessors()) {
    LDBG() << "    Processing predecessor: "
           << OpWithFlags(predecessor, OpPrintingFlags().skipRegions());
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
    LDBG() << "    Lattice at callee return: " << *latticeAtCalleeReturn;
    visitCallControlFlowTransfer(call, CallControlFlowAction::ExitCallee,
                                 *latticeAtCalleeReturn, latticeAfterCall);
  }
}

LogicalResult
AbstractDenseForwardDataFlowAnalysis::processOperation(Operation *op) {
  LDBG() << "processOperation (forward): "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  ProgramPoint *point = getProgramPointAfter(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<Executable>(point, getProgramPointBefore(op->getBlock()))
           ->isLive()) {
    LDBG() << "  Block not executable, skipping operation";
    return success();
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);

  // Get the dense state before the execution of the op.
  const AbstractDenseLattice *before =
      getLatticeFor(point, getProgramPointBefore(op));
  LDBG() << "  before state: " << *before;
  LDBG() << "  after state: " << *after;

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    LDBG() << "  Processing as region branch operation";
    visitRegionBranchOperation(point, branch, after);
    return success();
  }

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    LDBG() << "  Processing as call operation";
    visitCallOperation(call, *before, after);
    return success();
  }

  // Invoke the operation transfer function.
  LDBG() << "  Invoking operation transfer function";
  return visitOperationImpl(op, *before, after);
}

void AbstractDenseForwardDataFlowAnalysis::visitBlock(Block *block) {
  LDBG() << "visitBlock (forward): " << block;
  // If the block is not executable, bail out.
  ProgramPoint *point = getProgramPointBefore(block);
  if (!getOrCreateFor<Executable>(point, point)->isLive()) {
    LDBG() << "  Block not executable, skipping";
    return;
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);
  LDBG() << "  Block lattice state: " << *after;

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    LDBG() << "  Processing entry block";
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      LDBG() << "    Entry block of callable region";
      const auto *callsites = getOrCreateFor<PredecessorState>(
          point, getProgramPointAfter(callable));
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints. Do the same if
      // interprocedural analysis is not enabled.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural()) {
        LDBG() << "    Not all callsites known or non-interprocedural, setting "
                  "to entry state";
        return setToEntryState(after);
      }
      LDBG() << "    Processing " << callsites->getKnownPredecessors().size()
             << " known callsites";
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        LDBG() << "      Processing callsite: "
               << OpWithFlags(callsite, OpPrintingFlags().skipRegions());
        // Get the dense lattice before the callsite.
        const AbstractDenseLattice *before;
        before = getLatticeFor(point, getProgramPointBefore(callsite));
        LDBG() << "      Lattice before callsite: " << *before;

        visitCallControlFlowTransfer(cast<CallOpInterface>(callsite),
                                     CallControlFlowAction::EnterCallee,
                                     *before, after);
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      LDBG() << "    Entry block of region branch operation";
      return visitRegionBranchOperation(point, branch, after);
    }

    // Otherwise, we can't reason about the data-flow.
    LDBG() << "    Cannot reason about data-flow, setting to entry state";
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  LDBG() << "  Joining state from "
         << std::distance(block->pred_begin(), block->pred_end())
         << " predecessors";
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
    if (!getOrCreateFor<Executable>(
             point, getLatticeAnchor<CFGEdge>(predecessor, block))
             ->isLive()) {
      LDBG() << "    Skipping non-executable edge from " << predecessor;
      continue;
    }

    LDBG() << "    Joining state from predecessor " << predecessor;
    // Merge in the state from the predecessor's terminator.
    join(after, *getLatticeFor(
                    point, getProgramPointAfter(predecessor->getTerminator())));
  }
}

void AbstractDenseForwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint *point, RegionBranchOpInterface branch,
    AbstractDenseLattice *after) {
  LDBG() << "visitRegionBranchOperation (forward): "
         << OpWithFlags(branch.getOperation(), OpPrintingFlags().skipRegions());
  LDBG() << "  point: " << *point;
  LDBG() << "  after state: " << *after;

  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  LDBG() << "  Processing " << predecessors->getKnownPredecessors().size()
         << " known predecessors";
  for (Operation *op : predecessors->getKnownPredecessors()) {
    LDBG() << "    Processing predecessor: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      LDBG() << "      Predecessor is the branch itself, getting state before "
                "parent";
      before = getLatticeFor(point, getProgramPointBefore(op));
      // Otherwise, get the state after the terminator.
    } else {
      LDBG()
          << "      Predecessor is terminator, getting state after terminator";
      before = getLatticeFor(point, getProgramPointAfter(op));
    }
    LDBG() << "      before state: " << *before;

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
    LDBG() << "      regionFrom: "
           << (regionFrom ? std::to_string(*regionFrom) : "parent");

    if (point->isBlockStart()) {
      unsigned regionTo = point->getBlock()->getParent()->getRegionNumber();
      LDBG() << "      Point is block start, regionTo: " << regionTo;
      LDBG() << "      Calling visitRegionBranchControlFlowTransfer with "
                "regionFrom/regionTo";
      visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo,
                                           *before, after);
    } else {
      assert(point->getPrevOp() == branch &&
             "expected to be visiting the branch itself");
      LDBG() << "      Point is not block start, checking if predecessor is "
                "region or op itself";
      // Only need to call the arc transfer when the predecessor is the region
      // or the op itself, not the previous op.
      if (op->getParentOp() == branch || op == branch) {
        LDBG() << "      Predecessor is region or op itself, calling "
                  "visitRegionBranchControlFlowTransfer";
        visitRegionBranchControlFlowTransfer(
            branch, regionFrom, /*regionTo=*/std::nullopt, *before, after);
      } else {
        LDBG()
            << "      Predecessor is not region or op itself, performing join";
        join(after, *before);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AbstractDenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

void AbstractDenseBackwardDataFlowAnalysis::initializeEquivalentLatticeAnchor(
    Operation *top) {
  LDBG() << "initializeEquivalentLatticeAnchor (backward): "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  top->walk([&](Operation *op) {
    if (isa<RegionBranchOpInterface, CallOpInterface>(op)) {
      LDBG() << "  Skipping "
             << OpWithFlags(op, OpPrintingFlags().skipRegions())
             << " (region branch or call)";
      return;
    }
    LDBG() << "  Building equivalent lattice anchor for "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    buildOperationEquivalentLatticeAnchor(op);
  });
}

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::initialize(Operation *top) {
  LDBG() << "initialize (backward): "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  // Visit every operation and block.
  if (failed(processOperation(top))) {
    LDBG() << "  Failed to process top-level operation";
    return failure();
  }

  for (Region &region : top->getRegions()) {
    LDBG() << "  Processing region with " << region.getBlocks().size()
           << " blocks";
    for (Block &block : region) {
      LDBG() << "    Processing block with " << block.getOperations().size()
             << " operations";
      visitBlock(&block);
      for (Operation &op : llvm::reverse(block)) {
        LDBG() << "      Initializing operation (backward): "
               << OpWithFlags(&op, OpPrintingFlags().skipRegions());
        if (failed(initialize(&op))) {
          LDBG() << "      Failed to initialize operation";
          return failure();
        }
      }
    }
  }
  LDBG() << "  Backward initialization completed successfully";
  return success();
}

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::visit(ProgramPoint *point) {
  LDBG() << "visit (backward): " << *point;
  if (!point->isBlockEnd()) {
    LDBG() << "  Processing operation: "
           << OpWithFlags(point->getNextOp(), OpPrintingFlags().skipRegions());
    return processOperation(point->getNextOp());
  }
  LDBG() << "  Visiting block: " << point->getBlock();
  visitBlock(point->getBlock());
  return success();
}

void AbstractDenseBackwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &after,
    AbstractDenseLattice *before) {
  LDBG() << "visitCallOperation (backward): "
         << OpWithFlags(call.getOperation(), OpPrintingFlags().skipRegions());
  LDBG() << "  after state: " << after;
  LDBG() << "  before state: " << *before;

  // If the solver is not interprocedural, let the hook handle it as an external
  // callee.
  if (!getSolverConfig().isInterprocedural()) {
    LDBG() << "  Non-interprocedural analysis, handling as external callee";
    return visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExternalCallee, after, before);
  }

  // Find the callee.
  Operation *callee = call.resolveCallableInTable(&symbolTable);
  if (callee) {
    LDBG() << "  Resolved callee: "
           << OpWithFlags(callee, OpPrintingFlags().skipRegions());
  } else {
    LDBG() << "  Resolved callee: null";
  }

  auto callable = dyn_cast_or_null<CallableOpInterface>(callee);
  // No region means the callee is only declared in this module.
  // If that is the case or if the solver is not interprocedural,
  // let the hook handle it.
  if (callable && (!callable.getCallableRegion() ||
                   callable.getCallableRegion()->empty())) {
    LDBG() << "  Callee has no region or empty region, handling as external "
              "callee";
    return visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExternalCallee, after, before);
  }

  if (!callable) {
    LDBG() << "  No callable found, setting to exit state";
    return setToExitState(before);
  }

  Region *region = callable.getCallableRegion();
  LDBG() << "  Processing callable with region";

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
  LDBG() << "  Lattice at callee entry: " << latticeAtCalleeEntry;
  AbstractDenseLattice *latticeBeforeCall = before;
  visitCallControlFlowTransfer(call, CallControlFlowAction::EnterCallee,
                               latticeAtCalleeEntry, latticeBeforeCall);
}

LogicalResult
AbstractDenseBackwardDataFlowAnalysis::processOperation(Operation *op) {
  LDBG() << "processOperation (backward): "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  ProgramPoint *point = getProgramPointBefore(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<Executable>(point, getProgramPointBefore(op->getBlock()))
           ->isLive()) {
    LDBG() << "  Block not executable, skipping operation";
    return success();
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *before = getLattice(point);

  // Get the dense state after execution of this op.
  const AbstractDenseLattice *after =
      getLatticeFor(point, getProgramPointAfter(op));
  LDBG() << "  before state: " << *before;
  LDBG() << "  after state: " << *after;

  // Special cases where control flow may dictate data flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    LDBG() << "  Processing as region branch operation";
    visitRegionBranchOperation(point, branch, RegionBranchPoint::parent(),
                               before);
    return success();
  }
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    LDBG() << "  Processing as call operation";
    visitCallOperation(call, *after, before);
    return success();
  }

  // Invoke the operation transfer function.
  LDBG() << "  Invoking operation transfer function";
  return visitOperationImpl(op, *after, before);
}

void AbstractDenseBackwardDataFlowAnalysis::visitBlock(Block *block) {
  LDBG() << "visitBlock (backward): " << block;
  ProgramPoint *point = getProgramPointAfter(block);
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(point, getProgramPointBefore(block))
           ->isLive()) {
    LDBG() << "  Block not executable, skipping";
    return;
  }

  AbstractDenseLattice *before = getLattice(point);
  LDBG() << "  Block lattice state: " << *before;

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
    LDBG() << "  Processing exit block";
    // If this block is exiting from a callable, the successors of exiting from
    // a callable are the successors of all call sites. And the call sites
    // themselves are predecessors of the callable.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      LDBG() << "    Exit block of callable region";
      const auto *callsites = getOrCreateFor<PredecessorState>(
          point, getProgramPointAfter(callable));
      // If not all call sites are known, conservative mark all lattices as
      // having reached their pessimistic fix points.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural()) {
        LDBG() << "    Not all callsites known or non-interprocedural, setting "
                  "to exit state";
        return setToExitState(before);
      }

      LDBG() << "    Processing " << callsites->getKnownPredecessors().size()
             << " known callsites";
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        LDBG() << "      Processing callsite: "
               << OpWithFlags(callsite, OpPrintingFlags().skipRegions());
        const AbstractDenseLattice *after =
            getLatticeFor(point, getProgramPointAfter(callsite));
        LDBG() << "      Lattice after callsite: " << *after;
        visitCallControlFlowTransfer(cast<CallOpInterface>(callsite),
                                     CallControlFlowAction::ExitCallee, *after,
                                     before);
      }
      return;
    }

    // If this block is exiting from an operation with region-based control
    // flow, propagate the lattice back along the control flow edge.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      LDBG() << "    Exit block of region branch operation";
      visitRegionBranchOperation(point, branch, block->getParent(), before);
      return;
    }

    // Cannot reason about successors of an exit block, set the pessimistic
    // fixpoint.
    LDBG() << "    Cannot reason about successors, setting to exit state";
    return setToExitState(before);
  }

  // Meet the state with the state before block's successors.
  LDBG() << "  Meeting state from " << block->getSuccessors().size()
         << " successors";
  for (Block *successor : block->getSuccessors()) {
    if (!getOrCreateFor<Executable>(point,
                                    getLatticeAnchor<CFGEdge>(block, successor))
             ->isLive()) {
      LDBG() << "    Skipping non-executable edge to " << successor;
      continue;
    }

    LDBG() << "    Meeting state from successor " << successor;
    // Merge in the state from the successor: either the first operation, or the
    // block itself when empty.
    meet(before, *getLatticeFor(point, getProgramPointBefore(successor)));
  }
}

void AbstractDenseBackwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint branchPoint, AbstractDenseLattice *before) {
  LDBG() << "visitRegionBranchOperation (backward): "
         << OpWithFlags(branch.getOperation(), OpPrintingFlags().skipRegions());
  LDBG() << "  branchPoint: " << (branchPoint.isParent() ? "parent" : "region");
  LDBG() << "  before state: " << *before;

  // The successors of the operation may be either the first operation of the
  // entry block of each possible successor region, or the next operation when
  // the branch is a successor of itself.
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(branchPoint, successors);
  LDBG() << "  Processing " << successors.size() << " successor regions";
  for (const RegionSuccessor &successor : successors) {
    const AbstractDenseLattice *after;
    if (successor.isParent() || successor.getSuccessor()->empty()) {
      LDBG() << "    Successor is parent or empty region";
      after = getLatticeFor(point, getProgramPointAfter(branch));
    } else {
      Region *successorRegion = successor.getSuccessor();
      assert(!successorRegion->empty() && "unexpected empty successor region");
      Block *successorBlock = &successorRegion->front();
      LDBG() << "    Successor region with "
             << successorRegion->getBlocks().size() << " blocks";

      if (!getOrCreateFor<Executable>(point,
                                      getProgramPointBefore(successorBlock))
               ->isLive()) {
        LDBG() << "    Successor block not executable, skipping";
        continue;
      }

      after = getLatticeFor(point, getProgramPointBefore(successorBlock));
    }
    LDBG() << "    After state: " << *after;

    visitRegionBranchControlFlowTransfer(branch, branchPoint, successor, *after,
                                         before);
  }
}
