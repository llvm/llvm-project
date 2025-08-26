//===- DeadCodeAnalysis.cpp - Dead code analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "dead-code-analysis"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

ChangeResult Executable::setToLive() {
  if (live)
    return ChangeResult::NoChange;
  live = true;
  return ChangeResult::Change;
}

void Executable::print(raw_ostream &os) const {
  os << (live ? "live" : "dead");
}

void Executable::onUpdate(DataFlowSolver *solver) const {
  AnalysisState::onUpdate(solver);

  if (ProgramPoint *pp = llvm::dyn_cast_if_present<ProgramPoint *>(anchor)) {
    if (pp->isBlockStart()) {
      // Re-invoke the analyses on the block itself.
      for (DataFlowAnalysis *analysis : subscribers)
        solver->enqueue({pp, analysis});
      // Re-invoke the analyses on all operations in the block.
      for (DataFlowAnalysis *analysis : subscribers)
        for (Operation &op : *pp->getBlock())
          solver->enqueue({solver->getProgramPointAfter(&op), analysis});
    }
  } else if (auto *latticeAnchor =
                 llvm::dyn_cast_if_present<GenericLatticeAnchor *>(anchor)) {
    // Re-invoke the analysis on the successor block.
    if (auto *edge = dyn_cast<CFGEdge>(latticeAnchor)) {
      for (DataFlowAnalysis *analysis : subscribers)
        solver->enqueue(
            {solver->getProgramPointBefore(edge->getTo()), analysis});
    }
  }
}

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

void PredecessorState::print(raw_ostream &os) const {
  if (allPredecessorsKnown())
    os << "(all) ";
  os << "predecessors:";
  if (getKnownPredecessors().empty())
    os << " (none)";
  else
    os << "\n";
  llvm::interleave(
      getKnownPredecessors(), os,
      [&](Operation *op) {
        os << "  " << OpWithFlags(op, OpPrintingFlags().skipRegions());
      },
      "\n");
}

ChangeResult PredecessorState::join(Operation *predecessor) {
  return knownPredecessors.insert(predecessor) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

ChangeResult PredecessorState::join(Operation *predecessor, ValueRange inputs) {
  ChangeResult result = join(predecessor);
  if (!inputs.empty()) {
    ValueRange &curInputs = successorInputs[predecessor];
    if (curInputs != inputs) {
      curInputs = inputs;
      result |= ChangeResult::Change;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CFGEdge
//===----------------------------------------------------------------------===//

Location CFGEdge::getLoc() const {
  return FusedLoc::get(
      getFrom()->getParent()->getContext(),
      {getFrom()->getParent()->getLoc(), getTo()->getParent()->getLoc()});
}

void CFGEdge::print(raw_ostream &os) const {
  getFrom()->print(os);
  os << "\n -> \n";
  getTo()->print(os);
}

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

DeadCodeAnalysis::DeadCodeAnalysis(DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerAnchorKind<CFGEdge>();
}

LogicalResult DeadCodeAnalysis::initialize(Operation *top) {
  LDBG() << "Initializing DeadCodeAnalysis for top-level op: "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  // Mark the top-level blocks as executable.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    auto *state =
        getOrCreate<Executable>(getProgramPointBefore(&region.front()));
    propagateIfChanged(state, state->setToLive());
    LDBG() << "Marked entry block live for region in op: "
           << OpWithFlags(top, OpPrintingFlags().skipRegions());
  }

  // Mark as overdefined the predecessors of symbol callables with potentially
  // unknown predecessors.
  initializeSymbolCallables(top);

  return initializeRecursively(top);
}

void DeadCodeAnalysis::initializeSymbolCallables(Operation *top) {
  LDBG() << "[init] Entering initializeSymbolCallables for top-level op: "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
  analysisScope = top;
  auto walkFn = [&](Operation *symTable, bool allUsesVisible) {
    LDBG() << "[init] Processing symbol table op: "
           << OpWithFlags(symTable, OpPrintingFlags().skipRegions());
    Region &symbolTableRegion = symTable->getRegion(0);
    Block *symbolTableBlock = &symbolTableRegion.front();

    bool foundSymbolCallable = false;
    for (auto callable : symbolTableBlock->getOps<CallableOpInterface>()) {
      LDBG() << "[init] Found CallableOpInterface: "
             << OpWithFlags(callable.getOperation(),
                            OpPrintingFlags().skipRegions());
      Region *callableRegion = callable.getCallableRegion();
      if (!callableRegion)
        continue;
      auto symbol = dyn_cast<SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;

      // Public symbol callables or those for which we can't see all uses have
      // potentially unknown callsites.
      if (symbol.isPublic() || (!allUsesVisible && symbol.isNested())) {
        auto *state =
            getOrCreate<PredecessorState>(getProgramPointAfter(callable));
        propagateIfChanged(state, state->setHasUnknownPredecessors());
        LDBG() << "[init] Marked callable as having unknown predecessors: "
               << OpWithFlags(callable.getOperation(),
                              OpPrintingFlags().skipRegions());
      }
      foundSymbolCallable = true;
    }

    // Exit early if no eligible symbol callables were found in the table.
    if (!foundSymbolCallable)
      return;

    // Walk the symbol table to check for non-call uses of symbols.
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(&symbolTableRegion);
    if (!uses) {
      // If we couldn't gather the symbol uses, conservatively assume that
      // we can't track information for any nested symbols.
      LDBG() << "[init] Could not gather symbol uses, conservatively marking "
                "all nested callables as having unknown predecessors";
      return top->walk([&](CallableOpInterface callable) {
        auto *state =
            getOrCreate<PredecessorState>(getProgramPointAfter(callable));
        propagateIfChanged(state, state->setHasUnknownPredecessors());
        LDBG() << "[init] Marked nested callable as "
                  "having unknown predecessors: "
               << OpWithFlags(callable.getOperation(),
                              OpPrintingFlags().skipRegions());
      });
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      if (isa<CallOpInterface>(use.getUser()))
        continue;
      // If a callable symbol has a non-call use, then we can't be guaranteed to
      // know all callsites.
      Operation *symbol = symbolTable.lookupSymbolIn(top, use.getSymbolRef());
      if (!symbol)
        continue;
      auto *state = getOrCreate<PredecessorState>(getProgramPointAfter(symbol));
      propagateIfChanged(state, state->setHasUnknownPredecessors());
      LDBG() << "[init] Found non-call use for symbol, "
                "marked as having unknown predecessors: "
             << OpWithFlags(symbol, OpPrintingFlags().skipRegions());
    }
  };
  SymbolTable::walkSymbolTables(top, /*allSymUsesVisible=*/!top->getBlock(),
                                walkFn);
  LDBG() << "[init] Finished initializeSymbolCallables for top-level op: "
         << OpWithFlags(top, OpPrintingFlags().skipRegions());
}

/// Returns true if the operation is a returning terminator in region
/// control-flow or the terminator of a callable region.
static bool isRegionOrCallableReturn(Operation *op) {
  return op->getBlock() != nullptr && !op->getNumSuccessors() &&
         isa<RegionBranchOpInterface, CallableOpInterface>(op->getParentOp()) &&
         op->getBlock()->getTerminator() == op;
}

LogicalResult DeadCodeAnalysis::initializeRecursively(Operation *op) {
  LDBG() << "[init] Entering initializeRecursively for op: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  // Initialize the analysis by visiting every op with control-flow semantics.
  if (op->getNumRegions() || op->getNumSuccessors() ||
      isRegionOrCallableReturn(op) || isa<CallOpInterface>(op)) {
    LDBG() << "[init] Visiting op with control-flow semantics: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    // When the liveness of the parent block changes, make sure to
    // re-invoke the analysis on the op.
    if (op->getBlock())
      getOrCreate<Executable>(getProgramPointBefore(op->getBlock()))
          ->blockContentSubscribe(this);
    // Visit the op.
    if (failed(visit(getProgramPointAfter(op))))
      return failure();
  }
  // Recurse on nested operations.
  for (Region &region : op->getRegions()) {
    LDBG() << "[init] Recursing into region of op: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    for (Operation &nestedOp : region.getOps()) {
      LDBG() << "[init] Recursing into nested op: "
             << OpWithFlags(&nestedOp, OpPrintingFlags().skipRegions());
      if (failed(initializeRecursively(&nestedOp)))
        return failure();
    }
  }
  LDBG() << "[init] Finished initializeRecursively for op: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  return success();
}

void DeadCodeAnalysis::markEdgeLive(Block *from, Block *to) {
  LDBG() << "Marking edge live from block " << from << " to block " << to;
  auto *state = getOrCreate<Executable>(getProgramPointBefore(to));
  propagateIfChanged(state, state->setToLive());
  auto *edgeState =
      getOrCreate<Executable>(getLatticeAnchor<CFGEdge>(from, to));
  propagateIfChanged(edgeState, edgeState->setToLive());
}

void DeadCodeAnalysis::markEntryBlocksLive(Operation *op) {
  LDBG() << "Marking entry blocks live for op: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto *state =
        getOrCreate<Executable>(getProgramPointBefore(&region.front()));
    propagateIfChanged(state, state->setToLive());
    LDBG() << "Marked entry block live for region in op: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
  }
}

LogicalResult DeadCodeAnalysis::visit(ProgramPoint *point) {
  LDBG() << "Visiting program point: " << *point;
  if (point->isBlockStart())
    return success();
  Operation *op = point->getPrevOp();
  LDBG() << "Visiting operation: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());

  // If the parent block is not executable, there is nothing to do.
  if (op->getBlock() != nullptr &&
      !getOrCreate<Executable>(getProgramPointBefore(op->getBlock()))
           ->isLive()) {
    LDBG() << "Parent block not live, skipping op: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    return success();
  }

  // We have a live call op. Add this as a live predecessor of the callee.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    LDBG() << "Visiting call operation: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    visitCallOperation(call);
  }

  // Visit the regions.
  if (op->getNumRegions()) {
    // Check if we can reason about the region control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      LDBG() << "Visiting region branch operation: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      visitRegionBranchOperation(branch);

      // Check if this is a callable operation.
    } else if (auto callable = dyn_cast<CallableOpInterface>(op)) {
      LDBG() << "Visiting callable operation: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      const auto *callsites = getOrCreateFor<PredecessorState>(
          getProgramPointAfter(op), getProgramPointAfter(callable));

      // If the callsites could not be resolved or are known to be non-empty,
      // mark the callable as executable.
      if (!callsites->allPredecessorsKnown() ||
          !callsites->getKnownPredecessors().empty())
        markEntryBlocksLive(callable);

      // Otherwise, conservatively mark all entry blocks as executable.
    } else {
      LDBG() << "Marking all entry blocks live for op: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      markEntryBlocksLive(op);
    }
  }

  if (isRegionOrCallableReturn(op)) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op->getParentOp())) {
      LDBG() << "Visiting region terminator: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      // Visit the exiting terminator of a region.
      visitRegionTerminator(op, branch);
    } else if (auto callable =
                   dyn_cast<CallableOpInterface>(op->getParentOp())) {
      LDBG() << "Visiting callable terminator: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      // Visit the exiting terminator of a callable.
      visitCallableTerminator(op, callable);
    }
  }
  // Visit the successors.
  if (op->getNumSuccessors()) {
    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      LDBG() << "Visiting branch operation: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      visitBranchOperation(branch);

      // Otherwise, conservatively mark all successors as exectuable.
    } else {
      LDBG() << "Marking all successors live for op: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      for (Block *successor : op->getSuccessors())
        markEdgeLive(op->getBlock(), successor);
    }
  }

  return success();
}

void DeadCodeAnalysis::visitCallOperation(CallOpInterface call) {
  LDBG() << "visitCallOperation: "
         << OpWithFlags(call.getOperation(), OpPrintingFlags().skipRegions());
  Operation *callableOp = call.resolveCallableInTable(&symbolTable);

  // A call to a externally-defined callable has unknown predecessors.
  const auto isExternalCallable = [this](Operation *op) {
    // A callable outside the analysis scope is an external callable.
    if (!analysisScope->isAncestor(op))
      return true;
    // Otherwise, check if the callable region is defined.
    if (auto callable = dyn_cast<CallableOpInterface>(op))
      return !callable.getCallableRegion();
    return false;
  };

  // TODO: Add support for non-symbol callables when necessary. If the
  // callable has non-call uses we would mark as having reached pessimistic
  // fixpoint, otherwise allow for propagating the return values out.
  if (isa_and_nonnull<SymbolOpInterface>(callableOp) &&
      !isExternalCallable(callableOp)) {
    // Add the live callsite.
    auto *callsites =
        getOrCreate<PredecessorState>(getProgramPointAfter(callableOp));
    propagateIfChanged(callsites, callsites->join(call));
    LDBG() << "Added callsite as predecessor for callable: "
           << OpWithFlags(callableOp, OpPrintingFlags().skipRegions());
  } else {
    // Mark this call op's predecessors as overdefined.
    auto *predecessors =
        getOrCreate<PredecessorState>(getProgramPointAfter(call));
    propagateIfChanged(predecessors, predecessors->setHasUnknownPredecessors());
    LDBG() << "Marked call op's predecessors as unknown for: "
           << OpWithFlags(call.getOperation(), OpPrintingFlags().skipRegions());
  }
}

/// Get the constant values of the operands of an operation. If any of the
/// constant value lattices are uninitialized, return std::nullopt to indicate
/// the analysis should bail out.
static std::optional<SmallVector<Attribute>> getOperandValuesImpl(
    Operation *op,
    function_ref<const Lattice<ConstantValue> *(Value)> getLattice) {
  SmallVector<Attribute> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    const Lattice<ConstantValue> *cv = getLattice(operand);
    // If any of the operands' values are uninitialized, bail out.
    if (cv->getValue().isUninitialized())
      return {};
    operands.push_back(cv->getValue().getConstantValue());
  }
  return operands;
}

std::optional<SmallVector<Attribute>>
DeadCodeAnalysis::getOperandValues(Operation *op) {
  return getOperandValuesImpl(op, [&](Value value) {
    auto *lattice = getOrCreate<Lattice<ConstantValue>>(value);
    lattice->useDefSubscribe(this);
    return lattice;
  });
}

void DeadCodeAnalysis::visitBranchOperation(BranchOpInterface branch) {
  LDBG() << "visitBranchOperation: "
         << OpWithFlags(branch.getOperation(), OpPrintingFlags().skipRegions());
  // Try to deduce a single successor for the branch.
  std::optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  if (Block *successor = branch.getSuccessorForOperands(*operands)) {
    markEdgeLive(branch->getBlock(), successor);
    LDBG() << "Branch has single successor: " << successor;
  } else {
    // Otherwise, mark all successors as executable and outgoing edges.
    for (Block *successor : branch->getSuccessors())
      markEdgeLive(branch->getBlock(), successor);
    LDBG() << "Branch has multiple/all successors live";
  }
}

void DeadCodeAnalysis::visitRegionBranchOperation(
    RegionBranchOpInterface branch) {
  LDBG() << "visitRegionBranchOperation: "
         << OpWithFlags(branch.getOperation(), OpPrintingFlags().skipRegions());
  // Try to deduce which regions are executable.
  std::optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getEntrySuccessorRegions(*operands, successors);
  for (const RegionSuccessor &successor : successors) {
    // The successor can be either an entry block or the parent operation.
    ProgramPoint *point =
        successor.getSuccessor()
            ? getProgramPointBefore(&successor.getSuccessor()->front())
            : getProgramPointAfter(branch);
    // Mark the entry block as executable.
    auto *state = getOrCreate<Executable>(point);
    propagateIfChanged(state, state->setToLive());
    LDBG() << "Marked region successor live: " << point;
    // Add the parent op as a predecessor.
    auto *predecessors = getOrCreate<PredecessorState>(point);
    propagateIfChanged(
        predecessors,
        predecessors->join(branch, successor.getSuccessorInputs()));
    LDBG() << "Added region branch as predecessor for successor: " << point;
  }
}

void DeadCodeAnalysis::visitRegionTerminator(Operation *op,
                                             RegionBranchOpInterface branch) {
  LDBG() << "visitRegionTerminator: " << *op;
  std::optional<SmallVector<Attribute>> operands = getOperandValues(op);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op))
    terminator.getSuccessorRegions(*operands, successors);
  else
    branch.getSuccessorRegions(op->getParentRegion(), successors);

  // Mark successor region entry blocks as executable and add this op to the
  // list of predecessors.
  for (const RegionSuccessor &successor : successors) {
    PredecessorState *predecessors;
    if (Region *region = successor.getSuccessor()) {
      auto *state =
          getOrCreate<Executable>(getProgramPointBefore(&region->front()));
      propagateIfChanged(state, state->setToLive());
      LDBG() << "Marked region entry block live for region: " << region;
      predecessors = getOrCreate<PredecessorState>(
          getProgramPointBefore(&region->front()));
    } else {
      // Add this terminator as a predecessor to the parent op.
      predecessors =
          getOrCreate<PredecessorState>(getProgramPointAfter(branch));
    }
    propagateIfChanged(predecessors,
                       predecessors->join(op, successor.getSuccessorInputs()));
    LDBG() << "Added region terminator as predecessor for successor: "
           << (successor.getSuccessor() ? "region entry" : "parent op");
  }
}

void DeadCodeAnalysis::visitCallableTerminator(Operation *op,
                                               CallableOpInterface callable) {
  LDBG() << "visitCallableTerminator: " << *op;
  // Add as predecessors to all callsites this return op.
  auto *callsites = getOrCreateFor<PredecessorState>(
      getProgramPointAfter(op), getProgramPointAfter(callable));
  bool canResolve = op->hasTrait<OpTrait::ReturnLike>();
  for (Operation *predecessor : callsites->getKnownPredecessors()) {
    assert(isa<CallOpInterface>(predecessor));
    auto *predecessors =
        getOrCreate<PredecessorState>(getProgramPointAfter(predecessor));
    if (canResolve) {
      propagateIfChanged(predecessors, predecessors->join(op));
      LDBG() << "Added callable terminator as predecessor for callsite: "
             << OpWithFlags(predecessor, OpPrintingFlags().skipRegions());
    } else {
      // If the terminator is not a return-like, then conservatively assume we
      // can't resolve the predecessor.
      propagateIfChanged(predecessors,
                         predecessors->setHasUnknownPredecessors());
      LDBG() << "Could not resolve callable terminator for callsite: "
             << OpWithFlags(predecessor, OpPrintingFlags().skipRegions());
    }
  }
}
