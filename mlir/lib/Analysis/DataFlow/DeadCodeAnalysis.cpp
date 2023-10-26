//===- DeadCodeAnalysis.cpp - Dead code analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <optional>

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

  if (auto *block = llvm::dyn_cast_if_present<Block *>(point)) {
    // Re-invoke the analyses on the block itself.
    for (DataFlowAnalysis *analysis : subscribers)
      solver->enqueue({block, analysis});
    // Re-invoke the analyses on all operations in the block.
    for (DataFlowAnalysis *analysis : subscribers)
      for (Operation &op : *block)
        solver->enqueue({&op, analysis});
  } else if (auto *programPoint = llvm::dyn_cast_if_present<GenericProgramPoint *>(point)) {
    // Re-invoke the analysis on the successor block.
    if (auto *edge = dyn_cast<CFGEdge>(programPoint)) {
      for (DataFlowAnalysis *analysis : subscribers)
        solver->enqueue({edge->getTo(), analysis});
    }
  }
}

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

void PredecessorState::print(raw_ostream &os) const {
  if (allPredecessorsKnown())
    os << "(all) ";
  os << "predecessors:\n";
  for (Operation *op : getKnownPredecessors())
    os << "  " << *op << "\n";
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
  registerPointKind<CFGEdge>();
}

LogicalResult DeadCodeAnalysis::initialize(Operation *top) {
  // Mark the top-level blocks as executable.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Executable>(&region.front());
    propagateIfChanged(state, state->setToLive());
  }

  // Mark as overdefined the predecessors of symbol callables with potentially
  // unknown predecessors.
  initializeSymbolCallables(top);

  return initializeRecursively(top);
}

void DeadCodeAnalysis::initializeSymbolCallables(Operation *top) {
  analysisScope = top;
  auto walkFn = [&](Operation *symTable, bool allUsesVisible) {
    Region &symbolTableRegion = symTable->getRegion(0);
    Block *symbolTableBlock = &symbolTableRegion.front();

    bool foundSymbolCallable = false;
    for (auto callable : symbolTableBlock->getOps<CallableOpInterface>()) {
      Region *callableRegion = callable.getCallableRegion();
      if (!callableRegion)
        continue;
      auto symbol = dyn_cast<SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;

      // Public symbol callables or those for which we can't see all uses have
      // potentially unknown callsites.
      if (symbol.isPublic() || (!allUsesVisible && symbol.isNested())) {
        auto *state = getOrCreate<PredecessorState>(callable);
        propagateIfChanged(state, state->setHasUnknownPredecessors());
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
      return top->walk([&](CallableOpInterface callable) {
        auto *state = getOrCreate<PredecessorState>(callable);
        propagateIfChanged(state, state->setHasUnknownPredecessors());
      });
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      if (isa<CallOpInterface>(use.getUser()))
        continue;
      // If a callable symbol has a non-call use, then we can't be guaranteed to
      // know all callsites.
      Operation *symbol = symbolTable.lookupSymbolIn(top, use.getSymbolRef());
      auto *state = getOrCreate<PredecessorState>(symbol);
      propagateIfChanged(state, state->setHasUnknownPredecessors());
    }
  };
  SymbolTable::walkSymbolTables(top, /*allSymUsesVisible=*/!top->getBlock(),
                                walkFn);
}

/// Returns true if the operation is a returning terminator in region
/// control-flow or the terminator of a callable region.
static bool isRegionOrCallableReturn(Operation *op) {
  return !op->getNumSuccessors() &&
         isa<RegionBranchOpInterface, CallableOpInterface>(op->getParentOp()) &&
         op->getBlock()->getTerminator() == op;
}

LogicalResult DeadCodeAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every op with control-flow semantics.
  if (op->getNumRegions() || op->getNumSuccessors() ||
      isRegionOrCallableReturn(op) || isa<CallOpInterface>(op)) {
    // When the liveness of the parent block changes, make sure to re-invoke the
    // analysis on the op.
    if (op->getBlock())
      getOrCreate<Executable>(op->getBlock())->blockContentSubscribe(this);
    // Visit the op.
    if (failed(visit(op)))
      return failure();
  }
  // Recurse on nested operations.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (failed(initializeRecursively(&op)))
        return failure();
  return success();
}

void DeadCodeAnalysis::markEdgeLive(Block *from, Block *to) {
  auto *state = getOrCreate<Executable>(to);
  propagateIfChanged(state, state->setToLive());
  auto *edgeState = getOrCreate<Executable>(getProgramPoint<CFGEdge>(from, to));
  propagateIfChanged(edgeState, edgeState->setToLive());
}

void DeadCodeAnalysis::markEntryBlocksLive(Operation *op) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Executable>(&region.front());
    propagateIfChanged(state, state->setToLive());
  }
}

LogicalResult DeadCodeAnalysis::visit(ProgramPoint point) {
  if (point.is<Block *>())
    return success();
  auto *op = llvm::dyn_cast_if_present<Operation *>(point);
  if (!op)
    return emitError(point.getLoc(), "unknown program point kind");

  // If the parent block is not executable, there is nothing to do.
  if (!getOrCreate<Executable>(op->getBlock())->isLive())
    return success();

  // We have a live call op. Add this as a live predecessor of the callee.
  if (auto call = dyn_cast<CallOpInterface>(op))
    visitCallOperation(call);

  // Visit the regions.
  if (op->getNumRegions()) {
    // Check if we can reason about the region control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      visitRegionBranchOperation(branch);

      // Check if this is a callable operation.
    } else if (auto callable = dyn_cast<CallableOpInterface>(op)) {
      const auto *callsites = getOrCreateFor<PredecessorState>(op, callable);

      // If the callsites could not be resolved or are known to be non-empty,
      // mark the callable as executable.
      if (!callsites->allPredecessorsKnown() ||
          !callsites->getKnownPredecessors().empty())
        markEntryBlocksLive(callable);

      // Otherwise, conservatively mark all entry blocks as executable.
    } else {
      markEntryBlocksLive(op);
    }
  }

  if (isRegionOrCallableReturn(op)) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a region.
      visitRegionTerminator(op, branch);
    } else if (auto callable =
                   dyn_cast<CallableOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a callable.
      visitCallableTerminator(op, callable);
    }
  }
  // Visit the successors.
  if (op->getNumSuccessors()) {
    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      visitBranchOperation(branch);

      // Otherwise, conservatively mark all successors as exectuable.
    } else {
      for (Block *successor : op->getSuccessors())
        markEdgeLive(op->getBlock(), successor);
    }
  }

  return success();
}

void DeadCodeAnalysis::visitCallOperation(CallOpInterface call) {
  Operation *callableOp = call.resolveCallable(&symbolTable);

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
    auto *callsites = getOrCreate<PredecessorState>(callableOp);
    propagateIfChanged(callsites, callsites->join(call));
  } else {
    // Mark this call op's predecessors as overdefined.
    auto *predecessors = getOrCreate<PredecessorState>(call);
    propagateIfChanged(predecessors, predecessors->setHasUnknownPredecessors());
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
  // Try to deduce a single successor for the branch.
  std::optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  if (Block *successor = branch.getSuccessorForOperands(*operands)) {
    markEdgeLive(branch->getBlock(), successor);
  } else {
    // Otherwise, mark all successors as executable and outgoing edges.
    for (Block *successor : branch->getSuccessors())
      markEdgeLive(branch->getBlock(), successor);
  }
}

void DeadCodeAnalysis::visitRegionBranchOperation(
    RegionBranchOpInterface branch) {
  // Try to deduce which regions are executable.
  std::optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getEntrySuccessorRegions(*operands, successors);
  for (const RegionSuccessor &successor : successors) {
    // The successor can be either an entry block or the parent operation.
    ProgramPoint point = successor.getSuccessor()
                             ? &successor.getSuccessor()->front()
                             : ProgramPoint(branch);
    // Mark the entry block as executable.
    auto *state = getOrCreate<Executable>(point);
    propagateIfChanged(state, state->setToLive());
    // Add the parent op as a predecessor.
    auto *predecessors = getOrCreate<PredecessorState>(point);
    propagateIfChanged(
        predecessors,
        predecessors->join(branch, successor.getSuccessorInputs()));
  }
}

void DeadCodeAnalysis::visitRegionTerminator(Operation *op,
                                             RegionBranchOpInterface branch) {
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
      auto *state = getOrCreate<Executable>(&region->front());
      propagateIfChanged(state, state->setToLive());
      predecessors = getOrCreate<PredecessorState>(&region->front());
    } else {
      // Add this terminator as a predecessor to the parent op.
      predecessors = getOrCreate<PredecessorState>(branch);
    }
    propagateIfChanged(predecessors,
                       predecessors->join(op, successor.getSuccessorInputs()));
  }
}

void DeadCodeAnalysis::visitCallableTerminator(Operation *op,
                                               CallableOpInterface callable) {
  // Add as predecessors to all callsites this return op.
  auto *callsites = getOrCreateFor<PredecessorState>(op, callable);
  bool canResolve = op->hasTrait<OpTrait::ReturnLike>();
  for (Operation *predecessor : callsites->getKnownPredecessors()) {
    assert(isa<CallOpInterface>(predecessor));
    auto *predecessors = getOrCreate<PredecessorState>(predecessor);
    if (canResolve) {
      propagateIfChanged(predecessors, predecessors->join(op));
    } else {
      // If the terminator is not a return-like, then conservatively assume we
      // can't resolve the predecessor.
      propagateIfChanged(predecessors,
                         predecessors->setHasUnknownPredecessors());
    }
  }
}
