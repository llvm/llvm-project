//===- UniformityAnalysis.cpp - Uniformity analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "uniformity-analysis"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Control operands
//===----------------------------------------------------------------------===//

/// Marks in `forwarded` the positions of the operands in `range`.
static void markForwarded(OperandRange range, llvm::SmallBitVector &forwarded) {
  if (range.empty())
    return;
  unsigned begin = range.getBeginOperandIndex();
  for (unsigned i = 0, e = range.size(); i < e; ++i)
    forwarded.set(begin + i);
}

void mlir::dataflow::getControlOperands(
    Operation *op, SmallVectorImpl<Value> &controlOperands) {
  llvm::SmallBitVector forwarded(op->getNumOperands());
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (const RegionSuccessor &successor : successors)
      markForwarded(branch.getEntrySuccessorOperands(successor), forwarded);
  } else if (auto terminator =
                 dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    // The successors of a terminator are defined by the region branch that
    // encloses it; a return-like terminator of a callable has none.
    if (!isa<RegionBranchOpInterface>(op->getParentOp()))
      return;
    SmallVector<Attribute> operandAttrs(op->getNumOperands(), nullptr);
    SmallVector<RegionSuccessor> successors;
    terminator.getSuccessorRegions(operandAttrs, successors);
    for (const RegionSuccessor &successor : successors)
      markForwarded(terminator.getSuccessorOperands(successor), forwarded);
  } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    for (unsigned i = 0, e = op->getNumSuccessors(); i < e; ++i)
      markForwarded(branch.getSuccessorOperands(i).getForwardedOperands(),
                    forwarded);
  } else {
    return;
  }
  for (auto [index, operand] : llvm::enumerate(op->getOperands()))
    if (!forwarded.test(index))
      controlOperands.push_back(operand);
}

/// Returns the terminator of `block` if it is one of the branch interfaces
/// that steer control flow, null otherwise.
static Operation *getBranchTerminator(Block &block) {
  if (block.empty())
    return nullptr;
  Operation *terminator = &block.back();
  if (isa<BranchOpInterface, RegionBranchTerminatorOpInterface>(terminator))
    return terminator;
  return nullptr;
}

void mlir::dataflow::getEnclosingControlOperands(
    Operation *op, Region *limit, SmallVectorImpl<Value> &controlOperands) {
  for (Operation *current = op;;) {
    Block *block = current->getBlock();
    if (!block)
      return;
    Region *region = block->getParent();
    // Unstructured control flow within the region.
    if (!block->isEntryBlock())
      for (Block &other : *region)
        if (Operation *terminator = getBranchTerminator(other))
          if (isa<BranchOpInterface>(terminator))
            getControlOperands(terminator, controlOperands);
    if (region == limit)
      return;

    Operation *parent = region->getParentOp();
    if (!parent)
      return;
    // The body of a launch is executed by the threads of the launch, whatever
    // control flow the host wrapped the launch in.
    if (auto inferrable = dyn_cast<InferUniformityOpInterface>(parent);
        inferrable && inferrable.isLaunchBoundary())
      return;
    if (isa<RegionBranchOpInterface>(parent)) {
      getControlOperands(parent, controlOperands);
      for (Region &other : parent->getRegions())
        for (Block &otherBlock : other)
          if (Operation *terminator = getBranchTerminator(otherBlock))
            if (isa<RegionBranchTerminatorOpInterface>(terminator))
              getControlOperands(terminator, controlOperands);
    }
    if (isa<CallableOpInterface>(parent))
      return;
    current = parent;
  }
}

//===----------------------------------------------------------------------===//
// UniformityAnalysis
//===----------------------------------------------------------------------===//

static const StringRef kDefaultTransparentDialects[] = {
    "affine",  "arith", "bufferization", "builtin", "cf",
    "complex", "func",  "index",         "linalg",  "math",
    "memref",  "scf",   "tensor",        "ub",      "vector"};

ArrayRef<StringRef> UniformityAnalysis::getDefaultTransparentDialects() {
  return kDefaultTransparentDialects;
}

UniformityAnalysis::UniformityAnalysis(DataFlowSolver &solver,
                                       ArrayRef<StringRef> transparentDialects)
    : SparseForwardDataFlowAnalysis(solver) {
  for (StringRef dialect : transparentDialects)
    this->transparentDialects.insert(dialect);
}

bool UniformityAnalysis::isTransparent(Operation *op) const {
  return transparentDialects.contains(op->getName().getDialectNamespace());
}

void UniformityAnalysis::setToEntryState(UniformityLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(Uniformity::getDivergent()));
}

LogicalResult UniformityAnalysis::initialize(Operation *top) {
  if (failed(SparseForwardDataFlowAnalysis::initialize(top)))
    return failure();
  // The base class initializes blocks without going through `visit`, so the
  // control dependence of unstructured block arguments is added here.
  top->walk([&](Block *block) { visitUnstructuredBlockArguments(block); });
  return success();
}

LogicalResult UniformityAnalysis::visit(ProgramPoint *point) {
  if (failed(SparseForwardDataFlowAnalysis::visit(point)))
    return failure();
  if (point->isBlockStart())
    visitUnstructuredBlockArguments(point->getBlock());
  return success();
}

llvm::SmallBitVector UniformityAnalysis::inferThroughInterface(
    InferUniformityOpInterface op, ProgramPoint *point, ValueRange candidates,
    ArrayRef<AbstractSparseLattice *> lattices, bool &pending) {
  assert(candidates.size() == lattices.size() && "size mismatch");
  SmallVector<Uniformity> operandUniformity =
      llvm::map_to_vector(op->getOperands(), [&](Value operand) {
        return getLatticeElementFor(point, operand)->getValue();
      });
  // An implementation is allowed to describe nothing while an operand it needs
  // is still uninitialized, and several do. That is not the same as saying the
  // value is divergent: the join is a meet, so a lattice put in the entry state
  // can never recover, and the operand will settle on a later visit.
  pending = llvm::any_of(operandUniformity, [](const Uniformity &uniformity) {
    return uniformity.isUninitialized();
  });
  llvm::SmallBitVector set(lattices.size());
  op.inferUniformity(
      operandUniformity, [&](Value value, UniformityScope scope) {
        auto it = llvm::find(candidates, value);
        if (it == candidates.end())
          return;
        unsigned index = std::distance(candidates.begin(), it);
        set.set(index);
        auto *lattice = static_cast<UniformityLattice *>(lattices[index]);
        LDBG() << "Inferred " << scope << " for " << value;
        propagateIfChanged(lattice, lattice->join(Uniformity(scope)));
      });
  return set;
}

void UniformityAnalysis::setUnsetToEntryStates(
    const llvm::SmallBitVector &set,
    ArrayRef<AbstractSparseLattice *> lattices) {
  for (auto [index, lattice] : llvm::enumerate(lattices))
    if (!set.test(index))
      setToEntryState(static_cast<UniformityLattice *>(lattice));
}

LogicalResult
UniformityAnalysis::visitOperation(Operation *op,
                                   ArrayRef<const UniformityLattice *> operands,
                                   ArrayRef<UniformityLattice *> results) {
  if (auto inferrable = dyn_cast<InferUniformityOpInterface>(op)) {
    SmallVector<AbstractSparseLattice *> resultLattices(results.begin(),
                                                        results.end());
    bool pending = false;
    llvm::SmallBitVector set =
        inferThroughInterface(inferrable, getProgramPointAfter(op),
                              op->getResults(), resultLattices, pending);
    if (!pending)
      setUnsetToEntryStates(set, resultLattices);
    return success();
  }

  // Only a memory-effect-free operation of a transparent dialect that has no
  // region is known to compute a function of its operands. Anything else may
  // read memory written by another thread, thread identity, or a value its
  // region captures from above.
  if (!isTransparent(op) || !isMemoryEffectFree(op) ||
      op->getNumRegions() != 0) {
    setAllToEntryStates(results);
    return success();
  }

  Uniformity joined = Uniformity::getUniform();
  if (!operands.empty()) {
    joined = Uniformity::join(
        llvm::map_to_vector(operands, [](const UniformityLattice *lattice) {
          return lattice->getValue();
        }));
    if (joined.isUninitialized())
      return success();
  }
  LDBG() << "Joined operands to " << joined << " for "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  for (UniformityLattice *lattice : results)
    propagateIfChanged(lattice, lattice->join(joined));
  return success();
}

void UniformityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ValueRange nonSuccessorInputs,
    ArrayRef<UniformityLattice *> nonSuccessorInputLattices) {
  if (auto inferrable = dyn_cast<InferUniformityOpInterface>(op)) {
    ProgramPoint *point =
        !successor.isRegion()
            ? getProgramPointAfter(op)
            : getProgramPointBefore(&successor.getSuccessor()->front());
    SmallVector<AbstractSparseLattice *> lattices(
        nonSuccessorInputLattices.begin(), nonSuccessorInputLattices.end());
    bool pending = false;
    llvm::SmallBitVector set = inferThroughInterface(
        inferrable, point, nonSuccessorInputs, lattices, pending);
    if (pending)
      return;
    if (set.any()) {
      setUnsetToEntryStates(set, lattices);
      return;
    }
  }
  SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, nonSuccessorInputs, nonSuccessorInputLattices);
}

LogicalResult UniformityAnalysis::visitCallOperation(
    CallOpInterface call,
    ArrayRef<const AbstractSparseLattice *> operandLattices,
    ArrayRef<AbstractSparseLattice *> resultLattices) {
  if (failed(SparseForwardDataFlowAnalysis::visitCallOperation(
          call, operandLattices, resultLattices)))
    return failure();
  if (!getSolverConfig().isInterprocedural())
    return success();
  auto callable =
      dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  Region *body = callable ? callable.getCallableRegion() : nullptr;
  if (!body)
    return success();
  ProgramPoint *point = getProgramPointAfter(call);
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  if (!predecessors->allPredecessorsKnown())
    return success();

  // The base class joins what every return site yields. Threads that leave
  // the callee through different return sites observe different results, so
  // the results also depend on the control flow that selects the return site.
  SmallVector<Value> controlOperands;
  for (Operation *returnSite : predecessors->getKnownPredecessors())
    getEnclosingControlOperands(returnSite, body, controlOperands);
  Uniformity taint;
  for (Value operand : controlOperands)
    taint = Uniformity::join(taint,
                             getLatticeElementFor(point, operand)->getValue());
  if (taint.isUninitialized())
    return success();
  LDBG() << "Return site control dependence " << taint << " for "
         << OpWithFlags(call, OpPrintingFlags().skipRegions());
  for (AbstractSparseLattice *lattice : resultLattices) {
    auto *typed = static_cast<UniformityLattice *>(lattice);
    propagateIfChanged(typed, typed->join(taint));
  }
  return success();
}

void UniformityAnalysis::visitCallableOperation(
    CallableOpInterface callable,
    ArrayRef<AbstractSparseLattice *> argLattices) {
  auto inferrable =
      dyn_cast<InferUniformityOpInterface>(callable.getOperation());
  Region *body = callable.getCallableRegion();
  if (inferrable && body && !body->empty()) {
    Block &entry = body->front();
    bool pending = false;
    llvm::SmallBitVector set =
        inferThroughInterface(inferrable, getProgramPointBefore(&entry),
                              entry.getArguments(), argLattices, pending);
    if (pending)
      return;
    if (set.any()) {
      setUnsetToEntryStates(set, argLattices);
      return;
    }
  }
  AbstractSparseForwardDataFlowAnalysis::visitCallableOperation(callable,
                                                                argLattices);
}

void UniformityAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionSuccessor successor, ArrayRef<AbstractSparseLattice *> lattices) {
  Operation *op = branch.getOperation();
  if (auto inferrable = dyn_cast<InferUniformityOpInterface>(op)) {
    // The operation is authoritative for this successor if it sets one of
    // the inputs that region control flow would otherwise forward a value to.
    // Setting only a non-successor input (a loop induction variable) leaves
    // the forwarded ones to the usual join.
    ValueRange candidates = point->isBlockStart()
                                ? ValueRange(point->getBlock()->getArguments())
                                : ValueRange(op->getResults());
    bool pending = false;
    llvm::SmallBitVector set =
        inferThroughInterface(inferrable, point, candidates, lattices, pending);
    if (pending)
      return;
    ValueRange successorInputs = branch.getSuccessorInputs(successor);
    bool authoritative = false;
    for (auto [index, candidate] : llvm::enumerate(candidates))
      if (set.test(index) && llvm::is_contained(successorInputs, candidate))
        authoritative = true;
    if (authoritative) {
      setUnsetToEntryStates(set, lattices);
      return;
    }
  } else if (!isTransparent(op)) {
    // A region operation of an unknown dialect may run its regions with any
    // subset of threads, and its results may come from anywhere. (The typed
    // overload of setAllToEntryStates hides the untyped one.)
    AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(lattices);
    return;
  }

  AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors(
      point, branch, successor, lattices);
  // Threads that take different paths through the regions reach the results
  // with different values; the entry block arguments of a region are only
  // observed by the threads that entered it together.
  if (!point->isBlockStart())
    joinControlDependence(point, op, lattices);
}

void UniformityAnalysis::joinControlDependence(
    ProgramPoint *point, Operation *branch,
    ArrayRef<AbstractSparseLattice *> lattices) {
  SmallVector<Value> controlOperands;
  getControlOperands(branch, controlOperands);
  for (Region &region : branch->getRegions())
    for (Block &block : region)
      if (Operation *terminator = getBranchTerminator(block))
        getControlOperands(terminator, controlOperands);

  Uniformity taint;
  for (Value operand : controlOperands)
    taint = Uniformity::join(taint,
                             getLatticeElementFor(point, operand)->getValue());
  if (taint.isUninitialized())
    return;
  LDBG() << "Control dependence " << taint << " for "
         << OpWithFlags(branch, OpPrintingFlags().skipRegions());
  for (AbstractSparseLattice *lattice : lattices) {
    auto *typed = static_cast<UniformityLattice *>(lattice);
    propagateIfChanged(typed, typed->join(taint));
  }
}

void UniformityAnalysis::visitUnstructuredBlockArguments(Block *block) {
  if (block->isEntryBlock() || block->getNumArguments() == 0)
    return;
  ProgramPoint *point = getProgramPointBefore(block);
  SmallVector<Value> controlOperands;
  for (Block &other : *block->getParent())
    if (Operation *terminator = getBranchTerminator(other))
      if (isa<BranchOpInterface>(terminator))
        getControlOperands(terminator, controlOperands);

  Uniformity taint;
  for (Value operand : controlOperands)
    taint = Uniformity::join(taint,
                             getLatticeElementFor(point, operand)->getValue());
  if (taint.isUninitialized())
    return;
  for (BlockArgument argument : block->getArguments()) {
    UniformityLattice *lattice = getLatticeElement(argument);
    propagateIfChanged(lattice, lattice->join(taint));
  }
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

UniformityScope mlir::dataflow::getUniformity(DataFlowSolver &solver,
                                              Value value) {
  const auto *lattice = solver.lookupState<UniformityLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized())
    return UniformityScope::Divergent;
  return lattice->getValue().getScope();
}

UniformityScope
mlir::dataflow::getExecutionUniformity(DataFlowSolver &solver, Operation *op,
                                       Value *narrowingOperand) {
  SmallVector<Value> controlOperands;
  getEnclosingControlOperands(op, /*limit=*/nullptr, controlOperands);
  UniformityScope scope = UniformityScope::Uniform;
  Value narrowing;
  for (Value operand : controlOperands) {
    UniformityScope operandScope = getUniformity(solver, operand);
    if (operandScope < scope) {
      scope = operandScope;
      narrowing = operand;
    }
  }
  if (narrowingOperand)
    *narrowingOperand = narrowing;
  return scope;
}
