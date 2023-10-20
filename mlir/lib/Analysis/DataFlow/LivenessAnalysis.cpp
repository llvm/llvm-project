//===- LivenessAnalysis.cpp - Liveness analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SymbolTable.h"
#include <cassert>
#include <mlir/Analysis/DataFlow/LivenessAnalysis.h>

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Liveness
//===----------------------------------------------------------------------===//

void Liveness::print(raw_ostream &os) const {
  os << (isLive ? "live" : "not live");
}

ChangeResult Liveness::markLive() {
  bool wasLive = isLive;
  isLive = true;
  return wasLive ? ChangeResult::NoChange : ChangeResult::Change;
}

ChangeResult Liveness::meet(const AbstractSparseLattice &other) {
  const auto *otherLiveness = reinterpret_cast<const Liveness *>(&other);
  return otherLiveness->isLive ? markLive() : ChangeResult::NoChange;
}

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// For every value, liveness analysis determines whether or not it is "live".
///
/// A value is considered "live" iff it:
///   (1) has memory effects OR
///   (2) is returned by a public function OR
///   (3) is used to compute a value of type (1) or (2).
/// It is also to be noted that a value could be of multiple types (1/2/3) at
/// the same time.
///
/// A value "has memory effects" iff it:
///   (1.a) is an operand of an op with memory effects OR
///   (1.b) is a non-forwarded branch operand and its branch op could take the
///   control to a block that has an op with memory effects OR
///   (1.c) is a non-forwarded call operand.
///
/// A value `A` is said to be "used to compute" value `B` iff `B` cannot be
/// computed in the absence of `A`. Thus, in this implementation, we say that
/// value `A` is used to compute value `B` iff:
///   (3.a) `B` is a result of an op with operand `A` OR
///   (3.b) `A` is used to compute some value `C` and `C` is used to compute
///   `B`.

void LivenessAnalysis::visitOperation(Operation *op,
                                      ArrayRef<Liveness *> operands,
                                      ArrayRef<const Liveness *> results) {
  // This marks values of type (1.a) liveness as "live".
  if (!isMemoryEffectFree(op)) {
    for (auto *operand : operands)
      propagateIfChanged(operand, operand->markLive());
  }

  // This marks values of type (3) liveness as "live".
  bool foundLiveResult = false;
  for (const Liveness *r : results) {
    if (r->isLive && !foundLiveResult) {
      // It is assumed that each operand is used to compute each result of an
      // op. Thus, if at least one result is live, each operand is live.
      for (Liveness *operand : operands)
        meet(operand, *r);
      foundLiveResult = true;
    }
    addDependency(const_cast<Liveness *>(r), op);
  }
}

void LivenessAnalysis::visitBranchOperand(OpOperand &operand) {
  // We know (at the moment) and assume (for the future) that `operand` is a
  // non-forwarded branch operand of a `RegionBranchOpInterface`,
  // `BranchOpInterface`, `RegionBranchTerminatorOpInterface` or return-like op.
  Operation *op = operand.getOwner();
  assert((isa<RegionBranchOpInterface>(op) || isa<BranchOpInterface>(op) ||
          isa<RegionBranchTerminatorOpInterface>(op)) &&
         "expected the op to be `RegionBranchOpInterface`, "
         "`BranchOpInterface` or `RegionBranchTerminatorOpInterface`");

  // The lattices of the non-forwarded branch operands don't get updated like
  // the forwarded branch operands or the non-branch operands. Thus they need
  // to be handled separately. This is where we handle them.

  // This marks values of type (1.b) liveness as "live". A non-forwarded
  // branch operand will be live if a block where its op could take the control
  // has an op with memory effects.
  // Populating such blocks in `blocks`.
  SmallVector<Block *, 4> blocks;
  if (isa<RegionBranchOpInterface>(op)) {
    // When the op is a `RegionBranchOpInterface`, like an `scf.for` or an
    // `scf.index_switch` op, its branch operand controls the flow into this
    // op's regions.
    for (Region &region : op->getRegions()) {
      for (Block &block : region)
        blocks.push_back(&block);
    }
  } else if (isa<BranchOpInterface>(op)) {
    // When the op is a `BranchOpInterface`, like a `cf.cond_br` or a
    // `cf.switch` op, its branch operand controls the flow into this op's
    // successors.
    blocks = op->getSuccessors();
  } else {
    // When the op is a `RegionBranchTerminatorOpInterface`, like an
    // `scf.condition` op or return-like, like an `scf.yield` op, its branch
    // operand controls the flow into this op's parent's (which is a
    // `RegionBranchOpInterface`'s) regions.
    Operation *parentOp = op->getParentOp();
    assert(isa<RegionBranchOpInterface>(parentOp) &&
           "expected parent op to implement `RegionBranchOpInterface`");
    for (Region &region : parentOp->getRegions()) {
      for (Block &block : region)
        blocks.push_back(&block);
    }
  }
  bool foundMemoryEffectingOp = false;
  for (Block *block : blocks) {
    if (foundMemoryEffectingOp)
      break;
    for (Operation &nestedOp : *block) {
      if (!isMemoryEffectFree(&nestedOp)) {
        Liveness *operandLiveness = getLatticeElement(operand.get());
        propagateIfChanged(operandLiveness, operandLiveness->markLive());
        foundMemoryEffectingOp = true;
        break;
      }
    }
  }

  // Now that we have checked for memory-effecting ops in the blocks of concern,
  // we will simply visit the op with this non-forwarded operand to potentially
  // mark it "live" due to type (1.a/3) liveness.
  SmallVector<Liveness *, 4> operandLiveness;
  operandLiveness.push_back(getLatticeElement(operand.get()));
  SmallVector<const Liveness *, 4> resultsLiveness;
  for (const Value result : op->getResults())
    resultsLiveness.push_back(getLatticeElement(result));
  visitOperation(op, operandLiveness, resultsLiveness);

  // We also visit the parent op with the parent's results and this operand if
  // `op` is a `RegionBranchTerminatorOpInterface` because its non-forwarded
  // operand depends on not only its memory effects/results but also on those of
  // its parent's.
  if (!isa<RegionBranchTerminatorOpInterface>(op))
    return;
  Operation *parentOp = op->getParentOp();
  SmallVector<const Liveness *, 4> parentResultsLiveness;
  for (const Value parentResult : parentOp->getResults())
    parentResultsLiveness.push_back(getLatticeElement(parentResult));
  visitOperation(parentOp, operandLiveness, parentResultsLiveness);
}

void LivenessAnalysis::visitCallOperand(OpOperand &operand) {
  // We know (at the moment) and assume (for the future) that `operand` is a
  // non-forwarded call operand of an op implementing `CallOpInterface`.
  assert(isa<CallOpInterface>(operand.getOwner()) &&
         "expected the op to implement `CallOpInterface`");

  // The lattices of the non-forwarded call operands don't get updated like the
  // forwarded call operands or the non-call operands. Thus they need to be
  // handled separately. This is where we handle them.

  // This marks values of type (1.c) liveness as "live". A non-forwarded
  // call operand is live.
  Liveness *operandLiveness = getLatticeElement(operand.get());
  propagateIfChanged(operandLiveness, operandLiveness->markLive());
}

void LivenessAnalysis::setToExitState(Liveness *lattice) {
  // This marks values of type (2) liveness as "live".
  lattice->markLive();
}

//===----------------------------------------------------------------------===//
// RunLivenessAnalysis
//===----------------------------------------------------------------------===//

RunLivenessAnalysis::RunLivenessAnalysis(Operation *op) {
  SymbolTableCollection symbolTable;

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<LivenessAnalysis>(symbolTable);
  (void)solver.initializeAndRun(op);
}

const Liveness *RunLivenessAnalysis::getLiveness(Value val) {
  return solver.lookupState<Liveness>(val);
}
