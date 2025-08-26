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

#include <llvm/Support/DebugLog.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlow/Utils.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "liveness-analysis"

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
///   (3) is used to compute a value of type (1) or (2) OR
///   (4) is returned by a return-like op whose parent isn't a callable
///       nor a RegionBranchOpInterface (e.g.: linalg.yield, gpu.yield,...)
///       These ops have their own semantics, so we conservatively mark the
///       the yield value as live.
/// It is also to be noted that a value could be of multiple types (1/2/3) at
/// the same time.
///
/// A value "has memory effects" iff it:
///   (1.a) is an operand of an op with memory effects OR
///   (1.b) is a non-forwarded branch operand and its branch op could take the
///   control to a block that has an op with memory effects OR
///   (1.c) is a non-forwarded branch operand and its branch op could result
///   in different live result OR
///   (1.d) is a non-forwarded call operand.
///
/// A value `A` is said to be "used to compute" value `B` iff `B` cannot be
/// computed in the absence of `A`. Thus, in this implementation, we say that
/// value `A` is used to compute value `B` iff:
///   (3.a) `B` is a result of an op with operand `A` OR
///   (3.b) `A` is used to compute some value `C` and `C` is used to compute
///   `B`.

LogicalResult
LivenessAnalysis::visitOperation(Operation *op, ArrayRef<Liveness *> operands,
                                 ArrayRef<const Liveness *> results) {
  LDBG() << "[visitOperation] Enter: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  // This marks values of type (1.a) and (4) liveness as "live".
  if (!isMemoryEffectFree(op) || op->hasTrait<OpTrait::ReturnLike>()) {
    LDBG() << "[visitOperation] Operation has memory effects or is "
              "return-like, marking operands live";
    for (auto *operand : operands) {
      LDBG() << " [visitOperation] Marking operand live: " << operand << " ("
             << operand->isLive << ")";
      propagateIfChanged(operand, operand->markLive());
    }
  }

  // This marks values of type (3) liveness as "live".
  bool foundLiveResult = false;
  for (const Liveness *r : results) {
    if (r->isLive && !foundLiveResult) {
      LDBG() << "[visitOperation] Found live result, "
                "meeting all operands with result: "
             << r;
      // It is assumed that each operand is used to compute each result of an
      // op. Thus, if at least one result is live, each operand is live.
      for (Liveness *operand : operands) {
        LDBG() << " [visitOperation] Meeting operand: " << operand
               << " with result: " << r;
        meet(operand, *r);
      }
      foundLiveResult = true;
    }
    LDBG() << "[visitOperation] Adding dependency for result: " << r
           << " after op: " << *op;
    addDependency(const_cast<Liveness *>(r), getProgramPointAfter(op));
  }
  return success();
}

void LivenessAnalysis::visitBranchOperand(OpOperand &operand) {
  LDBG() << "Visiting branch operand: " << operand.get()
         << " in op: " << *operand.getOwner();
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

  // This marks values of type (1.b/1.c) liveness as "live". A non-forwarded
  // branch operand will be live if a block where its op could take the control
  // has an op with memory effects or could result in different results.
  // Populating such blocks in `blocks`.
  bool mayLive = false;
  SmallVector<Block *, 4> blocks;
  if (isa<RegionBranchOpInterface>(op)) {
    if (op->getNumResults() != 0) {
      // This mark value of type 1.c liveness as may live, because the region
      // branch operation has a return value, and the non-forwarded operand can
      // determine the region to jump to, it can thereby control the result of
      // the region branch operation.
      // Therefore, if the result value is live, we conservatively consider the
      // non-forwarded operand of the region branch operation with result may
      // live and record all result.
      for (Value result : op->getResults()) {
        if (getLatticeElement(result)->isLive) {
          mayLive = true;
          LDBG() << "[visitBranchOperand] Non-forwarded branch "
                    "operand may be live due to live result: "
                 << result;
          break;
        }
      }
    } else {
      // When the op is a `RegionBranchOpInterface`, like an `scf.for` or an
      // `scf.index_switch` op, its branch operand controls the flow into this
      // op's regions.
      for (Region &region : op->getRegions()) {
        for (Block &block : region)
          blocks.push_back(&block);
      }
    }
  } else if (isa<BranchOpInterface>(op)) {
    // We cannot track all successor blocks of the branch operation(More
    // specifically, it's the successor's successor). Additionally, different
    // blocks might also lead to the different block argument described in 1.c.
    // Therefore, we conservatively consider the non-forwarded operand of the
    // branch operation may live.
    mayLive = true;
    LDBG() << "[visitBranchOperand] Non-forwarded branch operand may "
              "be live due to branch op interface";
  } else {
    Operation *parentOp = op->getParentOp();
    assert(isa<RegionBranchOpInterface>(parentOp) &&
           "expected parent op to implement `RegionBranchOpInterface`");
    if (parentOp->getNumResults() != 0) {
      // This mark value of type 1.c liveness as may live, because the region
      // branch operation has a return value, and the non-forwarded operand can
      // determine the region to jump to, it can thereby control the result of
      // the region branch operation.
      // Therefore, if the result value is live, we conservatively consider the
      // non-forwarded operand of the region branch operation with result may
      // live and record all result.
      for (Value result : parentOp->getResults()) {
        if (getLatticeElement(result)->isLive) {
          mayLive = true;
          LDBG() << "[visitBranchOperand] Non-forwarded branch "
                    "operand may be live due to parent live result: "
                 << result;
          break;
        }
      }
    } else {
      // When the op is a `RegionBranchTerminatorOpInterface`, like an
      // `scf.condition` op or return-like, like an `scf.yield` op, its branch
      // operand controls the flow into this op's parent's (which is a
      // `RegionBranchOpInterface`'s) regions.
      for (Region &region : parentOp->getRegions()) {
        for (Block &block : region)
          blocks.push_back(&block);
      }
    }
  }
  for (Block *block : blocks) {
    if (mayLive)
      break;
    for (Operation &nestedOp : *block) {
      if (!isMemoryEffectFree(&nestedOp)) {
        mayLive = true;
        LDBG() << "Non-forwarded branch operand may be "
                  "live due to memory effect in block: "
               << block;
        break;
      }
    }
  }

  if (mayLive) {
    Liveness *operandLiveness = getLatticeElement(operand.get());
    LDBG() << "Marking branch operand live: " << operand.get();
    propagateIfChanged(operandLiveness, operandLiveness->markLive());
  }

  // Now that we have checked for memory-effecting ops in the blocks of concern,
  // we will simply visit the op with this non-forwarded operand to potentially
  // mark it "live" due to type (1.a/3) liveness.
  SmallVector<Liveness *, 4> operandLiveness;
  operandLiveness.push_back(getLatticeElement(operand.get()));
  SmallVector<const Liveness *, 4> resultsLiveness;
  for (const Value result : op->getResults())
    resultsLiveness.push_back(getLatticeElement(result));
  LDBG() << "Visiting operation for non-forwarded branch operand: " << *op;
  (void)visitOperation(op, operandLiveness, resultsLiveness);

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
  LDBG() << "Visiting parent operation for non-forwarded branch operand: "
         << *parentOp;
  (void)visitOperation(parentOp, operandLiveness, parentResultsLiveness);
}

void LivenessAnalysis::visitCallOperand(OpOperand &operand) {
  LDBG() << "Visiting call operand: " << operand.get()
         << " in op: " << *operand.getOwner();
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
  LDBG() << "Marking call operand live: " << operand.get();
  propagateIfChanged(operandLiveness, operandLiveness->markLive());
}

void LivenessAnalysis::setToExitState(Liveness *lattice) {
  LDBG() << "setToExitState for lattice: " << lattice;
  if (lattice->isLive) {
    LDBG() << "Lattice already live, nothing to do";
    return;
  }
  // This marks values of type (2) liveness as "live".
  LDBG() << "Marking lattice live due to exit state";
  (void)lattice->markLive();
  propagateIfChanged(lattice, ChangeResult::Change);
}

//===----------------------------------------------------------------------===//
// RunLivenessAnalysis
//===----------------------------------------------------------------------===//

RunLivenessAnalysis::RunLivenessAnalysis(Operation *op) {
  LDBG() << "Constructing RunLivenessAnalysis for op: " << op->getName();
  SymbolTableCollection symbolTable;

  loadBaselineAnalyses(solver);
  solver.load<LivenessAnalysis>(symbolTable);
  LDBG() << "Initializing and running solver";
  (void)solver.initializeAndRun(op);
  LDBG() << "RunLivenessAnalysis initialized for op: " << op->getName()
         << " check on unreachable code now:";
  // The framework doesn't visit operations in dead blocks, so we need to
  // explicitly mark them as dead.
  op->walk([&](Operation *op) {
    if (op->getNumResults() == 0)
      return;
    for (auto result : llvm::enumerate(op->getResults())) {
      if (getLiveness(result.value()))
        continue;
      LDBG() << "Result: " << result.index() << " of "
             << OpWithFlags(op, OpPrintingFlags().skipRegions())
             << " has no liveness info (unreachable), mark dead";
      solver.getOrCreateState<Liveness>(result.value());
    }
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto blockArg : llvm::enumerate(block.getArguments())) {
          if (getLiveness(blockArg.value()))
            continue;
          LDBG() << "Block argument: " << blockArg.index() << " of "
                 << OpWithFlags(op, OpPrintingFlags().skipRegions())
                 << " has no liveness info, mark dead";
          solver.getOrCreateState<Liveness>(blockArg.value());
        }
      }
    }
  });
}

const Liveness *RunLivenessAnalysis::getLiveness(Value val) {
  return solver.lookupState<Liveness>(val);
}
