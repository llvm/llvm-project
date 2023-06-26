//===- LivenessAnalysis.cpp - Liveness analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;


void Liveness::print(raw_ostream &os) const {
  os << (isLive ? "live" : "not live");
}

ChangeResult Liveness::markLive() {
  bool wasLive = this->isLive;
  this->isLive = true;
  return wasLive ? ChangeResult::NoChange : ChangeResult::Change;
}

ChangeResult Liveness::meet(const AbstractSparseLattice &other) {
  const auto *otherLiveness = reinterpret_cast<const Liveness *>(&other);
  return otherLiveness->isLive ? markLive() : ChangeResult::NoChange;
}

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

void LivenessAnalysis::backwardFlowLivenessFromResults(
    Operation *op, ArrayRef<Liveness *> operands,
    ArrayRef<const Liveness *> results) {
  bool foundLiveResult = false;
  for (const Liveness *r : results) {
    if (r->isLive && !foundLiveResult) {
      // By default, every result of an op depends on all its operands. Thus, if
      // any result is live, each operand is live.
      for (Liveness *operand : operands)
        meet(operand, *r);

      // TODO(srisrivastava): Enhance this backward flow of liveness. One
      // potential enhancement: If this op exists in a block which is in a
      // region of a region-based control flow op, then mark the non-forwarded
      // operands of that op as "live".

      foundLiveResult = true;
    }
    addDependency(const_cast<Liveness *>(r), op);
  }
  return;
}

void LivenessAnalysis::visitOperation(Operation *op,
                                      ArrayRef<Liveness *> operands,
                                      ArrayRef<const Liveness *> results) {
  // TODO(srisrivastava): Enhance this base case of liveness analysis to make it
  // more accurate.
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    propagateIfChanged(operands[0], operands[0]->markLive());
    return;
  }

  backwardFlowLivenessFromResults(op, operands, results);
  return;
}

void LivenessAnalysis::visitBranchOperand(OpOperand &operand) {
  // The lattices of the non-forwarded branch operands don't get updated like
  // the forwarded branch operands or the non-branch operands. Thus they need
  // to be handled separately. This is where we handle them. The liveness flows
  // backward (or, in other words, the lattices get updated) in such operands by
  // visiting their corresponding branch op (with all its operands).

  Operation *branchOp = operand.getOwner();

  SmallVector<Liveness *, 4> operandsLiveness;
  for (const Value operand : branchOp->getOperands()) {
    operandsLiveness.push_back(getLatticeElement(operand));
  }

  SmallVector<const Liveness *, 4> resultsLiveness;
  for (const Value result : branchOp->getResults()) {
    resultsLiveness.push_back(getLatticeElement(result));
  }

  backwardFlowLivenessFromResults(branchOp, operandsLiveness, resultsLiveness);
  return;
}

void LivenessAnalysis::setToExitState(Liveness *lattice) {
  // Unsure about this but seems like there is nothing to do here, for computing
  // liveness.
}