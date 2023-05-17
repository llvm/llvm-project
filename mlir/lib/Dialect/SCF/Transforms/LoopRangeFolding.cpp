//===- LoopRangeFolding.cpp - Code to perform loop range folding-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop range folding.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORLOOPRANGEFOLDING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {
struct ForLoopRangeFolding
    : public impl::SCFForLoopRangeFoldingBase<ForLoopRangeFolding> {
  void runOnOperation() override;
};
} // namespace

void ForLoopRangeFolding::runOnOperation() {
  getOperation()->walk([&](ForOp op) {
    Value indVar = op.getInductionVar();

    auto canBeFolded = [&](Value value) {
      return op.isDefinedOutsideOfLoop(value) || value == indVar;
    };

    // Fold until a fixed point is reached
    while (true) {

      // If the induction variable is used more than once, we can't fold its
      // arith ops into the loop range
      if (!indVar.hasOneUse())
        break;

      Operation *user = *indVar.getUsers().begin();
      if (!isa<arith::AddIOp, arith::MulIOp>(user))
        break;

      if (!llvm::all_of(user->getOperands(), canBeFolded))
        break;

      OpBuilder b(op);
      IRMapping lbMap;
      lbMap.map(indVar, op.getLowerBound());
      IRMapping ubMap;
      ubMap.map(indVar, op.getUpperBound());
      IRMapping stepMap;
      stepMap.map(indVar, op.getStep());

      if (isa<arith::AddIOp>(user)) {
        Operation *lbFold = b.clone(*user, lbMap);
        Operation *ubFold = b.clone(*user, ubMap);

        op.setLowerBound(lbFold->getResult(0));
        op.setUpperBound(ubFold->getResult(0));

      } else if (isa<arith::MulIOp>(user)) {
        Operation *ubFold = b.clone(*user, ubMap);
        Operation *stepFold = b.clone(*user, stepMap);

        op.setUpperBound(ubFold->getResult(0));
        op.setStep(stepFold->getResult(0));
      }

      ValueRange wrapIndvar(indVar);
      user->replaceAllUsesWith(wrapIndvar);
      user->erase();
    }
  });
}

std::unique_ptr<Pass> mlir::createForLoopRangeFoldingPass() {
  return std::make_unique<ForLoopRangeFolding>();
}
