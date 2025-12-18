//===- ShuffleVectorFMAOps.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

namespace {

static bool validateX86OpsHasOneUser(Value op) {

  if (auto x86Op = op.getDefiningOp<x86vector::CvtPackedEvenIndexedToF32Op>()) {
    if (!x86Op.getResult().hasOneUse())
      return false;
  } else if (auto x86Op = op.getDefiningOp<x86vector::BcstToPackedF32Op>()) {
    if (!x86Op.getResult().hasOneUse())
      return false;
  } else {
    return false;
  }
  return true;
}

static bool validateVectorFMAOp(vector::FMAOp fmaOp) {

  Value lhs = fmaOp.getLhs();
  Value rhs = fmaOp.getRhs();

  if (!isa<x86vector::CvtPackedEvenIndexedToF32Op>(lhs.getDefiningOp()) &&
      !isa<x86vector::CvtPackedEvenIndexedToF32Op>(rhs.getDefiningOp()))
    return false;

  if (!validateX86OpsHasOneUser(fmaOp.getLhs()) ||
      !validateX86OpsHasOneUser(fmaOp.getRhs()))
    return false;

  if (!fmaOp.getResult().hasOneUse())
    return false;

  return true;
}

static void moveFMA(vector::FMAOp fmaOp) {
  Operation *onlyUser = *fmaOp.getResult().getUsers().begin();

  if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(onlyUser)) {
    if (shapeCastOp.getResult().hasOneUse()) {
      onlyUser = *shapeCastOp.getResult().getUsers().begin();
      fmaOp.getLhs().getDefiningOp()->moveBefore(onlyUser);
      fmaOp.getRhs().getDefiningOp()->moveBefore(onlyUser);
      fmaOp->moveBefore(onlyUser);
      shapeCastOp->moveBefore(onlyUser);
      return;
    }
  }

  fmaOp.getLhs().getDefiningOp()->moveBefore(onlyUser);
  fmaOp.getRhs().getDefiningOp()->moveBefore(onlyUser);
  fmaOp->moveBefore(onlyUser);
  return;
}

struct ShuffleVectorFMAOps : public OpRewritePattern<vector::FMAOp> {
  using OpRewritePattern<vector::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fmaOp,
                                PatternRewriter &rewriter) const override {

    if (!validateVectorFMAOp(fmaOp))
      return failure();

    llvm::SmallVector<vector::FMAOp> fmaOps;
    Operation *nextOp = fmaOp;
    bool loopBreak = true;

    while ((nextOp = nextOp->getNextNode())) {
      if (auto fma = dyn_cast<vector::FMAOp>(nextOp)) {
        if (isa<x86vector::CvtPackedEvenIndexedToF32Op>(
                fma.getLhs().getDefiningOp()) ||
            isa<x86vector::CvtPackedEvenIndexedToF32Op>(
                fma.getRhs().getDefiningOp())) {
          if (loopBreak)
            break;
        }

        if (validateVectorFMAOp(fma))
          fmaOps.push_back(fma);

        loopBreak = false;
      }
    }

    if (fmaOps.empty())
      return failure();

    fmaOps.push_back(fmaOp);
    for (size_t i = 0; i < fmaOps.size(); i++) {
      moveFMA(fmaOps[i]);
    }

    return success();
  }
};

} // namespace

void x86vector::populateShuffleVectorFMAOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShuffleVectorFMAOps>(patterns.getContext());
}
