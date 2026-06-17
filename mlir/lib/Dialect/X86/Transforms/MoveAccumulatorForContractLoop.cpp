//===- MoveAccumulatorForContractLoop.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/Dialect/X86/X86Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86;

namespace {
// Transforms vector.contract(A, B, Acc) into vector.contract(A, B, 0) + Acc
// to decouple the contraction computation from the accumulator update.
struct MoveAccumulatorForContractLoop
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    Operation *accReadOp =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());

    Value contractValue = contractionUsersAfterYield(contractOp.getResult());

    if (!contractValue)
      return rewriter.notifyMatchFailure(
          contractOp, "Final acc write might have multiple users.");

    Operation *resultWriteOp = *contractValue.getUsers().begin();

    if (!accReadOp || !resultWriteOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Read/write from/to acc matrix is not by "
                      "transfer_read/load/transfer_write/store ops.");

    if (dyn_cast<arith::ConstantOp>(accReadOp))
      return rewriter.notifyMatchFailure(
          contractOp,
          "The input acc to contract is already a constant zero vector.");
    ;

    if ((accReadOp->getBlock() == contractOp->getBlock()) ||
        (resultWriteOp->getBlock() == contractOp->getBlock()))
      return rewriter.notifyMatchFailure(
          contractOp, "Acc read/write should be in a separate block.");

    // Replace acc of a contraction operation with vector constant.
    rewriter.setInsertionPointAfter(accReadOp);
    Value accValue = accReadOp->getResult(0);
    auto vecTy = llvm::dyn_cast<VectorType>(accValue.getType());
    if (!vecTy)
      return rewriter.notifyMatchFailure(contractOp, "Excepts vector type.");

    Location loc = accReadOp->getLoc();
    Type elemTy = vecTy.getElementType();

    Value zeroScalar = arith::ConstantOp::create(rewriter, loc, elemTy,
                                                 rewriter.getZeroAttr(elemTy));

    Value zeroVec =
        vector::BroadcastOp::create(rewriter, loc, vecTy, zeroScalar);

    accValue.replaceAllUsesWith(zeroVec);

    // Adds the initial acc value with acontract results before storing to acc
    // matrix.
    rewriter.setInsertionPoint(resultWriteOp);
    Location locUser = resultWriteOp->getLoc();

    Value addition;

    if (llvm::isa<FloatType>(elemTy)) {
      addition =
          arith::AddFOp::create(rewriter, locUser, contractValue, accValue);
    }

    if (llvm::isa<IntegerType>(elemTy)) {
      addition =
          arith::AddIOp::create(rewriter, locUser, contractValue, accValue);
    }

    resultWriteOp->replaceUsesOfWith(contractValue, addition);
    return success();
  }
};

} // namespace

void x86::populateMoveAccumulatorForContractLoopPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MoveAccumulatorForContractLoop>(patterns.getContext());
}
