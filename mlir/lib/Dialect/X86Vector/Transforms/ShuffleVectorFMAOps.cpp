//===- ShuffleVectorFMAOps.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

namespace {

// Validates whether the given operation is an x86vector operation and has only
// one consumer.
static bool validateFMAOperands(Value op) {
  if (auto cvt = op.getDefiningOp<x86vector::CvtPackedEvenIndexedToF32Op>())
    return cvt.getResult().hasOneUse();

  if (auto bcst = op.getDefiningOp<x86vector::BcstToPackedF32Op>())
    return bcst.getResult().hasOneUse();

  return false;
}

// Validates the vector.fma operation on the following conditions:
// (i) one of the lhs or rhs defining operation should be
// CvtPackedEvenIndexedToF32Op, (ii) the lhs or rhs defining operation should be
// an x86vector operation and has only one consumer, (iii) all operations
// are in the same block, and (iv) ths FMA has only one user.
static bool validateVectorFMAOp(vector::FMAOp fmaOp) {
  Value lhs = fmaOp.getLhs();
  Value rhs = fmaOp.getRhs();

  if (!isa<x86vector::CvtPackedEvenIndexedToF32Op>(lhs.getDefiningOp()) &&
      !isa<x86vector::CvtPackedEvenIndexedToF32Op>(rhs.getDefiningOp()))
    return false;

  if (!validateFMAOperands(lhs) || !validateFMAOperands(rhs))
    return false;

  if (lhs.getDefiningOp()->getBlock() != rhs.getDefiningOp()->getBlock())
    return false;

  if (lhs.getDefiningOp()->getBlock() != fmaOp->getBlock())
    return false;

  if (!fmaOp.getResult().hasOneUse())
    return false;

  Operation *consumer = *fmaOp.getResult().getUsers().begin();
  if (consumer->getBlock() != fmaOp->getBlock())
    return false;

  return true;
}

// Moves vector.fma along with the lhs and rhs defining operation before its
// consumer. If the consumer is vector.ShapeCastOp and has only one user then
// move before the consumer of vector.ShapeCastOp.
// TODO: Move before first consumer, if there are multiple.
static void moveFMA(PatternRewriter &rewriter, vector::FMAOp fmaOp) {
  Operation *consumer = *fmaOp.getResult().getUsers().begin();

  if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(consumer)) {
    if (shapeCastOp.getResult().hasOneUse()) {
      Operation *nxtConsumer = *shapeCastOp.getResult().getUsers().begin();
      if (nxtConsumer->getBlock() == fmaOp->getBlock()) {
        consumer = *shapeCastOp.getResult().getUsers().begin();
        rewriter.moveOpBefore(fmaOp.getLhs().getDefiningOp(), consumer);
        rewriter.moveOpBefore(fmaOp.getRhs().getDefiningOp(), consumer);
        rewriter.moveOpBefore(fmaOp.getOperation(), consumer);
        rewriter.moveOpBefore(shapeCastOp.getOperation(), consumer);
        return;
      }
    }
  }

  rewriter.moveOpBefore(fmaOp.getLhs().getDefiningOp(), consumer);
  rewriter.moveOpBefore(fmaOp.getRhs().getDefiningOp(), consumer);
  rewriter.moveOpBefore(fmaOp.getOperation(), consumer);

  return;
}

// Shuffle FMAs with x86vector operations as operands such that
// FMAs are grouped with respect to odd/even packed index.
//
// For example:
// ```
//   %1 = x86vector.avx.bcst_to_f32.packed
//   %2 = x86vector.avx.cvt.packed.odd.indexed_to_f32
//   %3 = vector.fma %1, %2, %arg1
//   %4 = x86vector.avx.bcst_to_f32.packed
//   %5 = x86vector.avx.cvt.packed.even.indexed_to_f32
//   %6 = vector.fma %4, %5, %3
//   %7 = x86vector.avx.bcst_to_f32.packed
//   %8 = x86vector.avx.cvt.packed.odd.indexed_to_f32
//   %9 = vector.fma %7, %8, %arg2
//   %10 = x86vector.avx.bcst_to_f32.packed
//   %11 = x86vector.avx.cvt.packed.even.indexed_to_f32
//   %12 = vector.fma %10, %11, %9
//   yield %6, %12
// ```
// to
// ```
//   %1 = x86vector.avx.bcst_to_f32.packed
//   %2 = x86vector.avx.cvt.packed.odd.indexed_to_f32
//   %3 = vector.fma %1, %2, %arg1
//   %7 = x86vector.avx.bcst_to_f32.packed
//   %8 = x86vector.avx.cvt.packed.odd.indexed_to_f32
//   %9 = vector.fma %7, %8, %arg2
//   %4 = x86vector.avx.bcst_to_f32.packed
//   %5 = x86vector.avx.cvt.packed.even.indexed_to_f32
//   %6 = vector.fma %4, %5, %3
//   %10 = x86vector.avx.bcst_to_f32.packed
//   %11 = x86vector.avx.cvt.packed.even.indexed_to_f32
//   %12 = vector.fma %10, %11, %9
//   yield %9, %12
// ```
// TODO: Shuffling supported only if the FMA, lhs/rhs defining operations
// have only one consumer. Have to extend this pass for multiple consumers.
struct ShuffleVectorFMAOps : public OpRewritePattern<vector::FMAOp> {
  using OpRewritePattern<vector::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fmaOp,
                                PatternRewriter &rewriter) const override {

    if (!validateVectorFMAOp(fmaOp))
      return failure();

    llvm::SmallVector<vector::FMAOp> fmaOps;
    Operation *nextOp = fmaOp;
    bool stopAtNextDependentFMA = true;

    // Break the loop and return failure if the immediate next FMA op
    // have CvtPackedEvenIndexedToF32Op in it's lhs/rhs defining ops.
    while ((nextOp = nextOp->getNextNode())) {
      auto fma = dyn_cast<vector::FMAOp>(nextOp);
      if (!fma)
        continue;

      bool hasX86CvtOperand = isa<x86vector::CvtPackedEvenIndexedToF32Op>(
                                  fma.getLhs().getDefiningOp()) ||
                              isa<x86vector::CvtPackedEvenIndexedToF32Op>(
                                  fma.getRhs().getDefiningOp());

      if (hasX86CvtOperand && stopAtNextDependentFMA)
        break;

      if (validateVectorFMAOp(fma))
        fmaOps.push_back(fma);

      stopAtNextDependentFMA = false;
    }

    if (fmaOps.empty())
      return rewriter.notifyMatchFailure(
          fmaOp, "No eligible FMA operations were found: the operation may "
                 "already be shuffled, there may be no following FMAs, or the "
                 "following FMAs do not satisfy the shuffle conditions.");

    fmaOps.push_back(fmaOp);
    for (auto fmaOp : fmaOps)
      moveFMA(rewriter, fmaOp);

    return success();
  }
};

} // namespace

void x86vector::populateShuffleVectorFMAOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShuffleVectorFMAOps>(patterns.getContext());
}
