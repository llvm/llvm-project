//===- ShuffleBF16VectorContractResult.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/Utils/X86VectorUtils.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

struct ShuffleBF16VectorContractResult
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    // TODO: Move this validation to a common utility folder. Planned to
    // do once (code refactoring), all architecture specific nanokernel
    // passes are merged into the repo.
    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only BF16 lowering is supported.");

    if (isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(),
                       /*blockingFactor=*/2))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices in VNNI format.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if (!accTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32 acumulation supported for BF16 type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });

    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B should be a non-unit dim in acc.");

    int64_t nonUnitDimValue = nonUnitDimAcc.front();

    if (nonUnitDimValue != 8 && nonUnitDimValue != 16)
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator dimension should be 8 or 16");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if ((nonUnitDimValue == 16) && (nonUnitDimLhs.size() - 1) > 0 &&
        (nonUnitDimRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape.");
    if (nonUnitDimValue == 8 && nonUnitDimLhs.size() > 0 &&
        nonUnitDimRhs.size() > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape.");

    vector::ContractionOp pairContractOp;
    bool rhsHasMultipleNonUnitDims = nonUnitDimValue == 16
                                         ? (nonUnitDimRhs.size() - 1) > 0
                                         : nonUnitDimRhs.size() > 0;

    Operation *nextOp = contractOp;
    while ((nextOp = nextOp->getNextNode())) {
      auto contOp = dyn_cast<vector::ContractionOp>(nextOp);

      if (!contOp)
        continue;

      if (validatePairVectorContract(
              contractOp, contOp, rhsHasMultipleNonUnitDims, nonUnitDimValue)) {
        pairContractOp = contOp;
        break;
      }
    }

    if (!pairContractOp)
      return failure();

    Operation *accReadOp0 =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());
    Operation *accReadOp1 =
        traceToVectorReadLikeParentOperation(pairContractOp.getAcc());

    Operation *resultWriteOp0 =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());
    Operation *resultWriteOp1 =
        traceToVectorWriteLikeUserOperation(pairContractOp.getResult());

    if (!accReadOp0 || !accReadOp1)
      return failure();

    if (!resultWriteOp0 || !resultWriteOp1)
      return failure();

    shuffleAfterReadLikeOp(rewriter, accReadOp0, accReadOp1, contractOp,
                           pairContractOp, nonUnitDimValue, accTy);
    shuffleBeforeWriteLikeOp(rewriter, resultWriteOp0, resultWriteOp1,
                             nonUnitDimValue, accTy);

    return success();
  }
};

void x86vector::populateShuffleBF16VectorContractResultPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShuffleBF16VectorContractResult>(patterns.getContext());
}
