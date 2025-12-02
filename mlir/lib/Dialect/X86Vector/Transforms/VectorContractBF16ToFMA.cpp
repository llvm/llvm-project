//===- VectorContractBF16ToFMA.cpp
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

static FailureOr<llvm::SmallVector<mlir::memref::SubViewOp>>
getSubviewFromVectorInput(Location loc, PatternRewriter &rewriter,
                          mlir::Value prodOp, int64_t mnDim, int64_t vnniDim,
                          int64_t mnDimIndx) {

  llvm::SmallVector<mlir::memref::SubViewOp> subviews;

  Value srcOperation;
  SmallVector<OpFoldResult> indexVals;

  if (auto transferRead =
          prodOp.getDefiningOp<mlir::vector::TransferReadOp>()) {
    srcOperation = transferRead.getOperand(0);
    SmallVector<OpFoldResult> indexValues(transferRead.getIndices().begin(),
                                          transferRead.getIndices().end());
    indexVals = indexValues;
  }

  if (auto load = prodOp.getDefiningOp<mlir::vector::LoadOp>()) {
    srcOperation = load.getOperand(0);
    SmallVector<OpFoldResult> indexValues(load.getIndices().begin(),
                                          load.getIndices().end());
    indexVals = indexValues;
  }

  if (!srcOperation)
    return failure();

  llvm::SmallVector<OpFoldResult> strides;
  llvm::SmallVector<OpFoldResult> sizes;

  for (unsigned int i = 0; i < indexVals.size(); i++) {
    strides.push_back(rewriter.getIndexAttr(1));
    sizes.push_back(rewriter.getIndexAttr(1));
  }

  sizes[indexVals.size() - 1] = rewriter.getIndexAttr(vnniDim);
  sizes[indexVals.size() - mnDimIndx] = rewriter.getIndexAttr(mnDim);

  if (mnDim == 1) {
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(1);
  }

  auto subview = memref::SubViewOp::create(rewriter, loc, srcOperation,
                                           indexVals, sizes, strides);
  subviews.push_back(subview);

  if (mnDim == 1) {
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(0);
    sizes[indexVals.size() - 1] = rewriter.getIndexAttr(1);

    auto unitDimEvenIndxSubview = memref::SubViewOp::create(
        rewriter, loc, srcOperation, indexVals, sizes, strides);
    subviews.push_back(unitDimEvenIndxSubview);
  }

  return subviews;
}

struct VectorContractBF16ToFMA
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only BF16 lowering is supported.");

    if (!isInVnniLayout(contractOp.getOperation(),
                        contractOp.getIndexingMapsArray(), 2))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices not in VNNI format.");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if ((nonUnitDimLhs.size() - 1) > 0 && (nonUnitDimRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape other than VNNI.");

    if ((nonUnitDimLhs.size() - 1) != 1 && (nonUnitDimRhs.size() - 1) != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if ((lhsTy.getElementType().isBF16() && !accTy.getElementType().isF32()))
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32 acumulation supported for BF16 type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });
    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B should be a non-unit dim in acc.");

    // Non-unit dimensions should match the vector length of BF16 or Int8
    // dot-product.
    unsigned int nonUnitDim = nonUnitDimLhs.size() == 2 ? nonUnitDimLhs.front()
                                                        : nonUnitDimRhs.front();
    if (nonUnitDim != 4 && nonUnitDim != 8 &&
        !(nonUnitDimAcc.front() == nonUnitDim))
      return rewriter.notifyMatchFailure(
          contractOp, "BF16 packed load operation expects non-unit (LHR or "
                      "RHS) dim and acc dim of size 4/8.");

    auto loc = contractOp.getLoc();
    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
        contractOp.getAcc());
    mlir::VectorType dstType =
        mlir::VectorType::get(nonUnitDimAcc.front(), rewriter.getF32Type());

    llvm::SmallVector<mlir::memref::SubViewOp> unitDimSubview;
    llvm::SmallVector<mlir::memref::SubViewOp> nonUnitDimSubview;

    if ((nonUnitDimRhs.size() - 1) > 0) {

      auto unitSubview = getSubviewFromVectorInput(
          loc, rewriter, contractOp.getLhs(), 1, 1, 2);
      auto nonUnitSubview = getSubviewFromVectorInput(
          loc, rewriter, contractOp.getRhs(), nonUnitDimRhs.front(), 2, 2);
      if (failed(unitSubview) || failed(nonUnitSubview))
        return failure();

      unitDimSubview = *unitSubview;
      nonUnitDimSubview = *nonUnitSubview;

    } else {
      auto nonUnitSubview = getSubviewFromVectorInput(
          loc, rewriter, contractOp.getLhs(), nonUnitDimRhs.front(), 2, 3);
      auto unitSubview = getSubviewFromVectorInput(
          loc, rewriter, contractOp.getRhs(), 1, 1, 2);
      if (failed(unitSubview) || failed(nonUnitSubview))
        return failure();

      unitDimSubview = *unitSubview;
      nonUnitDimSubview = *nonUnitSubview;
    }

    auto loadBcstOddIndxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[0]);
    auto loadOddIndxElementF32 = x86vector::CvtPackedOddIndexedToF32Op::create(
        rewriter, loc, dstType, nonUnitDimSubview[0]);
    auto oddIndxFMA =
        vector::FMAOp::create(rewriter, loc, loadBcstOddIndxElementToF32,
                              loadOddIndxElementF32, castAcc);

    llvm::SmallVector<Operation *> users;
    for (OpResult result : contractOp->getResults())
      for (Operation *user : result.getUsers())
        users.push_back(user);

    if (users.size() == 1) {
      rewriter.setInsertionPoint(users[0]);
    }

    auto loadBcstEvenIndxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[1]);
    auto loadEvenIndxElementF32 =
        x86vector::CvtPackedEvenIndexedToF32Op::create(rewriter, loc, dstType,
                                                       nonUnitDimSubview[0]);

    vector::FMAOp fma =
        vector::FMAOp::create(rewriter, loc, loadBcstEvenIndxElementToF32,
                              loadEvenIndxElementF32, oddIndxFMA);

    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);
    rewriter.replaceOp(contractOp, castFma);
    return success();
  }
};

void x86vector::populateVectorContractBF16ToFMAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractBF16ToFMA>(patterns.getContext());
}
