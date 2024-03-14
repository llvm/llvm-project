//===- LowerContractionToSMMLAPattern.cpp - Contract to SMMLA ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from vector.contract to
// arm_neon.intr.smmla
//
//===---

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "lower-contract-to-arm-neon"

using namespace mlir;
using namespace mlir::arm_neon;

namespace {

/// Return the shaped type with new element type.
static Type matchContainerType(Type element, Type container) {
  if (auto shapedTy = dyn_cast<ShapedType>(container)) {
    return shapedTy.clone(element);
  }
  return element;
}

/// Lowering from a single vector::contractOp directly to the arm neon smmla
/// intrinsic. The shapes of the contract and intrinsic must match.
class LowerContractionToSMMLAPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getAcc();

    // Check index maps that represent M N K in contract.
    auto indexingMaps = op.getIndexingMapsArray();
    if (llvm::any_of(indexingMaps, [](mlir::AffineMap affineMap) {
          return affineMap.isPermutation() || affineMap.getNumDims() != 3 ||
                 affineMap.getNumResults() != 2;
        })) {
      return failure();
    }

    // Check iterator types for contract.
    auto iteratorTypes = op.getIteratorTypesArray();
    if (iteratorTypes.size() != 3 ||
        iteratorTypes[0] != vector::IteratorType::parallel ||
        iteratorTypes[1] != vector::IteratorType::parallel ||
        iteratorTypes[2] != vector::IteratorType::reduction) {
      return failure();
    }

    // Check the tile size by mapping the dimensions of the contract.
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();
    auto dimM = lhsType.getDimSize(0);
    auto dimN = rhsType.getDimSize(0);
    auto dimK = lhsType.getDimSize(1);
    if (rhsType.getDimSize(1) != dimK || dimM != 2 || dimN != 2 || dimK != 8) {
      return failure();
    }

    // Check two extsi inputs Rhs Lhs for contract.
    arith::ExtSIOp origLhsExtOp =
        dyn_cast_or_null<arith::ExtSIOp>(lhs.getDefiningOp());
    arith::ExtSIOp origRhsExtOp =
        dyn_cast_or_null<arith::ExtSIOp>(rhs.getDefiningOp());
    if (!origLhsExtOp || !origRhsExtOp) {
      return failure();
    }

    // Match any iX to i32 for X<8 then turn into an i8 output. Feed into
    // following neon instruction. Check inputs for extsi are <=i8
    Value extsiLhs;
    Value extsiRhs;
    if (auto lhsExtInType =
            origLhsExtOp.getIn().getType().dyn_cast<mlir::VectorType>()) {
      if (lhsExtInType.getElementTypeBitWidth() <= 8) {
        Type targetLhsExtTy =
            matchContainerType(rewriter.getI8Type(), lhsExtInType);
        extsiLhs = rewriter.createOrFold<arith::ExtSIOp>(loc, targetLhsExtTy,
                                                         origLhsExtOp.getIn());
      }
    }
    if (auto rhsExtInType =
            origRhsExtOp.getIn().getType().dyn_cast<mlir::VectorType>()) {
      if (rhsExtInType.getElementTypeBitWidth() <= 8) {
        Type targetRhsExtTy =
            matchContainerType(rewriter.getI8Type(), rhsExtInType);
        extsiRhs = rewriter.createOrFold<arith::ExtSIOp>(loc, targetRhsExtTy,
                                                         origRhsExtOp.getIn());
      }
    }

    if (!extsiLhs || !extsiRhs) {
      return failure();
    }

    // Collapse to 1D vectors required by smmla intrinsic
    auto collapsedInputType = VectorType::get(
        {16}, extsiLhs.getType().cast<ShapedType>().getElementType());
    auto collapsedOutputType =
        VectorType::get({4}, res.getType().cast<ShapedType>().getElementType());
    auto collapsedLhs = rewriter.createOrFold<vector::ShapeCastOp>(
        extsiLhs.getLoc(), collapsedInputType, extsiLhs);
    auto collapsedRhs = rewriter.createOrFold<vector::ShapeCastOp>(
        extsiRhs.getLoc(), collapsedInputType, extsiRhs);
    auto collapsedRes = rewriter.createOrFold<vector::ShapeCastOp>(
        res.getLoc(), collapsedOutputType, res);

    // Replace the contract with a neon op
    auto smmlaOp = rewriter.createOrFold<arm_neon::SmmlaOp>(
        op.getLoc(), collapsedRes.getType(), collapsedRes, collapsedLhs,
        collapsedRhs);

    // Reshape output back to 2D
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getResultType(),
                                                     smmlaOp);
    return success();
  }
};

} // namespace

void mlir::arm_neon::populateLowerContractionToSMMLAPatternPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToSMMLAPattern>(context, /*benefit=*/1);
}
