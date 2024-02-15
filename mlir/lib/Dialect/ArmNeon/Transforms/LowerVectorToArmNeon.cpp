//===- LowerVectorToArmNeon.cpp - Lower 'arm_neon.intr.smmla' ops
//-----------===//
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

#define DEBUG_TYPE "arm-neon-vector-lowering"

using namespace mlir;
using namespace mlir::arm_neon;

namespace {

// Return the shaped type with new element type.
static Type matchContainerType(Type element, Type container) {
  if (auto shapedTy = dyn_cast<ShapedType>(container))
    return shapedTy.clone(element);

  return element;
}

// Lowering from vector::contractOp directly to the arm neon
// intrinsic.
class LowerVectorToArmNeonPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getAcc();

    // Check index maps represent M N K and aren't transposed.
    auto indexingMaps = op.getIndexingMapsArray();
    if (llvm::any_of(indexingMaps, [](mlir::AffineMap affineMap) {
          return affineMap.isPermutation() || affineMap.getNumDims() != 3 ||
                 affineMap.getNumResults() != 2;
        })) {
      return failure();
    }

    // Check iterator types for contract
    auto iteratorTypes = op.getIteratorTypesArray();
    if (iteratorTypes.size() != 3 ||
        iteratorTypes[0] != vector::IteratorType::parallel ||
        iteratorTypes[1] != vector::IteratorType::parallel ||
        iteratorTypes[2] != vector::IteratorType::reduction) {
      return failure();
    }

    // Check the tile size by mapping the dimensions of the contract
    //  -- Tile size: [2, 2, 8]
    // Infer tile sizes from operands. Check required tile size
    // Note: RHS is not transposed
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();
    auto dimM = lhsType.getDimSize(0);
    auto dimN = rhsType.getDimSize(0);
    auto dimK = lhsType.getDimSize(1);
    if (rhsType.getDimSize(1) != dimK || dimM != 2 || dimN != 2 || dimK != 8) {
      return failure();
    }

    // Check two extsi inputs Rhs Lhs
    arith::ExtSIOp origLhsExtOp;
    arith::ExtSIOp origRhsExtOp;
    if (!(origLhsExtOp =
              dyn_cast_or_null<arith::ExtSIOp>(lhs.getDefiningOp())) ||
        !(origRhsExtOp =
              dyn_cast_or_null<arith::ExtSIOp>(rhs.getDefiningOp()))) {
      return failure();
    }

    arith::ExtSIOp extsiLhs;
    arith::ExtSIOp extsiRhs;
    // Match any iX to i32 for X<8 then turn into an i8 output. Feed into
    // following neon instruction. Check inputs for extsi are <=i8
    if (auto lhsExtInType =
            origLhsExtOp.getIn().getType().dyn_cast<mlir::VectorType>()) {
      if (lhsExtInType.getElementTypeBitWidth() <= 8) {
        // Target lhs type with i8. This is likely redundant
        Type targetLhsExtTy =
            matchContainerType(rewriter.getI8Type(), lhsExtInType);
        extsiLhs = rewriter.create<arith::ExtSIOp>(loc, targetLhsExtTy,
                                                   origLhsExtOp.getIn());
      }
    }
    if (auto rhsExtInType =
            origRhsExtOp.getIn().getType().dyn_cast<mlir::VectorType>()) {
      if (rhsExtInType.getElementTypeBitWidth() <= 8) {
        // Target rhs type with i8
        Type targetRhsExtTy =
            matchContainerType(rewriter.getI8Type(), rhsExtInType);
        extsiRhs = rewriter.create<arith::ExtSIOp>(loc, targetRhsExtTy,
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
    auto collapsedLhs = rewriter.create<vector::ShapeCastOp>(
        extsiLhs.getLoc(), collapsedInputType, extsiLhs);
    auto collapsedRhs = rewriter.create<vector::ShapeCastOp>(
        extsiRhs.getLoc(), collapsedInputType, extsiRhs);
    auto collapsedRes = rewriter.create<vector::ShapeCastOp>(
        res.getLoc(), collapsedOutputType, res);

    // Replace the contract with a neon op
    auto smmlaOp = rewriter.create<arm_neon::SmmlaOp>(
        op.getLoc(), collapsedRes.getType(), collapsedRes, collapsedLhs,
        collapsedRhs);

    // Reshape output back to 2D
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getResultType(),
                                                     smmlaOp);
    return success();
  }
};

} // namespace

void mlir::arm_neon::populateLowerVectorToArmNeonPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerVectorToArmNeonPattern>(context, /*benefit=*/1);
}
