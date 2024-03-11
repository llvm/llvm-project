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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
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

/// Lowering from a vector::contractOp arm neon smmla intrinsic. This up to an
/// 8x8x8 vector contract that is tiled (up to 16) smmla instructions with
/// unrolling. If no unrolling is necessary, a single smmla instruction is
/// emitted.
class LowerContractionToSMMLAPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
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
    // Infer tile sizes from operands; Note: RHS is not transposed.
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();
    auto dimM = lhsType.getDimSize(0);
    auto dimN = rhsType.getDimSize(0);
    auto dimK = lhsType.getDimSize(1);

    // Unrolling patterns can handle [(2|4|8), (2|4|8), 8] shaped inputs for
    // tiling.
    if (dimM % 2 != 0 || dimM > 8 || dimN % 2 != 0 || dimN > 8 || dimK != 8) {
      return failure();
    }

    // Check two extsi inputs Rhs Lhs for contract.
    arith::ExtSIOp origLhsExtOp =
        dyn_cast_or_null<arith::ExtSIOp>(op.getLhs().getDefiningOp());
    arith::ExtSIOp origRhsExtOp =
        dyn_cast_or_null<arith::ExtSIOp>(op.getRhs().getDefiningOp());
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

    // Initial accumulator for the final result. This is the un-tiled result if
    // tiling is done.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, op.getResultType(), rewriter.getZeroAttr(op.getResultType()));

    SmallVector<int64_t> unrolledSize = *op.getShapeForUnroll();
    SmallVector<int64_t> smmlaShape{2, 2, 8};
    SmallVector<int64_t> loopOrder{0, 1, 2};
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(unrolledSize, smmlaShape, loopOrder)) {

      // Helper to compute the new shape of each operand and extract the slice.
      auto extractOperand = [&](Value operand, AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffsets) {
        SmallVector<int64_t> operandShape =
            applyPermutationMap(permutationMap, ArrayRef<int64_t>(smmlaShape));
        SmallVector<int64_t> operandStrides(operandOffsets.size(), 1);
        return rewriter.createOrFold<vector::ExtractStridedSliceOp>(
            loc, operand, operandOffsets, operandShape, operandStrides);
      };

      // Extract tiled lhs, rhs, and acc
      AffineMap lhsPermutationMap = op.getIndexingMapsArray()[0];
      SmallVector<int64_t> lhsOffsets =
          applyPermutationMap(lhsPermutationMap, ArrayRef<int64_t>(offsets));
      auto tiledLhs = extractOperand(extsiLhs, lhsPermutationMap, lhsOffsets);
      AffineMap rhsPermutationMap = op.getIndexingMapsArray()[1];
      SmallVector<int64_t> rhsOffsets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      auto tiledRhs = extractOperand(extsiRhs, rhsPermutationMap, rhsOffsets);
      AffineMap accPermutationMap = op.getIndexingMapsArray()[2];
      SmallVector<int64_t> accOffsets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      auto tiledAcc =
          extractOperand(op.getAcc(), accPermutationMap, accOffsets);

      // Collapse tiled operands to 1D vectors required by smmla intrinsic
      auto collapsedInputType = VectorType::get(
          tiledLhs.getType().cast<ShapedType>().getNumElements(),
          tiledLhs.getType().cast<ShapedType>().getElementType());
      auto collapsedOutputType = VectorType::get(
          {4}, tiledAcc.getType().cast<ShapedType>().getElementType());
      auto collapsedLhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledLhs.getLoc(), collapsedInputType, tiledLhs);
      auto collapsedRhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledRhs.getLoc(), collapsedInputType, tiledRhs);
      auto collapsedRes = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledAcc.getLoc(), collapsedOutputType, tiledAcc);

      // Insert contract op
      auto smmlaOp = rewriter.createOrFold<arm_neon::SmmlaOp>(
          op.getLoc(), collapsedRes.getType(), collapsedRes, collapsedLhs,
          collapsedRhs);

      // Reshape output back to 2D
      Value tiledRes = rewriter.createOrFold<vector::ShapeCastOp>(
          smmlaOp.getLoc(), tiledAcc.getType(), smmlaOp);

      // Insert the tiled result back into the non tiled result of the
      // contract op.
      SmallVector<int64_t> strides(
          tiledRes.getType().cast<ShapedType>().getRank(), 1);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, tiledRes, result, accOffsets, strides);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::arm_neon::populateLowerContractionToSMMLAPatternPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToSMMLAPattern>(context, /*benefit=*/1);
}
