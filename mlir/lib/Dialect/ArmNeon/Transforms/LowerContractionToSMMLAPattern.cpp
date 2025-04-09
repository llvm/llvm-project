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

/// Lowering from a vector::contractOp arm neon smmla intrinsic. This will tile
/// any vector.contract into multiple smmla instructions with unrolling so long
/// as [2,2,8] is a divisor of its shape. It can also process vecmats with dimM
/// = 1 (either explicitly or inferred if LHS has only dimK) If no unrolling is
/// necessary, a single smmla instruction is emitted.
class LowerContractionToSMMLAPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Infer tile sizes from operands. For vecmat, LHS may only have 1 dim.
    // Note: RHS is not transposed.
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();
    // Avoid 0-D vectors and 1-D rhs:
    if (!lhsType.hasRank() || !rhsType.hasRank() || rhsType.getRank() < 2)
      return failure();
    auto dimM = lhsType.getRank() == 1 ? 1 : lhsType.getDimSize(0);
    auto dimN = rhsType.getDimSize(0);
    auto dimK = rhsType.getDimSize(1);
    bool isVecmat = dimM == 1 ? true : false;
    if (lhsType.getDimSize(lhsType.getRank() - 1) !=
        rhsType.getDimSize(rhsType.getRank() - 1)) {
      return failure(); // dimK mismatch
    }
    // Unrolling patterns can handle any [2, 2, 8] shaped multiple of inputs for
    // tiling.
    if ((dimM % 2 != 0 && !isVecmat) || dimN % 2 != 0 || dimK % 8 != 0) {
      return failure();
    }

    // Check iterator types for contract. All iterators except inner-most
    // dimension must be parallel.
    auto iteratorTypes = op.getIteratorTypesArray();
    if (iteratorTypes.size() > 3 || iteratorTypes[iteratorTypes.size() - 1] !=
                                        vector::IteratorType::reduction) {
      return failure();
    }
    if (llvm::any_of(ArrayRef<vector::IteratorType>(iteratorTypes).drop_back(1),
                     [](vector::IteratorType iteratorType) {
                       return iteratorType != vector::IteratorType::parallel;
                     })) {
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
            dyn_cast<mlir::VectorType>(origLhsExtOp.getIn().getType())) {
      if (lhsExtInType.getElementTypeBitWidth() <= 8) {
        Type targetLhsExtTy =
            matchContainerType(rewriter.getI8Type(), lhsExtInType);
        extsiLhs = rewriter.createOrFold<arith::ExtSIOp>(loc, targetLhsExtTy,
                                                         origLhsExtOp.getIn());
      }
    }
    if (auto rhsExtInType =
            dyn_cast<mlir::VectorType>(origRhsExtOp.getIn().getType())) {
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
    SmallVector<int64_t> smmlaShape = {2, 8};
    SmallVector<int64_t> loopOrder = {0, 1};
    if (unrolledSize.size() == 3) {
      smmlaShape.insert(smmlaShape.begin(), isVecmat ? 1 : 2);
      loopOrder.push_back(2);
    }

    // Keep track of the previous accumulator when tiling over K.
    Value kAcc;
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
      Value tiledLhs = extractOperand(extsiLhs, lhsPermutationMap, lhsOffsets);
      AffineMap rhsPermutationMap = op.getIndexingMapsArray()[1];
      SmallVector<int64_t> rhsOffsets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      Value tiledRhs = extractOperand(extsiRhs, rhsPermutationMap, rhsOffsets);
      AffineMap accPermutationMap = op.getIndexingMapsArray()[2];
      SmallVector<int64_t> accOffsets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      Value tiledAcc =
          extractOperand(op.getAcc(), accPermutationMap, accOffsets);

      auto inputElementType =
          cast<ShapedType>(tiledLhs.getType()).getElementType();
      auto accElementType =
          cast<ShapedType>(tiledAcc.getType()).getElementType();
      auto inputExpandedType = VectorType::get({2, 8}, inputElementType);
      auto outputExpandedType = VectorType::get({2, 2}, accElementType);

      // With vecmat, tiled LHS and ACC will contain only one of 2 necessary
      // rows along dimM. Expand their shapes to match the smmla op.
      if (isVecmat) {
        auto expandForSMMLA = [&](Value tiledOperand,
                                  VectorType expandedTypeType) {
          auto emptyOperand = rewriter.create<arith::ConstantOp>(
              loc, expandedTypeType, rewriter.getZeroAttr(expandedTypeType));
          SmallVector<int64_t> offsets(
              cast<ShapedType>(emptyOperand.getType()).getRank(), 0);
          SmallVector<int64_t> strides(
              cast<ShapedType>(tiledOperand.getType()).getRank(), 1);
          return rewriter.createOrFold<vector::InsertStridedSliceOp>(
              loc, tiledOperand, emptyOperand, offsets, strides);
        };
        tiledLhs = expandForSMMLA(tiledLhs, inputExpandedType);
        tiledAcc = expandForSMMLA(tiledAcc, outputExpandedType);
      }

      // Collapse tiled operands to 1D vectors required by smmla intrinsic
      auto collapsedInputType =
          VectorType::get(inputExpandedType.getNumElements(), inputElementType);
      auto collapsedLhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledLhs.getLoc(), collapsedInputType, tiledLhs);
      auto collapsedRhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledRhs.getLoc(), collapsedInputType, tiledRhs);
      auto collapsedOutputType =
          VectorType::get(outputExpandedType.getNumElements(), accElementType);

      bool initialKAcc = offsets.back() == 0;
      Value collapsedRes;
      if (!initialKAcc) {
        collapsedRes = kAcc;
      } else {
        collapsedRes = rewriter.createOrFold<vector::ShapeCastOp>(
            tiledAcc.getLoc(), collapsedOutputType, tiledAcc);
      }

      // Insert contract op
      kAcc = rewriter.createOrFold<arm_neon::SmmlaOp>(
          op.getLoc(), collapsedRes.getType(), collapsedRes, collapsedLhs,
          collapsedRhs);

      // Reshape output back to 2D
      Value tiledRes = rewriter.createOrFold<vector::ShapeCastOp>(
          kAcc.getLoc(), tiledAcc.getType(), kAcc);

      // With vecmat, only one row of tiled ACC can be inserted into file result
      if (isVecmat) {
        tiledRes = rewriter.createOrFold<vector::ExtractOp>(loc, tiledRes, 0);
      }

      // Insert the tiled result back into the non tiled result of the
      // contract op.
      SmallVector<int64_t> strides(
          cast<ShapedType>(tiledRes.getType()).getRank(), 1);
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
