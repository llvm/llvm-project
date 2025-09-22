//===- LowerContractToNeonPatterns.cpp - Contract to I8MM/BF16 --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from vector.contract to operations
// that map to instructions from the Neon FEAT_I8MM extension.
//
// TODO: There may be opportunities to unify this with a similar pattern
// for SVE. See:
//   https://github.com/llvm/llvm-project/issues/145559
//   LowerContractToSVEPatterns.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "lower-contract-to-arm-neon"

using namespace mlir;
using namespace mlir::arm_neon;

namespace {
/// Get the operand of a `vector.contract`. This function is intended to
/// abstract away from the particular way a value is extended before feeding it
/// into the `vector.contract` - via zero-extend or an explicit or implicit
/// sign-extend (for implicit sign-extension see `vector.contract`
/// documentation).
///
/// The template parameter `Op` indicates the extension operation (explicit or
/// implicit) for which we are checking.
///
// Return success only for extensions from `iN` (N <= 8) to `i32`.
template <typename Op>
std::optional<Value> getExtOperand(Value v) {

  static_assert(llvm::is_one_of<Op, arith::ExtSIOp, arith::ExtUIOp>::value,
                "Must be instantiated with either sign- or zero- extension op");

  // If the operand is not defined by an explicit extend operation of the
  // accepted operation type allow for an implicit sign-extension.
  auto extOp = v.getDefiningOp<Op>();
  if (!extOp) {
    if constexpr (std::is_same<Op, arith::ExtSIOp>::value) {
      auto eltTy = cast<VectorType>(v.getType()).getElementType();
      if (!eltTy.isSignlessInteger() || eltTy.getIntOrFloatBitWidth() > 8)
        return {};
      return v;
    }
    return {};
  }

  // If the operand is defined by an explicit extend operation of the accepted
  // operation type, check it's extended from `iN` (N <= 8) to `i32`.
  auto inOp = extOp.getIn();
  auto inTy = dyn_cast<VectorType>(inOp.getType());
  if (!inTy)
    return {};
  auto inEltTy = inTy.getElementType();
  if (!inEltTy.isSignlessInteger() || inEltTy.getIntOrFloatBitWidth() > 8)
    return {};

  auto outTy = dyn_cast<VectorType>(extOp.getType());
  if (!(outTy && outTy.getElementType().isSignlessInteger(32)))
    return {};

  return inOp;
}

/// Helper function to extend a vector with elements iN, N < 8 to
/// a vector of i8. Do sign extension if the parameter `signExt` is true,
/// zero extension otherwise.
Value extendSmallIntVector(Location loc, VectorType srcTy, Value val,
                           bool signExt, PatternRewriter &rewriter) {
  Type targetTy = srcTy.clone(rewriter.getI8Type());
  return signExt ? rewriter.createOrFold<arith::ExtSIOp>(loc, targetTy, val)
                 : rewriter.createOrFold<arith::ExtUIOp>(loc, targetTy, val);
}

class VectorContractRewriter {
protected:
  // Designate the operation (resp. instruction) used to do sub-tile matrix
  // multiplications.
  enum class MMLA {
    Nop,
    SignedInt,   // smmla
    UnsignedInt, // ummla
    MixedInt,    // usmmla
    Bfloat       // bfmmla
  };

  // Lower-level operation to be emitted.
  MMLA mmlaOp = MMLA::Nop;

  // Indicate if the operands for the ArmNeon dialect operation need to be
  // swapped. Currently this is needed in order to emulate an "summla"
  // operation.
  bool swapOperands = false;

  // The operand tiles. These are not necessarily the operands of
  // `vector.contract`, for example they could be operands to `arith.extsi`
  // that is in turn fed into `vector.contract`.
  Value lhs;
  Value rhs;
  Value acc;

  // The dimensions logically corresponding to matrix multiplication of
  // MxK * KxN -> MxN. The operands and the result do not necessarily have these
  // shapes, for example RHS could be NxK with a transposing indexing map.
  int64_t dimM = 0;
  int64_t dimN = 0;
  int64_t dimK = 0;

  // Unroll iteration bounds. See documentaiton for `StaticTileOffsetRange`.
  SmallVector<int64_t> iterationBounds;

  // Sub-tile shape. The algorithm handles operand shapes, which are multiples
  // of this shape.
  SmallVector<int64_t> subTileShape;

  // Create the matrix multiply and accumulate operation according to `mmlaOp`.
  Value createMMLA(PatternRewriter &rewriter, Location loc, Value acc,
                   Value lhs, Value rhs) {

    if (swapOperands)
      std::swap(lhs, rhs);
    switch (mmlaOp) {
    case MMLA::SignedInt:
      return rewriter.createOrFold<arm_neon::SmmlaOp>(loc, acc.getType(), acc,
                                                      lhs, rhs);
    case MMLA::UnsignedInt:
      return rewriter.createOrFold<arm_neon::UmmlaOp>(loc, acc.getType(), acc,
                                                      lhs, rhs);
    case MMLA::MixedInt:
      return rewriter.createOrFold<arm_neon::UsmmlaOp>(loc, acc.getType(), acc,
                                                       lhs, rhs);
    case MMLA::Bfloat:
      return arm_neon::BfmmlaOp::create(rewriter, loc, acc.getType(), acc, lhs,
                                        rhs);
    case MMLA::Nop:
      llvm_unreachable("Uninitialized operation type");
    }
  }

  // Check common preconditions for applying the patterns and initialize
  // logical dimensions.
  LogicalResult matchAndInit(vector::ContractionOp op,
                             PatternRewriter &rewriter) {
    // Check iterator types for matrix multiplication.
    SmallVector<vector::IteratorType> itTypes = op.getIteratorTypesArray();
    if ((itTypes.size() != 3 || itTypes[0] != vector::IteratorType::parallel ||
         itTypes[1] != vector::IteratorType::parallel ||
         itTypes[2] != vector::IteratorType::reduction) &&
        (itTypes.size() != 2 || itTypes[0] != vector::IteratorType::parallel ||
         itTypes[1] != vector::IteratorType::reduction))
      return rewriter.notifyMatchFailure(
          op, "iterator types do not correspond to matrix multiplication");

    // Avoid 0-D vectors and 1-D rhs:
    VectorType lhsType = op.getLhsType();
    VectorType rhsType = op.getRhsType();
    if (!lhsType.hasRank() || !rhsType.hasRank() || lhsType.getRank() > 2 ||
        rhsType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Invalid operand rank");

    // This codegen does not work for scalable vectors. Return failure so this
    // pattern is not accidentally chosen over patterns that lower to ArmSVE.
    if (lhsType.isScalable() || rhsType.isScalable())
      return rewriter.notifyMatchFailure(op,
                                         "Not applicable to scalable vectors");

    // Initialize dimensions and check for a matching K dimension.
    dimM = lhsType.getDimSize(0);
    dimN = rhsType.getDimSize(0);
    dimK = rhsType.getDimSize(1);

    int64_t lhsDimK;
    if (lhsType.getRank() == 1) {
      dimM = 1;
      lhsDimK = lhsType.getDimSize(0);
    } else {
      lhsDimK = lhsType.getDimSize(1);
    }

    if (lhsDimK != dimK)
      return rewriter.notifyMatchFailure(op, "Dimensions mismatch");

    return success();
  }

public:
  void lower(vector::ContractionOp op, PatternRewriter &rewriter) {
    // Create some convenience types.
    auto inputElementType = cast<ShapedType>(lhs.getType()).getElementType();
    auto accElementType = cast<ShapedType>(acc.getType()).getElementType();
    auto inputExpandedType =
        VectorType::get({2, subTileShape.back()}, inputElementType);
    auto outputExpandedType = VectorType::get({2, 2}, accElementType);

    // One-dimensional representation of logical sub-tiles as required by the
    // ArmNeon ops.
    auto collapsedInputType =
        VectorType::get(inputExpandedType.getNumElements(), inputElementType);
    auto collapsedOutputType =
        VectorType::get(outputExpandedType.getNumElements(), accElementType);

    // Get indexing maps for a more concise/convenient access.
    auto indexingMaps = op.getIndexingMapsArray();
    AffineMap &lhsPermutationMap = indexingMaps[0];
    AffineMap &rhsPermutationMap = indexingMaps[1];
    AffineMap &accPermutationMap = indexingMaps[2];

    Location loc = op.getLoc();

    // Initial accumulator for the final result. This is the un-tiled result if
    // tiling is done.
    Value result =
        arith::ConstantOp::create(rewriter, loc, op.getResultType(),
                                  rewriter.getZeroAttr(op.getResultType()));

    SmallVector<int64_t, 3> loopOrder = {0, 1};
    if (iterationBounds.size() == 3)
      loopOrder.push_back(2);

    // Keep track of the previous accumulator when tiling over K.
    Value kAcc;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(iterationBounds, subTileShape, loopOrder)) {
      // Helper to compute the new shape of each operand and extract the slice.
      auto extractOperand = [&](Value operand, AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffsets) {
        SmallVector<int64_t> operandShape = applyPermutationMap(
            permutationMap, ArrayRef<int64_t>(subTileShape));
        SmallVector<int64_t> operandStrides(operandOffsets.size(), 1);
        return rewriter.createOrFold<vector::ExtractStridedSliceOp>(
            loc, operand, operandOffsets, operandShape, operandStrides);
      };

      // Extract tiled lhs, rhs, and acc
      SmallVector<int64_t> lhsOffsets =
          applyPermutationMap(lhsPermutationMap, ArrayRef<int64_t>(offsets));
      Value tiledLhs = extractOperand(lhs, lhsPermutationMap, lhsOffsets);
      SmallVector<int64_t> rhsOffsets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      Value tiledRhs = extractOperand(rhs, rhsPermutationMap, rhsOffsets);
      SmallVector<int64_t> accOffsets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      Value tiledAcc = extractOperand(acc, accPermutationMap, accOffsets);

      // With vecmat, tiled LHS and ACC will contain only one of 2 necessary
      // rows along dimM. Expand their shapes to match the ArmNeon op.
      if (dimM == 1) {
        auto expandRowVector = [&](Value tiledOperand,
                                   VectorType expandedTypeType) {
          auto emptyOperand =
              arith::ConstantOp::create(rewriter, loc, expandedTypeType,
                                        rewriter.getZeroAttr(expandedTypeType));
          SmallVector<int64_t> offsets(
              cast<ShapedType>(emptyOperand.getType()).getRank(), 0);
          SmallVector<int64_t> strides(
              cast<ShapedType>(tiledOperand.getType()).getRank(), 1);
          return rewriter.createOrFold<vector::InsertStridedSliceOp>(
              loc, tiledOperand, emptyOperand, offsets, strides);
        };
        tiledLhs = expandRowVector(tiledLhs, inputExpandedType);
        tiledAcc = expandRowVector(tiledAcc, outputExpandedType);
      }

      // Transpose ACC if doing signed by unsigned multiplication, because we're
      // using the instruction for unsigned by signed multiplication with
      // reversed operands.
      if (swapOperands)
        tiledAcc = vector::TransposeOp::create(rewriter, loc, tiledAcc,
                                               ArrayRef<int64_t>({1, 0}));

      // Collapse tiled operands to 1D vectors required by the ArmNeon ops
      auto collapsedLhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledLhs.getLoc(), collapsedInputType, tiledLhs);
      auto collapsedRhs = rewriter.createOrFold<vector::ShapeCastOp>(
          tiledRhs.getLoc(), collapsedInputType, tiledRhs);

      bool initialKAcc = offsets.back() == 0;
      Value collapsedRes;
      if (!initialKAcc) {
        collapsedRes = kAcc;
      } else {
        collapsedRes = rewriter.createOrFold<vector::ShapeCastOp>(
            tiledAcc.getLoc(), collapsedOutputType, tiledAcc);
      }

      // Insert contract op
      kAcc =
          createMMLA(rewriter, loc, collapsedRes, collapsedLhs, collapsedRhs);

      // Reshape output back to 2D
      Value tiledRes = rewriter.createOrFold<vector::ShapeCastOp>(
          kAcc.getLoc(), tiledAcc.getType(), kAcc);

      // Because of the reversed operands the result is obtained transposed.
      // Transpose it back,
      if (swapOperands)
        tiledRes = vector::TransposeOp::create(rewriter, loc, tiledRes,
                                               ArrayRef<int64_t>({1, 0}));

      // With vecmat, only one row of tiled ACC can be inserted into the final
      // result
      if (dimM == 1)
        tiledRes = rewriter.createOrFold<vector::ExtractOp>(loc, tiledRes, 0);

      // Insert the tiled result back into the non tiled result of the
      // contract op.
      SmallVector<int64_t> strides(
          cast<ShapedType>(tiledRes.getType()).getRank(), 1);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, tiledRes, result, accOffsets, strides);
    }

    rewriter.replaceOp(op, result);
  }
};

class VectorContractRewriterI8MM : public VectorContractRewriter {
public:
  LogicalResult matchAndInit(vector::ContractionOp op,
                             PatternRewriter &rewriter) {
    if (failed(VectorContractRewriter::matchAndInit(op, rewriter)))
      return failure();

    // Unrolling patterns can handle any [2, 2, 8] shaped multiple of inputs for
    // tiling.
    if ((dimM != 1 && dimM % 2 != 0) || dimN % 2 != 0 || dimK % 8 != 0)
      return rewriter.notifyMatchFailure(op, "Unsupported operand shapes");

    // Check inputs are sign-/zero- extensions from iN (N <= 8) to i32. Get the
    // values before the extension. All four signed/unsigned combinations for
    // input operands are supported, but they are lowered to different
    // operations. Determine which is the appropriate operation to lower to.
    mmlaOp = MMLA::SignedInt;
    auto maybeLhs = getExtOperand<arith::ExtSIOp>(op.getLhs());
    if (!maybeLhs) {
      mmlaOp = MMLA::UnsignedInt;
      maybeLhs = getExtOperand<arith::ExtUIOp>(op.getLhs());
    }
    if (!maybeLhs)
      return rewriter.notifyMatchFailure(
          op, "LHS is not a sign- or zero- extended iN, N <= 8");

    auto maybeRhs = getExtOperand<arith::ExtSIOp>(op.getRhs());
    if (maybeRhs) {
      if (mmlaOp == MMLA::UnsignedInt)
        mmlaOp = MMLA::MixedInt;
    } else {
      if (mmlaOp == MMLA::SignedInt) {
        mmlaOp = MMLA::MixedInt;
        swapOperands = true;
      }
      maybeRhs = getExtOperand<arith::ExtUIOp>(op.getRhs());
    }

    if (!maybeRhs)
      return rewriter.notifyMatchFailure(
          op, "RHS is not a sign- or zero- extended iN, N <= 8");

    lhs = *maybeLhs;
    rhs = *maybeRhs;
    acc = op.getAcc();

    // Extend inputs from iN, N < 8 to i8.
    Location loc = op.getLoc();
    auto lhsExtInType = cast<VectorType>(lhs.getType());
    if (lhsExtInType.getElementTypeBitWidth() < 8)
      lhs = extendSmallIntVector(loc, lhsExtInType, lhs,
                                 /* signExt */
                                 (mmlaOp == MMLA::SignedInt ||
                                  (mmlaOp == MMLA::MixedInt && !swapOperands)),
                                 rewriter);

    auto rhsExtInType = cast<VectorType>(rhs.getType());
    if (rhsExtInType.getElementTypeBitWidth() < 8)
      rhs = extendSmallIntVector(loc, rhsExtInType, rhs,
                                 /* signExt */
                                 (mmlaOp == MMLA::SignedInt ||
                                  (mmlaOp == MMLA::MixedInt && swapOperands)),
                                 rewriter);

    // Initialize parameters for unrolling.
    iterationBounds = *op.getShapeForUnroll();
    if (iterationBounds.size() == 3)
      subTileShape = SmallVector<int64_t>({dimM == 1 ? 1 : 2, 2, 8});
    else
      subTileShape = SmallVector<int64_t>({2, 8});

    return success();
  }
};

class VectorContractRewriterBFMMLA : public VectorContractRewriter {
public:
  LogicalResult matchAndInit(vector::ContractionOp op,
                             PatternRewriter &rewriter) {

    if (failed(VectorContractRewriter::matchAndInit(op, rewriter)))
      return failure();

    // Unrolling patterns can handle any [2, 2, 4] shaped multiple of inputs for
    // tiling.
    if ((dimM != 1 && dimM % 2 != 0) || dimN % 2 != 0 || dimK % 4 != 0)
      return rewriter.notifyMatchFailure(op, "Unsupported operand shapes");

    // Check the output is a vector of Float32 elements.
    auto outTy = dyn_cast<VectorType>(op.getResultType());
    if (!outTy || outTy.getElementType() != rewriter.getF32Type())
      return rewriter.notifyMatchFailure(op,
                                         "output type is not a vector of f32");

    // Check the inputs are vectors of BFloat16 elements.
    if (op.getLhsType().getElementType() != rewriter.getBF16Type())
      return rewriter.notifyMatchFailure(op,
                                         "input type is not a vector of bf16");

    mmlaOp = MMLA::Bfloat;
    swapOperands = false;
    lhs = op.getLhs();
    rhs = op.getRhs();
    acc = op.getAcc();

    // Initialize parameters for unrolling.
    iterationBounds = *op.getShapeForUnroll();
    if (iterationBounds.size() == 3)
      subTileShape = SmallVector<int64_t>({dimM == 1 ? 1 : 2, 2, 4});
    else
      subTileShape = SmallVector<int64_t>({2, 4});

    return success();
  }
};

/// Lowering from a vector::contractOp arm neon smmla intrinsic. This will tile
/// any vector.contract into multiple smmla instructions with unrolling so long
/// as [2,2,8] is a divisor of its shape. It can also process vecmats with dimM
/// = 1 (either explicitly or inferred if LHS has only dimK) If no unrolling is
/// necessary, a single smmla instruction is emitted.
class LowerContractionToNeonI8MMPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    VectorContractRewriterI8MM vcr;
    if (failed(vcr.matchAndInit(op, rewriter)))
      return failure();
    vcr.lower(op, rewriter);

    return success();
  }
};

class LowerContractionToNeonBFMMLAPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    VectorContractRewriterBFMMLA vcr;
    if (failed(vcr.matchAndInit(op, rewriter)))
      return failure();
    vcr.lower(op, rewriter);

    return success();
  }
};

} // namespace

void mlir::arm_neon::populateLowerContractionToNeonI8MMPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToNeonI8MMPattern>(context, /*benefit=*/2);
}

void mlir::arm_neon::populateLowerContractionToNeonBFMMLAPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToNeonBFMMLAPattern>(context, /*benefit=*/2);
}
