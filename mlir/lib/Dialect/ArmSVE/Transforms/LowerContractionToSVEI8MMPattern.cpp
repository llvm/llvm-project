//===- LowerContractionToSVEI8MMPattern.cpp - Contract to I8MM --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from vector.contract to operations
// that map to instructions from the SVE FEAT_I8MM extension.
//
// TODO: There may be opportunities to unify this with a similar pattern
// for Neon. See:
//   https://github.com/llvm/llvm-project/issues/145559
//   LowerContractionToNeonI8MMPattern.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/UB/IR/UBOps.h"

#define DEBUG_TYPE "lower-contract-to-arm-sve-i8mm"

using namespace mlir;

namespace {
// Get the operand of a `vector.contract`. This function is intended to abstract
// away from the particular way a value is extended before feeding it into the
// `vector.contract` - via zero-extend or an explicit or implicit sign-extend
// (for implicit sign-extension see `vector.contract` documentation).
//
// The template parameter `Op` indicates the extension operation (explicit or
// implicit) for which we are checking.
//
// Return success only for extensions from `i8` to `i32`.
template <typename Op>
std::optional<Value> getExtOperand(Value v) {

  static_assert(llvm::is_one_of<Op, arith::ExtSIOp, arith::ExtUIOp>::value,
                "Must be instantiated with either sign- or zero- extension op");

  // If the operand is not defined by an explicit extend operation of the
  // accepted operation type allow for an implicit sign-extension.
  auto extOp = dyn_cast_or_null<Op>(v.getDefiningOp());
  if (!extOp) {
    if constexpr (std::is_same<Op, arith::ExtSIOp>::value) {
      auto vTy = cast<VectorType>(v.getType());
      if (!vTy.getElementType().isSignlessInteger(8))
        return {};
      return v;
    }
    return {};
  }

  // If the operand is defined by an explicit extend operation of the accepted
  // operation type, check it's extended from `i8` to `i32`.
  auto inOp = extOp.getIn();
  auto inTy = dyn_cast<VectorType>(inOp.getType());
  if (!inTy || !inTy.getElementType().isSignlessInteger(8))
    return {};

  auto outTy = dyn_cast<VectorType>(extOp.getType());
  if (!outTy || !outTy.getElementType().isSignlessInteger(32))
    return {};

  return inOp;
}

// Designate the operation (resp. instruction) used to do sub-tile matrix
// multiplications.
enum class MMLA {
  Signed,      // smmla
  Unsigned,    // ummla
  Mixed,       // usmmla
  MixedSwapped // usmmla with LHS and RHS swapped
};

// Create the matrix mulitply and accumulate operation according to `op`.
Value createMMLA(PatternRewriter &rewriter, MMLA op, Location loc,
                 mlir::VectorType accType, Value acc, Value lhs, Value rhs) {
  switch (op) {
  case MMLA::Signed:
    return rewriter.create<arm_sve::SmmlaOp>(loc, accType, acc, lhs, rhs);
  case MMLA::Unsigned:
    return rewriter.create<arm_sve::UmmlaOp>(loc, accType, acc, lhs, rhs);
  case MMLA::Mixed:
    return rewriter.create<arm_sve::UsmmlaOp>(loc, accType, acc, lhs, rhs);
  case MMLA::MixedSwapped:
    // The accumulator comes transposed and the result will be transposed
    // later, so all we have to do here is swap the operands.
    return rewriter.create<arm_sve::UsmmlaOp>(loc, accType, acc, rhs, lhs);
  }
}

/// Lower a contraction operation that performs a matrix multiplication
/// of two 8-bit integer matrix tiles with logical dimensions <Mx8> and <8x[N]>
/// for the left-hand side and the right-hand side, respectively,
/// yielding a <Mx[N]> 32-bit integer result.
///
/// The operands' shapes are such that the operands can be evenly split into
/// sub-tiles with dimensions as expected by the targeted FEAT_I8MM
/// instructions. The intent is that M and N are chosen (by higher level
/// transforms) in such a way as to maximise register usage. The main use case
/// we envision as of now is MMT4D, thus the RHS operand is expected
/// pre-transposed.
///
/// The matrix multiplication is performed by unrolling the usual tiled matrix
/// multiplication algorithm using sub-tiles with dimensions <2x8> for the LHS,
/// <8x[2]> for the RHS, and <2x[2]> for the result and the input accumulator.
///
/// One way to illustrate the operation is as follows:
///
/// RHS<8x[N]>:       <8x[2]> <8x[2]> ... <8x[2]>
///                 +-----------------------------
/// LHS<Mx8>: <2x8> | <2x[2]> <2x[2]> ... <2x[2]>
///           <2x8> | <2x[2]> <2x[2]> ... <2x[2]>
///            ...  |   ...     ...   ...   ...
///           <2x8> | <2x[2]> <2x[2]> ... <2x[2]>
///
/// The RHS operand is unpacked into N/2 values, each representing a sequence of
/// VSCALE number of sub-tiles with dimensions <8x2>.
/// The LHS operand is initially unpacked into M/2 values, each representing a
/// sub-tile with dimensions <2x8>, and then each such sub-tile is replicated
/// VSCALE times.
/// Multiplying thus replicated LHS sub-tile by the corresponding RHS sub-tile
/// correctly computes an entire result sub-tile.
class LowerContractionToSVEI8MMPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();

    // Check the rank the types so we can safely examine their dimensions.
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "non-matching operand shape");

    auto M = lhsType.getDimSize(0);
    auto N = rhsType.getDimSize(0);
    auto K = rhsType.getDimSize(1);

    // Check the operands have the expected shape:
    //  * for LHS: fixed vector MxK
    //  * for RHS: scalable vector [N]xK
    //  * K == 8
    //  * M and N even and at least 2
    if (lhsType.isScalable() || !rhsType.getScalableDims()[0] ||
        rhsType.getScalableDims()[1] || lhsType.getDimSize(1) != K || K != 8 ||
        M < 2 || M % 2 != 0 || N < 2 || N % 2 != 0 ||
        !rhsType.getScalableDims()[0])
      return rewriter.notifyMatchFailure(op, "non-matching operand shape");

    // Check permutation maps. For now only accept
    //   lhs: (d0, d1, d2) -> (d0, d2)
    //   rhs: (d0, d1, d2) -> (d1, d2)
    //   acc: (d0, d1, d2) -> (d0, d1)
    // This corresponds to matrix multiplication with transposed RHS.
    if (op.getIndexingMapsArray()[0] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{0u, 2u},
                                                 op.getContext()) ||
        op.getIndexingMapsArray()[1] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{1u, 2u},
                                                 op.getContext()) ||
        op.getIndexingMapsArray()[2] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{0u, 1u},
                                                 op.getContext()))
      return rewriter.notifyMatchFailure(op, "non-matching permutation maps");

    // Check iterator types for matrix multiplication.
    auto itTypes = op.getIteratorTypesArray();
    if (itTypes.size() != 3 || itTypes[0] != vector::IteratorType::parallel ||
        itTypes[1] != vector::IteratorType::parallel ||
        itTypes[2] != vector::IteratorType::reduction)
      return rewriter.notifyMatchFailure(
          op, "iterator types do not correspond to matrix multiplication");

    // Check the combining kind is addition.
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(op,
                                         "combining kind is not an addition");

    // Check the output is a vector of i32 elements.
    auto outTy = dyn_cast<VectorType>(op.getResultType());
    if (!outTy || outTy.getElementType() != rewriter.getI32Type())
      return rewriter.notifyMatchFailure(op,
                                         "output type is not a vector of i32");

    // Check inputs are sign-/zero- extensions from i8 to i32. Get the values
    // before the extension. All four signed/unsigned combinations for input
    // operands are supported, but they are lowered to different operations.
    // Determine which is the appropriate operation to lower to.
    MMLA mmlaOp = MMLA::Signed;
    auto maybeLhs = getExtOperand<arith::ExtSIOp>(op.getLhs());
    if (!maybeLhs) {
      mmlaOp = MMLA::Unsigned;
      maybeLhs = getExtOperand<arith::ExtUIOp>(op.getLhs());
    }
    if (!maybeLhs)
      return rewriter.notifyMatchFailure(
          op, "LHS is not a sign- or zero- extended i8");

    auto maybeRhs = getExtOperand<arith::ExtSIOp>(op.getRhs());
    if (maybeRhs) {
      if (mmlaOp == MMLA::Unsigned)
        mmlaOp = MMLA::Mixed;
    } else {
      if (mmlaOp == MMLA::Signed)
        mmlaOp = MMLA::MixedSwapped;
      maybeRhs = getExtOperand<arith::ExtUIOp>(op.getRhs());
    }
    if (!maybeRhs)
      return rewriter.notifyMatchFailure(
          op, "RHS is not a sign- or zero- extended i8");

    // One-dimensional vector types for arm_sve.*mmla
    auto nxv16i8 = VectorType::get(/*shape=*/16, rewriter.getI8Type(),
                                   /*scalableDims=*/{true});
    auto nxv4i32 = VectorType::get(/*shape=*/4, rewriter.getI32Type(),
                                   /*scalableDims=*/{true});

    // Extract LHS sub-tiles with logicall shape <2x8>.
    SmallVector<Value> lhsTile;
    for (int64_t i = 0; i < M; i += 2) {
      // Extract two consecutive rows of the LHS tile.
      auto r0 = rewriter.create<vector::ExtractOp>(loc, *maybeLhs,
                                                   ArrayRef<int64_t>{i});
      auto r1 = rewriter.create<vector::ExtractOp>(loc, *maybeLhs,
                                                   ArrayRef<int64_t>{i + 1});
      // Concatenate to obtain a 16 x i8 flattened sub-tile.
      auto t = rewriter.create<vector::ShuffleOp>(
          loc, r0, r1,
          llvm::ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                  14, 15});
      // Turn it into a scalable vector.
      auto s = rewriter.create<vector::ScalableInsertOp>(
          loc, t, rewriter.create<ub::PoisonOp>(loc, nxv16i8), 0);
      // Replicate the sub-tile VSCALE times to fill the entire vector.
      auto r = rewriter.create<arm_sve::DupQLaneOp>(loc, s, 0);
      lhsTile.push_back(r);
    }

    // "Flatten" the RHS tile from <[N]x8> to <[8*N]>.
    auto rhs = rewriter.create<vector::ShapeCastOp>(
        maybeRhs->getLoc(),
        VectorType::get(/*shape=*/8 * N, rewriter.getI8Type(),
                        /*scalableDims=*/{true}),
        *maybeRhs);

    // Extract the RHS sub-tiles with logical shape <8x[2]>.
    SmallVector<Value> rhsTile;
    for (int64_t j = 0; j < N; j += 2)
      rhsTile.push_back(
          rewriter.create<vector::ScalableExtractOp>(loc, nxv16i8, rhs, j * 8));

    // Handy types for packing/unpacking of the accumulator tile.
    auto accRowTy = VectorType::get(/*shape=*/N, rewriter.getI32Type(),
                                    /*scalableDims=*/{true});
    auto accRowX2Ty = VectorType::get(/*shape=*/2 * N, rewriter.getI32Type(),
                                      /*scalableDims=*/{true});
    auto accRow64Ty = VectorType::get(/*shape=*/N / 2, rewriter.getI64Type(),
                                      /*scalableDims=*/{true});
    auto accRowX264Ty = VectorType::get(/*shape=*/N, rewriter.getI64Type(),
                                        /*scalableDims=*/{true});

    // Extract and pack the ACC sub-tiles.
    SmallVector<Value> accTile;
    for (int64_t i = 0; i < M; i += 2) {
      // Extract two consecutive rows of the accumulator tile.
      auto r0 = rewriter.create<vector::ExtractOp>(loc, op.getAcc(),
                                                   ArrayRef<int64_t>{i});
      auto r1 = rewriter.create<vector::ExtractOp>(loc, op.getAcc(),
                                                   ArrayRef<int64_t>{i + 1});
      Value accTileVec;
      if (mmlaOp == MMLA::MixedSwapped) {
        // We need to swap the positions of the LHS and RHS (since we don't have
        // a signed * unsigned operation), but then each individual 2x2 tile of
        // the acumulator and (later) the result need to be transposed.
        accTileVec = rewriter.create<vector::InterleaveOp>(loc, r0, r1);
      } else {
        // Bitcast them to 64-bit elements, so subsequent
        // interleave/deinterleave work on pairs of 32-bit numbers.
        auto r0I64 = rewriter.create<vector::BitCastOp>(loc, accRow64Ty, r0);
        auto r1I64 = rewriter.create<vector::BitCastOp>(loc, accRow64Ty, r1);

        // Interleave the rows, effectively flattening each 2x2 tile into 4
        // consecutive elements.
        auto intrI64 = rewriter.create<vector::InterleaveOp>(loc, r0I64, r1I64);

        // Bitcast back to 32-bit elements.
        accTileVec =
            rewriter.create<vector::BitCastOp>(loc, accRowX2Ty, intrI64);
      }
      // Extract ACC sub-tiles.
      for (int64_t j = 0; j < N; j += 2)
        accTile.push_back(rewriter.create<vector::ScalableExtractOp>(
            loc, nxv4i32, accTileVec, j * 2));
    }

    // Emit sub-tile matrix multiplications.
    SmallVector<Value> outTile;
    for (int64_t i = 0; i < M / 2; ++i)
      for (int64_t j = 0; j < N / 2; ++j) {
        Value mmla = createMMLA(rewriter, mmlaOp, loc, nxv4i32,
                                accTile[i * N / 2 + j], lhsTile[i], rhsTile[j]);
        outTile.push_back(mmla);
      }

    // Unpack the OUT sub-tiles and insert into the result.
    Value result = rewriter.create<ub::PoisonOp>(loc, op.getResultType());
    for (int64_t i = 0; i < M / 2; ++i) {
      // Collect a number of sub-tiles in a row.
      Value row = rewriter.create<ub::PoisonOp>(loc, accRowX2Ty);
      for (int64_t j = 0; j < N / 2; ++j)
        row = rewriter.create<vector::ScalableInsertOp>(
            loc, outTile[i * N / 2 + j], row, j * 4);

      // Unpack the row to obtain two rows of the output. If we have the out
      // sub-tiles transposed we obtain two consecutive output rows by
      // separating even and odd elements, i.e. a simple deinterleave.
      // Otherwise, the interleave is by pairs.
      Value out0, out1;
      if (mmlaOp == MMLA::MixedSwapped) {
        auto tmp = rewriter.create<vector::DeinterleaveOp>(loc, row);
        out0 = tmp.getRes1();
        out1 = tmp.getRes2();
      } else {
        // Deinterleave by pairs.
        auto row64 = rewriter.create<vector::BitCastOp>(loc, accRowX264Ty, row);
        auto deintr64 = rewriter.create<vector::DeinterleaveOp>(loc, row64);

        // Bitcast back into 32-bit elements and insert into the result.
        out0 = rewriter.create<vector::BitCastOp>(loc, accRowTy,
                                                  deintr64.getRes1());
        out1 = rewriter.create<vector::BitCastOp>(loc, accRowTy,
                                                  deintr64.getRes2());
      }
      result = rewriter.create<vector::InsertOp>(loc, out0, result, i * 2);
      result = rewriter.create<vector::InsertOp>(loc, out1, result, i * 2 + 1);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::populateLowerContractionToSVEI8MMPatternPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToSVEI8MMPattern>(context, /*benefit=*/2);
}
