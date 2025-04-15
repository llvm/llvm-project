//===- LowerContractionToSMMLAPattern.cpp - Contract to SMMLA ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from vector.contract to
// SVE I8MM operations.
//
//===---

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/UB/IR/UBOps.h"

#define DEBUG_TYPE "lower-contract-to-arm-sve-i8mm"

using namespace mlir;
using namespace mlir::arm_sve;

namespace {
// Check if the given value is a result of the operation `T` (which must be
// sign- or zero- extend) from i8 to i32. Return the value before the extension.
template <typename T>
inline std::enable_if_t<(std::is_base_of_v<arith::ExtSIOp, T> ||
                         std::is_base_of_v<arith::ExtUIOp, T>),
                        std::optional<Value>>
extractExtOperand(Value v, Type i8Ty, Type i32Ty) {
  auto extOp = dyn_cast_or_null<T>(v.getDefiningOp());
  if (!extOp)
    return {};

  auto inOp = extOp.getIn();
  auto inTy = dyn_cast<VectorType>(inOp.getType());
  if (!inTy || inTy.getElementType() != i8Ty)
    return {};

  auto outTy = dyn_cast<VectorType>(extOp.getType());
  if (!outTy || outTy.getElementType() != i32Ty)
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

// Create the matrix multply and accumulate operation according to `op`.
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

class LowerContractionToSVEI8MMPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    mlir::VectorType lhsType = op.getLhsType();
    mlir::VectorType rhsType = op.getRhsType();

    // For now handle LHS<Mx8> and RHS<8x[N]> - these are the types we
    // eventually expect from MMT4D. M and N dimensions must be even and at
    // least 2.
    if (!lhsType.hasRank() || lhsType.getRank() != 2 || !rhsType.hasRank() ||
        rhsType.getRank() != 2)
      return failure();

    if (lhsType.isScalable() || !rhsType.isScalable())
      return failure();

    // M, N, and K are the conventional names for matrix dimensions in the
    // context of matrix multiplication.
    auto M = lhsType.getDimSize(0);
    auto N = rhsType.getDimSize(0);
    auto K = rhsType.getDimSize(1);

    if (lhsType.getDimSize(1) != K || K != 8 || M < 2 || M % 2 != 0 || N < 2 ||
        N % 2 != 0 || !rhsType.getScalableDims()[0])
      return failure();

    // Check permutation maps. For now only accept
    //   lhs: (d0, d1, d2) -> (d0, d2)
    //   rhs: (d0, d1, d2) -> (d1, d2)
    //   acc: (d0, d1, d2) -> (d0, d1)
    // Note: RHS is transposed.
    if (op.getIndexingMapsArray()[0] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{0u, 2u},
                                                 op.getContext()) ||
        op.getIndexingMapsArray()[1] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{1u, 2u},
                                                 op.getContext()) ||
        op.getIndexingMapsArray()[2] !=
            AffineMap::getMultiDimMapWithTargets(3, ArrayRef{0u, 1u},
                                                 op.getContext()))
      return failure();

    // Check iterator types for matrix multiplication.
    auto itTypes = op.getIteratorTypesArray();
    if (itTypes.size() != 3 || itTypes[0] != vector::IteratorType::parallel ||
        itTypes[1] != vector::IteratorType::parallel ||
        itTypes[2] != vector::IteratorType::reduction)
      return failure();

    // Check the combining kind is addition.
    if (op.getKind() != vector::CombiningKind::ADD)
      return failure();

    // Check the output is a vector of i32 elements.
    auto outTy = dyn_cast<VectorType>(op.getType());
    if (!outTy || outTy.getElementType() != rewriter.getI32Type())
      return failure();

    // Check inputs are sign-/zero- extensions from i8 to i32. Get the values
    // before the extension. All four signed/unsigned combinations for input
    // operands are supported, but they are lowered to different operations.
    // Determina which is the appropriate operation to lower to.
    MMLA mmlaOp = MMLA::Signed;
    auto maybeLhs = extractExtOperand<arith::ExtSIOp>(
        op.getLhs(), rewriter.getI8Type(), rewriter.getI32Type());
    if (!maybeLhs) {
      mmlaOp = MMLA::Unsigned;
      maybeLhs = extractExtOperand<arith::ExtUIOp>(
          op.getLhs(), rewriter.getI8Type(), rewriter.getI32Type());
    }
    if (!maybeLhs)
      return failure();

    auto maybeRhs = extractExtOperand<arith::ExtSIOp>(
        op.getRhs(), rewriter.getI8Type(), rewriter.getI32Type());
    if (maybeRhs) {
      if (mmlaOp == MMLA::Unsigned)
        mmlaOp = MMLA::Mixed;
    } else {
      if (mmlaOp == MMLA::Signed)
        mmlaOp = MMLA::MixedSwapped;
      maybeRhs = extractExtOperand<arith::ExtUIOp>(
          op.getRhs(), rewriter.getI8Type(), rewriter.getI32Type());
    }
    if (!maybeRhs)
      return failure();

    // One-dimensional vector types for arm_sve.*mmla
    auto nxv16i8 = VectorType::get(16, rewriter.getI8Type(), {true});
    auto nxv4i32 = VectorType::get(4, rewriter.getI32Type(), {true});

    // Extract LHS sub-tiles.
    SmallVector<Value> lhsTile;
    for (int64_t i = 0; i < M; i += 2) {
      // Exract two consective rows of the LHS tile.
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
    auto RHS = rewriter.create<vector::ShapeCastOp>(
        maybeRhs->getLoc(),
        VectorType::get(8 * N, rewriter.getI8Type(), {true}), *maybeRhs);

    // Extract the RHS sub-tiles.
    SmallVector<Value> rhsTile;
    for (int64_t j = 0; j < N; j += 2)
      rhsTile.push_back(
          rewriter.create<vector::ScalableExtractOp>(loc, nxv16i8, RHS, j * 8));

    // Handy types for packing/unpacking of the accumulator tile.
    auto accRowTy = VectorType::get(N, rewriter.getI32Type(), {true});
    auto accRowX2Ty = VectorType::get(2 * N, rewriter.getI32Type(), {true});
    auto accRow64Ty = VectorType::get(N / 2, rewriter.getI64Type(), {true});
    auto accRowX264Ty = VectorType::get(N, rewriter.getI64Type(), {true});

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
        auto r0_i64 = rewriter.create<vector::BitCastOp>(loc, accRow64Ty, r0);
        auto r1_i64 = rewriter.create<vector::BitCastOp>(loc, accRow64Ty, r1);

        // Interleave the rows, effectively flattening each 2x2 tile into 4
        // consecutive elements.
        auto intr_i64 =
            rewriter.create<vector::InterleaveOp>(loc, r0_i64, r1_i64);

        // Bitcast back to 32-bit elements.
        accTileVec =
            rewriter.create<vector::BitCastOp>(loc, accRowX2Ty, intr_i64);
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
