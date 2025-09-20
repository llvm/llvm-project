//===- LowerContractToSVEPatterns.cpp - Contract to I8MM/BF16 ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from vector.contract to operations
// that map to instructions from the SVE FEAT_I8MM and FEAT_BF16 extensions.
//
// TODO: There may be opportunities to unify this with a similar pattern
// for Neon. See:
//   https://github.com/llvm/llvm-project/issues/145559
//   LowerContractToNeonPatterns.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#include <cassert>
#include <numeric>

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
  auto extOp = v.getDefiningOp<Op>();
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

/// This class encapsulates the algorithm and parametrisation (in terms of types
/// and dimensions) of lowering a `vector.contract` to "primitive" matrix
/// multiplication operations of the SVE dialect (here "primitive" would mean
/// corresponding to a single target instruction).
///
/// Supported are lowering to FEAT_I8MM `smmla`, `ummla`, and `usmmla`, and to
/// FEAT_BF16 `bfmmla`. All the transformations are very similar to each other
/// for concreteness the description below is given for `smmla`.
///
/// The lowering triggers for a contraction operation that performs a matrix
/// multiply of two 8-bit integer matrix tiles with logical dimensions
/// <Mx8> and <8x[N]> for the left-hand side (LHS) and the right-hand side
/// (RHS), respectively, added to a 32-bit integer accumulator operand (ACC)
/// with dimensions <Mx[N]>, yielding a <Mx[N]> 32-bit integer result (OUT).
///
/// The operands' shapes are such that the operands can be evenly split into
/// sub-tiles with dimensions as expected by the targeted FEAT_I8MM
/// instructions. The intent is that M and N are chosen (by higher level
/// transforms) in such a way as to maximise register usage. The main use case
/// we envision as of now is MMT4D, thus the RHS operand is expected
/// pre-transposed.
///
/// The matrix multiplication is performed by unrolling the usual tiled matrix
/// multiplication algorithm using sub-tiles with dimensions <2x8> for the
/// LHS, <8x[2]> for the RHS, and <2x[2]> for the result and the input
/// accumulator.
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
/// The RHS operand is unpacked into N/2 values, each representing a sequence
/// of VSCALE number of sub-tiles with dimensions <8x2>.
/// The LHS operand is initially unpacked into M/2 values, each representing a
/// sub-tile with dimensions <2x8>, and then each such sub-tile is replicated
/// VSCALE times. Multiplying thus replicated LHS sub-tile by the corresponding
/// RHS sub-tile correctly computes an entire result sub-tile.
/// The 2x2 sub-tiles of the ACC and OUT have rows that are not adjacent
/// (in memory or when imposing a row-major layout on the 2D vector value).
/// Reading the ACC is implemented as reading two consecutive rows and
/// interleaving the by pairs to obtain a vector having length twice the length
/// of an ACC row. This vector now is a sequence of one-dimensional tiles with
/// the exact layout needed by the `smmla`/`bfmmla`/etc instructions, which
/// tiles are extracted one by one. For illustration, if we have an 2x4 ACC tile
///   a0 a1 b0 b1
///   a2 a3 b2 b3
/// we read the two rows as separate values and then interleave by pairs
/// to obtain
///   a0 a1 a2 a3 b0 b1 b2 b3
/// from which we extract `a0 a1 a2 a3` and `b0 b1 b2 b3`.
///
/// Writing the OUT tile is done by the reverse of the above procedure,
/// concatenate two "flattened" sub-tiles into
///   c0 c1 c2 c3 d0 d1 d2 d3
/// deinterleave by pairs to obtain as separate values
///   c0 c1 d0 d1
///   c2 c3 d2 d3
/// which are then inserted into the final result.
///
/// Multiplication of a signed LHS by an unsigned LHS is performed by
/// swapping the order of the operands and emitting an `usmmla` (since there
/// isn't an `summla` instruction). Therefore each ACC sub-tile needs
/// to be transposed before the addition and the sum, an OUT sub-tile,
/// needs to be transposed before insertion into the final result.
/// This is done very elegantly by a modification of the above to
/// interleave/deinterleave not by pairs, but by individual elements, e.g.
/// after ordinary interleave we obtain
///   a0 a2 a1 a3 b0 b2 b1 b3
/// which is exactly the desired layout of having each individual 2x2 tile
/// transposed.
///
/// All of the above readily applies to FEAT_BF16 `bfmmla` with the
/// difference that the shapes of the LHS, RHS are <Mx4>, <4x[M]>, and
/// respectively, that is the "K" dimension is fixed to 4, instead of 8 (like
/// for the integer case).
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

  // Indicate if the operands for the ArmSVE dialect operation need to be
  // swapped. Currently this is needed in order to emulate an "summla"
  // operation.
  bool swapOperands = false;

  // The operand tiles. These are not necessarily the operends of
  // `vector.contract`, for example they could be operands to `arith.extsi`
  // that is in turn fed into `vector.contract`.
  Value lhs;
  Value rhs;
  Value acc;

  // Conventional names for matrix dimensions.
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  // Create the matrix mulitply and accumulate operation according to
  // `mmlaOp`.
  Value createMMLA(PatternRewriter &rewriter, Location loc, Value acc,
                   Value lhs, Value rhs);

  // Check general preconditions for applying the transformation, common to the
  // integer and the bfloat16 case.
  LogicalResult match(vector::ContractionOp op, PatternRewriter &rewriter);

public:
  VectorContractRewriter() = default;

  // Do the actuall rewrite. This member function is shared by both integer and
  // bfloat16 rewrites.
  Value lower(vector::ContractionOp op, PatternRewriter &rewriter);
};

Value VectorContractRewriter::createMMLA(PatternRewriter &rewriter,
                                         Location loc, Value acc, Value lhs,
                                         Value rhs) {

  Type resTy = acc.getType();
  if (swapOperands)
    std::swap(lhs, rhs);

  switch (mmlaOp) {
  case MMLA::SignedInt:
    return arm_sve::SmmlaOp::create(rewriter, loc, resTy, acc, lhs, rhs);
  case MMLA::UnsignedInt:
    return arm_sve::UmmlaOp::create(rewriter, loc, resTy, acc, lhs, rhs);
  case MMLA::MixedInt:
    return arm_sve::UsmmlaOp::create(rewriter, loc, resTy, acc, lhs, rhs);
  case MMLA::Bfloat:
    return arm_sve::BfmmlaOp::create(rewriter, loc, resTy, acc, lhs, rhs);
  default:
    llvm_unreachable("Uninitialized operation kind");
  }
}

LogicalResult VectorContractRewriter::match(vector::ContractionOp op,
                                            PatternRewriter &rewriter) {
  // Check iterator types for matrix multiplication.
  auto itTypes = op.getIteratorTypesArray();
  if (itTypes.size() != 3 || itTypes[0] != vector::IteratorType::parallel ||
      itTypes[1] != vector::IteratorType::parallel ||
      itTypes[2] != vector::IteratorType::reduction)
    return rewriter.notifyMatchFailure(
        op, "iterator types do not correspond to matrix multiplication");

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
      op.getIndexingMapsArray()[2] != AffineMap::getMultiDimMapWithTargets(
                                          3, ArrayRef{0u, 1u}, op.getContext()))
    return rewriter.notifyMatchFailure(op, "non-matching permutation maps");

  // Check the combining kind is addition.
  if (op.getKind() != vector::CombiningKind::ADD)
    return rewriter.notifyMatchFailure(op, "combining kind is not an addition");

  return success();
}

Value VectorContractRewriter::lower(vector::ContractionOp op,
                                    PatternRewriter &rewriter) {

  // Initialize some helper types.
  Type operandEltType = cast<VectorType>(lhs.getType()).getElementType();
  Type resultEltType = cast<VectorType>(op.getResultType()).getElementType();

  const int64_t numOperandSubTileElts =
      128 / operandEltType.getIntOrFloatBitWidth();

  assert(resultEltType.getIntOrFloatBitWidth() == 32 &&
         "Only implemented for i32 or f32 output");
  const int64_t numResultSubTileElts = 4;

  // Single-dimensional vector types for the operands of the ArmSVE dialect
  // op.
  auto flatLhsType =
      VectorType::get(/*shape=*/numOperandSubTileElts, operandEltType,
                      /*scalableDims=*/{true});
  auto flatRhsType =
      VectorType::get(/*shape=*/numOperandSubTileElts, operandEltType,
                      /*scalableDims=*/{true});
  auto flatAccType =
      VectorType::get(/*shape=*/numResultSubTileElts, resultEltType,
                      /*scalableDims=*/{true});

  // Single-dimension vector type for the entire RHS tile.

  auto flatRhsTileType = VectorType::get(/*shape=*/k * n, operandEltType,
                                         /*scalableDims=*/{true});

  // Vector type having the same number of elements as a row in the
  // accumulator/output tile and the same element type.
  auto accRowTy = VectorType::get(/*shape=*/n, resultEltType,
                                  /*scalableDims=*/{true});

  // Vector type having twice the number of elements as a row in the
  // accumulator/output tile the same element type.
  auto accRowX2Ty = VectorType::get(/*shape=*/2 * n, resultEltType,
                                    /*scalableDims=*/{true});
  // Vector type having half the number of elements as a row in the
  // accumulator/output tile and an integer element type with twice the bit
  // width.
  auto accRow64Ty = VectorType::get(/*shape=*/n / 2, rewriter.getI64Type(),
                                    /*scalableDims=*/{true});
  // Vector type having the same the number of elements as a row in the
  // accumulator/output tile and an integer element type with twice the bit
  // width.
  auto accRowX264Ty = VectorType::get(/*shape=*/n, rewriter.getI64Type(),
                                      /*scalableDims=*/{true});

  Location loc = op.getLoc();

  // Extract LHS sub-tiles with logical shape <2xK>.
  SmallVector<Value> lhsTile;
  for (int64_t i = 0; i < m; i += 2) {
    // Extract two consecutive rows of the LHS tile.
    auto r0 =
        vector::ExtractOp::create(rewriter, loc, lhs, ArrayRef<int64_t>{i});
    auto r1 =
        vector::ExtractOp::create(rewriter, loc, lhs, ArrayRef<int64_t>{i + 1});
    // Concatenate to obtain a 2 x K x <input-type> flattened sub-tile.
    SmallVector<int64_t> shuffleIdx(2 * k);
    std::iota(shuffleIdx.begin(), shuffleIdx.end(), 0);
    auto t = vector::ShuffleOp::create(rewriter, loc, r0, r1, shuffleIdx);
    // Turn it into a scalable vector.
    auto s = vector::ScalableInsertOp::create(
        rewriter, loc, t, ub::PoisonOp::create(rewriter, loc, flatLhsType), 0);
    // Replicate the sub-tile VSCALE times to fill the entire vector.
    auto r = arm_sve::DupQLaneOp::create(rewriter, loc, s, 0);
    lhsTile.push_back(r);
  }

  // "Flatten" the RHS tile from <[N]xK> to <[N*K]>.
  auto rhs = vector::ShapeCastOp::create(rewriter, this->rhs.getLoc(),
                                         flatRhsTileType, this->rhs);

  // Extract the RHS sub-tiles with logical shape <Kx[2]>.
  SmallVector<Value> rhsTile;
  for (int64_t j = 0; j < n; j += 2)
    rhsTile.push_back(vector::ScalableExtractOp::create(
        rewriter, loc, flatRhsType, rhs, j * k));

  // Extract and pack the ACC sub-tiles.
  SmallVector<Value> accTile;
  for (int64_t i = 0; i < m; i += 2) {
    // Extract two consecutive rows of the accumulator tile.
    auto r0 = vector::ExtractOp::create(rewriter, loc, op.getAcc(),
                                        ArrayRef<int64_t>{i});
    auto r1 = vector::ExtractOp::create(rewriter, loc, op.getAcc(),
                                        ArrayRef<int64_t>{i + 1});
    Value accTileVec;
    if (swapOperands) {
      // We are performing the operation with swapped LHS and RHS we need to
      // transpose each individual 2x2 tile of the accumulator and (later) the
      // final result.
      accTileVec = vector::InterleaveOp::create(rewriter, loc, r0, r1);
    } else {
      // Bitcast accumulator rows to double-width integer elements, so
      // subsequent interleave/deinterleave work on pairs of elements.
      auto r0I64 = vector::BitCastOp::create(rewriter, loc, accRow64Ty, r0);
      auto r1I64 = vector::BitCastOp::create(rewriter, loc, accRow64Ty, r1);

      // Interleave the rows, effectively flattening each 2x2 tile into 4
      // consecutive elements.
      auto intrI64 = vector::InterleaveOp::create(rewriter, loc, r0I64, r1I64);

      // Bitcast back to original element type.
      accTileVec =
          vector::BitCastOp::create(rewriter, loc, accRowX2Ty, intrI64);
    }
    // Extract ACC sub-tiles.
    for (int64_t j = 0; j < n; j += 2)
      accTile.push_back(vector::ScalableExtractOp::create(
          rewriter, loc, flatAccType, accTileVec, j * 2));
  }

  // Emit sub-tile matrix multiplications.
  SmallVector<Value> outTile;
  for (int64_t i = 0; i < m / 2; ++i)
    for (int64_t j = 0; j < n / 2; ++j) {
      Value mmla = createMMLA(rewriter, loc, accTile[i * n / 2 + j], lhsTile[i],
                              rhsTile[j]);
      outTile.push_back(mmla);
    }

  // Unpack the OUT sub-tiles and insert into the result.
  Value result = ub::PoisonOp::create(rewriter, loc, op.getResultType());
  for (int64_t i = 0; i < m / 2; ++i) {
    // Collect a number of sub-tiles in a row.
    Value row = ub::PoisonOp::create(rewriter, loc, accRowX2Ty);
    for (int64_t j = 0; j < n / 2; ++j)
      row = vector::ScalableInsertOp::create(
          rewriter, loc, outTile[i * n / 2 + j], row, j * 4);

    // Unpack the row to obtain two rows of the output. If we have the out
    // sub-tiles transposed we obtain two consecutive output rows by
    // separating even and odd elements, i.e. a simple deinterleave.
    // Otherwise, the interleave is by pairs.
    Value out0, out1;
    if (swapOperands) {
      auto tmp = vector::DeinterleaveOp::create(rewriter, loc, row);
      out0 = tmp.getRes1();
      out1 = tmp.getRes2();
    } else {
      // Deinterleave by pairs.
      auto row64 = vector::BitCastOp::create(rewriter, loc, accRowX264Ty, row);
      auto deintr64 = vector::DeinterleaveOp::create(rewriter, loc, row64);

      // Bitcast back into original element type and insert into the result.
      out0 = vector::BitCastOp::create(rewriter, loc, accRowTy,
                                       deintr64.getRes1());
      out1 = vector::BitCastOp::create(rewriter, loc, accRowTy,
                                       deintr64.getRes2());
    }
    result = vector::InsertOp::create(rewriter, loc, out0, result, i * 2);
    result = vector::InsertOp::create(rewriter, loc, out1, result, i * 2 + 1);
  }

  return result;
}

class VectorContractRewriterI8MM : public VectorContractRewriter {
public:
  // Check the specific preconditions for the integer case. Initialise
  // parametrisation types and dimensions.
  LogicalResult matchAndInit(vector::ContractionOp op,
                             PatternRewriter &rewriter) {
    if (failed(match(op, rewriter)))
      return failure();

    VectorType lhsType = op.getLhsType();
    VectorType rhsType = op.getRhsType();

    m = lhsType.getDimSize(0);
    n = rhsType.getDimSize(0);
    k = rhsType.getDimSize(1);

    // Check the operands have the expected shape:
    //  * for LHS: fixed vector MxK
    //  * for RHS: scalable vector [N]xK
    //  * K == 8
    //  * M and N even and at least 2
    if (lhsType.isScalable() || !rhsType.getScalableDims()[0] ||
        rhsType.getScalableDims()[1] || lhsType.getDimSize(1) != k || k != 8 ||
        m < 2 || m % 2 != 0 || n < 2 || n % 2 != 0 ||
        !rhsType.getScalableDims()[0])
      return rewriter.notifyMatchFailure(op, "non-matching operand shape");

    // Check the output is a vector of i32 elements.
    auto outTy = dyn_cast<VectorType>(op.getResultType());
    if (!outTy || outTy.getElementType() != rewriter.getI32Type())
      return rewriter.notifyMatchFailure(op,
                                         "output type is not a vector of i32");

    // Check inputs are sign-/zero- extensions from i8 to i32. Get the values
    // before the extension. All four signed/unsigned combinations for input
    // operands are supported, but they are lowered to different operations.
    // Determine which is the appropriate operation to lower to.
    mmlaOp = MMLA::SignedInt;
    swapOperands = false;
    auto maybeLhs = getExtOperand<arith::ExtSIOp>(op.getLhs());
    if (!maybeLhs) {
      mmlaOp = MMLA::UnsignedInt;
      maybeLhs = getExtOperand<arith::ExtUIOp>(op.getLhs());
    }
    if (!maybeLhs)
      return rewriter.notifyMatchFailure(
          op, "LHS is not a sign- or zero- extended i8");

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
          op, "RHS is not a sign- or zero- extended i8");

    // Initialise algorithm parameters.
    lhs = *maybeLhs;
    rhs = *maybeRhs;
    acc = op.getAcc();

    return success();
  }
};

class VectorContractRewriterBfloat : public VectorContractRewriter {
public:
  // Check the specific preconditions for the bfloat16 case. Initialise
  // parametrisation types and dimensions.
  LogicalResult matchAndInit(vector::ContractionOp op,
                             PatternRewriter &rewriter) {
    if (failed(match(op, rewriter)))
      return failure();

    VectorType lhsType = op.getLhsType();
    VectorType rhsType = op.getRhsType();

    m = lhsType.getDimSize(0);
    n = rhsType.getDimSize(0);
    k = rhsType.getDimSize(1);

    // Check the operands have the expected shape:
    //  * for LHS: fixed vector MxK
    //  * for RHS: scalable vector [N]xK
    //  * K == 4
    //  * M and N even and at least 2
    if (lhsType.isScalable() || !rhsType.getScalableDims()[0] ||
        rhsType.getScalableDims()[1] || lhsType.getDimSize(1) != k || k != 4 ||
        m < 2 || m % 2 != 0 || n < 2 || n % 2 != 0 ||
        !rhsType.getScalableDims()[0])
      return rewriter.notifyMatchFailure(op, "non-matching operand shape");

    // Check the output is a vector of Float32 elements.
    auto outTy = dyn_cast<VectorType>(op.getResultType());
    if (!outTy || outTy.getElementType() != rewriter.getF32Type())
      return rewriter.notifyMatchFailure(op,
                                         "output type is not a vector of f32");

    // Check the inputs are vectors of BFloat16 elements.
    if (lhsType.getElementType() != rewriter.getBF16Type())
      return rewriter.notifyMatchFailure(op,
                                         "input type is not a vector of bf16");

    // Initialise algorithm parameters.
    mmlaOp = MMLA::Bfloat;
    swapOperands = false;
    lhs = op.getLhs();
    rhs = op.getRhs();
    acc = op.getAcc();

    return success();
  }
};

class LowerContractionToSVEI8MMPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    // Match i8xi8 -> i32 matrix multiply and accumulate.
    VectorContractRewriterI8MM vcr;
    if (failed(vcr.matchAndInit(op, rewriter)))
      return failure();

    Value result = vcr.lower(op, rewriter);
    rewriter.replaceOp(op, result);

    return success();
  }
};

class LowerContractionToSVEBFMMLAPattern
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {

    // Match bf16xbf16 -> f32 matrix multiply and accumulate.
    VectorContractRewriterBfloat vcr;
    if (failed(vcr.matchAndInit(op, rewriter)))
      return failure();

    Value result = vcr.lower(op, rewriter);
    rewriter.replaceOp(op, result);

    return success();
  }
};

} // namespace

void mlir::populateLowerContractionToSVEI8MMPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToSVEI8MMPattern>(context, /*benefit=*/2);
}

void mlir::populateLowerContractionToSVEBFMMLAPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerContractionToSVEBFMMLAPattern>(context, /*benefit=*/2);
}
