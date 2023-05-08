//===- LowerVectorTranspose.cpp - Lower 'vector.transpose' operation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.transpose' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "lower-vector-transpose"

using namespace mlir;
using namespace mlir::vector;

/// Given a 'transpose' pattern, prune the rightmost dimensions that are not
/// transposed.
static void pruneNonTransposedDims(ArrayRef<int64_t> transpose,
                                   SmallVectorImpl<int64_t> &result) {
  size_t numTransposedDims = transpose.size();
  for (size_t transpDim : llvm::reverse(transpose)) {
    if (transpDim != numTransposedDims - 1)
      break;
    numTransposedDims--;
  }

  result.append(transpose.begin(), transpose.begin() + numTransposedDims);
}

/// Returns true if the lowering option is a vector shuffle based approach.
static bool isShuffleLike(VectorTransposeLowering lowering) {
  return lowering == VectorTransposeLowering::Shuffle1D ||
         lowering == VectorTransposeLowering::Shuffle16x16;
}

/// Returns a shuffle mask that builds on `vals`. `vals` is the offset base of
/// shuffle ops, i.e., the unpack pattern. The method iterates with `vals` to
/// create the mask for `numBits` bits vector. The `numBits` have to be a
/// multiple of 128. For example, if `vals` is {0, 1, 16, 17} and `numBits` is
/// 512, there should be 16 elements in the final result. It constructs the
/// below mask to get the unpack elements.
///   [0,    1,    16,    17,
///    0+4,  1+4,  16+4,  17+4,
///    0+8,  1+8,  16+8,  17+8,
///    0+12, 1+12, 16+12, 17+12]
static SmallVector<int64_t>
getUnpackShufflePermFor128Lane(ArrayRef<int64_t> vals, int numBits) {
  assert(numBits % 128 == 0 && "expected numBits is a multiple of 128");
  int numElem = numBits / 32;
  SmallVector<int64_t> res;
  for (int i = 0; i < numElem; i += 4)
    for (int64_t v : vals)
      res.push_back(v + i);
  return res;
}

/// Lower to vector.shuffle on v1 and v2 with UnpackLoPd shuffle mask. For
/// example, if it is targeting 512 bit vector, returns
///   vector.shuffle on v1, v2, [0,    1,    16,    17,
///                              0+4,  1+4,  16+4,  17+4,
///                              0+8,  1+8,  16+8,  17+8,
///                              0+12, 1+12, 16+12, 17+12].
static Value createUnpackLoPd(ImplicitLocOpBuilder &b, Value v1, Value v2,
                              int numBits) {
  int numElem = numBits / 32;
  return b.create<vector::ShuffleOp>(
      v1, v2,
      getUnpackShufflePermFor128Lane({0, 1, numElem, numElem + 1}, numBits));
}

/// Lower to vector.shuffle on v1 and v2 with UnpackHiPd shuffle mask. For
/// example, if it is targeting 512 bit vector, returns
///   vector.shuffle, v1, v2, [2,    3,    18,    19,
///                            2+4,  3+4,  18+4,  19+4,
///                            2+8,  3+8,  18+8,  19+8,
///                            2+12, 3+12, 18+12, 19+12].
static Value createUnpackHiPd(ImplicitLocOpBuilder &b, Value v1, Value v2,
                              int numBits) {
  int numElem = numBits / 32;
  return b.create<vector::ShuffleOp>(
      v1, v2,
      getUnpackShufflePermFor128Lane({2, 3, numElem + 2, numElem + 3},
                                     numBits));
}

/// Lower to vector.shuffle on v1 and v2 with UnpackLoPs shuffle mask. For
/// example, if it is targeting 512 bit vector, returns
///   vector.shuffle, v1, v2, [0,    16,    1,    17,
///                            0+4,  16+4,  1+4,  17+4,
///                            0+8,  16+8,  1+8,  17+8,
///                            0+12, 16+12, 1+12, 17+12].
static Value createUnpackLoPs(ImplicitLocOpBuilder &b, Value v1, Value v2,
                              int numBits) {
  int numElem = numBits / 32;
  auto shuffle = b.create<vector::ShuffleOp>(
      v1, v2,
      getUnpackShufflePermFor128Lane({0, numElem, 1, numElem + 1}, numBits));
  return shuffle;
}

/// Lower to vector.shuffle on v1 and v2 with UnpackHiPs shuffle mask. For
/// example, if it is targeting 512 bit vector, returns
///   vector.shuffle, v1, v2, [2,    18,    3,    19,
///                            2+4,  18+4,  3+4,  19+4,
///                            2+8,  18+8,  3+8,  19+8,
///                            2+12, 18+12, 3+12, 19+12].
static Value createUnpackHiPs(ImplicitLocOpBuilder &b, Value v1, Value v2,
                              int numBits) {
  int numElem = numBits / 32;
  return b.create<vector::ShuffleOp>(
      v1, v2,
      getUnpackShufflePermFor128Lane({2, numElem + 2, 3, numElem + 3},
                                     numBits));
}

/// Returns a vector.shuffle that shuffles 128-bit lanes (composed of 4 32-bit
/// elements) selected by `mask` from `v1` and `v2`. I.e.,
///
/// DEFINE SELECT4(src, control) {
///	CASE(control[1:0]) OF
///	0:	tmp[127:0] := src[127:0]
///	1:	tmp[127:0] := src[255:128]
///	2:	tmp[127:0] := src[383:256]
///	3:	tmp[127:0] := src[511:384]
///	ESAC
///	RETURN tmp[127:0]
/// }
/// dst[127:0]   := SELECT4(v1[511:0], mask[1:0])
/// dst[255:128] := SELECT4(v1[511:0], mask[3:2])
/// dst[383:256] := SELECT4(v2[511:0], mask[5:4])
/// dst[511:384] := SELECT4(v2[511:0], mask[7:6])
static Value create4x128BitSuffle(ImplicitLocOpBuilder &b, Value v1, Value v2,
                                  uint8_t mask) {
  assert(cast<VectorType>(v1.getType()).getShape()[0] == 16 &&
         "expected a vector with length=16");
  SmallVector<int64_t> shuffleMask;
  auto appendToMask = [&](int64_t base, uint8_t control) {
    switch (control) {
    case 0:
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{base + 0, base + 1,
                                                        base + 2, base + 3});
      break;
    case 1:
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{base + 4, base + 5,
                                                        base + 6, base + 7});
      break;
    case 2:
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{base + 8, base + 9,
                                                        base + 10, base + 11});
      break;
    case 3:
      llvm::append_range(shuffleMask, ArrayRef<int64_t>{base + 12, base + 13,
                                                        base + 14, base + 15});
      break;
    default:
      llvm_unreachable("control > 3 : overflow");
    }
  };
  uint8_t b01 = mask & 0x3;
  uint8_t b23 = (mask >> 2) & 0x3;
  uint8_t b45 = (mask >> 4) & 0x3;
  uint8_t b67 = (mask >> 6) & 0x3;
  appendToMask(0, b01);
  appendToMask(0, b23);
  appendToMask(16, b45);
  appendToMask(16, b67);
  return b.create<vector::ShuffleOp>(v1, v2, shuffleMask);
}

/// Lowers the value to a vector.shuffle op. The `source` is expected to be a
/// 1-D vector and have `m`x`n` elements.
static Value transposeToShuffle1D(OpBuilder &b, Value source, int m, int n) {
  SmallVector<int64_t> mask;
  mask.reserve(m * n);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < m; ++i)
      mask.push_back(i * n + j);
  return b.create<vector::ShuffleOp>(source.getLoc(), source, source, mask);
}

/// Lowers the value to a sequence of vector.shuffle ops. The `source` is
/// expected to be a 16x16 vector.
static Value transposeToShuffle16x16(OpBuilder &builder, Value source, int m,
                                     int n) {
  ImplicitLocOpBuilder b(source.getLoc(), builder);
  SmallVector<Value> vs;
  for (int64_t i = 0; i < m; ++i)
    vs.push_back(b.create<vector::ExtractOp>(source, i));

  // Interleave 32-bit lanes using
  //   8x _mm512_unpacklo_epi32
  //   8x _mm512_unpackhi_epi32
  Value t0 = createUnpackLoPs(b, vs[0x0], vs[0x1], 512);
  Value t1 = createUnpackHiPs(b, vs[0x0], vs[0x1], 512);
  Value t2 = createUnpackLoPs(b, vs[0x2], vs[0x3], 512);
  Value t3 = createUnpackHiPs(b, vs[0x2], vs[0x3], 512);
  Value t4 = createUnpackLoPs(b, vs[0x4], vs[0x5], 512);
  Value t5 = createUnpackHiPs(b, vs[0x4], vs[0x5], 512);
  Value t6 = createUnpackLoPs(b, vs[0x6], vs[0x7], 512);
  Value t7 = createUnpackHiPs(b, vs[0x6], vs[0x7], 512);
  Value t8 = createUnpackLoPs(b, vs[0x8], vs[0x9], 512);
  Value t9 = createUnpackHiPs(b, vs[0x8], vs[0x9], 512);
  Value ta = createUnpackLoPs(b, vs[0xa], vs[0xb], 512);
  Value tb = createUnpackHiPs(b, vs[0xa], vs[0xb], 512);
  Value tc = createUnpackLoPs(b, vs[0xc], vs[0xd], 512);
  Value td = createUnpackHiPs(b, vs[0xc], vs[0xd], 512);
  Value te = createUnpackLoPs(b, vs[0xe], vs[0xf], 512);
  Value tf = createUnpackHiPs(b, vs[0xe], vs[0xf], 512);

  // Interleave 64-bit lanes using
  //   8x _mm512_unpacklo_epi64
  //   8x _mm512_unpackhi_epi64
  Value r0 = createUnpackLoPd(b, t0, t2, 512);
  Value r1 = createUnpackHiPd(b, t0, t2, 512);
  Value r2 = createUnpackLoPd(b, t1, t3, 512);
  Value r3 = createUnpackHiPd(b, t1, t3, 512);
  Value r4 = createUnpackLoPd(b, t4, t6, 512);
  Value r5 = createUnpackHiPd(b, t4, t6, 512);
  Value r6 = createUnpackLoPd(b, t5, t7, 512);
  Value r7 = createUnpackHiPd(b, t5, t7, 512);
  Value r8 = createUnpackLoPd(b, t8, ta, 512);
  Value r9 = createUnpackHiPd(b, t8, ta, 512);
  Value ra = createUnpackLoPd(b, t9, tb, 512);
  Value rb = createUnpackHiPd(b, t9, tb, 512);
  Value rc = createUnpackLoPd(b, tc, te, 512);
  Value rd = createUnpackHiPd(b, tc, te, 512);
  Value re = createUnpackLoPd(b, td, tf, 512);
  Value rf = createUnpackHiPd(b, td, tf, 512);

  // Permute 128-bit lanes using
  //   16x _mm512_shuffle_i32x4
  t0 = create4x128BitSuffle(b, r0, r4, 0x88);
  t1 = create4x128BitSuffle(b, r1, r5, 0x88);
  t2 = create4x128BitSuffle(b, r2, r6, 0x88);
  t3 = create4x128BitSuffle(b, r3, r7, 0x88);
  t4 = create4x128BitSuffle(b, r0, r4, 0xdd);
  t5 = create4x128BitSuffle(b, r1, r5, 0xdd);
  t6 = create4x128BitSuffle(b, r2, r6, 0xdd);
  t7 = create4x128BitSuffle(b, r3, r7, 0xdd);
  t8 = create4x128BitSuffle(b, r8, rc, 0x88);
  t9 = create4x128BitSuffle(b, r9, rd, 0x88);
  ta = create4x128BitSuffle(b, ra, re, 0x88);
  tb = create4x128BitSuffle(b, rb, rf, 0x88);
  tc = create4x128BitSuffle(b, r8, rc, 0xdd);
  td = create4x128BitSuffle(b, r9, rd, 0xdd);
  te = create4x128BitSuffle(b, ra, re, 0xdd);
  tf = create4x128BitSuffle(b, rb, rf, 0xdd);

  // Permute 256-bit lanes using again
  //   16x _mm512_shuffle_i32x4
  vs[0x0] = create4x128BitSuffle(b, t0, t8, 0x88);
  vs[0x1] = create4x128BitSuffle(b, t1, t9, 0x88);
  vs[0x2] = create4x128BitSuffle(b, t2, ta, 0x88);
  vs[0x3] = create4x128BitSuffle(b, t3, tb, 0x88);
  vs[0x4] = create4x128BitSuffle(b, t4, tc, 0x88);
  vs[0x5] = create4x128BitSuffle(b, t5, td, 0x88);
  vs[0x6] = create4x128BitSuffle(b, t6, te, 0x88);
  vs[0x7] = create4x128BitSuffle(b, t7, tf, 0x88);
  vs[0x8] = create4x128BitSuffle(b, t0, t8, 0xdd);
  vs[0x9] = create4x128BitSuffle(b, t1, t9, 0xdd);
  vs[0xa] = create4x128BitSuffle(b, t2, ta, 0xdd);
  vs[0xb] = create4x128BitSuffle(b, t3, tb, 0xdd);
  vs[0xc] = create4x128BitSuffle(b, t4, tc, 0xdd);
  vs[0xd] = create4x128BitSuffle(b, t5, td, 0xdd);
  vs[0xe] = create4x128BitSuffle(b, t6, te, 0xdd);
  vs[0xf] = create4x128BitSuffle(b, t7, tf, 0xdd);

  auto reshInputType = VectorType::get(
      {m, n}, cast<VectorType>(source.getType()).getElementType());
  Value res =
      b.create<arith::ConstantOp>(reshInputType, b.getZeroAttr(reshInputType));
  for (int64_t i = 0; i < m; ++i)
    res = b.create<vector::InsertOp>(vs[i], res, i);
  return res;
}

namespace {
/// Progressive lowering of TransposeOp.
/// One:
///   %x = vector.transpose %y, [1, 0]
/// is replaced by:
///   %z = arith.constant dense<0.000000e+00>
///   %0 = vector.extract %y[0, 0]
///   %1 = vector.insert %0, %z [0, 0]
///   ..
///   %x = vector.insert .., .. [.., ..]
class TransposeOpLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value input = op.getVector();
    VectorType inputType = op.getSourceVectorType();
    VectorType resType = op.getResultVectorType();

    // Set up convenience transposition table.
    SmallVector<int64_t> transp;
    for (auto attr : op.getTransp())
      transp.push_back(cast<IntegerAttr>(attr).getInt());

    if (isShuffleLike(vectorTransformOptions.vectorTransposeLowering) &&
        succeeded(isTranspose2DSlice(op)))
      return rewriter.notifyMatchFailure(
          op, "Options specifies lowering to shuffle");

    // Handle a true 2-D matrix transpose differently when requested.
    if (vectorTransformOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Flat &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0) {
      Type flattenedType =
          VectorType::get(resType.getNumElements(), resType.getElementType());
      auto matrix =
          rewriter.create<vector::ShapeCastOp>(loc, flattenedType, input);
      auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
      auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
      Value trans = rewriter.create<vector::FlatTransposeOp>(
          loc, flattenedType, matrix, rows, columns);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
      return success();
    }

    // Generate unrolled extract/insert ops. We do not unroll the rightmost
    // (i.e., highest-order) dimensions that are not transposed and leave them
    // in vector form to improve performance. Therefore, we prune those
    // dimensions from the shape/transpose data structures used to generate the
    // extract/insert ops.
    SmallVector<int64_t> prunedTransp;
    pruneNonTransposedDims(transp, prunedTransp);
    size_t numPrunedDims = transp.size() - prunedTransp.size();
    auto prunedInShape = inputType.getShape().drop_back(numPrunedDims);
    auto prunedInStrides = computeStrides(prunedInShape);

    // Generates the extract/insert operations for every scalar/vector element
    // of the leftmost transposed dimensions. We traverse every transpose
    // element using a linearized index that we delinearize to generate the
    // appropriate indices for the extract/insert operations.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t numTransposedElements = ShapedType::getNumElements(prunedInShape);

    for (int64_t linearIdx = 0; linearIdx < numTransposedElements;
         ++linearIdx) {
      auto extractIdxs = delinearize(linearIdx, prunedInStrides);
      SmallVector<int64_t> insertIdxs(extractIdxs);
      applyPermutationToVector(insertIdxs, prunedTransp);
      Value extractOp =
          rewriter.create<vector::ExtractOp>(loc, input, extractIdxs);
      result =
          rewriter.create<vector::InsertOp>(loc, extractOp, result, insertIdxs);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};

/// Rewrite a 2-D vector.transpose as a sequence of shuffle ops.
/// If the strategy is Shuffle1D, it will be lowered to:
///   vector.shape_cast 2D -> 1D
///   vector.shuffle
///   vector.shape_cast 1D -> 2D
/// If the strategy is Shuffle16x16, it will be lowered to a sequence of shuffle
/// ops on 16xf32 vectors.
class TransposeOp2DToShuffleLowering
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOp2DToShuffleLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!isShuffleLike(vectorTransformOptions.vectorTransposeLowering))
      return rewriter.notifyMatchFailure(
          op, "not using vector shuffle based lowering");

    auto srcGtOneDims = isTranspose2DSlice(op);
    if (failed(srcGtOneDims))
      return rewriter.notifyMatchFailure(
          op, "expected transposition on a 2D slice");

    VectorType srcType = op.getSourceVectorType();
    int64_t m = srcType.getDimSize(std::get<0>(srcGtOneDims.value()));
    int64_t n = srcType.getDimSize(std::get<1>(srcGtOneDims.value()));

    // Reshape the n-D input vector with only two dimensions greater than one
    // to a 2-D vector.
    Location loc = op.getLoc();
    auto flattenedType = VectorType::get({n * m}, srcType.getElementType());
    auto reshInputType = VectorType::get({m, n}, srcType.getElementType());
    auto reshInput = rewriter.create<vector::ShapeCastOp>(loc, flattenedType,
                                                          op.getVector());

    Value res;
    if (vectorTransformOptions.vectorTransposeLowering ==
            VectorTransposeLowering::Shuffle16x16 &&
        m == 16 && n == 16) {
      reshInput =
          rewriter.create<vector::ShapeCastOp>(loc, reshInputType, reshInput);
      res = transposeToShuffle16x16(rewriter, reshInput, m, n);
    } else {
      // Fallback to shuffle on 1D approach.
      res = transposeToShuffle1D(rewriter, reshInput, m, n);
    }

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), res);

    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};
} // namespace

void mlir::vector::populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit) {
  patterns.add<TransposeOpLowering, TransposeOp2DToShuffleLowering>(
      options, patterns.getContext(), benefit);
}
