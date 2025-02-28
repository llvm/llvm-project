//===- WinogradConv2D.cpp - Winograd Conv2D implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Winograd Conv2D algorithm. The implementation is based on the
// paper: Fast Algorithms for Convolutional Neural Networks
// (https://arxiv.org/abs/1509.09308)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace linalg {

namespace {

// clang-format off
/// Winograd Conv2D uses a minimal 2D filtering algorithm to calculate its
/// result. The formula of minimal 2D filtering algorithm F(m x m, r x r),
/// m is the output dimension and r is the filter dimension, is
///
/// Y = A^T x [ (G x g x G^T) x (B^T x d x B) ] x A
///
/// g is filter and d is input data. We need to prepare 6 constant
/// transformation matrices, G, G^T, B^T, B, A^T, and A for this formula.
///
/// The following tables define these constant transformation matrices for
/// F(2 x 2, 3 x 3), F(4 x 4, 3 x 3), and F(2 x 2, 5 x 5)
constexpr float G_2x2_3x3[] = {
   -1,     0,   0,
 1./2, -1./2, 1./2,
 1./2,  1./2, 1./2,
    0,     0,    1
};

constexpr float GT_2x2_3x3[] = {
   -1,  1./2, 1./2, 0,
    0, -1./2, 1./2, 0,
    0,  1./2, 1./2, 1
};

constexpr float BT_2x2_3x3[] = {
   -1,    0,   1,   0,
    0,   -1,   1,   0,
    0,    1,   1,   0,
    0,   -1,   0,   1
};

constexpr float B_2x2_3x3[] = {
   -1,    0,   0,   0,
    0,   -1,   1,  -1,
    1,    1,   1,   0,
    0,    0,   0,   1
};

constexpr float AT_2x2_3x3[] = {
    1,    1,   1,   0,
    0,   -1,   1,   1
};

constexpr float A_2x2_3x3[] = {
    1,    0,
    1,   -1,
    1,    1,
    0,    1
};

constexpr float G_4x4_3x3[] = {
     1,     0,     0,
 -1./3,  1./3, -1./3,
 -1./3, -1./3, -1./3,
 1./12, -1./6,  1./3,
 1./12,  1./6,  1./3,
     0,     0,     1
};

constexpr float GT_4x4_3x3[] = {
 1,  -1./3, -1./3, 1./12, 1./12, 0,
 0,   1./3, -1./3, -1./6,  1./6, 0,
 0,  -1./3, -1./3,  1./3,  1./3, 1
};

constexpr float BT_4x4_3x3[] = {
 1./4,     0, -5./16,      0, 1./16,     0,
    0,  1./4,  -1./4, -1./16, 1./16,     0,
    0, -1./4,  -1./4,  1./16, 1./16,     0,
    0,  1./4,  -1./8,  -1./4,  1./8,     0,
    0, -1./4,  -1./8,   1./4,  1./8,     0,
    0,  1./4,      0, -5./16,     0, 1./16
};

constexpr float B_4x4_3x3[] = {
   1./4,      0,     0,     0,     0,      0,
      0,   1./4, -1./4,  1./4, -1./4,   1./4,
 -5./16,  -1./4, -1./4, -1./8, -1./8,      0,
      0, -1./16, 1./16, -1./4,  1./4, -5./16,
  1./16,  1./16, 1./16,  1./8,  1./8,      0,
      0,      0,     0,     0,     0,  1./16
};

constexpr float AT_4x4_3x3[] = {
 1./8,  1./4, 1./4,  1./8, 1./8,    0,
    0, -1./4, 1./4, -1./4, 1./4,    0,
    0,  1./4, 1./4,  1./2, 1./2,    0,
    0, -1./4, 1./4,    -1,    1, 1./2
};

constexpr float A_4x4_3x3[] = {
  1./8,     0,    0,     0,
  1./4, -1./4, 1./4, -1./4,
  1./4,  1./4, 1./4,  1./4,
  1./8, -1./4, 1./2,    -1,
  1./8,  1./4, 1./2,     1,
     0,     0,    0,  1./2
};

constexpr float G_2x2_5x5[] = {
     1,     0,      0,      0,      0,
  1./6, -1./6,   1./6,  -1./6,   1./6,
 -1./6, -1./6,  -1./6,  -1./6,  -1./6,
-4./15, 2./15, -1./15,  1./30, -1./60,
 1./60, 1./30,  1./15,  2./15,  4./15,
     0,     0,      0,      0,      1
};

constexpr float GT_2x2_5x5[] = {
   1,  1./6, -1./6, -4./15, 1./60, 0,
   0, -1./6, -1./6,  2./15, 1./30, 0,
   0,  1./6, -1./6, -1./15, 1./15, 0,
   0, -1./6, -1./6,  1./30, 2./15, 0,
   0,  1./6, -1./6, -1./60, 4./15, 1
};

constexpr float BT_2x2_5x5[] = {
 1./8,  3./16,  -1./4,  -3./16,   1./8,    0,
    0,   1./8,  1./16,  -5./16,   1./8,    0,
    0,  -1./8, -5./16,  -1./16,   1./8,    0,
    0,   1./4,  -1./8,   -1./4,   1./8,    0,
    0,  -1./8,  -1./4,    1./8,   1./4,    0,
    0,   1./8,  3./16,   -1./4, -3./16, 1./8
};

constexpr float B_2x2_5x5[] = {
   1./8,      0,      0,     0,     0,      0,
  3./16,   1./8,  -1./8,  1./4, -1./8,   1./8,
  -1./4,  1./16, -5./16, -1./8, -1./4,  3./16,
 -3./16, -5./16, -1./16, -1./4,  1./8,  -1./4,
   1./8,   1./8,   1./8,  1./8,  1./4, -3./16,
      0,      0,      0,     0,     0,   1./8
};

constexpr float AT_2x2_5x5[] = {
  1./2,  1, 1,  2, 1,    0,
     0, -1, 1, -1, 2, 1./2
};

constexpr float A_2x2_5x5[] = {
 1./2,    0,
    1,   -1,
    1,    1,
    2,   -1,
    1,    2,
    0, 1./2
};
// clang-format on

using TransformMapKeyTy = std::pair<int, int>;

/// We use F(m, r) to define the size of minimal filtering algorithms.
/// m is the output dimension and r is the filter dimension. We can get
/// the input dimension, alpha, from the formula, alpha = m + r - 1.
///
/// For example, when m = 2 and r = 3, we know its input size is 4.
/// The Conv2D will operate on 4x4 input data with 3x3 filter and get
/// 2x2 output result.
constexpr TransformMapKeyTy F_2_3{2, 3};
constexpr TransformMapKeyTy F_4_3{4, 3};
constexpr TransformMapKeyTy F_2_5{2, 5};

/// Structure to keep information of constant transform matrices.
struct TransformMatrix {
  TransformMatrix(const float *table, int64_t rows, int64_t cols,
                  int64_t scalarFactor = 1)
      : table(table), rows(rows), cols(cols), scalarFactor(scalarFactor) {}

  const float *table;
  int64_t rows;
  int64_t cols;
  int64_t scalarFactor;
};

/// Utility function to convert constant array to arith.constant Value.
Value create2DTransformMatrix(OpBuilder &builder, Location loc,
                              TransformMatrix transform, Type type) {
  ArrayRef<float> constVec(transform.table, transform.rows * transform.cols);

  return builder.create<arith::ConstantOp>(
      loc, DenseFPElementsAttr::get(
               RankedTensorType::get(
                   SmallVector<int64_t>{transform.rows, transform.cols}, type),
               constVec));
}

/// Extract height x width data from 4D tensors.
Value extract2DDataFrom4D(OpBuilder &builder, Location loc, Value source,
                          Value loopNorFIndex, Value loopCorFIndex,
                          Value heightOffset, Value widthOffset,
                          int64_t extractHeight, int64_t extractWidth,
                          int64_t loopNorFIdx, int64_t loopCorFIdx,
                          int64_t heightIdx, int64_t widthIdx) {
  auto sourceType = cast<ShapedType>(source.getType());
  Type elementType = sourceType.getElementType();
  int64_t srcSize = sourceType.getRank();

  auto oneIndex = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets;
  offsets.resize(srcSize);
  offsets[loopNorFIdx] = loopNorFIndex;
  offsets[loopCorFIdx] = loopCorFIndex;
  offsets[heightIdx] = heightOffset;
  offsets[widthIdx] = widthOffset;
  SmallVector<OpFoldResult> sizes(srcSize, oneIndex);
  sizes[heightIdx] = builder.getIndexAttr(extractHeight);
  sizes[widthIdx] = builder.getIndexAttr(extractWidth);
  SmallVector<OpFoldResult> strides(srcSize, oneIndex);

  auto extractFilterType =
      RankedTensorType::get({extractHeight, extractWidth}, elementType);
  auto extractFilterOp = builder.create<tensor::ExtractSliceOp>(
      loc, extractFilterType, source, offsets, sizes, strides);

  return extractFilterOp;
}

/// Extract height x width data from 6D tensors.
Value extract2DDataFrom6D(OpBuilder &builder, Location loc, Value source,
                          Value tileHIndex, Value tileWIndex,
                          Value loopNorFIndex, Value loopCorFIndex,
                          int64_t tileHIdx, int64_t tileWIdx,
                          int64_t loopNorFIdx, int64_t loopCorFIdx,
                          int64_t heightIdx, int64_t widthIdx) {
  auto sourceType = cast<ShapedType>(source.getType());
  Type elementType = sourceType.getElementType();
  auto sourceShape = sourceType.getShape();
  int64_t srcSize = sourceType.getRank();
  int64_t height = sourceShape[heightIdx];
  int64_t width = sourceShape[widthIdx];

  auto zeroIndex = builder.getIndexAttr(0);
  auto oneIndex = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(srcSize, zeroIndex);
  offsets.resize(srcSize);
  offsets[tileHIdx] = tileHIndex;
  offsets[tileWIdx] = tileWIndex;
  offsets[loopNorFIdx] = loopNorFIndex;
  offsets[loopCorFIdx] = loopCorFIndex;
  SmallVector<OpFoldResult> sizes(srcSize, oneIndex);
  sizes[heightIdx] = builder.getIndexAttr(height);
  sizes[widthIdx] = builder.getIndexAttr(width);
  SmallVector<OpFoldResult> strides(srcSize, oneIndex);

  auto extractFilterType = RankedTensorType::get({height, width}, elementType);
  auto extractFilterOp = builder.create<tensor::ExtractSliceOp>(
      loc, extractFilterType, source, offsets, sizes, strides);

  return extractFilterOp;
}

/// Insert transformed height x width data to 4D tensors which it is
/// extracted from.
Value insert2DDataTo4D(OpBuilder &builder, Location loc, Value source,
                       Value dest, Value loopNorFIndex, Value loopCorFIndex,
                       Value heightOffset, Value widthOffset, int64_t height,
                       int64_t width, int64_t loopNorFIdx, int64_t loopCorFIdx,
                       int64_t heightIdx, int64_t widthIdx) {
  int64_t destSize = cast<ShapedType>(dest.getType()).getRank();
  auto oneIndex = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> retOffsets;
  retOffsets.resize(destSize);
  retOffsets[loopNorFIdx] = loopNorFIndex;
  retOffsets[loopCorFIdx] = loopCorFIndex;
  retOffsets[heightIdx] = heightOffset;
  retOffsets[widthIdx] = widthOffset;
  SmallVector<OpFoldResult> retSizes(destSize, oneIndex);
  retSizes[heightIdx] = builder.getIndexAttr(height);
  retSizes[widthIdx] = builder.getIndexAttr(width);
  SmallVector<OpFoldResult> strides(destSize, oneIndex);

  auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
      loc, source, dest, retOffsets, retSizes, strides);

  return insertSliceOp;
}

/// Insert transformed height x width data to 6D tensors which it is
/// extracted from.
Value insert2DDataTo6D(OpBuilder &builder, Location loc, Value source,
                       Value dest, Value tileHIndex, Value tileWIndex,
                       Value loopNorFIndex, Value loopCorFIndex, int64_t height,
                       int64_t width, int64_t tileHIdx, int64_t tileWIdx,
                       int64_t loopNorFIdx, int64_t loopCorFIdx,
                       int64_t heightIdx, int64_t widthIdx) {
  int64_t destSize = cast<ShapedType>(dest.getType()).getRank();
  auto zeroIndex = builder.getIndexAttr(0);
  auto oneIndex = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> retOffsets(destSize, zeroIndex);
  retOffsets.resize(destSize);
  retOffsets[tileHIdx] = tileHIndex;
  retOffsets[tileWIdx] = tileWIndex;
  retOffsets[loopNorFIdx] = loopNorFIndex;
  retOffsets[loopCorFIdx] = loopCorFIndex;
  SmallVector<OpFoldResult> retSizes(destSize, oneIndex);
  retSizes[heightIdx] = builder.getIndexAttr(height);
  retSizes[widthIdx] = builder.getIndexAttr(width);
  SmallVector<OpFoldResult> strides(destSize, oneIndex);

  auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
      loc, source, dest, retOffsets, retSizes, strides);

  return insertSliceOp;
}

/// This function transforms the filter. The data layout of the filter is FHWC.
/// The transformation matrix is 2-dimension. We need to extract H x W from
/// FHWC first. We need to generate 2 levels of loops to iterate on F and C.
/// After the transformation, we get
///
/// scf.for %f = lo_f to hi_f step 1
///   scf.for %c = lo_c to hi_c step 1
///     %extracted = extract filter<h x w> from filter<f x h x w x c>
///     %ret = linalg.matmul G, %extracted
///     %ret = linalg.matmul %ret, GT
///     %inserted = insert %ret into filter<h x w x c x f>
Value filterTransform(RewriterBase &rewriter, Location loc, Value filter,
                      Value retValue, int64_t m, int64_t r,
                      bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to G transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      GMatrices = {
          {F_2_3, TransformMatrix(G_2x2_3x3, 4, 3)},
          {F_4_3, TransformMatrix(G_4x4_3x3, 6, 3)},
          {F_2_5, TransformMatrix(G_2x2_5x5, 6, 5)},
      };

  // Map from (m, r) to GT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      GTMatrices = {
          {F_2_3, TransformMatrix(GT_2x2_3x3, 3, 4)},
          {F_4_3, TransformMatrix(GT_4x4_3x3, 3, 6)},
          {F_2_5, TransformMatrix(GT_2x2_5x5, 5, 6)},
      };

  auto filterType = cast<ShapedType>(filter.getType());
  Type elementType = filterType.getElementType();
  auto filterShape = filterType.getShape(); // F, H, W, C
  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];

  if (filterH != r && filterH != 1)
    return Value();
  if (filterW != r && filterW != 1)
    return Value();

  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                       ValueRange args) -> scf::ValueVector {
    Value FIter = ivs[0];
    Value CIter = ivs[1];

    // Extract (H, W) from (F, H, W, C).
    auto extractFilter =
        extract2DDataFrom4D(builder, loc, filter, FIter, CIter, zeroIdx,
                            zeroIdx, filterH, filterW, /*loopNorFIdx=*/0,
                            /*loopCorFIdx=*/3, /*heightIdx=*/1, /*widthIdx=*/2);

    TransformMapKeyTy key = {m, r};
    int64_t retRows = 1;
    Value matmulRetValue = extractFilter;
    Value zero = builder.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    if (leftTransform) {
      // Get constant transform matrix G.
      auto it = GMatrices.find(key);
      if (it == GMatrices.end())
        return {};
      const TransformMatrix &GMatrix = it->second;

      retRows = GMatrix.rows;
      auto matmulType = RankedTensorType::get({retRows, filterW}, elementType);
      auto empty =
          builder
              .create<tensor::EmptyOp>(loc, matmulType.getShape(), elementType)
              .getResult();
      auto init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);

      Value G = create2DTransformMatrix(builder, loc, GMatrix, elementType);
      // Multiply G x g.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{G, extractFilter}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    if (rightTransform) {
      // Get constant transform matrix GT.
      auto it = GTMatrices.find(key);
      if (it == GTMatrices.end())
        return {};
      const TransformMatrix &GTMatrix = it->second;

      auto matmulType =
          RankedTensorType::get({retRows, GTMatrix.cols}, elementType);
      auto empty =
          builder
              .create<tensor::EmptyOp>(loc, matmulType.getShape(), elementType)
              .getResult();
      auto init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);

      Value GT = create2DTransformMatrix(builder, loc, GTMatrix, elementType);
      // Multiply u = (G x g) x GT.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{matmulRetValue, GT}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    // Insert (H, W) to (H, W, C, F).
    int64_t retHeight = leftTransform ? m + r - 1 : 1;
    int64_t retWidth = rightTransform ? m + r - 1 : 1;

    auto insertSliceOp =
        insert2DDataTo4D(builder, loc, matmulRetValue, args[0], FIter, CIter,
                         zeroIdx, zeroIdx, retHeight, retWidth,
                         /*loopNorFIdx=*/3, /*loopCorFIdx=*/2,
                         /*heightIdx=*/0, /*widthIdx=*/1);

    return {insertSliceOp};
  };

  auto fUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, filterF);
  auto cUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, filterC);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::LoopNest loops = scf::buildLoopNest(
      rewriter, loc, {zeroIdx, zeroIdx}, {fUpperBound, cUpperBound},
      {oneStep, oneStep}, {retValue}, buildBody);
  return loops.results[0];
}

/// This function transforms the input. The data layout of the input is NHWC.
/// The transformation matrix is 2-dimension. We need to extract H x W from
/// NHWC first. We need to generate 2 levels of loops to iterate on N and C.
/// After the transformation, we get
///
/// scf.for %h = 0 to tileH step 1
///   scf.for %w = 0 to tileW step 1
///     scf.for %n = 0 to N step 1
///       scf.for %c = 0 to C step 1
///         %extracted = extract %extracted<alphaH x alphaW> from
///                              %input<N x H x W x C>
///                              at [%n, (%h x m), (%w x m), %c]
///         %ret = linalg.matmul BT, %extracted
///         %ret = linalg.matmul %ret, B
///         %inserted = insert %ret<alphaH x alphaW> into
///                            %output<alphaH x alphaW x tileH x tileW x N x C>
///                            at [0, 0, %h, %w, %n, %c]
Value inputTransform(RewriterBase &rewriter, Location loc, Value input,
                     Value retValue, int64_t m, int64_t r,
                     bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to BT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      BTMatrices = {
          {F_2_3, TransformMatrix(BT_2x2_3x3, 4, 4)},
          {F_4_3, TransformMatrix(BT_4x4_3x3, 6, 6)},
          {F_2_5, TransformMatrix(BT_2x2_5x5, 6, 6)},
      };

  // Map from (m, r) to B transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      BMatrices = {
          {F_2_3, TransformMatrix(B_2x2_3x3, 4, 4)},
          {F_4_3, TransformMatrix(B_4x4_3x3, 6, 6)},
          {F_2_5, TransformMatrix(B_2x2_5x5, 6, 6)},
      };

  auto inputType = cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  auto inputShape = inputType.getShape(); // N, H, W, C
  int64_t inputN = inputShape[0];
  int64_t inputC = inputShape[3];
  auto valueType = cast<ShapedType>(retValue.getType());
  auto valueShape = valueType.getShape(); // alphaH, alphaW, HTile, WTile, N, C
  int64_t tileH = valueShape[2];
  int64_t tileW = valueShape[3];
  int64_t alphaH = leftTransform ? m + r - 1 : 1;
  int64_t alphaW = rightTransform ? m + r - 1 : 1;

  auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                       ValueRange args) -> scf::ValueVector {
    Value tileHIter = ivs[0];
    Value tileWIter = ivs[1];
    Value NIter = ivs[2];
    Value CIter = ivs[3];

    auto context = builder.getContext();

    auto identityAffineMap = rewriter.getMultiDimIdentityMap(1);
    auto affineMap =
        AffineMap::get(1, 0, {builder.getAffineDimExpr(0) * m}, context);
    Value heightOffset = builder.create<affine::AffineApplyOp>(
        loc, leftTransform ? affineMap : identityAffineMap, tileHIter);
    Value widthOffset = builder.create<affine::AffineApplyOp>(
        loc, rightTransform ? affineMap : identityAffineMap, tileWIter);

    // Extract (H, W) from (N, H, W, C).
    auto extractInput =
        extract2DDataFrom4D(builder, loc, input, NIter, CIter, heightOffset,
                            widthOffset, alphaH, alphaW, /*loopNorFIdx=*/0,
                            /*loopCorFIdx=*/3, /*heightIdx=*/1, /*widthIdx=*/2);

    TransformMapKeyTy key = {m, r};
    int64_t retRows = 1;
    int64_t retCols = 1;
    Value matmulRetValue = extractInput;
    Value zero = builder.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    if (leftTransform) {
      // Get constant transform matrix BT.
      auto it = BTMatrices.find(key);
      if (it == BTMatrices.end())
        return {};
      const TransformMatrix &BTMatrix = it->second;

      retRows = BTMatrix.rows;
      auto matmulType = RankedTensorType::get({retRows, alphaW}, elementType);
      auto empty =
          builder
              .create<tensor::EmptyOp>(loc, matmulType.getShape(), elementType)
              .getResult();
      auto init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);

      Value BT =
          create2DTransformMatrix(builder, loc, BTMatrix, builder.getF32Type());
      // Multiply BT x d.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{BT, matmulRetValue}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    if (rightTransform) {
      // Get constant transform matrix B.
      auto it = BMatrices.find(key);
      if (it == BMatrices.end())
        return {};
      const TransformMatrix &BMatrix = it->second;

      retCols = BMatrix.cols;
      auto matmulType = RankedTensorType::get({retRows, retCols}, elementType);
      auto empty =
          builder
              .create<tensor::EmptyOp>(loc, matmulType.getShape(), elementType)
              .getResult();
      auto init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
      Value B =
          create2DTransformMatrix(builder, loc, BMatrix, builder.getF32Type());
      // Multiply v = (BT x d) x B.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{matmulRetValue, B}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    // Insert (H, W) to (H, W, tileH, tileW, N, C).
    auto combinedVal = insert2DDataTo6D(
        builder, loc, matmulRetValue, args[0], tileHIter, tileWIter, NIter,
        CIter, retRows, retCols, 2, 3, /*loopNorFIdx=*/4, /*loopCorFIdx=*/5,
        /*heightIdx=*/0, /*widthIdx=*/1);

    return {combinedVal};
  };

  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto tileHBound = rewriter.create<arith::ConstantIndexOp>(loc, tileH);
  auto tileWBound = rewriter.create<arith::ConstantIndexOp>(loc, tileW);
  auto nUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, inputN);
  auto cUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, inputC);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::LoopNest loops = scf::buildLoopNest(
      rewriter, loc, {zeroIdx, zeroIdx, zeroIdx, zeroIdx},
      {tileHBound, tileWBound, nUpperBound, cUpperBound},
      {oneStep, oneStep, oneStep, oneStep}, {retValue}, buildBody);
  return loops.results[0];
}

/// This function generates linalg.batch_matmul to multiply input with filter.
/// linalg.batch_matmul only supports 3-dimensional inputs. We can treat
/// tileH x tileW x H x W data as the 1-dimensional data array. That is to
/// convert [tileH, tileW, H, W, N, C] to [tileH x tileW x H x W, N, C]. In this
/// way, we can convert 6-dimensional inputs to 3-dimensional representation
/// that is suitable for linalg.batch_matmul.
///
/// Batched matmul will do the matrix multiply with the reduction on channel.
///
/// We get
///
/// %collapsed_input = tensor.collapse_shape %input
/// %collapsed_filter = tensor.collapse_shape %filter
/// %ret = linalg.batch_matmul %collapsed_input, %collapsed_filter
/// %expanded_ret = tensor.expand_shape %ret
///
/// After this function, we get return value with data layout
/// (tileH, tileW, H, W, N, F).
static Value matrixMultiply(RewriterBase &rewriter, Location loc,
                            Value transformedFilter, Value transformedInput,
                            Type outputElementType) {
  // Convert (alphaH, alphaW, C, F) to (alphaH x alphaW, C, F) for filter.
  auto filterType = cast<ShapedType>(transformedFilter.getType());
  assert(filterType.hasStaticShape() && "only support static shapes.");
  ArrayRef<int64_t> filterShape = filterType.getShape();
  Type filterElementType = filterType.getElementType();
  auto filterReassocType = RankedTensorType::get(
      {filterShape[0] * filterShape[1], filterShape[2], filterShape[3]},
      filterElementType);
  SmallVector<ReassociationIndices> filterReassoc = {{0, 1}, {2}, {3}};
  Value collapseFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, filterReassocType, transformedFilter, filterReassoc);

  // Convert (alphaH, alphaW, tileH, tileW, N, C) to
  // (alphaH x alphaW, tileH x tileW x N, C) for input.
  auto inputType = cast<ShapedType>(transformedInput.getType());
  assert(inputType.hasStaticShape() && "only support static shapes.");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  Type inputElementType = inputType.getElementType();
  auto inputReassocType = RankedTensorType::get(
      {inputShape[0] * inputShape[1],
       inputShape[2] * inputShape[3] * inputShape[4], inputShape[5]},
      inputElementType);
  SmallVector<ReassociationIndices> inputReassoc = {{0, 1}, {2, 3, 4}, {5}};
  Value collapseInput = rewriter.create<tensor::CollapseShapeOp>(
      loc, inputReassocType, transformedInput, inputReassoc);

  // Batched matrix multiply.
  auto matmulType = RankedTensorType::get(
      {inputShape[0] * inputShape[1],
       inputShape[2] * inputShape[3] * inputShape[4], filterShape[3]},
      outputElementType);
  Value empty = rewriter
                    .create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                             outputElementType)
                    .getResult();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(outputElementType));
  Value init = rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);

  auto matmulOp = rewriter.create<linalg::BatchMatmulOp>(
      loc, matmulType, ValueRange({collapseInput, collapseFilter}),
      ValueRange{init});

  // The result shape of batch matmul is (alphaH x alphaW, tileH x tileW x N, F)
  // Expand matmul result to (alphaH, alphaW, tileH, tileW, N, F).
  SmallVector<ReassociationIndices> outputReassoc = {{0, 1}, {2, 3, 4}, {5}};
  auto outputReassocType =
      RankedTensorType::get({inputShape[0], inputShape[1], inputShape[2],
                             inputShape[3], inputShape[4], filterShape[3]},
                            outputElementType);
  auto expandOutput = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputReassocType, matmulOp.getResult(0), outputReassoc);
  return expandOutput;
}

/// This function transforms the output. The data layout of the output is HWNF.
/// The transformation matrix is 2-dimension. We need to extract H x W from
/// HWNF first. We need to generate 2 levels of loops to iterate on N and F.
/// After the transformation, we get
///
/// scf.for %h = 0 to tileH step 1
///   scf.for %w = 0 to tileW step 1
///     scf.for %n = 0 to N step 1
///       scf.for %f = 0 to F step 1
///         %extracted = extract %extracted<alphaH x alphaW> from
///                              %input<alphaH x alphaW x tileH x tileW x N x F>
///                              at [0, 0, %h, %w, %n, %f]
///         %ret = linalg.matmul AT, %extracted
///         %ret = linalg.matmul %ret, A
///         %inserted = insert %ret<alphaH x alphaW> into
///                            output<N x H x W x F>
///                            at [%n, (%h x m), (%w x m), %f]
Value outputTransform(RewriterBase &rewriter, Location loc, Value value,
                      Value output, int64_t m, int64_t r,
                      bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to AT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      ATMatrices = {
          {F_2_3, TransformMatrix(AT_2x2_3x3, 2, 4)},
          {F_4_3, TransformMatrix(AT_4x4_3x3, 4, 6, 32)},
          {F_2_5, TransformMatrix(AT_2x2_5x5, 2, 6, 16)},
      };

  // Map from (m, r) to A transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      AMatrices = {
          {F_2_3, TransformMatrix(A_2x2_3x3, 4, 2)},
          {F_4_3, TransformMatrix(A_4x4_3x3, 6, 4, 32)},
          {F_2_5, TransformMatrix(A_2x2_5x5, 6, 2, 16)},
      };

  auto valueType = cast<ShapedType>(value.getType());
  Type elementType = valueType.getElementType();
  auto valueShape = valueType.getShape(); // H, W, TileH, TileW, N, F
  int64_t valueH = valueShape[0];
  int64_t valueW = valueShape[1];
  int64_t valueN = valueShape[4];
  int64_t valueF = valueShape[5];
  int64_t alphaH = leftTransform ? m + r - 1 : 1;
  int64_t alphaW = rightTransform ? m + r - 1 : 1;

  if (valueH != alphaH && valueH != 1)
    return Value();
  if (valueW != alphaW && valueW != 1)
    return Value();

  auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                       ValueRange args) -> scf::ValueVector {
    auto context = builder.getContext();
    Value tileHIter = ivs[0];
    Value tileWIter = ivs[1];
    Value NIter = ivs[2];
    Value FIter = ivs[3];

    // Extract (H, W) from (H, W, tileH, tileW, N, F).
    auto extractValue =
        extract2DDataFrom6D(builder, loc, value, tileHIter, tileWIter, NIter,
                            FIter, 2, 3, /*loopNorFIdx=*/4,
                            /*loopCorFIdx=*/5, /*heightIdx=*/0, /*widthIdx=*/1);

    const TransformMapKeyTy key = {m, r};
    const TransformMatrix &AMatrix = AMatrices.at(key);
    const TransformMatrix &ATMatrix = ATMatrices.at(key);
    int64_t scalarFactor = (rightTransform ? AMatrix.scalarFactor : 1) *
                           (leftTransform ? ATMatrix.scalarFactor : 1);
    int64_t retCols = rightTransform ? AMatrix.cols : 1;
    int64_t retRows = leftTransform ? ATMatrix.rows : 1;

    Value matmulRetValue = extractValue;
    Value zero = builder.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    auto identityAffineMap = rewriter.getMultiDimIdentityMap(1);
    auto affineMap =
        AffineMap::get(1, 0, {builder.getAffineDimExpr(0) * m}, context);
    Value heightOffset = builder.create<affine::AffineApplyOp>(
        loc, leftTransform ? affineMap : identityAffineMap, tileHIter);
    Value widthOffset = builder.create<affine::AffineApplyOp>(
        loc, rightTransform ? affineMap : identityAffineMap, tileWIter);

    Value outInitVal =
        extract2DDataFrom4D(builder, loc, args[0], NIter, FIter, heightOffset,
                            widthOffset, retRows, retCols,
                            /*loopNorFIdx=*/0,
                            /*loopCorFIdx=*/3, /*heightIdx=*/1,
                            /*widthIdx=*/2);
    if (leftTransform) {
      auto matmulType = RankedTensorType::get({retRows, valueW}, elementType);
      Value init = outInitVal;
      if (rightTransform || scalarFactor != 1) {
        auto empty = builder
                         .create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                  elementType)
                         .getResult();
        init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
      }

      Value AT = create2DTransformMatrix(builder, loc, ATMatrix, elementType);
      // Multiply AT x m.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{AT, matmulRetValue}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    if (rightTransform) {
      auto matmulType =
          RankedTensorType::get({retRows, AMatrix.cols}, elementType);
      Value init = outInitVal;
      if (scalarFactor != 1) {
        auto empty = builder
                         .create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                  elementType)
                         .getResult();
        init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
      }

      Value A = create2DTransformMatrix(builder, loc, AMatrix, elementType);
      // Multiply y = (AT x m) x A.
      auto matmulOp = builder.create<linalg::MatmulOp>(
          loc, matmulType, ValueRange{matmulRetValue, A}, ValueRange{init});
      matmulRetValue = matmulOp.getResult(0);
    }

    if (scalarFactor != 1) {
      // Multiply by scalar factor and add outInitVal.
      Value scalarFactorValue = builder.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, scalarFactor));
      auto matmulType = RankedTensorType::get({retRows, retCols}, elementType);
      auto identityAffineMap = rewriter.getMultiDimIdentityMap(2);
      SmallVector<AffineMap> affineMaps = {
          AffineMap::get(2, 0, context), identityAffineMap, identityAffineMap};

      matmulRetValue =
          rewriter
              .create<linalg::GenericOp>(
                  loc, matmulType,
                  ValueRange{scalarFactorValue, matmulRetValue},
                  ValueRange{outInitVal}, affineMaps,
                  llvm::ArrayRef<utils::IteratorType>{
                      utils::IteratorType::parallel,
                      utils::IteratorType::parallel},
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    auto mulf = nestedBuilder.create<arith::MulFOp>(
                        nestedLoc, args[0], args[1]);
                    auto addf = nestedBuilder.create<arith::AddFOp>(
                        nestedLoc, mulf.getResult(), args[2]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                          addf.getResult());
                  })
              .getResult(0);
    }

    // Insert (H, W) to (N, H, W, F).
    Value combinedVal =
        insert2DDataTo4D(builder, loc, matmulRetValue, args[0], NIter, FIter,
                         heightOffset, widthOffset, retRows, retCols,
                         /*loopNorFIdx=*/0,
                         /*loopCorFIdx=*/3, /*heightIdx=*/1,
                         /*widthIdx=*/2);

    return {combinedVal};
  };

  int64_t tilwH = valueShape[2];
  int64_t tileW = valueShape[3];
  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto tileHBound = rewriter.create<arith::ConstantIndexOp>(loc, tilwH);
  auto tileWBound = rewriter.create<arith::ConstantIndexOp>(loc, tileW);
  auto nUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, valueN);
  auto fUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, valueF);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::LoopNest loops = scf::buildLoopNest(
      rewriter, loc, {zeroIdx, zeroIdx, zeroIdx, zeroIdx},
      {tileHBound, tileWBound, nUpperBound, fUpperBound},
      {oneStep, oneStep, oneStep, oneStep}, {output}, buildBody);
  return loops.results[0];
}

/// Create an empty tensor with alignedType and insert the value into the
/// created empty tensor with aligned size.
static Value padToAlignedTensor(RewriterBase &rewriter, Location loc,
                                Value value, ArrayRef<int64_t> alignedShape) {
  auto valueType = cast<ShapedType>(value.getType());
  Type elementType = valueType.getElementType();
  auto alignedType = RankedTensorType::get(alignedShape, elementType);
  Value padValue = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getZeroAttr(elementType));

  return linalg::makeComposedPadHighOp(rewriter, loc, alignedType, value,
                                       padValue, false);
}

/// Extract sub-tensor with extractedType from value.
static Value extractFromAlignedTensor(RewriterBase &rewriter, Location loc,
                                      Value value,
                                      RankedTensorType extractedType) {
  OpFoldResult zeroIndex = rewriter.getIndexAttr(0);
  OpFoldResult oneIndex = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult, 4> offsets(4, zeroIndex);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);

  ArrayRef<int64_t> extractedShape = extractedType.getShape();
  SmallVector<OpFoldResult> sizes =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(extractedShape));

  return rewriter.create<tensor::ExtractSliceOp>(loc, extractedType, value,
                                                 offsets, sizes, strides);
}

/// Utility function to check all values in the attribute are 1.
static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

/// A helper function to convert linalg.conv_2d_nhwc_fhwc to
/// linalg.winograd_*_transform ops.
static FailureOr<Operation *>
winogradConv2DHelper(RewriterBase &rewriter, linalg::Conv2DNhwcFhwcOp convOp,
                     int64_t m, int64_t r) {
  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];
  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());
  auto outputType = cast<ShapedType>(output.getType());

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  if (!hasAllOneValues(convOp.getStrides()))
    return rewriter.notifyMatchFailure(convOp, "expected all ones for strides");

  ArrayRef<int64_t> filterShape = filterType.getShape();
  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputN = inputShape[0];
  int64_t inputH = inputShape[1];
  int64_t inputW = inputShape[2];
  int64_t inputC = inputShape[3];
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t outputN = outputShape[0];
  int64_t outputH = outputShape[1];
  int64_t outputW = outputShape[2];
  int64_t outputF = outputShape[3];

  // Only support F(m x m, r x r), F(m x 1, r x 1) or F(1 x m, 1 x r).
  bool isSupportedFilter = false;
  if (filterH == filterW && filterH == r)
    isSupportedFilter = true;
  if (filterH == r && filterW == 1)
    isSupportedFilter = true;
  if (filterH == 1 && filterW == r)
    isSupportedFilter = true;

  if (!isSupportedFilter)
    return rewriter.notifyMatchFailure(
        convOp, "only support filter (r x r), (r x 1) or (1 x r)");

  // Currently, we support (m, r) = (2, 3) or (4, 3) or (2, 5).
  static const llvm::SmallVector<TransformMapKeyTy, 3> validConfigs = {
      F_2_3, F_4_3, F_2_5};

  TransformMapKeyTy key = {m, r};
  auto it = std::find(validConfigs.begin(), validConfigs.end(), key);
  // If we cannot find the constant transformation matrix, it means we do
  // not support this configuration yet.
  if (it == validConfigs.end())
    return failure();

  // All the criterias are satisfied. We can do Winograd Conv2D.
  Location loc = convOp.getLoc();

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = filterH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = filterW != 1;
  int64_t heightM = leftTransform ? m : 1;
  int64_t widthM = rightTransform ? m : 1;
  int64_t heightR = leftTransform ? r : 1;
  int64_t widthR = rightTransform ? r : 1;

  // --- Create operation for filter transform ---
  Type filterElementType = filterType.getElementType();
  int64_t alphaH = heightM + heightR - 1;
  int64_t alphaW = widthM + widthR - 1;
  int64_t tileH = llvm::divideCeilSigned(outputH, heightM);
  int64_t tileW = llvm::divideCeilSigned(outputW, widthM);
  auto retType = RankedTensorType::get({alphaH, alphaW, filterC, filterF},
                                       filterElementType);
  Value retValue = rewriter.create<tensor::EmptyOp>(loc, retType.getShape(),
                                                    filterElementType);
  auto transformedFilter = rewriter.create<linalg::WinogradFilterTransformOp>(
      loc, retType, filter, retValue, m, r);

  // --- Create operation for input transform ---

  // When input size - (r - 1) is not aligned with output tile size, we need to
  // pad the input data to create the full tiles as tiling.
  Type inputElementType = inputType.getElementType();
  int64_t alignedInputH = tileH * heightM + (heightR - 1);
  int64_t alignedInputW = tileW * widthM + (widthR - 1);
  if (alignedInputH != inputH || alignedInputW != inputW) {
    input = padToAlignedTensor(rewriter, loc, input,
                               {inputN, alignedInputH, alignedInputW, inputC});
  }

  retType = RankedTensorType::get(
      {alphaH, alphaW, tileH, tileW, inputN, inputC}, inputElementType);
  retValue = rewriter.create<tensor::EmptyOp>(loc, retType.getShape(),
                                              inputElementType);
  auto transformedInput = rewriter.create<linalg::WinogradInputTransformOp>(
      loc, retType, input, retValue, m, r);

  Type outputElementType = outputType.getElementType();
  Value matmulRet = matrixMultiply(rewriter, loc, transformedFilter,
                                   transformedInput, outputElementType);

  // --- Create operation for output transform ---

  // When output size is not aligned with output tile size, we need to pad the
  // output buffer to insert the full tiles after tiling.
  int64_t alignedOutputH = tileH * heightM;
  int64_t alignedOutputW = tileW * widthM;
  bool isOutputUnaligned =
      ((alignedOutputH != outputH) || (alignedOutputW != outputW));
  if (isOutputUnaligned) {
    auto alignedOutputType = RankedTensorType::get(
        {outputN, alignedOutputH, alignedOutputW, outputF}, outputElementType);
    output =
        padToAlignedTensor(rewriter, loc, output, alignedOutputType.getShape());
    outputType = alignedOutputType;
  }

  Value transformedOutput = rewriter.create<linalg::WinogradOutputTransformOp>(
      loc, outputType, matmulRet, output, m, r);

  // When output size is not aligned with output tile size, extract the
  // value from the padded buffer.
  if (isOutputUnaligned) {
    transformedOutput = extractFromAlignedTensor(
        rewriter, loc, transformedOutput,
        RankedTensorType::get({outputN, outputH, outputW, outputF},
                              outputElementType));
  }

  rewriter.replaceOp(convOp, transformedOutput);

  return transformedOutput.getDefiningOp();
}

/// A helper function to decompose linalg.winograd_filter_transform.
FailureOr<Operation *>
decomposeWinogradFilterTransformHelper(RewriterBase &rewriter,
                                       linalg::WinogradFilterTransformOp op) {
  Location loc = op.getLoc();
  Value filter = op.getFilter();
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = filterH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = filterW != 1;
  Value transformedFilter =
      filterTransform(rewriter, loc, filter, op.getOutput(), op.getM(),
                      op.getR(), leftTransform, rightTransform);
  if (!transformedFilter)
    return failure();

  rewriter.replaceOp(op, transformedFilter);

  return transformedFilter.getDefiningOp();
}

/// A helper function to decompose linalg.winograd_input_transform.
FailureOr<Operation *>
decomposeWinogradInputTransformHelper(RewriterBase &rewriter,
                                      linalg::WinogradInputTransformOp op) {
  Location loc = op.getLoc();
  Value output = op.getOutput();
  auto outputType = cast<ShapedType>(output.getType());
  auto outputShape = outputType.getShape();

  int64_t outputH = outputShape[0];
  int64_t outputW = outputShape[1];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = outputH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = outputW != 1;
  Value transformedInput =
      inputTransform(rewriter, loc, op.getInput(), op.getOutput(), op.getM(),
                     op.getR(), leftTransform, rightTransform);
  if (!transformedInput)
    return failure();

  rewriter.replaceOp(op, transformedInput);

  return transformedInput.getDefiningOp();
}

/// A helper function to decompose linalg.winograd_output_transform.
FailureOr<Operation *>
decomposeWinogradOutputTransformHelper(RewriterBase &rewriter,
                                       linalg::WinogradOutputTransformOp op) {
  Location loc = op.getLoc();
  Value value = op.getValue();
  auto valueType = cast<ShapedType>(value.getType());
  auto valueShape = valueType.getShape();
  int64_t valueH = valueShape[0];
  int64_t valueW = valueShape[1];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = valueH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = valueW != 1;
  Value transformedOutput =
      outputTransform(rewriter, loc, value, op.getOutput(), op.getM(),
                      op.getR(), leftTransform, rightTransform);
  if (!transformedOutput)
    return failure();

  rewriter.replaceOp(op, transformedOutput);

  return transformedOutput.getDefiningOp();
}

/// A rewrite pattern to decompose linalg.winograd_filter_transform operations.
class DecomposeWinogradFilterTransform final
    : public OpRewritePattern<linalg::WinogradFilterTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradFilterTransformOp op,
                                PatternRewriter &rewriter) const override {
    return decomposeWinogradFilterTransformHelper(rewriter, op);
  }
};

/// A rewrite pattern to decompose linalg.winograd_input_transform operations.
class DecomposeWinogradInputTransform final
    : public OpRewritePattern<linalg::WinogradInputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradInputTransformOp op,
                                PatternRewriter &rewriter) const override {
    return decomposeWinogradInputTransformHelper(rewriter, op);
  }
};

/// A rewrite pattern to decompose linalg.winograd_output_transform operations.
class DecomposeWinogradOutputTransform final
    : public OpRewritePattern<linalg::WinogradOutputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradOutputTransformOp op,
                                PatternRewriter &rewriter) const override {
    return decomposeWinogradOutputTransformHelper(rewriter, op);
  }
};

/// A rewrite pattern for Winograd Conv2D algorithm.
class WinogradConv2DNhwcFhwc final
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  WinogradConv2DNhwcFhwc(mlir::MLIRContext *context, int64_t m, int64_t r)
      : OpRewritePattern(context), m(m), r(r) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(winogradConv2DHelper(rewriter, convOp, m, r)))
      return failure();

    return success();
  }

private:
  int64_t m;
  int64_t r;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
FailureOr<Operation *> winogradConv2D(RewriterBase &rewriter,
                                      linalg::Conv2DNhwcFhwcOp op, int64_t m,
                                      int64_t r) {
  return winogradConv2DHelper(rewriter, op, m, r);
}

FailureOr<Operation *>
decomposeWinogradFilterTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradFilterTransformOp op) {
  return decomposeWinogradFilterTransformHelper(rewriter, op);
}

FailureOr<Operation *>
decomposeWinogradInputTransformOp(RewriterBase &rewriter,
                                  linalg::WinogradInputTransformOp op) {
  return decomposeWinogradInputTransformHelper(rewriter, op);
}

FailureOr<Operation *>
decomposeWinogradOutputTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradOutputTransformOp op) {
  return decomposeWinogradOutputTransformHelper(rewriter, op);
}

void populateWinogradConv2DPatterns(RewritePatternSet &patterns, int64_t m,
                                    int64_t r) {
  MLIRContext *context = patterns.getContext();
  // TODO: Support more Conv2D data layout, e.g., conv_2d_nchw_fchw
  patterns.insert<WinogradConv2DNhwcFhwc>(context, m, r);
}

void populateDecomposeWinogradOpsPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns
      .insert<DecomposeWinogradFilterTransform, DecomposeWinogradInputTransform,
              DecomposeWinogradOutputTransform>(context);
}

} // end namespace linalg
} // end namespace mlir
