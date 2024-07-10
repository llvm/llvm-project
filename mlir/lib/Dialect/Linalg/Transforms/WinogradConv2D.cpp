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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace linalg {

namespace {

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
  Value init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                outputElementType);

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

  // TODO: Should we support dynamic shapes?
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
void populateWinogradConv2DPatterns(RewritePatternSet &patterns, int64_t m,
                                    int64_t r) {
  MLIRContext *context = patterns.getContext();
  // TODO: Support more Conv2D data layout, e.g., conv_2d_nchw_fchw
  patterns.insert<WinogradConv2DNhwcFhwc>(context, m, r);
}

} // end namespace linalg
} // end namespace mlir
