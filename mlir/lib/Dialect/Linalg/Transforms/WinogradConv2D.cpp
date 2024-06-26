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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace linalg {

namespace {

using TransformMapKeyTy = std::pair<int, int>;

// We use F(m, r) to define the size of minimal filtering algorithms.
// m is the output dimension and r is the filter dimension. We can get
// the input dimension, alpha, from the formula, alpha = m + r - 1.
//
// For example, when m = 2 and r = 3, we know its input size is 4.
// The Conv2D will operate on 4x4 input data with 3x3 filter and get
// 2x2 output result.
constexpr TransformMapKeyTy F_2_3{2, 3};
constexpr TransformMapKeyTy F_4_3{4, 3};
constexpr TransformMapKeyTy F_2_5{2, 5};

Value collapse2DData(RewriterBase &rewriter, Location loc, Value data) {
  auto type = cast<ShapedType>(data.getType());
  auto elementType = type.getElementType();
  auto shape = type.getShape();
  auto collapseType = RankedTensorType::get(
      {shape[0] * shape[1] * shape[2] * shape[3], shape[4], shape[5]},
      elementType);
  SmallVector<ReassociationIndices> reassociation = {{0, 1, 2, 3}, {4}, {5}};
  return rewriter.create<tensor::CollapseShapeOp>(loc, collapseType, data,
                                                  reassociation);
}

// This function generates linalg.batch_matmul to multiply input with filter.
// linalg.batch_matmul only supports 3-dimension data sets. We can treat
// tileH x tileW x H x W data as the 1-dimension data array. That is to convert
// [tileH, tileW, H, W, N, C] to [tileH x tileW x H x W, N, C]. In this way, we
// can convert 6-dimension input data to 3-dimension representation that is
// suitable for linalg.batch_matmul.
//
// Batched matmul will do the matrix multiply with the reduction on channel.
//
// We get
//
// %collapsed_input = tensor.collapse_shape %input
// %collapsed_filter = tensor.collapse_shape %filter
// %ret = linalg.batch_matmul %collapsed_input, %collapsed_filter
// %expanded_ret = tensor.expand_shape %ret
//
// After this function, we get return value with data layout
// (tileH, tileW, H, W, N, F).
Value matrixMultiply(RewriterBase &rewriter, Location loc,
                     Value transformedFilter, Value transformedInput) {
  auto collapseFilter = collapse2DData(rewriter, loc, transformedFilter);
  auto collapseInput = collapse2DData(rewriter, loc, transformedInput);

  // Batched matrix multiply
  auto filterType = cast<ShapedType>(transformedFilter.getType());
  auto filterShape = filterType.getShape();
  auto inputType = cast<ShapedType>(transformedInput.getType());
  auto inputElemType = inputType.getElementType();
  auto inputShape = inputType.getShape();

  auto matmulType = RankedTensorType::get(
      {inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3],
       inputShape[4], filterShape[5]},
      inputElemType);
  Value init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                inputElemType);

  auto matmulOp = rewriter.create<linalg::BatchMatmulOp>(
      loc, matmulType, ValueRange({collapseInput, collapseFilter}),
      ValueRange{init});

  // Expand matmul result
  SmallVector<ReassociationIndices> reassociation = {{0, 1, 2, 3}, {4}, {5}};
  auto expandType =
      RankedTensorType::get({inputShape[0], inputShape[1], inputShape[2],
                             inputShape[3], inputShape[4], filterShape[5]},
                            inputElemType);
  auto expandOutput = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandType, matmulOp.getResult(0), reassociation);
  return expandOutput;
}

Value insertToAlignedTensor(RewriterBase &rewriter, Location loc, Value value,
                            RankedTensorType alignedType) {
  Value alignedInput = rewriter.create<tensor::EmptyOp>(
      loc, alignedType.getShape(), alignedType.getElementType());

  auto zeroIndex = rewriter.getIndexAttr(0);
  auto oneIndex = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult, 4> offsets(4, zeroIndex);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);

  auto valueType = cast<ShapedType>(value.getType());
  auto valueShape = valueType.getShape();
  SmallVector<OpFoldResult, 4> sizes;
  sizes.emplace_back(rewriter.getIndexAttr(valueShape[0]));
  sizes.emplace_back(rewriter.getIndexAttr(valueShape[1]));
  sizes.emplace_back(rewriter.getIndexAttr(valueShape[2]));
  sizes.emplace_back(rewriter.getIndexAttr(valueShape[3]));

  return rewriter.create<tensor::InsertSliceOp>(loc, value, alignedInput,
                                                offsets, sizes, strides);
}

Value extractFromAlignedTensor(RewriterBase &rewriter, Location loc,
                               Value value, RankedTensorType extractedType) {
  auto zeroIndex = rewriter.getIndexAttr(0);
  auto oneIndex = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult, 4> offsets(4, zeroIndex);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);

  auto extractedShape = extractedType.getShape();
  SmallVector<OpFoldResult, 4> sizes;
  sizes.emplace_back(rewriter.getIndexAttr(extractedShape[0]));
  sizes.emplace_back(rewriter.getIndexAttr(extractedShape[1]));
  sizes.emplace_back(rewriter.getIndexAttr(extractedShape[2]));
  sizes.emplace_back(rewriter.getIndexAttr(extractedShape[3]));

  return rewriter.create<tensor::ExtractSliceOp>(loc, extractedType, value,
                                                 offsets, sizes, strides);
}

bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

FailureOr<Operation *> winogradConv2DHelper(RewriterBase &rewriter,
                                            linalg::Conv2DNhwcFhwcOp convOp,
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

  auto filterShape = filterType.getShape();
  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];
  auto inputShape = inputType.getShape();
  int64_t inputN = inputShape[0];
  int64_t inputH = inputShape[1];
  int64_t inputW = inputShape[2];
  int64_t inputC = inputShape[3];
  auto outputShape = outputType.getShape();
  int64_t outputN = outputShape[0];
  int64_t outputH = outputShape[1];
  int64_t outputW = outputShape[2];
  int64_t outputF = outputShape[3];

  // Only support F(m x m, r x r), F(m x 1, r x 1) or F(1 x m, 1 x r)
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

  // Currently, we support (m, r) = (2, 3) or (4, 3) or (2, 5)
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

  // --- Create operator for filter transform ---
  Type elementType = filterType.getElementType();
  int64_t alphaH = heightM + heightR - 1;
  int64_t alphaW = widthM + widthR - 1;
  int64_t tileH = llvm::divideCeilSigned(outputH, heightM);
  int64_t tileW = llvm::divideCeilSigned(outputW, widthM);
  auto retType = RankedTensorType::get(
      {tileH, tileW, alphaH, alphaW, filterC, filterF}, elementType);
  Value retValue =
      rewriter.create<tensor::EmptyOp>(loc, retType.getShape(), elementType);
  auto transformedFilter = rewriter.create<linalg::WinogradFilterTransformOp>(
      loc, retType, filter, retValue, m, r);

  // --- Create operator for input transform ---

  // When input size - (r - 1) is not aligned with output tile size, we need to
  // pad the input data to create the full tiles as tiling.
  int64_t alignedInputH = tileH * heightM + (heightR - 1);
  int64_t alignedInputW = tileW * widthM + (widthR - 1);
  if (alignedInputH != inputH || alignedInputW != inputW) {
    auto alignedInputType = RankedTensorType::get(
        {inputN, alignedInputH, alignedInputW, inputC}, elementType);
    input = insertToAlignedTensor(rewriter, loc, input, alignedInputType);
  }

  retType = RankedTensorType::get(
      {tileH, tileW, alphaH, alphaW, inputN, inputC}, elementType);
  retValue =
      rewriter.create<tensor::EmptyOp>(loc, retType.getShape(), elementType);
  auto transformedInput = rewriter.create<linalg::WinogradInputTransformOp>(
      loc, retType, input, retValue, m, r);

  Value matmulRet =
      matrixMultiply(rewriter, loc, transformedFilter, transformedInput);

  // --- Create operator for output transform ---

  // When output size is not aligned with output tile size, we need to pad the
  // output buffer to insert the full tiles after tiling.
  int64_t alignedOutputH = tileH * heightM;
  int64_t alignedOutputW = tileW * widthM;
  bool isOutputUnaligned =
      ((alignedOutputH != outputH) || (alignedOutputW != outputW));
  if (isOutputUnaligned) {
    auto alignedOutputType = RankedTensorType::get(
        {outputN, alignedOutputH, alignedOutputW, outputF}, elementType);
    output = insertToAlignedTensor(rewriter, loc, output, alignedOutputType);
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
                              elementType));
  }

  rewriter.replaceOp(convOp, transformedOutput);

  return transformedOutput.getDefiningOp();
}

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
  patterns.insert<WinogradConv2DNhwcFhwc>(context, m, r);
}

} // end namespace linalg
} // end namespace mlir
