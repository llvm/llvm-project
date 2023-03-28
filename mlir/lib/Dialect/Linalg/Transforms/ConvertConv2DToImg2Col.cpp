//===- ConvertConv2DToImg2Col.cpp - im2col implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <utility>

namespace mlir {
namespace linalg {
static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = x.getType().isa<IntegerType>();
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (accType.isa<IntegerType>())
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

// Delinearizes the given composite `index` by the basis specified in `factors`.
static SmallVector<Value> unrollIndex(OpBuilder &b, Location loc, Value index,
                                      ArrayRef<int64_t> factors) {
  assert(!factors.empty() && "empty factor list");
  SmallVector<Value> basis;
  for (int64_t f : factors)
    basis.push_back(b.create<arith::ConstantOp>(loc, b.getIndexAttr(f)));
  FailureOr<SmallVector<Value>> multiIndex =
      delinearizeIndex(b, loc, index, basis);
  assert(!failed(multiIndex) && "Failed to linearize img2col index");
  return *multiIndex;
}

// Given indices corresponding to iterators in the output (oIndex) and filter
// (fIndex) for a convolution, compute the convolved index for the
// input as `oIndex * stride + fIndex`.
static Value getConvolvedIndex(OpBuilder &b, Location loc, Value oIndex,
                               Value fIndex, int64_t stride) {
  AffineExpr oExpr, fExpr;
  bindSymbols(b.getContext(), oExpr, fExpr);
  AffineMap convMap = AffineMap::get(0, 2, stride * oExpr + fExpr);
  return makeComposedAffineApply(b, loc, convMap, ValueRange{oIndex, fIndex});
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<ShapedType>();
  auto filterType = convOp.getInputs()[1].getType().cast<ShapedType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<ShapedType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  MLIRContext *context = rewriter.getContext();
  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  ArrayRef<int64_t> filterShape = filterType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t n = outputShape[0];
  int64_t oh = outputShape[1];
  int64_t ow = outputShape[2];
  int64_t oc = outputShape[3];
  int64_t fh = filterShape[0];
  int64_t fw = filterShape[1];
  int64_t ic = filterShape[2];

  Location loc = convOp.getLoc();

  // Reshape output and filter to the LHS and result of a (B)MNK matmul.
  SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
  auto reshapedFilterType =
      RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());
  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}};
  RankedTensorType reshapedOutputType =
      RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  SmallVector<int64_t> colTensorShape = {n, oh * ow, fh * fw * ic};
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  // Convert the input to a (BMK) column tensor.
  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> img2colIterators(nloops, parallel);

  SmallVector<AffineMap> img2colIndexingMaps = {
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/ValueRange{}, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Get the iterators named based on the matmul (batch, m, k).
        Value bIndex = nestedBuilder.create<linalg::IndexOp>(loc, 0);
        Value mIndex = nestedBuilder.create<linalg::IndexOp>(loc, 1);
        Value kIndex = nestedBuilder.create<linalg::IndexOp>(loc, 2);

        // Recover the original iteration indices from the problem/input sizes.
        SmallVector<Value> mIndices = unrollIndex(
            nestedBuilder, nestedLoc, mIndex, ArrayRef<int64_t>{oh, ow});
        auto ohIndex = mIndices[0];
        auto owIndex = mIndices[1];

        SmallVector<Value> kIndices = unrollIndex(
            nestedBuilder, nestedLoc, kIndex, ArrayRef<int64_t>{fh, fw, ic});
        auto fhIndex = kIndices[0];
        auto fwIndex = kIndices[1];
        auto icIndex = kIndices[2];

        // Extract the input element corresponding to the expanded indices.
        Value hIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, ohIndex, fhIndex,
                              convOp.getStrides().getValues<int64_t>()[0]);
        Value wIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, owIndex, fwIndex,
                              convOp.getStrides().getValues<int64_t>()[1]);

        // im2col[n, oh*ow, fh*fw*ic] = input[n, sh*oh + fh, sw*ow + fw, ic]
        SmallVector<Value> extractionIndices{bIndex, hIndex, wIndex, icIndex};
        Value inputVal = nestedBuilder.create<tensor::ExtractOp>(
            loc, input, extractionIndices);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
      });

  // Because the filter does not share the same batch dimension,
  // the batch dimension is only used in indexing the input and output. Thus
  // we cannot use existing linalg named ops like linalg.batch_matmul.
  // i.e. (B x) M x K * K x N = (B x) M x N
  AffineExpr bDim, mDim, nDim, kDim;
  bindDims(context, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, context);
  auto rhsMap = AffineMap::get(4, 0, {kDim, nDim}, context);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, context);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputType,
      /*inputs=*/ValueRange{img2ColTensor.getResult(0), reshapedFilter},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter,
                linalg::DepthwiseConv2DNhwcHwcOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<RankedTensorType>();
  auto filterType = convOp.getInputs()[1].getType().cast<RankedTensorType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<RankedTensorType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  Location loc = convOp.getLoc();

  auto transposeOperand = [&](Value operand, ArrayRef<int64_t> indices) {
    auto operandTensorType = operand.getType().cast<RankedTensorType>();
    auto nloops = indices.size();
    ArrayRef<int64_t> inputShape = operandTensorType.getShape();

    SmallVector<AffineExpr> exprs = llvm::to_vector<4>(
        llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
          return rewriter.getAffineDimExpr(index);
        }));

    SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
        indices, [&](int64_t index) -> int64_t { return inputShape[index]; }));

    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, targetShape, operandTensorType.getElementType());

    SmallVector<utils::IteratorType> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        inversePermutation(
            AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto transposedOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensor.getType(),
        /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    return transposedOp.getResult(0);
  };

  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  // Transpose input, filter so channels are outermost
  Value inputT = transposeOperand(input, {0, 3, 1, 2});
  Value filterT = transposeOperand(filter, {2, 0, 1});
  ArrayRef<int64_t> filterTShape =
      filterT.getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int n = outputShape[0];
  int oh = outputShape[1];
  int ow = outputShape[2];
  int c = outputShape[3];
  int fh = filterTShape[1];
  int fw = filterTShape[2];

  SmallVector<int64_t> colTensorShape = {n, c, oh, ow, fh, fw};
  Value transposedOutputTensor = transposeOperand(output, {0, 3, 1, 2});

  AffineExpr nDim, cDim, ohDim, owDim, khDim, kwDim;
  bindDims(rewriter.getContext(), nDim, cDim, ohDim, owDim, khDim, kwDim);

  AffineExpr shSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[0]);
  AffineExpr swSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[1]);

  SmallVector<AffineExpr> inputExprs = {nDim, cDim, ohDim * shSym + khDim,
                                        owDim * swSym + kwDim};

  auto nloops = colTensorShape.size();

  SmallVector<utils::IteratorType> loopAttributeTypes(
      nloops, utils::IteratorType::parallel);

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/inputT, /*outputs=*/colTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  SmallVector<ReassociationIndices> img2ColTensorReassocIndices = {
      {0, 1}, {2, 3}, {4, 5}};
  SmallVector<ReassociationIndices> filterReassociationIndice = {{0}, {1, 2}};
  SmallVector<ReassociationIndices> outputReassociationIndice = {{0, 1},
                                                                 {2, 3}};

  auto reshapedImg2ColTensorType = RankedTensorType::get(
      {n * c, oh * ow, fh * fw}, inputType.getElementType());
  auto reshapedFilterTensorType =
      RankedTensorType::get({c, fh * fw}, filterType.getElementType());
  auto reshapedOutputTensorType =
      RankedTensorType::get({n * c, oh * ow}, outputType.getElementType());

  Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
      img2ColTensorReassocIndices);
  Value reshapedFilterTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterTensorType, filterT, filterReassociationIndice);
  Value reshapedoutputTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputTensorType, transposedOutputTensor,
      outputReassociationIndice);

  auto batchMatVecResult = rewriter.create<linalg::BatchMatvecOp>(
      loc, TypeRange{reshapedoutputTensor.getType()},
      ValueRange{reshapedImg2ColTensor, reshapedFilterTensor},
      ValueRange{reshapedoutputTensor});

  SmallVector<ReassociationIndices> batchMatVecReassociationIndice = {{0, 1},
                                                                      {2, 3}};

  Value batchMatVecResultReshaped = rewriter.create<tensor::ExpandShapeOp>(
      loc, transposedOutputTensor.getType(), batchMatVecResult.getResult(0),
      batchMatVecReassociationIndice);

  Value transposedResult =
      transposeOperand(batchMatVecResultReshaped, {0, 2, 3, 1});

  rewriter.replaceOp(convOp, ArrayRef<Value>{transposedResult});
  return std::make_pair(img2ColTensor.getOperation(),
                        transposedResult.getDefiningOp());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<ShapedType>();
  auto filterType = convOp.getInputs()[1].getType().cast<ShapedType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<ShapedType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  auto filterShape = filterType.getShape();
  auto outputShape = outputType.getShape();

  int64_t n = outputShape[0];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];
  int64_t ic = filterShape[1];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];

  auto loc = convOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
  auto reshapedFilterType =
      RankedTensorType::get({oc, ic * fh * fw}, inputType.getElementType());
  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType =
      RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  // Convert the input to a (BKN) tensor.
  SmallVector<int64_t, 4> colTensorShape = {n, ic * fh * fw, oh * ow};
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

  SmallVector<AffineMap, 4> img2colIndexingMaps = {
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/ValueRange{}, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Get the iterators named based on the matmul (batch, m, k).
        Value bIndex = nestedBuilder.create<linalg::IndexOp>(loc, 0);
        Value kIndex = nestedBuilder.create<linalg::IndexOp>(loc, 1);
        Value nIndex = nestedBuilder.create<linalg::IndexOp>(loc, 2);

        // Recover the original iteration indices from the problem/input sizes.
        SmallVector<Value> kIndices = unrollIndex(
            nestedBuilder, nestedLoc, kIndex, ArrayRef<int64_t>{ic, fh, fw});
        auto icIndex = kIndices[0];
        auto fhIndex = kIndices[1];
        auto fwIndex = kIndices[2];

        SmallVector<Value> nIndices = unrollIndex(
            nestedBuilder, nestedLoc, nIndex, ArrayRef<int64_t>{oh, ow});
        auto ohIndex = nIndices[0];
        auto owIndex = nIndices[1];

        // Extract the input element corresponding to the expanded indices.
        Value hIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, ohIndex, fhIndex,
                              convOp.getStrides().getValues<int64_t>()[0]);
        Value wIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, owIndex, fwIndex,
                              convOp.getStrides().getValues<int64_t>()[1]);

        // im2col[n, ic*fh*fw, oh*ow] = input[n, ic, sh*oh + fh, sw*ow + fw]
        SmallVector<Value> extractionIndices{bIndex, icIndex, hIndex, wIndex};
        Value inputVal = nestedBuilder.create<tensor::ExtractOp>(
            loc, input, extractionIndices);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
      });

  // Because the filter does not share the same batch dimension,
  // the batch dimension is only used in indexing the input and output. Thus
  // we cannot use existing linalg named ops like linalg.batch_matmul.
  // i.e. M x K * (B x) K x N = (B x) M x N
  AffineExpr bDim, mDim, nDim, kDim;
  bindDims(context, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {mDim, kDim}, context);
  auto rhsMap = AffineMap::get(4, 0, {bDim, kDim, nDim}, context);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, context);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputType,
      /*inputs=*/ValueRange{reshapedFilter, img2ColTensor.getResult(0)},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

namespace {

class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};

class ConvertDepthwiseConv2DNhwcHwc final
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp> {
public:
  using OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};

class ConvertConv2DNchwFchw final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};
} // end anonymous namespace

void populateConvertConv2DToImg2ColPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ConvertConv2DNhwcHwcf, ConvertDepthwiseConv2DNhwcHwc,
                  ConvertConv2DNchwFchw>(context);
}
} // end namespace linalg
} // end namespace mlir
