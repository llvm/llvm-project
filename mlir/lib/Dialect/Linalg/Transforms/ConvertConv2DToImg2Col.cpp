//===- ConvertConv2DToImg2Col.cpp - im2col implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = x.getType().isa<IntegerType>();
  if (isInt)
    return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
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

  int n = outputShape[0];
  int oh = outputShape[1];
  int ow = outputShape[2];
  int oc = outputShape[3];
  int fh = filterShape[0];
  int fw = filterShape[1];
  int ic = filterShape[2];

  Location loc = convOp.getLoc();

  SmallVector<int64_t> colTensorShape = {n, oh, ow, fh, fw, ic};

  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  AffineExpr nDim, ohDim, owDim, khDim, kwDim, icDim;
  bindDims(context, nDim, ohDim, owDim, khDim, kwDim, icDim);

  AffineExpr shSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[0]);
  AffineExpr swSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[1]);

  SmallVector<AffineExpr> inputExprs = {nDim, ohDim * shSym + khDim,
                                        owDim * swSym + kwDim, icDim};

  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> img2colIterators(nloops, parallel);

  SmallVector<AffineMap> img2colIndexingMaps = {
      AffineMap::get(nloops, 0, inputExprs, context),
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  SmallVector<ReassociationIndices> img2ColTensorReassocIndices;
  SmallVector<ReassociationIndices> outputReassocIndices;
  RankedTensorType reshapedImg2ColTensorType, reshapedOutputType;
  if (n == 1) {
    img2ColTensorReassocIndices = {{0, 1, 2}, {3, 4, 5}};
    outputReassocIndices = {{0, 1, 2}, {3}};

    reshapedImg2ColTensorType = RankedTensorType::get(
        {oh * ow, fh * fw * ic}, inputType.getElementType());
    reshapedOutputType =
        RankedTensorType::get({oh * ow, oc}, outputType.getElementType());
  } else {
    img2ColTensorReassocIndices = {{0}, {1, 2}, {3, 4, 5}};
    outputReassocIndices = {{0}, {1, 2}, {3}};

    reshapedImg2ColTensorType = RankedTensorType::get(
        {n, oh * ow, fh * fw * ic}, inputType.getElementType());
    reshapedOutputType =
        RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
  }

  SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
  auto reshapedFilterType =
      RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());

  Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
      img2ColTensorReassocIndices);

  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  Value result;
  if (n == 1) {
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType,
        ArrayRef<Value>{reshapedImg2ColTensor, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});
    result = matmulOp.getResults().front();
  } else {
    // For cases where batch is not 1, we need to keep the batch dimension
    // separate. Because the filter does not share the same batch dimension,
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
        /*inputs=*/ValueRange{reshapedImg2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{reshapedOutput},
        ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value mul = createMul(loc, args[0], args[1], nestedBuilder);
          Value add = createAdd(loc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });
    result = genericOp.getResults().front();
  }

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

  int n = outputShape[0];
  int oc = outputShape[1];
  int oh = outputShape[2];
  int ow = outputShape[3];
  int ic = filterShape[1];
  int fh = filterShape[2];
  int fw = filterShape[3];

  auto loc = convOp.getLoc();

  SmallVector<int64_t, 4> colTensorShape = {n, ic, fh, fw, oh, ow};

  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  MLIRContext *context = rewriter.getContext();

  AffineExpr nDim, icDim, khDim, kwDim, ohDim, owDim;
  bindDims(context, nDim, icDim, khDim, kwDim, ohDim, owDim);

  auto shSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[0]);
  auto swSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[1]);

  SmallVector<AffineExpr, 4> inputExprs = {nDim, icDim, ohDim * shSym + khDim,
                                           owDim * swSym + kwDim};

  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

  SmallVector<AffineMap, 4> img2colIndexingMaps = {
      AffineMap::get(nloops, 0, inputExprs, context),
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
  auto reshapedFilterType =
      RankedTensorType::get({oc, fh * fw * ic}, inputType.getElementType());
  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> img2ColTensorReassocIndices;
  SmallVector<ReassociationIndices> outputReassocIndices;
  RankedTensorType reshapedImg2ColTensorType, reshapedOutputType;
  if (n == 1) {
    img2ColTensorReassocIndices = {{0, 1, 2, 3}, {4, 5}};
    outputReassocIndices = {{0, 1}, {2, 3}};

    reshapedImg2ColTensorType = RankedTensorType::get(
        {fh * fw * ic, oh * ow}, inputType.getElementType());
    reshapedOutputType =
        RankedTensorType::get({oc, oh * ow}, outputType.getElementType());
  } else {
    img2ColTensorReassocIndices = {{0}, {1, 2, 3}, {4, 5}};
    outputReassocIndices = {{0}, {1}, {2, 3}};

    reshapedImg2ColTensorType = RankedTensorType::get(
        {n, fh * fw * ic, oh * ow}, inputType.getElementType());
    reshapedOutputType =
        RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  }

  Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
      img2ColTensorReassocIndices);

  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  Value result;
  if (n == 1) {
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType,
        ArrayRef<Value>{reshapedFilter, reshapedImg2ColTensor},
        ArrayRef<Value>{reshapedOutput});
    result = matmulOp.getResults().front();
  } else {
    // For cases where batch is not 1, we need to keep the batch dimension
    // separate. Because the filter does not share the same batch dimension,
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
        /*inputs=*/ValueRange{reshapedFilter, reshapedImg2ColTensor},
        /*outputs=*/ValueRange{reshapedOutput},
        ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value mul = createMul(loc, args[0], args[1], nestedBuilder);
          Value add = createAdd(loc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });
    result = genericOp.getResults().front();
  }

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
