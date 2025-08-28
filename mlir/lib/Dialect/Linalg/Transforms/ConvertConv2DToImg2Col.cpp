//===- ConvertConv2DToImg2Col.cpp - im2col implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <utility>

namespace mlir {
namespace linalg {
static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  if (isa<IntegerType>(x.getType()))
    return arith::AddIOp::create(builder, loc, x, y);
  if (isa<ComplexType>(x.getType()))
    return complex::AddOp::create(builder, loc, x, y);
  return arith::AddFOp::create(builder, loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (isa<ComplexType>(accType))
    return complex::MulOp::create(builder, loc, xConvert, yConvert);
  if (isa<IntegerType>(accType))
    return arith::MulIOp::create(builder, loc, xConvert, yConvert);
  return arith::MulFOp::create(builder, loc, xConvert, yConvert);
}

// Generate the affine expression to compute the convolved index
// for the input as `oIndex * stride + fIndex`,
// where oIndex: output iterator; fIndex: filter iterator.
static AffineExpr getConvolvedExpr(OpBuilder &b, int64_t stride,
                                   bool useSymbols = true) {
  AffineExpr oExpr, fExpr;
  if (useSymbols)
    bindSymbols(b.getContext(), oExpr, fExpr);
  else
    bindDims(b.getContext(), oExpr, fExpr);
  return AffineExpr(stride * oExpr + fExpr);
}

// Stores the affine expressions to map the iteration space of the im2col matrix
// to the corresponding indices of the output and filter matrices
struct Im2ColToOperandsExprs {
  AffineExpr fhIndex;
  AffineExpr fwIndex;
  AffineExpr icIndex;
  AffineExpr ohIndex;
  AffineExpr owIndex;
};

// Stores the affine expressions to map the iteration space of the im2col matrix
// to the input matrix indices
struct Im2ColToInputDimsExprs {
  AffineExpr bIndex;
  AffineExpr hIndex;
  AffineExpr wIndex;
  AffineExpr cIndex;
};

/// Construct the affine expressions that map the indices of the im2col matrix
/// to the corresponding input tensor indices for a 2D convolution with the the
/// provided strides.
///
/// @param exprs      Affine expressions for output and filter indices.
/// @param strides    [height, width] stride values for the convolution.
/// @param rewriter   Pattern rewriter.
/// @return           Affine expressions mapping im2col matrix indices to input
/// offsets.
static Im2ColToInputDimsExprs
getIm2ColInputExpressions(Im2ColToOperandsExprs exprs,
                          ArrayRef<int64_t> strides, RewriterBase &rewriter) {
  // maps the iteration space of the im2col matrix to (output_y, filter_y)
  auto hIndicesMap = AffineMap::inferFromExprList(
      {ArrayRef{exprs.ohIndex, exprs.fhIndex}}, rewriter.getContext())[0];
  // maps the iteration space of the im2col matrix to (output_x, filter_x)
  auto wIndicesMap = AffineMap::inferFromExprList(
      {ArrayRef{exprs.owIndex, exprs.fwIndex}}, rewriter.getContext())[0];
  // Compute the input indexing map, to map the indices of the im2col matrix to
  // the original input offsets. Each element of the im2col matrix corresponds
  // to a pair of (out_element, filter_element). First, we build the expressions
  // to compute the input (ix, iy) indices from [out_x/y, filter_x/y] pairs;
  // then we compose them with the maps that map the im2col matrix elements to
  // the (out_element, filter_element) pairs.
  auto bIndexExpr = rewriter.getAffineDimExpr(0U);
  auto hIndexExpr = getConvolvedExpr(rewriter, strides[0],
                                     /*useSymbols*/ false);
  hIndexExpr = hIndexExpr.compose(hIndicesMap);
  auto wIndexExpr = getConvolvedExpr(rewriter, strides[1],
                                     /*useSymbols*/ false);
  wIndexExpr = wIndexExpr.compose(wIndicesMap);
  auto cIndexExpr = exprs.icIndex;
  return {bIndexExpr, hIndexExpr, wIndexExpr, cIndexExpr};
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp convOp) {
  auto inputType = cast<ShapedType>(convOp.getInputs()[0].getType());
  auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
  auto outputType = cast<ShapedType>(convOp.getOutputs()[0].getType());

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
      RankedTensorType::get({fh * fw * ic, oc}, filterType.getElementType());
  Value reshapedFilter = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}};
  RankedTensorType reshapedOutputType =
      RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
  Value reshapedOutput = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedOutputType, output, outputReassocIndices);

  SmallVector<int64_t> colTensorShape = {n, oh * ow, fh * fw * ic};
  Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                            inputType.getElementType());

  // Convert the input to a (BMK) column tensor.
  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> img2colIterators(nloops, parallel);

  // Given an index of the im2col matrix, retrieve the corresponding indices of
  // the output and filter matrices
  auto mIndicesExprs =
      delinearize(rewriter.getAffineDimExpr(1U), ArrayRef<int64_t>{ow, 1});
  auto kIndicesExprs = delinearize(rewriter.getAffineDimExpr(2U),
                                   ArrayRef<int64_t>{fw * ic, ic, 1});
  Im2ColToOperandsExprs i2cToOperExprs;
  i2cToOperExprs.fhIndex = kIndicesExprs[0];
  i2cToOperExprs.fwIndex = kIndicesExprs[1];
  i2cToOperExprs.icIndex = kIndicesExprs[2];
  i2cToOperExprs.ohIndex = mIndicesExprs[0];
  i2cToOperExprs.owIndex = mIndicesExprs[1];

  // im2col[n, oh*ow, fh*fw*ic] = input[n, sh*oh + fh, sw*ow + fw, ic]
  Im2ColToInputDimsExprs inExprs = getIm2ColInputExpressions(
      i2cToOperExprs, llvm::to_vector(convOp.getStrides().getValues<int64_t>()),
      rewriter);
  auto inMap =
      AffineMap::inferFromExprList({ArrayRef{inExprs.bIndex, inExprs.hIndex,
                                             inExprs.wIndex, inExprs.cIndex}},
                                   rewriter.getContext())[0];

  SmallVector<AffineMap> img2colIndexingMaps = {
      inMap, AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = linalg::GenericOp::create(
      rewriter, loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
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

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, reshapedOutputType,
      /*inputs=*/ValueRange{img2ColTensor.getResult(0), reshapedFilter},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = tensor::ExpandShapeOp::create(
      rewriter, loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter,
                linalg::DepthwiseConv2DNhwcHwcOp convOp) {
  auto inputType = cast<RankedTensorType>(convOp.getInputs()[0].getType());
  auto filterType = cast<RankedTensorType>(convOp.getInputs()[1].getType());
  auto outputType = cast<RankedTensorType>(convOp.getOutputs()[0].getType());

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
    auto operandTensorType = cast<RankedTensorType>(operand.getType());
    auto nloops = indices.size();
    ArrayRef<int64_t> inputShape = operandTensorType.getShape();

    SmallVector<AffineExpr> exprs = llvm::to_vector<4>(
        llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
          return rewriter.getAffineDimExpr(index);
        }));

    SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
        indices, [&](int64_t index) -> int64_t { return inputShape[index]; }));

    Value outputTensor = tensor::EmptyOp::create(
        rewriter, loc, targetShape, operandTensorType.getElementType());

    SmallVector<utils::IteratorType> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        inversePermutation(
            AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto transposedOp = linalg::GenericOp::create(
        rewriter, loc, outputTensor.getType(),
        /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
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
      cast<RankedTensorType>(filterT.getType()).getShape();
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

  Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                            inputType.getElementType());

  auto img2ColTensor = linalg::GenericOp::create(
      rewriter, loc, colTensor.getType(),
      /*inputs=*/inputT, /*outputs=*/colTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
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

  Value reshapedImg2ColTensor = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
      img2ColTensorReassocIndices);
  Value reshapedFilterTensor =
      tensor::CollapseShapeOp::create(rewriter, loc, reshapedFilterTensorType,
                                      filterT, filterReassociationIndice);
  Value reshapedoutputTensor = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedOutputTensorType, transposedOutputTensor,
      outputReassociationIndice);

  auto batchMatVecResult = linalg::BatchMatvecOp::create(
      rewriter, loc, TypeRange{reshapedoutputTensor.getType()},
      ValueRange{reshapedImg2ColTensor, reshapedFilterTensor},
      ValueRange{reshapedoutputTensor});

  SmallVector<ReassociationIndices> batchMatVecReassociationIndice = {{0, 1},
                                                                      {2, 3}};

  auto batchMatVecResultReshaped = tensor::ExpandShapeOp::create(
      rewriter, loc, transposedOutputTensor.getType(),
      batchMatVecResult.getResult(0), batchMatVecReassociationIndice);

  Value transposedResult =
      transposeOperand(batchMatVecResultReshaped, {0, 2, 3, 1});

  rewriter.replaceOp(convOp, ArrayRef<Value>{transposedResult});
  return std::make_pair(img2ColTensor.getOperation(),
                        transposedResult.getDefiningOp());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp convOp) {
  auto inputType = cast<ShapedType>(convOp.getInputs()[0].getType());
  auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
  auto outputType = cast<ShapedType>(convOp.getOutputs()[0].getType());

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
  Value reshapedFilter = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType =
      RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedOutputType, output, outputReassocIndices);

  // Convert the input to a (BKN) tensor.
  SmallVector<int64_t, 4> colTensorShape = {n, ic * fh * fw, oh * ow};
  Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                            inputType.getElementType());

  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

  // Recover the original iteration indices from the problem/input sizes:
  // given an index of the im2col matrix, retrieve the corresponding indices of
  // the output and filter matrices
  auto kIndicesExprs = delinearize(rewriter.getAffineDimExpr(1U),
                                   ArrayRef<int64_t>{fh * fw, fw, 1});
  auto mIndicesExprs =
      delinearize(rewriter.getAffineDimExpr(2U), ArrayRef<int64_t>{ow, 1});
  Im2ColToOperandsExprs i2cToOperExprs;
  i2cToOperExprs.icIndex = kIndicesExprs[0];
  i2cToOperExprs.fhIndex = kIndicesExprs[1];
  i2cToOperExprs.fwIndex = kIndicesExprs[2];
  i2cToOperExprs.ohIndex = mIndicesExprs[0];
  i2cToOperExprs.owIndex = mIndicesExprs[1];
  Im2ColToInputDimsExprs inExprs = getIm2ColInputExpressions(
      i2cToOperExprs, llvm::to_vector(convOp.getStrides().getValues<int64_t>()),
      rewriter);
  auto inMap =
      AffineMap::inferFromExprList({ArrayRef{inExprs.bIndex, inExprs.cIndex,
                                             inExprs.hIndex, inExprs.wIndex}},
                                   rewriter.getContext())[0];
  // im2col[n, ic*fh*fw, oh*ow] = input[n, ic, sh*oh + fh, sw*ow + fw]
  SmallVector<AffineMap> img2colIndexingMaps = {
      inMap, AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = linalg::GenericOp::create(
      rewriter, loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
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
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, reshapedOutputType,
      /*inputs=*/ValueRange{reshapedFilter, img2ColTensor.getResult(0)},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = tensor::ExpandShapeOp::create(
      rewriter, loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcFhwcOp convOp) {
  auto inputType = cast<ShapedType>(convOp.getInputs()[0].getType());
  auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
  auto outputType = cast<ShapedType>(convOp.getOutputs()[0].getType());

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
  int64_t fh = filterShape[1];
  int64_t fw = filterShape[2];
  int64_t ic = filterShape[3];

  Location loc = convOp.getLoc();

  // Reshape output and filter to the LHS and result of a "row-wise" matrix
  // multiplication.
  SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
  auto reshapedFilterType =
      RankedTensorType::get({oc, fh * fw * ic}, filterType.getElementType());
  Value reshapedFilter = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}};
  RankedTensorType reshapedOutputType =
      RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
  Value reshapedOutput = tensor::CollapseShapeOp::create(
      rewriter, loc, reshapedOutputType, output, outputReassocIndices);

  // Shape of the Toeplitz matrix produced by Im2col.
  SmallVector<int64_t> colTensorShape = {n, oh * ow, fh * fw * ic};
  Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                            inputType.getElementType());

  // Convert the input to a (BMK) column tensor.
  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> img2colIterators(nloops, parallel);

  // Given an index of the im2col matrix, retrieve the corresponding indices of
  // the output and filter matrices
  auto mIndicesExprs =
      delinearize(rewriter.getAffineDimExpr(1U), ArrayRef<int64_t>{ow, 1});
  auto kIndicesExprs = delinearize(rewriter.getAffineDimExpr(2U),
                                   ArrayRef<int64_t>{fw * ic, ic, 1});
  Im2ColToOperandsExprs i2cToOperExprs;
  i2cToOperExprs.fhIndex = kIndicesExprs[0];
  i2cToOperExprs.fwIndex = kIndicesExprs[1];
  i2cToOperExprs.icIndex = kIndicesExprs[2];
  i2cToOperExprs.ohIndex = mIndicesExprs[0];
  i2cToOperExprs.owIndex = mIndicesExprs[1];

  // im2col[n, oh*ow, fh*fw*ic] = input[n, sh*oh + fh, sw*ow + fw, ic]
  Im2ColToInputDimsExprs inExprs = getIm2ColInputExpressions(
      i2cToOperExprs, llvm::to_vector(convOp.getStrides().getValues<int64_t>()),
      rewriter);
  auto inMap =
      AffineMap::inferFromExprList({ArrayRef{inExprs.bIndex, inExprs.hIndex,
                                             inExprs.wIndex, inExprs.cIndex}},
                                   rewriter.getContext())[0];
  SmallVector<AffineMap> img2colIndexingMaps = {
      inMap, AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = linalg::GenericOp::create(
      rewriter, loc, colTensor.getType(),
      /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
      });

  // Because we didn't transpose the filters we don't actually have a batched
  // matrix multiply. Instead, we have an operation consisting of "row-wise" dot
  // products.
  AffineExpr bDim, mDim, nDim, kDim;
  bindDims(context, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, context);
  auto rhsMap = AffineMap::get(4, 0, {nDim, kDim}, context);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, context);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, reshapedOutputType,
      /*inputs=*/ValueRange{img2ColTensor.getResult(0), reshapedFilter},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = tensor::ExpandShapeOp::create(
      rewriter, loc, outputType, result, outputReassocIndices);

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

class ConvertConv2DNhwcFhwc final
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
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
                  ConvertConv2DNchwFchw, ConvertConv2DNhwcFhwc>(context);
}
} // end namespace linalg
} // end namespace mlir
