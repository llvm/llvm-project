//===- TosaDecomposeTransposeConv.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose TOSA TransposeConv operation to a series of TOSA Ops specifically
// (1) Convert a Dilated TransposeConv2D to Conv2D including reversing/reshaping
// etc.. of the weights (2) Convert a Strided TransposeConv2D to Conv2D
// including transposing/reversing/reshaping etc..
//     of the weights and input/output tenors and reversing/reshaping etc .. of
//     the weights
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct StridedTransposeConvPrep {
  explicit StridedTransposeConvPrep(Location l) : loc(l) {}

  Location loc;
  Value input;
  Value weight;
  Value bias;

  ShapedType inputTy;
  ShapedType weightTy;
  ShapedType biasTy;
  ShapedType resultTy;

  Type inputETy;
  Type weightETy;
  Type biasETy;
  Type resultETy;

  llvm::ArrayRef<int64_t> pad;
  llvm::ArrayRef<int64_t> stride;

  int64_t batch;
  int64_t outputChannels;
  int64_t inputChannels;
  int64_t inputZpVal;
  int64_t weightZpVal;
};

static LogicalResult prepareStridedTransposeConv(tosa::TransposeConv2DOp op,
                                                 PatternRewriter &rewriter,
                                                 StridedTransposeConvPrep &p) {
  p.loc = op->getLoc();
  p.input = op->getOperand(0);
  p.weight = op->getOperand(1);
  p.bias = op->getOperand(2);

  p.inputTy = cast<ShapedType>(p.input.getType());
  p.weightTy = cast<ShapedType>(p.weight.getType());
  p.biasTy = cast<ShapedType>(p.bias.getType());
  p.resultTy = cast<ShapedType>(op->getResult(0).getType());

  p.inputETy = p.inputTy.getElementType();
  p.weightETy = p.weightTy.getElementType();
  p.biasETy = p.biasTy.getElementType();
  p.resultETy = p.resultTy.getElementType();

  p.pad = op.getOutPad();
  p.stride = op.getStride();

  if (llvm::all_of(p.stride, [](int64_t v) { return v == 1; }))
    return rewriter.notifyMatchFailure(op, "non-one stride found.");

  for (unsigned int i = 1; i < 4; ++i) {
    if (p.inputTy.isDynamicDim(i) || p.resultTy.isDynamicDim(i))
      return failure();
  }

  if (!p.weightTy.hasStaticShape() || !p.biasTy.hasStaticShape())
    return failure();

  FailureOr<int64_t> maybeIZp = op.getInputZeroPoint();
  if (failed(maybeIZp))
    return rewriter.notifyMatchFailure(
        op, "input zero point cannot be statically determined");

  FailureOr<int64_t> maybeWZp = op.getWeightZeroPoint();
  if (failed(maybeWZp))
    return rewriter.notifyMatchFailure(
        op, "weight zero point cannot be statically determined");

  p.inputZpVal = *maybeIZp;
  p.weightZpVal = *maybeWZp;

  if (op.verifyInputZeroPoint(p.inputZpVal).failed())
    return rewriter.notifyMatchFailure(
        op, "input zero point must be zero for non-int8 integer types");

  if (op.verifyWeightZeroPoint(p.weightZpVal).failed())
    return rewriter.notifyMatchFailure(
        op, "weight zero point must be zero for non-int8 integer types");

  p.batch = p.inputTy.getDimSize(0);
  p.outputChannels = p.weightTy.getDimSize(0);
  int64_t weightHeight = p.weightTy.getDimSize(1);
  int64_t weightWidth = p.weightTy.getDimSize(2);
  p.inputChannels = p.weightTy.getDimSize(3);

  llvm::SmallVector<int64_t, 8> weightPadding = {0, 0, 0, 0, 0, 0, 0, 0};
  weightPadding[3] = (weightHeight % p.stride[0])
                         ? (p.stride[0] - weightHeight % p.stride[0])
                         : 0;
  weightPadding[5] = (weightWidth % p.stride[1])
                         ? (p.stride[1] - weightWidth % p.stride[1])
                         : 0;

  Value weightPaddingVal =
      getTosaConstShape(rewriter, op->getLoc(), weightPadding);

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  const Value inputPadConst =
      createPadConstTensor(builder, op->getLoc(), p.input, p.inputZpVal);
  const Value weightPadConst =
      createPadConstTensor(builder, op->getLoc(), p.input, p.weightZpVal);

  p.weight = CreateOpAndInferShape<tosa::PadOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.weightETy), p.weight,
      weightPaddingVal, weightPadConst);

  p.weightTy = cast<ShapedType>(p.weight.getType());
  weightHeight = p.weightTy.getDimSize(1);
  weightWidth = p.weightTy.getDimSize(2);

  llvm::SmallVector<int64_t, 6> weightReshapeDims0 = {
      p.outputChannels, weightHeight / p.stride[0],
      p.stride[0],      weightWidth / p.stride[1],
      p.stride[1],      p.inputChannels};

  p.weight = CreateOpAndInferShape<tosa::ReshapeOp>(
      builder, UnrankedTensorType::get(p.weightETy), p.weight,
      getTosaConstShape(rewriter, p.loc, weightReshapeDims0));

  p.weight = CreateOpAndInferShape<tosa::TransposeOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.weightETy), p.weight,
      rewriter.getDenseI32ArrayAttr({2, 4, 0, 1, 3, 5}));

  llvm::SmallVector<int64_t, 4> weightReshapeDims1 = {
      p.outputChannels * p.stride[0] * p.stride[1], weightHeight / p.stride[0],
      weightWidth / p.stride[1], p.inputChannels};

  p.weight = CreateOpAndInferShape<tosa::ReshapeOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.weightETy), p.weight,
      getTosaConstShape(rewriter, p.loc, weightReshapeDims1));
  ShapedType restridedWeightTy = cast<ShapedType>(p.weight.getType());

  p.weight = CreateOpAndInferShape<tosa::ReverseOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.weightETy), p.weight,
      /* axis = */ rewriter.getI32IntegerAttr(1));
  p.weight = CreateOpAndInferShape<tosa::ReverseOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.weightETy), p.weight,
      /* axis = */ rewriter.getI32IntegerAttr(2));

  llvm::SmallVector<int64_t, 8> inputPadding = {0, 0, 0, 0, 0, 0, 0, 0};
  inputPadding[2] += restridedWeightTy.getDimSize(1) - 1;
  inputPadding[3] += restridedWeightTy.getDimSize(1) - 1;
  inputPadding[4] += restridedWeightTy.getDimSize(2) - 1;
  inputPadding[5] += restridedWeightTy.getDimSize(2) - 1;

  Value inputPaddingVal =
      getTosaConstShape(rewriter, op->getLoc(), inputPadding);

  p.input = CreateOpAndInferShape<tosa::PadOp>(
      rewriter, p.loc, UnrankedTensorType::get(p.inputETy), p.input,
      inputPaddingVal, inputPadConst);

  return success();
}

static FailureOr<Value> buildConv2DFromPreparedInput(
    tosa::TransposeConv2DOp op, PatternRewriter &rewriter,
    const StridedTransposeConvPrep &p, Value convBias, Type convResultElemTy) {
  auto inputZp =
      createZeroPointTensor(rewriter, p.loc, p.input.getType(), p.inputZpVal);
  auto weightZp =
      createZeroPointTensor(rewriter, p.loc, p.weight.getType(), p.weightZpVal);

  if (!inputZp.has_value() || !weightZp.has_value()) {
    return failure();
  }

  Value conv2d =
      CreateOpAndInferShape<tosa::Conv2DOp>(
          rewriter, p.loc, UnrankedTensorType::get(convResultElemTy), p.input,
          p.weight, convBias, inputZp.value(), weightZp.value(),
          /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
          /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
          /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}),
          /* acc_type = */ op.getAccType())
          .getResult();

  return conv2d;
}

static Value shuffleSliceAndPadResult(PatternRewriter &rewriter, Location loc,
                                      Value value, Type resultElemTy,
                                      ShapedType resultTy, int64_t batch,
                                      int64_t outputChannels,
                                      llvm::ArrayRef<int64_t> stride,
                                      llvm::ArrayRef<int64_t> pad) {
  ShapedType convTy = cast<ShapedType>(value.getType());
  Type convETy = convTy.getElementType();

  int64_t convHeight = convTy.getDimSize(1);
  int64_t convWidth = convTy.getDimSize(2);

  llvm::SmallVector<int64_t, 6> convReshapeDims0 = {
      batch, convHeight, convWidth, stride[0], stride[1], outputChannels};

  auto convReshapeDims0Value =
      getTosaConstShape(rewriter, loc, convReshapeDims0);

  value = CreateOpAndInferShape<tosa::ReshapeOp>(
      rewriter, loc, UnrankedTensorType::get(resultElemTy), value,
      convReshapeDims0Value);

  value = CreateOpAndInferShape<tosa::TransposeOp>(
      rewriter, loc, UnrankedTensorType::get(convETy), value,
      rewriter.getDenseI32ArrayAttr({0, 1, 3, 2, 4, 5}));

  llvm::SmallVector<int64_t, 6> convReshapeDims1 = {
      batch, convHeight * stride[0], convWidth * stride[1], outputChannels};

  auto convReshapeDims1Value =
      getTosaConstShape(rewriter, loc, convReshapeDims1);

  value = CreateOpAndInferShape<tosa::ReshapeOp>(
      rewriter, loc, UnrankedTensorType::get(resultElemTy), value,
      convReshapeDims1Value);

  int64_t resultSliceTop = std::max<int64_t>(0, -pad[0]);
  int64_t resultSliceLeft = std::max<int64_t>(0, -pad[2]);
  int64_t resultPadTop = std::max<int64_t>(0, pad[0]);
  int64_t resultPadLeft = std::max<int64_t>(0, pad[2]);

  int64_t resultSliceHeight =
      std::min<int64_t>(convReshapeDims1[1] - resultSliceTop,
                        resultTy.getDimSize(1) - resultPadTop);
  int64_t resultSliceWidth =
      std::min<int64_t>(convReshapeDims1[2] - resultSliceLeft,
                        resultTy.getDimSize(2) - resultPadLeft);

  llvm::SmallVector<int64_t, 4> sliceBegin = {0, resultSliceTop,
                                              resultSliceLeft, 0};
  llvm::SmallVector<int64_t, 4> sliceSize(convReshapeDims1.begin(),
                                          convReshapeDims1.end());
  sliceSize[1] = resultSliceHeight;
  sliceSize[2] = resultSliceWidth;

  auto slice = CreateOpAndInferShape<tosa::SliceOp>(
                   rewriter, loc, UnrankedTensorType::get(resultElemTy), value,
                   getTosaConstShape(rewriter, loc, sliceBegin),
                   getTosaConstShape(rewriter, loc, sliceSize))
                   .getResult();

  llvm::SmallVector<int64_t, 8> resultPadding = {0, 0, 0, 0, 0, 0, 0, 0};
  resultPadding[2] = resultPadTop;
  resultPadding[3] = resultTy.getDimSize(1) - resultPadTop - sliceSize[1];
  resultPadding[4] = resultPadLeft;
  resultPadding[5] = resultTy.getDimSize(2) - resultPadLeft - sliceSize[2];

  Value resultPaddingVal = getTosaConstShape(rewriter, loc, resultPadding);

  return CreateOpAndInferShape<tosa::PadOp>(
      rewriter, loc, UnrankedTensorType::get(resultElemTy), slice,
      resultPaddingVal);
}

class TransposeConvRescaleDecompose : public OpRewritePattern<tosa::RescaleOp> {
public:
  explicit TransposeConvRescaleDecompose(MLIRContext *ctx)
      : OpRewritePattern<tosa::RescaleOp>(ctx, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(tosa::RescaleOp rescaleOp,
                                PatternRewriter &rewriter) const final {
    // Match the pair: transpose_conv2d -> rescale.
    // We intentionally anchor on rescale (instead of transpose_conv2d) so we
    // can move quantization earlier in the decomposition flow.
    //
    // Why this matters:
    // - transpose_conv2d may produce i48 accumulators (for example i16 input +
    // i8 weight).
    // - Some follow-up reshape/transpose paths are not legal/efficient on i48.
    // - Running rescale early converts conv output to a supported narrow type
    //   (typically i16/i8), so later reshape/transpose/slice are cheaper and
    //   legal.
    //
    // This corresponds to the preferred strategy:
    //   conv2d + bias + rescale -> reshape + transpose + reshape + slice
    // and avoids carrying wide accumulator types through depth-to-space style
    // ops.
    auto op = dyn_cast_or_null<tosa::TransposeConv2DOp>(
        rescaleOp.getInput().getDefiningOp());
    if (!op)
      return rewriter.notifyMatchFailure(rescaleOp,
                                         "input is not transpose_conv2d");

    StridedTransposeConvPrep p(op.getLoc());
    if (failed(prepareStridedTransposeConv(op, rewriter, p)))
      return failure();

    int64_t newOC = p.outputChannels * p.stride[0] * p.stride[1];

    DenseElementsAttr biasAttr;
    if (!matchPattern(p.bias, m_Constant(&biasAttr)))
      return rewriter.notifyMatchFailure(op, "bias must be a static constant");

    auto biasVals = llvm::to_vector(biasAttr.getValues<Attribute>());
    // Handle both scalar bias (shape [1]) and per-channel bias (shape [OC])
    if (!(biasVals.size() == 1 ||
          static_cast<int64_t>(biasVals.size()) == p.outputChannels))
      return rewriter.notifyMatchFailure(
          op, "bias must be scalar [1] or per-channel [OC]");

    SmallVector<Attribute> tiledBias(newOC);
    for (int64_t s0 = 0; s0 < p.stride[0]; ++s0)
      for (int64_t s1 = 0; s1 < p.stride[1]; ++s1)
        for (int64_t oc = 0; oc < p.outputChannels; ++oc) {
          int64_t biasIdx = (biasVals.size() == 1) ? 0 : oc;
          tiledBias[s0 * p.stride[1] * p.outputChannels +
                    s1 * p.outputChannels + oc] = biasVals[biasIdx];
        }

    auto tiledBiasTy = RankedTensorType::get({newOC}, p.biasETy);
    Value tiledBiasConst =
        tosa::ConstOp::create(
            rewriter, p.loc, tiledBiasTy,
            DenseElementsAttr::get(tiledBiasTy, ArrayRef<Attribute>(tiledBias)))
            .getResult();

    FailureOr<Value> maybeConv = buildConv2DFromPreparedInput(
        op, rewriter, p, tiledBiasConst,
        cast<ShapedType>(op->getResult(0).getType()).getElementType());
    if (failed(maybeConv))
      return rewriter.notifyMatchFailure(
          op, "fail to create transpose-conv decomposition conv2d");

    Value multiplier = rescaleOp.getMultiplier();
    Value shift = rescaleOp.getShift();
    if (rescaleOp.getPerChannel()) {
      DenseElementsAttr multAttr;
      DenseElementsAttr shiftAttr;
      if (!matchPattern(multiplier, m_Constant(&multAttr)) ||
          !matchPattern(shift, m_Constant(&shiftAttr)))
        return rewriter.notifyMatchFailure(
            rescaleOp,
            "per-channel rescale requires static multiplier/shift constants");

      auto multVals = llvm::to_vector(multAttr.getValues<Attribute>());
      auto shiftVals = llvm::to_vector(shiftAttr.getValues<Attribute>());
      // Allow both scalar and per-channel multiplier/shift
      if (!((multVals.size() == 1 ||
             static_cast<int64_t>(multVals.size()) == p.outputChannels) &&
            (shiftVals.size() == 1 ||
             static_cast<int64_t>(shiftVals.size()) == p.outputChannels)))
        return rewriter.notifyMatchFailure(
            rescaleOp,
            "rescale multiplier/shift must be scalar [1] or per-channel [OC]");

      SmallVector<Attribute> tiledMult(newOC);
      SmallVector<Attribute> tiledShift(newOC);
      for (int64_t s0 = 0; s0 < p.stride[0]; ++s0)
        for (int64_t s1 = 0; s1 < p.stride[1]; ++s1)
          for (int64_t oc = 0; oc < p.outputChannels; ++oc) {
            int64_t ix = s0 * p.stride[1] * p.outputChannels +
                         s1 * p.outputChannels + oc;
            int64_t multIdx = (multVals.size() == 1) ? 0 : oc;
            int64_t shiftIdx = (shiftVals.size() == 1) ? 0 : oc;
            tiledMult[ix] = multVals[multIdx];
            tiledShift[ix] = shiftVals[shiftIdx];
          }

      auto multETy = cast<ShapedType>(multiplier.getType()).getElementType();
      auto shiftETy = cast<ShapedType>(shift.getType()).getElementType();
      auto tiledMultTy = RankedTensorType::get({newOC}, multETy);
      auto tiledShiftTy = RankedTensorType::get({newOC}, shiftETy);

      multiplier = tosa::ConstOp::create(
                       rewriter, p.loc, tiledMultTy,
                       DenseElementsAttr::get(tiledMultTy,
                                              ArrayRef<Attribute>(tiledMult)))
                       .getResult();
      shift = tosa::ConstOp::create(
                  rewriter, p.loc, tiledShiftTy,
                  DenseElementsAttr::get(tiledShiftTy,
                                         ArrayRef<Attribute>(tiledShift)))
                  .getResult();
    }

    Type rescaleOutETy =
        cast<ShapedType>(rescaleOp.getResult().getType()).getElementType();
    Value rescaled =
        CreateOpAndInferShape<tosa::RescaleOp>(
            rewriter, p.loc, UnrankedTensorType::get(rescaleOutETy), *maybeConv,
            multiplier, shift, rescaleOp.getInputZp(), rescaleOp.getOutputZp(),
            rescaleOp.getScale32Attr(), rescaleOp.getRoundingModeAttr(),
            rescaleOp.getPerChannelAttr(), rescaleOp.getInputUnsignedAttr(),
            rescaleOp.getOutputUnsignedAttr())
            .getResult();

    ShapedType rescaleTy = cast<ShapedType>(rescaleOp.getResult().getType());
    Value resultPad = shuffleSliceAndPadResult(
        rewriter, p.loc, rescaled, rescaleOutETy, rescaleTy, p.batch,
        p.outputChannels, p.stride, p.pad);

    rewriter.replaceOp(rescaleOp, resultPad);
    if (op->use_empty())
      rewriter.eraseOp(op);
    return success();
  }
};

class TransposeConvNonStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    llvm::ArrayRef<int64_t> stride = op.getStride();
    llvm::ArrayRef<int64_t> pad = op.getOutPad();

    // If striding is all 1 we can modify padding and reverse the kernel along
    // the x/y direction to make it a regular convolution. This is much simpler
    // then handling striding....
    if (llvm::any_of(stride, [](int64_t v) { return v != 1; }))
      return failure();

    // Any dimensions other than batchSize cannot be dynamic for input/output
    for (unsigned int i = 1; i < 4; ++i) {
      if (inputTy.isDynamicDim(i) || resultTy.isDynamicDim(i))
        return failure();
    }

    if (!weightTy.hasStaticShape() || !biasTy.hasStaticShape())
      return failure();

    int64_t kernelHeight = weightTy.getDimSize(1);
    int64_t kernelWidth = weightTy.getDimSize(2);

    llvm::SmallVector<int64_t> convPad(4, 0);
    convPad[0] = kernelHeight - 1 + pad[0];
    convPad[1] = kernelHeight - 1 + pad[1];
    convPad[2] = kernelWidth - 1 + pad[2];
    convPad[3] = kernelWidth - 1 + pad[3];

    auto reverse1 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, weight,
                                /* axis = */ rewriter.getI32IntegerAttr(1));
    auto reverse2 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, reverse1,
                                /* axis = */ rewriter.getI32IntegerAttr(2));

    Value conv2d = tosa::Conv2DOp::create(
        rewriter, loc, resultTy, input, reverse2, bias, op.getInputZp(),
        op.getWeightZp(), rewriter.getDenseI64ArrayAttr(convPad),
        rewriter.getDenseI64ArrayAttr(stride),
        rewriter.getDenseI64ArrayAttr({1, 1}),
        /* acc_type = */ op.getAccType());

    rewriter.replaceOp(op, conv2d);
    return success();
  }
};

class TransposeConvStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    StridedTransposeConvPrep p(op.getLoc());
    if (failed(prepareStridedTransposeConv(op, rewriter, p)))
      return failure();

    // We use a zero bias as we need to broadcast the bias.
    auto zeroBias = tosa::ConstOp::create(
        rewriter, p.loc,
        RankedTensorType::get({p.outputChannels * p.stride[0] * p.stride[1]},
                              p.biasETy),
        DenseElementsAttr::get(
            RankedTensorType::get(
                {p.outputChannels * p.stride[0] * p.stride[1]}, p.biasETy),
            rewriter.getZeroAttr(p.biasETy)));

    FailureOr<Value> maybeConv = buildConv2DFromPreparedInput(
        op, rewriter, p, zeroBias,
        cast<ShapedType>(op->getResult(0).getType()).getElementType());
    if (failed(maybeConv))
      return rewriter.notifyMatchFailure(
          op, "fail to create a const zero point tensor");

    Value resultPad = shuffleSliceAndPadResult(
        rewriter, p.loc, *maybeConv, p.resultETy, p.resultTy, p.batch,
        p.outputChannels, p.stride, p.pad);

    if (EqualizeRanks(rewriter, op.getLoc(), resultPad, p.bias).failed()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(), resultPad,
                                             p.bias);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeTransposeConv(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  // Preferred strided transpose-conv lowering with quantization:
  //   tosa.transpose_conv2d + tosa.rescale
  //      -> conv2d + (bias handling) + rescale + reshape/transpose/slice
  // We keep rescale close to conv2d so later tensor reordering works on i8/i16
  // instead of wide accumulator types. This reduces memory traffic on reshape &
  // transpose
  patterns.add<TransposeConvRescaleDecompose>(ctx);
  patterns.add<TransposeConvNonStridedConverter>(ctx);
  patterns.add<TransposeConvStridedConverter>(ctx);
}
