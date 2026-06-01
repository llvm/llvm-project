//===- TosaToSPIRVTosaOps.cpp - TOSA to SPIR-V Graph/TOSA patterns --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert TOSA IR to SPIR-V Graph/TOSA.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "tosa-to-spirv-tosa-ops-pattern"

namespace mlir::tosa {
namespace {

constexpr unsigned maxConcatOpInputs = 64;

template <typename OpAdaptor>
spirv::TosaExtNaNPropagationModeType getNanMode(OpAdaptor adaptor) {
  return static_cast<spirv::TosaExtNaNPropagationModeType>(
      adaptor.getNanMode());
}

template <typename OpAdaptor>
spirv::TosaExtResizeModeType getResizeMode(OpAdaptor adaptor) {
  return static_cast<spirv::TosaExtResizeModeType>(adaptor.getMode());
}

template <typename OpAdaptor>
spirv::TosaExtRoundingModeType getRoundingMode(OpAdaptor adaptor) {
  return static_cast<spirv::TosaExtRoundingModeType>(adaptor.getRoundingMode());
}

spirv::TosaExtAccType getAccType(Type accType) {
  if (accType.isInteger(32))
    return spirv::TosaExtAccType::INT32;
  else if (accType.isF16())
    return spirv::TosaExtAccType::FP16;
  else if (accType.isF32())
    return spirv::TosaExtAccType::FP32;
  else if (accType.isInteger(48))
    return spirv::TosaExtAccType::INT48;
  llvm_unreachable("unknown accumulator type");
}

DenseIntElementsAttr getI32TensorArmAttr(ArrayRef<int32_t> values,
                                         ConversionPatternRewriter &rewriter) {
  return DenseIntElementsAttr::get(
      spirv::TensorArmType::get(static_cast<int64_t>(values.size()),
                                IntegerType::get(rewriter.getContext(), 32)),
      values);
}

// TOSA stores many integer array attributes as i64 in MLIR, while the
// SPIR-V TOSA extended instruction set models the same attributes as i32.
DenseIntElementsAttr getI32TensorArmAttr(ArrayRef<int64_t> values,
                                         ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> i32Values(values.begin(), values.end());
  return getI32TensorArmAttr(i32Values, rewriter);
}

FailureOr<DenseElementsAttr>
convertDenseElementsAttr(DenseElementsAttr values, ShapedType convertedType) {
  Type convertedElementType = convertedType.getElementType();
  if (values.getElementType() == convertedElementType)
    return values.reshape(convertedType);

  // Constant attributes still have the source TOSA element type. Rebuild them
  // for the converted SPIR-V tensor type, including integer type-converter
  // changes such as index to i32, i4 to i8, and i48 to i64.
  auto integerType = dyn_cast<IntegerType>(convertedElementType);
  if (!integerType)
    return failure();

  // The SPIR-V ARM tensor type does not represent scalar shape constants
  // directly, so model tensor<0xindex> as a one-element i32 tensor.
  if (values.empty() && values.getElementType().isIndex())
    return DenseIntElementsAttr::get(convertedType, {1});

  DenseElementsAttr convertedValues =
      values.mapValues(integerType, [&](const APInt &value) {
        return value.sextOrTrunc(integerType.getWidth());
      });
  return convertedValues.reshape(convertedType);
}

// Split a large concat into smaller concat operations so the generated SPIR-V
// instructions stay below the binary operand count limit.
LogicalResult splitConcat(tosa::ConcatOp op, Type resultType, int32_t axis,
                          ValueRange inputs,
                          ConversionPatternRewriter &rewriter) {
  auto resultTensorType = dyn_cast<spirv::TensorArmType>(resultType);
  if (!resultTensorType)
    return rewriter.notifyMatchFailure(op, "expected tensor result type");
  if (!resultTensorType.hasRank())
    return rewriter.notifyMatchFailure(op,
                                       "expected ranked tensor result type");

  SmallVector<Value> concatInputs;
  SmallVector<int64_t> concatShape(resultTensorType.getShape());
  concatShape[axis] = 0;

  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputType = dyn_cast<spirv::TensorArmType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected tensor input type");
    if (!inputType.hasRank())
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensor input type");

    int64_t inputAxisDim = inputType.getShape()[axis];
    if (ShapedType::isDynamic(inputAxisDim) ||
        ShapedType::isDynamic(concatShape[axis]))
      concatShape[axis] = ShapedType::kDynamic;
    else
      concatShape[axis] += inputAxisDim;

    concatInputs.push_back(input);
    if (concatInputs.size() != maxConcatOpInputs || index == inputs.size() - 1)
      continue;

    Type concatType = spirv::TensorArmType::get(
        concatShape, resultTensorType.getElementType());
    auto concat = spirv::TosaConcatOp::create(rewriter, op.getLoc(), concatType,
                                              axis, concatInputs);
    concatInputs.clear();
    concatInputs.push_back(concat.getOutput());
  }

  rewriter.replaceOpWithNewOp<spirv::TosaConcatOp>(op, resultType, axis,
                                                   concatInputs);
  return success();
}

template <typename SourceOp, auto Replace>
struct TosaOpConvert final : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    return Replace(op, adaptor, type, rewriter);
  }
};

template <typename SourceOp, auto Replace>
struct TosaMultiResultOpConvert final : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      types)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    return Replace(op, adaptor, types, rewriter);
  }
};

template <typename SourceOp, typename TargetOp>
LogicalResult replaceUnaryInput1(SourceOp op,
                                 typename SourceOp::Adaptor adaptor, Type type,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput1());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult replaceUnaryInput(SourceOp op, typename SourceOp::Adaptor adaptor,
                                Type type,
                                ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult
replaceBinaryElementwise(SourceOp op, typename SourceOp::Adaptor adaptor,
                         Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput1(),
                                        adaptor.getInput2());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult
replaceBinaryNanModeElementwise(SourceOp op, typename SourceOp::Adaptor adaptor,
                                Type type,
                                ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(
      op, type, getNanMode(adaptor), adaptor.getInput1(), adaptor.getInput2());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult replaceReduction(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getAxis(),
                                        adaptor.getInput());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult
replaceNanModeReduction(SourceOp op, typename SourceOp::Adaptor adaptor,
                        Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(
      op, type, adaptor.getAxis(), getNanMode(adaptor), adaptor.getInput());
  return success();
}

LogicalResult replaceAvgPool2d(tosa::AvgPool2dOp op,
                               tosa::AvgPool2dOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaAvgPool2DOp>(
      op, type, getI32TensorArmAttr(adaptor.getKernel(), rewriter),
      getI32TensorArmAttr(adaptor.getStride(), rewriter),
      getI32TensorArmAttr(adaptor.getPad(), rewriter),
      getAccType(adaptor.getAccType()), adaptor.getInput(),
      adaptor.getInputZp(), adaptor.getOutputZp());
  return success();
}

template <typename SourceOp, typename TargetOp>
LogicalResult replaceConvolution(SourceOp op,
                                 typename SourceOp::Adaptor adaptor, Type type,
                                 ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TargetOp>(
      op, type, getI32TensorArmAttr(adaptor.getPad(), rewriter),
      getI32TensorArmAttr(adaptor.getStride(), rewriter),
      getI32TensorArmAttr(adaptor.getDilation(), rewriter),
      getAccType(adaptor.getAccType()), adaptor.getLocalBound(),
      adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
      adaptor.getInputZp(), adaptor.getWeightZp());
  return success();
}

LogicalResult replaceFFT2d(tosa::FFT2dOp op, tosa::FFT2dOpAdaptor adaptor,
                           ArrayRef<Type> types,
                           ConversionPatternRewriter &rewriter) {
  auto structType = spirv::StructType::get(types);
  auto result = spirv::TosaFFT2DOp::create(
      rewriter, op.getLoc(), structType, adaptor.getInverse(),
      adaptor.getLocalBound(), adaptor.getInputReal(), adaptor.getInputImag());
  auto outputReal =
      spirv::CompositeExtractOp::create(rewriter, op.getLoc(), result, {0});
  auto outputImag =
      spirv::CompositeExtractOp::create(rewriter, op.getLoc(), result, {1});
  rewriter.replaceOp(op, {outputReal, outputImag});
  return success();
}

LogicalResult replaceMatMul(tosa::MatMulOp op, tosa::MatMulOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaMatMulOp>(
      op, type, adaptor.getA(), adaptor.getB(), adaptor.getAZp(),
      adaptor.getBZp());
  return success();
}

LogicalResult replaceMaxPool2d(tosa::MaxPool2dOp op,
                               tosa::MaxPool2dOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaMaxPool2DOp>(
      op, type, getI32TensorArmAttr(adaptor.getKernel(), rewriter),
      getI32TensorArmAttr(adaptor.getStride(), rewriter),
      getI32TensorArmAttr(adaptor.getPad(), rewriter), getNanMode(adaptor),
      adaptor.getInput());
  return success();
}

LogicalResult replaceRFFT2d(tosa::RFFT2dOp op, tosa::RFFT2dOpAdaptor adaptor,
                            ArrayRef<Type> types,
                            ConversionPatternRewriter &rewriter) {
  auto structType = spirv::StructType::get(types);
  auto result = spirv::TosaRFFT2DOp::create(rewriter, op.getLoc(), structType,
                                            adaptor.getLocalBound(),
                                            adaptor.getInputReal());
  auto outputReal =
      spirv::CompositeExtractOp::create(rewriter, op.getLoc(), result, {0});
  auto outputImag =
      spirv::CompositeExtractOp::create(rewriter, op.getLoc(), result, {1});
  rewriter.replaceOp(op, {outputReal, outputImag});
  return success();
}

LogicalResult replaceTransposeConv2d(tosa::TransposeConv2DOp op,
                                     tosa::TransposeConv2DOpAdaptor adaptor,
                                     Type type,
                                     ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaTransposeConv2DOp>(
      op, type, getI32TensorArmAttr(adaptor.getOutPad(), rewriter),
      getI32TensorArmAttr(adaptor.getStride(), rewriter),
      getAccType(adaptor.getAccType()), adaptor.getLocalBound(),
      adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
      adaptor.getInputZp(), adaptor.getWeightZp());
  return success();
}

LogicalResult replaceClamp(tosa::ClampOp op, tosa::ClampOpAdaptor adaptor,
                           Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaClampOp>(
      op, type, adaptor.getMinVal(), adaptor.getMaxVal(), getNanMode(adaptor),
      adaptor.getInput());
  return success();
}

LogicalResult
replaceArithmeticRightShift(tosa::ArithmeticRightShiftOp op,
                            tosa::ArithmeticRightShiftOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaArithmeticRightShiftOp>(
      op, type, adaptor.getRound(), adaptor.getInput1(), adaptor.getInput2());
  return success();
}

LogicalResult replaceMul(tosa::MulOp op, tosa::MulOpAdaptor adaptor, Type type,
                         ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaMulOp>(
      op, type, adaptor.getInput1(), adaptor.getInput2(), adaptor.getShift());
  return success();
}

LogicalResult replaceTable(tosa::TableOp op, tosa::TableOpAdaptor adaptor,
                           Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaTableOp>(op, type, adaptor.getInput1(),
                                                  adaptor.getTable());
  return success();
}

LogicalResult replaceNegate(tosa::NegateOp op, tosa::NegateOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaNegateOp>(
      op, type, adaptor.getInput1(), adaptor.getInput1Zp(),
      adaptor.getOutputZp());
  return success();
}

LogicalResult replaceSelect(tosa::SelectOp op, tosa::SelectOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaSelectOp>(
      op, type, adaptor.getInput1(), adaptor.getInput2(), adaptor.getInput3());
  return success();
}

LogicalResult replaceConcat(tosa::ConcatOp op, tosa::ConcatOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  // Large TOSA concats can produce SPIR-V instructions with too many
  // operands and fail validation. Split them into conservative 64-input
  // chunks to keep the generated SPIR-V valid.
  if (adaptor.getInput1().size() > maxConcatOpInputs)
    return splitConcat(op, type, adaptor.getAxis(), adaptor.getInput1(),
                       rewriter);

  rewriter.replaceOpWithNewOp<spirv::TosaConcatOp>(op, type, adaptor.getAxis(),
                                                   adaptor.getInput1());
  return success();
}

LogicalResult replacePad(tosa::PadOp op, tosa::PadOpAdaptor adaptor, Type type,
                         ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaPadOp>(op, type, adaptor.getInput1(),
                                                adaptor.getPadding(),
                                                adaptor.getPadConst());
  return success();
}

LogicalResult replaceReshape(tosa::ReshapeOp op, tosa::ReshapeOpAdaptor adaptor,
                             Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaReshapeOp>(
      op, type, adaptor.getInput1(), adaptor.getShape());
  return success();
}

LogicalResult replaceReverse(tosa::ReverseOp op, tosa::ReverseOpAdaptor adaptor,
                             Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaReverseOp>(op, type, adaptor.getAxis(),
                                                    adaptor.getInput1());
  return success();
}

LogicalResult replaceSlice(tosa::SliceOp op, tosa::SliceOpAdaptor adaptor,
                           Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaSliceOp>(
      op, type, adaptor.getInput1(), adaptor.getStart(), adaptor.getSize());
  return success();
}

LogicalResult replaceTile(tosa::TileOp op, tosa::TileOpAdaptor adaptor,
                          Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaTileOp>(op, type, adaptor.getInput1(),
                                                 adaptor.getMultiples());
  return success();
}

LogicalResult replaceTranspose(tosa::TransposeOp op,
                               tosa::TransposeOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
  DenseIntElementsAttr perms =
      getI32TensorArmAttr(adaptor.getPerms(), rewriter);
  rewriter.replaceOpWithNewOp<spirv::TosaTransposeOp>(op, type, perms,
                                                      adaptor.getInput1());
  return success();
}

LogicalResult replaceGather(tosa::GatherOp op, tosa::GatherOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaGatherOp>(
      op, type, adaptor.getValues(), adaptor.getIndices());
  return success();
}

LogicalResult replaceScatter(tosa::ScatterOp op, tosa::ScatterOpAdaptor adaptor,
                             Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaScatterOp>(
      op, type, adaptor.getValuesIn(), adaptor.getIndices(),
      adaptor.getInput());
  return success();
}

LogicalResult replaceResize(tosa::ResizeOp op, tosa::ResizeOpAdaptor adaptor,
                            Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaResizeOp>(
      op, type, getResizeMode(adaptor), adaptor.getInput(), adaptor.getScale(),
      adaptor.getOffset(), adaptor.getBorder());
  return success();
}

LogicalResult replaceRescale(tosa::RescaleOp op, tosa::RescaleOpAdaptor adaptor,
                             Type type, ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<spirv::TosaRescaleOp>(
      op, type, adaptor.getScale32(), getRoundingMode(adaptor),
      adaptor.getPerChannel(), adaptor.getInputUnsigned(),
      adaptor.getOutputUnsigned(), adaptor.getInput(), adaptor.getMultiplier(),
      adaptor.getShift(), adaptor.getInputZp(), adaptor.getOutputZp());
  return success();
}

template <typename SourceOp>
LogicalResult replaceConstant(SourceOp op, typename SourceOp::Adaptor adaptor,
                              Type type, ConversionPatternRewriter &rewriter) {
  auto convertedType = dyn_cast<ShapedType>(type);
  auto values = dyn_cast<DenseElementsAttr>(adaptor.getValues());
  if (!convertedType || !values)
    return failure();

  FailureOr<DenseElementsAttr> convertedValues =
      convertDenseElementsAttr(values, convertedType);
  if (failed(convertedValues))
    return failure();

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(op, type, *convertedValues);
  return success();
}

LogicalResult replaceIdentity(tosa::IdentityOp op,
                              tosa::IdentityOpAdaptor adaptor, Type type,
                              ConversionPatternRewriter &rewriter) {
  rewriter.replaceOp(op, adaptor.getInput1());
  return success();
}

} // namespace

void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      TosaOpConvert<tosa::ArgMaxOp, replaceNanModeReduction<
                                        tosa::ArgMaxOp, spirv::TosaArgMaxOp>>,
      TosaOpConvert<tosa::AvgPool2dOp, replaceAvgPool2d>,
      TosaOpConvert<tosa::Conv2DOp,
                    replaceConvolution<tosa::Conv2DOp, spirv::TosaConv2DOp>>,
      TosaOpConvert<tosa::Conv3DOp,
                    replaceConvolution<tosa::Conv3DOp, spirv::TosaConv3DOp>>,
      TosaOpConvert<tosa::DepthwiseConv2DOp,
                    replaceConvolution<tosa::DepthwiseConv2DOp,
                                       spirv::TosaDepthwiseConv2DOp>>,
      TosaMultiResultOpConvert<tosa::FFT2dOp, replaceFFT2d>,
      TosaOpConvert<tosa::MatMulOp, replaceMatMul>,
      TosaOpConvert<tosa::MaxPool2dOp, replaceMaxPool2d>,
      TosaMultiResultOpConvert<tosa::RFFT2dOp, replaceRFFT2d>,
      TosaOpConvert<tosa::TransposeConv2DOp, replaceTransposeConv2d>,
      TosaOpConvert<tosa::ClampOp, replaceClamp>,
      TosaOpConvert<tosa::ErfOp,
                    replaceUnaryInput<tosa::ErfOp, spirv::TosaErfOp>>,
      TosaOpConvert<tosa::SigmoidOp,
                    replaceUnaryInput<tosa::SigmoidOp, spirv::TosaSigmoidOp>>,
      TosaOpConvert<tosa::TanhOp,
                    replaceUnaryInput<tosa::TanhOp, spirv::TosaTanhOp>>,
      TosaOpConvert<tosa::AddOp,
                    replaceBinaryElementwise<tosa::AddOp, spirv::TosaAddOp>>,
      TosaOpConvert<tosa::ArithmeticRightShiftOp, replaceArithmeticRightShift>,
      TosaOpConvert<tosa::BitwiseAndOp,
                    replaceBinaryElementwise<tosa::BitwiseAndOp,
                                             spirv::TosaBitwiseAndOp>>,
      TosaOpConvert<
          tosa::BitwiseOrOp,
          replaceBinaryElementwise<tosa::BitwiseOrOp, spirv::TosaBitwiseOrOp>>,
      TosaOpConvert<tosa::BitwiseXorOp,
                    replaceBinaryElementwise<tosa::BitwiseXorOp,
                                             spirv::TosaBitwiseXorOp>>,
      TosaOpConvert<tosa::IntDivOp, replaceBinaryElementwise<
                                        tosa::IntDivOp, spirv::TosaIntDivOp>>,
      TosaOpConvert<tosa::LogicalAndOp,
                    replaceBinaryElementwise<tosa::LogicalAndOp,
                                             spirv::TosaLogicalAndOp>>,
      TosaOpConvert<tosa::LogicalLeftShiftOp,
                    replaceBinaryElementwise<tosa::LogicalLeftShiftOp,
                                             spirv::TosaLogicalLeftShiftOp>>,
      TosaOpConvert<tosa::LogicalRightShiftOp,
                    replaceBinaryElementwise<tosa::LogicalRightShiftOp,
                                             spirv::TosaLogicalRightShiftOp>>,
      TosaOpConvert<
          tosa::LogicalOrOp,
          replaceBinaryElementwise<tosa::LogicalOrOp, spirv::TosaLogicalOrOp>>,
      TosaOpConvert<tosa::LogicalXorOp,
                    replaceBinaryElementwise<tosa::LogicalXorOp,
                                             spirv::TosaLogicalXorOp>>,
      TosaOpConvert<tosa::MaximumOp,
                    replaceBinaryNanModeElementwise<tosa::MaximumOp,
                                                    spirv::TosaMaximumOp>>,
      TosaOpConvert<tosa::MinimumOp,
                    replaceBinaryNanModeElementwise<tosa::MinimumOp,
                                                    spirv::TosaMinimumOp>>,
      TosaOpConvert<tosa::MulOp, replaceMul>,
      TosaOpConvert<tosa::PowOp,
                    replaceBinaryElementwise<tosa::PowOp, spirv::TosaPowOp>>,
      TosaOpConvert<tosa::SubOp,
                    replaceBinaryElementwise<tosa::SubOp, spirv::TosaSubOp>>,
      TosaOpConvert<tosa::TableOp, replaceTable>,
      TosaOpConvert<tosa::AbsOp,
                    replaceUnaryInput1<tosa::AbsOp, spirv::TosaAbsOp>>,
      TosaOpConvert<
          tosa::BitwiseNotOp,
          replaceUnaryInput1<tosa::BitwiseNotOp, spirv::TosaBitwiseNotOp>>,
      TosaOpConvert<tosa::CeilOp,
                    replaceUnaryInput1<tosa::CeilOp, spirv::TosaCeilOp>>,
      TosaOpConvert<tosa::ClzOp,
                    replaceUnaryInput1<tosa::ClzOp, spirv::TosaClzOp>>,
      TosaOpConvert<tosa::CosOp,
                    replaceUnaryInput1<tosa::CosOp, spirv::TosaCosOp>>,
      TosaOpConvert<tosa::ExpOp,
                    replaceUnaryInput1<tosa::ExpOp, spirv::TosaExpOp>>,
      TosaOpConvert<tosa::FloorOp,
                    replaceUnaryInput1<tosa::FloorOp, spirv::TosaFloorOp>>,
      TosaOpConvert<tosa::LogOp,
                    replaceUnaryInput1<tosa::LogOp, spirv::TosaLogOp>>,
      TosaOpConvert<
          tosa::LogicalNotOp,
          replaceUnaryInput1<tosa::LogicalNotOp, spirv::TosaLogicalNotOp>>,
      TosaOpConvert<tosa::NegateOp, replaceNegate>,
      TosaOpConvert<
          tosa::ReciprocalOp,
          replaceUnaryInput1<tosa::ReciprocalOp, spirv::TosaReciprocalOp>>,
      TosaOpConvert<tosa::RsqrtOp,
                    replaceUnaryInput1<tosa::RsqrtOp, spirv::TosaRsqrtOp>>,
      TosaOpConvert<tosa::SinOp,
                    replaceUnaryInput1<tosa::SinOp, spirv::TosaSinOp>>,
      TosaOpConvert<tosa::SelectOp, replaceSelect>,
      TosaOpConvert<tosa::EqualOp, replaceBinaryElementwise<
                                       tosa::EqualOp, spirv::TosaEqualOp>>,
      TosaOpConvert<
          tosa::GreaterOp,
          replaceBinaryElementwise<tosa::GreaterOp, spirv::TosaGreaterOp>>,
      TosaOpConvert<tosa::GreaterEqualOp,
                    replaceBinaryElementwise<tosa::GreaterEqualOp,
                                             spirv::TosaGreaterEqualOp>>,
      TosaOpConvert<
          tosa::ReduceAllOp,
          replaceReduction<tosa::ReduceAllOp, spirv::TosaReduceAllOp>>,
      TosaOpConvert<
          tosa::ReduceAnyOp,
          replaceReduction<tosa::ReduceAnyOp, spirv::TosaReduceAnyOp>>,
      TosaOpConvert<
          tosa::ReduceMaxOp,
          replaceNanModeReduction<tosa::ReduceMaxOp, spirv::TosaReduceMaxOp>>,
      TosaOpConvert<
          tosa::ReduceMinOp,
          replaceNanModeReduction<tosa::ReduceMinOp, spirv::TosaReduceMinOp>>,
      TosaOpConvert<
          tosa::ReduceProductOp,
          replaceReduction<tosa::ReduceProductOp, spirv::TosaReduceProductOp>>,
      TosaOpConvert<
          tosa::ReduceSumOp,
          replaceReduction<tosa::ReduceSumOp, spirv::TosaReduceSumOp>>,
      TosaOpConvert<tosa::ConcatOp, replaceConcat>,
      TosaOpConvert<tosa::PadOp, replacePad>,
      TosaOpConvert<tosa::ReshapeOp, replaceReshape>,
      TosaOpConvert<tosa::ReverseOp, replaceReverse>,
      TosaOpConvert<tosa::SliceOp, replaceSlice>,
      TosaOpConvert<tosa::TileOp, replaceTile>,
      TosaOpConvert<tosa::TransposeOp, replaceTranspose>,
      TosaOpConvert<tosa::GatherOp, replaceGather>,
      TosaOpConvert<tosa::ScatterOp, replaceScatter>,
      TosaOpConvert<tosa::ResizeOp, replaceResize>,
      TosaOpConvert<tosa::CastOp,
                    replaceUnaryInput<tosa::CastOp, spirv::TosaCastOp>>,
      TosaOpConvert<tosa::RescaleOp, replaceRescale>,
      TosaOpConvert<tosa::ConstOp, replaceConstant<tosa::ConstOp>>,
      TosaOpConvert<tosa::ConstShapeOp, replaceConstant<tosa::ConstShapeOp>>,
      TosaOpConvert<tosa::IdentityOp, replaceIdentity>>(typeConverter,
                                                        patterns.getContext());
}

} // namespace mlir::tosa
