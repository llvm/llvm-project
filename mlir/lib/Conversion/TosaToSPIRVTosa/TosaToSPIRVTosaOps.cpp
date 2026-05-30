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

template <typename OpAdaptor>
spirv::TosaExtNaNPropagationModeType getNanMode(OpAdaptor adaptor) {
  return static_cast<spirv::TosaExtNaNPropagationModeType>(
      adaptor.getNanMode());
}

template <typename OpAdaptor>
spirv::TosaExtResizeModeType getResizeMode(OpAdaptor adaptor) {
  return static_cast<spirv::TosaExtResizeModeType>(adaptor.getMode());
}

DenseIntElementsAttr getI32TensorArmAttr(ArrayRef<int32_t> values,
                                         ConversionPatternRewriter &rewriter) {
  return DenseIntElementsAttr::get(
      spirv::TensorArmType::get(static_cast<int64_t>(values.size()),
                                IntegerType::get(rewriter.getContext(), 32)),
      values);
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

LogicalResult replaceConstShape(tosa::ConstShapeOp op,
                                tosa::ConstShapeOpAdaptor adaptor, Type type,
                                ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> values;
  for (const APInt &value : adaptor.getValues().getValues<APInt>())
    values.push_back(value.getSExtValue());

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
      op, type, getI32TensorArmAttr(values, rewriter));
  return success();
}

} // namespace

void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      TosaOpConvert<tosa::ArgMaxOp, replaceNanModeReduction<
                                        tosa::ArgMaxOp, spirv::TosaArgMaxOp>>,
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
      TosaOpConvert<tosa::ConstShapeOp, replaceConstShape>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::tosa
