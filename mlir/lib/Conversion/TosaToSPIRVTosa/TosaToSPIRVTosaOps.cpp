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
Value getInput1(OpAdaptor adaptor) {
  return adaptor.getInput1();
}

Value getInput1(tosa::ErfOpAdaptor adaptor) { return adaptor.getInput(); }

Value getInput1(tosa::SigmoidOpAdaptor adaptor) { return adaptor.getInput(); }

Value getInput1(tosa::TanhOpAdaptor adaptor) { return adaptor.getInput(); }

template <typename SourceOp, typename TargetOp>
struct UnaryElementwiseOpConvert final : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, getInput1(adaptor));
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryElementwiseOpConvert final : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput1(),
                                          adaptor.getInput2());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryNanModeElementwiseOpConvert final
    : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nanMode =
        static_cast<spirv::TosaExtNaNPropagationModeType>(adaptor.getNanMode());
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, type, nanMode, adaptor.getInput1(), adaptor.getInput2());
    return success();
  }
};

} // namespace

void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      UnaryElementwiseOpConvert<tosa::ErfOp, spirv::TosaErfOp>,
      UnaryElementwiseOpConvert<tosa::SigmoidOp, spirv::TosaSigmoidOp>,
      UnaryElementwiseOpConvert<tosa::TanhOp, spirv::TosaTanhOp>,
      BinaryElementwiseOpConvert<tosa::AddOp, spirv::TosaAddOp>,
      BinaryElementwiseOpConvert<tosa::BitwiseAndOp, spirv::TosaBitwiseAndOp>,
      BinaryElementwiseOpConvert<tosa::BitwiseOrOp, spirv::TosaBitwiseOrOp>,
      BinaryElementwiseOpConvert<tosa::BitwiseXorOp, spirv::TosaBitwiseXorOp>,
      BinaryElementwiseOpConvert<tosa::IntDivOp, spirv::TosaIntDivOp>,
      BinaryElementwiseOpConvert<tosa::LogicalAndOp, spirv::TosaLogicalAndOp>,
      BinaryElementwiseOpConvert<tosa::LogicalLeftShiftOp,
                                 spirv::TosaLogicalLeftShiftOp>,
      BinaryElementwiseOpConvert<tosa::LogicalRightShiftOp,
                                 spirv::TosaLogicalRightShiftOp>,
      BinaryElementwiseOpConvert<tosa::LogicalOrOp, spirv::TosaLogicalOrOp>,
      BinaryElementwiseOpConvert<tosa::LogicalXorOp, spirv::TosaLogicalXorOp>,
      BinaryNanModeElementwiseOpConvert<tosa::MaximumOp, spirv::TosaMaximumOp>,
      BinaryNanModeElementwiseOpConvert<tosa::MinimumOp, spirv::TosaMinimumOp>,
      BinaryElementwiseOpConvert<tosa::PowOp, spirv::TosaPowOp>,
      BinaryElementwiseOpConvert<tosa::SubOp, spirv::TosaSubOp>,
      UnaryElementwiseOpConvert<tosa::AbsOp, spirv::TosaAbsOp>,
      UnaryElementwiseOpConvert<tosa::BitwiseNotOp, spirv::TosaBitwiseNotOp>,
      UnaryElementwiseOpConvert<tosa::CeilOp, spirv::TosaCeilOp>,
      UnaryElementwiseOpConvert<tosa::ClzOp, spirv::TosaClzOp>,
      UnaryElementwiseOpConvert<tosa::CosOp, spirv::TosaCosOp>,
      UnaryElementwiseOpConvert<tosa::ExpOp, spirv::TosaExpOp>,
      UnaryElementwiseOpConvert<tosa::FloorOp, spirv::TosaFloorOp>,
      UnaryElementwiseOpConvert<tosa::LogOp, spirv::TosaLogOp>,
      UnaryElementwiseOpConvert<tosa::LogicalNotOp, spirv::TosaLogicalNotOp>,
      UnaryElementwiseOpConvert<tosa::ReciprocalOp, spirv::TosaReciprocalOp>,
      UnaryElementwiseOpConvert<tosa::RsqrtOp, spirv::TosaRsqrtOp>,
      UnaryElementwiseOpConvert<tosa::SinOp, spirv::TosaSinOp>,
      BinaryElementwiseOpConvert<tosa::EqualOp, spirv::TosaEqualOp>,
      BinaryElementwiseOpConvert<tosa::GreaterOp, spirv::TosaGreaterOp>,
      BinaryElementwiseOpConvert<tosa::GreaterEqualOp,
                                 spirv::TosaGreaterEqualOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::tosa
