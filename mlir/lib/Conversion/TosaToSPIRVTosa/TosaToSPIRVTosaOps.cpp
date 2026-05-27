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

template <typename SourceOp, typename Replacer>
struct TosaOpConvert final : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    return Replacer::replace(op, adaptor, type, rewriter);
  }
};

template <typename TargetOp>
struct UnaryInput1Replace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput1());
    return success();
  }
};

template <typename TargetOp>
struct UnaryInputReplace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput());
    return success();
  }
};

template <typename TargetOp>
struct BinaryElementwiseReplace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getInput1(),
                                          adaptor.getInput2());
    return success();
  }
};

template <typename TargetOp>
struct BinaryNanModeElementwiseReplace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, getNanMode(adaptor),
                                          adaptor.getInput1(),
                                          adaptor.getInput2());
    return success();
  }
};

} // namespace

void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      TosaOpConvert<tosa::ErfOp, UnaryInputReplace<spirv::TosaErfOp>>,
      TosaOpConvert<tosa::SigmoidOp, UnaryInputReplace<spirv::TosaSigmoidOp>>,
      TosaOpConvert<tosa::TanhOp, UnaryInputReplace<spirv::TosaTanhOp>>,
      TosaOpConvert<tosa::AddOp, BinaryElementwiseReplace<spirv::TosaAddOp>>,
      TosaOpConvert<tosa::BitwiseAndOp,
                    BinaryElementwiseReplace<spirv::TosaBitwiseAndOp>>,
      TosaOpConvert<tosa::BitwiseOrOp,
                    BinaryElementwiseReplace<spirv::TosaBitwiseOrOp>>,
      TosaOpConvert<tosa::BitwiseXorOp,
                    BinaryElementwiseReplace<spirv::TosaBitwiseXorOp>>,
      TosaOpConvert<tosa::IntDivOp,
                    BinaryElementwiseReplace<spirv::TosaIntDivOp>>,
      TosaOpConvert<tosa::LogicalAndOp,
                    BinaryElementwiseReplace<spirv::TosaLogicalAndOp>>,
      TosaOpConvert<tosa::LogicalLeftShiftOp,
                    BinaryElementwiseReplace<spirv::TosaLogicalLeftShiftOp>>,
      TosaOpConvert<tosa::LogicalRightShiftOp,
                    BinaryElementwiseReplace<spirv::TosaLogicalRightShiftOp>>,
      TosaOpConvert<tosa::LogicalOrOp,
                    BinaryElementwiseReplace<spirv::TosaLogicalOrOp>>,
      TosaOpConvert<tosa::LogicalXorOp,
                    BinaryElementwiseReplace<spirv::TosaLogicalXorOp>>,
      TosaOpConvert<tosa::MaximumOp,
                    BinaryNanModeElementwiseReplace<spirv::TosaMaximumOp>>,
      TosaOpConvert<tosa::MinimumOp,
                    BinaryNanModeElementwiseReplace<spirv::TosaMinimumOp>>,
      TosaOpConvert<tosa::PowOp, BinaryElementwiseReplace<spirv::TosaPowOp>>,
      TosaOpConvert<tosa::SubOp, BinaryElementwiseReplace<spirv::TosaSubOp>>,
      TosaOpConvert<tosa::AbsOp, UnaryInput1Replace<spirv::TosaAbsOp>>,
      TosaOpConvert<tosa::BitwiseNotOp,
                    UnaryInput1Replace<spirv::TosaBitwiseNotOp>>,
      TosaOpConvert<tosa::CeilOp, UnaryInput1Replace<spirv::TosaCeilOp>>,
      TosaOpConvert<tosa::ClzOp, UnaryInput1Replace<spirv::TosaClzOp>>,
      TosaOpConvert<tosa::CosOp, UnaryInput1Replace<spirv::TosaCosOp>>,
      TosaOpConvert<tosa::ExpOp, UnaryInput1Replace<spirv::TosaExpOp>>,
      TosaOpConvert<tosa::FloorOp, UnaryInput1Replace<spirv::TosaFloorOp>>,
      TosaOpConvert<tosa::LogOp, UnaryInput1Replace<spirv::TosaLogOp>>,
      TosaOpConvert<tosa::LogicalNotOp,
                    UnaryInput1Replace<spirv::TosaLogicalNotOp>>,
      TosaOpConvert<tosa::ReciprocalOp,
                    UnaryInput1Replace<spirv::TosaReciprocalOp>>,
      TosaOpConvert<tosa::RsqrtOp, UnaryInput1Replace<spirv::TosaRsqrtOp>>,
      TosaOpConvert<tosa::SinOp, UnaryInput1Replace<spirv::TosaSinOp>>,
      TosaOpConvert<tosa::EqualOp,
                    BinaryElementwiseReplace<spirv::TosaEqualOp>>,
      TosaOpConvert<tosa::GreaterOp,
                    BinaryElementwiseReplace<spirv::TosaGreaterOp>>,
      TosaOpConvert<tosa::GreaterEqualOp,
                    BinaryElementwiseReplace<spirv::TosaGreaterEqualOp>>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::tosa
