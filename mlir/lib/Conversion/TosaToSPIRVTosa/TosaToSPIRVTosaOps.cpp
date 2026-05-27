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

template <typename TargetOp>
struct ReductionReplace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(op, type, adaptor.getAxis(),
                                          adaptor.getInput());
    return success();
  }
};

template <typename TargetOp>
struct NanModeReductionReplace {
  template <typename SourceOp>
  static LogicalResult replace(SourceOp op, typename SourceOp::Adaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, type, adaptor.getAxis(), getNanMode(adaptor), adaptor.getInput());
    return success();
  }
};

struct ClampReplace {
  static LogicalResult replace(tosa::ClampOp op, tosa::ClampOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaClampOp>(
        op, type, adaptor.getMinVal(), adaptor.getMaxVal(), getNanMode(adaptor),
        adaptor.getInput());
    return success();
  }
};

struct ArithmeticRightShiftReplace {
  static LogicalResult replace(tosa::ArithmeticRightShiftOp op,
                               tosa::ArithmeticRightShiftOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaArithmeticRightShiftOp>(
        op, type, adaptor.getRound(), adaptor.getInput1(), adaptor.getInput2());
    return success();
  }
};

struct MulReplace {
  static LogicalResult replace(tosa::MulOp op, tosa::MulOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaMulOp>(
        op, type, adaptor.getInput1(), adaptor.getInput2(), adaptor.getShift());
    return success();
  }
};

struct TableReplace {
  static LogicalResult replace(tosa::TableOp op, tosa::TableOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaTableOp>(
        op, type, adaptor.getInput1(), adaptor.getTable());
    return success();
  }
};

struct NegateReplace {
  static LogicalResult replace(tosa::NegateOp op, tosa::NegateOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaNegateOp>(
        op, type, adaptor.getInput1(), adaptor.getInput1Zp(),
        adaptor.getOutputZp());
    return success();
  }
};

struct SelectReplace {
  static LogicalResult replace(tosa::SelectOp op, tosa::SelectOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaSelectOp>(
        op, type, adaptor.getInput1(), adaptor.getInput2(),
        adaptor.getInput3());
    return success();
  }
};

struct ReshapeReplace {
  static LogicalResult replace(tosa::ReshapeOp op,
                               tosa::ReshapeOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaReshapeOp>(
        op, type, adaptor.getInput1(), adaptor.getShape());
    return success();
  }
};

struct ReverseReplace {
  static LogicalResult replace(tosa::ReverseOp op,
                               tosa::ReverseOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaReverseOp>(
        op, type, adaptor.getAxis(), adaptor.getInput1());
    return success();
  }
};

struct SliceReplace {
  static LogicalResult replace(tosa::SliceOp op, tosa::SliceOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaSliceOp>(
        op, type, adaptor.getInput1(), adaptor.getStart(), adaptor.getSize());
    return success();
  }
};

struct TileReplace {
  static LogicalResult replace(tosa::TileOp op, tosa::TileOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaTileOp>(
        op, type, adaptor.getInput1(), adaptor.getMultiples());
    return success();
  }
};

struct TransposeReplace {
  static LogicalResult replace(tosa::TransposeOp op,
                               tosa::TransposeOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
    DenseIntElementsAttr perms =
        getI32TensorArmAttr(adaptor.getPerms(), rewriter);
    rewriter.replaceOpWithNewOp<spirv::TosaTransposeOp>(op, type, perms,
                                                        adaptor.getInput1());
    return success();
  }
};

struct GatherReplace {
  static LogicalResult replace(tosa::GatherOp op, tosa::GatherOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaGatherOp>(
        op, type, adaptor.getValues(), adaptor.getIndices());
    return success();
  }
};

struct ScatterReplace {
  static LogicalResult replace(tosa::ScatterOp op,
                               tosa::ScatterOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaScatterOp>(
        op, type, adaptor.getValuesIn(), adaptor.getIndices(),
        adaptor.getInput());
    return success();
  }
};

struct ResizeReplace {
  static LogicalResult replace(tosa::ResizeOp op, tosa::ResizeOpAdaptor adaptor,
                               Type type, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<spirv::TosaResizeOp>(
        op, type, getResizeMode(adaptor), adaptor.getInput(),
        adaptor.getScale(), adaptor.getOffset(), adaptor.getBorder());
    return success();
  }
};

struct ConstShapeReplace {
  static LogicalResult replace(tosa::ConstShapeOp op,
                               tosa::ConstShapeOpAdaptor adaptor, Type type,
                               ConversionPatternRewriter &rewriter) {
    SmallVector<int32_t> values;
    for (const APInt &value : adaptor.getValues().getValues<APInt>())
      values.push_back(value.getSExtValue());

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, type, getI32TensorArmAttr(values, rewriter));
    return success();
  }
};

} // namespace

void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      TosaOpConvert<tosa::ArgMaxOp,
                    NanModeReductionReplace<spirv::TosaArgMaxOp>>,
      TosaOpConvert<tosa::ClampOp, ClampReplace>,
      TosaOpConvert<tosa::ErfOp, UnaryInputReplace<spirv::TosaErfOp>>,
      TosaOpConvert<tosa::SigmoidOp, UnaryInputReplace<spirv::TosaSigmoidOp>>,
      TosaOpConvert<tosa::TanhOp, UnaryInputReplace<spirv::TosaTanhOp>>,
      TosaOpConvert<tosa::AddOp, BinaryElementwiseReplace<spirv::TosaAddOp>>,
      TosaOpConvert<tosa::ArithmeticRightShiftOp, ArithmeticRightShiftReplace>,
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
      TosaOpConvert<tosa::MulOp, MulReplace>,
      TosaOpConvert<tosa::PowOp, BinaryElementwiseReplace<spirv::TosaPowOp>>,
      TosaOpConvert<tosa::SubOp, BinaryElementwiseReplace<spirv::TosaSubOp>>,
      TosaOpConvert<tosa::TableOp, TableReplace>,
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
      TosaOpConvert<tosa::NegateOp, NegateReplace>,
      TosaOpConvert<tosa::ReciprocalOp,
                    UnaryInput1Replace<spirv::TosaReciprocalOp>>,
      TosaOpConvert<tosa::RsqrtOp, UnaryInput1Replace<spirv::TosaRsqrtOp>>,
      TosaOpConvert<tosa::SinOp, UnaryInput1Replace<spirv::TosaSinOp>>,
      TosaOpConvert<tosa::SelectOp, SelectReplace>,
      TosaOpConvert<tosa::EqualOp,
                    BinaryElementwiseReplace<spirv::TosaEqualOp>>,
      TosaOpConvert<tosa::GreaterOp,
                    BinaryElementwiseReplace<spirv::TosaGreaterOp>>,
      TosaOpConvert<tosa::GreaterEqualOp,
                    BinaryElementwiseReplace<spirv::TosaGreaterEqualOp>>,
      TosaOpConvert<tosa::ReduceAllOp,
                    ReductionReplace<spirv::TosaReduceAllOp>>,
      TosaOpConvert<tosa::ReduceAnyOp,
                    ReductionReplace<spirv::TosaReduceAnyOp>>,
      TosaOpConvert<tosa::ReduceMaxOp,
                    NanModeReductionReplace<spirv::TosaReduceMaxOp>>,
      TosaOpConvert<tosa::ReduceMinOp,
                    NanModeReductionReplace<spirv::TosaReduceMinOp>>,
      TosaOpConvert<tosa::ReduceProductOp,
                    ReductionReplace<spirv::TosaReduceProductOp>>,
      TosaOpConvert<tosa::ReduceSumOp,
                    ReductionReplace<spirv::TosaReduceSumOp>>,
      TosaOpConvert<tosa::ReshapeOp, ReshapeReplace>,
      TosaOpConvert<tosa::ReverseOp, ReverseReplace>,
      TosaOpConvert<tosa::SliceOp, SliceReplace>,
      TosaOpConvert<tosa::TileOp, TileReplace>,
      TosaOpConvert<tosa::TransposeOp, TransposeReplace>,
      TosaOpConvert<tosa::GatherOp, GatherReplace>,
      TosaOpConvert<tosa::ScatterOp, ScatterReplace>,
      TosaOpConvert<tosa::ResizeOp, ResizeReplace>,
      TosaOpConvert<tosa::CastOp, UnaryInputReplace<spirv::TosaCastOp>>,
      TosaOpConvert<tosa::ConstShapeOp, ConstShapeReplace>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::tosa
