//===- MathToLLVM.cpp - Math to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename SourceOp, typename TargetOp>
using ConvertFastMath = arith::AttrConvertFastMathToLLVM<SourceOp, TargetOp>;

template <typename SourceOp, typename TargetOp>
using ConvertFMFMathToLLVMPattern =
    VectorConvertToLLVMPattern<SourceOp, TargetOp, ConvertFastMath>;

using AbsFOpLowering = ConvertFMFMathToLLVMPattern<math::AbsFOp, LLVM::FAbsOp>;
using CeilOpLowering = ConvertFMFMathToLLVMPattern<math::CeilOp, LLVM::FCeilOp>;
using CopySignOpLowering =
    ConvertFMFMathToLLVMPattern<math::CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = ConvertFMFMathToLLVMPattern<math::CosOp, LLVM::CosOp>;
using CtPopFOpLowering =
    VectorConvertToLLVMPattern<math::CtPopOp, LLVM::CtPopOp>;
using Exp2OpLowering = ConvertFMFMathToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using ExpOpLowering = ConvertFMFMathToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using FloorOpLowering =
    ConvertFMFMathToLLVMPattern<math::FloorOp, LLVM::FFloorOp>;
using FmaOpLowering = ConvertFMFMathToLLVMPattern<math::FmaOp, LLVM::FMAOp>;
using Log10OpLowering =
    ConvertFMFMathToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering = ConvertFMFMathToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = ConvertFMFMathToLLVMPattern<math::LogOp, LLVM::LogOp>;
using PowFOpLowering = ConvertFMFMathToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using FPowIOpLowering =
    ConvertFMFMathToLLVMPattern<math::FPowIOp, LLVM::PowIOp>;
using RoundEvenOpLowering =
    ConvertFMFMathToLLVMPattern<math::RoundEvenOp, LLVM::RoundEvenOp>;
using RoundOpLowering =
    ConvertFMFMathToLLVMPattern<math::RoundOp, LLVM::RoundOp>;
using SinOpLowering = ConvertFMFMathToLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering = ConvertFMFMathToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;
using FTruncOpLowering =
    ConvertFMFMathToLLVMPattern<math::TruncOp, LLVM::FTruncOp>;

// A `CtLz/CtTz/absi(a)` is converted into `CtLz/CtTz/absi(a, false)`.
template <typename MathOp, typename LLVMOp>
struct IntOpWithFlagLowering : public ConvertOpToLLVMPattern<MathOp> {
  using ConvertOpToLLVMPattern<MathOp>::ConvertOpToLLVMPattern;
  using Super = IntOpWithFlagLowering<MathOp, LLVMOp>;

  LogicalResult
  matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getOperand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();

    if (!isa<LLVM::LLVMArrayType>(operandType)) {
      rewriter.replaceOpWithNewOp<LLVMOp>(op, resultType, adaptor.getOperand(),
                                          false);
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType)
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *this->getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          return rewriter.create<LLVMOp>(loc, llvm1DVectorTy, operands[0],
                                         false);
        },
        rewriter);
  }
};

using CountLeadingZerosOpLowering =
    IntOpWithFlagLowering<math::CountLeadingZerosOp, LLVM::CountLeadingZerosOp>;
using CountTrailingZerosOpLowering =
    IntOpWithFlagLowering<math::CountTrailingZerosOp,
                          LLVM::CountTrailingZerosOp>;
using AbsIOpLowering = IntOpWithFlagLowering<math::AbsIOp, LLVM::AbsOp>;

// A `expm1` is converted into `exp - 1`.
struct ExpM1OpLowering : public ConvertOpToLLVMPattern<math::ExpM1Op> {
  using ConvertOpToLLVMPattern<math::ExpM1Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getOperand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(getElementTypeOrSelf(resultType));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::ExpM1Op, LLVM::ExpOp> expAttrs(op);
    ConvertFastMath<math::ExpM1Op, LLVM::FSubOp> subAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(operandType)) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(cast<ShapedType>(resultType), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto exp = rewriter.create<LLVM::ExpOp>(loc, adaptor.getOperand(),
                                              expAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::FSubOp>(
          op, operandType, ValueRange{exp, one}, subAttrs.getAttrs());
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto exp = rewriter.create<LLVM::ExpOp>(
              loc, llvm1DVectorTy, operands[0], expAttrs.getAttrs());
          return rewriter.create<LLVM::FSubOp>(
              loc, llvm1DVectorTy, ValueRange{exp, one}, subAttrs.getAttrs());
        },
        rewriter);
  }
};

// A `log1p` is converted into `log(1 + ...)`.
struct Log1pOpLowering : public ConvertOpToLLVMPattern<math::Log1pOp> {
  using ConvertOpToLLVMPattern<math::Log1pOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getOperand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return rewriter.notifyMatchFailure(op, "unsupported operand type");

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(getElementTypeOrSelf(resultType));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::Log1pOp, LLVM::FAddOp> addAttrs(op);
    ConvertFastMath<math::Log1pOp, LLVM::LogOp> logAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(operandType)) {
      LLVM::ConstantOp one =
          LLVM::isCompatibleVectorType(operandType)
              ? rewriter.create<LLVM::ConstantOp>(
                    loc, operandType,
                    SplatElementsAttr::get(cast<ShapedType>(resultType),
                                           floatOne))
              : rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);

      auto add = rewriter.create<LLVM::FAddOp>(
          loc, operandType, ValueRange{one, adaptor.getOperand()},
          addAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::LogOp>(op, operandType, ValueRange{add},
                                               logAttrs.getAttrs());
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto add = rewriter.create<LLVM::FAddOp>(loc, llvm1DVectorTy,
                                                   ValueRange{one, operands[0]},
                                                   addAttrs.getAttrs());
          return rewriter.create<LLVM::LogOp>(
              loc, llvm1DVectorTy, ValueRange{add}, logAttrs.getAttrs());
        },
        rewriter);
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering : public ConvertOpToLLVMPattern<math::RsqrtOp> {
  using ConvertOpToLLVMPattern<math::RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getOperand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(getElementTypeOrSelf(resultType));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::RsqrtOp, LLVM::SqrtOp> sqrtAttrs(op);
    ConvertFastMath<math::RsqrtOp, LLVM::FDivOp> divAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(operandType)) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(cast<ShapedType>(resultType), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto sqrt = rewriter.create<LLVM::SqrtOp>(loc, adaptor.getOperand(),
                                                sqrtAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(
          op, operandType, ValueRange{one, sqrt}, divAttrs.getAttrs());
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType)
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto sqrt = rewriter.create<LLVM::SqrtOp>(
              loc, llvm1DVectorTy, operands[0], sqrtAttrs.getAttrs());
          return rewriter.create<LLVM::FDivOp>(
              loc, llvm1DVectorTy, ValueRange{one, sqrt}, divAttrs.getAttrs());
        },
        rewriter);
  }
};

struct ConvertMathToLLVMPass
    : public impl::ConvertMathToLLVMPassBase<ConvertMathToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateMathToLLVMConversionPatterns(converter, patterns, approximateLog1p);
    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateMathToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns,
                                                bool approximateLog1p) {
  if (approximateLog1p)
    patterns.add<Log1pOpLowering>(converter);
  // clang-format off
  patterns.add<
    AbsFOpLowering,
    AbsIOpLowering,
    CeilOpLowering,
    CopySignOpLowering,
    CosOpLowering,
    CountLeadingZerosOpLowering,
    CountTrailingZerosOpLowering,
    CtPopFOpLowering,
    Exp2OpLowering,
    ExpM1OpLowering,
    ExpOpLowering,
    FPowIOpLowering,
    FloorOpLowering,
    FmaOpLowering,
    Log10OpLowering,
    Log2OpLowering,
    LogOpLowering,
    PowFOpLowering,
    RoundEvenOpLowering,
    RoundOpLowering,
    RsqrtOpLowering,
    SinOpLowering,
    SqrtOpLowering,
    FTruncOpLowering
  >(converter);
  // clang-format on
}
