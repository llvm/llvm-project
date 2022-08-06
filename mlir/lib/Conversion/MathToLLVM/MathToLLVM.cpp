//===- MathToLLVM.cpp - Math to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
using AbsFOpLowering = VectorConvertToLLVMPattern<math::AbsFOp, LLVM::FAbsOp>;
using CeilOpLowering = VectorConvertToLLVMPattern<math::CeilOp, LLVM::FCeilOp>;
using CopySignOpLowering =
    VectorConvertToLLVMPattern<math::CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = VectorConvertToLLVMPattern<math::CosOp, LLVM::CosOp>;
using CtPopFOpLowering =
    VectorConvertToLLVMPattern<math::CtPopOp, LLVM::CtPopOp>;
using Exp2OpLowering = VectorConvertToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using ExpOpLowering = VectorConvertToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using FloorOpLowering =
    VectorConvertToLLVMPattern<math::FloorOp, LLVM::FFloorOp>;
using FmaOpLowering = VectorConvertToLLVMPattern<math::FmaOp, LLVM::FMAOp>;
using Log10OpLowering =
    VectorConvertToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = VectorConvertToLLVMPattern<math::LogOp, LLVM::LogOp>;
using PowFOpLowering = VectorConvertToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using RoundOpLowering =
    VectorConvertToLLVMPattern<math::RoundOp, LLVM::RoundOp>;
using SinOpLowering = VectorConvertToLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;

// A `CtLz/CtTz(a)` is converted into `CtLz/CtTz(a, false)`.
template <typename MathOp, typename LLVMOp>
struct CountOpLowering : public ConvertOpToLLVMPattern<MathOp> {
  using ConvertOpToLLVMPattern<MathOp>::ConvertOpToLLVMPattern;
  using Super = CountOpLowering<MathOp, LLVMOp>;

  LogicalResult
  matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getOperand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto boolType = rewriter.getIntegerType(1);
    auto boolZero = rewriter.getIntegerAttr(boolType, 0);

    if (!operandType.template isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp zero =
          rewriter.create<LLVM::ConstantOp>(loc, boolType, boolZero);
      rewriter.replaceOpWithNewOp<LLVMOp>(op, resultType, adaptor.getOperand(),
                                          zero);
      return success();
    }

    auto vectorType = resultType.template dyn_cast<VectorType>();
    if (!vectorType)
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *this->getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          LLVM::ConstantOp zero =
              rewriter.create<LLVM::ConstantOp>(loc, boolType, boolZero);
          return rewriter.create<LLVMOp>(loc, llvm1DVectorTy, operands[0],
                                         zero);
        },
        rewriter);
  }
};

using CountLeadingZerosOpLowering =
    CountOpLowering<math::CountLeadingZerosOp, LLVM::CountLeadingZerosOp>;
using CountTrailingZerosOpLowering =
    CountOpLowering<math::CountTrailingZerosOp, LLVM::CountTrailingZerosOp>;

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
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto exp = rewriter.create<LLVM::ExpOp>(loc, adaptor.getOperand());
      rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, operandType, exp, one);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
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
          auto exp =
              rewriter.create<LLVM::ExpOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FSubOp>(loc, llvm1DVectorTy, exp, one);
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
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one =
          LLVM::isCompatibleVectorType(operandType)
              ? rewriter.create<LLVM::ConstantOp>(
                    loc, operandType,
                    SplatElementsAttr::get(resultType.cast<ShapedType>(),
                                           floatOne))
              : rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);

      auto add = rewriter.create<LLVM::FAddOp>(loc, operandType, one,
                                               adaptor.getOperand());
      rewriter.replaceOpWithNewOp<LLVM::LogOp>(op, operandType, add);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
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
          auto add = rewriter.create<LLVM::FAddOp>(loc, llvm1DVectorTy, one,
                                                   operands[0]);
          return rewriter.create<LLVM::LogOp>(loc, llvm1DVectorTy, add);
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
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto sqrt = rewriter.create<LLVM::SqrtOp>(loc, adaptor.getOperand());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, operandType, one, sqrt);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
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
          auto sqrt =
              rewriter.create<LLVM::SqrtOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FDivOp>(loc, llvm1DVectorTy, one, sqrt);
        },
        rewriter);
  }
};

struct ConvertMathToLLVMPass
    : public ConvertMathToLLVMBase<ConvertMathToLLVMPass> {
  ConvertMathToLLVMPass() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateMathToLLVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateMathToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    AbsFOpLowering,
    CeilOpLowering,
    CopySignOpLowering,
    CosOpLowering,
    CountLeadingZerosOpLowering,
    CountTrailingZerosOpLowering,
    CtPopFOpLowering,
    Exp2OpLowering,
    ExpM1OpLowering,
    ExpOpLowering,
    FloorOpLowering,
    FmaOpLowering,
    Log10OpLowering,
    Log1pOpLowering,
    Log2OpLowering,
    LogOpLowering,
    PowFOpLowering,
    RoundOpLowering,
    RsqrtOpLowering,
    SinOpLowering,
    SqrtOpLowering
  >(converter);
  // clang-format on
}

std::unique_ptr<Pass> mlir::createConvertMathToLLVMPass() {
  return std::make_unique<ConvertMathToLLVMPass>();
}
