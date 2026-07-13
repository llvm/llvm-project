//===- MathToLLVM.cpp - Math to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/FloatingPointMode.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename SourceOp, typename TargetOp>
using ConvertFastMath = arith::AttrConvertFastMathToLLVM<SourceOp, TargetOp>;

template <typename SourceOp, typename TargetOp, bool FailOnUnsupportedFP = true>
using ConvertFMFMathToLLVMPattern =
    VectorConvertToLLVMPattern<SourceOp, TargetOp, ConvertFastMath,
                               FailOnUnsupportedFP>;

/// Lowering pattern that matches only when the source op's constrained
/// floating-point environment presence agrees with `IsConstrained`. A source
/// op is considered constrained when it carries either the deprecated
/// `roundingmode` attribute or the `#arith.fenv` attribute. Mirrors the helper
/// of the same name in `mlir/lib/Conversion/ArithToLLVM/ArithToLLVM.cpp`. This
/// lets us register two patterns for one math op: an unconstrained one that
/// lowers to a regular LLVM op, and a constrained one that lowers to an
/// `llvm.intr.experimental.constrained.*` intrinsic.
template <typename SourceOp, typename TargetOp, bool IsConstrained,
          template <typename, typename> typename AttrConvert =
              AttrConvertPassThrough,
          bool FailOnUnsupportedFP = true>
struct ConstrainedVectorConvertToLLVMPattern
    : public VectorConvertToLLVMPattern<SourceOp, TargetOp, AttrConvert,
                                        FailOnUnsupportedFP> {
  using VectorConvertToLLVMPattern<
      SourceOp, TargetOp, AttrConvert,
      FailOnUnsupportedFP>::VectorConvertToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool opIsConstrained = static_cast<bool>(op.getFenvAttr());
    // Operations that predate the `#arith.fenv` attribute may also carry the
    // deprecated `roundingmode` attribute.
    if constexpr (SourceOp::template hasTrait<
                      arith::ArithRoundingModeInterface::Trait>())
      opIsConstrained =
          opIsConstrained || static_cast<bool>(op.getRoundingModeAttr());
    if (IsConstrained != opIsConstrained)
      return failure();
    return VectorConvertToLLVMPattern<
        SourceOp, TargetOp, AttrConvert,
        FailOnUnsupportedFP>::matchAndRewrite(op, adaptor, rewriter);
  }
};

// Convenience alias for the pattern that lowers a math op carrying no
// floating-point environment constraint to a regular LLVM intrinsic op,
// converting the fastmath flags.
template <typename SourceOp, typename TargetOp>
using ConvertUnconstrainedMathToLLVMPattern =
    ConstrainedVectorConvertToLLVMPattern<SourceOp, TargetOp,
                                          /*IsConstrained=*/false,
                                          ConvertFastMath,
                                          /*FailOnUnsupportedFP=*/true>;

// Convenience alias for the pattern that lowers a math op carrying a
// floating-point environment constraint (the `#arith.fenv` attribute) to the
// matching `llvm.intr.experimental.constrained.*` intrinsic.
template <typename SourceOp, typename TargetOp>
using ConvertConstrainedMathToLLVMPattern =
    ConstrainedVectorConvertToLLVMPattern<
        SourceOp, TargetOp, /*IsConstrained=*/true,
        arith::AttrConverterConstrainedFPToLLVM, /*FailOnUnsupportedFP=*/true>;

using AbsFOpLowering =
    ConvertFMFMathToLLVMPattern<math::AbsFOp, LLVM::FAbsOp,
                                /*FailOnUnsupportedFP=*/true>;
using CeilOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::CeilOp, LLVM::FCeilOp>;
using ConstrainedCeilOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::CeilOp,
                                        LLVM::ConstrainedCeilIntr>;
using CopySignOpLowering =
    ConvertFMFMathToLLVMPattern<math::CopySignOp, LLVM::CopySignOp>;
using CosOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::CosOp, LLVM::CosOp>;
using ConstrainedCosOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::CosOp, LLVM::ConstrainedCosIntr>;
using CoshOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::CoshOp, LLVM::CoshOp>;
using ConstrainedCoshOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::CoshOp,
                                        LLVM::ConstrainedCoshIntr>;
using AcosOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::AcosOp, LLVM::ACosOp>;
using ConstrainedAcosOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::AcosOp,
                                        LLVM::ConstrainedACosIntr>;
using CtPopFOpLowering =
    VectorConvertToLLVMPattern<math::CtPopOp, LLVM::CtPopOp,
                               AttrConvertPassThrough,
                               /*FailOnUnsupportedFP=*/true>;
using Exp2OpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using ConstrainedExp2OpLowering =
    ConvertConstrainedMathToLLVMPattern<math::Exp2Op,
                                        LLVM::ConstrainedExp2Intr>;
using ExpOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using ConstrainedExpOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::ExpOp, LLVM::ConstrainedExpIntr>;
using FloorOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::FloorOp, LLVM::FFloorOp>;
using ConstrainedFloorOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::FloorOp,
                                        LLVM::ConstrainedFloorIntr>;
using FmaOpLowering =
    ConstrainedVectorConvertToLLVMPattern<math::FmaOp, LLVM::FMAOp,
                                          /*IsConstrained=*/false,
                                          ConvertFastMath,
                                          /*FailOnUnsupportedFP=*/true>;
using ConstrainedFmaOpLowering = ConstrainedVectorConvertToLLVMPattern<
    math::FmaOp, LLVM::ConstrainedFMAIntr, /*IsConstrained=*/true,
    arith::AttrConverterConstrainedFPToLLVM, /*FailOnUnsupportedFP=*/true>;
using Log10OpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using ConstrainedLog10OpLowering =
    ConvertConstrainedMathToLLVMPattern<math::Log10Op,
                                        LLVM::ConstrainedLog10Intr>;
using Log2OpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using ConstrainedLog2OpLowering =
    ConvertConstrainedMathToLLVMPattern<math::Log2Op,
                                        LLVM::ConstrainedLog2Intr>;
using LogOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::LogOp, LLVM::LogOp>;
using ConstrainedLogOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::LogOp, LLVM::ConstrainedLogIntr>;
using PowFOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using ConstrainedPowFOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::PowFOp, LLVM::ConstrainedPowIntr>;
using FPowIOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::FPowIOp, LLVM::PowIOp>;
using RoundEvenOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::RoundEvenOp, LLVM::RoundEvenOp>;
using ConstrainedRoundEvenOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::RoundEvenOp,
                                        LLVM::ConstrainedRoundEvenIntr>;
using RoundOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::RoundOp, LLVM::RoundOp>;
using ConstrainedRoundOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::RoundOp,
                                        LLVM::ConstrainedRoundIntr>;
using SinOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::SinOp, LLVM::SinOp>;
using ConstrainedSinOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::SinOp, LLVM::ConstrainedSinIntr>;
using SinhOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::SinhOp, LLVM::SinhOp>;
using ConstrainedSinhOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::SinhOp,
                                        LLVM::ConstrainedSinhIntr>;
using ASinOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::AsinOp, LLVM::ASinOp>;
using ConstrainedASinOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::AsinOp,
                                        LLVM::ConstrainedASinIntr>;
using SqrtOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;
using ConstrainedSqrtOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::SqrtOp,
                                        LLVM::ConstrainedSqrtIntr>;
using FTruncOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::TruncOp, LLVM::FTruncOp>;
using ConstrainedFTruncOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::TruncOp,
                                        LLVM::ConstrainedTruncIntr>;
using TanOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::TanOp, LLVM::TanOp>;
using ConstrainedTanOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::TanOp, LLVM::ConstrainedTanIntr>;
using TanhOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::TanhOp, LLVM::TanhOp>;
using ConstrainedTanhOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::TanhOp,
                                        LLVM::ConstrainedTanhIntr>;
using ATanOpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::AtanOp, LLVM::ATanOp>;
using ConstrainedATanOpLowering =
    ConvertConstrainedMathToLLVMPattern<math::AtanOp,
                                        LLVM::ConstrainedATanIntr>;
using ATan2OpLowering =
    ConvertUnconstrainedMathToLLVMPattern<math::Atan2Op, LLVM::ATan2Op>;
using ConstrainedATan2OpLowering =
    ConvertConstrainedMathToLLVMPattern<math::Atan2Op,
                                        LLVM::ConstrainedATan2Intr>;

// A `math.fpowi` carrying a floating-point environment (`#arith.fenv`) lowers
// to the constrained `powi` intrinsic.
struct ConstrainedFPowIOpLowering
    : public ConvertOpToLLVMPattern<math::FPowIOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::FPowIOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::FPowIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getFenvAttr())
      return rewriter.notifyMatchFailure(
          op, "unconstrained fpowi is handled by a separate pattern");

    // `llvm.intr.experimental.constrained.powi` requires a scalar operand with
    // an `i32` exponent.
    Type resultType = op.getResult().getType();
    if (!isa<FloatType>(resultType))
      return rewriter.notifyMatchFailure(
          op, "constrained fpowi only supports scalar operands");
    if (!op.getRhs().getType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "constrained fpowi requires an i32 exponent");

    Type llvmResultType = getTypeConverter()->convertType(resultType);
    if (!llvmResultType)
      return failure();

    arith::AttrConverterConstrainedFPToLLVM<math::FPowIOp,
                                            LLVM::ConstrainedPowIIntr>
        attrConvert(op);
    rewriter.replaceOpWithNewOp<LLVM::ConstrainedPowIIntr>(
        op, llvmResultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        attrConvert.getAttrs());
    return success();
  }
};

// A `CtLz/CtTz/absi(a)` is converted into `CtLz/CtTz/absi(a, false)`.
// TODO: Result and operand types match for `absi` as opposed to `ct*z`, so it
// may be better to separate the patterns.
template <typename MathOp, typename LLVMOp>
struct IntOpWithFlagLowering
    : public ConvertOpToLLVMPattern<MathOp, /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      MathOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;
  using Super = IntOpWithFlagLowering<MathOp, LLVMOp>;

  LogicalResult
  matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType = adaptor.getOperand().getType();
    auto llvmOperandType = typeConverter.convertType(operandType);
    if (!llvmOperandType)
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto llvmResultType = typeConverter.convertType(resultType);
    if (!llvmResultType)
      return failure();

    if (!isa<LLVM::LLVMArrayType>(llvmOperandType)) {
      rewriter.replaceOpWithNewOp<LLVMOp>(op, llvmResultType,
                                          adaptor.getOperand(), false);
      return success();
    }

    if (!isa<VectorType>(resultType))
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), typeConverter,
        [&](Type llvm1DVectorTy, ValueRange operands) {
          return LLVMOp::create(rewriter, loc, llvm1DVectorTy, operands[0],
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

// A `sincos` is converted into `llvm.intr.sincos` followed by extractvalue ops.
struct SincosOpLowering
    : public ConvertOpToLLVMPattern<math::SincosOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::SincosOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::SincosOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const LLVMTypeConverter &typeConverter = *this->getTypeConverter();
    mlir::Location loc = op.getLoc();
    mlir::Type operandType = adaptor.getOperand().getType();
    mlir::Type llvmOperandType = typeConverter.convertType(operandType);
    mlir::Type sinType = typeConverter.convertType(op.getSin().getType());
    mlir::Type cosType = typeConverter.convertType(op.getCos().getType());
    if (!llvmOperandType || !sinType || !cosType)
      return failure();

    ConvertFastMath<math::SincosOp, LLVM::SincosOp> attrs(op);

    auto structType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmOperandType, llvmOperandType});

    auto sincosOp = LLVM::SincosOp::create(
        rewriter, loc, structType, adaptor.getOperand(), attrs.getAttrs());

    auto sinValue = LLVM::ExtractValueOp::create(rewriter, loc, sincosOp, 0);
    auto cosValue = LLVM::ExtractValueOp::create(rewriter, loc, sincosOp, 1);

    rewriter.replaceOp(op, {sinValue, cosValue});
    return success();
  }
};

// A `expm1` is converted into `exp - 1`.
struct ExpM1OpLowering
    : public ConvertOpToLLVMPattern<math::ExpM1Op,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::ExpM1Op, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType = adaptor.getOperand().getType();
    auto llvmOperandType = typeConverter.convertType(operandType);
    if (!llvmOperandType)
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(
        typeConverter.convertType(getElementTypeOrSelf(resultType)));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::ExpM1Op, LLVM::ExpOp> expAttrs(op);
    ConvertFastMath<math::ExpM1Op, LLVM::FSubOp> subAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(llvmOperandType)) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(llvmOperandType)) {
        one = LLVM::ConstantOp::create(
            rewriter, loc, llvmOperandType,
            SplatElementsAttr::get(cast<ShapedType>(llvmOperandType),
                                   floatOne));
      } else {
        one =
            LLVM::ConstantOp::create(rewriter, loc, llvmOperandType, floatOne);
      }
      auto exp = LLVM::ExpOp::create(rewriter, loc, adaptor.getOperand(),
                                     expAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::FSubOp>(
          op, llvmOperandType, ValueRange{exp, one}, subAttrs.getAttrs());
      return success();
    }

    if (!isa<VectorType>(resultType))
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), typeConverter,
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto numElements = LLVM::getVectorNumElements(llvm1DVectorTy);
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get({numElements.getKnownMinValue()}, floatType,
                                    {numElements.isScalable()}),
              floatOne);
          auto one = LLVM::ConstantOp::create(rewriter, loc, llvm1DVectorTy,
                                              splatAttr);
          auto exp = LLVM::ExpOp::create(rewriter, loc, llvm1DVectorTy,
                                         operands[0], expAttrs.getAttrs());
          return LLVM::FSubOp::create(rewriter, loc, llvm1DVectorTy,
                                      ValueRange{exp, one},
                                      subAttrs.getAttrs());
        },
        rewriter);
  }
};

// A `log1p` is converted into `log(1 + ...)`.
struct Log1pOpLowering
    : public ConvertOpToLLVMPattern<math::Log1pOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::Log1pOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType = adaptor.getOperand().getType();
    auto llvmOperandType = typeConverter.convertType(operandType);
    if (!llvmOperandType)
      return rewriter.notifyMatchFailure(op, "unsupported operand type");

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(
        typeConverter.convertType(getElementTypeOrSelf(resultType)));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::Log1pOp, LLVM::FAddOp> addAttrs(op);
    ConvertFastMath<math::Log1pOp, LLVM::LogOp> logAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(llvmOperandType)) {
      LLVM::ConstantOp one =
          isa<VectorType>(llvmOperandType)
              ? LLVM::ConstantOp::create(
                    rewriter, loc, llvmOperandType,
                    SplatElementsAttr::get(cast<ShapedType>(llvmOperandType),
                                           floatOne))
              : LLVM::ConstantOp::create(rewriter, loc, llvmOperandType,
                                         floatOne);

      auto add = LLVM::FAddOp::create(rewriter, loc, llvmOperandType,
                                      ValueRange{one, adaptor.getOperand()},
                                      addAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::LogOp>(
          op, llvmOperandType, ValueRange{add}, logAttrs.getAttrs());
      return success();
    }

    if (!isa<VectorType>(resultType))
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), typeConverter,
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto numElements = LLVM::getVectorNumElements(llvm1DVectorTy);
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get({numElements.getKnownMinValue()}, floatType,
                                    {numElements.isScalable()}),
              floatOne);
          auto one = LLVM::ConstantOp::create(rewriter, loc, llvm1DVectorTy,
                                              splatAttr);
          auto add = LLVM::FAddOp::create(rewriter, loc, llvm1DVectorTy,
                                          ValueRange{one, operands[0]},
                                          addAttrs.getAttrs());
          return LLVM::LogOp::create(rewriter, loc, llvm1DVectorTy,
                                     ValueRange{add}, logAttrs.getAttrs());
        },
        rewriter);
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering
    : public ConvertOpToLLVMPattern<math::RsqrtOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::RsqrtOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType = adaptor.getOperand().getType();
    auto llvmOperandType = typeConverter.convertType(operandType);
    if (!llvmOperandType)
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = cast<FloatType>(
        typeConverter.convertType(getElementTypeOrSelf(resultType)));
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
    ConvertFastMath<math::RsqrtOp, LLVM::SqrtOp> sqrtAttrs(op);
    ConvertFastMath<math::RsqrtOp, LLVM::FDivOp> divAttrs(op);

    if (!isa<LLVM::LLVMArrayType>(llvmOperandType)) {
      LLVM::ConstantOp one;
      if (isa<VectorType>(llvmOperandType)) {
        one = LLVM::ConstantOp::create(
            rewriter, loc, llvmOperandType,
            SplatElementsAttr::get(cast<ShapedType>(llvmOperandType),
                                   floatOne));
      } else {
        one =
            LLVM::ConstantOp::create(rewriter, loc, llvmOperandType, floatOne);
      }
      auto sqrt = LLVM::SqrtOp::create(rewriter, loc, adaptor.getOperand(),
                                       sqrtAttrs.getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(
          op, llvmOperandType, ValueRange{one, sqrt}, divAttrs.getAttrs());
      return success();
    }

    if (!isa<VectorType>(resultType))
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), typeConverter,
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto numElements = LLVM::getVectorNumElements(llvm1DVectorTy);
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get({numElements.getKnownMinValue()}, floatType,
                                    {numElements.isScalable()}),
              floatOne);
          auto one = LLVM::ConstantOp::create(rewriter, loc, llvm1DVectorTy,
                                              splatAttr);
          auto sqrt = LLVM::SqrtOp::create(rewriter, loc, llvm1DVectorTy,
                                           operands[0], sqrtAttrs.getAttrs());
          return LLVM::FDivOp::create(rewriter, loc, llvm1DVectorTy,
                                      ValueRange{one, sqrt},
                                      divAttrs.getAttrs());
        },
        rewriter);
  }
};

struct IsNaNOpLowering
    : public ConvertOpToLLVMPattern<math::IsNaNOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::IsNaNOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::IsNaNOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType =
        typeConverter.convertType(adaptor.getOperand().getType());
    auto resultType = typeConverter.convertType(op.getResult().getType());
    if (!operandType || !resultType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::IsFPClass>(
        op, resultType, adaptor.getOperand(), llvm::fcNan);
    return success();
  }
};

struct IsFiniteOpLowering
    : public ConvertOpToLLVMPattern<math::IsFiniteOp,
                                    /*FailOnUnsupportedFP=*/true> {
  using ConvertOpToLLVMPattern<
      math::IsFiniteOp, /*FailOnUnsupportedFP=*/true>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::IsFiniteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *this->getTypeConverter();
    auto operandType =
        typeConverter.convertType(adaptor.getOperand().getType());
    auto resultType = typeConverter.convertType(op.getResult().getType());
    if (!operandType || !resultType)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::IsFPClass>(
        op, resultType, adaptor.getOperand(), llvm::fcFinite);
    return success();
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

void mlir::populateMathToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool approximateLog1p, PatternBenefit benefit) {
  if (approximateLog1p)
    patterns.add<Log1pOpLowering>(converter, benefit);
  // clang-format off
  patterns.add<
    IsNaNOpLowering,
    IsFiniteOpLowering,
    AbsFOpLowering,
    AbsIOpLowering,
    CeilOpLowering,
    ConstrainedCeilOpLowering,
    CopySignOpLowering,
    CosOpLowering,
    ConstrainedCosOpLowering,
    CoshOpLowering,
    ConstrainedCoshOpLowering,
    AcosOpLowering,
    ConstrainedAcosOpLowering,
    CountLeadingZerosOpLowering,
    CountTrailingZerosOpLowering,
    CtPopFOpLowering,
    Exp2OpLowering,
    ConstrainedExp2OpLowering,
    ExpM1OpLowering,
    ExpOpLowering,
    ConstrainedExpOpLowering,
    FPowIOpLowering,
    ConstrainedFPowIOpLowering,
    FloorOpLowering,
    ConstrainedFloorOpLowering,
    FmaOpLowering,
    ConstrainedFmaOpLowering,
    Log10OpLowering,
    ConstrainedLog10OpLowering,
    Log2OpLowering,
    ConstrainedLog2OpLowering,
    LogOpLowering,
    ConstrainedLogOpLowering,
    PowFOpLowering,
    ConstrainedPowFOpLowering,
    RoundEvenOpLowering,
    ConstrainedRoundEvenOpLowering,
    RoundOpLowering,
    ConstrainedRoundOpLowering,
    RsqrtOpLowering,
    SincosOpLowering,
    SinOpLowering,
    ConstrainedSinOpLowering,
    SinhOpLowering,
    ConstrainedSinhOpLowering,
    ASinOpLowering,
    ConstrainedASinOpLowering,
    SqrtOpLowering,
    ConstrainedSqrtOpLowering,
    FTruncOpLowering,
    ConstrainedFTruncOpLowering,
    TanOpLowering,
    ConstrainedTanOpLowering,
    TanhOpLowering,
    ConstrainedTanhOpLowering,
    ATanOpLowering,
    ConstrainedATanOpLowering,
    ATan2OpLowering,
    ConstrainedATan2OpLowering
  >(converter, benefit);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert Math to LLVM.
struct MathToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  MathToLLVMDialectInterface(Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertMathToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, math::MathDialect *dialect) {
    dialect->addInterfaces<MathToLLVMDialectInterface>();
  });
}
