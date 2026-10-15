//===- ComplexToSPIRV.cpp - Complex to SPIR-V Patterns --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "complex-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {
/// Creates a scalar floating-point constant of the given value.
Value createFPConstant(OpBuilder &builder, Location loc, Type type,
                       double value) {
  return spirv::ConstantOp::create(
      builder, loc, type, builder.getFloatAttr(cast<FloatType>(type), value));
}

struct ConstantOpPattern final : OpConversionPattern<complex::ConstantOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto spirvType =
        getTypeConverter()->convertType<ShapedType>(constOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(constOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        constOp, spirvType,
        DenseElementsAttr::get(spirvType, constOp.getValue().getValue()));
    return success();
  }
};

struct CreateOpPattern final : OpConversionPattern<complex::CreateOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::CreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(createOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(createOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        createOp, spirvType, adaptor.getOperands());
    return success();
  }
};

struct ReOpPattern final : OpConversionPattern<complex::ReOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ReOp reOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(reOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(reOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        reOp, adaptor.getComplex(), llvm::ArrayRef(0));
    return success();
  }
};

struct ImOpPattern final : OpConversionPattern<complex::ImOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ImOp imOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(imOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(imOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        imOp, adaptor.getComplex(), llvm::ArrayRef(1));
    return success();
  }
};

template <typename ComplexOp, typename SPIRVOp>
struct ElementwiseBinaryOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value lhsRe = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value lhsIm = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value rhsRe = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value rhsIm = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value resultRe = SPIRVOp::create(rewriter, loc, lhsRe, rhsRe);
    Value resultIm = SPIRVOp::create(rewriter, loc, lhsIm, rhsIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename ComplexOp, typename SPIRVCompareOp, typename SPIRVCombinerOp>
struct ComparisonOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value lhsRe = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value lhsIm = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value rhsRe = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value rhsIm = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value cmpRe = SPIRVCompareOp::create(rewriter, loc, lhsRe, rhsRe);
    Value cmpIm = SPIRVCompareOp::create(rewriter, loc, lhsIm, rhsIm);

    rewriter.replaceOpWithNewOp<SPIRVCombinerOp>(op, spirvType, cmpRe, cmpIm);
    return success();
  }
};

struct MulOpPattern final : OpConversionPattern<complex::MulOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value a = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value b = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value c = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value d = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value ac = spirv::FMulOp::create(rewriter, loc, a, c);
    Value bd = spirv::FMulOp::create(rewriter, loc, b, d);
    Value ad = spirv::FMulOp::create(rewriter, loc, a, d);
    Value bc = spirv::FMulOp::create(rewriter, loc, b, c);
    Value resultRe = spirv::FSubOp::create(rewriter, loc, ac, bd);
    Value resultIm = spirv::FAddOp::create(rewriter, loc, ad, bc);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename SqrtOp>
struct AbsOpPattern final : OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value reSq = spirv::FMulOp::create(rewriter, loc, re, re);
    Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
    Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);

    rewriter.replaceOpWithNewOp<SqrtOp>(op, sum);
    return success();
  }
};

template <typename ComplexOp, bool NegateReal>
struct NegationOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value resultRe =
        NegateReal ? spirv::FNegateOp::create(rewriter, loc, re) : re;
    Value resultIm = spirv::FNegateOp::create(rewriter, loc, im);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

struct DivOpPattern final : OpConversionPattern<complex::DivOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value a = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value b = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value c = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value d = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value ac = spirv::FMulOp::create(rewriter, loc, a, c);
    Value bd = spirv::FMulOp::create(rewriter, loc, b, d);
    Value bc = spirv::FMulOp::create(rewriter, loc, b, c);
    Value ad = spirv::FMulOp::create(rewriter, loc, a, d);
    Value cc = spirv::FMulOp::create(rewriter, loc, c, c);
    Value dd = spirv::FMulOp::create(rewriter, loc, d, d);
    Value denom = spirv::FAddOp::create(rewriter, loc, cc, dd);
    Value numRe = spirv::FAddOp::create(rewriter, loc, ac, bd);
    Value numIm = spirv::FSubOp::create(rewriter, loc, bc, ad);
    Value resultRe = spirv::FDivOp::create(rewriter, loc, numRe, denom);
    Value resultIm = spirv::FDivOp::create(rewriter, loc, numIm, denom);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename Atan2Op>
struct AngleOpPattern final : OpConversionPattern<complex::AngleOp> {
  using OpConversionPattern<complex::AngleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AngleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    rewriter.replaceOpWithNewOp<Atan2Op>(op, im, re);
    return success();
  }
};

/// Computes complex.sign as z / abs(z), selecting a zero result when z is
/// zero to avoid a division by zero. abs(z) is computed as
/// max * sqrt(1 + (min / max)^2) to avoid overflow in squaring.
template <typename SqrtOp, typename FAbsOp, typename FMaxOp, typename FMinOp>
struct SignOpPattern final : OpConversionPattern<complex::SignOp> {
  using OpConversionPattern<complex::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value absRe = FAbsOp::create(rewriter, loc, re);
    Value absIm = FAbsOp::create(rewriter, loc, im);
    Value max = FMaxOp::create(rewriter, loc, absRe, absIm);
    Value min = FMinOp::create(rewriter, loc, absRe, absIm);
    Value ratio = spirv::FDivOp::create(rewriter, loc, min, max);
    Value ratioSq = spirv::FMulOp::create(rewriter, loc, ratio, ratio);
    Value one = createFPConstant(rewriter, loc, re.getType(), 1.0);
    Value ratioSqPlusOne = spirv::FAddOp::create(rewriter, loc, ratioSq, one);
    Value sqrt = SqrtOp::create(rewriter, loc, ratioSqPlusOne);
    Value abs = spirv::FMulOp::create(rewriter, loc, max, sqrt);

    Value signRe = spirv::FDivOp::create(rewriter, loc, re, abs);
    Value signIm = spirv::FDivOp::create(rewriter, loc, im, abs);

    Value zero = createFPConstant(rewriter, loc, re.getType(), 0.0);
    Value reIsZero = spirv::FOrdEqualOp::create(rewriter, loc, re, zero);
    Value imIsZero = spirv::FOrdEqualOp::create(rewriter, loc, im, zero);
    Value isZero =
        spirv::LogicalAndOp::create(rewriter, loc, reIsZero, imIsZero);

    // Select per component: spirv.Select with a scalar condition and a
    // composite result requires SPIR-V 1.4, so operate on scalars here and
    // assemble the result afterwards.
    Value resultRe = spirv::SelectOp::create(rewriter, loc, isZero, re, signRe);
    Value resultIm = spirv::SelectOp::create(rewriter, loc, isZero, im, signIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

enum class SinCos { Sin, Cos };

/// Lowers complex.sin/cos following the identities
/// sin(x + iy) = sin(x) * (0.5*exp(y) + 0.5/exp(y))
///              + i * cos(x) * (0.5*exp(y) - 0.5/exp(y))
/// cos(x + iy) = cos(x) * (0.5/exp(y) + 0.5*exp(y))
///              + i * sin(x) * (0.5/exp(y) - 0.5*exp(y))
template <SinCos Kind, typename SinOp, typename CosOp, typename ExpOp,
          typename ComplexOp = std::conditional_t<
              Kind == SinCos::Sin, complex::SinOp, complex::CosOp>>
struct TrigonometricOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value half = createFPConstant(rewriter, loc, re.getType(), 0.5);
    Value exp = ExpOp::create(rewriter, loc, im);
    Value scaledExp = spirv::FMulOp::create(rewriter, loc, half, exp);
    Value reciprocalExp = spirv::FDivOp::create(rewriter, loc, half, exp);
    Value sin = SinOp::create(rewriter, loc, re);
    Value cos = CosOp::create(rewriter, loc, re);

    bool isSin = Kind == SinCos::Sin;
    Value lhs = isSin ? scaledExp : reciprocalExp;
    Value rhs = isSin ? reciprocalExp : scaledExp;
    Value sum = spirv::FAddOp::create(rewriter, loc, lhs, rhs);
    Value resultRe =
        spirv::FMulOp::create(rewriter, loc, sum, isSin ? sin : cos);
    Value diff = spirv::FSubOp::create(rewriter, loc, lhs, rhs);
    Value resultIm =
        spirv::FMulOp::create(rewriter, loc, diff, isSin ? cos : sin);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateComplexToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns
      .add<ConstantOpPattern, CreateOpPattern, ReOpPattern, ImOpPattern,
           ElementwiseBinaryOpPattern<complex::AddOp, spirv::FAddOp>,
           ElementwiseBinaryOpPattern<complex::SubOp, spirv::FSubOp>,
           ComparisonOpPattern<complex::EqualOp, spirv::FOrdEqualOp,
                               spirv::LogicalAndOp>,
           ComparisonOpPattern<complex::NotEqualOp, spirv::FUnordNotEqualOp,
                               spirv::LogicalOrOp>,
           MulOpPattern, DivOpPattern,
           NegationOpPattern<complex::NegOp, /*NegateReal=*/true>,
           NegationOpPattern<complex::ConjOp, /*NegateReal=*/false>,
           AbsOpPattern<spirv::GLSqrtOp>, AbsOpPattern<spirv::CLSqrtOp>,
           AngleOpPattern<spirv::GLAtan2Op>, AngleOpPattern<spirv::CLAtan2Op>,
           SignOpPattern<spirv::GLSqrtOp, spirv::GLFAbsOp, spirv::GLFMaxOp,
                         spirv::GLFMinOp>,
           SignOpPattern<spirv::CLSqrtOp, spirv::CLFAbsOp, spirv::CLFMaxOp,
                         spirv::CLFMinOp>,
           TrigonometricOpPattern<SinCos::Sin, spirv::GLSinOp, spirv::GLCosOp,
                                  spirv::GLExpOp>,
           TrigonometricOpPattern<SinCos::Sin, spirv::CLSinOp, spirv::CLCosOp,
                                  spirv::CLExpOp>,
           TrigonometricOpPattern<SinCos::Cos, spirv::GLSinOp, spirv::GLCosOp,
                                  spirv::GLExpOp>,
           TrigonometricOpPattern<SinCos::Cos, spirv::CLSinOp, spirv::CLCosOp,
                                  spirv::CLExpOp>>(typeConverter, context);
}
