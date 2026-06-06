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

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateComplexToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<ConstantOpPattern, CreateOpPattern, ReOpPattern, ImOpPattern,
               ElementwiseBinaryOpPattern<complex::AddOp, spirv::FAddOp>,
               ElementwiseBinaryOpPattern<complex::SubOp, spirv::FSubOp>,
               MulOpPattern, DivOpPattern, AbsOpPattern<spirv::GLSqrtOp>,
               AbsOpPattern<spirv::CLSqrtOp>>(typeConverter, context);
}
