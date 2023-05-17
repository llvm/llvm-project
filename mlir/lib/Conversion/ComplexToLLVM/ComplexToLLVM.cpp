//===- ComplexToLLVM.cpp - conversion from Complex to LLVM dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// ComplexStructBuilder implementation.
//===----------------------------------------------------------------------===//

static constexpr unsigned kRealPosInComplexNumberStruct = 0;
static constexpr unsigned kImaginaryPosInComplexNumberStruct = 1;

ComplexStructBuilder ComplexStructBuilder::undef(OpBuilder &builder,
                                                 Location loc, Type type) {
  Value val = builder.create<LLVM::UndefOp>(loc, type);
  return ComplexStructBuilder(val);
}

void ComplexStructBuilder::setReal(OpBuilder &builder, Location loc,
                                   Value real) {
  setPtr(builder, loc, kRealPosInComplexNumberStruct, real);
}

Value ComplexStructBuilder::real(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kRealPosInComplexNumberStruct);
}

void ComplexStructBuilder::setImaginary(OpBuilder &builder, Location loc,
                                        Value imaginary) {
  setPtr(builder, loc, kImaginaryPosInComplexNumberStruct, imaginary);
}

Value ComplexStructBuilder::imaginary(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kImaginaryPosInComplexNumberStruct);
}

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//

namespace {

struct AbsOpConversion : public ConvertOpToLLVMPattern<complex::AbsOp> {
  using ConvertOpToLLVMPattern<complex::AbsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    ComplexStructBuilder complexStruct(adaptor.getComplex());
    Value real = complexStruct.real(rewriter, op.getLoc());
    Value imag = complexStruct.imaginary(rewriter, op.getLoc());

    auto fmf = LLVM::FastmathFlagsAttr::get(op.getContext(), {});
    Value sqNorm = rewriter.create<LLVM::FAddOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, real, real, fmf),
        rewriter.create<LLVM::FMulOp>(loc, imag, imag, fmf), fmf);

    rewriter.replaceOpWithNewOp<LLVM::SqrtOp>(op, sqNorm);
    return success();
  }
};

struct ConstantOpLowering : public ConvertOpToLLVMPattern<complex::ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return LLVM::detail::oneToOneRewrite(
        op, LLVM::ConstantOp::getOperationName(), adaptor.getOperands(),
        op->getAttrs(), *getTypeConverter(), rewriter);
  }
};

struct CreateOpConversion : public ConvertOpToLLVMPattern<complex::CreateOp> {
  using ConvertOpToLLVMPattern<complex::CreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::CreateOp complexOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Pack real and imaginary part in a complex number struct.
    auto loc = complexOp.getLoc();
    auto structType = typeConverter->convertType(complexOp.getType());
    auto complexStruct = ComplexStructBuilder::undef(rewriter, loc, structType);
    complexStruct.setReal(rewriter, loc, adaptor.getReal());
    complexStruct.setImaginary(rewriter, loc, adaptor.getImaginary());

    rewriter.replaceOp(complexOp, {complexStruct});
    return success();
  }
};

struct ReOpConversion : public ConvertOpToLLVMPattern<complex::ReOp> {
  using ConvertOpToLLVMPattern<complex::ReOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::ReOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract real part from the complex number struct.
    ComplexStructBuilder complexStruct(adaptor.getComplex());
    Value real = complexStruct.real(rewriter, op.getLoc());
    rewriter.replaceOp(op, real);

    return success();
  }
};

struct ImOpConversion : public ConvertOpToLLVMPattern<complex::ImOp> {
  using ConvertOpToLLVMPattern<complex::ImOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::ImOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract imaginary part from the complex number struct.
    ComplexStructBuilder complexStruct(adaptor.getComplex());
    Value imaginary = complexStruct.imaginary(rewriter, op.getLoc());
    rewriter.replaceOp(op, imaginary);

    return success();
  }
};

struct BinaryComplexOperands {
  std::complex<Value> lhs;
  std::complex<Value> rhs;
};

template <typename OpTy>
BinaryComplexOperands
unpackBinaryComplexOperands(OpTy op, typename OpTy::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();

  // Extract real and imaginary values from operands.
  BinaryComplexOperands unpacked;
  ComplexStructBuilder lhs(adaptor.getLhs());
  unpacked.lhs.real(lhs.real(rewriter, loc));
  unpacked.lhs.imag(lhs.imaginary(rewriter, loc));
  ComplexStructBuilder rhs(adaptor.getRhs());
  unpacked.rhs.real(rhs.real(rewriter, loc));
  unpacked.rhs.imag(rhs.imaginary(rewriter, loc));

  return unpacked;
}

struct AddOpConversion : public ConvertOpToLLVMPattern<complex::AddOp> {
  using ConvertOpToLLVMPattern<complex::AddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::AddOp>(op, adaptor, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    auto fmf = LLVM::FastmathFlagsAttr::get(op.getContext(), {});
    Value real =
        rewriter.create<LLVM::FAddOp>(loc, arg.lhs.real(), arg.rhs.real(), fmf);
    Value imag =
        rewriter.create<LLVM::FAddOp>(loc, arg.lhs.imag(), arg.rhs.imag(), fmf);
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct DivOpConversion : public ConvertOpToLLVMPattern<complex::DivOp> {
  using ConvertOpToLLVMPattern<complex::DivOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::DivOp>(op, adaptor, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    auto fmf = LLVM::FastmathFlagsAttr::get(op.getContext(), {});
    Value rhsRe = arg.rhs.real();
    Value rhsIm = arg.rhs.imag();
    Value lhsRe = arg.lhs.real();
    Value lhsIm = arg.lhs.imag();

    Value rhsSqNorm = rewriter.create<LLVM::FAddOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, rhsRe, rhsRe, fmf),
        rewriter.create<LLVM::FMulOp>(loc, rhsIm, rhsIm, fmf), fmf);

    Value resultReal = rewriter.create<LLVM::FAddOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsRe, fmf),
        rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsIm, fmf), fmf);

    Value resultImag = rewriter.create<LLVM::FSubOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsRe, fmf),
        rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsIm, fmf), fmf);

    result.setReal(
        rewriter, loc,
        rewriter.create<LLVM::FDivOp>(loc, resultReal, rhsSqNorm, fmf));
    result.setImaginary(
        rewriter, loc,
        rewriter.create<LLVM::FDivOp>(loc, resultImag, rhsSqNorm, fmf));

    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct MulOpConversion : public ConvertOpToLLVMPattern<complex::MulOp> {
  using ConvertOpToLLVMPattern<complex::MulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::MulOp>(op, adaptor, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    auto fmf = LLVM::FastmathFlagsAttr::get(op.getContext(), {});
    Value rhsRe = arg.rhs.real();
    Value rhsIm = arg.rhs.imag();
    Value lhsRe = arg.lhs.real();
    Value lhsIm = arg.lhs.imag();

    Value real = rewriter.create<LLVM::FSubOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, rhsRe, lhsRe, fmf),
        rewriter.create<LLVM::FMulOp>(loc, rhsIm, lhsIm, fmf), fmf);

    Value imag = rewriter.create<LLVM::FAddOp>(
        loc, rewriter.create<LLVM::FMulOp>(loc, lhsIm, rhsRe, fmf),
        rewriter.create<LLVM::FMulOp>(loc, lhsRe, rhsIm, fmf), fmf);

    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct SubOpConversion : public ConvertOpToLLVMPattern<complex::SubOp> {
  using ConvertOpToLLVMPattern<complex::SubOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::SubOp>(op, adaptor, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to substract complex numbers.
    auto fmf = LLVM::FastmathFlagsAttr::get(op.getContext(), {});
    Value real =
        rewriter.create<LLVM::FSubOp>(loc, arg.lhs.real(), arg.rhs.real(), fmf);
    Value imag =
        rewriter.create<LLVM::FSubOp>(loc, arg.lhs.imag(), arg.rhs.imag(), fmf);
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

void mlir::populateComplexToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AbsOpConversion,
      AddOpConversion,
      ConstantOpLowering,
      CreateOpConversion,
      DivOpConversion,
      ImOpConversion,
      MulOpConversion,
      ReOpConversion,
      SubOpConversion
    >(converter);
  // clang-format on
}

namespace {
struct ConvertComplexToLLVMPass
    : public impl::ConvertComplexToLLVMPassBase<ConvertComplexToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertComplexToLLVMPass::runOnOperation() {
  // Convert to the LLVM IR dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  populateComplexToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addIllegalDialect<complex::ComplexDialect>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
