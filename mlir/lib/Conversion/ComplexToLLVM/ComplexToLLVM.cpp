//===- ComplexToLLVM.cpp - conversion from Complex to LLVM dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ComplexCommon/DivisionConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// ComplexStructBuilder implementation.
//===----------------------------------------------------------------------===//

static constexpr unsigned kRealPosInComplexNumberStruct = 0;
static constexpr unsigned kImaginaryPosInComplexNumberStruct = 1;

ComplexStructBuilder ComplexStructBuilder::poison(OpBuilder &builder,
                                                  Location loc, Type type) {
  Value val = LLVM::PoisonOp::create(builder, loc, type);
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

    arith::FastMathFlagsAttr complexFMFAttr = op.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        op.getContext(),
        convertArithFastMathFlagsToLLVM(complexFMFAttr.getValue()));
    Value sqNorm = LLVM::FAddOp::create(
        rewriter, loc, LLVM::FMulOp::create(rewriter, loc, real, real, fmf),
        LLVM::FMulOp::create(rewriter, loc, imag, imag, fmf), fmf);

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
        op->getAttrs(), /*propAttr=*/Attribute{}, *getTypeConverter(),
        rewriter);
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
    auto complexStruct =
        ComplexStructBuilder::poison(rewriter, loc, structType);
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
    auto result = ComplexStructBuilder::poison(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    arith::FastMathFlagsAttr complexFMFAttr = op.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        op.getContext(),
        convertArithFastMathFlagsToLLVM(complexFMFAttr.getValue()));
    Value real = LLVM::FAddOp::create(rewriter, loc, arg.lhs.real(),
                                      arg.rhs.real(), fmf);
    Value imag = LLVM::FAddOp::create(rewriter, loc, arg.lhs.imag(),
                                      arg.rhs.imag(), fmf);
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct DivOpConversion : public ConvertOpToLLVMPattern<complex::DivOp> {
  DivOpConversion(const LLVMTypeConverter &converter,
                  complex::ComplexRangeFlags target)
      : ConvertOpToLLVMPattern<complex::DivOp>(converter),
        complexRange(target) {}

  using ConvertOpToLLVMPattern<complex::DivOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::DivOp>(op, adaptor, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::poison(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    arith::FastMathFlagsAttr complexFMFAttr = op.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        op.getContext(),
        convertArithFastMathFlagsToLLVM(complexFMFAttr.getValue()));
    Value rhsRe = arg.rhs.real();
    Value rhsIm = arg.rhs.imag();
    Value lhsRe = arg.lhs.real();
    Value lhsIm = arg.lhs.imag();

    Value resultRe, resultIm;

    if (complexRange == complex::ComplexRangeFlags::basic ||
        complexRange == complex::ComplexRangeFlags::none) {
      mlir::complex::convertDivToLLVMUsingAlgebraic(
          rewriter, loc, lhsRe, lhsIm, rhsRe, rhsIm, fmf, &resultRe, &resultIm);
    } else if (complexRange == complex::ComplexRangeFlags::improved) {
      mlir::complex::convertDivToLLVMUsingRangeReduction(
          rewriter, loc, lhsRe, lhsIm, rhsRe, rhsIm, fmf, &resultRe, &resultIm);
    }

    result.setReal(rewriter, loc, resultRe);
    result.setImaginary(rewriter, loc, resultIm);

    rewriter.replaceOp(op, {result});
    return success();
  }

private:
  complex::ComplexRangeFlags complexRange;
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
    auto result = ComplexStructBuilder::poison(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    arith::FastMathFlagsAttr complexFMFAttr = op.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        op.getContext(),
        convertArithFastMathFlagsToLLVM(complexFMFAttr.getValue()));
    Value rhsRe = arg.rhs.real();
    Value rhsIm = arg.rhs.imag();
    Value lhsRe = arg.lhs.real();
    Value lhsIm = arg.lhs.imag();

    Value real = LLVM::FSubOp::create(
        rewriter, loc, LLVM::FMulOp::create(rewriter, loc, rhsRe, lhsRe, fmf),
        LLVM::FMulOp::create(rewriter, loc, rhsIm, lhsIm, fmf), fmf);

    Value imag = LLVM::FAddOp::create(
        rewriter, loc, LLVM::FMulOp::create(rewriter, loc, lhsIm, rhsRe, fmf),
        LLVM::FMulOp::create(rewriter, loc, lhsRe, rhsIm, fmf), fmf);

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
    auto result = ComplexStructBuilder::poison(rewriter, loc, structType);

    // Emit IR to substract complex numbers.
    arith::FastMathFlagsAttr complexFMFAttr = op.getFastMathFlagsAttr();
    LLVM::FastmathFlagsAttr fmf = LLVM::FastmathFlagsAttr::get(
        op.getContext(),
        convertArithFastMathFlagsToLLVM(complexFMFAttr.getValue()));
    Value real = LLVM::FSubOp::create(rewriter, loc, arg.lhs.real(),
                                      arg.rhs.real(), fmf);
    Value imag = LLVM::FSubOp::create(rewriter, loc, arg.lhs.imag(),
                                      arg.rhs.imag(), fmf);
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

void mlir::populateComplexToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    complex::ComplexRangeFlags complexRange) {
  // clang-format off
  patterns.add<
      AbsOpConversion,
      AddOpConversion,
      ConstantOpLowering,
      CreateOpConversion,
      ImOpConversion,
      MulOpConversion,
      ReOpConversion,
      SubOpConversion
    >(converter);

  patterns.add<DivOpConversion>(converter, complexRange);
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
  populateComplexToLLVMConversionPatterns(converter, patterns, complexRange);

  LLVMConversionTarget target(getContext());
  target.addIllegalDialect<complex::ComplexDialect>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert MemRef to LLVM.
struct ComplexToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateComplexToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertComplexToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, complex::ComplexDialect *dialect) {
        dialect->addInterfaces<ComplexToLLVMDialectInterface>();
      });
}
