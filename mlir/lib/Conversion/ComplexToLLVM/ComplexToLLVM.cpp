//===- ComplexToLLVM.cpp - conversion from Complex to LLVM dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

struct CreateOpConversion : public ConvertOpToLLVMPattern<complex::CreateOp> {
  using ConvertOpToLLVMPattern<complex::CreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::CreateOp complexOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::CreateOp::Adaptor transformed(operands);

    // Pack real and imaginary part in a complex number struct.
    auto loc = complexOp.getLoc();
    auto structType = typeConverter->convertType(complexOp.getType());
    auto complexStruct = ComplexStructBuilder::undef(rewriter, loc, structType);
    complexStruct.setReal(rewriter, loc, transformed.real());
    complexStruct.setImaginary(rewriter, loc, transformed.imaginary());

    rewriter.replaceOp(complexOp, {complexStruct});
    return success();
  }
};

struct ReOpConversion : public ConvertOpToLLVMPattern<complex::ReOp> {
  using ConvertOpToLLVMPattern<complex::ReOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::ReOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::ReOp::Adaptor transformed(operands);

    // Extract real part from the complex number struct.
    ComplexStructBuilder complexStruct(transformed.complex());
    Value real = complexStruct.real(rewriter, op.getLoc());
    rewriter.replaceOp(op, real);

    return success();
  }
};

struct ImOpConversion : public ConvertOpToLLVMPattern<complex::ImOp> {
  using ConvertOpToLLVMPattern<complex::ImOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::ImOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::ImOp::Adaptor transformed(operands);

    // Extract imaginary part from the complex number struct.
    ComplexStructBuilder complexStruct(transformed.complex());
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
unpackBinaryComplexOperands(OpTy op, ArrayRef<Value> operands,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  typename OpTy::Adaptor transformed(operands);

  // Extract real and imaginary values from operands.
  BinaryComplexOperands unpacked;
  ComplexStructBuilder lhs(transformed.lhs());
  unpacked.lhs.real(lhs.real(rewriter, loc));
  unpacked.lhs.imag(lhs.imaginary(rewriter, loc));
  ComplexStructBuilder rhs(transformed.rhs());
  unpacked.rhs.real(rhs.real(rewriter, loc));
  unpacked.rhs.imag(rhs.imaginary(rewriter, loc));

  return unpacked;
}

struct AddOpConversion : public ConvertOpToLLVMPattern<complex::AddOp> {
  using ConvertOpToLLVMPattern<complex::AddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::AddOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::AddOp>(op, operands, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    auto fmf = LLVM::FMFAttr::get({}, op.getContext());
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

struct SubOpConversion : public ConvertOpToLLVMPattern<complex::SubOp> {
  using ConvertOpToLLVMPattern<complex::SubOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(complex::SubOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<complex::SubOp>(op, operands, rewriter);

    // Initialize complex number struct for result.
    auto structType = typeConverter->convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to substract complex numbers.
    auto fmf = LLVM::FMFAttr::get({}, op.getContext());
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
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      AddOpConversion,
      CreateOpConversion,
      ImOpConversion,
      ReOpConversion,
      SubOpConversion
    >(converter);
  // clang-format on
}

namespace {
struct ConvertComplexToLLVMPass
    : public ConvertComplexToLLVMBase<ConvertComplexToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertComplexToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateComplexToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertComplexToLLVMPass() {
  return std::make_unique<ConvertComplexToLLVMPass>();
}
