//===- LegalizeForLLVMExport.cpp - Prepare X86 for LLVM translation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86/Transforms.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::x86;

namespace {

/// Generic one-to-one conversion of simply mappable operations into calls
/// to their respective LLVM intrinsics.
struct X86IntrinsicOpConversion
    : public ConvertOpInterfaceToLLVMPattern<x86::X86IntrinsicOp> {
  using ConvertOpInterfaceToLLVMPattern::ConvertOpInterfaceToLLVMPattern;

  LogicalResult
  matchAndRewrite(x86::X86IntrinsicOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const LLVMTypeConverter &typeConverter = *getTypeConverter();
    return LLVM::detail::intrinsicRewrite(
        op, rewriter.getStringAttr(op.getIntrinsicName()),
        op.getIntrinsicOperands(operands, typeConverter, rewriter),
        typeConverter, rewriter);
  }
};

} // namespace

/// Populate the given list with patterns that convert from X86 to LLVM.
void mlir::populateX86LegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<X86IntrinsicOpConversion>(converter);
  converter.addConversion([&](x86::amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(&converter.getContext());
  });
}

void mlir::configureX86LegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addIllegalDialect<X86Dialect>();
}

namespace {
/// Implement the interface to convert X86 to LLVM.
struct X86ToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateX86LegalizeForLLVMExportPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertX86ToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, x86::X86Dialect *dialect) {
    dialect->addInterfaces<X86ToLLVMDialectInterface>();
  });
}
