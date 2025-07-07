//===- LegalizeForLLVMExport.cpp - Prepare AMX for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/Transforms.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::amx;

namespace {

/// Generic one-to-one conversion of simply mappable operations into calls
/// to their respective LLVM intrinsics.
struct AMXIntrinsicOpConversion
    : public ConvertOpInterfaceToLLVMPattern<amx::AMXIntrinsicOp> {
  using ConvertOpInterfaceToLLVMPattern::ConvertOpInterfaceToLLVMPattern;

  LogicalResult
  matchAndRewrite(amx::AMXIntrinsicOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const LLVMTypeConverter &typeConverter = *getTypeConverter();
    return LLVM::detail::intrinsicRewrite(
        op, rewriter.getStringAttr(op.getIntrinsicName()),
        op.getIntrinsicOperands(operands, typeConverter, rewriter),
        typeConverter, rewriter);
  }
};

} // namespace

void mlir::populateAMXLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<AMXIntrinsicOpConversion>(converter);
  converter.addConversion([&](amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(&converter.getContext());
  });
}

void mlir::configureAMXLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addIllegalDialect<AMXDialect>();
}

namespace {
/// Implement the interface to convert AMX to LLVM.
struct AMXToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateAMXLegalizeForLLVMExportPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertAMXToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, amx::AMXDialect *dialect) {
    dialect->addInterfaces<AMXToLLVMDialectInterface>();
  });
}
