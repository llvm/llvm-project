//===- LegalizeForLLVMExport.cpp - Prepare X86Vector for LLVM translation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/Transforms.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::x86vector;

namespace {

/// Generic one-to-one conversion of simply mappable operations into calls
/// to their respective LLVM intrinsics.
struct X86IntrinsicOpConversion
    : public OpInterfaceConversionPattern<x86vector::X86IntrinsicOp> {
  using OpInterfaceConversionPattern<
      x86vector::X86IntrinsicOp>::OpInterfaceConversionPattern;

  X86IntrinsicOpConversion(const LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern(typeConverter, &typeConverter.getContext(),
                                     benefit),
        typeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(x86vector::X86IntrinsicOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return LLVM::detail::intrinsicRewrite(
        op, rewriter.getStringAttr(op.getIntrinsicName()),
        op.getIntrinsicOperands(operands, typeConverter, rewriter),
        typeConverter, rewriter);
  }

private:
  const LLVMTypeConverter &typeConverter;
};

} // namespace

/// Populate the given list with patterns that convert from X86Vector to LLVM.
void mlir::populateX86VectorLegalizeForLLVMExportPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<X86IntrinsicOpConversion>(converter);
}

void mlir::configureX86VectorLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addIllegalDialect<X86VectorDialect>();
}
