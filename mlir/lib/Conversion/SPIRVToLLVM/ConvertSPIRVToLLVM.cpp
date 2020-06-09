//===- ConvertSPIRVToLLVM.cpp - SPIR-V dialect to LLVM dialect conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SPIRVToLLVM/ConvertSPIRVToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

class BitwiseAndOpConversion : public ConvertToLLVMPattern {
public:
  explicit BitwiseAndOpConversion(MLIRContext *context,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(spirv::BitwiseAndOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bitwiseAndOp = cast<spirv::BitwiseAndOp>(op);
    auto dstType = typeConverter.convertType(bitwiseAndOp.getType());
    if (!dstType)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(bitwiseAndOp, dstType, operands);
    return success();
  }
};
} // namespace

void mlir::populateSPIRVToLLVMConversionPatterns(
    MLIRContext *context, LLVMTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BitwiseAndOpConversion>(context, typeConverter);
}
