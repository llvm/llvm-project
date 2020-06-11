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

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Converts SPIR-V operations that have straightforward LLVM equivalent
/// into LLVM dialect operations.
template <typename SPIRVOp, typename LLVMOp>
class DirectConversionPattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->typeConverter.convertType(operation.getType());
    if (!dstType)
      return failure();
    rewriter.template replaceOpWithNewOp<LLVMOp>(operation, dstType, operands);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateSPIRVToLLVMConversionPatterns(
    MLIRContext *context, LLVMTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<DirectConversionPattern<spirv::IAddOp, LLVM::AddOp>,
                  DirectConversionPattern<spirv::IMulOp, LLVM::MulOp>,
                  DirectConversionPattern<spirv::ISubOp, LLVM::SubOp>,
                  DirectConversionPattern<spirv::FAddOp, LLVM::FAddOp>,
                  DirectConversionPattern<spirv::FNegateOp, LLVM::FNegOp>,
                  DirectConversionPattern<spirv::FDivOp, LLVM::FDivOp>,
                  DirectConversionPattern<spirv::FRemOp, LLVM::FRemOp>,
                  DirectConversionPattern<spirv::FSubOp, LLVM::FSubOp>,
                  DirectConversionPattern<spirv::UDivOp, LLVM::UDivOp>,
                  DirectConversionPattern<spirv::SDivOp, LLVM::SDivOp>,
                  DirectConversionPattern<spirv::SRemOp, LLVM::SRemOp>,
                  DirectConversionPattern<spirv::BitwiseAndOp, LLVM::AndOp>,
                  DirectConversionPattern<spirv::BitwiseOrOp, LLVM::OrOp>,
                  DirectConversionPattern<spirv::BitwiseXorOp, LLVM::XOrOp>>(
      context, typeConverter);
}
