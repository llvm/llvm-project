//===- SPIRVToLLVM.h - SPIR-V to LLVM Patterns ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVM_H
#define MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;

template <typename SPIRVOp>
class SPIRVToLLVMConversion : public OpConversionPattern<SPIRVOp> {
public:
  SPIRVToLLVMConversion(MLIRContext *context, LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit = 1)
      : OpConversionPattern<SPIRVOp>(context, benefit),
        typeConverter(typeConverter) {}

protected:
  LLVMTypeConverter &typeConverter;
};

/// Encodes global variable's descriptor set and binding into its name if they
/// both exist.
void encodeBindAttribute(ModuleOp module);

/// Populates type conversions with additional SPIR-V types.
void populateSPIRVToLLVMTypeConversion(LLVMTypeConverter &typeConverter);

/// Populates the given list with patterns that convert from SPIR-V to LLVM.
void populateSPIRVToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

/// Populates the given list with patterns for function conversion from SPIR-V
/// to LLVM.
void populateSPIRVToLLVMFunctionConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

/// Populates the given patterns for module conversion from SPIR-V to LLVM.
void populateSPIRVToLLVMModuleConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVTOLLVM_SPIRVTOLLVM_H
