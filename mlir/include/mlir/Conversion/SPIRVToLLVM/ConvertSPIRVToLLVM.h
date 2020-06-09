//===- ConvertSPIRVToLLVM.h - Convert SPIR-V to LLVM dialect ----*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVM_H
#define MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;

/// Populates the given list with patterns that convert from SPIR-V to LLVM.
void populateSPIRVToLLVMConversionPatterns(MLIRContext *context,
                                           LLVMTypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVTOLLVM_CONVERTSPIRVTOLLVM_H
