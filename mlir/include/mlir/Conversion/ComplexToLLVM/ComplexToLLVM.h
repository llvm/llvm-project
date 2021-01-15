//===- ComplexToLLVM.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_
#define MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;

/// Populate the given list with patterns that convert from Complex to LLVM.
void populateComplexToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Create a pass to convert Complex operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertComplexToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_
