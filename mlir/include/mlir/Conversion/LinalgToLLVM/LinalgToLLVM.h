//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
#define MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from Linalg to LLVM.
void populateLinalgToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Create a pass to convert Linalg operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
