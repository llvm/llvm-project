//===- OpenMPToLLVM.h - Utils to convert from the OpenMP dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENMPTOLLVM_OPENMPTOLLVM_H_
#define MLIR_CONVERSION_OPENMPTOLLVM_OPENMPTOLLVM_H_

#include<memory>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class OwningRewritePatternList;

/// Populate the given list with patterns that convert from OpenMP to LLVM.
void populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

/// Create a pass to convert OpenMP operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenMPToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENMPTOLLVM_OPENMPTOLLVM_H_
