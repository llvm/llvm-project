//===- OpenMPToLLVM.h - Utils to convert from the OpenMP dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H
#define MLIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTOOPENMPTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

/// Configure dynamic conversion legality of regionless operations from OpenMP
/// to LLVM.
void configureOpenMPToLLVMConversionLegality(ConversionTarget &target,
                                             LLVMTypeConverter &typeConverter);

/// Populate the given list with patterns that convert from OpenMP to LLVM.
void populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Create a pass to convert OpenMP operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenMPToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H
