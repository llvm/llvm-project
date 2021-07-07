//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class LowerToLLVMOptions;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding non-memory-related
/// operations and FuncOp.
void populateStdToLLVMMemoryConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateStdToLLVMFuncOpConversionPattern(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);

/// Collect the patterns to convert from the Standard dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// stdlib malloc/free is used by default for allocating memrefs allocated with
/// memref.alloc, while LLVM's alloca is used for those allocated with
/// memref.alloca.
std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToLLVMPass(const LowerToLLVMOptions &options);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
