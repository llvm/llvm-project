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
class ModuleOp;
template <typename T> class OpPassBase;
class OwningRewritePatternList;

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding non-memory-related
/// operations and FuncOp.
void populateStdToLLVMMemoryConversionPatters(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlloca);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateStdToLLVMDefaultFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool emitCWrappers = false);

/// Collect a set of default patterns to convert from the Standard dialect to
/// LLVM. If `useAlloca` is set, the patterns for AllocOp and DeallocOp will
/// generate `llvm.alloca` instead of calls to "malloc".
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns,
                                         bool useAlloca = false,
                                         bool emitCWrappers = false);

/// Collect a set of patterns to convert from the Standard dialect to
/// LLVM using the bare pointer calling convention for MemRef function
/// arguments. If `useAlloca` is set, the patterns for AllocOp and DeallocOp
/// will generate `llvm.alloca` instead of calls to "malloc".
void populateStdToLLVMBarePtrConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlloca = false);

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// By default stdlib malloc/free are used for allocating MemRef payloads.
/// Specifying `useAlloca-true` emits stack allocations instead. In the future
/// this may become an enum when we have concrete uses for other options.
std::unique_ptr<OpPassBase<ModuleOp>>
createLowerToLLVMPass(bool useAlloca = false, bool useBarePtrCallConv = false,
                      bool emitCWrappers = false);

namespace LLVM {
/// Make argument-taking successors of each block distinct.  PHI nodes in LLVM
/// IR use the predecessor ID to identify which value to take.  They do not
/// support different values coming from the same predecessor.  If a block has
/// another block as a successor more than once with different values, insert
/// a new dummy block for LLVM PHI nodes to tell the sources apart.
void ensureDistinctSuccessors(ModuleOp m);
} // namespace LLVM

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
