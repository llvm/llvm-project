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

/// Value to pass as bitwidth for the index type when the converter is expected
/// to derive the bitwith from the LLVM data layout.
static constexpr unsigned kDeriveIndexBitwidthFromDataLayout = 0;

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// By default stdlib malloc/free are used for allocating MemRef payloads.
/// Specifying `useAlloca-true` emits stack allocations instead. In the future
/// this may become an enum when we have concrete uses for other options.
std::unique_ptr<OpPassBase<ModuleOp>> createLowerToLLVMPass(
    bool useAlloca = false, bool useBarePtrCallConv = false,
    bool emitCWrappers = false,
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
