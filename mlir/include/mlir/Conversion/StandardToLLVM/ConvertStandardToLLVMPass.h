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
template <typename T>
class OperationPass;
class OwningRewritePatternList;

/// Value to pass as bitwidth for the index type when the converter is expected
/// to derive the bitwidth from the LLVM data layout.
static constexpr unsigned kDeriveIndexBitwidthFromDataLayout = 0;

struct LowerToLLVMOptions {
  bool useBarePtrCallConv = false;
  bool emitCWrappers = false;
  unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
  /// Use aligned_alloc for heap allocations.
  bool useAlignedAlloc = false;
};

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding non-memory-related
/// operations and FuncOp.
void populateStdToLLVMMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    const LowerToLLVMOptions &options);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    const LowerToLLVMOptions &options);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateStdToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    const LowerToLLVMOptions &options);

/// Collect the patterns to convert from the Standard dialect to LLVM.
void populateStdToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    const LowerToLLVMOptions &options = {
        /*useBarePtrCallConv=*/false, /*emitCWrappers=*/false,
        /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
        /*useAlignedAlloc=*/false});

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// stdlib malloc/free is used by default for allocating memrefs allocated with
/// std.alloc, while LLVM's alloca is used for those allocated with std.alloca.
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToLLVMPass(const LowerToLLVMOptions &options = {
                          /*useBarePtrCallConv=*/false, /*emitCWrappers=*/false,
                          /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
                          /*useAlignedAlloc=*/false});

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
