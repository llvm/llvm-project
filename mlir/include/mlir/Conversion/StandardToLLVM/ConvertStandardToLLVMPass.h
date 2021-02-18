//===- ConvertStandardToLLVMPass.h - Pass entrypoint ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_

#include "llvm/IR/DataLayout.h"

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

/// Options to control the Standard dialect to LLVM lowering. The struct is used
/// to share lowering options between passes, patterns, and type converter.
struct LowerToLLVMOptions {
  bool useBarePtrCallConv = false;
  bool emitCWrappers = false;
  unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
  /// Use aligned_alloc for heap allocations.
  bool useAlignedAlloc = false;

  /// The data layout of the module to produce. This must be consistent with the
  /// data layout used in the upper levels of the lowering pipeline.
  // TODO: this should be replaced by MLIR data layout when one exists.
  llvm::DataLayout dataLayout = llvm::DataLayout("");

  /// Get a statically allocated copy of the default LowerToLLVMOptions.
  static const LowerToLLVMOptions &getDefaultOptions() {
    static LowerToLLVMOptions options;
    return options;
  }
};

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding non-memory-related
/// operations and FuncOp.
void populateStdToLLVMMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateStdToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);

/// Collect the patterns to convert from the Standard dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

/// Creates a pass to convert the Standard dialect into the LLVMIR dialect.
/// stdlib malloc/free is used by default for allocating memrefs allocated with
/// memref.alloc, while LLVM's alloca is used for those allocated with
/// memref.alloca.
std::unique_ptr<OperationPass<ModuleOp>>
createLowerToLLVMPass(const LowerToLLVMOptions &options =
                          LowerToLLVMOptions::getDefaultOptions());

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVMPASS_H_
