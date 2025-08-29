//===- PtrToLLVM.h - Ptr to LLVM dialect conversion -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PTRTOLLVM_PTRTOLLVM_H
#define MLIR_CONVERSION_PTRTOLLVM_PTRTOLLVM_H

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
namespace ptr {
/// Populate the convert to LLVM patterns for the `ptr` dialect.
void populatePtrToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
/// Register the convert to LLVM interface for the `ptr` dialect.
void registerConvertPtrToLLVMInterface(DialectRegistry &registry);
} // namespace ptr
} // namespace mlir

#endif // MLIR_CONVERSION_PTRTOLLVM_PTRTOLLVM_H
