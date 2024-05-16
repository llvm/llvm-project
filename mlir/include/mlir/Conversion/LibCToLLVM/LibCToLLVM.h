//===- LibCToLLVM.h - Utils to convert from the libc dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LIBCTOLLVM_LIBCTOLLVM_H_
#define MLIR_CONVERSION_LIBCTOLLVM_LIBCTOLLVM_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
#define GEN_PASS_DECL_CONVERTLIBCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from LibC to Func.
void populateLibCToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

void registerConvertLibCToLLVMInterface(DialectRegistry &registry);
} // namespace mlir

#endif // MLIR_CONVERSION_LIBCTOLLVM_LIBCTOLLVM_H_
