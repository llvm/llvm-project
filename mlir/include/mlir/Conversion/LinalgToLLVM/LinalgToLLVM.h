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
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTLINALGTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Linalg to LLVM.
void populateLinalgToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H_
