//===- UBToLLVM.h - UB to LLVM dialect conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_UBTOLLVM_UBLLVM_H
#define MLIR_CONVERSION_UBTOLLVM_UBLLVM_H

#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_UBTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace ub {
void populateUBToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);
} // namespace ub
} // namespace mlir

#endif // MLIR_CONVERSION_UBTOLLVM_UBLLVM_H
