//===- LLVMToControlFlow.h - Convert LLVM to CF dialect ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H
#define MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTLLVMTOCONTROLFLOW
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H
