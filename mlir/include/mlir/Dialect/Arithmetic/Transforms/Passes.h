//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {

/// Create a pass to bufferize Arithmetic ops.
std::unique_ptr<Pass> createArithmeticBufferizePass();

/// Add patterns to expand Arithmetic ops for LLVM lowering.
void populateArithmeticExpandOpsPatterns(RewritePatternSet &patterns);

/// Create a pass to legalize Arithmetic ops for LLVM lowering.
std::unique_ptr<Pass> createArithmeticExpandOpsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h.inc"

} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES_H_
