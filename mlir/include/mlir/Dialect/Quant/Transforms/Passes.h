//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DECL
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

void populateLowerQuantOpsPatterns(RewritePatternSet &patterns);

} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANT_TRANSFORMS_PASSES_H_
