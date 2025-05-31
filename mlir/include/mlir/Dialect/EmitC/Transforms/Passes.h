//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_EMITC_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

#define GEN_PASS_DECL_FORMEXPRESSIONSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES_H_
