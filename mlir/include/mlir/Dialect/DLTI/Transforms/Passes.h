//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_DLTI_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "mlir/Dialect/DLTI/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/DLTI/Transforms/Passes.h.inc"

/// Sets the target specs using the target attached to the module.
LogicalResult setTargetSpecsFromTarget(Operation *op);
} // namespace mlir

#endif // MLIR_DIALECT_DLTI_TRANSFORMS_PASSES_H
