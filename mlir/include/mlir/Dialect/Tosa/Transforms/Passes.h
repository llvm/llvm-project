//===-- Passes.h - TOSA optimization pass declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

std::unique_ptr<Pass> createTosaMakeBroadcastablePass();
std::unique_ptr<Pass> createTosaTestQuantUtilAPIPass();

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSES_H
