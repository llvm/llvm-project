//===- Passes.h - XeGPU Patterns and Passes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace xegpu {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"

} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_H
