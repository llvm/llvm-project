//===- Passes.h - GPU XeVM pass entrypoints ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of the XeVM-specific GPU passes and
// patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "mlir/Dialect/GPU/Transforms/XeVM/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Transforms/XeVM/Passes.h.inc"
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES_H_
