//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DECL
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"

/// Creates an instance of the `vector` dialect bufferization pass.
std::unique_ptr<Pass> createVectorBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector

} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H_
