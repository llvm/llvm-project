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

#ifndef MLIR_DIALECT_SCF_PASSES_H_
#define MLIR_DIALECT_SCF_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass that specializes for loop for unrolling and
/// vectorization.
std::unique_ptr<Pass> createForLoopSpecializationPass();

/// Creates a loop fusion pass which fuses parallel loops.
std::unique_ptr<Pass> createParallelLoopFusionPass();

/// Creates a pass that specializes parallel loop for unrolling and
/// vectorization.
std::unique_ptr<Pass> createParallelLoopSpecializationPass();

/// Creates a pass which tiles innermost parallel loops.
std::unique_ptr<Pass>
createParallelLoopTilingPass(llvm::ArrayRef<int64_t> tileSize = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SCF/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SCF_PASSES_H_
