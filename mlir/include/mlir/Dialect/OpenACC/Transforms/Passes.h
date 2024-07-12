//===- Passes.h - OpenACC Passes Construction and Registration ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace acc {

/// Create a pass to replace ssa values in region with device/host values.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeDataInRegion();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES_H
