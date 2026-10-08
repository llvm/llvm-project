//===- Passes.h - NVVM Pass Construction and Registration -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVVM_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_NVVM_TRANSFORMS_PASSES_H

#include "mlir/Dialect/NVVM/Transforms/OptimizeForNVVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace NVVM {

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/NVVM/Transforms/Passes.h.inc"

} // namespace NVVM
} // namespace mlir

#endif // MLIR_DIALECT_NVVM_TRANSFORMS_PASSES_H
