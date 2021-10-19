//===- Passes.h - Reducer Pass Construction and Registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_REDUCER_PASSES_H
#define MLIR_REDUCER_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createReductionTreePass();

std::unique_ptr<Pass> createOptReductionPass();

/// Generate the code for registering reducer passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Reducer/Passes.h.inc"

} // end namespace mlir

#endif // MLIR_REDUCER_PASSES_H
