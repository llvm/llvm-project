//===- Passes.h - OpenMP passes entry points -----------------------*- C++
//-*-===//
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

#ifndef MLIR_DIALECT_OPENMP_PASSES_H
#define MLIR_DIALECT_OPENMP_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createOpenMPTaskBasedTargetPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace omp {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenMP/Passes.h.inc"

} // namespace omp
} // namespace mlir

#endif
