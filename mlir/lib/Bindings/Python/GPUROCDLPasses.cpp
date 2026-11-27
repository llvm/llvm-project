//===- GPUROCDLPasses.cpp - Nanobind module for GPU ROCDL passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU/ROCDLPasses.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirGPUROCDLPasses, m) {
  m.doc() = "MLIR GPU ROCDL Passes";

  // Register all GPU ROCDL passes on load.
  mlirRegisterGPUROCDLPasses();
}
