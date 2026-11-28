//===- GPUNVVMPasses.cpp - Nanobind module for GPU NVVM passes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU/NVVMPasses.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirGPUNVVMPasses, m) {
  m.doc() = "MLIR GPU NVVM Passes";

  // Register all GPU NVVM passes on load.
  mlirRegisterGPUNVVMPasses();
}
