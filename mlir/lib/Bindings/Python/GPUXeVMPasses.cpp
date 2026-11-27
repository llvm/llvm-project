//===- GPUXeVMPasses.cpp - Nanobind module for GPU XeVM passes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU/XeVMPasses.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirGPUXeVMPasses, m) {
  m.doc() = "MLIR GPU XeVM Passes";

  // Register all GPU XeVM passes on load.
  mlirRegisterGPUXeVMPasses();
}
