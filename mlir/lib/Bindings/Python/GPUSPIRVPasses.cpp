//===- GPUSPIRVPasses.cpp - Nanobind module for GPU SPIRV passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU/SPIRVPasses.h"

#include "mlir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirGPUSPIRVPasses, m) {
  m.doc() = "MLIR GPU SPIRV Passes";

  // Register all GPU SPIRV passes on load.
  mlirRegisterGPUSPIRVPasses();
}
