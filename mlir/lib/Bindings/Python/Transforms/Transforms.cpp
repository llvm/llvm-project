//===- Transforms.cpp - Pybind module for the Transforms library ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Transforms.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirTransforms, m) {
  m.doc() = "MLIR Transforms library";

  // Register all the passes in the Transforms library on load.
  mlirRegisterTransformsPasses();
}
