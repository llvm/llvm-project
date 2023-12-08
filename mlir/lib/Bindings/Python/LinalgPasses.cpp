//===- LinalgPasses.cpp - Pybind module for the Linalg passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirLinalgPasses, m) {
  m.doc() = "MLIR Linalg Dialect Passes";

  // Register all Linalg passes on load.
  mlirRegisterLinalgPasses();
}
