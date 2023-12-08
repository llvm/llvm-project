//===- SparseTensorPasses.cpp - Pybind module for the SparseTensor passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SparseTensor.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirSparseTensorPasses, m) {
  m.doc() = "MLIR SparseTensor Dialect Passes";

  // Register all SparseTensor passes on load.
  mlirRegisterSparseTensorPasses();
}
