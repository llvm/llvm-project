//===- Conversions.cpp - Pybind module for the Conversionss library -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Conversion.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirConversions, m) {
  m.doc() = "MLIR Conversions library";

  // Register all the passes in the Conversions library on load.
  mlirRegisterConversionPasses();
}
