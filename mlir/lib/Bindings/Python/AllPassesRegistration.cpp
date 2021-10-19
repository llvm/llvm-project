//===- AllPassesRegistration.cpp - Pybind module to register all passes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Registration.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirAllPassesRegistration, m) {
  m.doc() = "MLIR All Passes Convenience Module";

  // Register all passes on load.
  mlirRegisterAllPasses();
}
