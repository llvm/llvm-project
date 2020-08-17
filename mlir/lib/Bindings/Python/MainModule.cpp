//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include <pybind11/pybind11.h>

#include "IRModules.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

PYBIND11_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRSubmodule(irModule);
}
