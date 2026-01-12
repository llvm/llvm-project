//===--- DialectIRDL.cpp - Pybind module for IRDL dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/IRDL.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectIRDLSubmodule(nb::module_ &m) {
  m.def(
      "load_dialects",
      [](PyModule &module) {
        if (mlirLogicalResultIsFailure(mlirLoadIRDLDialects(module.get())))
          throw std::runtime_error(
              "failed to load IRDL dialects from the input module");
      },
      nb::arg("module"), "Load IRDL dialects from the given module.");
}

NB_MODULE(_mlirDialectsIRDL, m) {
  m.doc() = "MLIR IRDL dialect.";

  populateDialectIRDLSubmodule(m);
}
