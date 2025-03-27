//===- DialectLinalg.cpp - Pybind module for Linalg dialect API support --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;

static void populateDialectLinalgSubmodule(nb::module_ m) {
  m.def(
      "fill_builtin_region",
      [](MlirOperation op) { mlirLinalgFillBuiltinNamedOpRegion(op); },
      nb::arg("op"),
      "Fill the region for `op`, which is assumed to be a builtin named Linalg "
      "op.");
}

NB_MODULE(_mlirDialectsLinalg, m) {
  m.doc() = "MLIR Linalg dialect.";

  populateDialectLinalgSubmodule(m);
}
