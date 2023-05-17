//===- StandaloneExtension.cpp - Extension module -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_standaloneDialects, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("standalone");

  standaloneM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__standalone__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
