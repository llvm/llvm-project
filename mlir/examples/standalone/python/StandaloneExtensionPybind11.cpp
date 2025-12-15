//===- StandaloneExtensionPybind11.cpp - Extension module -----------------===//
//
// This is the pybind11 version of the example module. There is also a nanobind
// example in StandaloneExtensionNanobind.cpp.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_standaloneDialectsPybind11, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("standalone");

  standaloneM.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle arithHandle = mlirGetDialectHandle__arith__();
        MlirDialectHandle standaloneHandle =
            mlirGetDialectHandle__standalone__();
        mlirDialectHandleRegisterDialect(arithHandle, context);
        mlirDialectHandleRegisterDialect(standaloneHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(arithHandle, context);
          mlirDialectHandleRegisterDialect(standaloneHandle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
