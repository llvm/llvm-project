//===- StandaloneExtension.cpp - Extension module -------------------------===//
//
// This is the nanobind version of the example module. There is also a pybind11
// example in StandaloneExtensionPybind11.cpp.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_standaloneDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("standalone");

  standaloneM.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle standaloneHandle =
            mlirGetDialectHandle__standalone__();
        MlirDialectHandle arithHandle = mlirGetDialectHandle__arith__();
        mlirDialectHandleRegisterDialect(standaloneHandle, context);
        mlirDialectHandleRegisterDialect(arithHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(standaloneHandle, context);
          mlirDialectHandleLoadDialect(arithHandle, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
