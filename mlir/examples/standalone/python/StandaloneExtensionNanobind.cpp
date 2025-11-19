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
      nb::arg("context").none() = nb::none(), nb::arg("load") = true,
      // clang-format off
      nb::sig("def register_dialects(context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") ", load: bool = True) -> None")
      // clang-format on
  );
}
