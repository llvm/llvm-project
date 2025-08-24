//===- DialectSMT.cpp - Pybind module for SMT dialect API support ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NanobindUtils.h"

#include "mlir-c/Dialect/SMT.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Target/ExportSMTLIB.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectSMTSubmodule(nanobind::module_ &m) {

  auto smtBoolType = mlir_type_subclass(m, "BoolType", mlirSMTTypeIsABool)
                         .def_classmethod(
                             "get",
                             [](const nb::object &, MlirContext context) {
                               return mlirSMTTypeGetBool(context);
                             },
                             "cls"_a, "context"_a.none() = nb::none());
  auto smtBitVectorType =
      mlir_type_subclass(m, "BitVectorType", mlirSMTTypeIsABitVector)
          .def_classmethod(
              "get",
              [](const nb::object &, int32_t width, MlirContext context) {
                return mlirSMTTypeGetBitVector(context, width);
              },
              "cls"_a, "width"_a, "context"_a.none() = nb::none());

  auto exportSMTLIB = [](MlirOperation module, bool inlineSingleUseValues,
                         bool indentLetBody) {
    mlir::python::CollectDiagnosticsToStringScope scope(
        mlirOperationGetContext(module));
    PyPrintAccumulator printAccum;
    MlirLogicalResult result = mlirTranslateOperationToSMTLIB(
        module, printAccum.getCallback(), printAccum.getUserData(),
        inlineSingleUseValues, indentLetBody);
    if (mlirLogicalResultIsSuccess(result))
      return printAccum.join();
    throw nb::value_error(
        ("Failed to export smtlib.\nDiagnostic message " + scope.takeMessage())
            .c_str());
  };

  m.def(
      "export_smtlib",
      [&exportSMTLIB](MlirOperation module, bool inlineSingleUseValues,
                      bool indentLetBody) {
        return exportSMTLIB(module, inlineSingleUseValues, indentLetBody);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false);
  m.def(
      "export_smtlib",
      [&exportSMTLIB](MlirModule module, bool inlineSingleUseValues,
                      bool indentLetBody) {
        return exportSMTLIB(mlirModuleGetOperation(module),
                            inlineSingleUseValues, indentLetBody);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false);
}

NB_MODULE(_mlirDialectsSMT, m) {
  m.doc() = "MLIR SMT Dialect";

  populateDialectSMTSubmodule(m);
}
