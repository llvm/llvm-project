//===- TransformInterpreter.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pybind classes for the transform dialect interpreter.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Transform/Interpreter.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;

namespace {
struct PyMlirTransformOptions {
  PyMlirTransformOptions() { options = mlirTransformOptionsCreate(); };
  PyMlirTransformOptions(PyMlirTransformOptions &&other) {
    options = other.options;
    other.options.ptr = nullptr;
  }
  PyMlirTransformOptions(const PyMlirTransformOptions &) = delete;

  ~PyMlirTransformOptions() { mlirTransformOptionsDestroy(options); }

  MlirTransformOptions options;
};
} // namespace

static void populateTransformInterpreterSubmodule(nb::module_ &m) {
  nb::class_<PyMlirTransformOptions>(m, "TransformOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "expensive_checks",
          [](const PyMlirTransformOptions &self) {
            return mlirTransformOptionsGetExpensiveChecksEnabled(self.options);
          },
          [](PyMlirTransformOptions &self, bool value) {
            mlirTransformOptionsEnableExpensiveChecks(self.options, value);
          })
      .def_prop_rw(
          "enforce_single_top_level_transform_op",
          [](const PyMlirTransformOptions &self) {
            return mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(
                self.options);
          },
          [](PyMlirTransformOptions &self, bool value) {
            mlirTransformOptionsEnforceSingleTopLevelTransformOp(self.options,
                                                                 value);
          });

  m.def(
      "apply_named_sequence",
      [](MlirOperation payloadRoot, MlirOperation transformRoot,
         MlirOperation transformModule, const PyMlirTransformOptions &options) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(transformRoot));

        // Calling back into Python to invalidate everything under the payload
        // root. This is awkward, but we don't have access to PyMlirContext
        // object here otherwise.
        nb::object obj = nb::cast(payloadRoot);
        obj.attr("context").attr("_clear_live_operations_inside")(payloadRoot);

        MlirLogicalResult result = mlirTransformApplyNamedSequence(
            payloadRoot, transformRoot, transformModule, options.options);
        if (mlirLogicalResultIsSuccess(result))
          return;

        throw nb::value_error(
            ("Failed to apply named transform sequence.\nDiagnostic message " +
             scope.takeMessage())
                .c_str());
      },
      nb::arg("payload_root"), nb::arg("transform_root"),
      nb::arg("transform_module"),
      nb::arg("transform_options") = PyMlirTransformOptions());

  m.def(
      "copy_symbols_and_merge_into",
      [](MlirOperation target, MlirOperation other) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(target));

        MlirLogicalResult result = mlirMergeSymbolsIntoFromClone(target, other);
        if (mlirLogicalResultIsFailure(result)) {
          throw nb::value_error(
              ("Failed to merge symbols.\nDiagnostic message " +
               scope.takeMessage())
                  .c_str());
        }
      },
      nb::arg("target"), nb::arg("other"));
}

NB_MODULE(_mlirTransformInterpreter, m) {
  m.doc() = "MLIR Transform dialect interpreter functionality.";
  populateTransformInterpreterSubmodule(m);
}
