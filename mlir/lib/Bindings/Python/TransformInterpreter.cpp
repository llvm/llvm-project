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
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

static void populateTransformInterpreterSubmodule(py::module &m) {
  py::class_<PyMlirTransformOptions>(m, "TransformOptions", py::module_local())
      .def(py::init())
      .def_property(
          "expensive_checks",
          [](const PyMlirTransformOptions &self) {
            return mlirTransformOptionsGetExpensiveChecksEnabled(self.options);
          },
          [](PyMlirTransformOptions &self, bool value) {
            mlirTransformOptionsEnableExpensiveChecks(self.options, value);
          })
      .def_property(
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
        py::object obj = py::cast(payloadRoot);
        obj.attr("context").attr("_clear_live_operations_inside")(payloadRoot);

        MlirLogicalResult result = mlirTransformApplyNamedSequence(
            payloadRoot, transformRoot, transformModule, options.options);
        if (mlirLogicalResultIsSuccess(result))
          return;

        throw py::value_error(
            "Failed to apply named transform sequence.\nDiagnostic message " +
            scope.takeMessage());
      },
      py::arg("payload_root"), py::arg("transform_root"),
      py::arg("transform_module"),
      py::arg("transform_options") = PyMlirTransformOptions());
}

PYBIND11_MODULE(_mlirTransformInterpreter, m) {
  m.doc() = "MLIR Transform dialect interpreter functionality.";
  populateTransformInterpreterSubmodule(m);
}
