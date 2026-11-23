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
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace transform_interpreter {
struct PyTransformOptions {
  PyTransformOptions() { options = mlirTransformOptionsCreate(); };
  PyTransformOptions(PyTransformOptions &&other) {
    options = other.options;
    other.options.ptr = nullptr;
  }
  PyTransformOptions(const PyTransformOptions &) = delete;

  ~PyTransformOptions() { mlirTransformOptionsDestroy(options); }

  MlirTransformOptions options;
};
} // namespace transform_interpreter
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

static void populateTransformInterpreterSubmodule(nb::module_ &m) {
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
  using namespace transform_interpreter;
  nb::class_<PyTransformOptions>(m, "TransformOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "expensive_checks",
          [](const PyTransformOptions &self) {
            return mlirTransformOptionsGetExpensiveChecksEnabled(self.options);
          },
          [](PyTransformOptions &self, bool value) {
            mlirTransformOptionsEnableExpensiveChecks(self.options, value);
          })
      .def_prop_rw(
          "enforce_single_top_level_transform_op",
          [](const PyTransformOptions &self) {
            return mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(
                self.options);
          },
          [](PyTransformOptions &self, bool value) {
            mlirTransformOptionsEnforceSingleTopLevelTransformOp(self.options,
                                                                 value);
          });

  m.def(
      "apply_named_sequence",
      [](PyOperationBase &payloadRoot, PyOperationBase &transformRoot,
         PyOperationBase &transformModule, const PyTransformOptions &options) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(transformRoot.getOperation()));
        MlirLogicalResult result = mlirTransformApplyNamedSequence(
            payloadRoot.getOperation(), transformRoot.getOperation(),
            transformModule.getOperation(), options.options);
        if (mlirLogicalResultIsSuccess(result)) {
          // Even in cases of success, we might have diagnostics to report:
          std::string msg;
          if ((msg = scope.takeMessage()).size() > 0) {
            fprintf(stderr,
                    "Diagnostic generated while applying "
                    "transform.named_sequence:\n%s",
                    msg.data());
          }
          return;
        }

        throw nb::value_error(
            ("Failed to apply named transform sequence.\nDiagnostic message " +
             scope.takeMessage())
                .c_str());
      },
      nb::arg("payload_root"), nb::arg("transform_root"),
      nb::arg("transform_module"),
      nb::arg("transform_options") = PyTransformOptions());

  m.def(
      "copy_symbols_and_merge_into",
      [](PyOperationBase &target, PyOperationBase &other) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(target.getOperation()));

        MlirLogicalResult result = mlirMergeSymbolsIntoFromClone(
            target.getOperation(), other.getOperation());
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
