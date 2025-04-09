//===- DialectLinalg.cpp - Pybind module for Linalg dialect API support --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

static std::optional<MlirLinalgContractionDimensions>
InferContractionDimensions(MlirOperation op) {
  MlirLinalgContractionDimensions dims =
      mlirLinalgInferContractionDimensions(op);

  // Detect "empty" result. This occurs when `op` is not a contraction op,
  // or when `linalg::inferContractionDims` fails.
  if (mlirAttributeIsNull(dims.batch) && mlirAttributeIsNull(dims.m) &&
      mlirAttributeIsNull(dims.n) && mlirAttributeIsNull(dims.k)) {
    return std::nullopt;
  }
  return dims;
}

static void populateDialectLinalgSubmodule(nb::module_ m) {
  m.def(
      "fill_builtin_region",
      [](MlirOperation op) { mlirLinalgFillBuiltinNamedOpRegion(op); },
      nb::arg("op"),
      "Fill the region for `op`, which is assumed to be a builtin named Linalg "
      "op.");

  m.def("isa_contraction_op", &mlirLinalgIsContractionOp,
        "Checks if the given operation is a Linalg contraction operation.",
        nb::arg("op"));

  nb::class_<MlirLinalgContractionDimensions>(m, "ContractionDimensions")
      .def_prop_ro("batch",
                   [](const MlirLinalgContractionDimensions &self) {
                     return self.batch;
                   })
      .def_prop_ro(
          "m",
          [](const MlirLinalgContractionDimensions &self) { return self.m; })
      .def_prop_ro(
          "n",
          [](const MlirLinalgContractionDimensions &self) { return self.n; })
      .def_prop_ro("k", [](const MlirLinalgContractionDimensions &self) {
        return self.k;
      });

  m.def("infer_contraction_dimensions", &InferContractionDimensions,
        "Infers contraction dimensions (batch/m/n/k) for a Linalg contraction "
        "op.",
        nb::arg("op"));
}

NB_MODULE(_mlirDialectsLinalg, m) {
  m.doc() = "MLIR Linalg dialect.";

  populateDialectLinalgSubmodule(m);
}
