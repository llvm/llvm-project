//===- DialectLinalg.cpp - Pybind module for Linalg dialect API support --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

struct PyContractionDimensions {
  MlirLinalgContractionDimensions value;

  PyContractionDimensions() = default;
  PyContractionDimensions(const MlirLinalgContractionDimensions &v)
      : value(v) {}
};

static std::optional<PyContractionDimensions>
mlirLinalgInferContractionDimensionsBinding(MlirOperation op) {
  MlirLinalgContractionDimensions dims =
      mlirLinalgInferContractionDimensions(op);

  // Detect "empty" result.
  if (mlirAttributeIsNull(dims.batch) && mlirAttributeIsNull(dims.m) &&
      mlirAttributeIsNull(dims.n) && mlirAttributeIsNull(dims.k)) {
    return std::nullopt;
  }
  return PyContractionDimensions{dims};
}

static std::vector<int32_t> convertDenseI32AttrToList(MlirAttribute attr) {
  std::vector<int32_t> result;
  int64_t size = mlirDenseArrayGetNumElements(attr);
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    result.push_back(mlirDenseI32ArrayGetElement(attr, i));
  }
  return result;
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

  nb::class_<PyContractionDimensions>(m, "ContractionDimensions")
      .def_prop_ro("batch",
                   [](const PyContractionDimensions &self) {
                     return convertDenseI32AttrToList(self.value.batch);
                   })
      .def_prop_ro("m",
                   [](const PyContractionDimensions &self) {
                     return convertDenseI32AttrToList(self.value.m);
                   })
      .def_prop_ro("n",
                   [](const PyContractionDimensions &self) {
                     return convertDenseI32AttrToList(self.value.n);
                   })
      .def_prop_ro("k", [](const PyContractionDimensions &self) {
        return convertDenseI32AttrToList(self.value.k);
      });

  m.def("infer_contraction_dimensions",
        &mlirLinalgInferContractionDimensionsBinding,
        "Infers contraction dimensions (batch/m/n/k) for a Linalg contraction "
        "op.",
        nb::arg("op"));
}

NB_MODULE(_mlirDialectsLinalg, m) {
  m.doc() = "MLIR Linalg dialect.";

  populateDialectLinalgSubmodule(m);
}
