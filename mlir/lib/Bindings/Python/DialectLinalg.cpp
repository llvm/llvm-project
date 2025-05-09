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

static std::optional<MlirLinalgConvolutionDimensions>
InferConvolutionDimensions(MlirOperation op) {
  MlirLinalgConvolutionDimensions dims =
      mlirLinalgInferConvolutionDimensions(op);

  // Detect "empty" result. This occurs when `op` is not a convolution op,
  // or when `linalg::inferConvolutionDims` fails.
  if (mlirAttributeIsNull(dims.batch) &&
      mlirAttributeIsNull(dims.outputImage) &&
      mlirAttributeIsNull(dims.outputChannel) &&
      mlirAttributeIsNull(dims.filterLoop) &&
      mlirAttributeIsNull(dims.inputChannel) &&
      mlirAttributeIsNull(dims.depth) && mlirAttributeIsNull(dims.strides) &&
      mlirAttributeIsNull(dims.dilations)) {
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

  m.def("isa_contraction_op", &mlirLinalgIsAContractionOp,
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

  m.def("isa_convolution_op", &mlirLinalgIsAConvolutionOp,
        "Checks if the given operation is a Linalg convolution operation.",
        nb::arg("op"));

  nb::class_<MlirLinalgConvolutionDimensions>(m, "ConvolutionDimensions")
      .def_prop_ro("batch",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.batch;
                   })
      .def_prop_ro("output_image",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.outputImage;
                   })
      .def_prop_ro("output_channel",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.outputChannel;
                   })
      .def_prop_ro("filter_loop",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.filterLoop;
                   })
      .def_prop_ro("input_channel",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.inputChannel;
                   })
      .def_prop_ro("depth",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.depth;
                   })
      .def_prop_ro("strides",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.strides;
                   })
      .def_prop_ro("dilations",
                   [](const MlirLinalgConvolutionDimensions &self) {
                     return self.dilations;
                   });

  m.def("infer_convolution_dimensions", &InferConvolutionDimensions,
        "Infers convolution dimensions", nb::arg("op"));

  m.def(
      "get_indexing_maps",
      [](MlirOperation op) -> std::optional<MlirAttribute> {
        MlirAttribute attr = mlirLinalgGetIndexingMapsAttribute(op);
        if (mlirAttributeIsNull(attr))
          return std::nullopt;
        return attr;
      },
      "Returns the indexing_maps attribute for a linalg op.");
}

NB_MODULE(_mlirDialectsLinalg, m) {
  m.doc() = "MLIR Linalg dialect.";

  populateDialectLinalgSubmodule(m);
}
