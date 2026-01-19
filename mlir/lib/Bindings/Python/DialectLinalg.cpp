//===- DialectLinalg.cpp - Nanobind module for Linalg dialect API support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;
namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace linalg {

struct PyLinalgContractionDimensions : MlirLinalgContractionDimensions {
  PyLinalgContractionDimensions(const MlirLinalgContractionDimensions &dims) {
    batch = dims.batch;
    m = dims.m;
    n = dims.n;
    k = dims.k;
  }
};

struct PyLinalgConvolutionDimensions : MlirLinalgConvolutionDimensions {
  PyLinalgConvolutionDimensions(const MlirLinalgConvolutionDimensions &dims) {
    batch = dims.batch;
    outputImage = dims.outputImage;
    outputChannel = dims.outputChannel;
    filterLoop = dims.filterLoop;
    inputChannel = dims.inputChannel;
    depth = dims.depth;
    strides = dims.strides;
    dilations = dims.dilations;
  }
};

static std::optional<PyLinalgContractionDimensions>
InferContractionDimensions(PyOperationBase &op) {
  MlirLinalgContractionDimensions dims =
      mlirLinalgInferContractionDimensions(op.getOperation());

  // Detect "empty" result. This occurs when `op` is not a contraction op,
  // or when `linalg::inferContractionDims` fails.
  if (mlirAttributeIsNull(dims.batch) && mlirAttributeIsNull(dims.m) &&
      mlirAttributeIsNull(dims.n) && mlirAttributeIsNull(dims.k)) {
    return std::nullopt;
  }
  return dims;
}

static std::optional<PyLinalgConvolutionDimensions>
InferConvolutionDimensions(PyOperationBase &op) {
  MlirLinalgConvolutionDimensions dims =
      mlirLinalgInferConvolutionDimensions(op.getOperation());

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
      [](PyOperationBase &op) {
        mlirLinalgFillBuiltinNamedOpRegion(op.getOperation());
      },
      nb::arg("op"),
      "Fill the region for `op`, which is assumed to be a builtin named Linalg "
      "op.");

  m.def(
      "isa_contraction_op",
      [](PyOperationBase &op) {
        return mlirLinalgIsAContractionOp(op.getOperation());
      },
      "Checks if the given operation is a Linalg contraction operation.",
      nb::arg("op"));

  nb::class_<PyLinalgContractionDimensions>(m, "ContractionDimensions")
      .def_prop_ro(
          "batch",
          [](const PyLinalgContractionDimensions &self) { return self.batch; })
      .def_prop_ro(
          "m", [](const PyLinalgContractionDimensions &self) { return self.m; })
      .def_prop_ro(
          "n", [](const PyLinalgContractionDimensions &self) { return self.n; })
      .def_prop_ro("k", [](const PyLinalgContractionDimensions &self) {
        return self.k;
      });

  m.def("infer_contraction_dimensions", &InferContractionDimensions,
        "Infers contraction dimensions (batch/m/n/k) for a Linalg contraction "
        "op.",
        nb::arg("op"));

  m.def(
      "infer_contraction_dimensions_from_maps",
      [](std::vector<PyAffineMap> indexingMaps)
          -> std::optional<PyLinalgContractionDimensions> {
        if (indexingMaps.empty())
          return std::nullopt;

        std::vector<MlirAffineMap> indexingMaps_(indexingMaps.size());
        std::copy(indexingMaps.begin(), indexingMaps.end(),
                  indexingMaps_.begin());
        MlirLinalgContractionDimensions dims =
            mlirLinalgInferContractionDimensionsFromMaps(indexingMaps_.data(),
                                                         indexingMaps_.size());

        // Detect "empty" result from invalid input or failed inference.
        if (mlirAttributeIsNull(dims.batch) && mlirAttributeIsNull(dims.m) &&
            mlirAttributeIsNull(dims.n) && mlirAttributeIsNull(dims.k)) {
          return std::nullopt;
        }
        return dims;
      },
      "Infers contraction dimensions (batch/m/n/k) from a list of affine "
      "maps.",
      nb::arg("indexing_maps"));

  m.def(
      "isa_convolution_op",
      [](PyOperationBase &op) {
        return mlirLinalgIsAConvolutionOp(op.getOperation());
      },
      "Checks if the given operation is a Linalg convolution operation.",
      nb::arg("op"));

  nb::class_<PyLinalgConvolutionDimensions>(m, "ConvolutionDimensions")
      .def_prop_ro(
          "batch",
          [](const PyLinalgConvolutionDimensions &self) { return self.batch; })
      .def_prop_ro("output_image",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.outputImage;
                   })
      .def_prop_ro("output_channel",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.outputChannel;
                   })
      .def_prop_ro("filter_loop",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.filterLoop;
                   })
      .def_prop_ro("input_channel",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.inputChannel;
                   })
      .def_prop_ro(
          "depth",
          [](const PyLinalgConvolutionDimensions &self) { return self.depth; })
      .def_prop_ro("strides",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.strides;
                   })
      .def_prop_ro("dilations", [](const PyLinalgConvolutionDimensions &self) {
        return self.dilations;
      });

  m.def("infer_convolution_dimensions", &InferConvolutionDimensions,
        "Infers convolution dimensions", nb::arg("op"));

  m.def(
      "get_indexing_maps",
      [](PyOperationBase &op) -> std::optional<PyArrayAttribute> {
        MlirAttribute attr =
            mlirLinalgGetIndexingMapsAttribute(op.getOperation());
        if (mlirAttributeIsNull(attr))
          return std::nullopt;
        return PyArrayAttribute(op.getOperation().getContext(), attr);
      },
      "Returns the indexing_maps attribute for a linalg op.");
}
} // namespace linalg
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsLinalg, m) {
  m.doc() = "MLIR Linalg dialect.";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::linalg::
      populateDialectLinalgSubmodule(m);
}
