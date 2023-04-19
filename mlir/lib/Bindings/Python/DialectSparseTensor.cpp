//===- DialectSparseTensor.cpp - 'sparse_tensor' dialect submodule --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <optional>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectSparseTensorSubmodule(const py::module &m) {
  py::enum_<MlirSparseTensorDimLevelType>(m, "DimLevelType", py::module_local())
      .value("dense", MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE)
      .value("compressed", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED)
      .value("compressed-nu", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU)
      .value("compressed-no", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO)
      .value("compressed-nu-no", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO)
      .value("singleton", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON)
      .value("singleton-nu", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU)
      .value("singleton-no", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO)
      .value("singleton-nu-no", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO)
      .value("compressed-hi", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI)
      .value("compressed-hi-nu",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU)
      .value("compressed-hi-no",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NO)
      .value("compressed-hi-nu-no",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU_NO);

  mlir_attribute_subclass(m, "EncodingAttr",
                          mlirAttributeIsASparseTensorEncodingAttr)
      .def_classmethod(
          "get",
          [](py::object cls,
             std::vector<MlirSparseTensorDimLevelType> dimLevelTypes,
             std::optional<MlirAffineMap> dimOrdering,
             std::optional<MlirAffineMap> higherOrdering, int posWidth,
             int crdWidth, MlirContext context) {
            return cls(mlirSparseTensorEncodingAttrGet(
                context, dimLevelTypes.size(), dimLevelTypes.data(),
                dimOrdering ? *dimOrdering : MlirAffineMap{nullptr},
                higherOrdering ? *higherOrdering : MlirAffineMap{nullptr},
                posWidth, crdWidth));
          },
          py::arg("cls"), py::arg("dim_level_types"), py::arg("dim_ordering"),
          py::arg("higher_ordering"), py::arg("pos_width"),
          py::arg("crd_width"), py::arg("context") = py::none(),
          "Gets a sparse_tensor.encoding from parameters.")
      .def_property_readonly(
          "dim_level_types",
          [](MlirAttribute self) {
            const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
            std::vector<MlirSparseTensorDimLevelType> ret;
            ret.reserve(lvlRank);
            for (int l = 0; l < lvlRank; ++l)
              ret.push_back(
                  mlirSparseTensorEncodingAttrGetDimLevelType(self, l));
            return ret;
          })
      .def_property_readonly(
          "dim_ordering",
          [](MlirAttribute self) -> std::optional<MlirAffineMap> {
            MlirAffineMap ret =
                mlirSparseTensorEncodingAttrGetDimOrdering(self);
            if (mlirAffineMapIsNull(ret))
              return {};
            return ret;
          })
      .def_property_readonly(
          "higher_ordering",
          [](MlirAttribute self) -> std::optional<MlirAffineMap> {
            MlirAffineMap ret =
                mlirSparseTensorEncodingAttrGetHigherOrdering(self);
            if (mlirAffineMapIsNull(ret))
              return {};
            return ret;
          })
      .def_property_readonly("pos_width",
                             mlirSparseTensorEncodingAttrGetPosWidth)
      .def_property_readonly("crd_width",
                             mlirSparseTensorEncodingAttrGetCrdWidth);
}

PYBIND11_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  populateDialectSparseTensorSubmodule(m);
}
