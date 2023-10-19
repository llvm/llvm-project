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
      .value("compressed24", MLIR_SPARSE_TENSOR_DIM_LEVEL_TWO_OUT_OF_FOUR)
      .value("compressed", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED)
      .value("compressed_nu", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU)
      .value("compressed_no", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO)
      .value("compressed_nu_no", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO)
      .value("singleton", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON)
      .value("singleton_nu", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU)
      .value("singleton_no", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO)
      .value("singleton_nu_no", MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO)
      .value("loose_compressed", MLIR_SPARSE_TENSOR_DIM_LEVEL_LOOSE_COMPRESSED)
      .value("loose_compressed_nu",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_LOOSE_COMPRESSED_NU)
      .value("loose_compressed_no",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_LOOSE_COMPRESSED_NO)
      .value("loose_compressed_nu_no",
             MLIR_SPARSE_TENSOR_DIM_LEVEL_LOOSE_COMPRESSED_NU_NO);

  mlir_attribute_subclass(m, "EncodingAttr",
                          mlirAttributeIsASparseTensorEncodingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<MlirSparseTensorDimLevelType> lvlTypes,
             std::optional<MlirAffineMap> dimToLvl,
             std::optional<MlirAffineMap> lvlToDim, int posWidth, int crdWidth,
             MlirContext context) {
            return cls(mlirSparseTensorEncodingAttrGet(
                context, lvlTypes.size(), lvlTypes.data(),
                dimToLvl ? *dimToLvl : MlirAffineMap{nullptr},
                lvlToDim ? *lvlToDim : MlirAffineMap{nullptr}, posWidth,
                crdWidth));
          },
          py::arg("cls"), py::arg("lvl_types"), py::arg("dim_to_lvl"),
          py::arg("lvl_to_dim"), py::arg("pos_width"), py::arg("crd_width"),
          py::arg("context") = py::none(),
          "Gets a sparse_tensor.encoding from parameters.")
      .def_property_readonly(
          "lvl_types",
          [](MlirAttribute self) {
            const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
            std::vector<MlirSparseTensorDimLevelType> ret;
            ret.reserve(lvlRank);
            for (int l = 0; l < lvlRank; ++l)
              ret.push_back(mlirSparseTensorEncodingAttrGetLvlType(self, l));
            return ret;
          })
      .def_property_readonly(
          "dim_to_lvl",
          [](MlirAttribute self) -> std::optional<MlirAffineMap> {
            MlirAffineMap ret = mlirSparseTensorEncodingAttrGetDimToLvl(self);
            if (mlirAffineMapIsNull(ret))
              return {};
            return ret;
          })
      .def_property_readonly(
          "lvl_to_dim",
          [](MlirAttribute self) -> std::optional<MlirAffineMap> {
            MlirAffineMap ret = mlirSparseTensorEncodingAttrGetLvlToDim(self);
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
