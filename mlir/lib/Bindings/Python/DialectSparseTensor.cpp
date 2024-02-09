//===- DialectSparseTensor.cpp - 'sparse_tensor' dialect submodule --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/AffineMap.h"
#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <optional>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectSparseTensorSubmodule(const py::module &m) {
  py::enum_<MlirBaseSparseTensorLevelType>(m, "LevelType", py::module_local())
      .value("dense", MLIR_SPARSE_TENSOR_LEVEL_DENSE)
      .value("n_out_of_m", MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M)
      .value("compressed", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED)
      .value("compressed_nu", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU)
      .value("compressed_no", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NO)
      .value("compressed_nu_no", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU_NO)
      .value("singleton", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON)
      .value("singleton_nu", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU)
      .value("singleton_no", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NO)
      .value("singleton_nu_no", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU_NO)
      .value("loose_compressed", MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED)
      .value("loose_compressed_nu",
             MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU)
      .value("loose_compressed_no",
             MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NO)
      .value("loose_compressed_nu_no",
             MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU_NO);

  mlir_attribute_subclass(m, "EncodingAttr",
                          mlirAttributeIsASparseTensorEncodingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<MlirSparseTensorLevelType> lvlTypes,
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
      .def_classmethod(
          "build_level_type",
          [](py::object cls, MlirBaseSparseTensorLevelType lvlType, unsigned n,
             unsigned m) {
            return mlirSparseTensorEncodingAttrBuildLvlType(lvlType, n, m);
          },
          py::arg("cls"), py::arg("lvl_type"), py::arg("n") = 0,
          py::arg("m") = 0,
          "Builds a sparse_tensor.encoding.level_type from parameters.")
      .def_property_readonly(
          "lvl_types",
          [](MlirAttribute self) {
            const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
            std::vector<MlirSparseTensorLevelType> ret;
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
                             mlirSparseTensorEncodingAttrGetCrdWidth)
      .def_property_readonly(
          "structured_n",
          [](MlirAttribute self) -> unsigned {
            const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
            return mlirSparseTensorEncodingAttrGetStructuredN(
                mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
          })
      .def_property_readonly(
          "structured_m",
          [](MlirAttribute self) -> unsigned {
            const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
            return mlirSparseTensorEncodingAttrGetStructuredM(
                mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
          })
      .def_property_readonly("lvl_types_enum", [](MlirAttribute self) {
        const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
        std::vector<MlirBaseSparseTensorLevelType> ret;
        ret.reserve(lvlRank);
        for (int l = 0; l < lvlRank; l++) {
          // Convert level type to 32 bits to ignore n and m for n_out_of_m
          // format.
          ret.push_back(
              static_cast<MlirBaseSparseTensorLevelType>(static_cast<uint32_t>(
                  mlirSparseTensorEncodingAttrGetLvlType(self, l))));
        }
        return ret;
      });
}

PYBIND11_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  populateDialectSparseTensorSubmodule(m);
}
