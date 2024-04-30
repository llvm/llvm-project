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
  py::enum_<MlirSparseTensorLevelFormat>(m, "LevelFormat", py::module_local())
      .value("dense", MLIR_SPARSE_TENSOR_LEVEL_DENSE)
      .value("n_out_of_m", MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M)
      .value("compressed", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED)
      .value("singleton", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON)
      .value("loose_compressed", MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED);

  py::enum_<MlirSparseTensorLevelPropertyNondefault>(m, "LevelProperty",
                                                     py::module_local())
      .value("non_ordered", MLIR_SPARSE_PROPERTY_NON_ORDERED)
      .value("non_unique", MLIR_SPARSE_PROPERTY_NON_UNIQUE);

  mlir_attribute_subclass(m, "EncodingAttr",
                          mlirAttributeIsASparseTensorEncodingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<MlirSparseTensorLevelType> lvlTypes,
             std::optional<MlirAffineMap> dimToLvl,
             std::optional<MlirAffineMap> lvlToDim, int posWidth, int crdWidth,
             std::optional<MlirAttribute> explicitVal,
             std::optional<MlirAttribute> implicitVal, MlirContext context) {
            return cls(mlirSparseTensorEncodingAttrGet(
                context, lvlTypes.size(), lvlTypes.data(),
                dimToLvl ? *dimToLvl : MlirAffineMap{nullptr},
                lvlToDim ? *lvlToDim : MlirAffineMap{nullptr}, posWidth,
                crdWidth, explicitVal ? *explicitVal : MlirAttribute{nullptr},
                implicitVal ? *implicitVal : MlirAttribute{nullptr}));
          },
          py::arg("cls"), py::arg("lvl_types"), py::arg("dim_to_lvl"),
          py::arg("lvl_to_dim"), py::arg("pos_width"), py::arg("crd_width"),
          py::arg("explicit_val") = py::none(),
          py::arg("implicit_val") = py::none(), py::arg("context") = py::none(),
          "Gets a sparse_tensor.encoding from parameters.")
      .def_classmethod(
          "build_level_type",
          [](py::object cls, MlirSparseTensorLevelFormat lvlFmt,
             const std::vector<MlirSparseTensorLevelPropertyNondefault>
                 &properties,
             unsigned n, unsigned m) {
            return mlirSparseTensorEncodingAttrBuildLvlType(
                lvlFmt, properties.data(), properties.size(), n, m);
          },
          py::arg("cls"), py::arg("lvl_fmt"),
          py::arg("properties") =
              std::vector<MlirSparseTensorLevelPropertyNondefault>(),
          py::arg("n") = 0, py::arg("m") = 0,
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
          "explicit_val",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            MlirAttribute ret =
                mlirSparseTensorEncodingAttrGetExplicitVal(self);
            if (mlirAttributeIsNull(ret))
              return {};
            return ret;
          })
      .def_property_readonly(
          "implicit_val",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            MlirAttribute ret =
                mlirSparseTensorEncodingAttrGetImplicitVal(self);
            if (mlirAttributeIsNull(ret))
              return {};
            return ret;
          })
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
      .def_property_readonly("lvl_formats_enum", [](MlirAttribute self) {
        const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
        std::vector<MlirSparseTensorLevelFormat> ret;
        ret.reserve(lvlRank);
        for (int l = 0; l < lvlRank; l++)
          ret.push_back(mlirSparseTensorEncodingAttrGetLvlFmt(self, l));
        return ret;
      });
}

PYBIND11_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  populateDialectSparseTensorSubmodule(m);
}
