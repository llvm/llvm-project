//===- DialectSparseTensor.cpp - 'sparse_tensor' dialect submodule --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <vector>

#include "mlir-c/AffineMap.h"
#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace sparse_tensor {

enum class PySparseTensorLevelFormat : std::underlying_type_t<
    MlirSparseTensorLevelFormat> {
  DENSE = MLIR_SPARSE_TENSOR_LEVEL_DENSE,
  N_OUT_OF_M = MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M,
  COMPRESSED = MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED,
  SINGLETON = MLIR_SPARSE_TENSOR_LEVEL_SINGLETON,
  LOOSE_COMPRESSED = MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED
};

enum class PySparseTensorLevelPropertyNondefault : std::underlying_type_t<
    MlirSparseTensorLevelPropertyNondefault> {
  NON_ORDERED = MLIR_SPARSE_PROPERTY_NON_ORDERED,
  NON_UNIQUE = MLIR_SPARSE_PROPERTY_NON_UNIQUE,
  SOA = MLIR_SPARSE_PROPERTY_SOA,
};

struct EncodingAttr : PyConcreteAttribute<EncodingAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsASparseTensorEncodingAttr;
  static constexpr const char *pyClassName = "EncodingAttr";
  static inline const MlirStringRef name =
      mlirSparseTensorEncodingAttrGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirSparseTensorLevelType> lvlTypes,
           std::optional<PyAffineMap> dimToLvl,
           std::optional<PyAffineMap> lvlToDim, int posWidth, int crdWidth,
           std::optional<PyAttribute> explicitVal,
           std::optional<PyAttribute> implicitVal,
           DefaultingPyMlirContext context) {
          return EncodingAttr(
              context->getRef(),
              mlirSparseTensorEncodingAttrGet(
                  context.get()->get(), lvlTypes.size(), lvlTypes.data(),
                  dimToLvl ? *dimToLvl : MlirAffineMap{nullptr},
                  lvlToDim ? *lvlToDim : MlirAffineMap{nullptr}, posWidth,
                  crdWidth, explicitVal ? *explicitVal : MlirAttribute{nullptr},
                  implicitVal ? *implicitVal : MlirAttribute{nullptr}));
        },
        nb::arg("lvl_types"), nb::arg("dim_to_lvl").none(),
        nb::arg("lvl_to_dim").none(), nb::arg("pos_width"),
        nb::arg("crd_width"), nb::arg("explicit_val") = nb::none(),
        nb::arg("implicit_val") = nb::none(), nb::arg("context") = nb::none(),
        "Gets a sparse_tensor.encoding from parameters.");

    c.def_static(
        "build_level_type",
        [](PySparseTensorLevelFormat lvlFmt,
           const std::vector<PySparseTensorLevelPropertyNondefault> &properties,
           unsigned n, unsigned m) {
          std::vector<MlirSparseTensorLevelPropertyNondefault> props;
          props.reserve(properties.size());
          for (auto prop : properties) {
            props.push_back(
                static_cast<MlirSparseTensorLevelPropertyNondefault>(prop));
          }
          return mlirSparseTensorEncodingAttrBuildLvlType(
              static_cast<MlirSparseTensorLevelFormat>(lvlFmt), props.data(),
              props.size(), n, m);
        },
        nb::arg("lvl_fmt"),
        nb::arg("properties") =
            std::vector<PySparseTensorLevelPropertyNondefault>(),
        nb::arg("n") = 0, nb::arg("m") = 0,
        "Builds a sparse_tensor.encoding.level_type from parameters.");

    c.def_prop_ro("lvl_types", [](const EncodingAttr &self) {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      std::vector<MlirSparseTensorLevelType> ret;
      ret.reserve(lvlRank);
      for (int l = 0; l < lvlRank; ++l)
        ret.push_back(mlirSparseTensorEncodingAttrGetLvlType(self, l));
      return ret;
    });

    c.def_prop_ro(
        "dim_to_lvl", [](EncodingAttr &self) -> std::optional<PyAffineMap> {
          MlirAffineMap ret = mlirSparseTensorEncodingAttrGetDimToLvl(self);
          if (mlirAffineMapIsNull(ret))
            return {};
          return PyAffineMap(self.getContext(), ret);
        });

    c.def_prop_ro(
        "lvl_to_dim", [](EncodingAttr &self) -> std::optional<PyAffineMap> {
          MlirAffineMap ret = mlirSparseTensorEncodingAttrGetLvlToDim(self);
          if (mlirAffineMapIsNull(ret))
            return {};
          return PyAffineMap(self.getContext(), ret);
        });

    c.def_prop_ro("pos_width", mlirSparseTensorEncodingAttrGetPosWidth);
    c.def_prop_ro("crd_width", mlirSparseTensorEncodingAttrGetCrdWidth);

    c.def_prop_ro("explicit_val",
                  [](EncodingAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    MlirAttribute ret =
                        mlirSparseTensorEncodingAttrGetExplicitVal(self);
                    if (mlirAttributeIsNull(ret))
                      return {};
                    return PyAttribute(self.getContext(), ret).maybeDownCast();
                  });

    c.def_prop_ro("implicit_val",
                  [](EncodingAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    MlirAttribute ret =
                        mlirSparseTensorEncodingAttrGetImplicitVal(self);
                    if (mlirAttributeIsNull(ret))
                      return {};
                    return PyAttribute(self.getContext(), ret).maybeDownCast();
                  });

    c.def_prop_ro("structured_n", [](const EncodingAttr &self) -> unsigned {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      return mlirSparseTensorEncodingAttrGetStructuredN(
          mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("structured_m", [](const EncodingAttr &self) -> unsigned {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      return mlirSparseTensorEncodingAttrGetStructuredM(
          mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("lvl_formats_enum", [](const EncodingAttr &self) {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      std::vector<PySparseTensorLevelFormat> ret;
      ret.reserve(lvlRank);

      for (int l = 0; l < lvlRank; l++)
        ret.push_back(static_cast<PySparseTensorLevelFormat>(
            mlirSparseTensorEncodingAttrGetLvlFmt(self, l)));
      return ret;
    });
  }
};

static void populateDialectSparseTensorSubmodule(nb::module_ &m) {
  nb::enum_<PySparseTensorLevelFormat>(m, "LevelFormat", nb::is_arithmetic(),
                                       nb::is_flag())
      .value("dense", PySparseTensorLevelFormat::DENSE)
      .value("n_out_of_m", PySparseTensorLevelFormat::N_OUT_OF_M)
      .value("compressed", PySparseTensorLevelFormat::COMPRESSED)
      .value("singleton", PySparseTensorLevelFormat::SINGLETON)
      .value("loose_compressed", PySparseTensorLevelFormat::LOOSE_COMPRESSED);
  nb::enum_<PySparseTensorLevelPropertyNondefault>(m, "LevelProperty")
      .value("non_ordered", PySparseTensorLevelPropertyNondefault::NON_ORDERED)
      .value("non_unique", PySparseTensorLevelPropertyNondefault::NON_UNIQUE)
      .value("soa", PySparseTensorLevelPropertyNondefault::SOA);

  EncodingAttr::bind(m);
}
} // namespace sparse_tensor
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::sparse_tensor::
      populateDialectSparseTensorSubmodule(m);
}
