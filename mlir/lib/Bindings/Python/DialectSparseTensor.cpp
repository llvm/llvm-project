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

struct EncodingAttr : PyConcreteAttribute<EncodingAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsASparseTensorEncodingAttr;
  static constexpr const char *pyClassName = "EncodingAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirSparseTensorLevelType> lvlTypes,
           std::optional<MlirAffineMap> dimToLvl,
           std::optional<MlirAffineMap> lvlToDim, int posWidth, int crdWidth,
           std::optional<MlirAttribute> explicitVal,
           std::optional<MlirAttribute> implicitVal,
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
        [](MlirSparseTensorLevelFormat lvlFmt,
           const std::vector<MlirSparseTensorLevelPropertyNondefault>
               &properties,
           unsigned n, unsigned m) {
          return mlirSparseTensorEncodingAttrBuildLvlType(
              lvlFmt, properties.data(), properties.size(), n, m);
        },
        nb::arg("lvl_fmt"),
        nb::arg("properties") =
            std::vector<MlirSparseTensorLevelPropertyNondefault>(),
        nb::arg("n") = 0, nb::arg("m") = 0,
        "Builds a sparse_tensor.encoding.level_type from parameters.");

    c.def_prop_ro("lvl_types", [](MlirAttribute self) {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      std::vector<MlirSparseTensorLevelType> ret;
      ret.reserve(lvlRank);
      for (int l = 0; l < lvlRank; ++l)
        ret.push_back(mlirSparseTensorEncodingAttrGetLvlType(self, l));
      return ret;
    });

    c.def_prop_ro(
        "dim_to_lvl", [](MlirAttribute self) -> std::optional<MlirAffineMap> {
          MlirAffineMap ret = mlirSparseTensorEncodingAttrGetDimToLvl(self);
          if (mlirAffineMapIsNull(ret))
            return {};
          return ret;
        });

    c.def_prop_ro(
        "lvl_to_dim", [](MlirAttribute self) -> std::optional<MlirAffineMap> {
          MlirAffineMap ret = mlirSparseTensorEncodingAttrGetLvlToDim(self);
          if (mlirAffineMapIsNull(ret))
            return {};
          return ret;
        });

    c.def_prop_ro("pos_width", mlirSparseTensorEncodingAttrGetPosWidth);
    c.def_prop_ro("crd_width", mlirSparseTensorEncodingAttrGetCrdWidth);

    c.def_prop_ro(
        "explicit_val", [](MlirAttribute self) -> std::optional<MlirAttribute> {
          MlirAttribute ret = mlirSparseTensorEncodingAttrGetExplicitVal(self);
          if (mlirAttributeIsNull(ret))
            return {};
          return ret;
        });

    c.def_prop_ro(
        "implicit_val", [](MlirAttribute self) -> std::optional<MlirAttribute> {
          MlirAttribute ret = mlirSparseTensorEncodingAttrGetImplicitVal(self);
          if (mlirAttributeIsNull(ret))
            return {};
          return ret;
        });

    c.def_prop_ro("structured_n", [](MlirAttribute self) -> unsigned {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      return mlirSparseTensorEncodingAttrGetStructuredN(
          mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("structured_m", [](MlirAttribute self) -> unsigned {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      return mlirSparseTensorEncodingAttrGetStructuredM(
          mlirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("lvl_formats_enum", [](MlirAttribute self) {
      const int lvlRank = mlirSparseTensorEncodingGetLvlRank(self);
      std::vector<MlirSparseTensorLevelFormat> ret;
      ret.reserve(lvlRank);
      for (int l = 0; l < lvlRank; l++)
        ret.push_back(mlirSparseTensorEncodingAttrGetLvlFmt(self, l));
      return ret;
    });
  }
};

static void populateDialectSparseTensorSubmodule(nb::module_ &m) {
  nb::enum_<MlirSparseTensorLevelFormat>(m, "LevelFormat", nb::is_arithmetic(),
                                         nb::is_flag())
      .value("dense", MLIR_SPARSE_TENSOR_LEVEL_DENSE)
      .value("n_out_of_m", MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M)
      .value("compressed", MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED)
      .value("singleton", MLIR_SPARSE_TENSOR_LEVEL_SINGLETON)
      .value("loose_compressed", MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED);

  nb::enum_<MlirSparseTensorLevelPropertyNondefault>(m, "LevelProperty")
      .value("non_ordered", MLIR_SPARSE_PROPERTY_NON_ORDERED)
      .value("non_unique", MLIR_SPARSE_PROPERTY_NON_UNIQUE)
      .value("soa", MLIR_SPARSE_PROPERTY_SOA);

  EncodingAttr::bind(m);
}
} // namespace sparse_tensor
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  mlir::python::mlir::sparse_tensor::populateDialectSparseTensorSubmodule(m);
}
