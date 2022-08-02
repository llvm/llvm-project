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

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectSparseTensorSubmodule(const py::module &m) {
  py::enum_<MlirSparseTensorDimLevelType>(m, "DimLevelType", py::module_local())
      .value("dense", MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE)
      .value("compressed", MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED);

  mlir_attribute_subclass(m, "EncodingAttr",
                          mlirAttributeIsASparseTensorEncodingAttr)
      .def_classmethod(
          "get",
          [](py::object cls,
             std::vector<MlirSparseTensorDimLevelType> dimLevelTypes,
             llvm::Optional<MlirAffineMap> dimOrdering, int pointerBitWidth,
             int indexBitWidth, MlirContext context) {
            return cls(mlirSparseTensorEncodingAttrGet(
                context, dimLevelTypes.size(), dimLevelTypes.data(),
                dimOrdering ? *dimOrdering : MlirAffineMap{nullptr},
                pointerBitWidth, indexBitWidth));
          },
          py::arg("cls"), py::arg("dim_level_types"), py::arg("dim_ordering"),
          py::arg("pointer_bit_width"), py::arg("index_bit_width"),
          py::arg("context") = py::none(),
          "Gets a sparse_tensor.encoding from parameters.")
      .def_property_readonly(
          "dim_level_types",
          [](MlirAttribute self) {
            std::vector<MlirSparseTensorDimLevelType> ret;
            for (int i = 0,
                     e = mlirSparseTensorEncodingGetNumDimLevelTypes(self);
                 i < e; ++i)
              ret.push_back(
                  mlirSparseTensorEncodingAttrGetDimLevelType(self, i));
            return ret;
          })
      .def_property_readonly(
          "dim_ordering",
          [](MlirAttribute self) -> llvm::Optional<MlirAffineMap> {
            MlirAffineMap ret =
                mlirSparseTensorEncodingAttrGetDimOrdering(self);
            if (mlirAffineMapIsNull(ret))
              return {};
            return ret;
          })
      .def_property_readonly(
          "pointer_bit_width",
          [](MlirAttribute self) {
            return mlirSparseTensorEncodingAttrGetPointerBitWidth(self);
          })
      .def_property_readonly("index_bit_width", [](MlirAttribute self) {
        return mlirSparseTensorEncodingAttrGetIndexBitWidth(self);
      });
}

PYBIND11_MODULE(_mlirDialectsSparseTensor, m) {
  m.doc() = "MLIR SparseTensor dialect.";
  populateDialectSparseTensorSubmodule(m);
}
