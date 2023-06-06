//===- PythonTestModule.cpp - Python extension for the PythonTest dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;
using namespace pybind11::literals;

static bool mlirTypeIsARankedIntegerTensor(MlirType t) {
  return mlirTypeIsARankedTensor(t) &&
         mlirTypeIsAInteger(mlirShapedTypeGetElementType(t));
}

PYBIND11_MODULE(_mlirPythonTest, m) {
  m.def(
      "register_python_test_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle pythonTestDialect =
            mlirGetDialectHandle__python_test__();
        mlirDialectHandleRegisterDialect(pythonTestDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(pythonTestDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  mlir_attribute_subclass(m, "TestAttr",
                          mlirAttributeIsAPythonTestTestAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(mlirPythonTestTestAttributeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());
  mlir_type_subclass(m, "TestType", mlirTypeIsAPythonTestTestType,
                     mlirPythonTestTestTypeGetTypeID)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(mlirPythonTestTestTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());
  auto cls =
      mlir_type_subclass(m, "TestIntegerRankedTensorType",
                         mlirTypeIsARankedIntegerTensor,
                         py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                             .attr("RankedTensorType"))
          .def_classmethod(
              "get",
              [](const py::object &cls, std::vector<int64_t> shape,
                 unsigned width, MlirContext ctx) {
                MlirAttribute encoding = mlirAttributeGetNull();
                return cls(mlirRankedTensorTypeGet(
                    shape.size(), shape.data(), mlirIntegerTypeGet(ctx, width),
                    encoding));
              },
              "cls"_a, "shape"_a, "width"_a, "context"_a = py::none());
  assert(py::hasattr(cls.get_class(), "static_typeid") &&
         "TestIntegerRankedTensorType has no static_typeid");
  MlirTypeID mlirTypeID = mlirRankedTensorTypeGetTypeID();
  py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
          mlirTypeID, pybind11::cpp_function([cls](const py::object &mlirType) {
            return cls.get_class()(mlirType);
          }),
          /*replace=*/true);
  mlir_value_subclass(m, "TestTensorValue",
                      mlirTypeIsAPythonTestTestTensorValue)
      .def("is_null", [](MlirValue &self) { return mlirValueIsNull(self); });
}
