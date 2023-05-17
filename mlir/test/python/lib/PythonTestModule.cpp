//===- PythonTestModule.cpp - Python extension for the PythonTest dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

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
  mlir_type_subclass(m, "TestType", mlirTypeIsAPythonTestTestType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(mlirPythonTestTestTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());
  mlir_value_subclass(m, "TestTensorValue",
                      mlirTypeIsAPythonTestTestTensorValue)
      .def("is_null", [](MlirValue &self) { return mlirValueIsNull(self); });
}
