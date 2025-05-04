//===- PythonTestModule.cpp - Python extension for the PythonTest dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the pybind11 edition of the PythonTest dialect module.
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

PYBIND11_MODULE(_mlirPythonTestPybind11, m) {
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

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) {
        MlirDialectHandle pythonTestDialect =
            mlirGetDialectHandle__python_test__();
        mlirDialectHandleInsertDialect(pythonTestDialect, registry);
      },
      py::arg("registry"));

  mlir_attribute_subclass(m, "TestAttr",
                          mlirAttributeIsAPythonTestTestAttribute,
                          mlirPythonTestTestAttributeGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirContext ctx) {
            return cls(mlirPythonTestTestAttributeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());

  mlir_type_subclass(m, "TestType", mlirTypeIsAPythonTestTestType,
                     mlirPythonTestTestTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirContext ctx) {
            return cls(mlirPythonTestTestTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());

  auto typeCls =
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

  assert(py::hasattr(typeCls.get_class(), "static_typeid") &&
         "TestIntegerRankedTensorType has no static_typeid");

  MlirTypeID mlirRankedTensorTypeID = mlirRankedTensorTypeGetTypeID();

  py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(mlirRankedTensorTypeID,
                                                        "replace"_a = true)(
          pybind11::cpp_function([typeCls](const py::object &mlirType) {
            return typeCls.get_class()(mlirType);
          }));

  auto valueCls = mlir_value_subclass(m, "TestTensorValue",
                                      mlirTypeIsAPythonTestTestTensorValue)
                      .def("is_null", [](MlirValue &self) {
                        return mlirValueIsNull(self);
                      });

  py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr(MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR)(
          mlirRankedTensorTypeID)(
          pybind11::cpp_function([valueCls](const py::object &valueObj) {
            py::object capsule = mlirApiObjectToCapsule(valueObj);
            MlirValue v = mlirPythonCapsuleToValue(capsule.ptr());
            MlirType t = mlirValueGetType(v);
            // This is hyper-specific in order to exercise/test registering a
            // value caster from cpp (but only for a single test case; see
            // testTensorValue python_test.py).
            if (mlirShapedTypeHasStaticShape(t) &&
                mlirShapedTypeGetDimSize(t, 0) == 1 &&
                mlirShapedTypeGetDimSize(t, 1) == 2 &&
                mlirShapedTypeGetDimSize(t, 2) == 3)
              return valueCls.get_class()(valueObj);
            return valueObj;
          }));
}
