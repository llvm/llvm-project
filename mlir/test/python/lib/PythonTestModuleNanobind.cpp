//===- PythonTestModuleNanobind.cpp - PythonTest dialect extension --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the nanobind edition of the PythonTest dialect module.
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

static bool mlirTypeIsARankedIntegerTensor(MlirType t) {
  return mlirTypeIsARankedTensor(t) &&
         mlirTypeIsAInteger(mlirShapedTypeGetElementType(t));
}

NB_MODULE(_mlirPythonTestNanobind, m) {
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
      nb::arg("context"), nb::arg("load") = true,
      // clang-format off
      nb::sig("def register_python_test_dialect(context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") ", load: bool = True) -> None"));
  // clang-format on

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) {
        MlirDialectHandle pythonTestDialect =
            mlirGetDialectHandle__python_test__();
        mlirDialectHandleInsertDialect(pythonTestDialect, registry);
      },
      nb::arg("registry"),
      // clang-format off
      nb::sig("def register_dialect(registry: " MAKE_MLIR_PYTHON_QUALNAME("ir.DialectRegistry") ") -> None"));
  // clang-format on

  m.def(
      "test_diagnostics_with_errors_and_notes",
      [](MlirContext ctx) {
        mlir::python::CollectDiagnosticsToStringScope handler(ctx);
        mlirPythonTestEmitDiagnosticWithNote(ctx);
        throw nb::value_error(handler.takeMessage().c_str());
      },
      // clang-format off
      nb::sig("def test_diagnostics_with_errors_and_notes(arg: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") ", /) -> None"));
  // clang-format on

  mlir_attribute_subclass(m, "TestAttr",
                          mlirAttributeIsAPythonTestTestAttribute,
                          mlirPythonTestTestAttributeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext ctx) {
            return cls(mlirPythonTestTestAttributeGet(ctx));
          },
          // clang-format off
          nb::sig("def get(cls: object, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> object"),
          // clang-format on
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  mlir_type_subclass(m, "TestType", mlirTypeIsAPythonTestTestType,
                     mlirPythonTestTestTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext ctx) {
            return cls(mlirPythonTestTestTypeGet(ctx));
          },
          // clang-format off
          nb::sig("def get(cls: object, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> object"),
          // clang-format on
          nb::arg("cls"), nb::arg("context").none() = nb::none());

  auto typeCls =
      mlir_type_subclass(m, "TestIntegerRankedTensorType",
                         mlirTypeIsARankedIntegerTensor,
                         nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                             .attr("RankedTensorType"))
          .def_classmethod(
              "get",
              [](const nb::object &cls, std::vector<int64_t> shape,
                 unsigned width, MlirContext ctx) {
                MlirAttribute encoding = mlirAttributeGetNull();
                return cls(mlirRankedTensorTypeGet(
                    shape.size(), shape.data(), mlirIntegerTypeGet(ctx, width),
                    encoding));
              },
              // clang-format off
              nb::sig("def get(cls: object, shape: collections.abc.Sequence[int], width: int, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> object"),
              // clang-format on
              nb::arg("cls"), nb::arg("shape"), nb::arg("width"),
              nb::arg("context").none() = nb::none());

  assert(nb::hasattr(typeCls.get_class(), "static_typeid") &&
         "TestIntegerRankedTensorType has no static_typeid");

  MlirTypeID mlirRankedTensorTypeID = mlirRankedTensorTypeGetTypeID();

  nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
          mlirRankedTensorTypeID, nb::arg("replace") = true)(
          nanobind::cpp_function([typeCls](const nb::object &mlirType) {
            return typeCls.get_class()(mlirType);
          }));

  auto valueCls = mlir_value_subclass(m, "TestTensorValue",
                                      mlirTypeIsAPythonTestTestTensorValue)
                      .def("is_null", [](MlirValue &self) {
                        return mlirValueIsNull(self);
                      });

  nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
      .attr(MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR)(
          mlirRankedTensorTypeID)(
          nanobind::cpp_function([valueCls](const nb::object &valueObj) {
            std::optional<nb::object> capsule =
                mlirApiObjectToCapsule(valueObj);
            assert(capsule.has_value() && "capsule is not null");
            MlirValue v = mlirPythonCapsuleToValue(capsule.value().ptr());
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
