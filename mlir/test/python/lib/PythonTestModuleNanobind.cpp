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
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;
namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace python_test {
static bool mlirTypeIsARankedIntegerTensor(MlirType t) {
  return mlirTypeIsARankedTensor(t) &&
         mlirTypeIsAInteger(mlirShapedTypeGetElementType(t));
}

struct PyTestType : PyConcreteType<PyTestType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPythonTestTestType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPythonTestTestTypeGetTypeID;
  static constexpr const char *pyClassName = "TestType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyTestType(context->getRef(),
                            mlirPythonTestTestTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyTestIntegerRankedTensorType
    : PyConcreteType<PyTestIntegerRankedTensorType, PyRankedTensorType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedIntegerTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TestIntegerRankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, unsigned width,
           DefaultingPyMlirContext ctx) {
          MlirAttribute encoding = mlirAttributeGetNull();
          return PyTestIntegerRankedTensorType(
              ctx->getRef(),
              mlirRankedTensorTypeGet(
                  shape.size(), shape.data(),
                  mlirIntegerTypeGet(ctx.get()->get(), width), encoding));
        },
        nb::arg("shape"), nb::arg("width"),
        nb::arg("context").none() = nb::none());
  }
};

struct PyTestTensorValue : PyConcreteValue<PyTestTensorValue> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAPythonTestTestTensorValue;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TestTensorValue";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def("is_null", [](MlirValue &self) { return mlirValueIsNull(self); });
  }
};

class PyTestAttr : public PyConcreteAttribute<PyTestAttr> {
public:
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAPythonTestTestAttribute;
  static constexpr const char *pyClassName = "TestAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPythonTestTestAttributeGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyTestAttr(context->getRef(), mlirPythonTestTestAttributeGet(
                                                   context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};
} // namespace python_test
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirPythonTestNanobind, m) {
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
  m.def(
      "register_python_test_dialect",
      [](DefaultingPyMlirContext context, bool load) {
        MlirDialectHandle pythonTestDialect =
            mlirGetDialectHandle__python_test__();
        mlirDialectHandleRegisterDialect(pythonTestDialect,
                                         context.get()->get());
        if (load) {
          mlirDialectHandleLoadDialect(pythonTestDialect, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

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
      [](DefaultingPyMlirContext ctx) {
        mlir::python::CollectDiagnosticsToStringScope handler(ctx.get()->get());
        mlirPythonTestEmitDiagnosticWithNote(ctx.get()->get());
        throw nb::value_error(handler.takeMessage().c_str());
      },
      nb::arg("context").none() = nb::none());

  using namespace python_test;
  PyTestAttr::bind(m);
  PyTestType::bind(m);
  PyTestIntegerRankedTensorType::bind(m);
  PyTestTensorValue::bind(m);
}
