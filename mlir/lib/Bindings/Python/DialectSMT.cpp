//===- DialectSMT.cpp - Pybind module for SMT dialect API support ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindUtils.h"

#include "mlir-c/Dialect/SMT.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Target/ExportSMTLIB.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace smt {
struct BoolType : PyConcreteType<BoolType> {
  static constexpr IsAFunctionTy isaFunction = mlirSMTTypeIsABool;
  static constexpr const char *pyClassName = "BoolType";
  static inline const MlirStringRef name = mlirSMTBoolTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return BoolType(context->getRef(),
                          mlirSMTTypeGetBool(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct BitVectorType : PyConcreteType<BitVectorType> {
  static constexpr IsAFunctionTy isaFunction = mlirSMTTypeIsABitVector;
  static constexpr const char *pyClassName = "BitVectorType";
  static inline const MlirStringRef name = mlirSMTBitVectorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t width, DefaultingPyMlirContext context) {
          return BitVectorType(
              context->getRef(),
              mlirSMTTypeGetBitVector(context.get()->get(), width));
        },
        nb::arg("width"), nb::arg("context").none() = nb::none());
  }
};

struct IntType : PyConcreteType<IntType> {
  static constexpr IsAFunctionTy isaFunction = mlirSMTTypeIsAInt;
  static constexpr const char *pyClassName = "IntType";
  static inline const MlirStringRef name = mlirSMTIntTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return IntType(context->getRef(),
                         mlirSMTTypeGetInt(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectSMTSubmodule(nanobind::module_ &m) {
  BoolType::bind(m);
  BitVectorType::bind(m);
  IntType::bind(m);

  auto exportSMTLIB = [](MlirOperation module, bool inlineSingleUseValues,
                         bool indentLetBody) {
    CollectDiagnosticsToStringScope scope(mlirOperationGetContext(module));
    PyPrintAccumulator printAccum;
    MlirLogicalResult result = mlirTranslateOperationToSMTLIB(
        module, printAccum.getCallback(), printAccum.getUserData(),
        inlineSingleUseValues, indentLetBody);
    if (mlirLogicalResultIsSuccess(result))
      return printAccum.join();
    throw nb::value_error(
        ("Failed to export smtlib.\nDiagnostic message " + scope.takeMessage())
            .c_str());
  };

  m.def(
      "export_smtlib",
      [&exportSMTLIB](const PyOperation &module, bool inlineSingleUseValues,
                      bool indentLetBody) {
        return exportSMTLIB(module, inlineSingleUseValues, indentLetBody);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false);
  m.def(
      "export_smtlib",
      [&exportSMTLIB](PyModule &module, bool inlineSingleUseValues,
                      bool indentLetBody) {
        return exportSMTLIB(mlirModuleGetOperation(module.get()),
                            inlineSingleUseValues, indentLetBody);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false);
}
} // namespace smt
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsSMT, m) {
  m.doc() = "MLIR SMT Dialect";

  python::MLIR_BINDINGS_PYTHON_DOMAIN::smt::populateDialectSMTSubmodule(m);
}
