//===- DialectTransform.cpp - 'transform' dialect submodule ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

void populateDialectTransformSubmodule(const nb::module_ &m) {
  //===-------------------------------------------------------------------===//
  // AnyOpType
  //===-------------------------------------------------------------------===//

  auto anyOpType =
      mlir_type_subclass(m, "AnyOpType", mlirTypeIsATransformAnyOpType,
                         mlirTransformAnyOpTypeGetTypeID);
  anyOpType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyOpTypeGet(ctx));
      },
      "Get an instance of AnyOpType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // AnyParamType
  //===-------------------------------------------------------------------===//

  auto anyParamType =
      mlir_type_subclass(m, "AnyParamType", mlirTypeIsATransformAnyParamType,
                         mlirTransformAnyParamTypeGetTypeID);
  anyParamType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyParamTypeGet(ctx));
      },
      "Get an instance of AnyParamType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // AnyValueType
  //===-------------------------------------------------------------------===//

  auto anyValueType =
      mlir_type_subclass(m, "AnyValueType", mlirTypeIsATransformAnyValueType,
                         mlirTransformAnyValueTypeGetTypeID);
  anyValueType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirTransformAnyValueTypeGet(ctx));
      },
      "Get an instance of AnyValueType in the given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // OperationType
  //===-------------------------------------------------------------------===//

  auto operationType =
      mlir_type_subclass(m, "OperationType", mlirTypeIsATransformOperationType,
                         mlirTransformOperationTypeGetTypeID);
  operationType.def_classmethod(
      "get",
      [](nb::object cls, const std::string &operationName, MlirContext ctx) {
        MlirStringRef cOperationName =
            mlirStringRefCreate(operationName.data(), operationName.size());
        return cls(mlirTransformOperationTypeGet(ctx, cOperationName));
      },
      "Get an instance of OperationType for the given kind in the given "
      "context",
      nb::arg("cls"), nb::arg("operation_name"),
      nb::arg("context").none() = nb::none());
  operationType.def_property_readonly(
      "operation_name",
      [](MlirType type) {
        MlirStringRef operationName =
            mlirTransformOperationTypeGetOperationName(type);
        return nb::str(operationName.data, operationName.length);
      },
      "Get the name of the payload operation accepted by the handle.");

  //===-------------------------------------------------------------------===//
  // ParamType
  //===-------------------------------------------------------------------===//

  auto paramType =
      mlir_type_subclass(m, "ParamType", mlirTypeIsATransformParamType,
                         mlirTransformParamTypeGetTypeID);
  paramType.def_classmethod(
      "get",
      [](nb::object cls, MlirType type, MlirContext ctx) {
        return cls(mlirTransformParamTypeGet(ctx, type));
      },
      "Get an instance of ParamType for the given type in the given context.",
      nb::arg("cls"), nb::arg("type"), nb::arg("context").none() = nb::none());
  paramType.def_property_readonly(
      "type",
      [](MlirType type) {
        MlirType paramType = mlirTransformParamTypeGetType(type);
        return paramType;
      },
      "Get the type this ParamType is associated with.");
}

NB_MODULE(_mlirDialectsTransform, m) {
  m.doc() = "MLIR Transform dialect.";
  populateDialectTransformSubmodule(m);
}
