//===- DialectPDL.cpp - 'pdl' dialect submodule ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/PDL.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

void populateDialectPDLSubmodule(const nanobind::module_ &m) {
  //===-------------------------------------------------------------------===//
  // PDLType
  //===-------------------------------------------------------------------===//

  auto pdlType = mlir_type_subclass(m, "PDLType", mlirTypeIsAPDLType);

  //===-------------------------------------------------------------------===//
  // AttributeType
  //===-------------------------------------------------------------------===//

  auto attributeType =
      mlir_type_subclass(m, "AttributeType", mlirTypeIsAPDLAttributeType);
  attributeType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirPDLAttributeTypeGet(ctx));
      },
      "Get an instance of AttributeType in given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // OperationType
  //===-------------------------------------------------------------------===//

  auto operationType =
      mlir_type_subclass(m, "OperationType", mlirTypeIsAPDLOperationType);
  operationType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirPDLOperationTypeGet(ctx));
      },
      "Get an instance of OperationType in given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // RangeType
  //===-------------------------------------------------------------------===//

  auto rangeType = mlir_type_subclass(m, "RangeType", mlirTypeIsAPDLRangeType);
  rangeType.def_classmethod(
      "get",
      [](nb::object cls, MlirType elementType) {
        return cls(mlirPDLRangeTypeGet(elementType));
      },
      "Gets an instance of RangeType in the same context as the provided "
      "element type.",
      nb::arg("cls"), nb::arg("element_type"));
  rangeType.def_property_readonly(
      "element_type",
      [](MlirType type) { return mlirPDLRangeTypeGetElementType(type); },
      "Get the element type.");

  //===-------------------------------------------------------------------===//
  // TypeType
  //===-------------------------------------------------------------------===//

  auto typeType = mlir_type_subclass(m, "TypeType", mlirTypeIsAPDLTypeType);
  typeType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirPDLTypeTypeGet(ctx));
      },
      "Get an instance of TypeType in given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // ValueType
  //===-------------------------------------------------------------------===//

  auto valueType = mlir_type_subclass(m, "ValueType", mlirTypeIsAPDLValueType);
  valueType.def_classmethod(
      "get",
      [](nb::object cls, MlirContext ctx) {
        return cls(mlirPDLValueTypeGet(ctx));
      },
      "Get an instance of TypeType in given context.", nb::arg("cls"),
      nb::arg("context").none() = nb::none());
}

NB_MODULE(_mlirDialectsPDL, m) {
  m.doc() = "MLIR PDL dialect.";
  populateDialectPDLSubmodule(m);
}
