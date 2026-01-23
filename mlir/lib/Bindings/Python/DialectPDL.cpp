//===- DialectPDL.cpp - 'pdl' dialect submodule ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/PDL.h"
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
namespace pdl {

//===-------------------------------------------------------------------===//
// PDLType
//===-------------------------------------------------------------------===//

struct PDLType : PyConcreteType<PDLType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLType;
  static constexpr const char *pyClassName = "PDLType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {}
};

//===-------------------------------------------------------------------===//
// AttributeType
//===-------------------------------------------------------------------===//

struct AttributeType : PyConcreteType<AttributeType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLAttributeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPDLAttributeTypeGetTypeID;
  static constexpr const char *pyClassName = "AttributeType";
  static inline const MlirStringRef name = mlirPDLAttributeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AttributeType(context->getRef(),
                               mlirPDLAttributeTypeGet(context.get()->get()));
        },
        "Get an instance of AttributeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// OperationType
//===-------------------------------------------------------------------===//

struct OperationType : PyConcreteType<OperationType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLOperationType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPDLOperationTypeGetTypeID;
  static constexpr const char *pyClassName = "OperationType";
  static inline const MlirStringRef name = mlirPDLOperationTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return OperationType(context->getRef(),
                               mlirPDLOperationTypeGet(context.get()->get()));
        },
        "Get an instance of OperationType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// RangeType
//===-------------------------------------------------------------------===//

struct RangeType : PyConcreteType<RangeType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLRangeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPDLRangeTypeGetTypeID;
  static constexpr const char *pyClassName = "RangeType";
  static inline const MlirStringRef name = mlirPDLRangeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, DefaultingPyMlirContext context) {
          return RangeType(context->getRef(), mlirPDLRangeTypeGet(elementType));
        },
        "Gets an instance of RangeType in the same context as the provided "
        "element type.",
        nb::arg("element_type"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "element_type",
        [](RangeType &type) {
          return PyType(type.getContext(), mlirPDLRangeTypeGetElementType(type))
              .maybeDownCast();
        },
        "Get the element type.");
  }
};

//===-------------------------------------------------------------------===//
// TypeType
//===-------------------------------------------------------------------===//

struct TypeType : PyConcreteType<TypeType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLTypeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPDLTypeTypeGetTypeID;
  static constexpr const char *pyClassName = "TypeType";
  static inline const MlirStringRef name = mlirPDLTypeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return TypeType(context->getRef(),
                          mlirPDLTypeTypeGet(context.get()->get()));
        },
        "Get an instance of TypeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// ValueType
//===-------------------------------------------------------------------===//

struct ValueType : PyConcreteType<ValueType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAPDLValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirPDLValueTypeGetTypeID;
  static constexpr const char *pyClassName = "ValueType";
  static inline const MlirStringRef name = mlirPDLValueTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return ValueType(context->getRef(),
                           mlirPDLValueTypeGet(context.get()->get()));
        },
        "Get an instance of TypeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectPDLSubmodule(nanobind::module_ &m) {
  PDLType::bind(m);
  AttributeType::bind(m);
  OperationType::bind(m);
  RangeType::bind(m);
  TypeType::bind(m);
  ValueType::bind(m);
}
} // namespace pdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsPDL, m) {
  m.doc() = "MLIR PDL dialect.";
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::pdl::populateDialectPDLSubmodule(
      m);
}
