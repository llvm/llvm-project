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
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace transform {
//===-------------------------------------------------------------------===//
// AnyOpType
//===-------------------------------------------------------------------===//

struct AnyOpType : PyConcreteType<AnyOpType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyOpType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyOpTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyOpType";
  static inline const MlirStringRef name = mlirTransformAnyOpTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyOpType(context->getRef(),
                           mlirTransformAnyOpTypeGet(context.get()->get()));
        },
        "Get an instance of AnyOpType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyParamType
//===-------------------------------------------------------------------===//

struct AnyParamType : PyConcreteType<AnyParamType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyParamTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyParamType";
  static inline const MlirStringRef name = mlirTransformAnyParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyParamType(context->getRef(), mlirTransformAnyParamTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyParamType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyValueType
//===-------------------------------------------------------------------===//

struct AnyValueType : PyConcreteType<AnyValueType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformAnyValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformAnyValueTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyValueType";
  static inline const MlirStringRef name = mlirTransformAnyValueTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyValueType(context->getRef(), mlirTransformAnyValueTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyValueType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// OperationType
//===-------------------------------------------------------------------===//

struct OperationType : PyConcreteType<OperationType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsATransformOperationType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformOperationTypeGetTypeID;
  static constexpr const char *pyClassName = "OperationType";
  static inline const MlirStringRef name = mlirTransformOperationTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &operationName, DefaultingPyMlirContext context) {
          MlirStringRef cOperationName =
              mlirStringRefCreate(operationName.data(), operationName.size());
          return OperationType(context->getRef(),
                               mlirTransformOperationTypeGet(
                                   context.get()->get(), cOperationName));
        },
        "Get an instance of OperationType for the given kind in the given "
        "context",
        nb::arg("operation_name"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "operation_name",
        [](const OperationType &type) {
          MlirStringRef operationName =
              mlirTransformOperationTypeGetOperationName(type);
          return nb::str(operationName.data, operationName.length);
        },
        "Get the name of the payload operation accepted by the handle.");
  }
};

//===-------------------------------------------------------------------===//
// ParamType
//===-------------------------------------------------------------------===//

struct ParamType : PyConcreteType<ParamType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATransformParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTransformParamTypeGetTypeID;
  static constexpr const char *pyClassName = "ParamType";
  static inline const MlirStringRef name = mlirTransformParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &type, DefaultingPyMlirContext context) {
          return ParamType(context->getRef(), mlirTransformParamTypeGet(
                                                  context.get()->get(), type));
        },
        "Get an instance of ParamType for the given type in the given context.",
        nb::arg("type"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "type",
        [](ParamType type) {
          return PyType(type.getContext(), mlirTransformParamTypeGetType(type))
              .maybeDownCast();
        },
        "Get the type this ParamType is associated with.");
  }
};

static void populateDialectTransformSubmodule(nb::module_ &m) {
  AnyOpType::bind(m);
  AnyParamType::bind(m);
  AnyValueType::bind(m);
  OperationType::bind(m);
  ParamType::bind(m);
}
} // namespace transform
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsTransform, m) {
  m.doc() = "MLIR Transform dialect.";
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::transform::
      populateDialectTransformSubmodule(m);
}
