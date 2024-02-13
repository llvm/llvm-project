//===- Transform.cpp - C Interface for Transform dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Transform, transform,
                                      transform::TransformDialect)

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyOpType(MlirType type) {
  return isa<transform::AnyOpType>(unwrap(type));
}

MlirTypeID mlirTransformAnyOpTypeGetTypeID(void) {
  return wrap(transform::AnyOpType::getTypeID());
}

MlirType mlirTransformAnyOpTypeGet(MlirContext ctx) {
  return wrap(transform::AnyOpType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyParamType(MlirType type) {
  return isa<transform::AnyParamType>(unwrap(type));
}

MlirTypeID mlirTransformAnyParamTypeGetTypeID(void) {
  return wrap(transform::AnyParamType::getTypeID());
}

MlirType mlirTransformAnyParamTypeGet(MlirContext ctx) {
  return wrap(transform::AnyParamType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyValueType(MlirType type) {
  return isa<transform::AnyValueType>(unwrap(type));
}

MlirTypeID mlirTransformAnyValueTypeGetTypeID(void) {
  return wrap(transform::AnyValueType::getTypeID());
}

MlirType mlirTransformAnyValueTypeGet(MlirContext ctx) {
  return wrap(transform::AnyValueType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformOperationType(MlirType type) {
  return isa<transform::OperationType>(unwrap(type));
}

MlirTypeID mlirTransformOperationTypeGetTypeID(void) {
  return wrap(transform::OperationType::getTypeID());
}

MlirType mlirTransformOperationTypeGet(MlirContext ctx,
                                       MlirStringRef operationName) {
  return wrap(
      transform::OperationType::get(unwrap(ctx), unwrap(operationName)));
}

MlirStringRef mlirTransformOperationTypeGetOperationName(MlirType type) {
  return wrap(cast<transform::OperationType>(unwrap(type)).getOperationName());
}

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformParamType(MlirType type) {
  return isa<transform::ParamType>(unwrap(type));
}

MlirTypeID mlirTransformParamTypeGetTypeID(void) {
  return wrap(transform::ParamType::getTypeID());
}

MlirType mlirTransformParamTypeGet(MlirContext ctx, MlirType type) {
  return wrap(transform::ParamType::get(unwrap(ctx), unwrap(type)));
}

MlirType mlirTransformParamTypeGetType(MlirType type) {
  return wrap(cast<transform::ParamType>(unwrap(type)).getType());
}
