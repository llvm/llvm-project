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

MlirType mlirTransformAnyOpTypeGet(MlirContext ctx) {
  return wrap(transform::AnyOpType::get(unwrap(ctx)));
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
