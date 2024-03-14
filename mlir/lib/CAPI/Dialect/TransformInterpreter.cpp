//===- TransformTransforms.cpp - C Interface for Transform dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C interface to transforms for the transform dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Transform/Interpreter.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirTransformOptions, transform::TransformOptions)

extern "C" {

MlirTransformOptions mlirTransformOptionsCreate() {
  return wrap(new transform::TransformOptions);
}

void mlirTransformOptionsEnableExpensiveChecks(
    MlirTransformOptions transformOptions, bool enable) {
  unwrap(transformOptions)->enableExpensiveChecks(enable);
}

bool mlirTransformOptionsGetExpensiveChecksEnabled(
    MlirTransformOptions transformOptions) {
  return unwrap(transformOptions)->getExpensiveChecksEnabled();
}

void mlirTransformOptionsEnforceSingleTopLevelTransformOp(
    MlirTransformOptions transformOptions, bool enable) {
  unwrap(transformOptions)->enableEnforceSingleToplevelTransformOp(enable);
}

bool mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(
    MlirTransformOptions transformOptions) {
  return unwrap(transformOptions)->getEnforceSingleToplevelTransformOp();
}

void mlirTransformOptionsDestroy(MlirTransformOptions transformOptions) {
  delete unwrap(transformOptions);
}

MlirLogicalResult mlirTransformApplyNamedSequence(
    MlirOperation payload, MlirOperation transformRoot,
    MlirOperation transformModule, MlirTransformOptions transformOptions) {
  Operation *transformRootOp = unwrap(transformRoot);
  Operation *transformModuleOp = unwrap(transformModule);
  if (!isa<transform::TransformOpInterface>(transformRootOp)) {
    transformRootOp->emitError()
        << "must implement TransformOpInterface to be used as transform root";
    return mlirLogicalResultFailure();
  }
  if (!isa<ModuleOp>(transformModuleOp)) {
    transformModuleOp->emitError()
        << "must be a " << ModuleOp::getOperationName();
    return mlirLogicalResultFailure();
  }
  return wrap(transform::applyTransformNamedSequence(
      unwrap(payload), unwrap(transformRoot),
      cast<ModuleOp>(unwrap(transformModule)), *unwrap(transformOptions)));
}
}
