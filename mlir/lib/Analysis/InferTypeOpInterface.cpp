//===- InferTypeOpInterface.cpp - Infer Type Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `InferTypeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/InferTypeOpInterface.h"

#include "mlir/IR/StandardTypes.h"

using namespace mlir;

namespace mlir {
#include "mlir/Analysis/InferTypeOpInterface.cpp.inc"
} // namespace mlir

LogicalResult mlir::detail::inferReturnTensorTypes(
    function_ref<LogicalResult(
        MLIRContext *, Optional<Location> location, ValueRange operands,
        ArrayRef<NamedAttribute> attributes, RegionRange regions,
        SmallVectorImpl<ShapedTypeComponents> &retComponents)>
        componentTypeFn,
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  SmallVector<ShapedTypeComponents, 2> retComponents;
  if (failed(componentTypeFn(context, location, operands, attributes, regions,
                             retComponents)))
    return failure();
  for (auto shapeAndType : retComponents) {
    assert(shapeAndType.getAttribute() == nullptr && "attribute not supported");
    if (shapeAndType.hasRank())
      inferedReturnTypes.push_back(RankedTensorType::get(
          shapeAndType.getDims(), shapeAndType.getElementType()));
    else
      inferedReturnTypes.push_back(
          UnrankedTensorType::get(shapeAndType.getElementType()));
  }
  return success();
}

LogicalResult mlir::detail::verifyInferredResultTypes(Operation *op) {
  SmallVector<Type, 4> inferedReturnTypes;
  auto retTypeFn = cast<InferTypeOpInterface>(op);
  if (failed(retTypeFn.inferReturnTypes(op->getContext(), op->getLoc(),
                                        op->getOperands(), op->getAttrs(),
                                        op->getRegions(), inferedReturnTypes)))
    return failure();
  if (!retTypeFn.isCompatibleReturnTypes(inferedReturnTypes,
                                         op->getResultTypes()))
    return op->emitOpError(
        "inferred type incompatible with return type of operation");
  return success();
}
