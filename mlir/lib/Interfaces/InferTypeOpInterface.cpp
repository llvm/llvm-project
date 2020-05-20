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

#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/IR/StandardTypes.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/InferTypeOpInterface.cpp.inc"
} // namespace mlir

LogicalResult mlir::detail::inferReturnTensorTypes(
    function_ref<LogicalResult(
        MLIRContext *, Optional<Location> location, ValueRange operands,
        DictionaryAttr attributes, RegionRange regions,
        SmallVectorImpl<ShapedTypeComponents> &retComponents)>
        componentTypeFn,
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SmallVector<ShapedTypeComponents, 2> retComponents;
  if (failed(componentTypeFn(context, location, operands, attributes, regions,
                             retComponents)))
    return failure();
  for (auto shapeAndType : retComponents) {
    assert(shapeAndType.getAttribute() == nullptr && "attribute not supported");
    if (shapeAndType.hasRank())
      inferredReturnTypes.push_back(RankedTensorType::get(
          shapeAndType.getDims(), shapeAndType.getElementType()));
    else
      inferredReturnTypes.push_back(
          UnrankedTensorType::get(shapeAndType.getElementType()));
  }
  return success();
}

LogicalResult mlir::detail::verifyInferredResultTypes(Operation *op) {
  SmallVector<Type, 4> inferredReturnTypes;
  auto retTypeFn = cast<InferTypeOpInterface>(op);
  if (failed(retTypeFn.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getRegions(), inferredReturnTypes)))
    return failure();
  if (!retTypeFn.isCompatibleReturnTypes(inferredReturnTypes,
                                         op->getResultTypes()))
    return op->emitOpError(
        "inferred type incompatible with return type of operation");
  return success();
}
