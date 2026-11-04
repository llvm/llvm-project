//===- Traits.cpp - Traits for MLIR DLTI dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

LogicalResult mlir::impl::verifyHasDefaultDLTIDataLayoutTrait(Operation *op) {
  // TODO: consider having trait inheritance so that HasDefaultDLTIDataLayout
  // trait can inherit DataLayoutOpInterface::Trait and enforce the validity of
  // the assertion below.
  assert(
      isa<DataLayoutOpInterface>(op) &&
      "HasDefaultDLTIDataLayout trait unexpectedly attached to an op that does "
      "not implement DataLayoutOpInterface");
  return success();
}

DataLayoutSpecInterface mlir::impl::getDataLayoutSpec(Operation *op) {
  return op->getAttrOfType<DataLayoutSpecInterface>(
      DLTIDialect::kDataLayoutAttrName);
}

TargetSystemSpecInterface mlir::impl::getTargetSystemSpec(Operation *op) {
  return op->getAttrOfType<TargetSystemSpecAttr>(
      DLTIDialect::kTargetSystemDescAttrName);
}
