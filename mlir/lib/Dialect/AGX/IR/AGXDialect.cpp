//===- AGXDialect.cpp - MLIR AGX dialect implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AGX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AGX/IR/AGXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::agx;

#include "mlir/Dialect/AGX/IR/AGXDialect.cpp.inc"
#include "mlir/Dialect/AGX/IR/AGXEnums.cpp.inc"

void AGXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AGX/IR/AGX.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AGX/IR/AGXAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleOp::verify() {
  Type type = getValue().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (!elementType.isIntOrFloat())
    return emitOpError() << "requires integer or floating point values, got "
                         << type;
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (vectorType.isScalable())
      return emitOpError() << "does not support scalable vectors";
    bits *= vectorType.getNumElements();
  }
  // The value is split into 32-bit parts; 16-bit values are shuffled
  // directly.
  if (bits != 16 && bits % 32 != 0)
    return emitOpError() << "requires a 16-bit value or a multiple of 32 bits, "
                            "got "
                         << type;
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  bool isFloatKind = getKind() == ReduceKind::minnumf ||
                     getKind() == ReduceKind::maxnumf;
  bool isArithKind = getKind() == ReduceKind::add ||
                     getKind() == ReduceKind::mul;
  Type type = getValue().getType();
  if (isa<FloatType>(type)) {
    if (!isFloatKind && !isArithKind)
      return emitOpError() << "reduction '" << stringifyReduceKind(getKind())
                           << "' does not apply to floating point values";
  } else if (isFloatKind) {
    return emitOpError() << "reduction '" << stringifyReduceKind(getKind())
                         << "' requires floating point values";
  } else if (getKind() == ReduceKind::mul) {
    return emitOpError() << "there is no integer multiplication reduction";
  }
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AGX/IR/AGXAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AGX/IR/AGX.cpp.inc"
