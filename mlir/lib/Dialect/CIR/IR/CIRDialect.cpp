//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::cir;

#include "mlir/Dialect/CIR/IR/CIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void cir::CIRDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(getOperation()->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType)
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

static void printAllocaOpTypeAndInitialValue(OpAsmPrinter &p, AllocaOp op,
                                             TypeAttr type,
                                             Attribute initialValue) {
  p << type;
  p << " = ";
  if (op.isUninitialized())
    p << "uninitialized";
  else
    p.printAttributeWithoutType(initialValue);
}

static ParseResult parseAllocaOpTypeAndInitialValue(OpAsmParser &parser,
                                                    TypeAttr &typeAttr,
                                                    Attribute &initialValue) {
  Type type;
  if (parser.parseType(type))
    return failure();
  typeAttr = TypeAttr::get(type);

  if (parser.parseEqual())
    return success();

  if (succeeded(parser.parseOptionalKeyword("uninitialized")))
    initialValue = UnitAttr::get(parser.getBuilder().getContext());

  if (!initialValue.isa<UnitAttr>())
    return parser.emitError(parser.getNameLoc())
           << "constant operation not implemented yet";

  return success();
}

LogicalResult AllocaOp::verify() {
  // Verify that the initial value, is either a unit attribute or
  // an elements attribute.
  Attribute initValue = getInitialValue();
  if (!initValue.isa<UnitAttr>())
    return emitOpError("initial value should be a unit "
                       "attribute, but got ")
           << initValue;

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
