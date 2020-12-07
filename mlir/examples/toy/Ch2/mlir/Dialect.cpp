//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::toy;

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<ToyDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
  printer << "toy.constant ";
  printer.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// GenericCallOp

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

//===----------------------------------------------------------------------===//
// MulOp

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op.getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand (" << inputType
                        << ") doesn't match function result type ("
                        << resultType << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

static mlir::LogicalResult verify(TransposeOp op) {
  auto inputType = op.getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = op.getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return op.emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
