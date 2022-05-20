//===- MLProgramOps.cpp - MLProgram dialect ops implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::ml_program;

//===----------------------------------------------------------------------===//
// Custom asm helpers
//===----------------------------------------------------------------------===//

/// some.op custom<TypeOrAttr>($type, $attr)
///
/// Uninitialized:
///   some.op : tensor<3xi32>
/// Initialized to narrower type than op:
///   some.op (dense<0> : tensor<3xi32>) : tensor<?xi32>
static ParseResult parseTypedInitialValue(OpAsmParser &parser,
                                          TypeAttr &typeAttr, Attribute &attr) {
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseAttribute(attr)))
      return failure();
    if (failed(parser.parseRParen()))
      return failure();
  }

  Type type;
  if (failed(parser.parseColonType(type)))
    return failure();
  typeAttr = TypeAttr::get(type);
  return success();
}

static void printTypedInitialValue(OpAsmPrinter &p, Operation *op,
                                   TypeAttr type, Attribute attr) {
  if (attr) {
    p << "(";
    p.printAttribute(attr);
    p << ")";
  }

  p << " : ";
  p.printAttribute(type);
}

/// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
/// ->
/// some.op public @foo
/// some.op private @foo
static ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                         StringAttr &symVisibilityAttr) {
  StringRef symVisibility;
  (void)parser.parseOptionalKeyword(&symVisibility,
                                    {"public", "private", "nested"});
  if (symVisibility.empty())
    return parser.emitError(parser.getCurrentLocation())
           << "expected 'public', 'private', or 'nested'";
  if (!symVisibility.empty())
    symVisibilityAttr = parser.getBuilder().getStringAttr(symVisibility);
  return success();
}

static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr symVisibilityAttr) {
  if (!symVisibilityAttr)
    p << "public";
  else
    p << symVisibilityAttr.getValue();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MLProgram/IR/MLProgramOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  if (!getIsMutable() && !getValue())
    return emitOpError() << "immutable global must have an initial value";
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalLoadConstOp
//===----------------------------------------------------------------------===//

GlobalOp GlobalLoadConstOp::getGlobalOp(SymbolTableCollection &symbolTable) {
  return symbolTable.lookupNearestSymbolFrom<GlobalOp>(
      getOperation()->getParentOp(), getGlobalAttr());
}

LogicalResult
GlobalLoadConstOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  GlobalOp referrent = getGlobalOp(symbolTable);
  if (!referrent)
    return emitOpError() << "undefined global: " << getGlobal();

  if (referrent.getIsMutable())
    return emitOpError() << "cannot load as const from mutable global "
                         << getGlobal();

  if (referrent.getType() != getResult().getType())
    return emitOpError() << "cannot load from global typed "
                         << referrent.getType() << " as "
                         << getResult().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

ParseResult SubgraphOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void SubgraphOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  auto function = cast<SubgraphOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") outputs " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of output operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}
