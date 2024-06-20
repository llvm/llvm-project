//===- MLProgramOps.cpp - MLProgram dialect ops implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::ml_program;

//===----------------------------------------------------------------------===//
// Custom asm helpers
//===----------------------------------------------------------------------===//

/// Parse and print an ordering clause for a variadic of consuming tokens
/// and an producing token.
///
/// Syntax:
///   ordering(%0, %1 -> !ml_program.token)
///   ordering(() -> !ml_program.token)
///
/// If both the consuming and producing token are not present on the op, then
/// the clause prints nothing.
static ParseResult parseTokenOrdering(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &consumeTokens,
    Type &produceTokenType) {
  if (failed(parser.parseOptionalKeyword("ordering")) ||
      failed(parser.parseLParen()))
    return success();

  // Parse consuming token list. If there are no consuming tokens, the
  // '()' null list represents this.
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseRParen()))
      return failure();
  } else {
    if (failed(parser.parseOperandList(consumeTokens,
                                       /*requiredOperandCount=*/-1)))
      return failure();
  }

  // Parse producer token.
  if (failed(parser.parseArrow()))
    return failure();
  if (failed(parser.parseType(produceTokenType)))
    return failure();

  if (failed(parser.parseRParen()))
    return failure();

  return success();
}

static void printTokenOrdering(OpAsmPrinter &p, Operation *op,
                               OperandRange consumeTokens,
                               Type produceTokenType) {
  if (consumeTokens.empty() && !produceTokenType)
    return;

  p << " ordering(";
  if (consumeTokens.empty())
    p << "()";
  else
    p.printOperands(consumeTokens);
  if (produceTokenType) {
    p << " -> ";
    p.printType(produceTokenType);
  }
  p << ")";
}

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
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
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
// GlobalLoadOp
//===----------------------------------------------------------------------===//

GlobalOp GlobalLoadOp::getGlobalOp(SymbolTableCollection &symbolTable) {
  for (auto *parent = getOperation()->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto nearest = symbolTable.lookupNearestSymbolFrom<GlobalOp>(
            parent, getGlobalAttr())) {
      return nearest;
    }
  }
  return {};
}

LogicalResult
GlobalLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  GlobalOp referrent = getGlobalOp(symbolTable);
  if (!referrent)
    return emitOpError() << "undefined global: " << getGlobal();

  if (referrent.getType() != getResult().getType()) {
    return emitOpError() << "cannot load from global typed "
                         << referrent.getType() << " as "
                         << getResult().getType();
  }

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
// GlobalLoadGraphOp
//===----------------------------------------------------------------------===//

GlobalOp GlobalLoadGraphOp::getGlobalOp(SymbolTableCollection &symbolTable) {
  return symbolTable.lookupNearestSymbolFrom<GlobalOp>(
      getOperation()->getParentOp(), getGlobalAttr());
}

LogicalResult
GlobalLoadGraphOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  GlobalOp referrent = getGlobalOp(symbolTable);
  if (!referrent)
    return emitOpError() << "undefined global: " << getGlobal();

  if (referrent.getType() != getResult().getType()) {
    return emitOpError() << "cannot load from global typed "
                         << referrent.getType() << " as "
                         << getResult().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalStoreOp
//===----------------------------------------------------------------------===//

GlobalOp GlobalStoreOp::getGlobalOp(SymbolTableCollection &symbolTable) {
  for (auto *parent = getOperation()->getParentOp(); parent;) {
    if (auto nearest = symbolTable.lookupNearestSymbolFrom<GlobalOp>(
            parent, getGlobalAttr())) {
      return nearest;
    }
    parent = parent->getParentOp();
  }
  return {};
}

LogicalResult
GlobalStoreOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  GlobalOp referrent = getGlobalOp(symbolTable);
  if (!referrent)
    return emitOpError() << "undefined global: " << getGlobal();

  if (!referrent.getIsMutable()) {
    return emitOpError() << "cannot store to an immutable global "
                         << getGlobal();
  }

  if (referrent.getType() != getValue().getType()) {
    return emitOpError() << "cannot store to a global typed "
                         << referrent.getType() << " from "
                         << getValue().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalStoreGraphOp
//===----------------------------------------------------------------------===//

GlobalOp GlobalStoreGraphOp::getGlobalOp(SymbolTableCollection &symbolTable) {
  return symbolTable.lookupNearestSymbolFrom<GlobalOp>(
      getOperation()->getParentOp(), getGlobalAttr());
}

LogicalResult
GlobalStoreGraphOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  GlobalOp referrent = getGlobalOp(symbolTable);
  if (!referrent)
    return emitOpError() << "undefined global: " << getGlobal();

  if (!referrent.getIsMutable()) {
    return emitOpError() << "cannot store to an immutable global "
                         << getGlobal();
  }

  if (referrent.getType() != getValue().getType()) {
    return emitOpError() << "cannot store to a global typed "
                         << referrent.getType() << " from "
                         << getValue().getType();
  }

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
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void SubgraphOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
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
