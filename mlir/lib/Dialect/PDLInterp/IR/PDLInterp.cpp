//===- PDLInterp.cpp - PDL Interpreter Dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::pdl_interp;

#include "mlir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

void PDLInterpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
      >();
}

template <typename OpT>
static LogicalResult verifySwitchOp(OpT op) {
  // Verify that the number of case destinations matches the number of case
  // values.
  size_t numDests = op.getCases().size();
  size_t numValues = op.getCaseValues().size();
  if (numDests != numValues) {
    return op.emitOpError(
               "expected number of cases to match the number of case "
               "values, got ")
           << numDests << " but expected " << numValues;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseCreateOperationOpAttributes(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &attrOperands,
    ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    do {
      StringAttr nameAttr;
      OpAsmParser::UnresolvedOperand operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOperands.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  return success();
}

static void printCreateOperationOpAttributes(OpAsmPrinter &p,
                                             CreateOperationOp op,
                                             OperandRange attrArgs,
                                             ArrayAttr attrNames) {
  if (attrNames.empty())
    return;
  p << " {";
  interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                  [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
  p << '}';
}

//===----------------------------------------------------------------------===//
// pdl_interp::ForEachOp
//===----------------------------------------------------------------------===//

void ForEachOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                      Value range, Block *successor, bool initLoop) {
  build(builder, state, range, successor);
  if (initLoop) {
    // Create the block and the loop variable.
    // FIXME: Allow passing in a proper location for the loop variable.
    auto rangeType = range.getType().cast<pdl::RangeType>();
    state.regions.front()->emplaceBlock();
    state.regions.front()->addArgument(rangeType.getElementType(),
                                       state.location);
  }
}

ParseResult ForEachOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the loop variable followed by type.
  OpAsmParser::UnresolvedOperand loopVariable;
  Type loopVariableType;
  if (parser.parseRegionArgument(loopVariable) ||
      parser.parseColonType(loopVariableType))
    return failure();

  // Parse the "in" keyword.
  if (parser.parseKeyword("in", " after loop variable"))
    return failure();

  // Parse the operand (value range).
  OpAsmParser::UnresolvedOperand operandInfo;
  if (parser.parseOperand(operandInfo))
    return failure();

  // Resolve the operand.
  Type rangeType = pdl::RangeType::get(loopVariableType);
  if (parser.resolveOperand(operandInfo, rangeType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {loopVariable}, {loopVariableType}))
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the successor.
  Block *successor;
  if (parser.parseArrow() || parser.parseSuccessor(successor))
    return failure();
  result.addSuccessors(successor);

  return success();
}

void ForEachOp::print(OpAsmPrinter &p) {
  BlockArgument arg = getLoopVariable();
  p << ' ' << arg << " : " << arg.getType() << " in " << getValues() << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " -> ";
  p.printSuccessor(getSuccessor());
}

LogicalResult ForEachOp::verify() {
  // Verify that the operation has exactly one argument.
  if (getRegion().getNumArguments() != 1)
    return emitOpError("requires exactly one argument");

  // Verify that the loop variable and the operand (value range)
  // have compatible types.
  BlockArgument arg = getLoopVariable();
  Type rangeType = pdl::RangeType::get(arg.getType());
  if (rangeType != getValues().getType())
    return emitOpError("operand must be a range of loop variable type");

  return success();
}

//===----------------------------------------------------------------------===//
// pdl_interp::FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

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
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

/// Given the result type of a `GetValueTypeOp`, return the expected input type.
static Type getGetValueTypeOpValueType(Type type) {
  Type valueTy = pdl::ValueType::get(type.getContext());
  return type.isa<pdl::RangeType>() ? pdl::RangeType::get(valueTy) : valueTy;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchAttributeOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchAttributeOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperandCountOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchOperandCountOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperationNameOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchOperationNameOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchResultCountOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchResultCountOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypeOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchTypeOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypesOp
//===----------------------------------------------------------------------===//

LogicalResult SwitchTypesOp::verify() { return verifySwitchOp(*this); }

//===----------------------------------------------------------------------===//
// TableGen Auto-Generated Op and Interface Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
