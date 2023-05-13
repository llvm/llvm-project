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

LogicalResult CreateOperationOp::verify() {
  if (!getInferredResultTypes())
    return success();
  if (!getInputResultTypes().empty()) {
    return emitOpError("with inferred results cannot also have "
                       "explicit result types");
  }
  OperationName opName(getName(), getContext());
  if (!opName.hasInterface<InferTypeOpInterface>()) {
    return emitOpError()
           << "has inferred results, but the created operation '" << opName
           << "' does not support result type inference (or is not "
              "registered)";
  }
  return success();
}

static ParseResult parseCreateOperationOpAttributes(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &attrOperands,
    ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    auto parseOperands = [&]() {
      StringAttr nameAttr;
      OpAsmParser::UnresolvedOperand operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOperands.push_back(operand);
      return success();
    };
    if (p.parseCommaSeparatedList(parseOperands) || p.parseRBrace())
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

static ParseResult parseCreateOperationOpResults(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultOperands,
    SmallVectorImpl<Type> &resultTypes, UnitAttr &inferredResultTypes) {
  if (failed(p.parseOptionalArrow()))
    return success();

  // Handle the case of inferred results.
  if (succeeded(p.parseOptionalLess())) {
    if (p.parseKeyword("inferred") || p.parseGreater())
      return failure();
    inferredResultTypes = p.getBuilder().getUnitAttr();
    return success();
  }

  // Otherwise, parse the explicit results.
  return failure(p.parseLParen() || p.parseOperandList(resultOperands) ||
                 p.parseColonTypeList(resultTypes) || p.parseRParen());
}

static void printCreateOperationOpResults(OpAsmPrinter &p, CreateOperationOp op,
                                          OperandRange resultOperands,
                                          TypeRange resultTypes,
                                          UnitAttr inferredResultTypes) {
  // Handle the case of inferred results.
  if (inferredResultTypes) {
    p << " -> <inferred>";
    return;
  }

  // Otherwise, handle the explicit results.
  if (!resultTypes.empty())
    p << " -> (" << resultOperands << " : " << resultTypes << ")";
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
    auto rangeType = llvm::cast<pdl::RangeType>(range.getType());
    state.regions.front()->emplaceBlock();
    state.regions.front()->addArgument(rangeType.getElementType(),
                                       state.location);
  }
}

ParseResult ForEachOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the loop variable followed by type.
  OpAsmParser::Argument loopVariable;
  OpAsmParser::UnresolvedOperand operandInfo;
  if (parser.parseArgument(loopVariable, /*allowType=*/true) ||
      parser.parseKeyword("in", " after loop variable") ||
      // Parse the operand (value range).
      parser.parseOperand(operandInfo))
    return failure();

  // Resolve the operand.
  Type rangeType = pdl::RangeType::get(loopVariable.type);
  if (parser.resolveOperand(operandInfo, rangeType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  Block *successor;
  if (parser.parseRegion(*body, loopVariable) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      // Parse the successor.
      parser.parseArrow() || parser.parseSuccessor(successor))
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
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

/// Given the result type of a `GetValueTypeOp`, return the expected input type.
static Type getGetValueTypeOpValueType(Type type) {
  Type valueTy = pdl::ValueType::get(type.getContext());
  return llvm::isa<pdl::RangeType>(type) ? pdl::RangeType::get(valueTy)
                                         : valueTy;
}

//===----------------------------------------------------------------------===//
// pdl::CreateRangeOp
//===----------------------------------------------------------------------===//

static ParseResult parseRangeType(OpAsmParser &p, TypeRange argumentTypes,
                                  Type &resultType) {
  // If arguments were provided, infer the result type from the argument list.
  if (!argumentTypes.empty()) {
    resultType =
        pdl::RangeType::get(pdl::getRangeElementTypeOrSelf(argumentTypes[0]));
    return success();
  }
  // Otherwise, parse the type as a trailing type.
  return p.parseColonType(resultType);
}

static void printRangeType(OpAsmPrinter &p, CreateRangeOp op,
                           TypeRange argumentTypes, Type resultType) {
  if (argumentTypes.empty())
    p << ": " << resultType;
}

LogicalResult CreateRangeOp::verify() {
  Type elementType = getType().getElementType();
  for (Type operandType : getOperandTypes()) {
    Type operandElementType = pdl::getRangeElementTypeOrSelf(operandType);
    if (operandElementType != elementType) {
      return emitOpError("expected operand to have element type ")
             << elementType << ", but got " << operandElementType;
    }
  }
  return success();
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
