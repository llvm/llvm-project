//===- ArmGraphOps.cpp - MLIR SPIR-V SPV_ARM_graph operations -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SPV_ARM_graph operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVParsingUtils.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/InterleavedRange.h"

using namespace mlir;
using namespace mlir::spirv::AttrNames;

//===----------------------------------------------------------------------===//
// spirv.GraphARM
//===----------------------------------------------------------------------===//

ParseResult spirv::GraphARMOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  Builder &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  SmallVector<Type> argTypes = llvm::map_to_vector(
      entryArgs, [](const OpAsmParser::Argument &arg) { return arg.type; });
  GraphType grType = builder.getGraphType(argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(grType));

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  // Parse the optional function body.
  Region *body = result.addRegion();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs);
  return failure(parseResult.has_value() && failed(*parseResult));
}

void spirv::GraphARMOp::print(OpAsmPrinter &printer) {
  // Print graph name, signature, and control.
  printer << " ";
  printer.printSymbolName(getSymName());
  GraphType grType = getFunctionType();
  function_interface_impl::printFunctionSignature(
      printer, *this, grType.getInputs(),
      /*isVariadic=*/false, grType.getResults());
  function_interface_impl::printFunctionAttributes(printer, *this,
                                                   {getFunctionTypeAttrName(),
                                                    getArgAttrsAttrName(),
                                                    getResAttrsAttrName()});

  // Print the body.
  Region &body = this->getBody();
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

LogicalResult spirv::GraphARMOp::verifyType() {
  if (getFunctionType().getNumResults() < 1)
    return emitOpError("there should be at least one result");
  return success();
}

LogicalResult spirv::GraphARMOp::verifyBody() {
  for (auto [index, graphArgType] : llvm::enumerate(getArgumentTypes())) {
    if (!isa<spirv::TensorArmType>(graphArgType)) {
      return emitOpError("type of argument #")
             << index << " must be a TensorArmType, but got " << graphArgType;
    }
  }
  for (auto [index, graphResType] : llvm::enumerate(getResultTypes())) {
    if (!isa<spirv::TensorArmType>(graphResType)) {
      return emitOpError("type of result #")
             << index << " must be a TensorArmType, but got " << graphResType;
    }
  }

  if (!isExternal()) {
    Block &entryBlock = front();

    unsigned numArguments = this->getNumArguments();
    if (entryBlock.getNumArguments() != numArguments)
      return emitOpError("entry block must have ")
             << numArguments << " arguments to match graph signature";

    for (auto [index, grArgType, blockArgType] :
         llvm::enumerate(getArgumentTypes(), entryBlock.getArgumentTypes())) {
      if (blockArgType != grArgType) {
        return emitOpError("type of entry block argument #")
               << index << '(' << blockArgType
               << ") must match the type of the corresponding argument in "
               << "graph signature(" << grArgType << ')';
      }
    }
  }

  GraphType grType = getFunctionType();
  auto walkResult = walk([grType](spirv::GraphOutputsARMOp op) -> WalkResult {
    if (grType.getNumResults() != op.getNumOperands())
      return op.emitOpError("is returning ")
             << op.getNumOperands()
             << " value(s) but enclosing spirv.ARM.Graph requires "
             << grType.getNumResults() << " result(s)";

    ValueTypeRange<OperandRange> graphOutputOperandTypes =
        op.getValue().getType();
    for (auto [index, type] : llvm::enumerate(graphOutputOperandTypes)) {
      if (type != grType.getResult(index))
        return op.emitError("type of return operand ")
               << index << " (" << type << ") doesn't match graph result type ("
               << grType.getResult(index) << ")";
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

void spirv::GraphARMOp::build(OpBuilder &builder, OperationState &state,
                              StringRef name, GraphType type,
                              ArrayRef<NamedAttribute> attrs, bool entryPoint) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs);
  state.addAttribute(getEntryPointAttrName(state.name),
                     builder.getBoolAttr(entryPoint));
  state.addRegion();
}

ArrayRef<Type> spirv::GraphARMOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

ArrayRef<Type> spirv::GraphARMOp::getResultTypes() {
  return getFunctionType().getResults();
}

Region *spirv::GraphARMOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

//===----------------------------------------------------------------------===//
// spirv.GraphOutputsARM
//===----------------------------------------------------------------------===//

LogicalResult spirv::GraphOutputsARMOp::verify() {
  auto graph = cast<GraphARMOp>((*this)->getParentOp());

  // The operand number and types must match the graph signature.
  const ArrayRef<Type> &results = graph.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing  spirv.ARM.Graph (@"
           << graph.getName() << ") returns " << results.size();

  for (auto [index, result] : llvm::enumerate(results))
    if (getOperand(index).getType() != result)
      return emitError() << "type of return operand " << index << " ("
                         << getOperand(index).getType()
                         << ") doesn't match  spirv.ARM.Graph result type ("
                         << result << ")"
                         << " in graph @" << graph.getName();
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.GraphEntryPointARM
//===----------------------------------------------------------------------===//

void spirv::GraphEntryPointARMOp::build(OpBuilder &builder,
                                        OperationState &state,
                                        spirv::GraphARMOp graph,
                                        ArrayRef<Attribute> interfaceVars) {
  build(builder, state, SymbolRefAttr::get(graph),
        builder.getArrayAttr(interfaceVars));
}

ParseResult spirv::GraphEntryPointARMOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  FlatSymbolRefAttr fn;
  if (parser.parseAttribute(fn, Type(), kFnNameAttrName, result.attributes))
    return failure();

  SmallVector<Attribute, 4> interfaceVars;
  if (!parser.parseOptionalComma()) {
    // Parse the interface variables.
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          // The name of the interface variable attribute is not important.
          FlatSymbolRefAttr var;
          NamedAttrList attrs;
          if (parser.parseAttribute(var, Type(), "var_symbol", attrs))
            return failure();
          interfaceVars.push_back(var);
          return success();
        }))
      return failure();
  }
  result.addAttribute("interface",
                      parser.getBuilder().getArrayAttr(interfaceVars));
  return success();
}

void spirv::GraphEntryPointARMOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getFn());
  ArrayRef<Attribute> interfaceVars = getInterface().getValue();
  if (!interfaceVars.empty()) {
    printer << ", " << llvm::interleaved(interfaceVars);
  }
}
