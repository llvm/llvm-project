//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/Casting.h"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOps.cpp.inc"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::wasmssa;

namespace {
inline LogicalResult
inferTeeGetResType(ValueRange operands,
                   ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto opType = llvm::dyn_cast<LocalRefType>(operands.front().getType());
  if (!opType)
    return failure();
  inferredReturnTypes.push_back(opType.getElementType());
  return success();
}

ParseResult parseImportOp(OpAsmParser &parser, OperationState &result) {
  std::string importName;
  auto *ctx = parser.getContext();
  ParseResult res = parser.parseString(&importName);
  result.addAttribute("importName", StringAttr::get(ctx, importName));

  std::string fromStr;
  res = parser.parseKeywordOrString(&fromStr);
  if (failed(res) || fromStr != "from")
    return failure();

  std::string moduleName;
  res = parser.parseString(&moduleName);
  if (failed(res))
    return failure();
  result.addAttribute("moduleName", StringAttr::get(ctx, moduleName));

  std::string asStr;
  res = parser.parseKeywordOrString(&asStr);
  if (failed(res) || asStr != "as")
    return failure();

  StringAttr symbolName;
  res = parser.parseSymbolName(symbolName, SymbolTable::getSymbolAttrName(),
                               result.attributes);
  return res;
}
} // namespace

//===----------------------------------------------------------------------===//
// BlockOp
//===----------------------------------------------------------------------===//

Block *BlockOp::getLabelTarget() { return getTarget(); }

//===----------------------------------------------------------------------===//
// BlockReturnOp
//===----------------------------------------------------------------------===//

std::size_t BlockReturnOp::getExitLevel() { return 0; }

Block *BlockReturnOp::getTarget() {
  return cast<LabelBranchingOpInterface>(getOperation())
      .getTargetOp()
      .getOperation()
      ->getSuccessor(0);
}

//===----------------------------------------------------------------------===//
// ExtendLowBitsSOp
//===----------------------------------------------------------------------===//

ParseResult ExtendLowBitsSOp::parse(::mlir::OpAsmParser &parser,
                                    ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  uint64_t nBits;
  ParseResult parseRes = parser.parseInteger(nBits);
  parseRes = parser.parseKeyword("low");
  parseRes = parser.parseKeyword("bits");
  parseRes = parser.parseKeyword("from");
  parseRes = parser.parseOperand(operand);
  parseRes = parser.parseColon();
  Type inType;
  parseRes = parser.parseType(inType);
  if (!inType.isInteger())
    return failure();
  llvm::SmallVector<Value, 1> opVal;
  parseRes = parser.resolveOperand(operand, inType, opVal);
  if (parseRes.failed())
    return failure();
  result.addOperands(opVal);
  result.addAttribute(
      ExtendLowBitsSOp::getBitsToTakeAttrName(OperationName{
          ExtendLowBitsSOp::getOperationName(), parser.getContext()}),
      parser.getBuilder().getI64IntegerAttr(nBits));
  result.addTypes(inType);
  return success();
}

void ExtendLowBitsSOp::print(OpAsmPrinter &p) {
  p << " " << getBitsToTake().getUInt() << " low bits from ";
  p.printOperand(getInput());
  p << ": " << getInput().getType();
}

LogicalResult ExtendLowBitsSOp::verify() {
  auto bitsToTake = getBitsToTake().getValue().getLimitedValue();
  if (bitsToTake != 32 && bitsToTake != 16 && bitsToTake != 8)
    return emitError("extend op can only take 8, 16 or 32 bits. Got ")
           << bitsToTake;

  if (bitsToTake >= getInput().getType().getIntOrFloatBitWidth())
    return emitError("trying to extend the ")
           << bitsToTake << " low bits from a " << getInput().getType()
           << " value";
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

Block *FuncOp::addEntryBlock() {
  if (!getBody().empty()) {
    emitError("adding entry block to a FuncOp which already has one.");
    return &getBody().front();
  }
  Block &block = getBody().emplaceBlock();
  for (auto argType : getFunctionType().getInputs())
    block.addArgument(LocalRefType::get(argType), getLoc());
  return &block;
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState, llvm::StringRef symbol,
                   FunctionType funcType) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("functionType", TypeAttr::get(funcType));
  odsState.addRegion();
}

ParseResult FuncOp::parse(::mlir::OpAsmParser &parser,
                          ::mlir::OperationState &result) {
  auto buildFuncType = [&parser](Builder &builder, ArrayRef<Type> argTypes,
                                 ArrayRef<Type> results,
                                 function_interface_impl::VariadicFlag,
                                 std::string &) {
    llvm::SmallVector<Type> argTypesWithoutLocal{};
    argTypesWithoutLocal.reserve(argTypes.size());
    llvm::for_each(argTypes, [&parser, &argTypesWithoutLocal](Type argType) {
      auto refType = dyn_cast<LocalRefType>(argType);
      auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
      if (!refType) {
        mlir::emitError(loc, "invalid type for wasm.func argument. Expecting "
                             "!wasm<local T>, got ")
            << argType << ".";
        return;
      }
      argTypesWithoutLocal.push_back(refType.getElementType());
    });

    return builder.getFunctionType(argTypesWithoutLocal, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

LogicalResult FuncOp::verifyBody() {
  if (getBody().empty())
    return success();
  Block &entry = getBody().front();
  if (entry.getNumArguments() != getFunctionType().getNumInputs())
    return emitError("entry block should have same number of arguments as "
                     "function type. Function type has ")
           << getFunctionType().getNumInputs() << ", entry block has "
           << entry.getNumArguments() << ".";

  for (auto [argNo, funcSignatureType, blockType] : llvm::enumerate(
           getFunctionType().getInputs(), entry.getArgumentTypes())) {
    auto blockLocalRefType = dyn_cast<LocalRefType>(blockType);
    if (!blockLocalRefType)
      return emitError("entry block argument type should be LocalRefType, got ")
             << blockType << " for block argument " << argNo << ".";
    if (blockLocalRefType.getElementType() != funcSignatureType)
      return emitError("func argument type #")
             << argNo << "(" << funcSignatureType
             << ") doesn't match entry block referenced type ("
             << blockLocalRefType.getElementType() << ").";
  }
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// FuncImportOp
//===----------------------------------------------------------------------===//

void FuncImportOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState, StringRef symbol,
                         StringRef moduleName, StringRef importName,
                         FunctionType type) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("moduleName", odsBuilder.getStringAttr(moduleName));
  odsState.addAttribute("importName", odsBuilder.getStringAttr(importName));
  odsState.addAttribute("type", TypeAttr::get(type));
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

void GlobalOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState, llvm::StringRef symbol,
                     Type type, bool isMutable) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("type", TypeAttr::get(type));
  if (isMutable)
    odsState.addAttribute("isMutable", odsBuilder.getUnitAttr());
  odsState.addRegion();
}

// Custom formats
ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbolName;
  Type globalType;
  auto *ctx = parser.getContext();
  ParseResult res = parser.parseSymbolName(
      symbolName, SymbolTable::getSymbolAttrName(), result.attributes);

  res = parser.parseType(globalType);
  result.addAttribute(getTypeAttrName(result.name), TypeAttr::get(globalType));
  std::string mutableString;
  res = parser.parseOptionalKeywordOrString(&mutableString);
  if (res.succeeded() && mutableString == "mutable")
    result.addAttribute("isMutable", UnitAttr::get(ctx));
  std::string visibilityString;
  res = parser.parseOptionalKeywordOrString(&visibilityString);
  if (res.succeeded())
    result.addAttribute("sym_visibility",
                        StringAttr::get(ctx, visibilityString));
  res = parser.parseColon();
  Region *globalInitRegion = result.addRegion();
  res = parser.parseRegion(*globalInitRegion);
  return res;
}

void GlobalOp::print(OpAsmPrinter &printer) {
  printer << " @" << getSymName().str() << " " << getType();
  if (getIsMutable())
    printer << " mutable";
  if (auto vis = getSymVisibility())
    printer << " " << *vis;
  printer << " :";
  Region &body = getRegion();
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// GlobalGetOp
//===----------------------------------------------------------------------===//

// Custom interface overrides
LogicalResult GlobalGetOp::isValidInConstantExpr() {
  StringRef referencedSymbol = getGlobal();
  Operation *symTableOp =
      getOperation()->getParentWithTrait<OpTrait::SymbolTable>();
  Operation *definitionOp =
      SymbolTable::lookupSymbolIn(symTableOp, referencedSymbol);
  if (!definitionOp)
    return failure();
  auto definitionImport = llvm::dyn_cast<GlobalImportOp>(definitionOp);
  if (!definitionImport || definitionImport.getIsMutable()) {
    return emitError("global.get op is considered constant if it's referring "
                     "to a import.global symbol marked non-mutable.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalImportOp
//===----------------------------------------------------------------------===//

void GlobalImportOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, StringRef symbol,
                           StringRef moduleName, StringRef importName,
                           Type type, bool isMutable) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("moduleName", odsBuilder.getStringAttr(moduleName));
  odsState.addAttribute("importName", odsBuilder.getStringAttr(importName));
  odsState.addAttribute("type", TypeAttr::get(type));
  if (isMutable)
    odsState.addAttribute("isMutable", odsBuilder.getUnitAttr());
}

ParseResult GlobalImportOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *ctx = parser.getContext();
  ParseResult res = parseImportOp(parser, result);
  if (res.failed())
    return failure();
  std::string mutableOrSymVisString;
  res = parser.parseOptionalKeywordOrString(&mutableOrSymVisString);
  if (res.succeeded() && mutableOrSymVisString == "mutable") {
    result.addAttribute("isMutable", UnitAttr::get(ctx));
    res = parser.parseOptionalKeywordOrString(&mutableOrSymVisString);
  }

  if (res.succeeded())
    result.addAttribute("sym_visibility",
                        StringAttr::get(ctx, mutableOrSymVisString));
  res = parser.parseColon();

  Type importedType;
  res = parser.parseType(importedType);
  if (res.succeeded())
    result.addAttribute(getTypeAttrName(result.name),
                        TypeAttr::get(importedType));
  return res;
}

void GlobalImportOp::print(OpAsmPrinter &printer) {
  printer << " \"" << getImportName() << "\" from \"" << getModuleName()
          << "\" as @" << getSymName();
  if (getIsMutable())
    printer << " mutable";
  if (auto vis = getSymVisibility())
    printer << " " << *vis;
  printer << " : " << getType();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

Block *IfOp::getLabelTarget() { return getTarget(); }

//===----------------------------------------------------------------------===//
// LocalOp
//===----------------------------------------------------------------------===//

LogicalResult LocalOp::inferReturnTypes(
    MLIRContext *context, ::std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  LocalOp::GenericAdaptor<ValueRange> adaptor{operands, attributes, properties,
                                              regions};
  auto type = adaptor.getTypeAttr();
  if (!type)
    return failure();
  auto resType = LocalRefType::get(type.getContext(), type.getValue());
  inferredReturnTypes.push_back(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// LocalGetOp
//===----------------------------------------------------------------------===//

LogicalResult LocalGetOp::inferReturnTypes(
    MLIRContext *context, ::std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTeeGetResType(operands, inferredReturnTypes);
}

LogicalResult LocalGetOp::verify() {
  return success(getLocalVar().getType().getElementType() ==
                 getResult().getType());
}

//===----------------------------------------------------------------------===//
// LocalSetOp
//===----------------------------------------------------------------------===//

LogicalResult LocalSetOp::verify() {
  return success(getLocalVar().getType().getElementType() ==
                 getValue().getType());
}

//===----------------------------------------------------------------------===//
// LocalTeeOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTeeOp::inferReturnTypes(
    MLIRContext *context, ::std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTeeGetResType(operands, inferredReturnTypes);
}

LogicalResult LocalTeeOp::verify() {
  return success(getLocalVar().getType().getElementType() ==
                     getValue().getType() &&
                 getValue().getType() == getResult().getType());
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

Block *LoopOp::getLabelTarget() { return &getBody().front(); }

//===----------------------------------------------------------------------===//
// MemOp
//===----------------------------------------------------------------------===//

void MemOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, llvm::StringRef symbol,
                  LimitType limit) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("limits", TypeAttr::get(limit));
}

//===----------------------------------------------------------------------===//
// MemImportOp
//===----------------------------------------------------------------------===//

void MemImportOp::build(mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        llvm::StringRef symbol, llvm::StringRef moduleName,
                        llvm::StringRef importName, LimitType limits) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("moduleName", odsBuilder.getStringAttr(moduleName));
  odsState.addAttribute("importName", odsBuilder.getStringAttr(importName));
  odsState.addAttribute("limits", TypeAttr::get(limits));
}

//===----------------------------------------------------------------------===//
// ReinterpretOp
//===----------------------------------------------------------------------===//

LogicalResult ReinterpretOp::verify() {
  auto inT = getInput().getType();
  auto resT = getResult().getType();
  if (inT == resT)
    return emitError("reinterpret input and output type should be distinct.");
  if (inT.getIntOrFloatBitWidth() != resT.getIntOrFloatBitWidth())
    return emitError() << "input type (" << inT << ") and output type (" << resT
                       << ") have incompatible bit widths.";
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState) {}

//===----------------------------------------------------------------------===//
// TableOp
//===----------------------------------------------------------------------===//

void TableOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, llvm::StringRef symbol,
                    TableType type) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("type", TypeAttr::get(type));
}

//===----------------------------------------------------------------------===//
// TableImportOp
//===----------------------------------------------------------------------===//

void TableImportOp::build(mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          llvm::StringRef symbol, llvm::StringRef moduleName,
                          llvm::StringRef importName, TableType type) {
  odsState.addAttribute("sym_name", odsBuilder.getStringAttr(symbol));
  odsState.addAttribute("sym_visibility", odsBuilder.getStringAttr("nested"));
  odsState.addAttribute("moduleName", odsBuilder.getStringAttr(moduleName));
  odsState.addAttribute("importName", odsBuilder.getStringAttr(importName));
  odsState.addAttribute("type", TypeAttr::get(type));
}
