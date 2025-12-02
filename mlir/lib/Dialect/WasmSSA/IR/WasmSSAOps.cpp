//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/Casting.h"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace {
ParseResult parseElseRegion(OpAsmParser &opParser, Region &elseRegion) {
  std::string keyword;
  std::ignore = opParser.parseOptionalKeywordOrString(&keyword);
  if (keyword == "else")
    return opParser.parseRegion(elseRegion);
  return ParseResult::success();
}

void printElseRegion(OpAsmPrinter &opPrinter, Operation *op,
                     Region &elseRegion) {
  if (elseRegion.empty())
    return;
  opPrinter.printKeywordOrString("else ");
  opPrinter.printRegion(elseRegion);
}
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOps.cpp.inc"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

using namespace wasmssa;

namespace {
inline LogicalResult
inferTeeGetResType(ValueRange operands,
                   SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto opType = dyn_cast<LocalRefType>(operands.front().getType());
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

LogicalResult ExtendLowBitsSOp::verify() {
  auto bitsToTake = getBitsToTake().getValue().getLimitedValue();
  if (bitsToTake != 32 && bitsToTake != 16 && bitsToTake != 8)
    return emitError("extend op can only take 8, 16 or 32 bits. Got ")
           << bitsToTake;

  if (bitsToTake >= getInput().getType().getIntOrFloatBitWidth())
    return emitError("trying to extend the ")
           << bitsToTake << " low bits from a " << getInput().getType()
           << " value is illegal";
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

Block *FuncOp::addEntryBlock() {
  if (!getBody().empty()) {
    emitError("adding entry block to a FuncOp which already has one");
    return &getBody().front();
  }
  Block &block = getBody().emplaceBlock();
  for (auto argType : getFunctionType().getInputs())
    block.addArgument(LocalRefType::get(argType), getLoc());
  return &block;
}

void FuncOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   StringRef symbol, FunctionType funcType) {
  FuncOp::build(odsBuilder, odsState, symbol, funcType, {}, {});
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *ctx = parser.getContext();
  std::string visibilityString;
  auto loc = parser.getNameLoc();
  ParseResult res = parser.parseOptionalKeywordOrString(&visibilityString);
  bool exported{false};
  if (res.succeeded()) {
    if (visibilityString != "exported")
      return parser.emitError(
                 loc, "expecting either `exported` or symbol name. got ")
             << visibilityString;
    exported = true;
  }

  auto buildFuncType = [&parser](Builder &builder, ArrayRef<Type> argTypes,
                                 ArrayRef<Type> results,
                                 function_interface_impl::VariadicFlag,
                                 std::string &) {
    SmallVector<Type> argTypesWithoutLocal{};
    argTypesWithoutLocal.reserve(argTypes.size());
    llvm::for_each(argTypes, [&parser, &argTypesWithoutLocal](Type argType) {
      auto refType = dyn_cast<LocalRefType>(argType);
      auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
      if (!refType) {
        mlir::emitError(loc, "invalid type for wasm.func argument. Expecting "
                             "!wasm<local T>, got ")
            << argType;
        return;
      }
      argTypesWithoutLocal.push_back(refType.getElementType());
    });

    return builder.getFunctionType(argTypesWithoutLocal, results);
  };
  auto funcParseRes = function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  if (exported)
    result.addAttribute(getExportedAttrName(result.name), UnitAttr::get(ctx));
  return funcParseRes;
}

LogicalResult FuncOp::verifyBody() {
  if (getBody().empty())
    return success();
  Block &entry = getBody().front();
  if (entry.getNumArguments() != getFunctionType().getNumInputs())
    return emitError("entry block should have same number of arguments as "
                     "function type. Function type has ")
           << getFunctionType().getNumInputs() << ", entry block has "
           << entry.getNumArguments();

  for (auto [argNo, funcSignatureType, blockType] : llvm::enumerate(
           getFunctionType().getInputs(), entry.getArgumentTypes())) {
    auto blockLocalRefType = dyn_cast<LocalRefType>(blockType);
    if (!blockLocalRefType)
      return emitError("entry block argument type should be LocalRefType, got ")
             << blockType << " for block argument " << argNo;
    if (blockLocalRefType.getElementType() != funcSignatureType)
      return emitError("func argument type #")
             << argNo << "(" << funcSignatureType
             << ") doesn't match entry block referenced type ("
             << blockLocalRefType.getElementType() << ")";
  }
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  /// If exported, print it before and mask it before printing
  /// using generic interface.
  auto exported = getExported();
  if (exported) {
    p << " exported";
    removeExportedAttr();
  }
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
  if (exported)
    setExported(true);
}

//===----------------------------------------------------------------------===//
// FuncImportOp
//===----------------------------------------------------------------------===//

void FuncImportOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         StringRef symbol, StringRef moduleName,
                         StringRef importName, FunctionType type) {
  FuncImportOp::build(odsBuilder, odsState, symbol, moduleName, importName,
                      type, {}, {});
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//
// Custom formats
ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbolName;
  Type globalType;
  auto *ctx = parser.getContext();
  std::string visibilityString;
  auto loc = parser.getNameLoc();
  ParseResult res = parser.parseOptionalKeywordOrString(&visibilityString);
  if (res.succeeded()) {
    if (visibilityString != "exported")
      return parser.emitError(
                 loc, "expecting either `exported` or symbol name. got ")
             << visibilityString;
    result.addAttribute(getExportedAttrName(result.name), UnitAttr::get(ctx));
  }

  res = parser.parseSymbolName(symbolName, SymbolTable::getSymbolAttrName(),
                               result.attributes);
  res = parser.parseType(globalType);
  result.addAttribute(getTypeAttrName(result.name), TypeAttr::get(globalType));
  std::string mutableString;
  res = parser.parseOptionalKeywordOrString(&mutableString);
  if (res.succeeded() && mutableString == "mutable")
    result.addAttribute("isMutable", UnitAttr::get(ctx));

  res = parser.parseColon();
  Region *globalInitRegion = result.addRegion();
  res = parser.parseRegion(*globalInitRegion);
  return res;
}

void GlobalOp::print(OpAsmPrinter &printer) {
  if (getExported())
    printer << " exported";
  printer << " @" << getSymName().str() << " " << getType();
  if (getIsMutable())
    printer << " mutable";
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

LogicalResult
GlobalGetOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If the parent requires a constant context, verify that global.get is a
  // constant as defined per the wasm standard.
  if (!this->getOperation()
           ->getParentWithTrait<ConstantExpressionInitializerOpTrait>())
    return success();
  Operation *symTabOp = SymbolTable::getNearestSymbolTable(*this);
  StringRef referencedSymbol = getGlobal();
  Operation *definitionOp = symbolTable.lookupSymbolIn(
      symTabOp, StringAttr::get(this->getContext(), referencedSymbol));
  if (!definitionOp)
    return emitError() << "symbol @" << referencedSymbol << " is undefined";
  auto definitionImport = dyn_cast<GlobalImportOp>(definitionOp);
  if (!definitionImport || definitionImport.getIsMutable()) {
    return emitError("global.get op is considered constant if it's referring "
                     "to a import.global symbol marked non-mutable");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalImportOp
//===----------------------------------------------------------------------===//

ParseResult GlobalImportOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *ctx = parser.getContext();
  ParseResult res = parseImportOp(parser, result);
  if (res.failed())
    return failure();
  std::string mutableOrSymVisString;
  res = parser.parseOptionalKeywordOrString(&mutableOrSymVisString);
  if (res.succeeded() && mutableOrSymVisString == "mutable") {
    result.addAttribute("isMutable", UnitAttr::get(ctx));
  }

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
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
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
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTeeGetResType(operands, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// LocalSetOp
//===----------------------------------------------------------------------===//

LogicalResult LocalSetOp::verify() {
  if (getLocalVar().getType().getElementType() != getValue().getType())
    return emitError("input type and result type of local.set do not match");
  return success();
}

//===----------------------------------------------------------------------===//
// LocalTeeOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTeeOp::inferReturnTypes(
    MLIRContext *context, ::std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTeeGetResType(operands, inferredReturnTypes);
}

LogicalResult LocalTeeOp::verify() {
  if (getLocalVar().getType().getElementType() != getValue().getType() ||
      getValue().getType() != getResult().getType())
    return emitError("input type and output type of local.tee do not match");
  return success();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

Block *LoopOp::getLabelTarget() { return &getBody().front(); }

//===----------------------------------------------------------------------===//
// ReinterpretOp
//===----------------------------------------------------------------------===//

LogicalResult ReinterpretOp::verify() {
  auto inT = getInput().getType();
  auto resT = getResult().getType();
  if (inT == resT)
    return emitError("reinterpret input and output type should be distinct");
  if (inT.getIntOrFloatBitWidth() != resT.getIntOrFloatBitWidth())
    return emitError() << "input type (" << inT << ") and output type (" << resT
                       << ") have incompatible bit widths";
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}
