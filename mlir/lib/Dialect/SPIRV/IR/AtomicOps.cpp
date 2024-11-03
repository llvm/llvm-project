//===- AtomicOps.cpp - MLIR SPIR-V Atomic Ops  ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the atomic operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

// Parses an atomic update op. If the update op does not take a value (like
// AtomicIIncrement) `hasValue` must be false.
static ParseResult parseAtomicUpdateOp(OpAsmParser &parser,
                                       OperationState &state, bool hasValue) {
  spirv::Scope scope;
  spirv::MemorySemantics memoryScope;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfo;
  OpAsmParser::UnresolvedOperand ptrInfo, valueInfo;
  Type type;
  SMLoc loc;
  if (parseEnumStrAttr<spirv::ScopeAttr>(scope, parser, state,
                                         kMemoryScopeAttrName) ||
      parseEnumStrAttr<spirv::MemorySemanticsAttr>(memoryScope, parser, state,
                                                   kSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, (hasValue ? 2 : 1)) ||
      parser.getCurrentLocation(&loc) || parser.parseColonType(type))
    return failure();

  auto ptrType = llvm::dyn_cast<spirv::PointerType>(type);
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  SmallVector<Type, 2> operandTypes;
  operandTypes.push_back(ptrType);
  if (hasValue)
    operandTypes.push_back(ptrType.getPointeeType());
  if (parser.resolveOperands(operandInfo, operandTypes, parser.getNameLoc(),
                             state.operands))
    return failure();
  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

// Prints an atomic update op.
static void printAtomicUpdateOp(Operation *op, OpAsmPrinter &printer) {
  printer << " \"";
  auto scopeAttr = op->getAttrOfType<spirv::ScopeAttr>(kMemoryScopeAttrName);
  printer << spirv::stringifyScope(scopeAttr.getValue()) << "\" \"";
  auto memorySemanticsAttr =
      op->getAttrOfType<spirv::MemorySemanticsAttr>(kSemanticsAttrName);
  printer << spirv::stringifyMemorySemantics(memorySemanticsAttr.getValue())
          << "\" " << op->getOperands() << " : " << op->getOperand(0).getType();
}

template <typename T>
static StringRef stringifyTypeName();

template <>
StringRef stringifyTypeName<IntegerType>() {
  return "integer";
}

template <>
StringRef stringifyTypeName<FloatType>() {
  return "float";
}

// Verifies an atomic update op.
template <typename ExpectedElementType>
static LogicalResult verifyAtomicUpdateOp(Operation *op) {
  auto ptrType = llvm::cast<spirv::PointerType>(op->getOperand(0).getType());
  auto elementType = ptrType.getPointeeType();
  if (!llvm::isa<ExpectedElementType>(elementType))
    return op->emitOpError() << "pointer operand must point to an "
                             << stringifyTypeName<ExpectedElementType>()
                             << " value, found " << elementType;

  if (op->getNumOperands() > 1) {
    auto valueType = op->getOperand(1).getType();
    if (valueType != elementType)
      return op->emitOpError("expected value to have the same type as the "
                             "pointer operand's pointee type ")
             << elementType << ", but found " << valueType;
  }
  auto memorySemantics =
      op->getAttrOfType<spirv::MemorySemanticsAttr>(kSemanticsAttrName)
          .getValue();
  if (failed(verifyMemorySemantics(op, memorySemantics))) {
    return failure();
  }
  return success();
}

template <typename T>
static void printAtomicCompareExchangeImpl(T atomOp, OpAsmPrinter &printer) {
  printer << " \"" << stringifyScope(atomOp.getMemoryScope()) << "\" \""
          << stringifyMemorySemantics(atomOp.getEqualSemantics()) << "\" \""
          << stringifyMemorySemantics(atomOp.getUnequalSemantics()) << "\" "
          << atomOp.getOperands() << " : " << atomOp.getPointer().getType();
}

static ParseResult parseAtomicCompareExchangeImpl(OpAsmParser &parser,
                                                  OperationState &state) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics equalSemantics, unequalSemantics;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operandInfo;
  Type type;
  if (parseEnumStrAttr<spirv::ScopeAttr>(memoryScope, parser, state,
                                         kMemoryScopeAttrName) ||
      parseEnumStrAttr<spirv::MemorySemanticsAttr>(
          equalSemantics, parser, state, kEqualSemanticsAttrName) ||
      parseEnumStrAttr<spirv::MemorySemanticsAttr>(
          unequalSemantics, parser, state, kUnequalSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, 3))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type))
    return failure();

  auto ptrType = llvm::dyn_cast<spirv::PointerType>(type);
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  if (parser.resolveOperands(
          operandInfo,
          {ptrType, ptrType.getPointeeType(), ptrType.getPointeeType()},
          parser.getNameLoc(), state.operands))
    return failure();

  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

template <typename T>
static LogicalResult verifyAtomicCompareExchangeImpl(T atomOp) {
  // According to the spec:
  // "The type of Value must be the same as Result Type. The type of the value
  // pointed to by Pointer must be the same as Result Type. This type must also
  // match the type of Comparator."
  if (atomOp.getType() != atomOp.getValue().getType())
    return atomOp.emitOpError("value operand must have the same type as the op "
                              "result, but found ")
           << atomOp.getValue().getType() << " vs " << atomOp.getType();

  if (atomOp.getType() != atomOp.getComparator().getType())
    return atomOp.emitOpError(
               "comparator operand must have the same type as the op "
               "result, but found ")
           << atomOp.getComparator().getType() << " vs " << atomOp.getType();

  Type pointeeType =
      llvm::cast<spirv::PointerType>(atomOp.getPointer().getType())
          .getPointeeType();
  if (atomOp.getType() != pointeeType)
    return atomOp.emitOpError(
               "pointer operand's pointee type must have the same "
               "as the op result type, but found ")
           << pointeeType << " vs " << atomOp.getType();

  // TODO: Unequal cannot be set to Release or Acquire and Release.
  // In addition, Unequal cannot be set to a stronger memory-order then Equal.

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.AtomicAndOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicAndOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicAndOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicAndOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicCompareExchangeOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicCompareExchangeOp::verify() {
  return verifyAtomicCompareExchangeImpl(*this);
}

ParseResult AtomicCompareExchangeOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseAtomicCompareExchangeImpl(parser, result);
}

void AtomicCompareExchangeOp::print(OpAsmPrinter &p) {
  printAtomicCompareExchangeImpl(*this, p);
}

//===----------------------------------------------------------------------===//
// spirv.AtomicCompareExchangeWeakOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicCompareExchangeWeakOp::verify() {
  return verifyAtomicCompareExchangeImpl(*this);
}

ParseResult AtomicCompareExchangeWeakOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseAtomicCompareExchangeImpl(parser, result);
}

void AtomicCompareExchangeWeakOp::print(OpAsmPrinter &p) {
  printAtomicCompareExchangeImpl(*this, p);
}

//===----------------------------------------------------------------------===//
// spirv.AtomicExchange
//===----------------------------------------------------------------------===//

void AtomicExchangeOp::print(OpAsmPrinter &printer) {
  printer << " \"" << stringifyScope(getMemoryScope()) << "\" \""
          << stringifyMemorySemantics(getSemantics()) << "\" " << getOperands()
          << " : " << getPointer().getType();
}

ParseResult AtomicExchangeOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics semantics;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfo;
  Type type;
  if (parseEnumStrAttr<spirv::ScopeAttr>(memoryScope, parser, result,
                                         kMemoryScopeAttrName) ||
      parseEnumStrAttr<spirv::MemorySemanticsAttr>(semantics, parser, result,
                                                   kSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, 2))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type))
    return failure();

  auto ptrType = llvm::dyn_cast<spirv::PointerType>(type);
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  if (parser.resolveOperands(operandInfo, {ptrType, ptrType.getPointeeType()},
                             parser.getNameLoc(), result.operands))
    return failure();

  return parser.addTypeToList(ptrType.getPointeeType(), result.types);
}

LogicalResult AtomicExchangeOp::verify() {
  if (getType() != getValue().getType())
    return emitOpError("value operand must have the same type as the op "
                       "result, but found ")
           << getValue().getType() << " vs " << getType();

  Type pointeeType =
      llvm::cast<spirv::PointerType>(getPointer().getType()).getPointeeType();
  if (getType() != pointeeType)
    return emitOpError("pointer operand's pointee type must have the same "
                       "as the op result type, but found ")
           << pointeeType << " vs " << getType();

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIAddOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIAddOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicIAddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicIAddOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.EXT.AtomicFAddOp
//===----------------------------------------------------------------------===//

LogicalResult EXTAtomicFAddOp::verify() {
  return verifyAtomicUpdateOp<FloatType>(getOperation());
}

ParseResult EXTAtomicFAddOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void spirv::EXTAtomicFAddOp::print(OpAsmPrinter &p) {
  printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIDecrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIDecrementOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicIDecrementOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseAtomicUpdateOp(parser, result, false);
}

void AtomicIDecrementOp::print(OpAsmPrinter &p) {
  printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIIncrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIIncrementOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicIIncrementOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseAtomicUpdateOp(parser, result, false);
}

void AtomicIIncrementOp::print(OpAsmPrinter &p) {
  printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spirv.AtomicISubOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicISubOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicISubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicISubOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicOrOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicOrOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicOrOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicOrOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMaxOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicSMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicSMaxOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicSMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMinOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicSMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicSMinOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMaxOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicUMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicUMaxOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicUMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMinOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicUMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicUMinOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.AtomicXorOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicXorOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult AtomicXorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAtomicUpdateOp(parser, result, true);
}

void AtomicXorOp::print(OpAsmPrinter &p) { printAtomicUpdateOp(*this, p); }

} // namespace mlir::spirv
