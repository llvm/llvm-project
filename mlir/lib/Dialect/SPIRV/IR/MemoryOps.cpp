//===- MemoryOps.cpp - MLIR SPIR-V Memory Ops  ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the memory operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

/// Parses optional memory access (a.k.a. memory operand) attributes attached to
/// a memory access operand/pointer. Specifically, parses the following syntax:
///     (`[` memory-access `]`)?
/// where:
///     memory-access ::= `"None"` | `"Volatile"` | `"Aligned", `
///         integer-literal | `"NonTemporal"`
template <typename MemoryOpTy>
ParseResult parseMemoryAccessAttributes(OpAsmParser &parser,
                                        OperationState &state) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  StringAttr memoryAccessAttrName =
      MemoryOpTy::getMemoryAccessAttrName(state.name);
  if (spirv::parseEnumStrAttr<spirv::MemoryAccessAttr>(
          memoryAccessAttr, parser, state, memoryAccessAttrName))
    return failure();

  if (spirv::bitEnumContainsAll(memoryAccessAttr,
                                spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    StringAttr alignmentAttrName = MemoryOpTy::getAlignmentAttrName(state.name);
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type, alignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
template <typename MemoryOpTy>
static ParseResult parseSourceMemoryAccessAttributes(OpAsmParser &parser,
                                                     OperationState &state) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  StringRef memoryAccessAttrName =
      MemoryOpTy::getSourceMemoryAccessAttrName(state.name);
  if (spirv::parseEnumStrAttr<spirv::MemoryAccessAttr>(
          memoryAccessAttr, parser, state, memoryAccessAttrName))
    return failure();

  if (spirv::bitEnumContainsAll(memoryAccessAttr,
                                spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    StringAttr alignmentAttrName =
        MemoryOpTy::getSourceAlignmentAttrName(state.name);
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type, alignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
template <typename MemoryOpTy>
static void printSourceMemoryAccessAttribute(
    MemoryOpTy memoryOp, OpAsmPrinter &printer,
    SmallVectorImpl<StringRef> &elidedAttrs,
    std::optional<spirv::MemoryAccess> memoryAccessAtrrValue = std::nullopt,
    std::optional<uint32_t> alignmentAttrValue = std::nullopt) {

  printer << ", ";

  // Print optional memory access attribute.
  if (auto memAccess = (memoryAccessAtrrValue ? memoryAccessAtrrValue
                                              : memoryOp.getMemoryAccess())) {
    elidedAttrs.push_back(memoryOp.getSourceMemoryAccessAttrName());

    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    if (spirv::bitEnumContainsAll(*memAccess, spirv::MemoryAccess::Aligned)) {
      // Print integer alignment attribute.
      if (auto alignment = (alignmentAttrValue ? alignmentAttrValue
                                               : memoryOp.getAlignment())) {
        elidedAttrs.push_back(memoryOp.getSourceAlignmentAttrName());
        printer << ", " << *alignment;
      }
    }
    printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

template <typename MemoryOpTy>
static void printMemoryAccessAttribute(
    MemoryOpTy memoryOp, OpAsmPrinter &printer,
    SmallVectorImpl<StringRef> &elidedAttrs,
    std::optional<spirv::MemoryAccess> memoryAccessAtrrValue = std::nullopt,
    std::optional<uint32_t> alignmentAttrValue = std::nullopt) {
  // Print optional memory access attribute.
  if (auto memAccess = (memoryAccessAtrrValue ? memoryAccessAtrrValue
                                              : memoryOp.getMemoryAccess())) {
    elidedAttrs.push_back(memoryOp.getMemoryAccessAttrName());

    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    if (spirv::bitEnumContainsAll(*memAccess, spirv::MemoryAccess::Aligned)) {
      // Print integer alignment attribute.
      if (auto alignment = (alignmentAttrValue ? alignmentAttrValue
                                               : memoryOp.getAlignment())) {
        elidedAttrs.push_back(memoryOp.getAlignmentAttrName());
        printer << ", " << *alignment;
      }
    }
    printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

template <typename LoadStoreOpTy>
static LogicalResult verifyLoadStorePtrAndValTypes(LoadStoreOpTy op, Value ptr,
                                                   Value val) {
  // ODS already checks ptr is spirv::PointerType. Just check that the pointee
  // type of the pointer and the type of the value are the same
  //
  // TODO: Check that the value type satisfies restrictions of
  // SPIR-V OpLoad/OpStore operations
  if (val.getType() !=
      llvm::cast<spirv::PointerType>(ptr.getType()).getPointeeType()) {
    return op.emitOpError("mismatch in result type and pointer type");
  }
  return success();
}

template <typename MemoryOpTy>
static LogicalResult verifyMemoryAccessAttribute(MemoryOpTy memoryOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = memoryOp.getOperation();
  auto memAccessAttr = op->getAttr(memoryOp.getMemoryAccessAttrName());
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(memoryOp.getAlignmentAttrName())) {
      return memoryOp.emitOpError(
          "invalid alignment specification without aligned memory access "
          "specification");
    }
    return success();
  }

  auto memAccess = llvm::cast<spirv::MemoryAccessAttr>(memAccessAttr);

  if (!memAccess) {
    return memoryOp.emitOpError("invalid memory access specifier: ")
           << memAccessAttr;
  }

  if (spirv::bitEnumContainsAll(memAccess.getValue(),
                                spirv::MemoryAccess::Aligned)) {
    if (!op->getAttr(memoryOp.getAlignmentAttrName())) {
      return memoryOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(memoryOp.getAlignmentAttrName())) {
      return memoryOp.emitOpError(
          "invalid alignment specification with non-aligned memory access "
          "specification");
    }
  }
  return success();
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
template <typename MemoryOpTy>
static LogicalResult verifySourceMemoryAccessAttribute(MemoryOpTy memoryOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = memoryOp.getOperation();
  auto memAccessAttr = op->getAttr(memoryOp.getSourceMemoryAccessAttrName());
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(memoryOp.getSourceAlignmentAttrName())) {
      return memoryOp.emitOpError(
          "invalid alignment specification without aligned memory access "
          "specification");
    }
    return success();
  }

  auto memAccess = llvm::cast<spirv::MemoryAccessAttr>(memAccessAttr);

  if (!memAccess) {
    return memoryOp.emitOpError("invalid memory access specifier: ")
           << memAccess;
  }

  if (spirv::bitEnumContainsAll(memAccess.getValue(),
                                spirv::MemoryAccess::Aligned)) {
    if (!op->getAttr(memoryOp.getSourceAlignmentAttrName())) {
      return memoryOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(memoryOp.getSourceAlignmentAttrName())) {
      return memoryOp.emitOpError(
          "invalid alignment specification with non-aligned memory access "
          "specification");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.AccessChainOp
//===----------------------------------------------------------------------===//

static Type getElementPtrType(Type type, ValueRange indices, Location baseLoc) {
  auto ptrType = llvm::dyn_cast<spirv::PointerType>(type);
  if (!ptrType) {
    emitError(baseLoc, "'spirv.AccessChain' op expected a pointer "
                       "to composite type, but provided ")
        << type;
    return nullptr;
  }

  auto resultType = ptrType.getPointeeType();
  auto resultStorageClass = ptrType.getStorageClass();
  int32_t index = 0;

  for (auto indexSSA : indices) {
    auto cType = llvm::dyn_cast<spirv::CompositeType>(resultType);
    if (!cType) {
      emitError(
          baseLoc,
          "'spirv.AccessChain' op cannot extract from non-composite type ")
          << resultType << " with index " << index;
      return nullptr;
    }
    index = 0;
    if (llvm::isa<spirv::StructType>(resultType)) {
      Operation *op = indexSSA.getDefiningOp();
      if (!op) {
        emitError(baseLoc, "'spirv.AccessChain' op index must be an "
                           "integer spirv.Constant to access "
                           "element of spirv.struct");
        return nullptr;
      }

      // TODO: this should be relaxed to allow
      // integer literals of other bitwidths.
      if (failed(spirv::extractValueFromConstOp(op, index))) {
        emitError(
            baseLoc,
            "'spirv.AccessChain' index must be an integer spirv.Constant to "
            "access element of spirv.struct, but provided ")
            << op->getName();
        return nullptr;
      }
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        emitError(baseLoc, "'spirv.AccessChain' op index ")
            << index << " out of bounds for " << resultType;
        return nullptr;
      }
    }
    resultType = cType.getElementType(index);
  }
  return spirv::PointerType::get(resultType, resultStorageClass);
}

void AccessChainOp::build(OpBuilder &builder, OperationState &state,
                          Value basePtr, ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, indices);
}

ParseResult AccessChainOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand ptrInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indicesInfo;
  Type type;
  auto loc = parser.getCurrentLocation();
  SmallVector<Type, 4> indicesTypes;

  if (parser.parseOperand(ptrInfo) ||
      parser.parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptrInfo, type, result.operands)) {
    return failure();
  }

  // Check that the provided indices list is not empty before parsing their
  // type list.
  if (indicesInfo.empty()) {
    return mlir::emitError(result.location,
                           "'spirv.AccessChain' op expected at "
                           "least one index ");
  }

  if (parser.parseComma() || parser.parseTypeList(indicesTypes))
    return failure();

  // Check that the indices types list is not empty and that it has a one-to-one
  // mapping to the provided indices.
  if (indicesTypes.size() != indicesInfo.size()) {
    return mlir::emitError(
        result.location, "'spirv.AccessChain' op indices types' count must be "
                         "equal to indices info count");
  }

  if (parser.resolveOperands(indicesInfo, indicesTypes, loc, result.operands))
    return failure();

  auto resultType = getElementPtrType(
      type, llvm::ArrayRef(result.operands).drop_front(), result.location);
  if (!resultType) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

template <typename Op>
static void printAccessChain(Op op, ValueRange indices, OpAsmPrinter &printer) {
  printer << ' ' << op.getBasePtr() << '[' << indices
          << "] : " << op.getBasePtr().getType() << ", " << indices.getTypes();
}

void spirv::AccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, getIndices(), printer);
}

template <typename Op>
static LogicalResult verifyAccessChain(Op accessChainOp, ValueRange indices) {
  auto resultType = getElementPtrType(accessChainOp.getBasePtr().getType(),
                                      indices, accessChainOp.getLoc());
  if (!resultType)
    return failure();

  auto providedResultType =
      llvm::dyn_cast<spirv::PointerType>(accessChainOp.getType());
  if (!providedResultType)
    return accessChainOp.emitOpError(
               "result type must be a pointer, but provided")
           << providedResultType;

  if (resultType != providedResultType)
    return accessChainOp.emitOpError("invalid result type: expected ")
           << resultType << ", but provided " << providedResultType;

  return success();
}

LogicalResult AccessChainOp::verify() {
  return verifyAccessChain(*this, getIndices());
}

//===----------------------------------------------------------------------===//
// spirv.LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &builder, OperationState &state, Value basePtr,
                   MemoryAccessAttr memoryAccess, IntegerAttr alignment) {
  auto ptrType = llvm::cast<spirv::PointerType>(basePtr.getType());
  build(builder, state, ptrType.getPointeeType(), basePtr, memoryAccess,
        alignment);
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::UnresolvedOperand ptrInfo;
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) || parser.parseOperand(ptrInfo) ||
      parseMemoryAccessAttributes<LoadOp>(parser, result) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperand(ptrInfo, ptrType, result.operands)) {
    return failure();
  }

  result.addTypes(elementType);
  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      llvm::cast<spirv::PointerType>(getPtr().getType()).getStorageClass());
  printer << " \"" << sc << "\" " << getPtr();

  printMemoryAccessAttribute(*this, printer, elidedAttrs);

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << " : " << getType();
}

LogicalResult LoadOp::verify() {
  // SPIR-V spec : "Result Type is the type of the loaded object. It must be a
  // type with fixed size; i.e., it cannot be, nor include, any
  // OpTypeRuntimeArray types."
  if (failed(verifyLoadStorePtrAndValTypes(*this, getPtr(), getValue()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(*this);
}

//===----------------------------------------------------------------------===//
// spirv.StoreOp
//===----------------------------------------------------------------------===//

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfo;
  auto loc = parser.getCurrentLocation();
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) ||
      parser.parseOperandList(operandInfo, 2) ||
      parseMemoryAccessAttributes<StoreOp>(parser, result) ||
      parser.parseColon() || parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperands(operandInfo, {ptrType, elementType}, loc,
                             result.operands)) {
    return failure();
  }
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      llvm::cast<spirv::PointerType>(getPtr().getType()).getStorageClass());
  printer << " \"" << sc << "\" " << getPtr() << ", " << getValue();

  printMemoryAccessAttribute(*this, printer, elidedAttrs);

  printer << " : " << getValue().getType();
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult StoreOp::verify() {
  // SPIR-V spec : "Pointer is the pointer to store through. Its type must be an
  // OpTypePointer whose Type operand is the same as the type of Object."
  if (failed(verifyLoadStorePtrAndValTypes(*this, getPtr(), getValue())))
    return failure();
  return verifyMemoryAccessAttribute(*this);
}

//===----------------------------------------------------------------------===//
// spirv.CopyMemory
//===----------------------------------------------------------------------===//

void CopyMemoryOp::print(OpAsmPrinter &printer) {
  printer << ' ';

  StringRef targetStorageClass = stringifyStorageClass(
      llvm::cast<spirv::PointerType>(getTarget().getType()).getStorageClass());
  printer << " \"" << targetStorageClass << "\" " << getTarget() << ", ";

  StringRef sourceStorageClass = stringifyStorageClass(
      llvm::cast<spirv::PointerType>(getSource().getType()).getStorageClass());
  printer << " \"" << sourceStorageClass << "\" " << getSource();

  SmallVector<StringRef, 4> elidedAttrs;
  printMemoryAccessAttribute(*this, printer, elidedAttrs);
  printSourceMemoryAccessAttribute(*this, printer, elidedAttrs,
                                   getSourceMemoryAccess(),
                                   getSourceAlignment());

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  Type pointeeType =
      llvm::cast<spirv::PointerType>(getTarget().getType()).getPointeeType();
  printer << " : " << pointeeType;
}

ParseResult CopyMemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  spirv::StorageClass targetStorageClass;
  OpAsmParser::UnresolvedOperand targetPtrInfo;

  spirv::StorageClass sourceStorageClass;
  OpAsmParser::UnresolvedOperand sourcePtrInfo;

  Type elementType;

  if (parseEnumStrAttr(targetStorageClass, parser) ||
      parser.parseOperand(targetPtrInfo) || parser.parseComma() ||
      parseEnumStrAttr(sourceStorageClass, parser) ||
      parser.parseOperand(sourcePtrInfo) ||
      parseMemoryAccessAttributes<CopyMemoryOp>(parser, result)) {
    return failure();
  }

  if (!parser.parseOptionalComma()) {
    // Parse 2nd memory access attributes.
    if (parseSourceMemoryAccessAttributes<CopyMemoryOp>(parser, result)) {
      return failure();
    }
  }

  if (parser.parseColon() || parser.parseType(elementType))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto targetPtrType = spirv::PointerType::get(elementType, targetStorageClass);
  auto sourcePtrType = spirv::PointerType::get(elementType, sourceStorageClass);

  if (parser.resolveOperand(targetPtrInfo, targetPtrType, result.operands) ||
      parser.resolveOperand(sourcePtrInfo, sourcePtrType, result.operands)) {
    return failure();
  }

  return success();
}

LogicalResult CopyMemoryOp::verify() {
  Type targetType =
      llvm::cast<spirv::PointerType>(getTarget().getType()).getPointeeType();

  Type sourceType =
      llvm::cast<spirv::PointerType>(getSource().getType()).getPointeeType();

  if (targetType != sourceType)
    return emitOpError("both operands must be pointers to the same type");

  if (failed(verifyMemoryAccessAttribute(*this)))
    return failure();

  // TODO - According to the spec:
  //
  // If two masks are present, the first applies to Target and cannot include
  // MakePointerVisible, and the second applies to Source and cannot include
  // MakePointerAvailable.
  //
  // Add such verification here.

  return verifySourceMemoryAccessAttribute(*this);
}

static ParseResult parsePtrAccessChainOpImpl(StringRef opName,
                                             OpAsmParser &parser,
                                             OperationState &state) {
  OpAsmParser::UnresolvedOperand ptrInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indicesInfo;
  Type type;
  auto loc = parser.getCurrentLocation();
  SmallVector<Type, 4> indicesTypes;

  if (parser.parseOperand(ptrInfo) ||
      parser.parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptrInfo, type, state.operands))
    return failure();

  // Check that the provided indices list is not empty before parsing their
  // type list.
  if (indicesInfo.empty())
    return emitError(state.location) << opName << " expected element";

  if (parser.parseComma() || parser.parseTypeList(indicesTypes))
    return failure();

  // Check that the indices types list is not empty and that it has a one-to-one
  // mapping to the provided indices.
  if (indicesTypes.size() != indicesInfo.size())
    return emitError(state.location)
           << opName
           << " indices types' count must be equal to indices info count";

  if (parser.resolveOperands(indicesInfo, indicesTypes, loc, state.operands))
    return failure();

  auto resultType = getElementPtrType(
      type, llvm::ArrayRef(state.operands).drop_front(2), state.location);
  if (!resultType)
    return failure();

  state.addTypes(resultType);
  return success();
}

template <typename Op>
static auto concatElemAndIndices(Op op) {
  SmallVector<Value> ret(op.getIndices().size() + 1);
  ret[0] = op.getElement();
  llvm::copy(op.getIndices(), ret.begin() + 1);
  return ret;
}

//===----------------------------------------------------------------------===//
// spirv.InBoundsPtrAccessChainOp
//===----------------------------------------------------------------------===//

void InBoundsPtrAccessChainOp::build(OpBuilder &builder, OperationState &state,
                                     Value basePtr, Value element,
                                     ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, element, indices);
}

ParseResult InBoundsPtrAccessChainOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  return parsePtrAccessChainOpImpl(
      spirv::InBoundsPtrAccessChainOp::getOperationName(), parser, result);
}

void InBoundsPtrAccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, concatElemAndIndices(*this), printer);
}

LogicalResult InBoundsPtrAccessChainOp::verify() {
  return verifyAccessChain(*this, getIndices());
}

//===----------------------------------------------------------------------===//
// spirv.PtrAccessChainOp
//===----------------------------------------------------------------------===//

void PtrAccessChainOp::build(OpBuilder &builder, OperationState &state,
                             Value basePtr, Value element, ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, element, indices);
}

ParseResult PtrAccessChainOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parsePtrAccessChainOpImpl(spirv::PtrAccessChainOp::getOperationName(),
                                   parser, result);
}

void PtrAccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, concatElemAndIndices(*this), printer);
}

LogicalResult PtrAccessChainOp::verify() {
  return verifyAccessChain(*this, getIndices());
}

//===----------------------------------------------------------------------===//
// spirv.Variable
//===----------------------------------------------------------------------===//

ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional initializer
  std::optional<OpAsmParser::UnresolvedOperand> initInfo;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    initInfo = OpAsmParser::UnresolvedOperand();
    if (parser.parseLParen() || parser.parseOperand(*initInfo) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, result)) {
    return failure();
  }

  // Parse result pointer type
  Type type;
  if (parser.parseColon())
    return failure();
  auto loc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return failure();

  auto ptrType = llvm::dyn_cast<spirv::PointerType>(type);
  if (!ptrType)
    return parser.emitError(loc, "expected spirv.ptr type");
  result.addTypes(ptrType);

  // Resolve the initializer operand
  if (initInfo) {
    if (parser.resolveOperand(*initInfo, ptrType.getPointeeType(),
                              result.operands))
      return failure();
  }

  auto attr = parser.getBuilder().getAttr<spirv::StorageClassAttr>(
      ptrType.getStorageClass());
  result.addAttribute(spirv::attributeName<spirv::StorageClass>(), attr);

  return success();
}

void VariableOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  // Print optional initializer
  if (getNumOperands() != 0)
    printer << " init(" << getInitializer() << ")";

  printVariableDecorations(*this, printer, elidedAttrs);
  printer << " : " << getType();
}

LogicalResult VariableOp::verify() {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (getStorageClass() != spirv::StorageClass::Function) {
    return emitOpError(
        "can only be used to model function-level variables. Use "
        "spirv.GlobalVariable for module-level variables.");
  }

  auto pointerType = llvm::cast<spirv::PointerType>(getPointer().getType());
  if (getStorageClass() != pointerType.getStorageClass())
    return emitOpError(
        "storage class must match result pointer's storage class");

  if (getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    auto *initOp = getOperand(0).getDefiningOp();
    if (!initOp || !isa<spirv::ConstantOp,    // for normal constant
                        spirv::ReferenceOfOp, // for spec constant
                        spirv::AddressOfOp>(initOp))
      return emitOpError("initializer must be the result of a "
                         "constant or spirv.GlobalVariable op");
  }

  auto getDecorationAttr = [op = getOperation()](spirv::Decoration decoration) {
    return op->getAttr(
        llvm::convertToSnakeFromCamelCase(stringifyDecoration(decoration)));
  };

  // TODO: generate these strings using ODS.
  for (auto decoration :
       {spirv::Decoration::DescriptorSet, spirv::Decoration::Binding,
        spirv::Decoration::BuiltIn}) {
    if (auto attr = getDecorationAttr(decoration))
      return emitOpError("cannot have '")
             << llvm::convertToSnakeFromCamelCase(
                    stringifyDecoration(decoration))
             << "' attribute (only allowed in spirv.GlobalVariable)";
  }

  // From SPV_KHR_physical_storage_buffer:
  // > If an OpVariable's pointee type is a pointer (or array of pointers) in
  // > PhysicalStorageBuffer storage class, then the variable must be decorated
  // > with exactly one of AliasedPointer or RestrictPointer.
  auto pointeePtrType = dyn_cast<spirv::PointerType>(getPointeeType());
  if (!pointeePtrType) {
    if (auto pointeeArrayType = dyn_cast<spirv::ArrayType>(getPointeeType())) {
      pointeePtrType =
          dyn_cast<spirv::PointerType>(pointeeArrayType.getElementType());
    }
  }

  if (pointeePtrType && pointeePtrType.getStorageClass() ==
                            spirv::StorageClass::PhysicalStorageBuffer) {
    bool hasAliasedPtr =
        getDecorationAttr(spirv::Decoration::AliasedPointer) != nullptr;
    bool hasRestrictPtr =
        getDecorationAttr(spirv::Decoration::RestrictPointer) != nullptr;

    if (!hasAliasedPtr && !hasRestrictPtr)
      return emitOpError() << " with physical buffer pointer must be decorated "
                              "either 'AliasedPointer' or 'RestrictPointer'";

    if (hasAliasedPtr && hasRestrictPtr)
      return emitOpError()
             << " with physical buffer pointer must have exactly one "
                "aliasing decoration";
  }

  return success();
}

} // namespace mlir::spirv
