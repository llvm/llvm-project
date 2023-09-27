//===- SPIRVOps.cpp - MLIR SPIR-V operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::spirv::AttrNames;

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

LogicalResult spirv::extractValueFromConstOp(Operation *op, int32_t &value) {
  auto constOp = dyn_cast_or_null<spirv::ConstantOp>(op);
  if (!constOp) {
    return failure();
  }
  auto valueAttr = constOp.getValue();
  auto integerValueAttr = llvm::dyn_cast<IntegerAttr>(valueAttr);
  if (!integerValueAttr) {
    return failure();
  }

  if (integerValueAttr.getType().isSignlessInteger())
    value = integerValueAttr.getInt();
  else
    value = integerValueAttr.getSInt();

  return success();
}

LogicalResult
spirv::verifyMemorySemantics(Operation *op,
                             spirv::MemorySemantics memorySemantics) {
  // According to the SPIR-V specification:
  // "Despite being a mask and allowing multiple bits to be combined, it is
  // invalid for more than one of these four bits to be set: Acquire, Release,
  // AcquireRelease, or SequentiallyConsistent. Requesting both Acquire and
  // Release semantics is done by setting the AcquireRelease bit, not by setting
  // two bits."
  auto atMostOneInSet = spirv::MemorySemantics::Acquire |
                        spirv::MemorySemantics::Release |
                        spirv::MemorySemantics::AcquireRelease |
                        spirv::MemorySemantics::SequentiallyConsistent;

  auto bitCount =
      llvm::popcount(static_cast<uint32_t>(memorySemantics & atMostOneInSet));
  if (bitCount > 1) {
    return op->emitError(
        "expected at most one of these four memory constraints "
        "to be set: `Acquire`, `Release`,"
        "`AcquireRelease` or `SequentiallyConsistent`");
  }
  return success();
}

void spirv::printVariableDecorations(Operation *op, OpAsmPrinter &printer,
                                     SmallVectorImpl<StringRef> &elidedAttrs) {
  // Print optional descriptor binding
  auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::Binding));
  auto descriptorSet = op->getAttrOfType<IntegerAttr>(descriptorSetName);
  auto binding = op->getAttrOfType<IntegerAttr>(bindingName);
  if (descriptorSet && binding) {
    elidedAttrs.push_back(descriptorSetName);
    elidedAttrs.push_back(bindingName);
    printer << " bind(" << descriptorSet.getInt() << ", " << binding.getInt()
            << ")";
  }

  // Print BuiltIn attribute if present
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));
  if (auto builtin = op->getAttrOfType<StringAttr>(builtInName)) {
    printer << " " << builtInName << "(\"" << builtin.getValue() << "\")";
    elidedAttrs.push_back(builtInName);
  }

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

static ParseResult parseOneResultSameOperandTypeOp(OpAsmParser &parser,
                                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> ops;
  Type type;
  // If the operand list is in-between parentheses, then we have a generic form.
  // (see the fallback in `printOneResultOp`).
  SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(ops) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(type))
      return failure();
    auto fnType = llvm::dyn_cast<FunctionType>(type);
    if (!fnType) {
      parser.emitError(loc, "expected function type");
      return failure();
    }
    if (parser.resolveOperands(ops, fnType.getInputs(), loc, result.operands))
      return failure();
    result.addTypes(fnType.getResults());
    return success();
  }
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(ops, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

static void printOneResultOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumResults() == 1 && "op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type type) { return type != resultType; })) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  // Now we can output only one type for all operands and the result.
  p << " : " << resultType;
}

template <typename Op>
static LogicalResult verifyImageOperands(Op imageOp,
                                         spirv::ImageOperandsAttr attr,
                                         Operation::operand_range operands) {
  if (!attr) {
    if (operands.empty())
      return success();

    return imageOp.emitError("the Image Operands should encode what operands "
                             "follow, as per Image Operands");
  }

  // TODO: Add the validation rules for the following Image Operands.
  spirv::ImageOperands noSupportOperands =
      spirv::ImageOperands::Bias | spirv::ImageOperands::Lod |
      spirv::ImageOperands::Grad | spirv::ImageOperands::ConstOffset |
      spirv::ImageOperands::Offset | spirv::ImageOperands::ConstOffsets |
      spirv::ImageOperands::Sample | spirv::ImageOperands::MinLod |
      spirv::ImageOperands::MakeTexelAvailable |
      spirv::ImageOperands::MakeTexelVisible |
      spirv::ImageOperands::SignExtend | spirv::ImageOperands::ZeroExtend;

  if (spirv::bitEnumContainsAll(attr.getValue(), noSupportOperands))
    llvm_unreachable("unimplemented operands of Image Operands");

  return success();
}

template <typename BlockReadWriteOpTy>
static LogicalResult verifyBlockReadWritePtrAndValTypes(BlockReadWriteOpTy op,
                                                        Value ptr, Value val) {
  auto valType = val.getType();
  if (auto valVecTy = llvm::dyn_cast<VectorType>(valType))
    valType = valVecTy.getElementType();

  if (valType !=
      llvm::cast<spirv::PointerType>(ptr.getType()).getPointeeType()) {
    return op.emitOpError("mismatch in result type and pointer type");
  }
  return success();
}

/// Walks the given type hierarchy with the given indices, potentially down
/// to component granularity, to select an element type. Returns null type and
/// emits errors with the given loc on failure.
static Type
getElementType(Type type, ArrayRef<int32_t> indices,
               function_ref<InFlightDiagnostic(StringRef)> emitErrorFn) {
  if (indices.empty()) {
    emitErrorFn("expected at least one index for spirv.CompositeExtract");
    return nullptr;
  }

  for (auto index : indices) {
    if (auto cType = llvm::dyn_cast<spirv::CompositeType>(type)) {
      if (cType.hasCompileTimeKnownNumElements() &&
          (index < 0 ||
           static_cast<uint64_t>(index) >= cType.getNumElements())) {
        emitErrorFn("index ") << index << " out of bounds for " << type;
        return nullptr;
      }
      type = cType.getElementType(index);
    } else {
      emitErrorFn("cannot extract from non-composite type ")
          << type << " with index " << index;
      return nullptr;
    }
  }
  return type;
}

static Type
getElementType(Type type, Attribute indices,
               function_ref<InFlightDiagnostic(StringRef)> emitErrorFn) {
  auto indicesArrayAttr = llvm::dyn_cast<ArrayAttr>(indices);
  if (!indicesArrayAttr) {
    emitErrorFn("expected a 32-bit integer array attribute for 'indices'");
    return nullptr;
  }
  if (indicesArrayAttr.empty()) {
    emitErrorFn("expected at least one index for spirv.CompositeExtract");
    return nullptr;
  }

  SmallVector<int32_t, 2> indexVals;
  for (auto indexAttr : indicesArrayAttr) {
    auto indexIntAttr = llvm::dyn_cast<IntegerAttr>(indexAttr);
    if (!indexIntAttr) {
      emitErrorFn("expected an 32-bit integer for index, but found '")
          << indexAttr << "'";
      return nullptr;
    }
    indexVals.push_back(indexIntAttr.getInt());
  }
  return getElementType(type, indexVals, emitErrorFn);
}

static Type getElementType(Type type, Attribute indices, Location loc) {
  auto errorFn = [&](StringRef err) -> InFlightDiagnostic {
    return ::mlir::emitError(loc, err);
  };
  return getElementType(type, indices, errorFn);
}

static Type getElementType(Type type, Attribute indices, OpAsmParser &parser,
                           SMLoc loc) {
  auto errorFn = [&](StringRef err) -> InFlightDiagnostic {
    return parser.emitError(loc, err);
  };
  return getElementType(type, indices, errorFn);
}

template <typename ExtendedBinaryOp>
static LogicalResult verifyArithmeticExtendedBinaryOp(ExtendedBinaryOp op) {
  auto resultType = llvm::cast<spirv::StructType>(op.getType());
  if (resultType.getNumElements() != 2)
    return op.emitOpError("expected result struct type containing two members");

  if (!llvm::all_equal({op.getOperand1().getType(), op.getOperand2().getType(),
                        resultType.getElementType(0),
                        resultType.getElementType(1)}))
    return op.emitOpError(
        "expected all operand types and struct member types are the same");

  return success();
}

static ParseResult parseArithmeticExtendedBinaryOp(OpAsmParser &parser,
                                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOperandList(operands) || parser.parseColon())
    return failure();

  Type resultType;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(resultType))
    return failure();

  auto structType = llvm::dyn_cast<spirv::StructType>(resultType);
  if (!structType || structType.getNumElements() != 2)
    return parser.emitError(loc, "expected spirv.struct type with two members");

  SmallVector<Type, 2> operandTypes(2, structType.getElementType(0));
  if (parser.resolveOperands(operands, operandTypes, loc, result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

static void printArithmeticExtendedBinaryOp(Operation *op,
                                            OpAsmPrinter &printer) {
  printer << ' ';
  printer.printOptionalAttrDict(op->getAttrs());
  printer.printOperands(op->getOperands());
  printer << " : " << op->getResultTypes().front();
}

static LogicalResult verifyShiftOp(Operation *op) {
  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return op->emitError("expected the same type for the first operand and "
                         "result, but provided ")
           << op->getOperand(0).getType() << " and "
           << op->getResult(0).getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.mlir.addressof
//===----------------------------------------------------------------------===//

void spirv::AddressOfOp::build(OpBuilder &builder, OperationState &state,
                               spirv::GlobalVariableOp var) {
  build(builder, state, var.getType(), SymbolRefAttr::get(var));
}

LogicalResult spirv::AddressOfOp::verify() {
  auto varOp = dyn_cast_or_null<spirv::GlobalVariableOp>(
      SymbolTable::lookupNearestSymbolFrom((*this)->getParentOp(),
                                           getVariableAttr()));
  if (!varOp) {
    return emitOpError("expected spirv.GlobalVariable symbol");
  }
  if (getPointer().getType() != varOp.getType()) {
    return emitOpError(
        "result type mismatch with the referenced global variable's type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.CompositeConstruct
//===----------------------------------------------------------------------===//

LogicalResult spirv::CompositeConstructOp::verify() {
  operand_range constituents = this->getConstituents();

  // There are 4 cases with varying verification rules:
  // 1. Cooperative Matrices (1 constituent)
  // 2. Structs (1 constituent for each member)
  // 3. Arrays (1 constituent for each array element)
  // 4. Vectors (1 constituent (sub-)element for each vector element)

  auto coopElementType =
      llvm::TypeSwitch<Type, Type>(getType())
          .Case<spirv::CooperativeMatrixType, spirv::CooperativeMatrixNVType,
                spirv::JointMatrixINTELType>(
              [](auto coopType) { return coopType.getElementType(); })
          .Default([](Type) { return nullptr; });

  // Case 1. -- matrices.
  if (coopElementType) {
    if (constituents.size() != 1)
      return emitOpError("has incorrect number of operands: expected ")
             << "1, but provided " << constituents.size();
    if (coopElementType != constituents.front().getType())
      return emitOpError("operand type mismatch: expected operand type ")
             << coopElementType << ", but provided "
             << constituents.front().getType();
    return success();
  }

  // Case 2./3./4. -- number of constituents matches the number of elements.
  auto cType = llvm::cast<spirv::CompositeType>(getType());
  if (constituents.size() == cType.getNumElements()) {
    for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
      if (constituents[index].getType() != cType.getElementType(index)) {
        return emitOpError("operand type mismatch: expected operand type ")
               << cType.getElementType(index) << ", but provided "
               << constituents[index].getType();
      }
    }
    return success();
  }

  // Case 4. -- check that all constituents add up tp the expected vector type.
  auto resultType = llvm::dyn_cast<VectorType>(cType);
  if (!resultType)
    return emitOpError(
        "expected to return a vector or cooperative matrix when the number of "
        "constituents is less than what the result needs");

  SmallVector<unsigned> sizes;
  for (Value component : constituents) {
    if (!llvm::isa<VectorType>(component.getType()) &&
        !component.getType().isIntOrFloat())
      return emitOpError("operand type mismatch: expected operand to have "
                         "a scalar or vector type, but provided ")
             << component.getType();

    Type elementType = component.getType();
    if (auto vectorType = llvm::dyn_cast<VectorType>(component.getType())) {
      sizes.push_back(vectorType.getNumElements());
      elementType = vectorType.getElementType();
    } else {
      sizes.push_back(1);
    }

    if (elementType != resultType.getElementType())
      return emitOpError("operand element type mismatch: expected to be ")
             << resultType.getElementType() << ", but provided " << elementType;
  }
  unsigned totalCount = std::accumulate(sizes.begin(), sizes.end(), 0);
  if (totalCount != cType.getNumElements())
    return emitOpError("has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided " << totalCount;
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.CompositeExtractOp
//===----------------------------------------------------------------------===//

void spirv::CompositeExtractOp::build(OpBuilder &builder, OperationState &state,
                                      Value composite,
                                      ArrayRef<int32_t> indices) {
  auto indexAttr = builder.getI32ArrayAttr(indices);
  auto elementType =
      getElementType(composite.getType(), indexAttr, state.location);
  if (!elementType) {
    return;
  }
  build(builder, state, elementType, composite, indexAttr);
}

ParseResult spirv::CompositeExtractOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand compositeInfo;
  Attribute indicesAttr;
  Type compositeType;
  SMLoc attrLocation;

  if (parser.parseOperand(compositeInfo) ||
      parser.getCurrentLocation(&attrLocation) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, result.attributes) ||
      parser.parseColonType(compositeType) ||
      parser.resolveOperand(compositeInfo, compositeType, result.operands)) {
    return failure();
  }

  Type resultType =
      getElementType(compositeType, indicesAttr, parser, attrLocation);
  if (!resultType) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void spirv::CompositeExtractOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getComposite() << getIndices() << " : "
          << getComposite().getType();
}

LogicalResult spirv::CompositeExtractOp::verify() {
  auto indicesArrayAttr = llvm::dyn_cast<ArrayAttr>(getIndices());
  auto resultType =
      getElementType(getComposite().getType(), indicesArrayAttr, getLoc());
  if (!resultType)
    return failure();

  if (resultType != getType()) {
    return emitOpError("invalid result type: expected ")
           << resultType << " but provided " << getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.CompositeInsert
//===----------------------------------------------------------------------===//

void spirv::CompositeInsertOp::build(OpBuilder &builder, OperationState &state,
                                     Value object, Value composite,
                                     ArrayRef<int32_t> indices) {
  auto indexAttr = builder.getI32ArrayAttr(indices);
  build(builder, state, composite.getType(), object, composite, indexAttr);
}

ParseResult spirv::CompositeInsertOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  Type objectType, compositeType;
  Attribute indicesAttr;
  auto loc = parser.getCurrentLocation();

  return failure(
      parser.parseOperandList(operands, 2) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, result.attributes) ||
      parser.parseColonType(objectType) ||
      parser.parseKeywordType("into", compositeType) ||
      parser.resolveOperands(operands, {objectType, compositeType}, loc,
                             result.operands) ||
      parser.addTypesToList(compositeType, result.types));
}

LogicalResult spirv::CompositeInsertOp::verify() {
  auto indicesArrayAttr = llvm::dyn_cast<ArrayAttr>(getIndices());
  auto objectType =
      getElementType(getComposite().getType(), indicesArrayAttr, getLoc());
  if (!objectType)
    return failure();

  if (objectType != getObject().getType()) {
    return emitOpError("object operand type should be ")
           << objectType << ", but found " << getObject().getType();
  }

  if (getComposite().getType() != getType()) {
    return emitOpError("result type should be the same as "
                       "the composite type, but found ")
           << getComposite().getType() << " vs " << getType();
  }

  return success();
}

void spirv::CompositeInsertOp::print(OpAsmPrinter &printer) {
  printer << " " << getObject() << ", " << getComposite() << getIndices()
          << " : " << getObject().getType() << " into "
          << getComposite().getType();
}

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

ParseResult spirv::ConstantOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  Attribute value;
  if (parser.parseAttribute(value, kValueAttrName, result.attributes))
    return failure();

  Type type = NoneType::get(parser.getContext());
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(value))
    type = typedAttr.getType();
  if (llvm::isa<NoneType, TensorType>(type)) {
    if (parser.parseColonType(type))
      return failure();
  }

  return parser.addTypeToList(type, result.types);
}

void spirv::ConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getValue();
  if (llvm::isa<spirv::ArrayType>(getType()))
    printer << " : " << getType();
}

static LogicalResult verifyConstantType(spirv::ConstantOp op, Attribute value,
                                        Type opType) {
  if (llvm::isa<IntegerAttr, FloatAttr>(value)) {
    auto valueType = llvm::cast<TypedAttr>(value).getType();
    if (valueType != opType)
      return op.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  }
  if (llvm::isa<DenseIntOrFPElementsAttr, SparseElementsAttr>(value)) {
    auto valueType = llvm::cast<TypedAttr>(value).getType();
    if (valueType == opType)
      return success();
    auto arrayType = llvm::dyn_cast<spirv::ArrayType>(opType);
    auto shapedType = llvm::dyn_cast<ShapedType>(valueType);
    if (!arrayType)
      return op.emitOpError("result or element type (")
             << opType << ") does not match value type (" << valueType
             << "), must be the same or spirv.array";

    int numElements = arrayType.getNumElements();
    auto opElemType = arrayType.getElementType();
    while (auto t = llvm::dyn_cast<spirv::ArrayType>(opElemType)) {
      numElements *= t.getNumElements();
      opElemType = t.getElementType();
    }
    if (!opElemType.isIntOrFloat())
      return op.emitOpError("only support nested array result type");

    auto valueElemType = shapedType.getElementType();
    if (valueElemType != opElemType) {
      return op.emitOpError("result element type (")
             << opElemType << ") does not match value element type ("
             << valueElemType << ")";
    }

    if (numElements != shapedType.getNumElements()) {
      return op.emitOpError("result number of elements (")
             << numElements << ") does not match value number of elements ("
             << shapedType.getNumElements() << ")";
    }
    return success();
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(value)) {
    auto arrayType = llvm::dyn_cast<spirv::ArrayType>(opType);
    if (!arrayType)
      return op.emitOpError(
          "must have spirv.array result type for array value");
    Type elemType = arrayType.getElementType();
    for (Attribute element : arrayAttr.getValue()) {
      // Verify array elements recursively.
      if (failed(verifyConstantType(op, element, elemType)))
        return failure();
    }
    return success();
  }
  return op.emitOpError("cannot have attribute: ") << value;
}

LogicalResult spirv::ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return verifyConstantType(*this, getValueAttr(), getType());
}

bool spirv::ConstantOp::isBuildableWith(Type type) {
  // Must be valid SPIR-V type first.
  if (!llvm::isa<spirv::SPIRVType>(type))
    return false;

  if (isa<SPIRVDialect>(type.getDialect())) {
    // TODO: support constant struct
    return llvm::isa<spirv::ArrayType>(type);
  }

  return true;
}

spirv::ConstantOp spirv::ConstantOp::getZero(Type type, Location loc,
                                             OpBuilder &builder) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return builder.create<spirv::ConstantOp>(loc, type,
                                               builder.getBoolAttr(false));
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, APInt(width, 0)));
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getFloatAttr(floatType, 0.0));
  }
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    Type elemType = vectorType.getElementType();
    if (llvm::isa<IntegerType>(elemType)) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseElementsAttr::get(vectorType,
                                 IntegerAttr::get(elemType, 0).getValue()));
    }
    if (llvm::isa<FloatType>(elemType)) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseFPElementsAttr::get(vectorType,
                                   FloatAttr::get(elemType, 0.0).getValue()));
    }
  }

  llvm_unreachable("unimplemented types for ConstantOp::getZero()");
}

spirv::ConstantOp spirv::ConstantOp::getOne(Type type, Location loc,
                                            OpBuilder &builder) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return builder.create<spirv::ConstantOp>(loc, type,
                                               builder.getBoolAttr(true));
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, APInt(width, 1)));
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getFloatAttr(floatType, 1.0));
  }
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    Type elemType = vectorType.getElementType();
    if (llvm::isa<IntegerType>(elemType)) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseElementsAttr::get(vectorType,
                                 IntegerAttr::get(elemType, 1).getValue()));
    }
    if (llvm::isa<FloatType>(elemType)) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseFPElementsAttr::get(vectorType,
                                   FloatAttr::get(elemType, 1.0).getValue()));
    }
  }

  llvm_unreachable("unimplemented types for ConstantOp::getOne()");
}

void mlir::spirv::ConstantOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  Type type = getType();

  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "cst";

  IntegerType intTy = llvm::dyn_cast<IntegerType>(type);

  if (IntegerAttr intCst = llvm::dyn_cast<IntegerAttr>(getValue())) {
    if (intTy && intTy.getWidth() == 1) {
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));
    }

    if (intTy.isSignless()) {
      specialName << intCst.getInt();
    } else if (intTy.isUnsigned()) {
      specialName << intCst.getUInt();
    } else {
      specialName << intCst.getSInt();
    }
  }

  if (intTy || llvm::isa<FloatType>(type)) {
    specialName << '_' << type;
  }

  if (auto vecType = llvm::dyn_cast<VectorType>(type)) {
    specialName << "_vec_";
    specialName << vecType.getDimSize(0);

    Type elementType = vecType.getElementType();

    if (llvm::isa<IntegerType>(elementType) ||
        llvm::isa<FloatType>(elementType)) {
      specialName << "x" << elementType;
    }
  }

  setNameFn(getResult(), specialName.str());
}

void mlir::spirv::AddressOfOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << getVariable() << "_addr";
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// spirv.ControlBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ControlBarrierOp::verify() {
  return verifyMemorySemantics(getOperation(), getMemorySemantics());
}

//===----------------------------------------------------------------------===//
// spirv.EntryPoint
//===----------------------------------------------------------------------===//

void spirv::EntryPointOp::build(OpBuilder &builder, OperationState &state,
                                spirv::ExecutionModel executionModel,
                                spirv::FuncOp function,
                                ArrayRef<Attribute> interfaceVars) {
  build(builder, state,
        spirv::ExecutionModelAttr::get(builder.getContext(), executionModel),
        SymbolRefAttr::get(function), builder.getArrayAttr(interfaceVars));
}

ParseResult spirv::EntryPointOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  spirv::ExecutionModel execModel;
  SmallVector<OpAsmParser::UnresolvedOperand, 0> identifiers;
  SmallVector<Type, 0> idTypes;
  SmallVector<Attribute, 4> interfaceVars;

  FlatSymbolRefAttr fn;
  if (parseEnumStrAttr<spirv::ExecutionModelAttr>(execModel, parser, result) ||
      parser.parseAttribute(fn, Type(), kFnNameAttrName, result.attributes)) {
    return failure();
  }

  if (!parser.parseOptionalComma()) {
    // Parse the interface variables
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          // The name of the interface variable attribute isnt important
          FlatSymbolRefAttr var;
          NamedAttrList attrs;
          if (parser.parseAttribute(var, Type(), "var_symbol", attrs))
            return failure();
          interfaceVars.push_back(var);
          return success();
        }))
      return failure();
  }
  result.addAttribute(kInterfaceAttrName,
                      parser.getBuilder().getArrayAttr(interfaceVars));
  return success();
}

void spirv::EntryPointOp::print(OpAsmPrinter &printer) {
  printer << " \"" << stringifyExecutionModel(getExecutionModel()) << "\" ";
  printer.printSymbolName(getFn());
  auto interfaceVars = getInterface().getValue();
  if (!interfaceVars.empty()) {
    printer << ", ";
    llvm::interleaveComma(interfaceVars, printer);
  }
}

LogicalResult spirv::EntryPointOp::verify() {
  // Checks for fn and interface symbol reference are done in spirv::ModuleOp
  // verification.
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.ExecutionMode
//===----------------------------------------------------------------------===//

void spirv::ExecutionModeOp::build(OpBuilder &builder, OperationState &state,
                                   spirv::FuncOp function,
                                   spirv::ExecutionMode executionMode,
                                   ArrayRef<int32_t> params) {
  build(builder, state, SymbolRefAttr::get(function),
        spirv::ExecutionModeAttr::get(builder.getContext(), executionMode),
        builder.getI32ArrayAttr(params));
}

ParseResult spirv::ExecutionModeOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  spirv::ExecutionMode execMode;
  Attribute fn;
  if (parser.parseAttribute(fn, kFnNameAttrName, result.attributes) ||
      parseEnumStrAttr<spirv::ExecutionModeAttr>(execMode, parser, result)) {
    return failure();
  }

  SmallVector<int32_t, 4> values;
  Type i32Type = parser.getBuilder().getIntegerType(32);
  while (!parser.parseOptionalComma()) {
    NamedAttrList attr;
    Attribute value;
    if (parser.parseAttribute(value, i32Type, "value", attr)) {
      return failure();
    }
    values.push_back(llvm::cast<IntegerAttr>(value).getInt());
  }
  result.addAttribute(kValuesAttrName,
                      parser.getBuilder().getI32ArrayAttr(values));
  return success();
}

void spirv::ExecutionModeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getFn());
  printer << " \"" << stringifyExecutionMode(getExecutionMode()) << "\"";
  auto values = this->getValues();
  if (values.empty())
    return;
  printer << ", ";
  llvm::interleaveComma(values, printer, [&](Attribute a) {
    printer << llvm::cast<IntegerAttr>(a).getInt();
  });
}

//===----------------------------------------------------------------------===//
// spirv.func
//===----------------------------------------------------------------------===//

ParseResult spirv::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Parse the optional function control keyword.
  spirv::FunctionControl fnControl;
  if (parseEnumStrAttr<spirv::FunctionControlAttr>(fnControl, parser, result))
    return failure();

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  // Parse the optional function body.
  auto *body = result.addRegion();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs);
  return failure(parseResult.has_value() && failed(*parseResult));
}

void spirv::FuncOp::print(OpAsmPrinter &printer) {
  // Print function name, signature, and control.
  printer << " ";
  printer.printSymbolName(getSymName());
  auto fnType = getFunctionType();
  function_interface_impl::printFunctionSignature(
      printer, *this, fnType.getInputs(),
      /*isVariadic=*/false, fnType.getResults());
  printer << " \"" << spirv::stringifyFunctionControl(getFunctionControl())
          << "\"";
  function_interface_impl::printFunctionAttributes(
      printer, *this,
      {spirv::attributeName<spirv::FunctionControl>(),
       getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName(),
       getFunctionControlAttrName()});

  // Print the body if this is not an external function.
  Region &body = this->getBody();
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

LogicalResult spirv::FuncOp::verifyType() {
  if (getFunctionType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult spirv::FuncOp::verifyBody() {
  FunctionType fnType = getFunctionType();

  auto walkResult = walk([fnType](Operation *op) -> WalkResult {
    if (auto retOp = dyn_cast<spirv::ReturnOp>(op)) {
      if (fnType.getNumResults() != 0)
        return retOp.emitOpError("cannot be used in functions returning value");
    } else if (auto retOp = dyn_cast<spirv::ReturnValueOp>(op)) {
      if (fnType.getNumResults() != 1)
        return retOp.emitOpError(
                   "returns 1 value but enclosing function requires ")
               << fnType.getNumResults() << " results";

      auto retOperandType = retOp.getValue().getType();
      auto fnResultType = fnType.getResult(0);
      if (retOperandType != fnResultType)
        return retOp.emitOpError(" return value's type (")
               << retOperandType << ") mismatch with function's result type ("
               << fnResultType << ")";
    }
    return WalkResult::advance();
  });

  // TODO: verify other bits like linkage type.

  return failure(walkResult.wasInterrupted());
}

void spirv::FuncOp::build(OpBuilder &builder, OperationState &state,
                          StringRef name, FunctionType type,
                          spirv::FunctionControl control,
                          ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute(spirv::attributeName<spirv::FunctionControl>(),
                     builder.getAttr<spirv::FunctionControlAttr>(control));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

//===----------------------------------------------------------------------===//
// spirv.GLFClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLFClampOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLFClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.GLUClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLUClampOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLUClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.GLSClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLSClampOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLSClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.GLFmaOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLFmaOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLFmaOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spirv.GlobalVariable
//===----------------------------------------------------------------------===//

void spirv::GlobalVariableOp::build(OpBuilder &builder, OperationState &state,
                                    Type type, StringRef name,
                                    unsigned descriptorSet, unsigned binding) {
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::DescriptorSet),
      builder.getI32IntegerAttr(descriptorSet));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::Binding),
      builder.getI32IntegerAttr(binding));
}

void spirv::GlobalVariableOp::build(OpBuilder &builder, OperationState &state,
                                    Type type, StringRef name,
                                    spirv::BuiltIn builtin) {
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::BuiltIn),
      builder.getStringAttr(spirv::stringifyBuiltIn(builtin)));
}

ParseResult spirv::GlobalVariableOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  // Parse variable name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  // Parse optional initializer
  if (succeeded(parser.parseOptionalKeyword(kInitializerAttrName))) {
    FlatSymbolRefAttr initSymbol;
    if (parser.parseLParen() ||
        parser.parseAttribute(initSymbol, Type(), kInitializerAttrName,
                              result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, result)) {
    return failure();
  }

  Type type;
  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type)) {
    return failure();
  }
  if (!llvm::isa<spirv::PointerType>(type)) {
    return parser.emitError(loc, "expected spirv.ptr type");
  }
  result.addAttribute(kTypeAttrName, TypeAttr::get(type));

  return success();
}

void spirv::GlobalVariableOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};

  // Print variable name.
  printer << ' ';
  printer.printSymbolName(getSymName());
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());

  // Print optional initializer
  if (auto initializer = this->getInitializer()) {
    printer << " " << kInitializerAttrName << '(';
    printer.printSymbolName(*initializer);
    printer << ')';
    elidedAttrs.push_back(kInitializerAttrName);
  }

  elidedAttrs.push_back(kTypeAttrName);
  spirv::printVariableDecorations(*this, printer, elidedAttrs);
  printer << " : " << getType();
}

LogicalResult spirv::GlobalVariableOp::verify() {
  if (!llvm::isa<spirv::PointerType>(getType()))
    return emitOpError("result must be of a !spv.ptr type");

  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  // Also, Function storage class is reserved by spirv.Variable.
  auto storageClass = this->storageClass();
  if (storageClass == spirv::StorageClass::Generic ||
      storageClass == spirv::StorageClass::Function) {
    return emitOpError("storage class cannot be '")
           << stringifyStorageClass(storageClass) << "'";
  }

  if (auto init =
          (*this)->getAttrOfType<FlatSymbolRefAttr>(kInitializerAttrName)) {
    Operation *initOp = SymbolTable::lookupNearestSymbolFrom(
        (*this)->getParentOp(), init.getAttr());
    // TODO: Currently only variable initialization with specialization
    // constants and other variables is supported. They could be normal
    // constants in the module scope as well.
    if (!initOp ||
        !isa<spirv::GlobalVariableOp, spirv::SpecConstantOp>(initOp)) {
      return emitOpError("initializer must be result of a "
                         "spirv.SpecConstant or spirv.GlobalVariable op");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.INTEL.SubgroupBlockRead
//===----------------------------------------------------------------------===//

ParseResult spirv::INTELSubgroupBlockReadOp::parse(OpAsmParser &parser,
                                                   OperationState &result) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::UnresolvedOperand ptrInfo;
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) || parser.parseOperand(ptrInfo) ||
      parser.parseColon() || parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (auto valVecTy = llvm::dyn_cast<VectorType>(elementType))
    ptrType = spirv::PointerType::get(valVecTy.getElementType(), storageClass);

  if (parser.resolveOperand(ptrInfo, ptrType, result.operands)) {
    return failure();
  }

  result.addTypes(elementType);
  return success();
}

void spirv::INTELSubgroupBlockReadOp::print(OpAsmPrinter &printer) {
  printer << " " << getPtr() << " : " << getType();
}

LogicalResult spirv::INTELSubgroupBlockReadOp::verify() {
  if (failed(verifyBlockReadWritePtrAndValTypes(*this, getPtr(), getValue())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.INTEL.SubgroupBlockWrite
//===----------------------------------------------------------------------===//

ParseResult spirv::INTELSubgroupBlockWriteOp::parse(OpAsmParser &parser,
                                                    OperationState &result) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfo;
  auto loc = parser.getCurrentLocation();
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) ||
      parser.parseOperandList(operandInfo, 2) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (auto valVecTy = llvm::dyn_cast<VectorType>(elementType))
    ptrType = spirv::PointerType::get(valVecTy.getElementType(), storageClass);

  if (parser.resolveOperands(operandInfo, {ptrType, elementType}, loc,
                             result.operands)) {
    return failure();
  }
  return success();
}

void spirv::INTELSubgroupBlockWriteOp::print(OpAsmPrinter &printer) {
  printer << " " << getPtr() << ", " << getValue() << " : "
          << getValue().getType();
}

LogicalResult spirv::INTELSubgroupBlockWriteOp::verify() {
  if (failed(verifyBlockReadWritePtrAndValTypes(*this, getPtr(), getValue())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.IAddCarryOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::IAddCarryOp::verify() {
  return ::verifyArithmeticExtendedBinaryOp(*this);
}

ParseResult spirv::IAddCarryOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return ::parseArithmeticExtendedBinaryOp(parser, result);
}

void spirv::IAddCarryOp::print(OpAsmPrinter &printer) {
  ::printArithmeticExtendedBinaryOp(*this, printer);
}

//===----------------------------------------------------------------------===//
// spirv.ISubBorrowOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ISubBorrowOp::verify() {
  return ::verifyArithmeticExtendedBinaryOp(*this);
}

ParseResult spirv::ISubBorrowOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseArithmeticExtendedBinaryOp(parser, result);
}

void spirv::ISubBorrowOp::print(OpAsmPrinter &printer) {
  ::printArithmeticExtendedBinaryOp(*this, printer);
}

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

LogicalResult spirv::SMulExtendedOp::verify() {
  return ::verifyArithmeticExtendedBinaryOp(*this);
}

ParseResult spirv::SMulExtendedOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return ::parseArithmeticExtendedBinaryOp(parser, result);
}

void spirv::SMulExtendedOp::print(OpAsmPrinter &printer) {
  ::printArithmeticExtendedBinaryOp(*this, printer);
}

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

LogicalResult spirv::UMulExtendedOp::verify() {
  return ::verifyArithmeticExtendedBinaryOp(*this);
}

ParseResult spirv::UMulExtendedOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return ::parseArithmeticExtendedBinaryOp(parser, result);
}

void spirv::UMulExtendedOp::print(OpAsmPrinter &printer) {
  ::printArithmeticExtendedBinaryOp(*this, printer);
}

//===----------------------------------------------------------------------===//
// spirv.MemoryBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::MemoryBarrierOp::verify() {
  return verifyMemorySemantics(getOperation(), getMemorySemantics());
}

//===----------------------------------------------------------------------===//
// spirv.module
//===----------------------------------------------------------------------===//

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            std::optional<StringRef> name) {
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(state.addRegion());
  if (name) {
    state.attributes.append(mlir::SymbolTable::getSymbolAttrName(),
                            builder.getStringAttr(*name));
  }
}

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            spirv::AddressingModel addressingModel,
                            spirv::MemoryModel memoryModel,
                            std::optional<VerCapExtAttr> vceTriple,
                            std::optional<StringRef> name) {
  state.addAttribute(
      "addressing_model",
      builder.getAttr<spirv::AddressingModelAttr>(addressingModel));
  state.addAttribute("memory_model",
                     builder.getAttr<spirv::MemoryModelAttr>(memoryModel));
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(state.addRegion());
  if (vceTriple)
    state.addAttribute(getVCETripleAttrName(), *vceTriple);
  if (name)
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));
}

ParseResult spirv::ModuleOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  Region *body = result.addRegion();

  // If the name is present, parse it.
  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  // Parse attributes
  spirv::AddressingModel addrModel;
  spirv::MemoryModel memoryModel;
  if (spirv::parseEnumKeywordAttr<spirv::AddressingModelAttr>(addrModel, parser,
                                                              result) ||
      spirv::parseEnumKeywordAttr<spirv::MemoryModelAttr>(memoryModel, parser,
                                                          result))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("requires"))) {
    spirv::VerCapExtAttr vceTriple;
    if (parser.parseAttribute(vceTriple,
                              spirv::ModuleOp::getVCETripleAttrName(),
                              result.attributes))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*body, /*arguments=*/{}))
    return failure();

  // Make sure we have at least one block.
  if (body->empty())
    body->push_back(new Block());

  return success();
}

void spirv::ModuleOp::print(OpAsmPrinter &printer) {
  if (std::optional<StringRef> name = getName()) {
    printer << ' ';
    printer.printSymbolName(*name);
  }

  SmallVector<StringRef, 2> elidedAttrs;

  printer << " " << spirv::stringifyAddressingModel(getAddressingModel()) << " "
          << spirv::stringifyMemoryModel(getMemoryModel());
  auto addressingModelAttrName = spirv::attributeName<spirv::AddressingModel>();
  auto memoryModelAttrName = spirv::attributeName<spirv::MemoryModel>();
  elidedAttrs.assign({addressingModelAttrName, memoryModelAttrName,
                      mlir::SymbolTable::getSymbolAttrName()});

  if (std::optional<spirv::VerCapExtAttr> triple = getVceTriple()) {
    printer << " requires " << *triple;
    elidedAttrs.push_back(spirv::ModuleOp::getVCETripleAttrName());
  }

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  printer << ' ';
  printer.printRegion(getRegion());
}

LogicalResult spirv::ModuleOp::verifyRegions() {
  Dialect *dialect = (*this)->getDialect();
  DenseMap<std::pair<spirv::FuncOp, spirv::ExecutionModel>, spirv::EntryPointOp>
      entryPoints;
  mlir::SymbolTable table(*this);

  for (auto &op : *getBody()) {
    if (op.getDialect() != dialect)
      return op.emitError("'spirv.module' can only contain spirv.* ops");

    // For EntryPoint op, check that the function and execution model is not
    // duplicated in EntryPointOps. Also verify that the interface specified
    // comes from globalVariables here to make this check cheaper.
    if (auto entryPointOp = dyn_cast<spirv::EntryPointOp>(op)) {
      auto funcOp = table.lookup<spirv::FuncOp>(entryPointOp.getFn());
      if (!funcOp) {
        return entryPointOp.emitError("function '")
               << entryPointOp.getFn() << "' not found in 'spirv.module'";
      }
      if (auto interface = entryPointOp.getInterface()) {
        for (Attribute varRef : interface) {
          auto varSymRef = llvm::dyn_cast<FlatSymbolRefAttr>(varRef);
          if (!varSymRef) {
            return entryPointOp.emitError(
                       "expected symbol reference for interface "
                       "specification instead of '")
                   << varRef;
          }
          auto variableOp =
              table.lookup<spirv::GlobalVariableOp>(varSymRef.getValue());
          if (!variableOp) {
            return entryPointOp.emitError("expected spirv.GlobalVariable "
                                          "symbol reference instead of'")
                   << varSymRef << "'";
          }
        }
      }

      auto key = std::pair<spirv::FuncOp, spirv::ExecutionModel>(
          funcOp, entryPointOp.getExecutionModel());
      auto entryPtIt = entryPoints.find(key);
      if (entryPtIt != entryPoints.end()) {
        return entryPointOp.emitError("duplicate of a previous EntryPointOp");
      }
      entryPoints[key] = entryPointOp;
    } else if (auto funcOp = dyn_cast<spirv::FuncOp>(op)) {
      // If the function is external and does not have 'Import'
      // linkage_attributes(LinkageAttributes), throw an error. 'Import'
      // LinkageAttributes is used to import external functions.
      auto linkageAttr = funcOp.getLinkageAttributes();
      auto hasImportLinkage =
          linkageAttr && (linkageAttr.value().getLinkageType().getValue() ==
                          spirv::LinkageType::Import);
      if (funcOp.isExternal() && !hasImportLinkage)
        return op.emitError(
            "'spirv.module' cannot contain external functions "
            "without 'Import' linkage_attributes (LinkageAttributes)");

      // TODO: move this check to spirv.func.
      for (auto &block : funcOp)
        for (auto &op : block) {
          if (op.getDialect() != dialect)
            return op.emitError(
                "functions in 'spirv.module' can only contain spirv.* ops");
        }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.mlir.referenceof
//===----------------------------------------------------------------------===//

LogicalResult spirv::ReferenceOfOp::verify() {
  auto *specConstSym = SymbolTable::lookupNearestSymbolFrom(
      (*this)->getParentOp(), getSpecConstAttr());
  Type constType;

  auto specConstOp = dyn_cast_or_null<spirv::SpecConstantOp>(specConstSym);
  if (specConstOp)
    constType = specConstOp.getDefaultValue().getType();

  auto specConstCompositeOp =
      dyn_cast_or_null<spirv::SpecConstantCompositeOp>(specConstSym);
  if (specConstCompositeOp)
    constType = specConstCompositeOp.getType();

  if (!specConstOp && !specConstCompositeOp)
    return emitOpError(
        "expected spirv.SpecConstant or spirv.SpecConstantComposite symbol");

  if (getReference().getType() != constType)
    return emitOpError("result type mismatch with the referenced "
                       "specialization constant's type");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstant
//===----------------------------------------------------------------------===//

ParseResult spirv::SpecConstantOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  StringAttr nameAttr;
  Attribute valueAttr;

  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse optional spec_id.
  if (succeeded(parser.parseOptionalKeyword(kSpecIdAttrName))) {
    IntegerAttr specIdAttr;
    if (parser.parseLParen() ||
        parser.parseAttribute(specIdAttr, kSpecIdAttrName, result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseEqual() ||
      parser.parseAttribute(valueAttr, kDefaultValueAttrName,
                            result.attributes))
    return failure();

  return success();
}

void spirv::SpecConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  if (auto specID = (*this)->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    printer << ' ' << kSpecIdAttrName << '(' << specID.getInt() << ')';
  printer << " = " << getDefaultValue();
}

LogicalResult spirv::SpecConstantOp::verify() {
  if (auto specID = (*this)->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    if (specID.getValue().isNegative())
      return emitOpError("SpecId cannot be negative");

  auto value = getDefaultValue();
  if (llvm::isa<IntegerAttr, FloatAttr>(value)) {
    // Make sure bitwidth is allowed.
    if (!llvm::isa<spirv::SPIRVType>(value.getType()))
      return emitOpError("default value bitwidth disallowed");
    return success();
  }
  return emitOpError(
      "default value can only be a bool, integer, or float scalar");
}

//===----------------------------------------------------------------------===//
// spirv.VectorShuffle
//===----------------------------------------------------------------------===//

LogicalResult spirv::VectorShuffleOp::verify() {
  VectorType resultType = llvm::cast<VectorType>(getType());

  size_t numResultElements = resultType.getNumElements();
  if (numResultElements != getComponents().size())
    return emitOpError("result type element count (")
           << numResultElements
           << ") mismatch with the number of component selectors ("
           << getComponents().size() << ")";

  size_t totalSrcElements =
      llvm::cast<VectorType>(getVector1().getType()).getNumElements() +
      llvm::cast<VectorType>(getVector2().getType()).getNumElements();

  for (const auto &selector : getComponents().getAsValueRange<IntegerAttr>()) {
    uint32_t index = selector.getZExtValue();
    if (index >= totalSrcElements &&
        index != std::numeric_limits<uint32_t>().max())
      return emitOpError("component selector ")
             << index << " out of range: expected to be in [0, "
             << totalSrcElements << ") or 0xffffffff";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.MatrixTimesScalar
//===----------------------------------------------------------------------===//

LogicalResult spirv::MatrixTimesScalarOp::verify() {
  Type elementType =
      llvm::TypeSwitch<Type, Type>(getMatrix().getType())
          .Case<spirv::CooperativeMatrixType, spirv::CooperativeMatrixNVType,
                spirv::MatrixType>(
              [](auto matrixType) { return matrixType.getElementType(); })
          .Default([](Type) { return nullptr; });

  assert(elementType && "Unhandled type");

  // Check that the scalar type is the same as the matrix element type.
  if (getScalar().getType() != elementType)
    return emitOpError("input matrix components' type and scaling value must "
                       "have the same type");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.Transpose
//===----------------------------------------------------------------------===//

LogicalResult spirv::TransposeOp::verify() {
  auto inputMatrix = llvm::cast<spirv::MatrixType>(getMatrix().getType());
  auto resultMatrix = llvm::cast<spirv::MatrixType>(getResult().getType());

  // Verify that the input and output matrices have correct shapes.
  if (inputMatrix.getNumRows() != resultMatrix.getNumColumns())
    return emitError("input matrix rows count must be equal to "
                     "output matrix columns count");

  if (inputMatrix.getNumColumns() != resultMatrix.getNumRows())
    return emitError("input matrix columns count must be equal to "
                     "output matrix rows count");

  // Verify that the input and output matrices have the same component type
  if (inputMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("input and output matrices must have the same "
                     "component type");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.MatrixTimesMatrix
//===----------------------------------------------------------------------===//

LogicalResult spirv::MatrixTimesMatrixOp::verify() {
  auto leftMatrix = llvm::cast<spirv::MatrixType>(getLeftmatrix().getType());
  auto rightMatrix = llvm::cast<spirv::MatrixType>(getRightmatrix().getType());
  auto resultMatrix = llvm::cast<spirv::MatrixType>(getResult().getType());

  // left matrix columns' count and right matrix rows' count must be equal
  if (leftMatrix.getNumColumns() != rightMatrix.getNumRows())
    return emitError("left matrix columns' count must be equal to "
                     "the right matrix rows' count");

  // right and result matrices columns' count must be the same
  if (rightMatrix.getNumColumns() != resultMatrix.getNumColumns())
    return emitError(
        "right and result matrices must have equal columns' count");

  // right and result matrices component type must be the same
  if (rightMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("right and result matrices' component type must"
                     " be the same");

  // left and result matrices component type must be the same
  if (leftMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("left and result matrices' component type"
                     " must be the same");

  // left and result matrices rows count must be the same
  if (leftMatrix.getNumRows() != resultMatrix.getNumRows())
    return emitError("left and result matrices must have equal rows' count");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite
//===----------------------------------------------------------------------===//

ParseResult spirv::SpecConstantCompositeOp::parse(OpAsmParser &parser,
                                                  OperationState &result) {

  StringAttr compositeName;
  if (parser.parseSymbolName(compositeName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();

  SmallVector<Attribute, 4> constituents;

  do {
    // The name of the constituent attribute isn't important
    const char *attrName = "spec_const";
    FlatSymbolRefAttr specConstRef;
    NamedAttrList attrs;

    if (parser.parseAttribute(specConstRef, Type(), attrName, attrs))
      return failure();

    constituents.push_back(specConstRef);
  } while (!parser.parseOptionalComma());

  if (parser.parseRParen())
    return failure();

  result.addAttribute(kCompositeSpecConstituentsName,
                      parser.getBuilder().getArrayAttr(constituents));

  Type type;
  if (parser.parseColonType(type))
    return failure();

  result.addAttribute(kTypeAttrName, TypeAttr::get(type));

  return success();
}

void spirv::SpecConstantCompositeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " (";
  auto constituents = this->getConstituents().getValue();

  if (!constituents.empty())
    llvm::interleaveComma(constituents, printer);

  printer << ") : " << getType();
}

LogicalResult spirv::SpecConstantCompositeOp::verify() {
  auto cType = llvm::dyn_cast<spirv::CompositeType>(getType());
  auto constituents = this->getConstituents().getValue();

  if (!cType)
    return emitError("result type must be a composite type, but provided ")
           << getType();

  if (llvm::isa<spirv::CooperativeMatrixNVType>(cType))
    return emitError("unsupported composite type  ") << cType;
  if (llvm::isa<spirv::JointMatrixINTELType>(cType))
    return emitError("unsupported composite type  ") << cType;
  if (constituents.size() != cType.getNumElements())
    return emitError("has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided "
           << constituents.size();

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    auto constituent = llvm::cast<FlatSymbolRefAttr>(constituents[index]);

    auto constituentSpecConstOp =
        dyn_cast<spirv::SpecConstantOp>(SymbolTable::lookupNearestSymbolFrom(
            (*this)->getParentOp(), constituent.getAttr()));

    if (constituentSpecConstOp.getDefaultValue().getType() !=
        cType.getElementType(index))
      return emitError("has incorrect types of operands: expected ")
             << cType.getElementType(index) << ", but provided "
             << constituentSpecConstOp.getDefaultValue().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantOperation
//===----------------------------------------------------------------------===//

ParseResult spirv::SpecConstantOperationOp::parse(OpAsmParser &parser,
                                                  OperationState &result) {
  Region *body = result.addRegion();

  if (parser.parseKeyword("wraps"))
    return failure();

  body->push_back(new Block);
  Block &block = body->back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());

  if (!wrappedOp)
    return failure();

  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<spirv::YieldOp>(wrappedOp->getLoc(), wrappedOp->getResult(0));
  result.location = wrappedOp->getLoc();

  result.addTypes(wrappedOp->getResult(0).getType());

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void spirv::SpecConstantOperationOp::print(OpAsmPrinter &printer) {
  printer << " wraps ";
  printer.printGenericOp(&getBody().front().front());
}

LogicalResult spirv::SpecConstantOperationOp::verifyRegions() {
  Block &block = getRegion().getBlocks().front();

  if (block.getOperations().size() != 2)
    return emitOpError("expected exactly 2 nested ops");

  Operation &enclosedOp = block.getOperations().front();

  if (!enclosedOp.hasTrait<OpTrait::spirv::UsableInSpecConstantOp>())
    return emitOpError("invalid enclosed op");

  for (auto operand : enclosedOp.getOperands())
    if (!isa<spirv::ConstantOp, spirv::ReferenceOfOp,
             spirv::SpecConstantOperationOp>(operand.getDefiningOp()))
      return emitOpError(
          "invalid operand, must be defined by a constant operation");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.GL.FrexpStruct
//===----------------------------------------------------------------------===//

LogicalResult spirv::GLFrexpStructOp::verify() {
  spirv::StructType structTy =
      llvm::dyn_cast<spirv::StructType>(getResult().getType());

  if (structTy.getNumElements() != 2)
    return emitError("result type must be a struct type with two memebers");

  Type significandTy = structTy.getElementType(0);
  Type exponentTy = structTy.getElementType(1);
  VectorType exponentVecTy = llvm::dyn_cast<VectorType>(exponentTy);
  IntegerType exponentIntTy = llvm::dyn_cast<IntegerType>(exponentTy);

  Type operandTy = getOperand().getType();
  VectorType operandVecTy = llvm::dyn_cast<VectorType>(operandTy);
  FloatType operandFTy = llvm::dyn_cast<FloatType>(operandTy);

  if (significandTy != operandTy)
    return emitError("member zero of the resulting struct type must be the "
                     "same type as the operand");

  if (exponentVecTy) {
    IntegerType componentIntTy =
        llvm::dyn_cast<IntegerType>(exponentVecTy.getElementType());
    if (!componentIntTy || componentIntTy.getWidth() != 32)
      return emitError("member one of the resulting struct type must"
                       "be a scalar or vector of 32 bit integer type");
  } else if (!exponentIntTy || exponentIntTy.getWidth() != 32) {
    return emitError("member one of the resulting struct type "
                     "must be a scalar or vector of 32 bit integer type");
  }

  // Check that the two member types have the same number of components
  if (operandVecTy && exponentVecTy &&
      (exponentVecTy.getNumElements() == operandVecTy.getNumElements()))
    return success();

  if (operandFTy && exponentIntTy)
    return success();

  return emitError("member one of the resulting struct type must have the same "
                   "number of components as the operand type");
}

//===----------------------------------------------------------------------===//
// spirv.GL.Ldexp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GLLdexpOp::verify() {
  Type significandType = getX().getType();
  Type exponentType = getExp().getType();

  if (llvm::isa<FloatType>(significandType) !=
      llvm::isa<IntegerType>(exponentType))
    return emitOpError("operands must both be scalars or vectors");

  auto getNumElements = [](Type type) -> unsigned {
    if (auto vectorType = llvm::dyn_cast<VectorType>(type))
      return vectorType.getNumElements();
    return 1;
  };

  if (getNumElements(significandType) != getNumElements(exponentType))
    return emitOpError("operands must have the same number of elements");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.ImageDrefGather
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageDrefGatherOp::verify() {
  VectorType resultType = llvm::cast<VectorType>(getResult().getType());
  auto sampledImageType =
      llvm::cast<spirv::SampledImageType>(getSampledimage().getType());
  auto imageType =
      llvm::cast<spirv::ImageType>(sampledImageType.getImageType());

  if (resultType.getNumElements() != 4)
    return emitOpError("result type must be a vector of four components");

  Type elementType = resultType.getElementType();
  Type sampledElementType = imageType.getElementType();
  if (!llvm::isa<NoneType>(sampledElementType) &&
      elementType != sampledElementType)
    return emitOpError(
        "the component type of result must be the same as sampled type of the "
        "underlying image type");

  spirv::Dim imageDim = imageType.getDim();
  spirv::ImageSamplingInfo imageMS = imageType.getSamplingInfo();

  if (imageDim != spirv::Dim::Dim2D && imageDim != spirv::Dim::Cube &&
      imageDim != spirv::Dim::Rect)
    return emitOpError(
        "the Dim operand of the underlying image type must be 2D, Cube, or "
        "Rect");

  if (imageMS != spirv::ImageSamplingInfo::SingleSampled)
    return emitOpError("the MS operand of the underlying image type must be 0");

  spirv::ImageOperandsAttr attr = getImageoperandsAttr();
  auto operandArguments = getOperandArguments();

  return verifyImageOperands(*this, attr, operandArguments);
}

//===----------------------------------------------------------------------===//
// spirv.ShiftLeftLogicalOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftLeftLogicalOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spirv.ShiftRightArithmeticOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftRightArithmeticOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spirv.ShiftRightLogicalOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftRightLogicalOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spirv.BtiwiseAndOp
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::BitwiseAndOp::fold(spirv::BitwiseAndOp::FoldAdaptor adaptor) {
  APInt rhsMask;
  if (!matchPattern(adaptor.getOperand2(), m_ConstantInt(&rhsMask)))
    return {};

  // x & 0 -> 0
  if (rhsMask.isZero())
    return getOperand2();

  // x & <all ones> -> x
  if (rhsMask.isAllOnes())
    return getOperand1();

  // (UConvert x : iN to iK) & <mask with N low bits set> -> UConvert x
  if (auto zext = getOperand1().getDefiningOp<spirv::UConvertOp>()) {
    int valueBits =
        getElementTypeOrSelf(zext.getOperand()).getIntOrFloatBitWidth();
    if (rhsMask.zextOrTrunc(valueBits).isAllOnes())
      return getOperand1();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// spirv.BtiwiseOrOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::BitwiseOrOp::fold(spirv::BitwiseOrOp::FoldAdaptor adaptor) {
  APInt rhsMask;
  if (!matchPattern(adaptor.getOperand2(), m_ConstantInt(&rhsMask)))
    return {};

  // x | 0 -> x
  if (rhsMask.isZero())
    return getOperand1();

  // x | <all ones> -> <all ones>
  if (rhsMask.isAllOnes())
    return getOperand2();

  return {};
}

//===----------------------------------------------------------------------===//
// spirv.ImageQuerySize
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageQuerySizeOp::verify() {
  spirv::ImageType imageType =
      llvm::cast<spirv::ImageType>(getImage().getType());
  Type resultType = getResult().getType();

  spirv::Dim dim = imageType.getDim();
  spirv::ImageSamplingInfo samplingInfo = imageType.getSamplingInfo();
  spirv::ImageSamplerUseInfo samplerInfo = imageType.getSamplerUseInfo();
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Dim2D:
  case spirv::Dim::Dim3D:
  case spirv::Dim::Cube:
    if (samplingInfo != spirv::ImageSamplingInfo::MultiSampled &&
        samplerInfo != spirv::ImageSamplerUseInfo::SamplerUnknown &&
        samplerInfo != spirv::ImageSamplerUseInfo::NoSampler)
      return emitError(
          "if Dim is 1D, 2D, 3D, or Cube, "
          "it must also have either an MS of 1 or a Sampled of 0 or 2");
    break;
  case spirv::Dim::Buffer:
  case spirv::Dim::Rect:
    break;
  default:
    return emitError("the Dim operand of the image type must "
                     "be 1D, 2D, 3D, Buffer, Cube, or Rect");
  }

  unsigned componentNumber = 0;
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Buffer:
    componentNumber = 1;
    break;
  case spirv::Dim::Dim2D:
  case spirv::Dim::Cube:
  case spirv::Dim::Rect:
    componentNumber = 2;
    break;
  case spirv::Dim::Dim3D:
    componentNumber = 3;
    break;
  default:
    break;
  }

  if (imageType.getArrayedInfo() == spirv::ImageArrayedInfo::Arrayed)
    componentNumber += 1;

  unsigned resultComponentNumber = 1;
  if (auto resultVectorType = llvm::dyn_cast<VectorType>(resultType))
    resultComponentNumber = resultVectorType.getNumElements();

  if (componentNumber != resultComponentNumber)
    return emitError("expected the result to have ")
           << componentNumber << " component(s), but found "
           << resultComponentNumber << " component(s)";

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.VectorTimesScalarOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::VectorTimesScalarOp::verify() {
  if (getVector().getType() != getType())
    return emitOpError("vector operand and result type mismatch");
  auto scalarType = llvm::cast<VectorType>(getType()).getElementType();
  if (getScalar().getType() != scalarType)
    return emitOpError("scalar operand and result element type match");
  return success();
}
