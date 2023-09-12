//===- CooperativeMatrixOps.cpp - MLIR SPIR-V Cooperative Matrix Ops  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Cooperative Matrix operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "SPIRVParsingUtils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {
//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixLength
//===----------------------------------------------------------------------===//

LogicalResult KHRCooperativeMatrixLengthOp::verify() {
  if (!isa<CooperativeMatrixType>(getCooperativeMatrixType())) {
    return emitOpError(
               "type attribute must be a '!spirv.coopmatrix' type, found ")
           << getCooperativeMatrixType() << " instead";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixLoad
//===----------------------------------------------------------------------===//

static LogicalResult verifyPointerAndCoopMatrixType(Operation *op, Type pointer,
                                                    Type coopMatrix) {
  auto pointerType = cast<PointerType>(pointer);
  Type pointeeType = pointerType.getPointeeType();
  if (!isa<ScalarType, VectorType>(pointeeType)) {
    return op->emitError(
               "Pointer must point to a scalar or vector type but provided ")
           << pointeeType;
  }

  // TODO: Verify the memory object behind the pointer:
  // > If the Shader capability was declared, Pointer must point into an array
  // > and any ArrayStride decoration on Pointer is ignored.

  return success();
}

LogicalResult KHRCooperativeMatrixLoadOp::verify() {
  return verifyPointerAndCoopMatrixType(*this, getPointer().getType(),
                                        getResult().getType());
}

//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixStore
//===----------------------------------------------------------------------===//

ParseResult KHRCooperativeMatrixStoreOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  std::array<OpAsmParser::UnresolvedOperand, 3> operandInfo = {};
  for (auto &op : operandInfo) {
    if (parser.parseOperand(op) || parser.parseComma())
      return failure();
  }

  CooperativeMatrixLayoutKHR layout;
  if (parseEnumKeywordAttr<CooperativeMatrixLayoutKHRAttr>(
          layout, parser, result, kKhrCooperativeMatrixLayoutAttrName)) {
    return failure();
  }

  if (parseMemoryAccessAttributes(parser, result, kMemoryOperandAttrName))
    return failure();

  Type ptrType;
  Type objectType;
  if (parser.parseColon() || parser.parseType(ptrType) || parser.parseComma() ||
      parser.parseType(objectType)) {
    return failure();
  }

  Type strideType = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(operandInfo, {ptrType, objectType, strideType},
                             parser.getNameLoc(), result.operands)) {
    return failure();
  }

  return success();
}

void KHRCooperativeMatrixStoreOp::print(OpAsmPrinter &printer) {
  printer << " " << getPointer() << ", " << getObject() << ", " << getStride()
          << ", " << getMatrixLayout();

  // Print optional memory operand attribute.
  if (auto memOperand = getMemoryOperand())
    printer << " [\"" << *memOperand << "\"]";
  printer << " : " << getPointer().getType() << ", " << getObject().getType();
}

LogicalResult KHRCooperativeMatrixStoreOp::verify() {
  return verifyPointerAndCoopMatrixType(*this, getPointer().getType(),
                                        getObject().getType());
}

//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixMulAdd
//===----------------------------------------------------------------------===//

LogicalResult KHRCooperativeMatrixMulAddOp::verify() {
  auto typeA = cast<spirv::CooperativeMatrixType>(getA().getType());
  auto typeB = cast<spirv::CooperativeMatrixType>(getB().getType());
  auto typeC = cast<spirv::CooperativeMatrixType>(getC().getType());

  // Check element types. ODS enforces that `type(c) == type(result)`, so no
  // need to check it here.

  // Check the 'use' part of the type against the operands and the result.
  if (typeA.getUse() != CooperativeMatrixUseKHR::MatrixA)
    return emitOpError("operand #0 must be of use 'MatrixA'");
  if (typeB.getUse() != CooperativeMatrixUseKHR::MatrixB)
    return emitOpError("operand #1 must be of use 'MatrixB'");
  if (typeC.getUse() != CooperativeMatrixUseKHR::MatrixAcc)
    return emitOpError("operand #2 must be of use 'MatrixAcc'");

  // Check the 'scope' part of the type.
  if (!llvm::all_equal({typeA.getScope(), typeB.getScope(), typeC.getScope()}))
    return emitOpError("matrix scope mismatch");

  // Check dimension sizes. We expect 'MxK * KxN + MxN -> MxN'.
  if (typeA.getRows() != typeC.getRows())
    return emitOpError("matrix size mismatch on dimension 'M'");
  if (typeB.getColumns() != typeC.getColumns())
    return emitOpError("matrix size mismatch on dimension 'N'");
  if (typeA.getColumns() != typeB.getRows())
    return emitOpError("matrix size mismatch on dimension 'K'");

  // The spec does not restrict the element types:
  //  > A, B, C, and Result Type need not necessarily have the same component
  //  > type, this is defined by the client API.

  // Check that if Cooperative Matrix Operands are provided, the element type
  // is integer.
  if (getMatrixOperands()) {
    Type elementTypes[] = {typeA.getElementType(), typeB.getElementType(),
                           typeC.getElementType()};
    if (!llvm::all_of(elementTypes,
                      [](Type ty) { return isa<IntegerType>(ty); })) {
      return emitOpError("Matrix Operands require all matrix element types to "
                         "be Integer Types");
    }
  }

  // Any further requirements need to be checked against VCE.
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.NV.CooperativeMatrixLength
//===----------------------------------------------------------------------===//

LogicalResult NVCooperativeMatrixLengthOp::verify() {
  if (!isa<CooperativeMatrixNVType>(getCooperativeMatrixType())) {
    return emitOpError(
               "type attribute must be a '!spirv.NV.coopmatrix' type, found ")
           << getCooperativeMatrixType() << " instead";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.NV.CooperativeMatrixLoad
//===----------------------------------------------------------------------===//

ParseResult NVCooperativeMatrixLoadOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operandInfo;
  Type strideType = parser.getBuilder().getIntegerType(32);
  Type columnMajorType = parser.getBuilder().getIntegerType(1);
  Type ptrType;
  Type elementType;
  if (parser.parseOperandList(operandInfo, 3) ||
      parseMemoryAccessAttributes(parser, result) || parser.parseColon() ||
      parser.parseType(ptrType) || parser.parseKeywordType("as", elementType)) {
    return failure();
  }
  if (parser.resolveOperands(operandInfo,
                             {ptrType, strideType, columnMajorType},
                             parser.getNameLoc(), result.operands)) {
    return failure();
  }

  result.addTypes(elementType);
  return success();
}

void NVCooperativeMatrixLoadOp::print(OpAsmPrinter &printer) {
  printer << " " << getPointer() << ", " << getStride() << ", "
          << getColumnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = getMemoryAccess())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << getPointer().getType() << " as " << getType();
}

static LogicalResult
verifyPointerAndCoopMatrixNVType(Operation *op, Type pointer, Type coopMatrix) {
  Type pointeeType = llvm::cast<PointerType>(pointer).getPointeeType();
  if (!llvm::isa<ScalarType>(pointeeType) &&
      !llvm::isa<VectorType>(pointeeType))
    return op->emitError(
               "Pointer must point to a scalar or vector type but provided ")
           << pointeeType;
  StorageClass storage = llvm::cast<PointerType>(pointer).getStorageClass();
  if (storage != StorageClass::Workgroup &&
      storage != StorageClass::StorageBuffer &&
      storage != StorageClass::PhysicalStorageBuffer)
    return op->emitError(
               "Pointer storage class must be Workgroup, StorageBuffer or "
               "PhysicalStorageBufferEXT but provided ")
           << stringifyStorageClass(storage);
  return success();
}

LogicalResult NVCooperativeMatrixLoadOp::verify() {
  return verifyPointerAndCoopMatrixNVType(*this, getPointer().getType(),
                                          getResult().getType());
}

//===----------------------------------------------------------------------===//
// spirv.NV.CooperativeMatrixStore
//===----------------------------------------------------------------------===//

ParseResult NVCooperativeMatrixStoreOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operandInfo;
  Type strideType = parser.getBuilder().getIntegerType(32);
  Type columnMajorType = parser.getBuilder().getIntegerType(1);
  Type ptrType;
  Type elementType;
  if (parser.parseOperandList(operandInfo, 4) ||
      parseMemoryAccessAttributes(parser, result) || parser.parseColon() ||
      parser.parseType(ptrType) || parser.parseComma() ||
      parser.parseType(elementType)) {
    return failure();
  }
  if (parser.resolveOperands(
          operandInfo, {ptrType, elementType, strideType, columnMajorType},
          parser.getNameLoc(), result.operands)) {
    return failure();
  }

  return success();
}

void NVCooperativeMatrixStoreOp::print(OpAsmPrinter &printer) {
  printer << " " << getPointer() << ", " << getObject() << ", " << getStride()
          << ", " << getColumnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = getMemoryAccess())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << getPointer().getType() << ", " << getOperand(1).getType();
}

LogicalResult NVCooperativeMatrixStoreOp::verify() {
  return verifyPointerAndCoopMatrixNVType(*this, getPointer().getType(),
                                          getObject().getType());
}

//===----------------------------------------------------------------------===//
// spirv.NV.CooperativeMatrixMulAdd
//===----------------------------------------------------------------------===//

static LogicalResult verifyCoopMatrixMulAddNV(NVCooperativeMatrixMulAddOp op) {
  if (op.getC().getType() != op.getResult().getType())
    return op.emitOpError("result and third operand must have the same type");
  auto typeA = llvm::cast<CooperativeMatrixNVType>(op.getA().getType());
  auto typeB = llvm::cast<CooperativeMatrixNVType>(op.getB().getType());
  auto typeC = llvm::cast<CooperativeMatrixNVType>(op.getC().getType());
  auto typeR = llvm::cast<CooperativeMatrixNVType>(op.getResult().getType());
  if (typeA.getRows() != typeR.getRows() ||
      typeA.getColumns() != typeB.getRows() ||
      typeB.getColumns() != typeR.getColumns())
    return op.emitOpError("matrix size must match");
  if (typeR.getScope() != typeA.getScope() ||
      typeR.getScope() != typeB.getScope() ||
      typeR.getScope() != typeC.getScope())
    return op.emitOpError("matrix scope must match");
  auto elementTypeA = typeA.getElementType();
  auto elementTypeB = typeB.getElementType();
  if (isa<IntegerType>(elementTypeA) && isa<IntegerType>(elementTypeB)) {
    if (llvm::cast<IntegerType>(elementTypeA).getWidth() !=
        llvm::cast<IntegerType>(elementTypeB).getWidth())
      return op.emitOpError(
          "matrix A and B integer element types must be the same bit width");
  } else if (elementTypeA != elementTypeB) {
    return op.emitOpError(
        "matrix A and B non-integer element types must match");
  }
  if (typeR.getElementType() != typeC.getElementType())
    return op.emitOpError("matrix accumulator element type must match");
  return success();
}

LogicalResult NVCooperativeMatrixMulAddOp::verify() {
  return verifyCoopMatrixMulAddNV(*this);
}

} // namespace mlir::spirv
