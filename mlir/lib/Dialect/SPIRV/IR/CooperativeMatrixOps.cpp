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
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

static LogicalResult
verifyCoopMatrixAccess(Operation *op, Type pointer, Type coopMatrix,
                       spirv::MemoryAccessAttr memoryOperand) {
  auto pointerType = cast<PointerType>(pointer);
  Type pointeeType = pointerType.getPointeeType();
  if (!isa<ScalarType, VectorType>(pointeeType)) {
    return op->emitOpError(
               "Pointer must point to a scalar or vector type but provided ")
           << pointeeType;
  }

  if (memoryOperand) {
    spirv::MemoryAccess operandSet = memoryOperand.getValue();

    if (isa<spirv::KHRCooperativeMatrixLoadOp>(op) &&
        spirv::bitEnumContainsAll(operandSet,
                                  spirv::MemoryAccess::MakePointerAvailable)) {
      return op->emitOpError(
          "not compatible with memory operand 'MakePointerAvailable'");
    }

    if (isa<spirv::KHRCooperativeMatrixStoreOp>(op) &&
        spirv::bitEnumContainsAll(operandSet,
                                  spirv::MemoryAccess::MakePointerVisible)) {
      return op->emitOpError(
          "not compatible with memory operand 'MakePointerVisible'");
    }

    // The 'Aligned' memory operand requires an alignment literal to follow,
    // which needs to be implemented on the level of op parsing and
    // (de-)serialization.
    // TODO: Consider adding support for this attribute value.
    if (spirv::bitEnumContainsAll(memoryOperand.getValue(),
                                  spirv::MemoryAccess::Aligned)) {
      return op->emitOpError("has unhandled memory operand 'Aligned'");
    }
  }

  // TODO: Verify the memory object behind the pointer:
  // > If the Shader capability was declared, Pointer must point into an array
  // > and any ArrayStride decoration on Pointer is ignored.

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixLoad
//===----------------------------------------------------------------------===//

LogicalResult KHRCooperativeMatrixLoadOp::verify() {
  return verifyCoopMatrixAccess(*this, getPointer().getType(),
                                getResult().getType(), getMemoryOperandAttr());
}

//===----------------------------------------------------------------------===//
// spirv.KHR.CooperativeMatrixStore
//===----------------------------------------------------------------------===//

LogicalResult KHRCooperativeMatrixStoreOp::verify() {
  return verifyCoopMatrixAccess(*this, getPointer().getType(),
                                getObject().getType(), getMemoryOperandAttr());
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
    if (!llvm::all_of(elementTypes, llvm::IsaPred<IntegerType>)) {
      return emitOpError("Matrix Operands require all matrix element types to "
                         "be Integer Types");
    }
  }

  // Any further requirements need to be checked against VCE.
  return success();
}

} // namespace mlir::spirv
