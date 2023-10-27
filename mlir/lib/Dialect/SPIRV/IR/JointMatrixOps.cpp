//===- JointMatrixOps.cpp - MLIR SPIR-V Intel Joint Matrix Ops  -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Intel Joint Matrix operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// spirv.INTEL.JointMatrixLoad
//===----------------------------------------------------------------------===//

static LogicalResult
verifyPointerAndJointMatrixType(Operation *op, Type pointer, Type jointMatrix) {
  Type pointeeType = llvm::cast<spirv::PointerType>(pointer).getPointeeType();
  if (!llvm::isa<spirv::ScalarType>(pointeeType) &&
      !llvm::isa<VectorType>(pointeeType))
    return op->emitError(
               "Pointer must point to a scalar or vector type but provided ")
           << pointeeType;
  spirv::StorageClass storage =
      llvm::cast<spirv::PointerType>(pointer).getStorageClass();
  if (storage != spirv::StorageClass::Workgroup &&
      storage != spirv::StorageClass::CrossWorkgroup &&
      storage != spirv::StorageClass::UniformConstant &&
      storage != spirv::StorageClass::Generic)
    return op->emitError("Pointer storage class must be Workgroup or "
                         "CrossWorkgroup but provided ")
           << stringifyStorageClass(storage);
  return success();
}

LogicalResult spirv::INTELJointMatrixLoadOp::verify() {
  return verifyPointerAndJointMatrixType(*this, getPointer().getType(),
                                         getResult().getType());
}

//===----------------------------------------------------------------------===//
// spirv.INTEL.JointMatrixStore
//===----------------------------------------------------------------------===//

LogicalResult spirv::INTELJointMatrixStoreOp::verify() {
  return verifyPointerAndJointMatrixType(*this, getPointer().getType(),
                                         getObject().getType());
}

//===----------------------------------------------------------------------===//
// spirv.INTEL.JointMatrixMad
//===----------------------------------------------------------------------===//

static LogicalResult verifyJointMatrixMad(spirv::INTELJointMatrixMadOp op) {
  if (op.getC().getType() != op.getResult().getType())
    return op.emitOpError("result and third operand must have the same type");
  auto typeA = llvm::cast<spirv::JointMatrixINTELType>(op.getA().getType());
  auto typeB = llvm::cast<spirv::JointMatrixINTELType>(op.getB().getType());
  auto typeC = llvm::cast<spirv::JointMatrixINTELType>(op.getC().getType());
  auto typeR =
      llvm::cast<spirv::JointMatrixINTELType>(op.getResult().getType());
  if (typeA.getRows() != typeR.getRows() ||
      typeA.getColumns() != typeB.getRows() ||
      typeB.getColumns() != typeR.getColumns())
    return op.emitOpError("matrix size must match");
  if (typeR.getScope() != typeA.getScope() ||
      typeR.getScope() != typeB.getScope() ||
      typeR.getScope() != typeC.getScope())
    return op.emitOpError("matrix scope must match");
  if (typeA.getElementType() != typeB.getElementType() ||
      typeR.getElementType() != typeC.getElementType())
    return op.emitOpError("matrix element type must match");
  return success();
}

LogicalResult spirv::INTELJointMatrixMadOp::verify() {
  return verifyJointMatrixMad(*this);
}

} // namespace mlir
