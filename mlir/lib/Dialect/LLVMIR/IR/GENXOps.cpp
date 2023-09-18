//===- GENXOps.cpp - MLIR GENX operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/GENXTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// genx.matrix.load
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixLoadOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto resType = getResult().getType().cast<GENX::JointMatrixType>();
  if (getLayout() != resType.getMatrixLayout())
    return this->emitOpError("result layout must match layout attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.store
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixStoreOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto valType = getVal().getType().cast<GENX::JointMatrixType>();
  if (getLayout() != valType.getMatrixLayout())
    return this->emitOpError(
        "layout of value to store must match layout attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.mad
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixMadOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto AType = getA().getType().cast<GENX::JointMatrixType>();
  auto BType = getB().getType().cast<GENX::JointMatrixType>();
  auto CType = getC().getType().cast<GENX::JointMatrixType>();
  auto resType = getResult().getType().cast<GENX::JointMatrixType>();

  if (CType != resType)
    return this->emitOpError("result and 3rd operand must have the same type");

  // Check the matrices dimensions match - A(M,K) * B(K,N) + C(M,N).
  if (AType.getNumRows() != resType.getNumRows() ||
      AType.getNumColumns() != BType.getNumRows() ||
      BType.getNumColumns() != resType.getNumColumns())
    return this->emitOpError("matrix sizes must match");

  Type AElemType = AType.getElementType();
  Type BElemType = BType.getElementType();
  Type CElemType = CType.getElementType();
  Type resElemType = resType.getElementType();

  // Check valid sizes for the matrixes dimensions which on XMX are:
  //   M <= 8, N == 16, K == 32 (if A's element type is integer)
  //   M <= 8, N == 16, K == 16 (if A's element type is floating-point)
  if (resType.getNumRows() > 8)
    return this->emitOpError("result matrix must have a max of 8 rows");
  if (resType.getNumColumns() != 16)
    return this->emitOpError("result matrix must have 16 columns");
  if (isa<IntegerType>(AElemType) && AType.getNumColumns() != 32)
    return this->emitOpError("1st operand matrix must have 32 columns");
  if (isa<FloatType>(AElemType) && AType.getNumColumns() != 16)
    return this->emitOpError("1st operand matrix must have 16 columns");

  // Check that element types match.
  if (AElemType != BElemType || resElemType != CElemType)
    return this->emitOpError("matrix element types must match");

  // Allowed matrices element types on XMX are:
  //   Matrices  |     A      |     B      |   C   |
  //   Elem Type | uint8/int8 | uint8/int8 | int32 |
  //             |    fp16    |    fp16    | fp32  |
  //             |    bf16    |    bf16    | fp32  |
  if (auto t = dyn_cast<IntegerType>(AElemType)) {
    if (t.getWidth() != 8)
      return this->emitOpError(
          "1st operand element type must have bit-width equal to 8");
    if (!isa<IntegerType>(CElemType) ||
        cast<IntegerType>(CElemType).getWidth() != 32)
      return this->emitOpError("3rd operand element type must be i32");
  } else if (auto t = dyn_cast<FloatType>(AElemType)) {
    if (!t.isF16() && !t.isBF16())
      return this->emitOpError("1st operand element type must be f16 or bf16");
    if (!isa<FloatType>(CElemType) ||
        cast<FloatType>(CElemType).getWidth() != 32)
      return this->emitOpError("3rd operand element type must be f32");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.init
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixInitOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto matType = getMat().getType().cast<GENX::JointMatrixType>();
  if (matType.getElementType() != getVal().getType())
    return this->emitOpError("initializer type must match matrix element type");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.copy
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixCopyOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto resType = getResult().getType().cast<GENX::JointMatrixType>();
  auto srcType = getSrc().getType().cast<GENX::JointMatrixType>();

  if ((resType.getNumRows() != srcType.getNumRows()) ||
      (resType.getNumColumns() != srcType.getNumColumns()))
    return this->emitOpError("result shape must match source shape");

  if (resType.getMatrixLayout() != srcType.getMatrixLayout())
    return this->emitOpError("result layout must match source layout");

  return success();
}
