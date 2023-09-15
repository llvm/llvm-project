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
