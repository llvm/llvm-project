//===- LLVMFloatIntrinsicOps.cpp - LLVM floating-point intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `FloatIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMFloatIntrinsicOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// SincosOp and ModfOp (intrinsics)
//===----------------------------------------------------------------------===//

static LogicalResult verifyHomogeneousStructResult(Operation *op,
                                                   Type operandType,
                                                   Type resultType) {
  auto resultStructType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(resultType);
  if (!resultStructType || resultStructType.getBody().size() != 2 ||
      resultStructType.getBody()[0] != operandType ||
      resultStructType.getBody()[1] != operandType) {
    return op->emitOpError(
               "expected result type to be a homogeneous struct with "
               "two elements matching the operand type, but got ")
           << resultType;
  }
  return success();
}

LogicalResult LLVM::SincosOp::verify() {
  return verifyHomogeneousStructResult(getOperation(), getVal().getType(),
                                       getResult().getType());
}

LogicalResult LLVM::ModfOp::verify() {
  return verifyHomogeneousStructResult(getOperation(), getVal().getType(),
                                       getResult().getType());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMFloatIntrinsicOps.cpp.inc"
