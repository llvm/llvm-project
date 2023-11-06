//===- VCIXOps.cpp - VCIX dialect operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/VCIX/VCIXAttrs.h"
#include "mlir/Dialect/VCIX/VCIXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/VCIX/VCIX.cpp.inc"

static LogicalResult verifyOpcode(Attribute opcodeAttr,
                                  const unsigned expectedBitSize) {
  if (auto intAttr = opcodeAttr.dyn_cast<IntegerAttr>())
    return LogicalResult::success(intAttr.getType().isInteger(expectedBitSize));
  return failure();
}

static LogicalResult isWidenType(Type from, Type to) {
  if (isa<IntegerType>(from)) {
    return LogicalResult::success(2 * from.cast<IntegerType>().getWidth() ==
                                  to.cast<IntegerType>().getWidth());
  }
  if (isa<FloatType>(from)) {
    if (from.isF16() && to.isF32())
      return success();
    if (from.isF32() && to.isF64())
      return success();
  }
  return failure();
}

// Return true if type is a scalable vector that encodes LMUL and SEW correctly
// https://lists.llvm.org/pipermail/llvm-dev/2020-October/145850.html
static LogicalResult verifyVectorType(Type t) {
  auto vt = t.dyn_cast<VectorType>();
  if (!vt || vt.getRank() != 1)
    return failure();
  if (!vt.isScalable())
    return success();

  Type eltTy = vt.getElementType();
  unsigned sew = 0;
  if (eltTy.isF32())
    sew = 32;
  else if (eltTy.isF64())
    sew = 64;
  else if (auto intTy = eltTy.dyn_cast<IntegerType>())
    sew = intTy.getWidth();
  else
    return failure();

  unsigned eltCount = vt.getShape()[0];
  const unsigned lmul = eltCount * sew / 64;
  return lmul > 8 ? failure() : success();
}

template <typename OpT>
static LogicalResult verifyVCIXOpCommon(OpT op, Value result) {
  Type op1Type = op.getOp1().getType();
  VectorType op2Type = op.getOp2().getType().template cast<VectorType>();
  if (result && op2Type != result.getType())
    return op.emitOpError("Result type does not match to op2 type");

  if (failed(verifyVectorType(op2Type)))
    return op.emitOpError(
        "used type does not represent RVV-compatible scalable vector type");

  if (!op2Type.isScalable() && op.getRvl())
    return op.emitOpError(
        "'rvl' must not be specified if operation is done on a "
        "fixed vector type");

  if (op1Type.isa<VectorType>() && op1Type != op2Type)
    return op.emitOpError("op1 type does not match to op2 type");

  if (op1Type.isa<FloatType>()) {
    if (failed(verifyOpcode(op.getOpcodeAttr(), 1)))
      return op.emitOpError(
          "with a floating point scalar can only use 1-bit opcode");
    return success();
  }
  if (failed(verifyOpcode(op.getOpcodeAttr(), 2)))
    return op.emitOpError("must use 2-bit opcode");

  if (op1Type.isInteger(5)) {
    Operation *defOp = op.getOp1().getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
      return op.emitOpError("immediate operand must be a constant");
    return success();
  }
  if (op1Type.isa<IntegerType>() && !op1Type.isInteger(32) &&
      !op1Type.isInteger(64))
    return op.emitOpError(
        "non-constant integer first operand must be of a size 32 or 64");
  return success();
}

/// Unary operations
LogicalResult vcix::UnaryROOp::verify() {
  if (failed(verifyOpcode(getOpcodeAttr(), 2)))
    return emitOpError("must use 2-bit opcode");
  return success();
}

LogicalResult vcix::UnaryOp::verify() {
  if (failed(verifyOpcode(getOpcodeAttr(), 2)))
    return emitOpError("must use 2-bit opcode");

  if (failed(verifyVectorType(getResult().getType())))
    return emitOpError(
        "result type does not represent RVV-compatible scalable vector type");

  return success();
}

/// Binary operations
LogicalResult vcix::BinaryROOp::verify() {
  return verifyVCIXOpCommon(*this, nullptr);
}

LogicalResult vcix::BinaryOp::verify() {
  return verifyVCIXOpCommon(*this, getResult());
}

/// Ternary operations
LogicalResult vcix::TernaryROOp::verify() {
  VectorType op2Type = getOp2().getType().cast<VectorType>();
  VectorType op3Type = getOp3().getType().cast<VectorType>();
  if (op2Type != op3Type) {
    return emitOpError("op3 type does not match to op2 type");
  }
  return verifyVCIXOpCommon(*this, nullptr);
}

LogicalResult vcix::TernaryOp::verify() {
  VectorType op2Type = getOp2().getType().cast<VectorType>();
  VectorType op3Type = getOp3().getType().cast<VectorType>();
  if (op2Type != op3Type)
    return emitOpError("op3 type does not match to op2 type");

  return verifyVCIXOpCommon(*this, getResult());
}

/// Wide Ternary operations
LogicalResult vcix::WideTernaryROOp::verify() {
  VectorType op2Type = getOp2().getType().cast<VectorType>();
  VectorType op3Type = getOp3().getType().cast<VectorType>();
  if (failed(isWidenType(op2Type.getElementType(), op3Type.getElementType())))
    return emitOpError("result type is not widened type of op2");

  return verifyVCIXOpCommon(*this, nullptr);
}

LogicalResult vcix::WideTernaryOp::verify() {
  VectorType op2Type = getOp2().getType().cast<VectorType>();
  VectorType op3Type = getOp3().getType().cast<VectorType>();
  if (failed(isWidenType(op2Type.getElementType(), op3Type.getElementType())))
    return emitOpError("result type is not widened type of op2");

  // Don't compare result type for widended operations
  return verifyVCIXOpCommon(*this, nullptr);
}
