//===- AIRDialect.cpp - AIR dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::air;

#include "mlir/Dialect/AIR/IR/AIROpsDialect.cpp.inc"

void AIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AIR/IR/AIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"
      >();
  declarePromisedInterface<mlir::gpu::TargetAttrInterface,
                           mlir::air::AIRTargetAttr>();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"

static LogicalResult verifyMatrixAccess(Operation *op, MemRefType base,
                                        size_t numIndices, VectorType matrix) {
  if (numIndices != 2)
    return op->emitOpError("takes two indices");
  if (base.getElementType() != matrix.getElementType())
    return op->emitOpError("matrix and memref element types differ");
  return success();
}

LogicalResult MatrixLoadOp::verify() {
  return verifyMatrixAccess(*this, getBase().getType(), getIndices().size(),
                            getResult().getType());
}

LogicalResult MatrixStoreOp::verify() {
  return verifyMatrixAccess(*this, getBase().getType(), getIndices().size(),
                            getValue().getType());
}

LogicalResult MatrixMultiplyAccumulateOp::verify() {
  Type ab = getA().getType().getElementType();
  Type cd = getC().getType().getElementType();
  if (ab != cd && !((ab.isF16() || ab.isBF16()) && cd.isF32()))
    return emitOpError("multiplies ")
           << ab << " matrices into " << cd
           << " (only the same type, or f16 or bf16 into f32)";
  return success();
}

/// `: T` when A, B, C and D all have type T, else `: TA, TC`.
static ParseResult parseMatrixMMATypes(OpAsmParser &parser, Type &a, Type &c) {
  if (parser.parseType(a))
    return failure();
  c = a;
  if (succeeded(parser.parseOptionalComma()))
    return parser.parseType(c);
  return success();
}

static void printMatrixMMATypes(OpAsmPrinter &printer, Operation *, Type a,
                                Type c) {
  printer << a;
  if (a != c)
    printer << ", " << c;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AIR/IR/AIROps.cpp.inc"
