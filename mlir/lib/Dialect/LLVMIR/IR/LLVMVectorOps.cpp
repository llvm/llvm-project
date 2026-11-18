//===- LLVMVectorOps.cpp - LLVM dialect vector ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Vector` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMVectorOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// ShuffleVectorOp
//===----------------------------------------------------------------------===//

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, DenseI32ArrayAttr mask,
                            ArrayRef<NamedAttribute> attrs) {
  auto containerType = v1.getType();
  auto vType = LLVM::getVectorType(
      cast<VectorType>(containerType).getElementType(), mask.size(),
      LLVM::isScalableVectorType(containerType));
  build(builder, state, vType, v1, v2, mask);
  state.addAttributes(attrs);
}

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, ArrayRef<int32_t> mask) {
  build(builder, state, v1, v2, builder.getDenseI32ArrayAttr(mask));
}

/// Build the result type of a shuffle vector operation.
ParseResult mlir::LLVM::parseShuffleType(AsmParser &parser, Type v1Type,
                                         Type &resType,
                                         DenseI32ArrayAttr mask) {
  if (!LLVM::isCompatibleVectorType(v1Type))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected an LLVM compatible vector type");
  resType =
      LLVM::getVectorType(cast<VectorType>(v1Type).getElementType(),
                          mask.size(), LLVM::isScalableVectorType(v1Type));
  return success();
}

/// Nothing to do when the result type is inferred.
void mlir::LLVM::printShuffleType(AsmPrinter &printer, Operation *op,
                                  Type v1Type, Type resType,
                                  DenseI32ArrayAttr mask) {}

LogicalResult ShuffleVectorOp::verify() {
  if (LLVM::isScalableVectorType(getV1().getType()) &&
      llvm::any_of(getMask(), [](int32_t v) { return v != 0; }))
    return emitOpError("expected a splat operation for scalable vectors");
  return success();
}

// Folding for shufflevector op when v1 is single element 1D vector
// and the mask is a single zero. OpFoldResult will be v1 in this case.
OpFoldResult ShuffleVectorOp::fold(FoldAdaptor adaptor) {
  // Check if operand 0 is a single element vector.
  auto vecType = llvm::dyn_cast<VectorType>(getV1().getType());
  if (!vecType || vecType.getRank() != 1 || vecType.getNumElements() != 1)
    return {};
  // Check if the mask is a single zero.
  // Note: The mask is guaranteed to be non-empty.
  if (getMask().size() != 1 || getMask()[0] != 0)
    return {};
  return getV1();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMVectorOps.cpp.inc"
