//===- MemOpInterfaces.cpp - Memory operation interfaces ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/MemOpInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace mlir;

LogicalResult mlir::detail::verifyMemorySpaceCastOpInterface(Operation *op) {
  auto memCastOp = cast<MemorySpaceCastOpInterface>(op);

  // Verify that the source and target pointers are valid
  Value sourcePtr = memCastOp.getSourcePtr();
  Value targetPtr = memCastOp.getTargetPtr();

  if (!sourcePtr || !targetPtr) {
    return op->emitError()
           << "memory space cast op must have valid source and target pointers";
  }

  if (sourcePtr.getType().getTypeID() != targetPtr.getType().getTypeID()) {
    return op->emitError()
           << "expected source and target types of the same kind";
  }

  // Verify the Types are of `PtrLikeTypeInterface` type.
  auto sourceType = dyn_cast<PtrLikeTypeInterface>(sourcePtr.getType());
  if (!sourceType) {
    return op->emitError()
           << "source type must implement `PtrLikeTypeInterface`, but got: "
           << sourcePtr.getType();
  }

  auto targetType = dyn_cast<PtrLikeTypeInterface>(targetPtr.getType());
  if (!targetType) {
    return op->emitError()
           << "target type must implement `PtrLikeTypeInterface`, but got: "
           << targetPtr.getType();
  }

  // Verify that the operation has exactly one result
  if (op->getNumResults() != 1) {
    return op->emitError()
           << "memory space cast op must have exactly one result";
  }

  return success();
}

FailureOr<std::optional<SmallVector<Value>>>
mlir::detail::bubbleDownInPlaceMemorySpaceCastImpl(OpOperand &operand,
                                                   ValueRange results) {
  MemorySpaceCastOpInterface castOp =
      MemorySpaceCastOpInterface::getIfPromotableCast(operand.get());

  // Bail if the src is not valid.
  if (!castOp)
    return failure();

  // Modify the op.
  operand.set(castOp.getSourcePtr());
  return std::optional<SmallVector<Value>>();
}

#include "mlir/Interfaces/MemOpInterfaces.cpp.inc"
