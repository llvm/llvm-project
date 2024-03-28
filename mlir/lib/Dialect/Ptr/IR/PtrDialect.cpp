//===- PtrDialect.cpp - Pointer dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pointer dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

// Returns a pair containing:
// The underlying type of a vector or the type itself if it's not a vector.
// The number of elements in the vector or an error code if the type is not
// supported.
static std::pair<Type, int64_t> getVecOrScalarInfo(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemTy = vecTy.getElementType();
    // Vectors of rank greater than one or with scalable dimensions are not
    // supported.
    if (vecTy.getRank() != 1)
      return {elemTy, -1};
    else if (vecTy.getScalableDims()[0])
      return {elemTy, -2};
    return {elemTy, vecTy.getShape()[0]};
  }
  // `ty` is a scalar type.
  return {ty, 0};
}

LogicalResult mlir::ptr::isValidAddrSpaceCastImpl(Type tgt, Type src,
                                                  Operation *op) {
  std::pair<Type, int64_t> tgtInfo = getVecOrScalarInfo(tgt);
  std::pair<Type, int64_t> srcInfo = getVecOrScalarInfo(src);
  if (!isa<PtrType>(tgtInfo.first) || !isa<PtrType>(srcInfo.first))
    return op ? op->emitError("invalid ptr-like operand") : failure();
  // Check shape validity.
  if (tgtInfo.second == -1 || srcInfo.second == -1)
    return op ? op->emitError("vectors of rank != 1 are not supported")
              : failure();
  if (tgtInfo.second == -2 || srcInfo.second == -2)
    return op ? op->emitError(
                    "vectors with scalable dimensions are not supported")
              : failure();
  if (tgtInfo.second != srcInfo.second)
    return op ? op->emitError("incompatible operand shapes") : failure();
  return success();
}

LogicalResult mlir::ptr::isValidPtrIntCastImpl(Type intLikeTy, Type ptrLikeTy,
                                               Operation *op) {
  // Check int-like type.
  std::pair<Type, int64_t> intInfo = getVecOrScalarInfo(intLikeTy);
  if (!intInfo.first.isSignlessIntOrIndex())
    /// The int-like operand is invalid.
    return op ? op->emitError("invalid int-like type") : failure();
  // Check ptr-like type.
  std::pair<Type, int64_t> ptrInfo = getVecOrScalarInfo(ptrLikeTy);
  if (!isa<PtrType>(ptrInfo.first))
    /// The pointer-like operand is invalid.
    return op ? op->emitError("invalid ptr-like type") : failure();
  // Check shape validity.
  if (intInfo.second == -1 || ptrInfo.second == -1)
    return op ? op->emitError("vectors of rank != 1 are not supported")
              : failure();
  if (intInfo.second == -2 || ptrInfo.second == -2)
    return op ? op->emitError(
                    "vectors with scalable dimensions are not supported")
              : failure();
  if (intInfo.second != ptrInfo.second)
    return op ? op->emitError("incompatible operand shapes") : failure();
  return success();
}

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
