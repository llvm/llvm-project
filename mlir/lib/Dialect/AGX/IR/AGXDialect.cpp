//===- AGXDialect.cpp - MLIR AGX dialect implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AGX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AGX/IR/AGXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::agx;

#include "mlir/Dialect/AGX/IR/AGXDialect.cpp.inc"
#include "mlir/Dialect/AGX/IR/AGXEnums.cpp.inc"

void AGXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AGX/IR/AGX.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AGX/IR/AGXAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleOp::verify() {
  Type type = getValue().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (!elementType.isIntOrFloat())
    return emitOpError() << "requires integer or floating point values, got "
                         << type;
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (vectorType.isScalable())
      return emitOpError() << "does not support scalable vectors";
    bits *= vectorType.getNumElements();
  }
  // The value is split into 32-bit parts; 16-bit values are shuffled
  // directly.
  if (bits != 16 && bits % 32 != 0)
    return emitOpError() << "requires a 16-bit value or a multiple of 32 bits, "
                            "got "
                         << type;
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  bool isFloatKind = getKind() == ReduceKind::minnumf ||
                     getKind() == ReduceKind::maxnumf;
  bool isArithKind = getKind() == ReduceKind::add ||
                     getKind() == ReduceKind::mul;
  Type type = getValue().getType();
  if (isa<FloatType>(type)) {
    if (!isFloatKind && !isArithKind)
      return emitOpError() << "reduction '" << stringifyReduceKind(getKind())
                           << "' does not apply to floating point values";
  } else if (isFloatKind) {
    return emitOpError() << "reduction '" << stringifyReduceKind(getKind())
                         << "' requires floating point values";
  } else if (getKind() == ReduceKind::mul) {
    return emitOpError() << "there is no integer multiplication reduction";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

/// The accumulator may be the input type, or f32 for f16 and bf16 inputs.
static bool isValidAccumulator(Type input, Type acc) {
  return input == acc || (acc.isF32() && (input.isF16() || input.isBF16()));
}

LogicalResult MMAOp::verify() {
  Type input = getElementTypeOrSelf(getA().getType());
  Type acc = getElementTypeOrSelf(getC().getType());
  if (!isValidAccumulator(input, acc))
    return emitOpError() << "accumulator " << acc << " cannot accumulate "
                         << input << " products";
  return success();
}

static LogicalResult verifyFragmentAccess(Operation *op, MemRefType memref,
                                          VectorType fragment) {
  if (memref.getRank() != 2)
    return op->emitOpError() << "requires a 2-D memref, got " << memref;
  if (memref.getElementType() != fragment.getElementType())
    return op->emitOpError() << "fragment " << fragment
                             << " does not match the elements of " << memref;
  return success();
}

LogicalResult MatrixLoadOp::verify() {
  return verifyFragmentAccess(*this, getBase().getType(), getResult().getType());
}

LogicalResult MatrixStoreOp::verify() {
  return verifyFragmentAccess(*this, getBase().getType(), getValue().getType());
}

LogicalResult MMATileOp::verify() {
  auto a = getA().getType(), b = getB().getType();
  if (a.getRank() != 2 || b.getRank() != 2)
    return emitOpError() << "requires 2-D memrefs";
  if (a.getElementType() != b.getElementType())
    return emitOpError() << "requires A and B with the same element type";
  for (int64_t dim : {getM(), getN(), getK()})
    if (dim % 8 != 0)
      return emitOpError() << "requires m, n and k to be multiples of 8";
  VectorType acc = getAcc().getType();
  if (!isValidAccumulator(a.getElementType(), acc.getElementType()))
    return emitOpError() << "accumulator " << acc.getElementType()
                         << " cannot accumulate " << a.getElementType()
                         << " products";
  if (acc.getNumElements() != getM() * getN() / 32)
    return emitOpError() << "requires an accumulator of m * n / 32 = "
                         << getM() * getN() / 32 << " elements per thread, got "
                         << acc.getNumElements();
  return success();
}

//===----------------------------------------------------------------------===//
// FMALaneShiftOp
//===----------------------------------------------------------------------===//

LogicalResult FMALaneShiftOp::verify() {
  uint32_t width = getWidth();
  if (width != 2 && width != 4 && width != 16)
    return emitOpError() << "requires a segment width of 2, 4 or 16, got "
                         << width;
  if (getShift() >= width)
    return emitOpError() << "shift " << getShift()
                         << " does not stay inside a segment of " << width
                         << " lanes";
  return success();
}

//===----------------------------------------------------------------------===//
// UniformRegionOp
//===----------------------------------------------------------------------===//

LogicalResult UniformRegionOp::verifyRegions() {
  auto yield = dyn_cast<YieldOp>(getBody().front().getTerminator());
  if (!yield)
    return emitOpError() << "requires an agx.yield terminator";
  if (yield.getValues().getTypes() != getResultTypes())
    return emitOpError() << "yields " << yield.getValues().getTypes()
                         << " but returns " << getResultTypes();
  // Thread-dependent values would make the region differ between threads.
  WalkResult r = getBody().walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "gpu.thread_id" || name == "gpu.block_id" ||
        name == "gpu.global_id" || name == "gpu.lane_id" ||
        name == "gpu.subgroup_id" || isa<CoreIdOp>(op)) {
      op->emitOpError() << "is thread-dependent and cannot be in an "
                           "agx.uniform_region";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AGX/IR/AGXAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AGX/IR/AGX.cpp.inc"
