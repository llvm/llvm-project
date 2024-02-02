//===- AtomicOps.cpp - MLIR SPIR-V Atomic Ops  ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the atomic operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

template <typename T>
static StringRef stringifyTypeName();

template <>
StringRef stringifyTypeName<IntegerType>() {
  return "integer";
}

template <>
StringRef stringifyTypeName<FloatType>() {
  return "float";
}

// Verifies an atomic update op.
template <typename ExpectedElementType>
static LogicalResult verifyAtomicUpdateOp(Operation *op) {
  auto ptrType = llvm::cast<spirv::PointerType>(op->getOperand(0).getType());
  auto elementType = ptrType.getPointeeType();
  if (!llvm::isa<ExpectedElementType>(elementType))
    return op->emitOpError() << "pointer operand must point to an "
                             << stringifyTypeName<ExpectedElementType>()
                             << " value, found " << elementType;

  auto memorySemantics =
      op->getAttrOfType<spirv::MemorySemanticsAttr>(kSemanticsAttrName)
          .getValue();
  if (failed(verifyMemorySemantics(op, memorySemantics))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.AtomicAndOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicAndOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIAddOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIAddOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.EXT.AtomicFAddOp
//===----------------------------------------------------------------------===//

LogicalResult EXTAtomicFAddOp::verify() {
  return verifyAtomicUpdateOp<FloatType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIDecrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIDecrementOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIIncrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIIncrementOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicISubOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicISubOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicOrOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicOrOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMaxOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMinOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMaxOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMinOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicXorOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicXorOp::verify() {
  return verifyAtomicUpdateOp<IntegerType>(getOperation());
}

} // namespace mlir::spirv
