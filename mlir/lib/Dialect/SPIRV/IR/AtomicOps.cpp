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
template <typename AtomicOpTy, typename ExpectedElementType>
static LogicalResult verifyAtomicUpdateOp(Operation *op) {
  auto ptrType = llvm::cast<spirv::PointerType>(op->getOperand(0).getType());
  auto elementType = ptrType.getPointeeType();
  if (!llvm::isa<ExpectedElementType>(elementType))
    return op->emitOpError() << "pointer operand must point to an "
                             << stringifyTypeName<ExpectedElementType>()
                             << " value, found " << elementType;

  StringAttr semanticsAttrName =
      AtomicOpTy::getSemanticsAttrName(op->getName());
  auto memorySemantics =
      op->getAttrOfType<spirv::MemorySemanticsAttr>(semanticsAttrName)
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
  return verifyAtomicUpdateOp<AtomicAndOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIAddOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIAddOp::verify() {
  return verifyAtomicUpdateOp<AtomicIAddOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.EXT.AtomicFAddOp
//===----------------------------------------------------------------------===//

LogicalResult EXTAtomicFAddOp::verify() {
  return verifyAtomicUpdateOp<EXTAtomicFAddOp, FloatType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIDecrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIDecrementOp::verify() {
  return verifyAtomicUpdateOp<AtomicIDecrementOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIIncrementOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicIIncrementOp::verify() {
  return verifyAtomicUpdateOp<AtomicIIncrementOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicISubOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicISubOp::verify() {
  return verifyAtomicUpdateOp<AtomicISubOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicOrOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicOrOp::verify() {
  return verifyAtomicUpdateOp<AtomicOrOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMaxOp::verify() {
  return verifyAtomicUpdateOp<AtomicSMaxOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicSMinOp::verify() {
  return verifyAtomicUpdateOp<AtomicSMinOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMaxOp::verify() {
  return verifyAtomicUpdateOp<AtomicUMaxOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMinOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUMinOp::verify() {
  return verifyAtomicUpdateOp<AtomicUMinOp, IntegerType>(getOperation());
}

//===----------------------------------------------------------------------===//
// spirv.AtomicXorOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicXorOp::verify() {
  return verifyAtomicUpdateOp<AtomicXorOp, IntegerType>(getOperation());
}

} // namespace mlir::spirv
