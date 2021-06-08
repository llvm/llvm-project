//===- ArmSVEDialect.cpp - MLIR ArmSVE dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSVE dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

static Type getI1SameShape(Type type);

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSVE/ArmSVE.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/ArmSVE/ArmSVETypes.cpp.inc"

void arm_sve::ArmSVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSVE/ArmSVE.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/ArmSVE/ArmSVETypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ScalableVectorType
//===----------------------------------------------------------------------===//

Type arm_sve::ArmSVEDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  {
    Type genType;
    auto parseResult = generatedTypeParser(parser.getBuilder().getContext(),
                                           parser, "vector", genType);
    if (parseResult.hasValue())
      return genType;
  }
  parser.emitError(typeLoc, "unknown type in ArmSVE dialect");
  return Type();
}

void arm_sve::ArmSVEDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected 'arm_sve' type kind");
}

//===----------------------------------------------------------------------===//
// ScalableVector versions of general helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the scalable vector of the same shape and containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto sVectorType = type.dyn_cast<arm_sve::ScalableVectorType>())
    return arm_sve::ScalableVectorType::get(type.getContext(),
                                            sVectorType.getShape(), i1Type);
  return nullptr;
}
