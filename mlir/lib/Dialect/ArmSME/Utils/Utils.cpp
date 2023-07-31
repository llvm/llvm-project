//===- Utils.cpp - Utilities to support the ArmSME dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the ArmSME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/Utils/Utils.h"

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

using namespace mlir;
using namespace mlir::arm_sme;

static constexpr unsigned MinStreamingVectorLengthInBits = 128;

unsigned mlir::arm_sme::getSMETileSliceMinNumElts(Type type) {
  assert(isValidSMETileElementType(type) && "invalid tile type!");
  return MinStreamingVectorLengthInBits / type.getIntOrFloatBitWidth();
}

bool mlir::arm_sme::isValidSMETileElementType(Type type) {
  // TODO: add support for i128.
  return type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
         type.isInteger(64) || type.isF16() || type.isBF16() || type.isF32() ||
         type.isF64();
}

bool mlir::arm_sme::isValidSMETileVectorType(VectorType vType) {
  if ((vType.getRank() != 2) && vType.allDimsScalable())
    return false;

  // TODO: add support for i128.
  auto elemType = vType.getElementType();
  if (!isValidSMETileElementType(elemType))
    return false;

  unsigned minNumElts = arm_sme::getSMETileSliceMinNumElts(elemType);
  if (vType.getShape() != ArrayRef<int64_t>({minNumElts, minNumElts}))
    return false;

  return true;
}
