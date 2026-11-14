//===- NVVMIndexSRegOps.cpp - NVVM dialect index registers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `IndexSReg` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMIndexSRegOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/ConstantRange.h"

#include <optional>

using namespace mlir;
using namespace mlir::NVVM;

/// Infer the result ranges for the NVVM SpecialRangeableRegisterOp that might
/// have ConstantRangeAttr.
static void nvvmInferResultRanges(std::optional<LLVM::ConstantRangeAttr> range,
                                  Value result,
                                  ArrayRef<::mlir::ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRanges) {
  if (range) {
    // ConstantRange uses a half-open interval of bit patterns, whereas
    // ConstantIntRanges uses inclusive signed and unsigned bounds. Derive
    // each interpretation separately to handle wrapping and sign boundaries.
    llvm::ConstantRange valueRange(range->getLower(), range->getUpper());
    if (!valueRange.isEmptySet()) {
      setResultRanges(result,
                      {valueRange.getUnsignedMin(), valueRange.getUnsignedMax(),
                       valueRange.getSignedMin(), valueRange.getSignedMax()});
      return;
    }
  }
  // ConstantIntRanges cannot represent an empty set. Conservatively fall back
  // to the full range rather than deriving invalid inclusive bounds.
  setResultRanges(result, IntegerValueRange::getMaxRange(result).getValue());
}

/// Verify the range attribute satisfies LLVM ConstantRange constructor
/// requirements for NVVM SpecialRangeableRegisterOp.
static LogicalResult
verifyConstantRangeAttr(Operation *op,
                        std::optional<LLVM::ConstantRangeAttr> rangeAttr) {
  if (!rangeAttr)
    return success();

  const llvm::APInt &lower = rangeAttr->getLower();
  const llvm::APInt &upper = rangeAttr->getUpper();

  // Check LLVM ConstantRange constructor condition
  if (lower == upper && !lower.isMaxValue() && !lower.isMinValue()) {
    unsigned bitWidth = lower.getBitWidth();
    llvm::APInt minVal = llvm::APInt::getMinValue(bitWidth);
    llvm::APInt maxVal = llvm::APInt::getMaxValue(bitWidth);
    return op->emitOpError(
               "invalid range attribute: Lower == Upper, but they aren't min (")
           << llvm::toString(minVal, 10, false) << ") or max ("
           << llvm::toString(maxVal, 10, false)
           << ") value! This is an invalid constant range.";
  }

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMIndexSRegOps.cpp.inc"
