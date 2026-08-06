//===- AttrToLLVMConverter.cpp - Arith attributes conversion to LLVM ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"

using namespace mlir;

LLVM::FastmathFlags
mlir::arith::convertArithFastMathFlagsToLLVM(arith::FastMathFlags arithFMF) {
  LLVM::FastmathFlags llvmFMF{};
  const std::pair<arith::FastMathFlags, LLVM::FastmathFlags> flags[] = {
      {arith::FastMathFlags::nnan, LLVM::FastmathFlags::nnan},
      {arith::FastMathFlags::ninf, LLVM::FastmathFlags::ninf},
      {arith::FastMathFlags::nsz, LLVM::FastmathFlags::nsz},
      {arith::FastMathFlags::arcp, LLVM::FastmathFlags::arcp},
      {arith::FastMathFlags::contract, LLVM::FastmathFlags::contract},
      {arith::FastMathFlags::afn, LLVM::FastmathFlags::afn},
      {arith::FastMathFlags::reassoc, LLVM::FastmathFlags::reassoc}};
  for (auto [arithFlag, llvmFlag] : flags) {
    if (bitEnumContainsAny(arithFMF, arithFlag))
      llvmFMF = llvmFMF | llvmFlag;
  }
  return llvmFMF;
}

LLVM::FastmathFlagsAttr
mlir::arith::convertArithFastMathAttrToLLVM(arith::FastMathFlagsAttr fmfAttr) {
  arith::FastMathFlags arithFMF = fmfAttr.getValue();
  return LLVM::FastmathFlagsAttr::get(
      fmfAttr.getContext(), convertArithFastMathFlagsToLLVM(arithFMF));
}

LLVM::IntegerOverflowFlags mlir::arith::convertArithOverflowFlagsToLLVM(
    arith::IntegerOverflowFlags arithFlags) {
  LLVM::IntegerOverflowFlags llvmFlags{};
  const std::pair<arith::IntegerOverflowFlags, LLVM::IntegerOverflowFlags>
      flags[] = {
          {arith::IntegerOverflowFlags::nsw, LLVM::IntegerOverflowFlags::nsw},
          {arith::IntegerOverflowFlags::nuw, LLVM::IntegerOverflowFlags::nuw}};
  for (auto [arithFlag, llvmFlag] : flags) {
    if (bitEnumContainsAny(arithFlags, arithFlag))
      llvmFlags = llvmFlags | llvmFlag;
  }
  return llvmFlags;
}

LLVM::RoundingMode
mlir::arith::convertArithRoundingModeToLLVM(arith::RoundingMode roundingMode) {
  switch (roundingMode) {
  case arith::RoundingMode::downward:
    return LLVM::RoundingMode::TowardNegative;
  case arith::RoundingMode::to_nearest_away:
    return LLVM::RoundingMode::NearestTiesToAway;
  case arith::RoundingMode::to_nearest_even:
    return LLVM::RoundingMode::NearestTiesToEven;
  case arith::RoundingMode::toward_zero:
    return LLVM::RoundingMode::TowardZero;
  case arith::RoundingMode::upward:
    return LLVM::RoundingMode::TowardPositive;
  case arith::RoundingMode::unknown:
    return LLVM::RoundingMode::Dynamic;
  }
  llvm_unreachable("Unhandled rounding mode");
}

LLVM::RoundingModeAttr mlir::arith::convertArithRoundingModeAttrToLLVM(
    arith::RoundingModeAttr roundingModeAttr) {
  assert(roundingModeAttr && "Expecting valid attribute");
  return LLVM::RoundingModeAttr::get(
      roundingModeAttr.getContext(),
      convertArithRoundingModeToLLVM(roundingModeAttr.getValue()));
}

LLVM::FPExceptionBehaviorAttr
mlir::arith::getLLVMDefaultFPExceptionBehavior(MLIRContext &context) {
  return LLVM::FPExceptionBehaviorAttr::get(&context,
                                            LLVM::FPExceptionBehavior::Ignore);
}

LLVM::FPExceptionBehavior mlir::arith::convertArithFPExceptionBehaviorToLLVM(
    arith::FPExceptionMode exceptionMode, bool strictExcept) {
  // A strict-exception requirement always maps to `strict`, which preserves
  // both the side effects and the exact exception semantics regardless of the
  // exception mode.
  if (strictExcept)
    return LLVM::FPExceptionBehavior::Strict;
  // Without the strict requirement, a masked environment may ignore exceptions,
  // while an unmasked or unknown environment must conservatively assume that
  // traps may occur.
  if (exceptionMode == arith::FPExceptionMode::Masked)
    return LLVM::FPExceptionBehavior::Ignore;
  return LLVM::FPExceptionBehavior::MayTrap;
}
