//===- AttrToLLVMConverter.cpp - Arith attributes conversion to LLVM ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"

using namespace mlir;

// Map arithmetic fastmath enum values to LLVMIR enum values.
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

// Create an LLVM fastmath attribute from a given arithmetic fastmath attribute.
LLVM::FastmathFlagsAttr
mlir::arith::convertArithFastMathAttrToLLVM(arith::FastMathFlagsAttr fmfAttr) {
  arith::FastMathFlags arithFMF = fmfAttr.getValue();
  return LLVM::FastmathFlagsAttr::get(
      fmfAttr.getContext(), convertArithFastMathFlagsToLLVM(arithFMF));
}

// Map arithmetic overflow enum values to LLVMIR enum values.
LLVM::IntegerOverflowFlags mlir::arith::convertArithOveflowFlagsToLLVM(
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

// Create an LLVM overflow attribute from a given arithmetic overflow attribute.
LLVM::IntegerOverflowFlagsAttr mlir::arith::convertArithOveflowAttrToLLVM(
    arith::IntegerOverflowFlagsAttr flagsAttr) {
  arith::IntegerOverflowFlags arithFlags = flagsAttr.getValue();
  return LLVM::IntegerOverflowFlagsAttr::get(
      flagsAttr.getContext(), convertArithOveflowFlagsToLLVM(arithFlags));
}
