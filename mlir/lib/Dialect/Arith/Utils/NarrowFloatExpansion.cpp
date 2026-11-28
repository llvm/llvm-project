//===- NarrowFloatExpansion.cpp - Narrow float truncf/extf ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include <climits>

using namespace mlir;

arith::FloatFormatSpec
mlir::arith::floatFormatSpecFromSemantics(const llvm::fltSemantics &sem) {
  assert(sem.hasZero && sem.hasDenormals &&
         "the format needs a zero and subnormals (f8E8M0FNU has neither)");
  assert(!sem.hasExplicitIntegerBit && "the integer bit must be implicit");
  FloatFormatSpec spec;
  spec.sizeInBits = llvm::APFloat::semanticsSizeInBits(sem);
  assert(spec.sizeInBits <= 64 && "the encoding must fit in 64 bits");
  spec.mantBits = llvm::APFloat::semanticsPrecision(sem) - 1;
  spec.hasInf = llvm::APFloat::semanticsHasInf(sem);
  spec.isSigned = llvm::APFloat::semanticsHasSignedRepr(sem);
  if (spec.isSigned) {
    spec.signBitPos = spec.sizeInBits - 1;
    spec.expBits = spec.sizeInBits - 1 - spec.mantBits;
  } else {
    spec.signBitPos = UINT_MAX;
    spec.expBits = spec.sizeInBits - spec.mantBits;
  }
  assert(spec.expBits < 32 && "the exponent field must fit in 32 bits");
  spec.maxStoredExp = (1u << spec.expBits) - 1u;
  spec.bias = 1 - llvm::APFloat::semanticsMinExponent(sem);
  bool nanOnly = sem.nonFiniteBehavior == llvm::fltNonfiniteBehavior::NanOnly;
  spec.hasAllOnesNaN =
      nanOnly && sem.nanEncoding == llvm::fltNanEncoding::AllOnes;
  spec.isFNUZ =
      nanOnly && sem.nanEncoding == llvm::fltNanEncoding::NegativeZero;
  spec.maxFiniteBitsAbs =
      llvm::APFloat::getLargest(sem).bitcastToAPInt().getZExtValue();
  // getNaN asserts on a format without a NaN.
  if (sem.nonFiniteBehavior == llvm::fltNonfiniteBehavior::FiniteOnly)
    spec.quietNaNBits = spec.maxFiniteBitsAbs;
  else
    spec.quietNaNBits =
        llvm::APFloat::getNaN(sem).bitcastToAPInt().getZExtValue();
  return spec;
}

arith::FloatFormatSpec
mlir::arith::floatFormatSpecFromFloatType(FloatType floatTy) {
  return floatFormatSpecFromSemantics(floatTy.getFloatSemantics());
}
