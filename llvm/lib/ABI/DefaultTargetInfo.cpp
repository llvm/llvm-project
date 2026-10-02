//===- DefaultTargetInfo.cpp - Default ABI classification -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/DefaultTargetInfo.h"
#include "llvm/Support/Casting.h"

using namespace llvm::abi;
using llvm::dyn_cast;

ArgInfo DefaultTargetInfo::classifyArgumentType(const Type *Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (RecordArgABI RAA = getRecordArgABI(Ty))
      return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(),
                                     /*ByVal=*/RAA == RAA_DirectInMemory);
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace());
  }

  if (const auto *IT = dyn_cast<IntegerType>(Ty)) {
    // A _BitInt too wide for the largest integer register goes indirect.
    if (IT->isBitInt() &&
        IT->getSizeInBits().getFixedValue() > getBitIntRegThreshold())
      return getNaturalAlignIndirect(Ty, getAllocaAddrSpace());
    if (isPromotableInteger(IT))
      return ArgInfo::getExtend(Ty);
  }

  return ArgInfo::getDirect();
}

ArgInfo DefaultTargetInfo::classifyReturnType(const Type *RetTy) const {
  if (RetTy->isVoid())
    return ArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());

  if (const auto *IT = dyn_cast<IntegerType>(RetTy)) {
    // A _BitInt too wide for the largest integer register goes indirect.
    if (IT->isBitInt() &&
        IT->getSizeInBits().getFixedValue() > getBitIntRegThreshold())
      return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());
    if (isPromotableInteger(IT))
      return ArgInfo::getExtend(RetTy);
  }

  return ArgInfo::getDirect();
}

void DefaultTargetInfo::computeInfo(FunctionInfo &FI) const {
  if (!maybeCommonClassifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (auto &I : FI.arguments())
    I.Info = classifyArgumentType(I.ABIType);
}
