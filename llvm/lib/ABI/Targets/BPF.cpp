//===- BPF.cpp - BPF ABI Implementation ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"

namespace llvm::abi {

class BPFTargetInfo : public TargetInfo {
private:
  TypeBuilder &TB;

  ArgInfo classifyReturnType(const Type *RetTy) const {
    if (RetTy->isVoid())
      return ArgInfo::getIgnore();

    if (isAggregateTypeForABI(RetTy)) {
      if (RetTy->isZeroSize())
        return ArgInfo::getIgnore();
      return getNaturalAlignIndirect(RetTy, /*ByVal=*/false);
    }

    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy)) {
      if (IntTy->isBitInt() && IntTy->getSizeInBits().getFixedValue() > 128)
        return getNaturalAlignIndirect(RetTy, /*ByVal=*/false);
    }

    return ArgInfo::getDirect();
  }

  ArgInfo classifyArgumentType(const Type *ArgTy) const {
    if (const auto *RT = dyn_cast<RecordType>(ArgTy))
      if (RT->isTransparentUnion() && RT->getNumFields() > 0)
        ArgTy = RT->getFields()[0].FieldType;

    if (isAggregateTypeForABI(ArgTy)) {
      if (ArgTy->isZeroSize())
        return ArgInfo::getIgnore();

      auto SizeInBits = ArgTy->getSizeInBits().getFixedValue();
      if (SizeInBits <= 128) {
        const Type *CoerceTy;
        if (SizeInBits <= 64) {
          CoerceTy = TB.getIntegerType(alignTo(SizeInBits, 8), Align(8), false);
        } else {
          const Type *RegTy = TB.getIntegerType(64, Align(8), false);
          CoerceTy = TB.getArrayType(RegTy, 2, 128);
        }
        return ArgInfo::getDirect(CoerceTy);
      }

      return getNaturalAlignIndirect(ArgTy, /*ByVal=*/true);
    }

    if (const auto *IntTy = dyn_cast<IntegerType>(ArgTy)) {
      if (IntTy->isBitInt() && IntTy->getSizeInBits().getFixedValue() > 128)
        return getNaturalAlignIndirect(ArgTy, /*ByVal=*/true);

      if (isPromotableInteger(IntTy))
        return ArgInfo::getExtend(ArgTy);
    }

    return ArgInfo::getDirect();
  }

public:
  BPFTargetInfo(TypeBuilder &TB) : TB(TB) {}

  void computeInfo(FunctionInfo &FI) const override {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments())
      I.Info = classifyArgumentType(I.ABIType);
  }
};

std::unique_ptr<TargetInfo> createBPFTargetInfo(TypeBuilder &TB) {
  return std::make_unique<BPFTargetInfo>(TB);
}

} // namespace llvm::abi
