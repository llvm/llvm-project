//===- BPF.cpp - BPF ABI Implementation ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/ABIFunctionInfo.h"
#include "llvm/ABI/ABIInfo.h"
#include "llvm/ABI/TargetCodegenInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"

namespace llvm::abi {

class BPFABIInfo : public ABIInfo {
private:
  TypeBuilder &TB;

  bool isAggregateType(const Type *Ty) const {
    return Ty->isStruct() || Ty->isVector() || Ty->isArray();
  }

public:
  BPFABIInfo(TypeBuilder &TypeBuilder) : TB(TypeBuilder) {}

  ABIArgInfo classifyReturnType(const Type *RetTy) const {
    if (RetTy->isVoid())
      return ABIArgInfo::getIgnore();

    if (isAggregateType(RetTy)) {
      auto SizeInBits = RetTy->getSizeInBits().getFixedValue();
      if (SizeInBits == 0)
        return ABIArgInfo::getIgnore();
      return ABIArgInfo::getIndirect(RetTy->getAlignment().value());
    }

    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy)) {
      if (IntTy->isBitInt() && IntTy->getSizeInBits().getFixedValue() > 128)
        return ABIArgInfo::getIndirect(RetTy->getAlignment().value());
    }

    return ABIArgInfo::getDirect();
  }

  ABIArgInfo classifyArgumentType(const Type *ArgTy) const {
    if (isAggregateType(ArgTy)) {
      auto SizeInBits = ArgTy->getSizeInBits().getFixedValue();
      if (SizeInBits == 0)
        return ABIArgInfo::getIgnore();

      if (SizeInBits <= 128) {
        const Type *CoerceTy;
        if (SizeInBits <= 64) {
          auto AlignedBits = alignTo(SizeInBits, 8);
          CoerceTy = TB.getIntegerType(AlignedBits, Align(8), false);
        } else {
          const Type *RegTy = TB.getIntegerType(64, Align(8), false);
          CoerceTy = TB.getArrayType(RegTy, 2);
        }
        return ABIArgInfo::getDirect(CoerceTy);
      }

      return ABIArgInfo::getIndirect(ArgTy->getAlignment().value());
    }

    if (const auto *IntTy = dyn_cast<IntegerType>(ArgTy)) {
      auto BitWidth = IntTy->getSizeInBits().getFixedValue();
      if (IntTy->isBitInt() && BitWidth > 128)
        return ABIArgInfo::getIndirect(ArgTy->getAlignment().value());

      if (IntTy->isPromotableIntegerType())
        return ABIArgInfo::getExtend(ArgTy);
    }
    return ABIArgInfo::getDirect();
  }

  void computeInfo(ABIFunctionInfo &FI) const override {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments()) {
      I.ArgInfo = classifyArgumentType(I.ABIType);
    }
  }
};

class BPFTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  BPFTargetCodeGenInfo(TypeBuilder &TB)
      : TargetCodeGenInfo(std::make_unique<BPFABIInfo>(TB)) {}
};

std::unique_ptr<TargetCodeGenInfo> createBPFTargetCodeGenInfo(TypeBuilder &TB) {
  return std::make_unique<BPFTargetCodeGenInfo>(TB);
}

} // namespace llvm::abi
