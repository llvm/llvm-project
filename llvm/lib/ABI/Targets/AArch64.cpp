//===- AArch64.cpp - AArch64 ABI Implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace abi {

class AArch64TargetInfo : public TargetInfo {
public:
  AArch64TargetInfo(TypeBuilder &TB, AArch64ABIKind Kind)
      : TB(TB), Kind(Kind) {}

  void computeInfo(FunctionInfo &FI) const override {
    if (!maybeCommonClassifyReturnType(FI))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    unsigned ArgNo = 0;
    unsigned NSRN = 0, NPRN = 0;
    for (auto &I : FI.arguments()) {
      const bool IsNamedArg =
          !FI.isVariadic() || ArgNo < FI.getNumRequiredArgs();
      ++ArgNo;
      I.Info = classifyArgumentType(I.ABIType, FI.isVariadic(), IsNamedArg,
                                    FI.getCallingConvention(), NSRN, NPRN);
    }
  }

private:
  [[maybe_unused]] TypeBuilder &TB;
  AArch64ABIKind Kind;

  ArgInfo classifyReturnType(const Type *RetTy, bool IsVariadicFn) const;
  ArgInfo classifyArgumentType(const Type *Ty, bool IsVariadicFn,
                               bool IsNamedArg, unsigned CallingConvention,
                               unsigned &NSRN, unsigned &NPRN) const;

  bool isDarwinPCS() const { return Kind == AArch64ABIKind::DarwinPCS; }
  bool passAsAggregateType(const Type *Ty) const;
};

std::unique_ptr<TargetInfo> createAArch64TargetInfo(TypeBuilder &TB,
                                                    AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetInfo>(TB, Kind);
}

static void reportNYI(StringRef Feature) {
  WithColor::warning()
      << Feature
      << " is not yet implemented for AArch64 in the LLVM ABI library.\n";
}

ArgInfo AArch64TargetInfo::classifyReturnType(const Type *RetTy,
                                              bool IsVariadicFn) const {
  if (RetTy->isVoid())
    return ArgInfo::getIgnore();

  if (RetTy->isVector()) {
    reportNYI("Vector return type handling");
    return ArgInfo::getDirect();
  }

  if (!passAsAggregateType(RetTy)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy)) {
      if (IntTy->isBitInt()) {
        reportNYI("BitInt return type handling");
        return ArgInfo::getDirect();
      }
      if (isPromotableInteger(IntTy) && isDarwinPCS()) {
        return ArgInfo::getExtend(IntTy);
      }
    }

    // Everything not handled above is returned directly.
    return ArgInfo::getDirect();
  }

  reportNYI("Aggregate return type handling");
  return ArgInfo::getDirect();
}

ArgInfo AArch64TargetInfo::classifyArgumentType(
    const Type *Ty, bool IsVariadicFn, bool IsNamedArg,
    unsigned CallingConvention, unsigned &NSRN, unsigned &NPRN) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO: Handle variadic functins here when Windows Arm64 EC is supported.

  if (Ty->isVector()) {
    reportNYI("Vector argument type handling");
    return ArgInfo::getDirect();
  }

  if (!passAsAggregateType(Ty)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt()) {
        reportNYI("BitInt argument type handling");
        return ArgInfo::getDirect();
      }
      if (isPromotableInteger(IntTy) && isDarwinPCS()) {
        return ArgInfo::getExtend(IntTy);
      }
    }

    // TODO: Legal vector types will update NSRN or NPRN.

    if (Ty->isFloat())
      NSRN = std::min(NSRN + 1, 8u);

    // Everything not handled above is returned directly.
    return ArgInfo::getDirect();
  }

  reportNYI("Aggregate argument type handling");
  return ArgInfo::getDirect();
}

bool AArch64TargetInfo::passAsAggregateType(const Type *Ty) const {
  // TODO: Handle SVE types. For now, they don't get through the type mapper.
  return isAggregateTypeForABI(Ty);
}

} // namespace abi
} // namespace llvm
