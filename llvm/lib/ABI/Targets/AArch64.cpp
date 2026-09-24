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
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>
#include <cstdint>

namespace llvm {
namespace abi {

class AArch64TargetInfo : public TargetInfo {
public:
  AArch64TargetInfo(TypeBuilder &TB, const AArch64ABIOptions &Opts)
      : TargetInfo(TB), Opts(Opts) {}

  const ABICompatInfo &getABICompatInfo() const override {
    return Opts.CompatInfo;
  }

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
  AArch64ABIOptions Opts;

  ArgInfo classifyReturnType(const Type *RetTy, bool IsVariadicFn) const;
  ArgInfo classifyArgumentType(const Type *Ty, bool IsVariadicFn,
                               bool IsNamedArg, unsigned CallingConvention,
                               unsigned &NSRN, unsigned &NPRN) const;

  bool isDarwinPCS() const { return Opts.Kind == AArch64ABIKind::DarwinPCS; }
  bool isSoftFloat() const { return Opts.Kind == AArch64ABIKind::AAPCSSoft; }
  bool passAsAggregateType(const Type *Ty) const;

  bool isHomogeneousAggregateBaseType(const Type *Ty) const override;
  bool isHomogeneousAggregateSmallEnough(const Type *Base,
                                         uint64_t Members) const override;
  bool isZeroLengthBitfieldPermittedInHomogeneousAggregate() const override;
  bool isPermittedToBeHomogeneousAggregate(const RecordType *RT) const override;
};

std::unique_ptr<TargetInfo>
createAArch64TargetInfo(TypeBuilder &TB, const AArch64ABIOptions &Opts) {
  return std::make_unique<AArch64TargetInfo>(TB, Opts);
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
    return ArgInfo::getIgnore();
  }

  if (!passAsAggregateType(RetTy)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy)) {
      if (IntTy->isBitInt())
        if (RetTy->getSizeInBits().getFixedValue() > 128)
          return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());

      if (isPromotableInteger(IntTy) && isDarwinPCS())
        return ArgInfo::getExtend(IntTy);
    }

    // Everything not handled above is returned directly.
    return ArgInfo::getDirect();
  }

  uint64_t Size = RetTy->getFixedSizeInBitsOrZero();
  if (!RetTy->isSVESizelessType() && (RetTy->isEmptyRecord() || Size == 0))
    return ArgInfo::getIgnore();

  const Type *Base = nullptr;
  uint64_t Members = 0;
  if (isHomogeneousAggregate(RetTy, Base, Members) &&
      !(Opts.IsILP32 && IsVariadicFn)) {
    // Homogeneous Floating-point Aggregates (HFAs) are returned directly.
    return ArgInfo::getDirect();
  }

  reportNYI("Aggregate return type handling");
  return ArgInfo::getIgnore();
}

ArgInfo AArch64TargetInfo::classifyArgumentType(
    const Type *Ty, bool IsVariadicFn, bool IsNamedArg,
    unsigned CallingConvention, unsigned &NSRN, unsigned &NPRN) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (Ty->isVector()) {
    reportNYI("Vector argument type handling");
    return ArgInfo::getIgnore();
  }

  if (!passAsAggregateType(Ty)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt())
        if (Ty->getSizeInBits().getFixedValue() > 128)
          return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(),
                                         /*ByVal=*/false);

      if (isPromotableInteger(IntTy) && isDarwinPCS())
        return ArgInfo::getExtend(IntTy);
    }

    // TODO: Legal vector types will update NSRN or NPRN.

    if (Ty->isFloat())
      NSRN = std::min(NSRN + 1, 8u);

    // Everything not handled above is returned directly.
    return ArgInfo::getDirect();
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (auto RecordRAA = getRecordArgABI(Ty)) {
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(),
                                   /*ByVal=*/RecordRAA ==
                                       RecordArgABI::RAA_DirectInMemory);
  }

  // AAPCS64 does not say that empty C records are ignored as arguments,
  // but other compilers do so in certain situations, and we copy that behavior.
  uint64_t Size = Ty->getFixedSizeInBitsOrZero();
  if (!Ty->isSVESizelessType() && (Ty->isEmptyRecord() || Size == 0)) {
    // Darwin overrides the psABI here to ignore all empty records in all modes.
    // The ABI explicitly says that an empty class shall be treated as if its
    // type were an aggregate with a single member of type unsigned byte.
    if (!Opts.IsCXX || isDarwinPCS())
      return ArgInfo::getIgnore();

    // In C++ mode, arguments which have sizeof() == 0 (which are non-standard
    // C++) are ignored. This isn't defined by any standard, so we copy GCC's
    // behaviour here.
    if (Size == 0)
      return ArgInfo::getIgnore();
  }

  // Homogeneous Floating-point Aggregates (HFAs) need to be expanded.
  const Type *Base = nullptr;
  uint64_t Members = 0;
  bool IsWin64 = Opts.Kind == AArch64ABIKind::Win64 ||
                 CallingConvention == llvm::CallingConv::Win64;
  bool IsWinVariadic = IsWin64 && IsVariadicFn;
  // In variadic functions on Windows, all composite types are treated alike,
  // no special handling of HFAs/HVAs.
  if (!IsWinVariadic && isHomogeneousAggregate(Ty, Base, Members)) {
    NSRN = std::min(NSRN + Members, uint64_t(8));
    uint64_t BaseAllocSizeInBits = Base->getTypeAllocSize().getFixedValue() * 8;
    const Type *CoerceTy =
        TB.getArrayType(Base, Members, Members * BaseAllocSizeInBits);
    if (Opts.Kind != AArch64ABIKind::AAPCS)
      return ArgInfo::getDirect(CoerceTy);

    // For HFAs/HVAs, cap the argument alignment to 16, otherwise
    // set it to 8 according to the AAPCS64 document.
    unsigned TyAlign = Ty->getUnadjustedAlignment().value();
    TyAlign = (TyAlign >= 16) ? 16 : 8;
    return ArgInfo::getDirect(CoerceTy, /*Offset=*/0, llvm::Align(TyAlign));
  }

  reportNYI("Aggregate argument type handling");
  return ArgInfo::getIgnore();
}

bool AArch64TargetInfo::passAsAggregateType(const Type *Ty) const {
  // TODO: Handle SVE types. For now, they don't get through the type mapper.
  return isAggregateTypeForABI(Ty);
}

bool AArch64TargetInfo::isHomogeneousAggregateBaseType(const Type *Ty) const {
  // Soft-float ABI: no types are homogeneous aggregates.
  if (isSoftFloat())
    return false;

  // Homogeneous aggregates for AAPCS64 must have base types of a floating
  // point type or a short-vector type.
  if (Ty->isFloat())
    return true;

  if (const auto *VT = dyn_cast<VectorType>(Ty)) {
    if (VT->isScalable() || VT->isSVEData() || VT->isSVEPredicate())
      return false;

    // Clang's getTypeSize for non-power-of-2 vectors rounds the width up to
    // the next power-of-2 alignment (e.g. 3 x float is 96 bits of payload but
    // 128 bits of ABI size), so those vectors are short-vector HVA bases.
    uint64_t VecSize =
        bit_ceil(std::max<uint64_t>(8, VT->getSizeInBits().getFixedValue()));
    if (VecSize == 64 || VecSize == 128)
      return true;
  }
  return false;
}

bool AArch64TargetInfo::isHomogeneousAggregateSmallEnough(
    const Type * /*Base*/, uint64_t Members) const {
  return Members <= 4;
}

bool AArch64TargetInfo::isZeroLengthBitfieldPermittedInHomogeneousAggregate()
    const {
  // AAPCS64 applies homogeneity to the output of the data layout decision, so
  // zero-length bitfields do not affect homogeneity.
  return true;
}

bool AArch64TargetInfo::isPermittedToBeHomogeneousAggregate(
    const RecordType *RT) const {
  if (Opts.IsMicrosoftCXXABI && RT->isCXXRecord()) {
    // This won't always return false, but we don't have enough information to
    // perform the full check correctly yet.
    reportNYI("MicrosoftCXXABI homogeneous record classification");
    return false;
  }

  return true;
}

} // namespace abi
} // namespace llvm
