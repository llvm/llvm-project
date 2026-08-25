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
#include "llvm/Support/TypeSize.h"
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

  const VectorType *
  convertFixedToScalableVectorType(const VectorType *VT) const;

  ArgInfo coerceIllegalVector(const Type *Ty, unsigned &NSRN,
                              unsigned &NPRN) const;

  bool isIllegalVectorType(const Type *Ty) const;

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

  if (const auto *VT = dyn_cast<VectorType>(RetTy)) {
    if (VT->isFixedLengthSVEData() || VT->isFixedLengthSVEPredicate()) {
      unsigned NSRN = 0, NPRN = 0;
      return coerceIllegalVector(VT, NSRN, NPRN);
    }

    // Large vector types should be returned via memory.
    if (VT->getABISizeInBits() > 128)
      return getNaturalAlignIndirect(RetTy);
  }

  if (!passAsAggregateType(RetTy)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy)) {
      if (IntTy->isBitInt())
        if (RetTy->getSizeInBits().getFixedValue() > 128)
          return getNaturalAlignIndirect(RetTy);

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

  // Arm64EC variadic functions classify their arguments with the x86-64
  // rules rather than the AArch64 ones.
  if (IsVariadicFn && Opts.IsWindowsArm64EC) {
    reportNYI("Arm64EC variadic argument handling");
    return ArgInfo::getIgnore();
  }

  // Handle illegal vector types here.
  if (isIllegalVectorType(Ty))
    return coerceIllegalVector(Ty, NSRN, NPRN);

  if (!passAsAggregateType(Ty)) {
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt())
        if (Ty->getSizeInBits().getFixedValue() > 128)
          return getNaturalAlignIndirect(Ty, /*ByVal=*/false);

      if (isPromotableInteger(IntTy) && isDarwinPCS())
        return ArgInfo::getExtend(IntTy);
    }

    // Predicates and svcount_t are passed in a predicate register. Legal
    // vectors, SVE data vectors, and floating-point types are passed in a
    // SIMD and floating-point register. A tuple occupies one register of the
    // appropriate kind per vector it contains.
    if (const auto *VT = dyn_cast<VectorType>(Ty)) {
      if (VT->isSVEPredicate() || VT->isSVECount())
        NPRN = std::min(NPRN + 1, 4u);
      else
        NSRN = std::min(NSRN + 1, 8u);
    } else if (const auto *TT = dyn_cast<TupleType>(Ty)) {
      if (TT->getVectorType()->isSVEPredicate())
        NPRN = std::min(NPRN + TT->getNumVectors(), 4u);
      else
        NSRN = std::min(NSRN + TT->getNumVectors(), 8u);
    } else if (Ty->isFloat()) {
      NSRN = std::min(NSRN + 1, 8u);
    }

    // Everything not handled above is returned directly.
    return ArgInfo::getDirect();
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (auto RecordRAA = getRecordArgABI(Ty)) {
    return getNaturalAlignIndirect(Ty, RecordRAA ==
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
  if (Opts.Kind == AArch64ABIKind::AAPCS && Ty->isSVESizelessType()) {
    // svcount_t and the single-vector types occupy a register of their own,
    // so only the data and predicate tuples are passed as aggregates.
    const auto *TupleTy = dyn_cast<TupleType>(Ty);
    assert((!TupleTy || TupleTy->getNumVectors() > 1) &&
           "unexpected single vector tuple");
    return TupleTy && !TupleTy->getVectorType()->isSVECount();
  }
  return isAggregateTypeForABI(Ty);
}

/// Returns the scalable vector type that \p VT, a fixed-length SVE vector,
/// is passed as. A scalable SVE vector holds 128 bits per granule, so the
/// scalable element count is 128 divided by the element size, regardless of
/// how many elements the fixed-length type has.
const VectorType *AArch64TargetInfo::convertFixedToScalableVectorType(
    const VectorType *VT) const {
  // TODO: Verify that this correctly handles MFloat8 when we decide on a
  // mapping for that type.

  if (VT->isFixedLengthSVEPredicate())
    return TB.getScalablePredicateVectorType();

  assert(VT->isFixedLengthSVEData() && "expected a fixed-length SVE vector!");

  const Type *EltTy = VT->getElementType();
  uint64_t EltBits = EltTy->getSizeInBits().getFixedValue();
  assert(EltBits >= 8 && EltBits <= 64 && isPowerOf2_64(EltBits) &&
         "unexpected element type for SVE data vector!");

  return TB.getVectorType(EltTy, ElementCount::getScalable(128 / EltBits),
                          llvm::Align(16), VectorKind::SVEData);
}

ArgInfo AArch64TargetInfo::coerceIllegalVector(const Type *Ty, unsigned &NSRN,
                                               unsigned &NPRN) const {
  const auto *VT = cast<VectorType>(Ty);

  if (VT->isFixedLengthSVEPredicate()) {
    // Fixed-length predicates are described with 8-bit elements, but they are
    // passed in a predicate register as a scalable vector of 16 one-bit
    // elements.
    assert(isa<IntegerType>(VT->getElementType()) &&
           VT->getElementType()->getSizeInBits().getFixedValue() == 8 &&
           "unexpected element type for SVE predicate!");
    NPRN = std::min(NPRN + 1, 4u);
    return ArgInfo::getDirect(TB.getScalablePredicateVectorType());
  }

  if (VT->isFixedLengthSVEData()) {
    NSRN = std::min(NSRN + 1, 8u);
    return ArgInfo::getDirect(convertFixedToScalableVectorType(VT));
  }

  uint64_t Size = VT->getABISizeInBits();
  // Android promotes <2 x i8> to i16, not i32
  if (Opts.IsAndroidOrOHOS && (Size <= 16)) {
    auto *ResType = TB.getIntegerType(16, llvm::Align(2), /*Signed=*/false);
    return ArgInfo::getDirect(ResType);
  }
  const Type *I32 = TB.getIntegerType(32, llvm::Align(4), /*Signed=*/false);
  if (Size <= 32)
    return ArgInfo::getDirect(I32);
  if (Size == 64) {
    NSRN = std::min(NSRN + 1, 8u);
    return ArgInfo::getDirect(
        TB.getVectorType(I32, ElementCount::getFixed(2), llvm::Align(8)));
  }
  if (Size == 128) {
    NSRN = std::min(NSRN + 1, 8u);
    return ArgInfo::getDirect(
        TB.getVectorType(I32, ElementCount::getFixed(4), llvm::Align(16)));
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
}

bool AArch64TargetInfo::isIllegalVectorType(const Type *Ty) const {
  if (const auto *VT = dyn_cast<VectorType>(Ty)) {
    // Check whether VT is a fixed-length SVE vector. These types are
    // represented as scalable vectors in function args/return and must be
    // coerced from fixed vectors.
    if (VT->isFixedLengthSVEData() || VT->isFixedLengthSVEPredicate())
      return true;

    // Scalable SVE types are legal.
    if (VT->isScalable())
      return false;

    // Check whether VT is legal.
    assert(VT->getNumElements().isFixed() &&
           "expected fixed number of elements!");
    unsigned NumElements = VT->getNumElements().getKnownMinValue();
    uint64_t Size = VT->getABISizeInBits();
    // NumElements should be power of 2.
    if (!llvm::isPowerOf2_32(NumElements))
      return true;

    // arm64_32 has to be compatible with the ARM logic here, which allows huge
    // vectors for some reason.
    if (Opts.IsILP32 && Opts.IsMachO)
      return Size <= 32;

    return Size != 64 && (Size != 128 || NumElements == 1);
  }
  return false;
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

    uint64_t VecSize = VT->getABISizeInBits();
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
