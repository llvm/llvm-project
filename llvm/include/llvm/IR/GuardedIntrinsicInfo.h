//===- GuardedIntrinsicInfo.h - Guarded intrinsic table access -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GUARDEDINTRINSICINFO_H
#define LLVM_IR_GUARDEDINTRINSICINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <cassert>
#include <optional>

namespace llvm {

struct IntrinsicFeatureGuard {
  // Offset of this guard's first alternative in the feature clause table.
  unsigned FirstClause;
  // Number of alternative feature clauses.
  unsigned NumClauses;
  // Original feature expression, retained for diagnostics.
  const char *Expression;
  // Index of shared predicate and feature-bit check generated for this guard.
  unsigned Index;
};

enum class IntrinsicGuardCheckResult { Inactive, MissingFeatures, Available };

struct IntrinsicFeatureVariant {
  unsigned FirstType;
  // Zero means the guard applies to every valid overload.
  unsigned NumOverloadTypesToCheck;
  unsigned FirstGuard;
  unsigned NumGuards;
};

struct GuardedIntrinsicEntry {
  Intrinsic::ID ID;
  unsigned FirstVariant;
  unsigned NumVariants;
};

/// Provides access to the tables emitted by -gen-intrinsic-feature-checks.
template <typename Subtarget> class GuardedIntrinsicInfo {
  using GetFeatures = std::optional<FeatureBitset> (*)(
      const IntrinsicFeatureGuard &, const Subtarget &);

  ArrayRef<FeatureBitset> Clauses;
  ArrayRef<MVT::SimpleValueType> OverloadTypes;
  ArrayRef<unsigned> GuardIndices;
  ArrayRef<IntrinsicFeatureGuard> Guards;
  DenseMap<Intrinsic::ID, ArrayRef<IntrinsicFeatureVariant>> VariantsByID;
  GetFeatures GetFeaturesForGuard;

public:
  GuardedIntrinsicInfo(ArrayRef<FeatureBitset> Clauses,
                       ArrayRef<MVT::SimpleValueType> OverloadTypes,
                       ArrayRef<unsigned> GuardIndices,
                       ArrayRef<IntrinsicFeatureGuard> Guards,
                       ArrayRef<IntrinsicFeatureVariant> Variants,
                       ArrayRef<GuardedIntrinsicEntry> Entries,
                       GetFeatures GetFeaturesForGuard)
      : Clauses(Clauses), OverloadTypes(OverloadTypes),
        GuardIndices(GuardIndices), Guards(Guards),
        GetFeaturesForGuard(GetFeaturesForGuard) {
    VariantsByID.reserve(Entries.size());
    for (const GuardedIntrinsicEntry &Entry : Entries)
      VariantsByID.try_emplace(
          Entry.ID, Variants.slice(Entry.FirstVariant, Entry.NumVariants));
  }

  ArrayRef<IntrinsicFeatureVariant> getVariants(Intrinsic::ID ID) const {
    auto It = VariantsByID.find(ID);
    return It == VariantsByID.end() ? ArrayRef<IntrinsicFeatureVariant>{}
                                    : It->second;
  }

  const IntrinsicFeatureGuard &getGuard(const IntrinsicFeatureVariant &Variant,
                                        unsigned Index) const {
    assert(Index < Variant.NumGuards &&
           "invalid intrinsic feature guard index");
    return Guards[GuardIndices[Variant.FirstGuard + Index]];
  }

  ArrayRef<FeatureBitset>
  getAlternatives(const IntrinsicFeatureGuard &Guard) const {
    return Clauses.slice(Guard.FirstClause, Guard.NumClauses);
  }

  /// Check a guard using the target's predicate and feature-bit source.
  /// The feature getter returns nullopt when the predicate is inactive.
  IntrinsicGuardCheckResult checkGuard(const IntrinsicFeatureGuard &Guard,
                                       const Subtarget &ST) const {
    std::optional<FeatureBitset> Available = GetFeaturesForGuard(Guard, ST);
    if (!Available)
      return IntrinsicGuardCheckResult::Inactive;
    for (const FeatureBitset &Required : getAlternatives(Guard))
      if ((*Available & Required) == Required)
        return IntrinsicGuardCheckResult::Available;
    return IntrinsicGuardCheckResult::MissingFeatures;
  }

  /// The generated predicate decides whether each guard is active. Ignore
  /// inactive guards, but require every active guard to have its features.
  /// At least one guard must be active for the variant to be available.
  bool isVariantAvailable(const IntrinsicFeatureVariant &Variant,
                          const Subtarget &ST) const {
    bool HasActiveGuard = false;
    for (unsigned I = 0; I < Variant.NumGuards; ++I)
      switch (checkGuard(getGuard(Variant, I), ST)) {
      case IntrinsicGuardCheckResult::Inactive:
        break;
      case IntrinsicGuardCheckResult::MissingFeatures:
        return false;
      case IntrinsicGuardCheckResult::Available:
        HasActiveGuard = true;
        break;
      }
    return HasActiveGuard;
  }

  FunctionType *getFunctionType(LLVMContext &Context, Intrinsic::ID IID,
                                const IntrinsicFeatureVariant &Variant) const {
    assert((Variant.NumOverloadTypesToCheck > 0 ||
            !Intrinsic::isOverloaded(IID)) &&
           "No concrete type for variant");
    SmallVector<Type *, 4> Types;
    for (MVT::SimpleValueType VT : OverloadTypes.slice(
             Variant.FirstType, Variant.NumOverloadTypesToCheck))
      Types.push_back(VT == MVT::iPTR ? PointerType::get(Context, 0)
                                      : EVT(VT).getTypeForEVT(Context));
    return Intrinsic::getType(Context, IID, Types);
  }
};

} // namespace llvm

#endif // LLVM_IR_GUARDEDINTRINSICINFO_H
