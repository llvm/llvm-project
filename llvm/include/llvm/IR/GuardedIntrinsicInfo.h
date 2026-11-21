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

namespace llvm {

struct IntrinsicFeatureGuard {
  unsigned FirstClause;
  unsigned NumClauses;
  const char *Expression;
};

struct IntrinsicFeatureVariant {
  unsigned FirstType;
  unsigned NumTypes;
  unsigned FirstGuard;
  unsigned NumGuards;
  bool AppliesToAllTypes;
};

struct GuardedIntrinsicEntry {
  Intrinsic::ID ID;
  unsigned FirstVariant;
  unsigned NumVariants;
};

/// Provides access to the tables emitted by -gen-intrinsic-feature-checks.
class GuardedIntrinsicInfo {
  ArrayRef<FeatureBitset> Clauses;
  ArrayRef<MVT::SimpleValueType> OverloadTypes;
  ArrayRef<unsigned> GuardIndices;
  ArrayRef<IntrinsicFeatureGuard> Guards;
  DenseMap<Intrinsic::ID, ArrayRef<IntrinsicFeatureVariant>> VariantsByID;

public:
  GuardedIntrinsicInfo(ArrayRef<FeatureBitset> Clauses,
                       ArrayRef<MVT::SimpleValueType> OverloadTypes,
                       ArrayRef<unsigned> GuardIndices,
                       ArrayRef<IntrinsicFeatureGuard> Guards,
                       ArrayRef<IntrinsicFeatureVariant> Variants,
                       ArrayRef<GuardedIntrinsicEntry> Entries)
      : Clauses(Clauses), OverloadTypes(OverloadTypes),
        GuardIndices(GuardIndices), Guards(Guards) {
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

  FunctionType *getFunctionType(LLVMContext &Context, Intrinsic::ID ID,
                                const IntrinsicFeatureVariant &Variant) const {
    SmallVector<Type *, 4> Types;
    for (MVT::SimpleValueType VT :
         OverloadTypes.slice(Variant.FirstType, Variant.NumTypes))
      Types.push_back(VT == MVT::iPTR ? PointerType::get(Context, 0)
                                      : EVT(VT).getTypeForEVT(Context));
    return Intrinsic::getType(Context, ID, Types);
  }
};

} // namespace llvm

#endif // LLVM_IR_GUARDEDINTRINSICINFO_H
