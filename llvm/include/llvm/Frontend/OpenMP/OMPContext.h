//===- OpenMP/OMPContext.h ----- OpenMP context helper functions  - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides helper functions and classes to deal with OpenMP
/// contexts as used by `[begin/end] declare variant` and `metadirective`.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_CONTEXT_H
#define LLVM_OPENMP_CONTEXT_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace llvm {
namespace omp {

/// OpenMP Context related IDs and helpers
///
///{

/// IDs for all OpenMP context selector trait sets (construct/device/...).
enum class TraitSet {
#define OMP_TRAIT_SET(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

/// IDs for all OpenMP context selector trait (device={kind/isa...}/...).
enum class TraitSelector {
#define OMP_TRAIT_SELECTOR(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

/// IDs for all OpenMP context trait properties (host/gpu/bsc/llvm/...)
enum class TraitProperty {
#define OMP_TRAIT_PROPERTY(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

/// Parse \p Str and return the trait set it matches or TraitSet::invalid.
TraitSet getOpenMPContextTraitSetKind(StringRef Str);

/// Return the trait set for which \p Property is a property.
TraitSet getOpenMPContextTraitSetForProperty(TraitProperty Property);

/// Return a textual representation of the trait set \p Kind.
StringRef getOpenMPContextTraitSetName(TraitSet Kind);

/// Parse \p Str and return the trait set it matches or
/// TraitSelector::invalid.
TraitSelector getOpenMPContextTraitSelectorKind(StringRef Str);

/// Return the trait selector for which \p Property is a property.
TraitSelector getOpenMPContextTraitSelectorForProperty(TraitProperty Property);

/// Return a textual representation of the trait selector \p Kind.
StringRef getOpenMPContextTraitSelectorName(TraitSelector Kind);

/// Parse \p Str and return the trait set it matches or
/// TraitProperty::invalid.
TraitProperty getOpenMPContextTraitPropertyKind(TraitSet Set,
                                                TraitSelector Selector,
                                                StringRef Str);

/// Return the trait property for a singleton selector \p Selector.
TraitProperty getOpenMPContextTraitPropertyForSelector(TraitSelector Selector);

/// Return a textual representation of the trait property \p Kind.
StringRef getOpenMPContextTraitPropertyName(TraitProperty Kind);

/// Return a textual representation of the trait property \p Kind with selector
/// and set name included.
StringRef getOpenMPContextTraitPropertyFullName(TraitProperty Kind);
///}

/// Return true if \p Selector can be nested in \p Set. Also sets
/// \p AllowsTraitScore and \p RequiresProperty to true/false if the user can
/// specify a score for properties in \p Selector and if the \p Selector
/// requires at least one property.
bool isValidTraitSelectorForTraitSet(TraitSelector Selector, TraitSet Set,
                                     bool &AllowsTraitScore,
                                     bool &RequiresProperty);

/// Return true if \p Property can be nested in \p Selector and \p Set.
bool isValidTraitPropertyForTraitSetAndSelector(TraitProperty Property,
                                                TraitSelector Selector,
                                                TraitSet Set);

/// Variant match information describes the required traits and how they are
/// scored (via the ScoresMap). In addition, the required consturct nesting is
/// decribed as well.
struct VariantMatchInfo {
  /// Add the trait \p Property to the required trait set. If \p Score is not
  /// null, it recorded as well. If \p Property is in the `construct` set it
  /// is recorded in-order in the ConstructTraits as well.
  void addTrait(TraitProperty Property, APInt *Score = nullptr) {
    addTrait(getOpenMPContextTraitSetForProperty(Property), Property, Score);
  }
  /// Add the trait \p Property which is in set \p Set to the required trait
  /// set. If \p Score is not null, it recorded as well. If \p Set is the
  /// `construct` set it is recorded in-order in the ConstructTraits as well.
  void addTrait(TraitSet Set, TraitProperty Property, APInt *Score = nullptr) {
    if (Score)
      ScoreMap[Property] = *Score;
    RequiredTraits.insert(Property);
    if (Set == TraitSet::construct)
      ConstructTraits.push_back(Property);
  }

  SmallSet<TraitProperty, 8> RequiredTraits;
  SmallVector<TraitProperty, 8> ConstructTraits;
  SmallDenseMap<TraitProperty, APInt> ScoreMap;
};

/// The context for a source location is made up of active property traits,
/// e.g., device={kind(host)}, and constructs traits which describe the nesting
/// in OpenMP constructs at the location.
struct OMPContext {
  OMPContext(bool IsDeviceCompilation, Triple TargetTriple);

  void addTrait(TraitProperty Property) {
    addTrait(getOpenMPContextTraitSetForProperty(Property), Property);
  }
  void addTrait(TraitSet Set, TraitProperty Property) {
    ActiveTraits.insert(Property);
    if (Set == TraitSet::construct)
      ConstructTraits.push_back(Property);
  }

  SmallSet<TraitProperty, 8> ActiveTraits;
  SmallVector<TraitProperty, 8> ConstructTraits;
};

/// Return true if \p VMI is applicable in \p Ctx, that is, all traits required
/// by \p VMI are available in the OpenMP context \p Ctx.
bool isVariantApplicableInContext(const VariantMatchInfo &VMI,
                                  const OMPContext &Ctx);

/// Return the index (into \p VMIs) of the variant with the highest score
/// from the ones applicble in \p Ctx. See llvm::isVariantApplicableInContext.
int getBestVariantMatchForContext(const SmallVectorImpl<VariantMatchInfo> &VMIs,
                                  const OMPContext &Ctx);

} // namespace omp

template <> struct DenseMapInfo<omp::TraitProperty> {
  static inline omp::TraitProperty getEmptyKey() {
    return omp::TraitProperty(-1);
  }
  static inline omp::TraitProperty getTombstoneKey() {
    return omp::TraitProperty(-2);
  }
  static unsigned getHashValue(omp::TraitProperty val) {
    return std::hash<unsigned>{}(unsigned(val));
  }
  static bool isEqual(omp::TraitProperty LHS, omp::TraitProperty RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm
#endif // LLVM_OPENMP_CONTEXT_H
