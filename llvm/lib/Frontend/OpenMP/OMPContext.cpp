//===- OMPContext.cpp ------ Collection of helpers for OpenMP contexts ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements helper functions and classes to deal with OpenMP
/// contexts as used by `[begin/end] declare variant` and `metadirective`.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPContext.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;

OMPContext::OMPContext(bool IsDeviceCompilation, Triple TargetTriple,
                       Triple TargetOffloadTriple, int DeviceNum) {
  // Add the appropriate target device kind trait based on the target triple
  if (!TargetOffloadTriple.getTriple().empty() && DeviceNum > -1) {
    // If target triple is present, then target device is not a host
    ActiveTraits.set(unsigned(TraitProperty::target_device_kind_nohost));
    switch (TargetOffloadTriple.getArch()) {
    case Triple::arm:
    case Triple::armeb:
    case Triple::aarch64:
    case Triple::aarch64_be:
    case Triple::aarch64_32:
    case Triple::mips:
    case Triple::mipsel:
    case Triple::mips64:
    case Triple::mips64el:
    case Triple::ppc:
    case Triple::ppcle:
    case Triple::ppc64:
    case Triple::ppc64le:
    case Triple::systemz:
    case Triple::x86:
    case Triple::x86_64:
      ActiveTraits.set(unsigned(TraitProperty::target_device_kind_cpu));
      break;
    case Triple::amdgpu:
    case Triple::nvptx:
    case Triple::nvptx64:
    case Triple::spirv64:
      ActiveTraits.set(unsigned(TraitProperty::target_device_kind_gpu));
      break;
    default:
      break;
    }
    // Add the appropriate device architecture trait based on the triple.
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  if (TraitSelector::TraitSelectorEnum == TraitSelector::target_device_arch) { \
    if (TargetOffloadTriple.getArch() == Triple::parseArch(Str))               \
      ActiveTraits.set(unsigned(TraitProperty::Enum));                         \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  } else {
    // Add the appropriate device kind trait based on the triple and the
    // IsDeviceCompilation flag.
    ActiveTraits.set(unsigned(IsDeviceCompilation
                                  ? TraitProperty::device_kind_nohost
                                  : TraitProperty::device_kind_host));
    ActiveTraits.set(unsigned(TraitProperty::target_device_kind_host));
    switch (TargetTriple.getArch()) {
    case Triple::arm:
    case Triple::armeb:
    case Triple::aarch64:
    case Triple::aarch64_be:
    case Triple::aarch64_32:
    case Triple::mips:
    case Triple::mipsel:
    case Triple::mips64:
    case Triple::mips64el:
    case Triple::ppc:
    case Triple::ppcle:
    case Triple::ppc64:
    case Triple::ppc64le:
    case Triple::systemz:
    case Triple::x86:
    case Triple::x86_64:
      ActiveTraits.set(unsigned(TraitProperty::device_kind_cpu));
      ActiveTraits.set(unsigned(TraitProperty::target_device_kind_cpu));
      break;
    case Triple::amdgpu:
    case Triple::nvptx:
    case Triple::nvptx64:
    case Triple::spirv64:
      ActiveTraits.set(unsigned(TraitProperty::device_kind_gpu));
      ActiveTraits.set(unsigned(TraitProperty::target_device_kind_gpu));
      break;
    default:
      break;
    }

    // Add the appropriate device architecture trait based on the triple.
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  if (TraitSelector::TraitSelectorEnum == TraitSelector::device_arch ||        \
      TraitSelector::TraitSelectorEnum == TraitSelector::target_device_arch) { \
    if (TargetTriple.getArch() == Triple::parseArch(Str))                      \
      ActiveTraits.set(unsigned(TraitProperty::Enum));                         \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"

    // TODO: What exactly do we want to see as device ISA trait?
    //       The discussion on the list did not seem to have come to an agreed
    //       upon solution.

    // LLVM is the "OpenMP vendor" but we could also interpret vendor as the
    // target vendor.
    ActiveTraits.set(unsigned(TraitProperty::implementation_vendor_llvm));

    // The user condition true is accepted but not false.
    ActiveTraits.set(unsigned(TraitProperty::user_condition_true));

    // This is for sure some device.
    ActiveTraits.set(unsigned(TraitProperty::device_kind_any));

    LLVM_DEBUG({
      dbgs() << "[" << DEBUG_TYPE
             << "] New OpenMP context with the following properties:\n";
      for (unsigned Bit : ActiveTraits.set_bits()) {
        TraitProperty Property = TraitProperty(Bit);
        dbgs() << "\t " << getOpenMPContextTraitPropertyFullName(Property)
               << "\n";
      }
    });
  }
}

/// Return true if \p C0 is an ordered subsequence of \p C1.
template <typename T>
static bool isOrderedSubset(ArrayRef<T> C0, ArrayRef<T> C1) {
  if (C0.size() > C1.size())
    return false;
  auto It0 = C0.begin(), End0 = C0.end();
  auto It1 = C1.begin(), End1 = C1.end();
  while (It0 != End0) {
    if (It1 == End1)
      return false;
    if (*It0 == *It1) {
      ++It0;
      ++It1;
      continue;
    }
    ++It1;
  }
  return true;
}

static bool isStrictSubset(const VariantMatchInfo &VMI0,
                           const VariantMatchInfo &VMI1) {
  // kind(any) is equivalent to omitting the kind selector.
  BitVector Traits0 = VMI0.RequiredTraits, Traits1 = VMI1.RequiredTraits;
  for (TraitProperty Property : {TraitProperty::device_kind_any,
                                 TraitProperty::target_device_kind_any}) {
    Traits0.reset(unsigned(Property));
    Traits1.reset(unsigned(Property));
  }
  size_t TraitCount0 = Traits0.count() + VMI0.UnknownTraits.size();
  size_t TraitCount1 = Traits1.count() + VMI1.UnknownTraits.size();
  if (TraitCount0 > TraitCount1)
    return false;
  for (unsigned Bit : Traits0.set_bits())
    if (!Traits1.test(Bit))
      return false;
  for (const auto &Trait : VMI0.UnknownTraits)
    if (!llvm::is_contained(VMI1.UnknownTraits, Trait))
      return false;
  for (const auto &Trait : VMI0.ISATraits)
    if (!llvm::is_contained(VMI1.ISATraits, Trait))
      return false;
  bool HasAdditionalISATrait =
      llvm::any_of(VMI1.ISATraits, [&](const auto &Trait) {
        return !llvm::is_contained(VMI0.ISATraits, Trait);
      });
  if (!VMI0.UserCondition.empty() && VMI0.UserCondition != VMI1.UserCondition)
    return false;
  if (!isOrderedSubset<TraitProperty>(VMI0.ConstructTraits,
                                      VMI1.ConstructTraits))
    return false;
  // RequiredTraits is a bit vector, so repeated construct properties only
  // make the ordered vector strict.
  return TraitCount0 < TraitCount1 || HasAdditionalISATrait ||
         VMI0.ConstructTraits.size() < VMI1.ConstructTraits.size();
}

static int
isVariantApplicableInContextHelper(const VariantMatchInfo &VMI,
                                   const OMPContext &Ctx,
                                   SmallVectorImpl<unsigned> *ConstructMatches,
                                   bool DeviceOrImplementationSetOnly) {

  // The match kind determines if we need to match all traits, any of the
  // traits, or none of the traits for it to be an applicable context.
  enum MatchKind { MK_ALL, MK_ANY, MK_NONE };

  MatchKind MK = MK_ALL;
  // Determine the match kind the user wants, "all" is the default and provided
  // to the user only for completeness.
  if (VMI.RequiredTraits.test(
          unsigned(TraitProperty::implementation_extension_match_any)))
    MK = MK_ANY;
  if (VMI.RequiredTraits.test(
          unsigned(TraitProperty::implementation_extension_match_none)))
    MK = MK_NONE;

  bool AnyTraitMatched = false;

  // Apply the match kind selected by implementation={extension(...)} to
  // each property. Continue after match_any succeeds to record all construct
  // match positions needed for scoring.
  auto HandleTrait = [MK, &AnyTraitMatched](TraitProperty Property,
                                            bool WasFound) -> bool {
    AnyTraitMatched |= WasFound;
    if (MK == MK_ANY)
      return true;

    // In "all" or "none" mode we accept a matching or non-matching property
    // respectively and move on. We are not done yet!
    if ((WasFound && MK == MK_ALL) || (!WasFound && MK == MK_NONE))
      return true;

    // We missed a property, provide some debug output and indicate failure.
    LLVM_DEBUG({
      if (MK == MK_ALL)
        dbgs() << "[" << DEBUG_TYPE << "] Property "
               << getOpenMPContextTraitPropertyName(Property, "")
               << " was not in the OpenMP context but match kind is all.\n";
      if (MK == MK_NONE)
        dbgs() << "[" << DEBUG_TYPE << "] Property "
               << getOpenMPContextTraitPropertyName(Property, "")
               << " was in the OpenMP context but match kind is none.\n";
    });
    return false;
  };

  if (!VMI.UnknownTraits.empty() && !HandleTrait(TraitProperty::invalid, false))
    return false;

  for (unsigned Bit : VMI.RequiredTraits.set_bits()) {
    TraitProperty Property = TraitProperty(Bit);
    if (DeviceOrImplementationSetOnly &&
        getOpenMPContextTraitSetForProperty(Property) != TraitSet::device &&
        getOpenMPContextTraitSetForProperty(Property) !=
            TraitSet::implementation)
      continue;

    // So far all extensions are handled elsewhere, we skip them here as they
    // are not part of the OpenMP context.
    if (getOpenMPContextTraitSelectorForProperty(Property) ==
        TraitSelector::implementation_extension)
      continue;

    bool IsActiveTrait = Ctx.ActiveTraits.test(unsigned(Property));

    // We overwrite the isa trait as it is actually up to the OMPContext hook to
    // check the raw string(s).
    if (Property == TraitProperty::device_isa___ANY ||
        Property == TraitProperty::target_device_isa___ANY)
      IsActiveTrait = llvm::all_of(VMI.ISATraits, [&](const auto &Trait) {
        return Trait.Property != Property || Ctx.matchesISATrait(Trait.Name);
      });

    if (!HandleTrait(Property, IsActiveTrait))
      return false;
  }

  if (!DeviceOrImplementationSetOnly) {
    // Scan the construct sequence in order, recording matching context
    // positions for scoring.
    unsigned ConstructIdx = 0, NoConstructTraits = Ctx.ConstructTraits.size();
    for (TraitProperty Property : VMI.ConstructTraits) {
      assert(getOpenMPContextTraitSetForProperty(Property) ==
                 TraitSet::construct &&
             "Variant context is ill-formed!");

      // Verify the nesting. A failed match in match_any or match_none must not
      // consume the remaining context, since a later selector property can
      // still match.
      unsigned SearchStart = ConstructIdx;
      bool FoundInOrder = false;
      while (!FoundInOrder && ConstructIdx != NoConstructTraits)
        FoundInOrder = (Ctx.ConstructTraits[ConstructIdx++] == Property);
      if (!FoundInOrder && MK != MK_ALL)
        ConstructIdx = SearchStart;
      if (ConstructMatches && FoundInOrder)
        ConstructMatches->push_back(ConstructIdx - 1);

      if (!HandleTrait(Property, FoundInOrder)) {
        LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE << "] Construct property "
                          << getOpenMPContextTraitPropertyName(Property, "")
                          << " was not nested properly.\n");
        return false;
      }

      // TODO: Verify SIMD
    }

    // A complete ordered match can have several embeddings in the context.
    // Match backwards to choose the highest-valued one for scoring. Keep the
    // forward scan's partial matches for the match_any extension.
    if (ConstructMatches &&
        ConstructMatches->size() == VMI.ConstructTraits.size()) {
      ConstructIdx = NoConstructTraits;
      for (unsigned I = VMI.ConstructTraits.size(); I > 0; --I) {
        TraitProperty Property = VMI.ConstructTraits[I - 1];
        while (ConstructIdx > 0 &&
               Ctx.ConstructTraits[ConstructIdx - 1] != Property)
          --ConstructIdx;
        assert(ConstructIdx > 0 && "Previously matched construct not found!");
        (*ConstructMatches)[I - 1] = --ConstructIdx;
      }
    }

    if (MK == MK_ALL)
      assert(isOrderedSubset<TraitProperty>(VMI.ConstructTraits,
                                            Ctx.ConstructTraits) &&
             "Broken invariant!");
  }

  if (MK == MK_ANY && !AnyTraitMatched) {
    LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE
                      << "] None of the properties was in the OpenMP context "
                         "but match kind is any.\n");
    return false;
  }

  return true;
}

bool llvm::omp::isVariantApplicableInContext(
    const VariantMatchInfo &VMI, const OMPContext &Ctx,
    bool DeviceOrImplementationSetOnly) {
  return isVariantApplicableInContextHelper(
      VMI, Ctx, /* ConstructMatches */ nullptr, DeviceOrImplementationSetOnly);
}

static APInt getVariantMatchScore(const VariantMatchInfo &VMI,
                                  const OMPContext &Ctx,
                                  SmallVectorImpl<unsigned> &ConstructMatches) {
  APInt Score(1, 1);

  // A sum of valid scores can exceed the width of any individual score.
  // Retain all active bits and allow one more bit for each addition's carry.
  auto AddScore = [&](const APInt &Value) {
    unsigned Width = std::max(Score.getActiveBits(), Value.getActiveBits()) + 1;
    Score = Score.zextOrTrunc(Width);
    Score += Value.zextOrTrunc(Width);
  };
  auto AddPowerOfTwo = [&](unsigned Exponent) {
    AddScore(APInt::getOneBitSet(Exponent + 1, Exponent));
  };

  unsigned NoConstructTraits = Ctx.ConstructTraits.size();
  SmallDenseSet<TraitSelector, 8> ScoredSelectors;
  auto AddSelectorScore = [&](TraitSelector Selector, const APInt *UserScore) {
    // Scores belong to selectors, not to individual properties. Unknown
    // properties retain these scores even though they never match.
    if (!ScoredSelectors.insert(Selector).second)
      return;
    if (UserScore) {
      AddScore(*UserScore);
      return;
    }
    switch (Selector) {
    case TraitSelector::device_kind:
    case TraitSelector::target_device_kind:
      AddPowerOfTwo(NoConstructTraits);
      break;
    case TraitSelector::device_arch:
    case TraitSelector::target_device_arch:
      AddPowerOfTwo(NoConstructTraits + 1);
      break;
    case TraitSelector::device_isa:
    case TraitSelector::target_device_isa:
      AddPowerOfTwo(NoConstructTraits + 2);
      break;
    default:
      break;
    }
  };
  for (unsigned Bit : VMI.RequiredTraits.set_bits()) {
    TraitProperty Property = TraitProperty(Bit);
    // Construct scores use ordered positions below. kind(any) is treated as
    // if no kind selector were specified.
    if (getOpenMPContextTraitSetForProperty(Property) == TraitSet::construct ||
        Property == TraitProperty::device_kind_any ||
        Property == TraitProperty::target_device_kind_any)
      continue;
    auto It = VMI.ScoreMap.find(Property);
    AddSelectorScore(getOpenMPContextTraitSelectorForProperty(Property),
                     It == VMI.ScoreMap.end() ? nullptr : &It->second);
  }
  for (const auto &Trait : VMI.UnknownTraits)
    AddSelectorScore(Trait.Selector, Trait.Score ? &*Trait.Score : nullptr);

  assert(VMI.ConstructTraits.size() >= ConstructMatches.size() &&
         "Mismatch in the construct traits!");
  for (unsigned Match : ConstructMatches) {
    // ConstructMatches is the position p - 1 and we need 2^(p-1).
    AddPowerOfTwo(Match);
  }

  LLVM_DEBUG({
    dbgs() << "[" << DEBUG_TYPE << "] Variant has a score of ";
    Score.print(dbgs(), /*isSigned=*/false);
    dbgs() << "\n";
  });
  return Score;
}

int llvm::omp::getBestVariantMatchForContext(
    const SmallVectorImpl<VariantMatchInfo> &VMIs, const OMPContext &Ctx) {
  SmallVector<std::optional<APInt>, 4> Scores(VMIs.size());
  for (unsigned u = 0, e = VMIs.size(); u < e; ++u) {
    const VariantMatchInfo &VMI = VMIs[u];

    SmallVector<unsigned, 8> ConstructMatches;
    // Inapplicable variants do not participate in scoring or subset checks.
    if (!isVariantApplicableInContextHelper(
            VMI, Ctx, &ConstructMatches,
            /* DeviceOrImplementationSetOnly */ false))
      continue;
    Scores[u] = getVariantMatchScore(VMI, Ctx, ConstructMatches);
  }

  // A compatible selector that is a strict subset of another compatible
  // selector has score zero, irrespective of their scores before this step.
  // Apply this rule globally before choosing the maximum.
  for (unsigned u = 0, e = VMIs.size(); u < e; ++u) {
    if (!Scores[u])
      continue;
    for (unsigned v = 0; v < e; ++v) {
      if (u != v && Scores[v] && isStrictSubset(VMIs[u], VMIs[v])) {
        Scores[u] = APInt(1, 0);
        break;
      }
    }
  }

  APInt BestScore(1, 0);
  int BestVMIIdx = -1;
  for (unsigned u = 0, e = VMIs.size(); u < e; ++u) {
    if (!Scores[u])
      continue;
    const APInt &Score = *Scores[u];
    unsigned Width = std::max(Score.getBitWidth(), BestScore.getBitWidth());
    if (BestVMIIdx >= 0 &&
        !BestScore.zextOrTrunc(Width).ult(Score.zextOrTrunc(Width)))
      continue;
    BestVMIIdx = u;
    BestScore = Score;
  }

  return BestVMIIdx;
}

TraitSet llvm::omp::getOpenMPContextTraitSetKind(StringRef S) {
  return StringSwitch<TraitSet>(S)
#define OMP_TRAIT_SET(Enum, Str) .Case(Str, TraitSet::Enum)
#include "llvm/Frontend/OpenMP/OMPKinds.def"
      .Default(TraitSet::invalid);
}

TraitSet
llvm::omp::getOpenMPContextTraitSetForSelector(TraitSelector Selector) {
  switch (Selector) {
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, ReqProp)                   \
  case TraitSelector::Enum:                                                    \
    return TraitSet::TraitSetEnum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait selector!");
}
TraitSet
llvm::omp::getOpenMPContextTraitSetForProperty(TraitProperty Property) {
  switch (Property) {
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  case TraitProperty::Enum:                                                    \
    return TraitSet::TraitSetEnum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait set!");
}
StringRef llvm::omp::getOpenMPContextTraitSetName(TraitSet Kind) {
  switch (Kind) {
#define OMP_TRAIT_SET(Enum, Str)                                               \
  case TraitSet::Enum:                                                         \
    return Str;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait set!");
}

TraitSelector llvm::omp::getOpenMPContextTraitSelectorKind(StringRef S,
                                                           TraitSet Set) {
  if (Set == TraitSet::target_device && S == "kind")
    return TraitSelector::target_device_kind;
  if (Set == TraitSet::target_device && S == "arch")
    return TraitSelector::target_device_arch;
  if (Set == TraitSet::target_device && S == "isa")
    return TraitSelector::target_device_isa;
  return StringSwitch<TraitSelector>(S)
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, ReqProp)                   \
  .Case(Str, TraitSelector::Enum)
#include "llvm/Frontend/OpenMP/OMPKinds.def"
      .Default(TraitSelector::invalid);
}
TraitSelector
llvm::omp::getOpenMPContextTraitSelectorForProperty(TraitProperty Property) {
  switch (Property) {
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  case TraitProperty::Enum:                                                    \
    return TraitSelector::TraitSelectorEnum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait set!");
}
StringRef llvm::omp::getOpenMPContextTraitSelectorName(TraitSelector Kind) {
  switch (Kind) {
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, ReqProp)                   \
  case TraitSelector::Enum:                                                    \
    return Str;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait selector!");
}

TraitProperty llvm::omp::getOpenMPContextTraitPropertyKind(
    TraitSet Set, TraitSelector Selector, StringRef S) {
  // Special handling for `device={isa(...)}` as we accept anything here. It is
  // up to the target to decide if the feature is available.
  if (Set == TraitSet::device && Selector == TraitSelector::device_isa)
    return TraitProperty::device_isa___ANY;
  if (Set == TraitSet::target_device &&
      Selector == TraitSelector::target_device_isa)
    return TraitProperty::target_device_isa___ANY;
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  if (Set == TraitSet::TraitSetEnum && Str == S)                               \
    return TraitProperty::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  return TraitProperty::invalid;
}
TraitProperty
llvm::omp::getOpenMPContextTraitPropertyForSelector(TraitSelector Selector) {
  return StringSwitch<TraitProperty>(
             getOpenMPContextTraitSelectorName(Selector))
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  .Case(Str, Selector == TraitSelector::TraitSelectorEnum                      \
                 ? TraitProperty::Enum                                         \
                 : TraitProperty::invalid)
#include "llvm/Frontend/OpenMP/OMPKinds.def"
      .Default(TraitProperty::invalid);
}
StringRef llvm::omp::getOpenMPContextTraitPropertyName(TraitProperty Kind,
                                                       StringRef RawString) {
  if (Kind == TraitProperty::device_isa___ANY)
    return RawString;
  if (Kind == TraitProperty::target_device_isa___ANY)
    return RawString;
  switch (Kind) {
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  case TraitProperty::Enum:                                                    \
    return Str;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait property!");
}
StringRef llvm::omp::getOpenMPContextTraitPropertyFullName(TraitProperty Kind) {
  switch (Kind) {
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  case TraitProperty::Enum:                                                    \
    return "(" #TraitSetEnum "," #TraitSelectorEnum "," Str ")";
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait property!");
}

bool llvm::omp::isValidTraitSelectorForTraitSet(TraitSelector Selector,
                                                TraitSet Set,
                                                bool &AllowsTraitScore,
                                                bool &RequiresProperty) {
  AllowsTraitScore = Set != TraitSet::construct && Set != TraitSet::device &&
                     Set != TraitSet::target_device;
  switch (Selector) {
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, ReqProp)                   \
  case TraitSelector::Enum:                                                    \
    RequiresProperty = ReqProp;                                                \
    return Set == TraitSet::TraitSetEnum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait selector!");
}

bool llvm::omp::isValidTraitPropertyForTraitSetAndSelector(
    TraitProperty Property, TraitSelector Selector, TraitSet Set) {
  switch (Property) {
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  case TraitProperty::Enum:                                                    \
    return Set == TraitSet::TraitSetEnum &&                                    \
           Selector == TraitSelector::TraitSelectorEnum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Unknown trait property!");
}

std::string llvm::omp::listOpenMPContextTraitSets() {
  std::string S;
#define OMP_TRAIT_SET(Enum, Str)                                               \
  if (StringRef(Str) != "invalid")                                             \
    S.append("'").append(Str).append("'").append(" ");
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  S.pop_back();
  return S;
}

std::string llvm::omp::listOpenMPContextTraitSelectors(TraitSet Set) {
  std::string S;
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, ReqProp)                   \
  if (TraitSet::TraitSetEnum == Set && StringRef(Str) != "Invalid")            \
    S.append("'").append(Str).append("'").append(" ");
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  S.pop_back();
  return S;
}

std::string
llvm::omp::listOpenMPContextTraitProperties(TraitSet Set,
                                            TraitSelector Selector) {
  std::string S;
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  if (TraitSet::TraitSetEnum == Set &&                                         \
      TraitSelector::TraitSelectorEnum == Selector &&                          \
      StringRef(Str) != "invalid")                                             \
    S.append("'").append(Str).append("'").append(" ");
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  if (S.empty())
    return "<none>";
  S.pop_back();
  return S;
}
