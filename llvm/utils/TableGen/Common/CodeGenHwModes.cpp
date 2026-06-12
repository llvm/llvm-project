//===--- CodeGenHwModes.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Classes to parse and store HW mode information for instruction selection
//===----------------------------------------------------------------------===//

#include "CodeGenHwModes.h"
#include "SubtargetFeatureInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <set>

using namespace llvm;

#define DEBUG_TYPE "codegen-hwmodes"

StringRef CodeGenHwModes::DefaultModeName = "DefaultMode";

HwMode::HwMode(const Record *R) {
  Name = R->getName();
  Predicates = R->getValueAsListOfDefs("Predicates");
}

LLVM_DUMP_METHOD
void HwMode::dump() const {
  dbgs() << Name << ": ";
  ListSeparator LS;
  for (const Record *R : Predicates)
    dbgs() << LS << R->getName();
  dbgs() << '\n';
}

HwModeSelect::HwModeSelect(const Record *R, CodeGenHwModes &CGH) {
  std::vector<const Record *> Modes = R->getValueAsListOfDefs("Modes");
  std::vector<const Record *> Objects = R->getValueAsListOfDefs("Objects");
  for (auto [Mode, Object] : zip_equal(Modes, Objects)) {
    unsigned ModeId = CGH.getHwModeId(Mode);
    Items.emplace_back(ModeId, Object);
  }
}

LLVM_DUMP_METHOD
void HwModeSelect::dump() const {
  dbgs() << '{';
  for (const PairType &P : Items)
    dbgs() << " (" << P.first << ',' << P.second->getName() << ')';
  dbgs() << " }\n";
}
namespace llvm {
// Helper to represent predicates semantically.
struct HwModePredicates {
  // HwModePredicates represents a set of subtarget feature requirements in a
  // Conjunctive Normal Form (CNF) like structure:
  //
  // - FeaturesSet: Conjunction of literals (e.g., FeatureA && !FeatureB).
  //
  // - AnyOfFeatureSets: Conjunction of disjunctions
  //   (e.g., (FeatureC || !FeatureD) && (FeatureE || FeatureF)).
  //
  // This structure allows us to perform basic SAT-solving via unit propagation
  // to detect contradictions and prove mode compatibility.
  std::set<SubtargetFeatureLiteral> FeaturesSet;
  std::set<std::set<SubtargetFeatureLiteral>> AnyOfFeatureSets;

  HwModePredicates() = default;

  HwModePredicates(ArrayRef<const Record *> Preds) {
    getRequiredFeatures(FeaturesSet, AnyOfFeatureSets, Preds);
  }

  // DefaultMode is implicitly active only when *none* of the other target
  // HwModes are active.
  //
  // Mathematically: DefaultMode = !(Mode_1 || Mode_2 || ... || Mode_N)
  //                            = !Mode_1 && !Mode_2 && ... && !Mode_N
  //
  // If Mode_1 requires (FeatureA && FeatureB), then !Mode_1 requires
  // (!FeatureA || !FeatureB). We construct these implicit OR-sets for all
  // other modes and add them to DefaultMode's requirements.
  static HwModePredicates createForDefaultMode(const CodeGenHwModes &CGH) {
    HwModePredicates SP;
    for (unsigned M = 1; M < CGH.getNumModeIds(); ++M) {
      const HwMode &Mode = CGH.getMode(M);
      HwModePredicates ModePreds(Mode.Predicates);

      // Negate the features in FeaturesSet: !(F1 && F2) = !F1 || !F2
      std::set<SubtargetFeatureLiteral> NegatedFeatures;
      for (const auto &Op : ModePreds.FeaturesSet) {
        NegatedFeatures.insert({Op.Feature, !Op.IsNot});
      }

      if (ModePreds.AnyOfFeatureSets.empty()) {
        if (!NegatedFeatures.empty())
          SP.AnyOfFeatureSets.insert(std::move(NegatedFeatures));
      } else if (ModePreds.AnyOfFeatureSets.size() == 1) {
        // Negate a mode with a single OR-set:
        // !(F1 && F2 && (L1 || L2)) = !F1 || !F2 || (!L1 && !L2)
        // In CNF: (!F1 || !F2 || !L1) && (!F1 || !F2 || !L2)
        const auto &OrSet = *ModePreds.AnyOfFeatureSets.begin();
        for (const auto &Op : OrSet) {
          std::set<SubtargetFeatureLiteral> NewOrSet = NegatedFeatures;
          NewOrSet.insert({Op.Feature, !Op.IsNot});
          SP.AnyOfFeatureSets.insert(std::move(NewOrSet));
        }
      } else {
        LLVM_DEBUG(dbgs().indent(2)
                   << "Warning: HwMode '" << Mode.Name
                   << "' has multiple complex predicates. Ignoring for "
                      "DefaultMode negation.\n");
      }
    }
    return SP;
  }

  void add(const HwModePredicates &Other) {
    FeaturesSet.insert(Other.FeaturesSet.begin(), Other.FeaturesSet.end());
    AnyOfFeatureSets.insert(Other.AnyOfFeatureSets.begin(),
                            Other.AnyOfFeatureSets.end());
  }

  // Evaluates if the current set of predicates contains a contradiction.
  // Performs unit propagation: if we have a known feature F, we can simplify
  // OR-sets (A || B) containing F or !F.
  bool isSelfContradictory() {
    while (true) {
      // 1. Check for immediate contradictions (e.g. requiring both F and !F).
      for (const auto &Lit : FeaturesSet) {
        if (FeaturesSet.count({Lit.Feature, !Lit.IsNot}))
          return true;
      }

      // 2. Propagate known features to simplify OR-sets.
      std::set<std::set<SubtargetFeatureLiteral>> NewAnyOfs;
      bool MadeProgress = false;

      for (const auto &OrSet : AnyOfFeatureSets) {
        bool IsSatisfied = false;
        std::set<SubtargetFeatureLiteral> SimplifiedSet;

        for (const auto &Lit : OrSet) {
          // If a literal in the OR-set is already known to be true,
          // the entire OR-set is satisfied and can be discarded.
          if (FeaturesSet.count(Lit)) {
            IsSatisfied = true;
            break;
          }
          // If a literal's negation is known to be true (so this literal is
          // false), we can remove it from the OR-set.
          if (FeaturesSet.count({Lit.Feature, !Lit.IsNot})) {
            MadeProgress = true;
            continue;
          }
          SimplifiedSet.insert(Lit);
        }

        if (IsSatisfied) {
          MadeProgress = true;
          continue; // Discard satisfied set
        }

        if (SimplifiedSet.empty())
          return true; // All literals in this OR-set are false ->
                       // contradiction.

        if (SimplifiedSet.size() == 1) {
          // Unit clause: only one choice remains, so it must be true.
          FeaturesSet.insert(*SimplifiedSet.begin());
          MadeProgress = true;
        } else {
          NewAnyOfs.insert(std::move(SimplifiedSet));
        }
      }

      if (!MadeProgress)
        break;

      AnyOfFeatureSets = std::move(NewAnyOfs);
    }

    return false;
  }

  // Two predicate sets conflict if their union is self-contradictory.
  bool conflictsWith(const HwModePredicates &Other) const {
    HwModePredicates Combined(*this);
    Combined.add(Other);
    return Combined.isSelfContradictory();
  }
};

CodeGenHwModes::~CodeGenHwModes() = default;
} // namespace llvm

CodeGenHwModes::CodeGenHwModes(const RecordKeeper &RK) : Records(RK) {
  for (const Record *R : Records.getAllDerivedDefinitions("HwMode")) {
    // The default mode needs a definition in the .td sources for TableGen
    // to accept references to it. We need to ignore the definition here.
    if (R->getName() == DefaultModeName)
      continue;
    Modes.emplace_back(R);
    ModeIds.try_emplace(R, Modes.size());
  }

  assert(Modes.size() <= 32 && "number of HwModes exceeds maximum of 32");

  for (const Record *R : Records.getAllDerivedDefinitions("HwModeSelect")) {
    auto P = ModeSelects.emplace(R, HwModeSelect(R, *this));
    assert(P.second);
    (void)P;
  }

  // Populate the semantic predicates cache.
  PredicatesByMode.resize(getNumModeIds());
  PredicatesByMode[DefaultMode] = HwModePredicates::createForDefaultMode(*this);
  for (unsigned M = 1; M < getNumModeIds(); ++M) {
    PredicatesByMode[M] = HwModePredicates(getMode(M).Predicates);
  }
}

unsigned CodeGenHwModes::getHwModeId(const Record *R) const {
  if (R->getName() == DefaultModeName)
    return DefaultMode;
  auto F = ModeIds.find(R);
  assert(F != ModeIds.end() && "Unknown mode name");
  return F->second;
}

const HwModeSelect &CodeGenHwModes::getHwModeSelect(const Record *R) const {
  auto F = ModeSelects.find(R);
  assert(F != ModeSelects.end() && "Record is not a \"mode select\"");
  return F->second;
}

LLVM_DUMP_METHOD
void CodeGenHwModes::dump() const {
  dbgs() << "Modes: {\n";
  for (const HwMode &M : Modes) {
    dbgs() << "  ";
    M.dump();
  }
  dbgs() << "}\n";

  dbgs() << "ModeIds: {\n";
  for (const auto &P : ModeIds)
    dbgs() << "  " << P.first->getName() << " -> " << P.second << '\n';
  dbgs() << "}\n";

  dbgs() << "ModeSelects: {\n";
  for (const auto &P : ModeSelects) {
    dbgs() << "  " << P.first->getName() << " -> ";
    P.second.dump();
  }
  dbgs() << "}\n";
}

// Resolves a HwModeSelect record based on pattern predicates.
// Returns a unique resolved record if compatible, or nullptr if ambiguous.
const Record *
CodeGenHwModes::resolveModeSelect(const Record *SelectRec,
                                  ArrayRef<const Record *> PatPreds) const {
  if (!SelectRec->isSubClassOf("HwModeSelect"))
    return SelectRec;

  LLVM_DEBUG(dbgs() << "Trying to resolve HwModeSelect '"
                    << SelectRec->getName() << "'\n");

  const HwModeSelect &MS = getHwModeSelect(SelectRec);

  std::set<const Record *> ResolvedObjects;

  // Construct and parse the pattern predicates ONCE here
  HwModePredicates PatPredsSet(PatPreds);

  for (const auto &Item : MS.Items) {
    unsigned ModeId = Item.first;
    const Record *Obj = Item.second;

    // Use the pre-computed semantic predicates from cache.
    const HwModePredicates &ModePreds = PredicatesByMode[ModeId];

    if (!ModePreds.conflictsWith(PatPredsSet)) {
      LLVM_DEBUG(dbgs() << "  HwMode '" << getModeName(ModeId, true)
                        << "' is compatible -> " << Obj->getName() << "\n");
      ResolvedObjects.insert(Obj);
    } else {
      LLVM_DEBUG(dbgs() << "  HwMode '" << getModeName(ModeId, true)
                        << "' is incompatible due to semantic conflict\n");
    }
  }

  if (ResolvedObjects.size() == 1) {
    const Record *Resolved = *ResolvedObjects.begin();
    LLVM_DEBUG(dbgs() << "  Resolved to unique object: " << Resolved->getName()
                      << "\n");
    return Resolved;
  }

  if (ResolvedObjects.empty()) {
    LLVM_DEBUG(dbgs() << "  No active modes resolved for '"
                      << SelectRec->getName() << "'\n");
  } else {
    LLVM_DEBUG(dbgs().indent(2)
               << "Multiple active modes resolved to different objects for '"
               << SelectRec->getName() << "'\n");
  }
  return nullptr;
}
