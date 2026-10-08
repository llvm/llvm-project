//===- SubtargetFeatureInfo.cpp - Helpers for subtarget features ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubtargetFeatureInfo.h"
#include "Basic/PredicateExpanderDag.h"
#include "Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <iterator>
#include <vector>

using namespace llvm;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SubtargetFeatureInfo::dump() const {
  errs() << getEnumName() << " " << Index << "\n" << *TheDef;
}
#endif

SubtargetFeaturesInfoVec
SubtargetFeatureInfo::getAll(const RecordKeeper &Records) {
  SubtargetFeaturesInfoVec SubtargetFeatures;
  for (const Record *Pred : Records.getAllDerivedDefinitions("Predicate")) {
    // Ignore predicates that are not intended for the assembler.
    //
    // The "AssemblerMatcherPredicate" string should be promoted to an argument
    // if we re-use the machinery for non-assembler purposes in future.
    if (!Pred->getValueAsBit("AssemblerMatcherPredicate"))
      continue;

    if (Pred->getName().empty())
      PrintFatalError(Pred->getLoc(), "Predicate has no name!");

    // Ignore always true predicates.
    if (Pred->getValueAsString("CondString").empty())
      continue;

    SubtargetFeatures.emplace_back(
        Pred, SubtargetFeatureInfo(Pred, SubtargetFeatures.size()));
  }
  return SubtargetFeatures;
}

void SubtargetFeatureInfo::emitSubtargetFeatureBitEnumeration(
    const SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS,
    const std::map<std::string, unsigned> *HwModes) {
  OS << "// Bits for subtarget features that participate in "
     << "instruction matching.\n";
  unsigned Size = SubtargetFeatures.size();
  if (HwModes)
    Size += HwModes->size();

  OS << "enum SubtargetFeatureBits : " << getMinimalTypeForRange(Size)
     << " {\n";
  for (const auto &SF : SubtargetFeatures) {
    const SubtargetFeatureInfo &SFI = SF.second;
    OS << "  " << SFI.getEnumBitName() << " = " << SFI.Index << ",\n";
  }

  if (HwModes) {
    unsigned Offset = SubtargetFeatures.size();
    for (const auto &M : *HwModes) {
      OS << "  Feature_HwMode" << M.second << "Bit = " << (M.second + Offset)
         << ",\n";
    }
  }

  OS << "};\n\n";
}

void SubtargetFeatureInfo::emitNameTable(
    SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS) {
  // Need to sort the name table so that lookup by the log of the enum value
  // gives the proper name. More specifically, for a feature of value 1<<n,
  // SubtargetFeatureNames[n] should be the name of the feature.
  uint64_t IndexUB = 0;
  for (const auto &SF : SubtargetFeatures)
    if (IndexUB <= SF.second.Index)
      IndexUB = SF.second.Index + 1;

  std::vector<std::string> Names;
  if (IndexUB > 0)
    Names.resize(IndexUB);
  for (const auto &SF : SubtargetFeatures)
    Names[SF.second.Index] = SF.second.getEnumName();

  OS << "static const char *SubtargetFeatureNames[] = {\n";
  for (uint64_t I = 0; I < IndexUB; ++I)
    OS << "  \"" << Names[I] << "\",\n";

  // A small number of targets have no predicates. Null terminate the array to
  // avoid a zero-length array.
  OS << "  nullptr\n"
     << "};\n\n";
}

void SubtargetFeatureInfo::emitComputeAvailableFeatures(
    StringRef TargetName, StringRef ClassName, StringRef FuncName,
    const SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS,
    StringRef ExtraParams, const std::map<std::string, unsigned> *HwModes) {
  OS << "PredicateBitset " << ClassName << "::\n"
     << FuncName << "(const " << TargetName << "Subtarget *Subtarget";
  if (!ExtraParams.empty())
    OS << ", " << ExtraParams;
  OS << ") const {\n";
  OS << "  PredicateBitset Features{};\n";
  for (const auto &SF : SubtargetFeatures) {
    const SubtargetFeatureInfo &SFI = SF.second;
    StringRef CondStr = SFI.TheDef->getValueAsString("CondString");
    assert(!CondStr.empty() && "true predicate should have been filtered");

    OS << "  if (" << CondStr << ")\n";
    OS << "    Features.set(" << SFI.getEnumBitName() << ");\n";
  }

  if (HwModes) {
    for (const auto &M : *HwModes) {
      OS << "  if (" << M.first << ")\n";
      OS << "    Features.set(Feature_HwMode" << M.second << "Bit);\n";
    }
  }

  OS << "  return Features;\n";
  OS << "}\n\n";
}

// Emit a feature-bit test leaf: FB[<TargetName>::<feature>]. Returns true on
// error (the leaf is not a SubtargetFeature).
static bool emitFeatureBitLeaf(StringRef TargetName, const Init &Val,
                               raw_ostream &OS) {
  const auto *D = dyn_cast<DefInit>(&Val);
  if (!D || !D->getDef()->isSubClassOf("SubtargetFeature"))
    return true;
  OS << "FB[" << TargetName << "::" << D->getAsString() << ']';
  return false;
}

// If ParenIfBinOp is true, print a surrounding () if Val uses && or ||.
// Structural errors in the dag are diagnosed against \p Owner by the walker;
// a leaf that is not a SubtargetFeature is reported here via the true return.
static bool emitFeaturesAux(const Record *Owner, StringRef TargetName,
                            const Init &Val, bool ParenIfBinOp,
                            raw_ostream &OS) {
  return emitPredicateDag(Owner, Val, ParenIfBinOp, OS,
                          [&](const Init &Leaf, raw_ostream &OS) {
                            return emitFeatureBitLeaf(TargetName, Leaf, OS);
                          });
}

void SubtargetFeatureInfo::emitPredicateCheck(
    raw_ostream &OS, ArrayRef<const Record *> Predicates) {
  ListSeparator LS(" && ");
  for (const Record *R : Predicates) {
    StringRef CondString = R->getValueAsString("CondString");
    if (CondString.empty())
      continue;
    OS << LS << '(' << CondString << ')';
  }
}

void SubtargetFeatureInfo::emitMCPredicateCheck(
    raw_ostream &OS, StringRef TargetName,
    ArrayRef<const Record *> Predicates) {
  auto MCPredicates = make_filter_range(Predicates, [](const Record *R) {
    return R->getValueAsBit("AssemblerMatcherPredicate");
  });

  if (MCPredicates.empty()) {
    OS << "false";
    return;
  }

  ListSeparator LS(" && ");
  bool ParenIfBinOp = range_size(MCPredicates) > 1;
  for (const Record *R : MCPredicates) {
    OS << LS;
    if (emitFeaturesAux(R, TargetName, *R->getValueAsDag("AssemblerCondDag"),
                        ParenIfBinOp, OS))
      PrintFatalError(R, "Invalid AssemblerCondDag!");
  }
}

void SubtargetFeatureInfo::emitComputeAssemblerAvailableFeatures(
    StringRef TargetName, StringRef ClassName, StringRef FuncName,
    SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS) {
  OS << "FeatureBitset ";
  if (!ClassName.empty())
    OS << TargetName << ClassName << "::\n";
  OS << FuncName << "(const FeatureBitset &FB) ";
  if (!ClassName.empty())
    OS << "const ";
  OS << "{\n";
  OS << "  FeatureBitset Features;\n";
  for (const SubtargetFeatureInfo &SFI : make_second_range(SubtargetFeatures)) {
    const Record *Def = SFI.TheDef;

    OS << "  if (";
    if (emitFeaturesAux(Def, TargetName,
                        *Def->getValueAsDag("AssemblerCondDag"),
                        /*ParenIfBinOp=*/false, OS))
      PrintFatalError(Def, "Invalid AssemblerCondDag!");

    OS << ")\n";
    OS << "    Features.set(" << SFI.getEnumBitName() << ");\n";
  }
  OS << "  return Features;\n";
  OS << "}\n\n";
}

/// A disjunction of literals. The empty clause is unsatisfiable.
using CNFClause = std::set<SubtargetFeatureLiteral>;
/// A conjunction of clauses. The empty formula is trivially satisfiable.
using CNFFormula = std::vector<CNFClause>;

/// Returns true if \p Clause holds a literal together with its negation, which
/// makes the clause true regardless of how the features are assigned.
static bool isTautology(const CNFClause &Clause) {
  return any_of(Clause, [&Clause](const SubtargetFeatureLiteral &Lit) {
    return Clause.count({Lit.Feature, !Lit.IsNot}) != 0;
  });
}

/// Disjoins two formulas by distributing the clauses of one over the other:
/// (A && B) || (C && D) becomes (A || C) && (A || D) && (B || C) && (B || D).
static CNFFormula disjoin(const CNFFormula &LHS, const CNFFormula &RHS) {
  CNFFormula Result;
  for (const CNFClause &L : LHS) {
    for (const CNFClause &R : RHS) {
      CNFClause Clause = L;
      Clause.insert(R.begin(), R.end());
      if (!isTautology(Clause))
        Result.push_back(std::move(Clause));
    }
  }
  return Result;
}

/// Converts the predicate dag \p Val, owned by \p R, to conjunctive normal
/// form. \p Negate requests the form of the negation of \p Val instead, which
/// is how 'not' is pushed down to the leaves rather than represented
/// explicitly. The accepted grammar is the one emitPredicateDag() accepts:
/// arbitrarily nested all_of/any_of/not over SubtargetFeature leaves.
static CNFFormula convertToCNF(const Record *R, const Init &Val, bool Negate) {
  const auto *D = dyn_cast<DagInit>(&Val);
  if (!D) {
    const auto *Leaf = dyn_cast<DefInit>(&Val);
    if (!Leaf || !Leaf->getDef()->isSubClassOf("SubtargetFeature"))
      PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
    return {{{Leaf->getDef()->getName(), Negate}}};
  }

  const auto *Op = dyn_cast<DefInit>(D->getOperator());
  if (!Op)
    PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
  StringRef OpName = Op->getDef()->getName();

  if (OpName == "not") {
    if (D->getNumArgs() != 1)
      PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
    return convertToCNF(R, *D->getArg(0), !Negate);
  }

  if (OpName != "any_of" && OpName != "all_of")
    PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
  if (D->getNumArgs() == 0)
    PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");

  // By De Morgan, negating turns a conjunction into a disjunction and back.
  bool IsConjunction = (OpName == "all_of") != Negate;

  CNFFormula Result = convertToCNF(R, *D->getArg(0), Negate);
  for (const Init *Arg : D->getArgs().drop_front()) {
    CNFFormula ArgCNF = convertToCNF(R, *Arg, Negate);
    if (IsConjunction)
      Result.insert(Result.end(), std::make_move_iterator(ArgCNF.begin()),
                    std::make_move_iterator(ArgCNF.end()));
    else
      Result = disjoin(Result, ArgCNF);
  }
  return Result;
}

void SubtargetFeatureConstraints::add(
    const SubtargetFeatureConstraints &Other) {
  FeaturesSet.insert(Other.FeaturesSet.begin(), Other.FeaturesSet.end());
  AnyOfFeatureSets.insert(Other.AnyOfFeatureSets.begin(),
                          Other.AnyOfFeatureSets.end());
}

// Simplifies \p OrSet against the literals known to hold. Returns true if the
// set is already satisfied by one of them, and otherwise fills
// \p SimplifiedSet with the literals that are still undecided.
static bool simplifyOrSet(const std::set<SubtargetFeatureLiteral> &OrSet,
                          const std::set<SubtargetFeatureLiteral> &FeaturesSet,
                          std::set<SubtargetFeatureLiteral> &SimplifiedSet) {
  for (const auto &Lit : OrSet) {
    if (FeaturesSet.count(Lit))
      return true; // Satisfied
    if (!FeaturesSet.count({Lit.Feature, !Lit.IsNot}))
      SimplifiedSet.insert(Lit);
  }
  return false;
}

bool SubtargetFeatureConstraints::isSelfContradictory() {
  // Check for immediate contradictions (e.g. requiring both F and !F).
  for (const auto &Lit : FeaturesSet) {
    if (FeaturesSet.count({Lit.Feature, !Lit.IsNot}))
      return true;
  }

  std::set<std::set<SubtargetFeatureLiteral>> NewAnyOfs;
  bool FeaturesChanged = false;

  for (const auto &OrSet : AnyOfFeatureSets) {
    std::set<SubtargetFeatureLiteral> SimplifiedSet;
    if (simplifyOrSet(OrSet, FeaturesSet, SimplifiedSet))
      continue; // Discard satisfied set

    // All literals in this OR-set are false, causing a contradiction.
    if (SimplifiedSet.empty())
      return true;

    if (SimplifiedSet.size() == 1) {
      // Unit clause: only one choice remains, so it must be true.
      FeaturesSet.insert(*SimplifiedSet.begin());
      FeaturesChanged = true;
    } else {
      NewAnyOfs.insert(std::move(SimplifiedSet));
    }
  }

  if (FeaturesChanged) {
    AnyOfFeatureSets = std::move(NewAnyOfs);
    return isSelfContradictory();
  }

  return false;
}

bool SubtargetFeatureConstraints::conflictsWith(
    const SubtargetFeatureConstraints &Other) const {
  SubtargetFeatureConstraints Combined(*this);
  Combined.add(Other);
  return Combined.isSelfContradictory();
}

bool llvm::getRequiredFeatures(
    std::set<SubtargetFeatureLiteral> &FeaturesSet,
    std::set<std::set<SubtargetFeatureLiteral>> &AnyOfFeatureSets,
    ArrayRef<const Record *> ReqPredicates) {
  bool AllPredicatesParsed = true;
  for (const Record *R : ReqPredicates) {
    if (!R->getValueAsBit("AssemblerMatcherPredicate")) {
      AllPredicatesParsed = false;
      continue;
    }
    const RecordVal *V = R->getValue("AssemblerCondDag");
    if (!V || !V->getValue() || !isa<DagInit>(V->getValue())) {
      // Ignore this predicate for semantic analysis, as it has no subtarget
      // features.
      AllPredicatesParsed = false;
      continue;
    }

    // A clause of one literal says that literal must hold, so it belongs with
    // the unconditionally required features rather than among the groups.
    for (CNFClause &Clause :
         convertToCNF(R, *V->getValue(), /*Negate=*/false)) {
      if (Clause.size() == 1)
        FeaturesSet.insert(*Clause.begin());
      else
        AnyOfFeatureSets.insert(std::move(Clause));
    }
  }
  return AllPredicatesParsed;
}
