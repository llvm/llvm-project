//===- IntrinsicFeatureEmitter.cpp - Intrinsic feature table --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/FeatureGuardParser.h"
#include "llvm/TableGen/GuardedIntrinsic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>
#include <vector>

using namespace llvm;

namespace {

static cl::opt<std::string> IntrinsicFeaturePrefix(
    "intrinsic-feature-prefix",
    cl::desc("TargetPrefix of intrinsics whose feature checks are generated"),
    cl::value_desc("target prefix"));
static cl::opt<std::string> IntrinsicFeatureNamespace(
    "intrinsic-feature-namespace",
    cl::desc("C++ feature namespace and prefix for generated symbols"),
    cl::value_desc("namespace"));

// Translate guard spellings such as "sve" to SubtargetFeature record names
// such as "FeatureSVE", which the generated code uses as target enum values.
static FeatureGuardAlternatives
parseGuard(StringRef FeatureString,
           const std::map<std::string, std::string> &Features,
           const Record *Owner, StringRef TargetPrefix) {
  auto ErrorHandler = [&Owner](const Twine &Message) {
    PrintFatalError(Owner, Message);
  };

  FeatureGuardAlternatives Result;
  for (const RequiredFeatureSet &Names :
       FeatureGuardParser(FeatureString, ErrorHandler).parse()) {
    RequiredFeatureSet FeaturesInClause;
    for (const std::string &Name : Names) {
      auto It = Features.find(Name);
      if (It == Features.end())
        PrintFatalError(Owner, "unknown " + TargetPrefix +
                                   " feature in target guard: " + Name);
      FeaturesInClause.insert(It->second);
    }
    Result.insert(std::move(FeaturesInClause));
  }
  return Result;
}

// One unique feature guard expression. Its parsed clauses are alternatives:
// any clause can satisfy the guard, and each clause requires all its features.
// For example, IntrinsicsAArch64.td has:
//
//   let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_Overloads>] in {
//     ...
//   }
//
// Here the first string maps to a GuardEntry with clauses requiring {sve, sve2}
// or {sve, sme}; the second maps to a separate GuardEntry for "sme".
// The generated guard table refers to these clauses by offset and count.
struct GuardEntry {
  std::string RawGuardString;
  FeatureGuardAlternatives ParsedGuardClauses;
};

// One TargetGuard variant of an intrinsic.
struct VariantEntry {
  // Overload types in intrinsic signature order; empty when the guard applies
  // to every signature or the intrinsic is not overloaded.
  std::vector<std::string> Types;
  // Pointers to unique GuardEntry nodes, in TargetGuard order.
  std::vector<const GuardEntry *> Guards;
  bool AppliesToAllTypes = false;
};

// Add a check for one TargetGuard variant. For overloaded variants, record the
// overload types; otherwise the check applies to all types. Reuse
// existing guards, parsing each new guard expression once.
static void addVariant(std::vector<VariantEntry> &Entries,
                       const Record *Intrinsic, const Record *Guard,
                       const ListInit *Overload,
                       const std::map<std::string, std::string> &Features,
                       StringRef Prefix,
                       std::map<std::string, GuardEntry> &GuardsByExpression,
                       std::vector<const GuardEntry *> &GuardsInOrder) {
  auto GuardStrings = Guard->getValueAsListOfStrings("Guards");
  if (GuardStrings.empty())
    PrintFatalError(Guard, "intrinsic variant must have at least one guard");

  // Parse the guard strings (or use previously parsed GuardEntry from cache).
  VariantEntry Entry;
  for (StringRef RawGuardString : GuardStrings) {
    auto [It, Inserted] = GuardsByExpression.try_emplace(RawGuardString.str());
    if (Inserted) {
      It->second = {It->first,
                    parseGuard(RawGuardString, Features, Guard, Prefix)};
      GuardsInOrder.push_back(&It->second);
    }
    Entry.Guards.push_back(&It->second);
  }

  // Record the overload types used to construct the intrinsic's signature.
  Entry.AppliesToAllTypes = !Overload;
  if (Overload) {
    auto Types = getGuardedOverloadTypes(Overload);
    unsigned Expected = Intrinsic->getValueAsDef("TypeInfo")
                            ->getValueAsListOfDefs("OverloadTypes")
                            .size();
    if (Types.size() != Expected)
      PrintFatalError(Guard, "wrong number of intrinsic overload types for " +
                                 Intrinsic->getName());
    for (const Record *T : Types)
      Entry.Types.push_back(T->getValueAsDef("VT")->getName().str());
  }

  Entries.push_back(std::move(Entry));
}

static void emitIntrinsicFeatureChecks(const RecordKeeper &Records,
                                       raw_ostream &OS) {
  StringRef Prefix = IntrinsicFeaturePrefix;
  StringRef Namespace = IntrinsicFeatureNamespace;

  if (Prefix.empty() || Namespace.empty())
    PrintFatalError(
        "-gen-intrinsic-feature-checks requires "
        "-intrinsic-feature-prefix and -intrinsic-feature-namespace");

  // Collate all the target's SubtargetFeatures in 'Features'.
  std::map<std::string, std::string> Features;
  for (const Record *R : Records.getAllDerivedDefinitions("SubtargetFeature")) {
    std::string Name = R->getValueAsString("Name").str();
    if (Name.empty())
      PrintFatalError(R, "Empty feature name for SubtargetFeature");
    if (!Features.emplace(Name, R->getName().str()).second)
      PrintFatalError(R, "duplicate subtarget feature: " + Name);
  }

  // Map nodes keep GuardEntry pointers stable as new guards are inserted.
  std::map<std::string, GuardEntry> GuardsByExpression;
  std::vector<const GuardEntry *> GuardsInOrder;
  std::map<std::string, std::vector<VariantEntry>> Intrinsics;
  for (const Record *Intrinsic :
       Records.getAllDerivedDefinitions("GuardedIntrinsic")) {
    if (Intrinsic->getValueAsString("TargetPrefix") != Prefix)
      continue;
    if (!Intrinsic->getValueAsString("TargetFeatures").empty())
      PrintFatalError(Intrinsic,
                      "GuardedIntrinsic must not specify TargetFeatures; "
                      "use TargetGuards instead");
    auto TargetGuards = Intrinsic->getValueAsListOfDefs("TargetGuards");
    if (TargetGuards.empty())
      PrintFatalError(Intrinsic,
                      "intrinsic " + Intrinsic->getName() +
                          " must have at least one TargetGuard variant");
    auto &Entries = Intrinsics[Intrinsic->getName().str()];
    for (const GuardedIntrinsicVariant &V :
         collectGuardedIntrinsicVariants(Intrinsic)) {
      if (!V.OverloadTypes && TargetGuards.size() != 1)
        PrintFatalError(Intrinsic, "type-independent TargetGuard must be the "
                                   "intrinsic's only TargetGuard variant");
      addVariant(Entries, Intrinsic, V.Guard, V.OverloadTypes, Features, Prefix,
                 GuardsByExpression, GuardsInOrder);
    }
  }

  OS << "// Generated by -gen-intrinsic-feature-checks.\n";
  OS << "[[maybe_unused]] static ArrayRef<FeatureBitset> get" << Namespace
     << "IntrinsicFeatureAlternatives(const IntrinsicFeatureGuard &Guard) {\n"
        "  static constexpr FeatureBitset "
     << Namespace << "IntrinsicFeatureClauses[] = {\n";
  for (const GuardEntry *G : GuardsInOrder)
    for (const RequiredFeatureSet &C : G->ParsedGuardClauses) {
      OS << "    {";
      interleaveComma(C, OS, [&](const std::string &Feature) {
        OS << Namespace << "::" << Feature;
      });
      OS << "},\n";
    }
  OS << "  };\n"
        "  return ArrayRef("
     << Namespace
     << "IntrinsicFeatureClauses + Guard.FirstClause, Guard.NumClauses);\n"
        "}\n\n";

  OS << "[[maybe_unused]] static FunctionType *get" << Namespace
     << "IntrinsicFunctionType(LLVMContext &Context, Intrinsic::ID ID, "
        "const IntrinsicFeatureVariant &Variant) {\n"
        "  static constexpr MVT::SimpleValueType "
     << Namespace << "IntrinsicOverloadTypes[] = {\n";
  bool HasOverloadTypes = false;
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "    ";
      for (const std::string &T : V.Types) {
        OS << "MVT::" << T << ", ";
        HasOverloadTypes = true;
      }
      OS << "// " << Name << "\n";
    }
  if (!HasOverloadTypes)
    OS << "    MVT::INVALID_SIMPLE_VALUE_TYPE,\n";
  OS << "  };\n"
        "  SmallVector<Type *, 4> OverloadTypes;\n"
        "  for (MVT::SimpleValueType VT :\n"
        "       ArrayRef("
     << Namespace
     << "IntrinsicOverloadTypes + Variant.FirstType, Variant.NumTypes))\n"
        "    OverloadTypes.push_back(VT == MVT::iPTR\n"
        "                                ? PointerType::get(Context, 0)\n"
        "                                : EVT(VT).getTypeForEVT(Context));\n"
        "  return Intrinsic::getType(Context, ID, OverloadTypes);\n"
        "}\n\n";

  OS << "[[maybe_unused]] static const IntrinsicFeatureGuard &get" << Namespace
     << "IntrinsicFeatureGuard(const IntrinsicFeatureVariant &Variant, "
        "unsigned Index) {\n"
        "  static constexpr unsigned "
     << Namespace << "IntrinsicGuardIndices[] = {\n";
  std::map<const GuardEntry *, unsigned> GuardIndices;
  for (unsigned I = 0; I < GuardsInOrder.size(); ++I)
    GuardIndices.emplace(GuardsInOrder[I], I);
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "    ";
      interleaveComma(V.Guards, OS,
                      [&](const GuardEntry *G) { OS << GuardIndices.at(G); });
      OS << ", // " << Name << "\n";
    }
  OS << "  };\n"
        "  static constexpr IntrinsicFeatureGuard "
     << Namespace << "IntrinsicFeatureGuards[] = {\n";
  unsigned FirstClauseIndex = 0;
  for (const GuardEntry *G : GuardsInOrder) {
    OS << "    {" << FirstClauseIndex << ", " << G->ParsedGuardClauses.size()
       << ", \"" << G->RawGuardString << "\"},\n";
    FirstClauseIndex += G->ParsedGuardClauses.size();
  }
  OS << "  };\n"
        "  assert(Index < Variant.NumGuards && \"invalid intrinsic feature "
        "guard index\");\n"
        "  return "
     << Namespace << "IntrinsicFeatureGuards[" << Namespace
     << "IntrinsicGuardIndices[Variant.FirstGuard + Index]];\n"
        "}\n\n";

  OS << "[[maybe_unused]] static ArrayRef<IntrinsicFeatureVariant> get"
     << Namespace
     << "IntrinsicVariants(Intrinsic::ID ID) {\n"
        "  static constexpr IntrinsicFeatureVariant "
     << Namespace << "IntrinsicVariants[] = {\n";
  unsigned FirstType = 0;
  unsigned FirstGuard = 0;
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "    {" << FirstType << ", " << V.Types.size() << ", " << FirstGuard
         << ", " << V.Guards.size() << ", "
         << (V.AppliesToAllTypes ? "true" : "false") << "}, // " << Name
         << "\n";
      FirstType += V.Types.size();
      FirstGuard += V.Guards.size();
    }
  OS << "  };\n"
        "  switch (ID) {\n";
  unsigned FirstVariant = 0;
  for (const auto &[Name, Entries] : Intrinsics) {
    if (!StringRef(Name).starts_with("int_"))
      PrintFatalError("unexpected intrinsic record name: " + Name);
    OS << "  case Intrinsic::" << StringRef(Name).drop_front(4) << ":\n"
       << "    return ArrayRef(" << Namespace << "IntrinsicVariants + "
       << FirstVariant << ", " << Entries.size() << ");\n";
    FirstVariant += Entries.size();
  }
  OS << "  default:\n    return {};\n  }\n}\n";
}

static TableGen::Emitter::Opt
    X("gen-intrinsic-feature-checks", emitIntrinsicFeatureChecks,
      "Generate intrinsic feature checks from TargetGuards");

} // namespace
