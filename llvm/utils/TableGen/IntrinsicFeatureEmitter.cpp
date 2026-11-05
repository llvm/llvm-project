//===- IntrinsicFeatureEmitter.cpp - Intrinsic feature table --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/FeatureGuardParser.h"
#include "llvm/TableGen/GuardedIntrinsic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
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
  // Instantiated result types followed by input types for an overloaded
  // variant; empty when the guard applies to every signature.
  std::vector<std::string> Types;
  unsigned Results = 0;
  unsigned Inputs = 0;
  // Pointers to unique GuardEntry nodes, in TargetGuard order.
  std::vector<const GuardEntry *> Guards;
  bool AppliesToAllTypes = false;
};

// The handler must not return.
using ErrorHandler = function_ref<void(const Twine &)>;

// Find the ValueType for a concrete LLVMType with the given type signature.
static const Record *findFixedTypeVT(ArrayRef<int64_t> Sig,
                                     ArrayRef<const Record *> LLVMTypes,
                                     ErrorHandler ReportError) {
  for (const Record *Type : LLVMTypes) {
    if (Type->isSubClassOf("LLVMDependentType") ||
        Type->isSubClassOf("LLVMAnyType"))
      continue;
    auto Candidate = Type->getValueAsListOfInts("Sig");
    if (Candidate.size() == Sig.size() &&
        std::equal(Candidate.begin(), Candidate.end(), Sig.begin()))
      return Type->getValueAsDef("VT");
  }
  ReportError("no fixed LLVM type for dependent intrinsic type");
  return nullptr;
}

// Find a scalar element type by bit width and floating-point or integer kind.
static const Record *findScalarElement(int Size, bool IsFP,
                                       ArrayRef<const Record *> ValueTypes,
                                       ErrorHandler ReportError) {
  for (const Record *VT : ValueTypes) {
    // Subdividing a floating-point element to 16 bits uses IEEE half.
    if (IsFP && Size == 16 && VT->getName() != "f16")
      continue;
    if (!VT->getValueAsBit("isVector") &&
        VT->getValueAsBit(IsFP ? "isFP" : "isInteger") &&
        VT->getValueAsInt("Size") == Size)
      return VT;
  }
  ReportError("no scalar ValueType of size " + Twine(Size));
  return nullptr;
}

// Preserve the reference vector's scalability while changing its element
// type and number of elements.
static const Record *findVector(const Record *RefVT, const Record *ElementVT,
                                int Elements,
                                ArrayRef<const Record *> ValueTypes,
                                ErrorHandler ReportError) {
  for (const Record *VT : ValueTypes) {
    if (VT->getValueAsBit("isVector") &&
        VT->getValueAsBit("isScalable") == RefVT->getValueAsBit("isScalable") &&
        VT->getValueAsInt("nElem") == Elements &&
        VT->getValueAsDef("ElementType") == ElementVT)
      return VT;
  }
  ReportError("no LLVM vector type for dependent intrinsic type");
  return nullptr;
}

// Instantiate a dependent signature type from its referenced overload type.
static const Record *
instantiateDependentType(const Record *T, ArrayRef<const Record *> Overloads,
                         ArrayRef<const Record *> ValueTypes,
                         ArrayRef<const Record *> LLVMTypes,
                         ErrorHandler ReportError) {
  unsigned Index = T->getValueAsInt("OverloadIndex");
  const Record *RefVT = Overloads[Index]->getValueAsDef("VT");
  if (T->isSubClassOf("LLVMMatchType")) {
    return RefVT;
  } else if (T->isSubClassOf("LLVMScalarOrSameVectorWidth")) {
    auto Sig = T->getValueAsListOfInts("Sig");
    if (Sig.size() < 3)
      PrintFatalError(T, "missing scalar element type in intrinsic signature");
    const Record *ElementVT =
        findFixedTypeVT(ArrayRef(Sig).drop_front(2), LLVMTypes, ReportError);
    return RefVT->getValueAsBit("isVector")
               ? findVector(RefVT, ElementVT, RefVT->getValueAsInt("nElem"),
                            ValueTypes, ReportError)
               : ElementVT;
  } else if (T->isSubClassOf("LLVMVectorElementType")) {
    return RefVT->getValueAsDef("ElementType");
  } else if (T->isSubClassOf("LLVMExtendedType") ||
             T->isSubClassOf("LLVMTruncatedType") ||
             T->isSubClassOf("LLVMVectorOfBitcastsToInt")) {
    const Record *ElementVT = RefVT->getValueAsBit("isVector")
                                  ? RefVT->getValueAsDef("ElementType")
                                  : RefVT;
    int Size = ElementVT->getValueAsInt("Size");
    if (T->isSubClassOf("LLVMExtendedType"))
      Size *= 2;
    else if (T->isSubClassOf("LLVMTruncatedType"))
      Size /= 2;
    const Record *IntVT =
        findScalarElement(Size, false, ValueTypes, ReportError);
    return RefVT->getValueAsBit("isVector")
               ? findVector(RefVT, IntVT, RefVT->getValueAsInt("nElem"),
                            ValueTypes, ReportError)
               : IntVT;
  } else if (T->isSubClassOf("LLVMOneNthElementsVectorType") ||
             T->isSubClassOf("LLVMSubdivide2VectorType") ||
             T->isSubClassOf("LLVMSubdivide4VectorType")) {
    int Elements = RefVT->getValueAsInt("nElem");
    const Record *ElementVT = RefVT->getValueAsDef("ElementType");
    if (T->isSubClassOf("LLVMOneNthElementsVectorType")) {
      auto Sig = T->getValueAsListOfInts("Sig");
      if (Sig.size() != 3)
        PrintFatalError(T, "missing vector divisor in intrinsic signature");
      Elements /= Sig[2];
    } else {
      int Factor = T->isSubClassOf("LLVMSubdivide2VectorType") ? 2 : 4;
      Elements *= Factor;
      ElementVT = findScalarElement(ElementVT->getValueAsInt("Size") / Factor,
                                    ElementVT->getValueAsBit("isFP"),
                                    ValueTypes, ReportError);
    }
    return findVector(RefVT, ElementVT, Elements, ValueTypes, ReportError);
  }
  ReportError("unsupported dependent intrinsic type " + T->getName());
  return nullptr;
}

// Instantiate the intrinsic's result and parameter types for a guard variant.
// TypeInfo.OverloadTypes uses the same indices as LLVMDependentType.
static std::vector<const Record *>
instantiateSignatureForVariant(const Record *Intrinsic, const Record *Guard,
                               const ListInit *Overload,
                               const RecordKeeper &Records) {
  auto ReportError = [Guard](const Twine &Message) {
    PrintFatalError(Guard, Message);
  };
  const Record *TypeInfo = Intrinsic->getValueAsDef("TypeInfo");
  auto Overloads = getGuardedOverloadTypes(Overload);
  if (Overloads.size() !=
      TypeInfo->getValueAsListOfDefs("OverloadTypes").size())
    PrintFatalError(Guard, "wrong number of intrinsic overload types for " +
                               Intrinsic->getName());

  auto ValueTypes = Records.getAllDerivedDefinitions("ValueType");
  auto LLVMTypes = Records.getAllDerivedDefinitions("LLVMType");

  std::vector<const Record *> Signature;
  auto Results = Intrinsic->getValueAsListOfDefs("RetTypes");
  auto Inputs = Intrinsic->getValueAsListOfDefs("ParamTypes");
  Signature.insert(Signature.end(), Results.begin(), Results.end());
  Signature.insert(Signature.end(), Inputs.begin(), Inputs.end());

  std::vector<const Record *> Expanded;
  unsigned NextOverload = 0;
  for (const Record *T : Signature) {
    if (T->isSubClassOf("LLVMAnyType") ||
        T->isSubClassOf("LLVMPartiallyDependentType")) {
      Expanded.push_back(Overloads[NextOverload++]->getValueAsDef("VT"));
      continue;
    }
    if (!T->isSubClassOf("LLVMDependentType")) {
      Expanded.push_back(T->getValueAsDef("VT"));
      continue;
    }

    Expanded.push_back(instantiateDependentType(T, Overloads, ValueTypes,
                                                LLVMTypes, ReportError));
  }
  return Expanded;
}

// Add a check for one TargetGuard variant. For overloaded variants, record the
// instantiated signature; otherwise the check applies to all types. Reuse
// existing guards, parsing each new guard expression once.
static void addVariant(std::vector<VariantEntry> &Entries,
                       const Record *Intrinsic, const Record *Guard,
                       const ListInit *Overload, const RecordKeeper &Records,
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

  // If the target guard only works for certain types, then instantiate the
  // types for this variant.
  Entry.AppliesToAllTypes = !Overload;
  if (Overload) {
    auto Types =
        instantiateSignatureForVariant(Intrinsic, Guard, Overload, Records);
    Entry.Results = Intrinsic->getValueAsListOfDefs("RetTypes").size();
    Entry.Inputs = Intrinsic->getValueAsListOfDefs("ParamTypes").size();
    for (const Record *VT : Types)
      Entry.Types.push_back(VT->getName().str());
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
      addVariant(Entries, Intrinsic, V.Guard, V.OverloadTypes, Records,
                 Features, Prefix, GuardsByExpression, GuardsInOrder);
    }
  }

  OS << "// Generated by -gen-intrinsic-feature-checks.\n";
  OS << "static constexpr FeatureBitset " << Namespace
     << "IntrinsicFeatureClauses[] = {\n";
  for (const GuardEntry *G : GuardsInOrder)
    for (const RequiredFeatureSet &C : G->ParsedGuardClauses) {
      OS << "  {";
      interleaveComma(C, OS, [&](const std::string &Feature) {
        OS << Namespace << "::" << Feature;
      });
      OS << "},\n";
    }
  OS << "};\n\n";

  OS << "static constexpr IntrinsicFeatureGuard " << Namespace
     << "IntrinsicFeatureGuards[] = {\n";
  unsigned FirstClauseIndex = 0;
  for (const GuardEntry *G : GuardsInOrder) {
    OS << "  {" << FirstClauseIndex << ", " << G->ParsedGuardClauses.size()
       << ", \"" << G->RawGuardString << "\"},\n";
    FirstClauseIndex += G->ParsedGuardClauses.size();
  }
  OS << "};\n\n";

  OS << "static constexpr MVT::SimpleValueType " << Namespace
     << "IntrinsicTypes[] = {\n";
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "  ";
      for (const std::string &T : V.Types)
        OS << "MVT::" << T << ", ";
      OS << "// " << Name << "\n";
    }
  OS << "};\n\n";

  OS << "static constexpr unsigned " << Namespace
     << "IntrinsicGuardIndices[] = {\n";
  std::map<const GuardEntry *, unsigned> GuardIndices;
  for (unsigned I = 0; I < GuardsInOrder.size(); ++I)
    GuardIndices.emplace(GuardsInOrder[I], I);
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "  ";
      interleaveComma(V.Guards, OS,
                      [&](const GuardEntry *G) { OS << GuardIndices.at(G); });
      OS << ", // " << Name << "\n";
    }
  OS << "};\n\n";

  OS << "static constexpr IntrinsicFeatureVariant " << Namespace
     << "IntrinsicVariants[] = {\n";
  unsigned FirstType = 0;
  unsigned FirstGuard = 0;
  for (const auto &[Name, Entries] : Intrinsics)
    for (const VariantEntry &V : Entries) {
      OS << "  {" << FirstType << ", " << V.Results << ", " << V.Inputs << ", "
         << FirstGuard << ", " << V.Guards.size() << ", "
         << (V.AppliesToAllTypes ? "true" : "false") << "}, // " << Name
         << "\n";
      FirstType += V.Types.size();
      FirstGuard += V.Guards.size();
    }
  OS << "};\n\n";

  OS << "static ArrayRef<IntrinsicFeatureVariant> get" << Namespace
     << "IntrinsicVariants(Intrinsic::ID ID) {\n"
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
