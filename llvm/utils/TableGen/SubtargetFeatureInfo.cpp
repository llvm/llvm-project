//===- SubtargetFeatureInfo.cpp - Helpers for subtarget features ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubtargetFeatureInfo.h"

#include "Types.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/TableGen/Record.h"

#include <map>

using namespace llvm;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SubtargetFeatureInfo::dump() const {
  errs() << getEnumName() << " " << Index << "\n" << *TheDef;
}
#endif

std::vector<std::pair<Record *, SubtargetFeatureInfo>>
SubtargetFeatureInfo::getAll(const RecordKeeper &Records) {
  std::vector<std::pair<Record *, SubtargetFeatureInfo>> SubtargetFeatures;
  std::vector<Record *> AllPredicates =
      Records.getAllDerivedDefinitions("Predicate");
  for (Record *Pred : AllPredicates) {
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
    SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS) {
  OS << "// Bits for subtarget features that participate in "
     << "instruction matching.\n";
  OS << "enum SubtargetFeatureBits : "
     << getMinimalTypeForRange(SubtargetFeatures.size()) << " {\n";
  for (const auto &SF : SubtargetFeatures) {
    const SubtargetFeatureInfo &SFI = SF.second;
    OS << "  " << SFI.getEnumBitName() << " = " << SFI.Index << ",\n";
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
      IndexUB = SF.second.Index+1;

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
    SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS,
    StringRef ExtraParams) {
  OS << "PredicateBitset " << TargetName << ClassName << "::\n"
     << FuncName << "(const " << TargetName << "Subtarget *Subtarget";
  if (!ExtraParams.empty())
    OS << ", " << ExtraParams;
  OS << ") const {\n";
  OS << "  PredicateBitset Features;\n";
  for (const auto &SF : SubtargetFeatures) {
    const SubtargetFeatureInfo &SFI = SF.second;
    StringRef CondStr = SFI.TheDef->getValueAsString("CondString");
    assert(!CondStr.empty() && "true predicate should have been filtered");

    OS << "  if (" << CondStr << ")\n";
    OS << "    Features.set(" << SFI.getEnumBitName() << ");\n";
  }
  OS << "  return Features;\n";
  OS << "}\n\n";
}

void SubtargetFeatureInfo::emitComputeAssemblerAvailableFeatures(
    StringRef TargetName, StringRef ClassName, StringRef FuncName,
    SubtargetFeatureInfoMap &SubtargetFeatures, raw_ostream &OS) {
  OS << "FeatureBitset " << TargetName << ClassName << "::\n"
     << FuncName << "(const FeatureBitset& FB) const {\n";
  OS << "  FeatureBitset Features;\n";
  for (const auto &SF : SubtargetFeatures) {
    const SubtargetFeatureInfo &SFI = SF.second;

    OS << "  if (";

    const DagInit *D = SFI.TheDef->getValueAsDag("AssemblerCondDag");
    std::string CombineType = D->getOperator()->getAsString();
    if (CombineType != "any_of" && CombineType != "all_of")
      PrintFatalError(SFI.TheDef->getLoc(), "Invalid AssemblerCondDag!");
    if (D->getNumArgs() == 0)
      PrintFatalError(SFI.TheDef->getLoc(), "Invalid AssemblerCondDag!");
    bool IsOr = CombineType == "any_of";

    if (IsOr)
      OS << "(";

    bool First = true;
    for (auto *Arg : D->getArgs()) {
      if (!First) {
        if (IsOr)
          OS << " || ";
        else
          OS << " && ";
      }
      if (auto *NotArg = dyn_cast<DagInit>(Arg)) {
        if (NotArg->getOperator()->getAsString() != "not" ||
            NotArg->getNumArgs() != 1)
          PrintFatalError(SFI.TheDef->getLoc(), "Invalid AssemblerCondDag!");
        Arg = NotArg->getArg(0);
        OS << "!";
      }
      if (!isa<DefInit>(Arg) ||
          !cast<DefInit>(Arg)->getDef()->isSubClassOf("SubtargetFeature"))
        PrintFatalError(SFI.TheDef->getLoc(), "Invalid AssemblerCondDag!");
      OS << "FB[" << TargetName << "::" << Arg->getAsString() << "]";

      First = false;
    }

    if (IsOr)
      OS << ")";

    OS << ")\n";
    OS << "    Features.set(" << SFI.getEnumBitName() << ");\n";
  }
  OS << "  return Features;\n";
  OS << "}\n\n";
}
