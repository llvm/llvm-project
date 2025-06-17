//===- EmitTargetFeature.cpp - Generate CPU Targer feature ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This tablegen backend exports cpu target features
//  and cpu sub-type for all platform.  
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/TargetParser/SubtargetFeature.h"

using namespace llvm;

using FeatureMapTy = DenseMap<const Record *, unsigned>;
using ConstRecVec = std::vector<const Record *>;

struct LessRecordFieldNameAndID {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return std::tuple(Rec1->getValueAsString("Name"), Rec1->getID()) <
           std::tuple(Rec2->getValueAsString("Name"), Rec2->getID());
  }
};

static StringRef getTargetName(const RecordKeeper &Records) {
  ArrayRef<const Record *> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() == 0)
    PrintFatalError("No 'Target' subclasses defined!");
  if (Targets.size() != 1)
    PrintFatalError("Multiple subclasses of Target defined!");
  return Targets[0]->getName();
}

static FeatureMapTy enumeration(const RecordKeeper &Records, raw_ostream &OS) {
  ArrayRef<const Record *> DefList =
      Records.getAllDerivedDefinitions("SubtargetFeature");

  unsigned N = DefList.size();
  if (N == 0)
    return FeatureMapTy();

  if (N + 1 > MAX_SUBTARGET_FEATURES)
    PrintFatalError(
        "Too many subtarget features! Bump MAX_SUBTARGET_FEATURES.");

  StringRef Target = getTargetName(Records);

  OS << "namespace " << Target << " {\n";

  OS << "enum {\n";

  FeatureMapTy FeatureMap;
  for (unsigned I = 0; I < N; ++I) {
    const Record *Def = DefList[I];
    // Print the Feature Name.
    OS << "  " << Def->getName() << " = " << I << ",\n";

    FeatureMap[Def] = I;
  }

  OS << "  " << "NumSubtargetFeatures = " << N << "\n";

  // Close enumeration and namespace
  OS << "};\n";
  OS << "} // end namespace " << Target << "\n";
  return FeatureMap;
}

static void printFeatureMask(raw_ostream &OS,
                             ArrayRef<const Record *> FeatureList,
                             const FeatureMapTy &FeatureMap) {
  std::array<uint64_t, MAX_SUBTARGET_WORDS> Mask = {};
  for (const Record *Feature : FeatureList) {
    unsigned Bit = FeatureMap.lookup(Feature);
    Mask[Bit / 64] |= 1ULL << (Bit % 64);
  }

  OS << "{ { { ";
  for (unsigned I = 0; I != Mask.size(); ++I) {
    OS << "0x";
    OS.write_hex(Mask[I]);
    OS << "ULL, ";
  }
  OS << "} } }";
}

static void printFeatureKeyValues(const RecordKeeper &Records, raw_ostream &OS,
                                  const FeatureMapTy &FeatureMap) {
  std::vector<const Record *> FeatureList =
      Records.getAllDerivedDefinitions("SubtargetFeature");

  // Remove features with empty name.
  llvm::erase_if(FeatureList, [](const Record *Rec) {
    return Rec->getValueAsString("Name").empty();
  });

  if (FeatureList.empty())
    return;

  llvm::sort(FeatureList, LessRecordFieldNameAndID());

  StringRef Target = getTargetName(Records);
  // Begin feature table.
  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "extern const llvm::BasicSubtargetFeatureKV " << "Basic" << Target
     << "FeatureKV[] = {\n";

  for (const Record *Feature : FeatureList) {
    StringRef Name = Feature->getName();
    StringRef ValueName = Feature->getValueAsString("Name");

    OS << "  { " << "\"" << ValueName << "\", " << Target << "::" << Name
       << ", ";

    ConstRecVec ImpliesList = Feature->getValueAsListOfDefs("Implies");

    printFeatureMask(OS, ImpliesList, FeatureMap);

    OS << " },\n";
  }

  // End feature table.
  OS << "};\n";

  return;
}

void printCPUKeyValues(const RecordKeeper &Records, raw_ostream &OS,
                       const FeatureMapTy &FeatureMap) {
  // Gather and sort processor information
  std::vector<const Record *> ProcessorList =
      Records.getAllDerivedDefinitions("Processor");
  llvm::sort(ProcessorList, LessRecordFieldName());

  StringRef Target = getTargetName(Records);

  // Begin processor table.
  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "extern const llvm::BasicSubtargetSubTypeKV " << "Basic" << Target
     << "SubTypeKV[] = {\n";

  for (const Record *Processor : ProcessorList) {
    StringRef Name = Processor->getValueAsString("Name");
    ConstRecVec FeatureList = Processor->getValueAsListOfDefs("Features");

    OS << " { " << "\"" << Name << "\", ";

    printFeatureMask(OS, FeatureList, FeatureMap);
    OS << " },\n";
  }

  // End processor table.
  OS << "};\n";

  return;
}

static void emitTargetFeature(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "// Autogenerated by TargetFeatureEmitter.cpp\n\n";

  OS << "\n#ifdef GET_SUBTARGETFEATURES_ENUM\n";
  OS << "#undef GET_SUBTARGETFEATURES_ENUM\n\n";

  OS << "namespace llvm {\n";
  auto FeatureMap = enumeration(RK, OS);
  OS << "} // end namespace llvm\n\n";
  OS << "#endif // GET_SUBTARGETFEATURES_ENUM\n\n";

  OS << "\n#ifdef GET_SUBTARGETFEATURES_KV\n";
  OS << "#undef GET_SUBTARGETFEATURES_KV\n\n";

  OS << "namespace llvm {\n";
  printFeatureKeyValues(RK, OS, FeatureMap);
  OS << "\n";

  printCPUKeyValues(RK, OS, FeatureMap);
  OS << "\n";
  OS << "} // end namespace llvm\n\n";
  OS << "#endif // GET_SUBTARGETFEATURES_KV\n\n";
}

static TableGen::Emitter::Opt
    X("gen-target-features", emitTargetFeature,
      "Generate the default Target features and CPU sub types");
