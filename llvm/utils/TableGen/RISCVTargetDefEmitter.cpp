//===- RISCVTargetDefEmitter.cpp - Generate lists of RISCV CPUs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits the include file needed by the target
// parser to parse the RISC-V CPUs.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

using ISAInfoTy = llvm::Expected<std::unique_ptr<RISCVISAInfo>>;

static int getXLen(const Record &Rec) {
  std::vector<Record *> Features = Rec.getValueAsListOfDefs("Features");
  if (find_if(Features, [](const Record *R) {
        return R->getName() == "Feature64Bit";
      }) != Features.end())
    return 64;

  return 32;
}

// We can generate march string from target features as what has been described
// in RISCV ISA specification (version 20191213) 'Chapter 27. ISA Extension
// Naming Conventions'.
//
// This is almost the same as RISCVFeatures::parseFeatureBits, except that we
// get feature name from feature records instead of feature bits.
static std::string getMArch(int XLen, const Record &Rec) {
  std::vector<std::string> FeatureVector;

  // Convert features to FeatureVector.
  for (auto *Feature : Rec.getValueAsListOfDefs("Features")) {
    StringRef FeatureName = Feature->getValueAsString("Name");
    if (llvm::RISCVISAInfo::isSupportedExtensionFeature(FeatureName))
      FeatureVector.push_back((Twine("+") + FeatureName).str());
  }

  ISAInfoTy ISAInfo = llvm::RISCVISAInfo::parseFeatures(XLen, FeatureVector);
  if (!ISAInfo)
    report_fatal_error("Invalid features");

  // RISCVISAInfo::toString will generate a march string with all the extensions
  // we have added to it.
  return (*ISAInfo)->toString();
}

static std::string getEnumFeatures(int XLen) {
  return XLen == 64 ? "FK_64BIT" : "FK_NONE";
}

void llvm::EmitRISCVTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "#ifndef PROC\n"
     << "#define PROC(ENUM, NAME, FEATURES, DEFAULT_MARCH)\n"
     << "#endif\n\n";

  OS << "PROC(INVALID, {\"invalid\"}, FK_INVALID, {\"\"})\n";
  // Iterate on all definition records.
  for (const Record *Rec : RK.getAllDerivedDefinitions("RISCVProcessorModel")) {
    int XLen = getXLen(*Rec);
    std::string MArch = Rec->getValueAsString("DefaultMarch").str();

    // Compute MArch from features if we don't specify it.
    if (MArch.empty())
      MArch = getMArch(XLen, *Rec);

    OS << "PROC(" << Rec->getName() << ", "
       << "{\"" << Rec->getValueAsString("Name") << "\"},"
       << getEnumFeatures(XLen) << ", "
       << "{\"" << MArch << "\"})\n";
  }
  OS << "\n#undef PROC\n";
  OS << "\n";
  OS << "#ifndef TUNE_PROC\n"
     << "#define TUNE_PROC(ENUM, NAME)\n"
     << "#endif\n\n";
  OS << "TUNE_PROC(GENERIC, \"generic\")\n";

  for (const Record *Rec :
       RK.getAllDerivedDefinitions("RISCVTuneProcessorModel")) {
    OS << "TUNE_PROC(" << Rec->getName() << ", "
       << "\"" << Rec->getValueAsString("Name") << "\")\n";
  }

  OS << "\n#undef TUNE_PROC\n";
}
