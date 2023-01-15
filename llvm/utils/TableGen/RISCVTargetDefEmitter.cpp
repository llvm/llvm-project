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
#include "llvm/TableGen/Record.h"

using namespace llvm;

static std::string getEnumFeatures(const Record &Rec) {
  std::vector<Record *> Features = Rec.getValueAsListOfDefs("Features");
  if (find_if(Features, [](const Record *R) {
        return R->getName() == "Feature64Bit";
      }) != Features.end())
    return "FK_64BIT";

  return "FK_NONE";
}

void llvm::EmitRISCVTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  using MapTy = std::pair<const std::string, std::unique_ptr<llvm::Record>>;
  using RecordMap = std::map<std::string, std::unique_ptr<Record>, std::less<>>;
  const RecordMap &Map = RK.getDefs();

  OS << "#ifndef PROC\n"
     << "#define PROC(ENUM, NAME, FEATURES, DEFAULT_MARCH)\n"
     << "#endif\n\n";

  OS << "PROC(INVALID, {\"invalid\"}, FK_INVALID, {\"\"})\n";
  // Iterate on all definition records.
  for (const MapTy &Def : Map) {
    const Record &Rec = *(Def.second);
    if (Rec.isSubClassOf("RISCVProcessorModel"))
      OS << "PROC(" << Rec.getName() << ", "
         << "{\"" << Rec.getValueAsString("Name") << "\"},"
         << getEnumFeatures(Rec) << ", "
         << "{\"" << Rec.getValueAsString("DefaultMarch") << "\"})\n";
  }
  OS << "\n#undef PROC\n";
  OS << "\n";
  OS << "#ifndef TUNE_PROC\n"
     << "#define TUNE_PROC(ENUM, NAME)\n"
     << "#endif\n\n";
  OS << "TUNE_PROC(GENERIC, \"generic\")\n";
  for (const MapTy &Def : Map) {
    const Record &Rec = *(Def.second);
    if (Rec.isSubClassOf("RISCVTuneProcessorModel"))
      OS << "TUNE_PROC(" << Rec.getName() << ", "
         << "\"" << Rec.getValueAsString("Name") << "\")\n";
  }

  OS << "\n#undef TUNE_PROC\n";
}
