//===- RISCVTargetDefEmitter.cpp - Generate lists of RISC-V CPUs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits the include file needed by RISCVTargetParser.cpp
// and RISCVISAInfo.cpp to parse the RISC-V CPUs and extensions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/RISCVISAUtils.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

static StringRef getExtensionName(const Record *R) {
  StringRef Name = R->getValueAsString("Name");
  Name.consume_front("experimental-");
  return Name;
}

static void printExtensionTable(raw_ostream &OS,
                                ArrayRef<const Record *> Extensions,
                                bool Experimental) {
  OS << "static const RISCVSupportedExtension Supported";
  if (Experimental)
    OS << "Experimental";
  OS << "Extensions[] = {\n";

  for (const Record *R : Extensions) {
    if (R->getValueAsBit("Experimental") != Experimental)
      continue;

    OS.indent(4) << "{\"" << getExtensionName(R) << "\", {"
                 << R->getValueAsInt("MajorVersion") << ", "
                 << R->getValueAsInt("MinorVersion") << "}},\n";
  }

  OS << "};\n\n";
}

static void emitRISCVExtensions(const RecordKeeper &Records, raw_ostream &OS) {
  OS << "#ifdef GET_SUPPORTED_EXTENSIONS\n";
  OS << "#undef GET_SUPPORTED_EXTENSIONS\n\n";

  std::vector<const Record *> Extensions =
      Records.getAllDerivedDefinitionsIfDefined("RISCVExtension");
  llvm::sort(Extensions, [](const Record *Rec1, const Record *Rec2) {
    return getExtensionName(Rec1) < getExtensionName(Rec2);
  });

  if (!Extensions.empty()) {
    printExtensionTable(OS, Extensions, /*Experimental=*/false);
    printExtensionTable(OS, Extensions, /*Experimental=*/true);
  }

  OS << "#endif // GET_SUPPORTED_EXTENSIONS\n\n";

  OS << "#ifdef GET_IMPLIED_EXTENSIONS\n";
  OS << "#undef GET_IMPLIED_EXTENSIONS\n\n";

  if (!Extensions.empty()) {
    OS << "\nstatic constexpr ImpliedExtsEntry ImpliedExts[] = {\n";
    for (const Record *Ext : Extensions) {
      std::vector<const Record *> ImpliesList =
          Ext->getValueAsListOfDefs("Implies");
      if (ImpliesList.empty())
        continue;

      StringRef Name = getExtensionName(Ext);

      for (const Record *ImpliedExt : ImpliesList) {
        if (!ImpliedExt->isSubClassOf("RISCVExtension"))
          continue;

        OS.indent(4) << "{ {\"" << Name << "\"}, \""
                     << getExtensionName(ImpliedExt) << "\"},\n";
      }
    }

    OS << "};\n\n";
  }

  OS << "#endif // GET_IMPLIED_EXTENSIONS\n\n";
}

// We can generate march string from target features as what has been described
// in RISC-V ISA specification (version 20191213) 'Chapter 27. ISA Extension
// Naming Conventions'.
//
// This is almost the same as RISCVFeatures::parseFeatureBits, except that we
// get feature name from feature records instead of feature bits.
static void printMArch(raw_ostream &OS, ArrayRef<const Record *> Features) {
  RISCVISAUtils::OrderedExtensionMap Extensions;
  unsigned XLen = 0;

  // Convert features to FeatureVector.
  for (const Record *Feature : Features) {
    StringRef FeatureName = getExtensionName(Feature);
    if (Feature->isSubClassOf("RISCVExtension")) {
      unsigned Major = Feature->getValueAsInt("MajorVersion");
      unsigned Minor = Feature->getValueAsInt("MinorVersion");
      Extensions[FeatureName.str()] = {Major, Minor};
    } else if (FeatureName == "64bit") {
      assert(XLen == 0 && "Already determined XLen");
      XLen = 64;
    } else if (FeatureName == "32bit") {
      assert(XLen == 0 && "Already determined XLen");
      XLen = 32;
    }
  }

  assert(XLen != 0 && "Unable to determine XLen");

  OS << "rv" << XLen;

  ListSeparator LS("_");
  for (auto const &Ext : Extensions)
    OS << LS << Ext.first << Ext.second.Major << 'p' << Ext.second.Minor;
}

static void printProfileTable(raw_ostream &OS,
                              ArrayRef<const Record *> Profiles,
                              bool Experimental) {
  OS << "static constexpr RISCVProfile Supported";
  if (Experimental)
    OS << "Experimental";
  OS << "Profiles[] = {\n";

  for (const Record *Rec : Profiles) {
    if (Rec->getValueAsBit("Experimental") != Experimental)
      continue;

    StringRef Name = Rec->getValueAsString("Name");
    Name.consume_front("experimental-");
    OS.indent(4) << "{\"" << Name << "\",\"";
    printMArch(OS, Rec->getValueAsListOfDefs("Implies"));
    OS << "\"},\n";
  }

  OS << "};\n\n";
}

static void emitRISCVProfiles(const RecordKeeper &Records, raw_ostream &OS) {
  OS << "#ifdef GET_SUPPORTED_PROFILES\n";
  OS << "#undef GET_SUPPORTED_PROFILES\n\n";

  ArrayRef<const Record *> Profiles =
      Records.getAllDerivedDefinitionsIfDefined("RISCVProfile");

  if (!Profiles.empty()) {
    printProfileTable(OS, Profiles, /*Experimental=*/false);
    bool HasExperimentalProfiles = any_of(Profiles, [&](const Record *Rec) {
      return Rec->getValueAsBit("Experimental");
    });
    if (HasExperimentalProfiles)
      printProfileTable(OS, Profiles, /*Experimental=*/true);
  }

  OS << "#endif // GET_SUPPORTED_PROFILES\n\n";
}

static void emitRISCVProcs(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "#ifndef PROC\n"
     << "#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_SCALAR_UNALIGN"
     << ", FAST_VECTOR_UNALIGN, MVENDORID, MARCHID, MIMPID)\n"
     << "#endif\n\n";

  // Iterate on all definition records.
  for (const Record *Rec :
       RK.getAllDerivedDefinitionsIfDefined("RISCVProcessorModel")) {
    std::vector<const Record *> Features =
        Rec->getValueAsListOfDefs("Features");
    bool FastScalarUnalignedAccess =
        any_of(Features, [&](const Record *Feature) {
          return Feature->getValueAsString("Name") == "unaligned-scalar-mem";
        });

    bool FastVectorUnalignedAccess =
        any_of(Features, [&](const Record *Feature) {
          return Feature->getValueAsString("Name") == "unaligned-vector-mem";
        });

    OS << "PROC(" << Rec->getName() << ", {\"" << Rec->getValueAsString("Name")
       << "\"}, {\"";

    StringRef MArch = Rec->getValueAsString("DefaultMarch");

    // Compute MArch from features if we don't specify it.
    if (MArch.empty())
      printMArch(OS, Features);
    else
      OS << MArch;

    uint32_t MVendorID = Rec->getValueAsInt("MVendorID");
    uint64_t MArchID = Rec->getValueAsInt("MArchID");
    uint64_t MImpID = Rec->getValueAsInt("MImpID");

    OS << "\"}, " << FastScalarUnalignedAccess << ", "
       << FastVectorUnalignedAccess;
    OS << ", " << format_hex(MVendorID, 10);
    OS << ", " << format_hex(MArchID, 18);
    OS << ", " << format_hex(MImpID, 18);
    OS << ")\n";
  }
  OS << "\n#undef PROC\n";
  OS << "\n";
  OS << "#ifndef TUNE_PROC\n"
     << "#define TUNE_PROC(ENUM, NAME)\n"
     << "#endif\n\n";

  for (const Record *Rec :
       RK.getAllDerivedDefinitionsIfDefined("RISCVTuneProcessorModel")) {
    OS << "TUNE_PROC(" << Rec->getName() << ", "
       << "\"" << Rec->getValueAsString("Name") << "\")\n";
  }

  OS << "\n#undef TUNE_PROC\n";
}

static void emitRISCVExtensionBitmask(const RecordKeeper &RK, raw_ostream &OS) {
  std::vector<const Record *> Extensions =
      RK.getAllDerivedDefinitionsIfDefined("RISCVExtensionBitmask");
  llvm::sort(Extensions, [](const Record *Rec1, const Record *Rec2) {
    unsigned GroupID1 = Rec1->getValueAsInt("GroupID");
    unsigned GroupID2 = Rec2->getValueAsInt("GroupID");
    if (GroupID1 != GroupID2)
      return GroupID1 < GroupID2;

    return Rec1->getValueAsInt("BitPos") < Rec2->getValueAsInt("BitPos");
  });

#ifndef NDEBUG
  llvm::DenseSet<std::pair<uint64_t, uint64_t>> Seen;
#endif

  OS << "#ifdef GET_RISCVExtensionBitmaskTable_IMPL\n";
  OS << "static const RISCVExtensionBitmask ExtensionBitmask[]={\n";
  for (const Record *Rec : Extensions) {
    unsigned GroupIDVal = Rec->getValueAsInt("GroupID");
    unsigned BitPosVal = Rec->getValueAsInt("BitPos");

    StringRef ExtName = Rec->getValueAsString("Name");
    ExtName.consume_front("experimental-");

#ifndef NDEBUG
    assert(Seen.insert({GroupIDVal, BitPosVal}).second && "duplicated bitmask");
#endif

    OS.indent(4) << "{"
                 << "\"" << ExtName << "\""
                 << ", " << GroupIDVal << ", " << BitPosVal << "ULL"
                 << "},\n";
  }
  OS << "};\n";
  OS << "#endif\n\n";
}

static void emitRISCVTuneFeatures(const RecordKeeper &RK, raw_ostream &OS) {
  std::vector<const Record *> TuneFeatureRecords =
      RK.getAllDerivedDefinitionsIfDefined("RISCVTuneFeatureBase");

  // {Post Directive Idx, Neg Directive Idx, TuneFeature Record}
  SmallVector<std::tuple<unsigned, unsigned, const Record *>>
      TuneFeatureDirectives;
  // {Directive Idx -> Original Record}
  // This is primarily for diagnosing purposes -- when there is a duplication,
  // we are able to pointed out the previous definition.
  DenseMap<unsigned, const Record *> DirectiveToRecord;
  // A list of {Feature Name, Implied Feature Name}
  SmallVector<std::pair<StringRef, StringRef>> ImpliedFeatureList;
  StringToOffsetTable StrTable;

  auto tryInsertDirectives = [&](StringRef PosName, StringRef NegName,
                                 const Record *R) {
    unsigned PosIdx = StrTable.GetOrAddStringOffset(PosName);
    if (auto [ItEntry, Inserted] = DirectiveToRecord.try_emplace(PosIdx, R);
        !Inserted) {
      PrintError(R, "RISC-V tune feature positive directive '" +
                        Twine(PosName) + "' was already defined");
      PrintFatalNote(ItEntry->second, "Previously defined here");
    }
    unsigned NegIdx = StrTable.GetOrAddStringOffset(NegName);
    if (auto [ItEntry, Inserted] = DirectiveToRecord.try_emplace(NegIdx, R);
        !Inserted) {
      PrintError(R, "RISC-V tune feature negative directive '" +
                        Twine(NegName) + "' was already defined");
      PrintFatalNote(ItEntry->second, "Previously defined here");
    }

    TuneFeatureDirectives.emplace_back(PosIdx, NegIdx, R);
  };

  const std::string SimpleNegPrefix("no-");
  for (const auto *R : TuneFeatureRecords) {
    if (!R->isSubClassOf("SubtargetFeature"))
      PrintFatalError(
          R, "A RISC-V tune feature should also be a SubtargetFeature");
    // Preemptively insert feature name into the string table because we know
    // it will be used later.
    StringRef FeatureName = R->getValueAsString("Name");
    StrTable.GetOrAddStringOffset(FeatureName);
    if (R->isSubClassOf("RISCVSimpleTuneFeature")) {
      // The positive directve will be the feature name, and the negative
      // directive will be "no-" + feature name.
      std::string NegName = SimpleNegPrefix + FeatureName.str();
      tryInsertDirectives(FeatureName, NegName, R);
    } else if (R->isSubClassOf("RISCVTuneFeature")) {
      StringRef PosName = R->getValueAsString("PositiveDirectiveName");
      StringRef NegName = R->getValueAsString("NegativeDirectiveName");
      tryInsertDirectives(PosName, NegName, R);
    } else {
      llvm_unreachable("unrecognized RISCVTuneFeatureBase");
    }
  }

  for (const auto *R : TuneFeatureRecords) {
    std::vector<const Record *> Implies = R->getValueAsListOfDefs("Implies");
    for (const auto *ImpliedRecord : Implies) {
      if (!ImpliedRecord->isSubClassOf("RISCVTuneFeatureBase") ||
          ImpliedRecord == R) {
        PrintError(ImpliedRecord,
                   "A RISC-V tune feature can only imply another tune feature");
        PrintFatalNote(R, "implied by this tune feature");
      }
      StringRef CurrFeatureName = R->getValueAsString("Name");
      StringRef ImpliedFeatureName = ImpliedRecord->getValueAsString("Name");

      ImpliedFeatureList.emplace_back(CurrFeatureName, ImpliedFeatureName);
    }
  }

  OS << "#ifdef GET_TUNE_FEATURES\n";
  OS << "#undef GET_TUNE_FEATURES\n\n";

  StrTable.EmitStringTableDef(OS, "TuneFeatureStrings");
  OS << "\n";

  OS << "static constexpr RISCVTuneFeature TuneFeatures[] = {\n";
  for (const auto &[PosIdx, NegIdx, R] : TuneFeatureDirectives) {
    StringRef FeatureName = R->getValueAsString("Name");
    OS.indent(4) << formatv("{{ {0}, {1}, {2} },\t// '{3}'\n", PosIdx, NegIdx,
                            *StrTable.GetStringOffset(FeatureName),
                            FeatureName);
  }
  OS << "};\n\n";

  OS << "static constexpr RISCVImpliedTuneFeature ImpliedTuneFeatures[] = {\n";
  for (auto [Feature, ImpliedFeature] : ImpliedFeatureList)
    OS.indent(4) << formatv("{{ {0}, {1} }, // '{2}' -> '{3}'\n",
                            *StrTable.GetStringOffset(Feature),
                            *StrTable.GetStringOffset(ImpliedFeature), Feature,
                            ImpliedFeature);
  OS << "};\n\n";

  OS << "#endif // GET_TUNE_FEATURES\n";
}

static void emitRiscvTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  emitRISCVExtensions(RK, OS);
  emitRISCVProfiles(RK, OS);
  emitRISCVProcs(RK, OS);
  emitRISCVExtensionBitmask(RK, OS);
  emitRISCVTuneFeatures(RK, OS);
}

static TableGen::Emitter::Opt X("gen-riscv-target-def", emitRiscvTargetDef,
                                "Generate the list of CPUs and extensions for "
                                "RISC-V");
