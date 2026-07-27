//===- AMDGPUTargetDefEmitter.cpp - Generate lists of AMDGPU GPUs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits the AMDGPU GPU tables used by
// AMDGPUTargetParser.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

// Derive the GPUKind enum from a processor name, e.g. "gfx90a" -> "GK_GFX90A".
static void emitGPUKindEnum(raw_ostream &OS, StringRef Name) {
  OS << "GK_";
  for (char C : Name)
    OS << ((C == '-') ? '_' : toUpper(C));
}

// Derive the Triple::SubArchType from a "gfx..." GPU name, e.g. "gfx90a" ->
// Triple::AMDGPUSubArch90A, "gfx9-generic" -> Triple::AMDGPUSubArch9 (the
// family major). The name must be a real hardware GPU (not a pseudo target).
static void emitSubArchForName(raw_ostream &OS, StringRef Name) {
  StringRef Suffix = Name;
  Suffix.consume_front("gfx");
  Suffix.consume_back("-generic");

  OS << "Triple::AMDGPUSubArch";
  for (char C : Suffix)
    OS << ((C == '-') ? '_' : toUpper(C));
}

// Derive the Triple::SubArchType for a canonical GPU record. A pseudo target
// represents no hardware and maps to Triple::NoSubArch; otherwise the subarch
// is derived from the name.
static void emitSubArch(raw_ostream &OS, const Record *Rec) {
  if (Rec->getValueAsBit("IsPseudoTarget")) {
    OS << "Triple::NoSubArch";
    return;
  }
  emitSubArchForName(OS, Rec->getValueAsString("Name"));
}

// A canonical GPU record is a "gfxN-generic" family target if it covers a set
// of concrete GPUs (via CoveredGPUs) rather than being a single piece of
// hardware.
static bool isGenericTarget(const Record *Rec) {
  return !Rec->getValueAsListOfDefs("CoveredGPUs").empty();
}

// The gfx family for a canonical GPU record: the "-generic" family prefix (e.g.
// "gfx9-4-generic" -> "gfx9"), or the name with its last two chars dropped for
// a concrete GPU (e.g. "gfx90a" -> "gfx9", "gfx1030" -> "gfx10"). Empty for a
// pseudo target.
static StringRef getArchFamily(const Record *Rec) {
  if (Rec->getValueAsBit("IsPseudoTarget"))
    return "";
  StringRef Name = Rec->getValueAsString("Name");
  if (isGenericTarget(Rec))
    return Name.take_front(Name.find('-'));
  return Name.drop_back(2);
}

// Emit the ISA version tuple as "major, minor, stepping" wrapped in \p Open and
// \p Close (parens for the AMDGPU_GPU macro's ISAVERSION argument, braces for a
// struct initializer).
static void emitIsaVersion(raw_ostream &OS, const Record *Rec, char Open,
                           char Close) {
  std::vector<int64_t> V = Rec->getValueAsListOfInts("IsaVersion");
  if (V.size() != 3) {
    PrintFatalError(Rec->getLoc(),
                    "GPU '" + Rec->getValueAsString("Name") +
                        "' must have a 3-element [major, minor, stepping] "
                        "IsaVersion");
  }

  OS << Open << V[0] << ", " << V[1] << ", " << V[2] << Close;
}

// A canonical GPU or a ProcessorAlias.
namespace {
struct GPUEntry {
  const Record *Rec;
  bool IsAlias;

  // Whether this entry is (or aliases) a generic family target. \p Canonicals
  // maps canonical GPU names to their records.
  bool isGeneric(const StringMap<const Record *> &Canonicals) const {
    const Record *Canon =
        IsAlias ? Canonicals.lookup(Rec->getValueAsString("Alias")) : Rec;
    return Canon && isGenericTarget(Canon);
  }
};
} // namespace

// Emit the ArchFeature spellings joined with '|', or \p NoneSpelling when
// empty.
static void emitFeatureExpr(raw_ostream &OS, const Record *Rec,
                            StringRef NoneSpelling) {
  ListSeparator LS("|");
  bool Any = false;
  for (const Record *F : Rec->getValueAsListOfDefs("ArchFeatures")) {
    OS << LS << F->getValueAsString("Spelling");
    Any = true;
  }

  if (!Any)
    OS << NoneSpelling;
}

// Collect canonical GPUs and their aliases, in TableGen definition order. R600
// GPUs are plain Processor records; AMDGPU GPUs are ProcessorModel records (a
// Processor subclass), so \p WantR600 selects the family to emit.
static std::vector<GPUEntry> collectGPUs(const RecordKeeper &RK,
                                         bool WantR600) {
  ArrayRef<const Record *> GPUs = RK.getAllDerivedDefinitions("AMDGPUGPUInfo");
  std::vector<GPUEntry> Entries;
  Entries.reserve(GPUs.size());
  for (const Record *Rec : GPUs) {
    if (Rec->isSubClassOf("ProcessorModel") == WantR600)
      continue;
    Entries.push_back({Rec, /*IsAlias=*/false});
  }

  // Aliases only make sense when their canonical is present, so only gather
  // them for the family being emitted.
  if (!Entries.empty()) {
    for (const Record *Rec :
         RK.getAllDerivedDefinitionsIfDefined("ProcessorAlias"))
      Entries.push_back({Rec, /*IsAlias=*/true});
  }

  // Sort to preserve declaration order instead of name order.
  sort(Entries, [](const GPUEntry &A, const GPUEntry &B) {
    return A.Rec->getID() < B.Rec->getID();
  });

  return Entries;
}

// Check that every alias resolves to a canonical GPU and no name repeats.
static void validate(ArrayRef<GPUEntry> Entries) {
  StringMap<const Record *> Canonicals;
  for (const GPUEntry &E : Entries)
    if (!E.IsAlias)
      Canonicals[E.Rec->getValueAsString("Name")] = E.Rec;

  StringMap<const Record *> Seen;
  for (const GPUEntry &E : Entries) {
    StringRef Name = E.Rec->getValueAsString("Name");
    if (!Seen.insert({Name, E.Rec}).second) {
      PrintFatalError(E.Rec->getLoc(),
                      "duplicate AMDGPU processor name '" + Name + "'");
    }

    if (E.IsAlias) {
      StringRef Alias = E.Rec->getValueAsString("Alias");
      if (!Canonicals.count(Alias)) {
        PrintFatalError(E.Rec->getLoc(),
                        "ProcessorAlias '" + Name + "' aliases '" + Alias +
                            "' which is not a canonical AMDGPU GPU");
      }
    }
  }
}

static void emitR600(raw_ostream &OS, const RecordKeeper &RK) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/true);
  validate(Entries);
  if (Entries.empty())
    return;

  OS << "#ifndef R600_GPU\n"
        "#define R600_GPU(NAME, ENUM, FEATURES)\n"
        "#endif\n\n"
        "#ifndef R600_GPU_ALIAS\n"
        "#define R600_GPU_ALIAS(NAME, ENUM)\n"
        "#endif\n\n";

  for (const GPUEntry &E : Entries) {
    StringRef Name = E.Rec->getValueAsString("Name");
    if (E.IsAlias) {
      OS << "R600_GPU_ALIAS(\"" << Name << "\", ";
      emitGPUKindEnum(OS, E.Rec->getValueAsString("Alias"));
      OS << ")\n";
      continue;
    }
    OS << "R600_GPU(\"" << Name << "\", ";
    emitGPUKindEnum(OS, Name);
    OS << ", ";
    emitFeatureExpr(OS, E.Rec, "R600_FEATURE_NONE");
    OS << ")\n";
  }

  OS << "\n#undef R600_GPU\n"
        "#undef R600_GPU_ALIAS\n";
}

// Return \p Entries with the generic-family entries moved after the non-generic
// ones, each group keeping definition order. The GPUKind enum and GPUInfo table
// are positional and rely on the generics forming a contiguous block at the
// end, so both are emitted in this order.
static std::vector<GPUEntry>
orderGenericsLast(ArrayRef<GPUEntry> Entries,
                  const StringMap<const Record *> &Canonicals) {
  std::vector<GPUEntry> Ordered;
  Ordered.reserve(Entries.size());

  for (const GPUEntry &E : Entries) {
    if (!E.isGeneric(Canonicals))
      Ordered.push_back(E);
  }

  for (const GPUEntry &E : Entries) {
    if (E.isGeneric(Canonicals))
      Ordered.push_back(E);
  }

  return Ordered;
}

static void emitAMDGPUEntry(raw_ostream &OS, const GPUEntry &E) {
  StringRef Name = E.Rec->getValueAsString("Name");
  if (E.IsAlias) {
    OS << "AMDGPU_GPU_ALIAS(\"" << Name << "\", ";
    emitGPUKindEnum(OS, E.Rec->getValueAsString("Alias"));
    OS << ")\n";
  } else {
    OS << "AMDGPU_GPU(\"" << Name << "\", ";
    emitGPUKindEnum(OS, Name);
    OS << ")\n";
  }
}

static void emitAMDGPU(raw_ostream &OS, const RecordKeeper &RK) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/false);
  validate(Entries);
  if (Entries.empty())
    return;

  StringMap<const Record *> Canonicals;
  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias)
      Canonicals[E.Rec->getValueAsString("Name")] = E.Rec;
  }

  OS << "#ifndef AMDGPU_GPU\n"
        "#define AMDGPU_GPU(NAME, ENUM)\n"
        "#endif\n\n"
        "#ifndef AMDGPU_GPU_ALIAS\n"
        "#define AMDGPU_GPU_ALIAS(NAME, ENUM)\n"
        "#endif\n\n";

  for (const GPUEntry &E : orderGenericsLast(Entries, Canonicals))
    emitAMDGPUEntry(OS, E);

  OS << "\n#undef AMDGPU_GPU\n"
        "#undef AMDGPU_GPU_ALIAS\n";
}

/// Emit a GPUInfo table indexed by (GPUKind - AMDGPUFirstGPUKind). Name and
/// family strings are stored as offsets into the shared \p Names table.
static void emitAMDGPUTable(raw_ostream &OS, const RecordKeeper &RK,
                            StringToOffsetTable &Names) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/false);
  if (Entries.empty())
    return;

  StringMap<const Record *> Canonicals;
  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias)
      Canonicals[E.Rec->getValueAsString("Name")] = E.Rec;
  }

  // Canonicals only; aliases share a canonical's GPUKind row.
  std::vector<const Record *> Canon;
  for (const GPUEntry &E : orderGenericsLast(Entries, Canonicals)) {
    if (!E.IsAlias)
      Canon.push_back(E.Rec);
  }

  OS << "#ifdef GET_AMDGPU_GPU_TABLE\n"
        "#undef GET_AMDGPU_GPU_TABLE\n";
  OS << "static constexpr GPUKind AMDGPUFirstGPUKind = ";
  emitGPUKindEnum(OS, Canon.front()->getValueAsString("Name"));
  OS << ";\n"
        "static constexpr GPUInfo AMDGPUGPUTable[] = {\n";
  for (const Record *R : Canon) {
    StringRef Name = R->getValueAsString("Name");
    OS << "  {" << Names.GetOrAddStringOffset(Name) << ", ";
    emitSubArch(OS, R);
    OS << ", ";
    emitFeatureExpr(OS, R, "FEATURE_NONE");
    OS << ", ";
    emitIsaVersion(OS, R, '{', '}');
    OS << ", " << Names.GetOrAddStringOffset(getArchFamily(R)) << "},\n";
  }
  OS << "};\n"
        "#endif // GET_AMDGPU_GPU_TABLE\n\n";
}

// Emit the subarch -> major-family-subarch overrides; a subarch not listed here
// is its own major subarch. Each entry maps a member GPU's own subarch to its
// family's major subarch, from one of two sources: a "gfxN-generic" target,
// whose subarch is the major for every GPU it lists in CoveredGPUs, or an
// AMDGPUFamily's MajorSubArch for the gfx6/gfx7/gfx8 families that have no
// generic target.
static void emitAMDGPUMajorSubArch(raw_ostream &OS, const RecordKeeper &RK) {
  ArrayRef<const Record *> GPUs =
      RK.getAllDerivedDefinitionsIfDefined("AMDGPUGPUInfo");
  ArrayRef<const Record *> Families =
      RK.getAllDerivedDefinitionsIfDefined("AMDGPUFamily");

  // The overrides come from generic targets' CoveredGPUs and AMDGPUFamily
  // members. std::array makes the R600 case (zero entries) well-formed.
  size_t NumEntries = 0;
  for (const Record *G : GPUs)
    NumEntries += G->getValueAsListOfDefs("CoveredGPUs").size();
  for (const Record *F : Families)
    NumEntries += F->getValueAsListOfDefs("Members").size();

  OS << "#ifdef GET_AMDGPU_MAJOR_SUBARCH\n"
        "#undef GET_AMDGPU_MAJOR_SUBARCH\n"
        "struct AMDGPUMajorSubArchEntry {\n"
        "  Triple::SubArchType SubArch;\n"
        "  Triple::SubArchType Major;\n"
        "};\n"
        "static constexpr std::array<AMDGPUMajorSubArchEntry, "
     << NumEntries << "> AMDGPUMajorSubArch = {{\n";

  // A "gfxN-generic" target's subarch is the major for every GPU it covers.
  for (const Record *G : GPUs) {
    for (const Record *Member : G->getValueAsListOfDefs("CoveredGPUs")) {
      OS << "  {";
      emitSubArchForName(OS, Member->getValueAsString("Name"));
      OS << ", ";
      emitSubArch(OS, G);
      OS << "},\n";
    }
  }

  // The gfx6/gfx7/gfx8 families have no generic target, so their major comes
  // from AMDGPUFamily::MajorSubArch.
  for (const Record *F : Families) {
    StringRef Major = F->getValueAsString("MajorSubArch");
    for (const Record *Member : F->getValueAsListOfDefs("Members")) {
      OS << "  {";
      emitSubArchForName(OS, Member->getValueAsString("Name"));
      OS << ", Triple::AMDGPUSubArch" << Major << "},\n";
    }
  }

  OS << "}};\n"
        "#endif // GET_AMDGPU_MAJOR_SUBARCH\n\n";
}

// Emit the canonical GPU name for each AMDGPU subarch.
static void emitAMDGPUSubArchNames(raw_ostream &OS, const RecordKeeper &RK,
                                   StringToOffsetTable &Names) {
  // (subarch enumerator, name) pairs.
  std::vector<std::pair<std::string, StringRef>> Pairs;
  for (const GPUEntry &E : collectGPUs(RK, /*WantR600=*/false)) {
    if (E.IsAlias || E.Rec->getValueAsBit("IsPseudoTarget"))
      continue;
    StringRef Name = E.Rec->getValueAsString("Name");
    std::string SA;
    raw_string_ostream SO(SA);
    emitSubArchForName(SO, Name);
    Pairs.emplace_back(std::move(SA), Name);
  }
  for (const Record *F : RK.getAllDerivedDefinitionsIfDefined("AMDGPUFamily")) {
    std::vector<const Record *> Members = F->getValueAsListOfDefs("Members");
    Pairs.emplace_back("Triple::AMDGPUSubArch" +
                           F->getValueAsString("MajorSubArch").str(),
                       Members.front()->getValueAsString("Name"));
  }
  if (Pairs.empty())
    return;

  OS << "#ifdef GET_AMDGPU_SUBARCH_NAME\n"
        "#undef GET_AMDGPU_SUBARCH_NAME\n";
  OS << "struct AMDGPUSubArchNameEntry {\n"
        "  Triple::SubArchType SubArch;\n"
        "  StringTable::Offset NameOffset;\n"
        "};\n"
        "static constexpr AMDGPUSubArchNameEntry AMDGPUSubArchNames[] = {\n";
  for (const auto &[SubArch, Name] : Pairs)
    OS << "  {" << SubArch << ", " << Names.GetOrAddStringOffset(Name)
       << "},\n";
  OS << "};\n"
        "#endif // GET_AMDGPU_SUBARCH_NAME\n\n";
}

static void emitAMDGPUTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "// Autogenerated by AMDGPUTargetDefEmitter.cpp\n\n";
  // R600 processors are Processor records; AMDGPU processors are
  // ProcessorModel records. R600.td and AMDGPU.td are separate top-level files
  // (neither includes the other), so exactly one family is present in a given
  // run; the other section emits nothing.
  emitR600(OS, RK);
  emitAMDGPU(OS, RK);
  emitAMDGPUMajorSubArch(OS, RK);

  // The GPUInfo and SubArchName tables both reference GPU-name strings; pool
  // them into a single string table. Buffer the two tables first so every
  // referenced string is interned, then emit the shared table def (guarded so
  // consumers pull it in once) ahead of the buffered tables.
  StringToOffsetTable Names;
  std::string GPUTable, SubArchNames;
  raw_string_ostream GPUTableOS(GPUTable), SubArchNamesOS(SubArchNames);
  emitAMDGPUTable(GPUTableOS, RK, Names);
  emitAMDGPUSubArchNames(SubArchNamesOS, RK, Names);

  OS << "#ifdef GET_AMDGPU_NAME_TABLE\n"
        "#undef GET_AMDGPU_NAME_TABLE\n";
  Names.EmitStringTableDef(OS, "AMDGPUNameTable");
  OS << "#endif // GET_AMDGPU_NAME_TABLE\n\n";

  OS << GPUTable << SubArchNames;
}

static TableGen::Emitter::Opt X("gen-amdgpu-target-def", emitAMDGPUTargetDef,
                                "Generate the list of AMDGPU GPUs");
