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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
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

// Emit the Triple::AMDGPUSubArch enumerator suffix for a "gfx..." GPU name,
// e.g. "gfx90a" -> "90A", "gfx9-generic" -> "9" (the family major).
static void emitSubArchSuffix(raw_ostream &OS, StringRef Name) {
  StringRef Suffix = Name;
  Suffix.consume_front("gfx");
  Suffix.consume_back("-generic");

  for (char C : Suffix)
    OS << static_cast<char>((C == '-') ? '_' : toUpper(C));
}

/// Derive the Triple::SubArchType from a "gfx..." GPU name, e.g. "gfx90a" ->
/// Triple::AMDGPUSubArch90A
static void emitSubArchForName(raw_ostream &OS, StringRef Name) {
  OS << "Triple::AMDGPUSubArch";
  emitSubArchSuffix(OS, Name);
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

// Emit the triple subarch name for a concrete GPU, e.g. gfx90c / [9, 0, 12] ->
// "amdgpu9.0c" (stepping is a single lowercase hex digit).
static void emitConcreteSubArchTripleName(raw_ostream &OS, const Record *Rec) {
  std::vector<int64_t> V = Rec->getValueAsListOfInts("IsaVersion");

  // Assuming emitIsaVersion validated the number of elements.
  if (V[2] < 0 || V[2] > 15) {
    PrintFatalError(Rec->getLoc(), "GPU '" + Rec->getValueAsString("Name") +
                                       "' stepping must be a single hex digit");
  }

  OS << "amdgpu" << V[0] << '.' << V[1] << hexdigit(V[2], /*LowerCase=*/true);
}

// Emit the triple subarch name for a major-family subarch, e.g. "9" ->
// "amdgpu9", "9_4" -> "amdgpu9.4" (the enumerator suffix uses '_', the triple
// name '.').
static void emitFamilySubArchTripleName(raw_ostream &OS, StringRef Suffix) {
  OS << "amdgpu";
  for (char C : Suffix)
    OS << static_cast<char>((C == '_') ? '.' : C);
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

// The canonical R600 GPU records, in GPUKind-enum / TableGen definition order.
static std::vector<const Record *>
collectR600Canonicals(const RecordKeeper &RK) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/true);
  std::vector<const Record *> Canon;
  Canon.reserve(Entries.size());

  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias)
      Canon.push_back(E.Rec);
  }

  return Canon;
}

// Emit the R600 GPUKind enumerators (canonical GPUs only; aliases share a
// canonical's kind). Guarded by GET_R600_GPU_ENUM.
static void emitR600Enum(raw_ostream &OS, const RecordKeeper &RK) {
  std::vector<const Record *> Canon = collectR600Canonicals(RK);
  if (Canon.empty())
    return;
  OS << "#ifdef GET_R600_GPU_ENUM\n"
        "#undef GET_R600_GPU_ENUM\n";
  for (const Record *R : Canon) {
    OS << "  ";
    emitGPUKindEnum(OS, R->getValueAsString("Name"));
    OS << ",\n";
  }
  OS << "#endif // GET_R600_GPU_ENUM\n\n";
}

// Emit the R600Info table indexed by (GPUKind - R600FirstGPUKind). Names are
// offsets into the shared \p Names table. Guarded by GET_R600_GPU_TABLE.
static void emitR600Table(raw_ostream &OS, const RecordKeeper &RK,
                          StringToOffsetTable &Names) {
  std::vector<const Record *> Canon = collectR600Canonicals(RK);
  if (Canon.empty())
    return;

  OS << "#ifdef GET_R600_GPU_TABLE\n"
        "#undef GET_R600_GPU_TABLE\n";
  OS << "static constexpr GPUKind R600FirstGPUKind = ";
  emitGPUKindEnum(OS, Canon.front()->getValueAsString("Name"));
  OS << ";\n"
        "static constexpr R600Info R600GPUTable[] = {\n";
  for (const Record *R : Canon) {
    OS << "  {" << Names.GetOrAddStringOffset(R->getValueAsString("Name"))
       << ", ";
    emitFeatureExpr(OS, R, "R600_FEATURE_NONE");
    OS << "},\n";
  }
  OS << "};\n"
        "#endif // GET_R600_GPU_TABLE\n\n";
}

// Emit the R600 name -> GPUKind alias table. Guarded by
// GET_R600_GPU_ALIAS_TABLE; names are offsets into \p Names.
static void emitR600Aliases(raw_ostream &OS, const RecordKeeper &RK,
                            StringToOffsetTable &Names) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/true);
  validate(Entries);
  if (Entries.empty())
    return;

  OS << "#ifdef GET_R600_GPU_ALIAS_TABLE\n"
        "#undef GET_R600_GPU_ALIAS_TABLE\n"
        "static constexpr GPUNameAlias R600GPUAliases[] = {\n";
  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias)
      continue;
    OS << "  {" << Names.GetOrAddStringOffset(E.Rec->getValueAsString("Name"))
       << ", ";
    emitGPUKindEnum(OS, E.Rec->getValueAsString("Alias"));
    OS << "},\n";
  }
  OS << "};\n"
        "#endif // GET_R600_GPU_ALIAS_TABLE\n\n";
}

// Canonical AMDGPU GPUs in GPUKind-enum order: non-generic targets first, then
// the "gfxN-generic" targets. The enum and the GPUInfo table share this order.
static std::vector<const Record *>
collectAMDGPUCanonicals(const RecordKeeper &RK) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/false);
  std::vector<const Record *> Canon;
  Canon.reserve(Entries.size());

  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias && !isGenericTarget(E.Rec))
      Canon.push_back(E.Rec);
  }

  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias && isGenericTarget(E.Rec))
      Canon.push_back(E.Rec);
  }

  return Canon;
}

// Emit the AMDGPU GPUKind enumerators (canonical GPUs only; aliases share a
// canonical's kind). Guarded by GET_AMDGPU_GPU_ENUM.
static void emitAMDGPUEnum(raw_ostream &OS, const RecordKeeper &RK) {
  std::vector<const Record *> Canon = collectAMDGPUCanonicals(RK);
  if (Canon.empty())
    return;
  OS << "#ifdef GET_AMDGPU_GPU_ENUM\n"
        "#undef GET_AMDGPU_GPU_ENUM\n";
  for (const Record *R : Canon) {
    OS << "  ";
    emitGPUKindEnum(OS, R->getValueAsString("Name"));
    OS << ",\n";
  }
  OS << "#endif // GET_AMDGPU_GPU_ENUM\n\n";
}

// Emit the name -> GPUKind alias table (legacy names such as "tahiti" ->
// gfx600). Guarded by GET_AMDGPU_GPU_ALIAS_TABLE; names are offsets into \p
// Names.
static void emitAMDGPUAliases(raw_ostream &OS, const RecordKeeper &RK,
                              StringToOffsetTable &Names) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/false);
  validate(Entries);
  if (Entries.empty())
    return;

  OS << "#ifdef GET_AMDGPU_GPU_ALIAS_TABLE\n"
        "#undef GET_AMDGPU_GPU_ALIAS_TABLE\n"
        "static constexpr GPUNameAlias AMDGPUGPUAliases[] = {\n";
  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias)
      continue;
    OS << "  {" << Names.GetOrAddStringOffset(E.Rec->getValueAsString("Name"))
       << ", ";
    emitGPUKindEnum(OS, E.Rec->getValueAsString("Alias"));
    OS << "},\n";
  }
  OS << "};\n"
        "#endif // GET_AMDGPU_GPU_ALIAS_TABLE\n\n";
}

/// Emit a GPUInfo table indexed by (GPUKind - AMDGPUFirstGPUKind). Name and
/// family strings are stored as offsets into the shared \p Names table.
static void emitAMDGPUTable(raw_ostream &OS, const RecordKeeper &RK,
                            StringToOffsetTable &Names) {
  std::vector<const Record *> Canon = collectAMDGPUCanonicals(RK);
  if (Canon.empty())
    return;

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

// Emit the subarch -> major-family-subarch overrides for getMajorSubArch (a
// subarch not listed here is its own major). Each member GPU maps to its
// family's major, sourced from a "gfxN-generic" target's CoveredGPUs, or from
// an AMDGPUFamily's MajorSubArch for the gfx6/gfx7/gfx8 families that have no
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

/// Emit the canonical GPU name for each AMDGPU subarch ("gfx900"), and it's
/// corresponding subarch ("amdgpu9.00")
static void emitAMDGPUSubArchNames(raw_ostream &OS, const RecordKeeper &RK,
                                   StringToOffsetTable &Names) {
  // A row of the generated table. \p Suffix is emitted verbatim after
  // "Triple::AMDGPUSubArch"; the two name offsets index the shared string pool.
  struct SubArchEntry {
    SmallString<16> Suffix;
    StringRef GPUName; // e.g. "gfx900".
    unsigned TripleNameOffset;
  };
  std::vector<SubArchEntry> Entries;

  for (const GPUEntry &E : collectGPUs(RK, /*WantR600=*/false)) {
    if (E.IsAlias || E.Rec->getValueAsBit("IsPseudoTarget"))
      continue;
    SubArchEntry Entry;
    Entry.GPUName = E.Rec->getValueAsString("Name");
    {
      raw_svector_ostream SubArchOS(Entry.Suffix);
      emitSubArchSuffix(SubArchOS, Entry.GPUName);
    }

    // A "gfxN-generic" target maps to the major-family subarch, so it takes the
    // family triple name; a concrete GPU derives it from the ISA version.
    SmallString<16> TripleName;
    raw_svector_ostream TripleOS(TripleName);
    if (isGenericTarget(E.Rec))
      emitFamilySubArchTripleName(TripleOS, Entry.Suffix);
    else
      emitConcreteSubArchTripleName(TripleOS, E.Rec);
    Entry.TripleNameOffset = Names.GetOrAddStringOffset(TripleName);

    Entries.push_back(std::move(Entry));
  }

  for (const Record *F : RK.getAllDerivedDefinitionsIfDefined("AMDGPUFamily")) {
    std::vector<const Record *> Members = F->getValueAsListOfDefs("Members");
    StringRef Major = F->getValueAsString("MajorSubArch");
    SubArchEntry Entry;
    Entry.Suffix = Major;
    Entry.GPUName = Members.front()->getValueAsString("Name");

    SmallString<16> TripleName;
    raw_svector_ostream TripleOS(TripleName);
    emitFamilySubArchTripleName(TripleOS, Major);
    Entry.TripleNameOffset = Names.GetOrAddStringOffset(TripleName);

    Entries.push_back(std::move(Entry));
  }

  if (Entries.empty())
    return;

  unsigned NoSubArchOffset = Names.GetOrAddStringOffset("amdgpu");

  OS << "#ifdef GET_AMDGPU_SUBARCH_NAME\n"
        "#undef GET_AMDGPU_SUBARCH_NAME\n";
  OS << "static constexpr StringTable::Offset AMDGPUNoSubArchNameOffset = "
     << NoSubArchOffset << ";\n";
  OS << "struct AMDGPUSubArchNameEntry {\n"
        "  Triple::SubArchType SubArch;\n"
        "  StringTable::Offset NameOffset;\n"
        "  StringTable::Offset TripleNameOffset;\n"
        "};\n"
        "static constexpr AMDGPUSubArchNameEntry AMDGPUSubArchNames[] = {\n";
  for (const SubArchEntry &E : Entries)
    OS << "  {Triple::AMDGPUSubArch" << E.Suffix << ", "
       << Names.GetOrAddStringOffset(E.GPUName) << ", " << E.TripleNameOffset
       << "},\n";
  OS << "};\n"
        "#endif // GET_AMDGPU_SUBARCH_NAME\n\n";
}

static void emitAMDGPUTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "// Autogenerated by AMDGPUTargetDefEmitter.cpp\n\n";
  // R600.td and AMDGPU.td are separate top-level files, so a run sees exactly
  // one family; the other family's sections emit nothing.
  emitR600Enum(OS, RK);
  emitAMDGPUEnum(OS, RK);
  emitAMDGPUMajorSubArch(OS, RK);

  // Each family gets its own string pool with a distinct guard/symbol so the
  // two generated headers stay independent when a consumer includes both.
  // Buffer the tables first to intern their strings, then emit the pool ahead.
  {
    StringToOffsetTable Names;
    std::string Tables;
    raw_string_ostream TablesOS(Tables);
    emitR600Table(TablesOS, RK, Names);
    emitR600Aliases(TablesOS, RK, Names);
    if (!Tables.empty()) {
      OS << "#ifdef GET_R600_NAME_TABLE\n"
            "#undef GET_R600_NAME_TABLE\n";
      Names.EmitStringTableDef(OS, "R600NameTable");
      OS << "#endif // GET_R600_NAME_TABLE\n\n";
      OS << Tables;
    }
  }

  {
    StringToOffsetTable Names;
    std::string Tables;
    raw_string_ostream TablesOS(Tables);
    emitAMDGPUTable(TablesOS, RK, Names);
    emitAMDGPUAliases(TablesOS, RK, Names);
    emitAMDGPUSubArchNames(TablesOS, RK, Names);
    if (!Tables.empty()) {
      OS << "#ifdef GET_AMDGPU_NAME_TABLE\n"
            "#undef GET_AMDGPU_NAME_TABLE\n";
      Names.EmitStringTableDef(OS, "AMDGPUNameTable");
      OS << "#endif // GET_AMDGPU_NAME_TABLE\n\n";
      OS << Tables;
    }
  }
}

static TableGen::Emitter::Opt X("gen-amdgpu-target-def", emitAMDGPUTargetDef,
                                "Generate the list of AMDGPU GPUs");
