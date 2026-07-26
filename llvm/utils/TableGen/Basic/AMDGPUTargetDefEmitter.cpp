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
#include "llvm/TableGen/TableGenBackend.h"
#include <vector>

using namespace llvm;

// Derive the GPUKind enum from a processor name, e.g. "gfx90a" -> "GK_GFX90A".
static void emitGPUKindEnum(raw_ostream &OS, StringRef Name) {
  OS << "GK_";
  for (char C : Name)
    OS << ((C == '-') ? '_' : toUpper(C));
}

/// Derive the Triple::SubArchType for a canonical GPU record.
static void emitSubArch(raw_ostream &OS, const Record *Rec) {
  if (Rec->getValueAsBit("IsPseudoTarget")) {
    OS << "Triple::NoSubArch";
    return;
  }

  StringRef Suffix = Rec->getValueAsString("Name");
  Suffix.consume_front("gfx");
  Suffix.consume_back("-generic");

  OS << "Triple::AMDGPUSubArch";
  for (char C : Suffix)
    OS << ((C == '-') ? '_' : toUpper(C));
}

/// The gfx family for a canonical GPU record: the "-generic" family prefix
/// (e.g.  "gfx9-4-generic" -> "gfx9"), or the name with its last two chars
/// dropped for a concrete GPU (e.g. "gfx90a" -> "gfx9", "gfx1030" ->
/// "gfx10"). Empty for a pseudo target.
static StringRef getArchFamily(const Record *Rec) {
  if (Rec->getValueAsBit("IsPseudoTarget"))
    return "";
  StringRef Name = Rec->getValueAsString("Name");
  if (Name.ends_with("-generic"))
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

// A canonical GPU record is a "gfxN-generic" family target if it covers a set
// of concrete GPUs (via CoveredGPUs) rather than being a single piece of
// hardware.
static bool isGenericTarget(const Record *Rec) {
  return !Rec->getValueAsListOfDefs("CoveredGPUs").empty();
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

  // The GPUKind enum is positional and code relies on the generic targets being
  // a contiguous block at the end, so emit all non-generic entries first, then
  // the generics, each group preserving TableGen definition order.
  for (const GPUEntry &E : Entries) {
    if (!E.isGeneric(Canonicals))
      emitAMDGPUEntry(OS, E);
  }

  for (const GPUEntry &E : Entries) {
    if (E.isGeneric(Canonicals))
      emitAMDGPUEntry(OS, E);
  }

  OS << "\n#undef AMDGPU_GPU\n"
        "#undef AMDGPU_GPU_ALIAS\n";
}

/// Emit a GPUInfo table indexed by (GPUKind - AMDGPUFirstGPUKind).
static void emitAMDGPUTable(raw_ostream &OS, const RecordKeeper &RK) {
  std::vector<GPUEntry> Entries = collectGPUs(RK, /*WantR600=*/false);
  if (Entries.empty())
    return;

  // Canonicals only (aliases share a canonical's GPUKind row), in the same
  // non-generic-then-generic order as the GPUKind enum.
  std::vector<const Record *> Canon;
  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias && !isGenericTarget(E.Rec))
      Canon.push_back(E.Rec);
  }

  for (const GPUEntry &E : Entries) {
    if (!E.IsAlias && isGenericTarget(E.Rec))
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
    OS << "  {\"" << Name << "\", ";
    emitSubArch(OS, R);
    OS << ", ";
    emitFeatureExpr(OS, R, "FEATURE_NONE");
    OS << ", ";
    emitIsaVersion(OS, R, '{', '}');
    OS << ", \"" << getArchFamily(R) << "\"},\n";
  }
  OS << "};\n"
        "#endif // GET_AMDGPU_GPU_TABLE\n\n";
}

static void emitAMDGPUTargetDef(const RecordKeeper &RK, raw_ostream &OS) {
  OS << "// Autogenerated by AMDGPUTargetDefEmitter.cpp\n\n";
  // R600 processors are Processor records; AMDGPU processors are
  // ProcessorModel records. R600.td and AMDGPU.td are separate top-level files
  // (neither includes the other), so exactly one family is present in a given
  // run; the other section emits nothing.
  emitR600(OS, RK);
  emitAMDGPU(OS, RK);
  emitAMDGPUTable(OS, RK);
}

static TableGen::Emitter::Opt X("gen-amdgpu-target-def", emitAMDGPUTargetDef,
                                "Generate the list of AMDGPU GPUs");
