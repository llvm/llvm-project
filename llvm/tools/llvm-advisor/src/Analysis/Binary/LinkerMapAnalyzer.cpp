//===--- LinkerMapAnalyzer.cpp - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/LinkerMapAnalyzer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// ---- Map-file discovery -------------------------------------------------

static std::string findMapFile(const CapabilityContext &Context) {
  // Derive base name from the object path (e.g. "foo.o" → "foo.map").
  SmallVector<std::string, 8> Candidates;
  if (!Context.ObjectPath.empty()) {
    SmallString<256> Base(Context.ObjectPath);
    sys::path::replace_extension(Base, "map");
    Candidates.push_back(Base.str().str());

    // Also try stripping the last extension and appending .map.
    StringRef Stem = sys::path::stem(Context.ObjectPath);
    SmallString<256> Dir(sys::path::parent_path(Context.ObjectPath));
    sys::path::append(Dir, Stem + ".map");
    Candidates.push_back(Dir.str().str());
  }

  if (!Context.WorkingDirectory.empty()) {
    SmallString<256> WD(Context.WorkingDirectory);
    for (StringRef Name :
         {"output.map", "a.out.map", "linker.map", "link.map"}) {
      SmallString<256> P(WD);
      sys::path::append(P, Name);
      Candidates.push_back(P.str().str());
    }
  }

  for (const std::string &C : Candidates)
    if (!C.empty() && sys::fs::exists(C))
      return C;

  // Fall back: scan the working directory for any *.map file.
  if (!Context.WorkingDirectory.empty()) {
    std::error_code EC;
    for (sys::fs::directory_iterator It(Context.WorkingDirectory, EC), End;
         !EC && It != End; It.increment(EC)) {
      StringRef P = It->path();
      if (P.ends_with(".map"))
        return P.str();
    }
  }
  return {};
}

// ---- Format detection ---------------------------------------------------

enum class MapFormat { LLD, GNUld, Unknown };

static MapFormat detectFormat(StringRef Content) {
  // LLD map files begin with the column header line.
  if (Content.starts_with("VMA") || Content.starts_with("          VMA"))
    return MapFormat::LLD;
  // GNU ld map files contain a well-known section header.
  if (Content.contains("Linker script and memory map") ||
      Content.contains("Memory Configuration"))
    return MapFormat::GNUld;
  return MapFormat::Unknown;
}

// ---- Shared helpers -----------------------------------------------------

static bool parseHex(StringRef S, uint64_t &Out) {
  return !S.getAsInteger(16, Out);
}

// ---- LLD map parser -----------------------------------------------------
//
// LLD --print-map columns (space-separated, fixed-width):
//   VMA              LMA              Size             Align Out/In/File/Symbol
//
// The last token on a line (after the 4 fixed numeric columns) is the name.
// Hierarchy is encoded by the number of leading spaces before the name:
//   0 extra spaces  → output section  (.text, .data …)
//   9 extra spaces  → input section
//   18 extra spaces → object file
//   27 extra spaces → symbol
//
// We only care about output sections and symbols for aggregation.

struct SymEntry {
  std::string Name;
  uint64_t Size;
  std::string Section;
};

static json::Value parseLLD(StringRef Content) {
  struct SectionEntry {
    std::string Name;
    uint64_t VMA;
    uint64_t Size;
    uint64_t Align;
    SmallVector<SymEntry, 16> Symbols;
  };

  SmallVector<SectionEntry, 32> Sections;
  SectionEntry *Current = nullptr;

  SmallVector<StringRef, 4> Lines;
  Content.split(Lines, '\n');

  for (StringRef Raw : Lines) {
    // Skip the header and blank lines.
    if (Raw.trim().empty() || Raw.starts_with("VMA") ||
        Raw.starts_with("          VMA"))
      continue;

    // The first four tokens are hex/dec numbers.
    StringRef Line = Raw;
    auto nextTok = [&]() -> StringRef {
      Line = Line.ltrim(' ');
      auto Pos = Line.find(' ');
      StringRef T = Line.take_front(Pos == StringRef::npos ? Line.size() : Pos);
      Line = Line.drop_front(T.size());
      return T;
    };

    StringRef VMAStr = nextTok();
    StringRef LMAStr = nextTok();
    StringRef SizeStr = nextTok();
    StringRef AlignStr = nextTok();

    uint64_t VMA, LMA, Size, Align;
    if (!parseHex(VMAStr, VMA) || !parseHex(LMAStr, LMA) ||
        !parseHex(SizeStr, Size) || AlignStr.getAsInteger(10, Align))
      continue;

    // Everything remaining (after leading spaces) is the name.
    StringRef NameField = Line; // still has leading spaces
    size_t NameStart = NameField.find_first_not_of(' ');
    if (NameStart == StringRef::npos)
      continue;

    // Count leading spaces to determine hierarchy level.
    StringRef Name = NameField.drop_front(NameStart).trim();
    if (Name.empty())
      continue;

    // Leading spaces before name (in the fixed-width layout after the 4 cols):
    // LLD uses 4-space indentation per level.  Anything with ≤3 leading spaces
    // (i.e. effectively 0 indentation after the columns) is an output section.
    if (NameStart < 4) {
      // Output section.
      SectionEntry SE;
      SE.Name = Name.str();
      SE.VMA = VMA;
      SE.Size = Size;
      SE.Align = Align;
      Sections.push_back(std::move(SE));
      Current = &Sections.back();
    } else if (Current && NameStart >= 12 &&
               (Name.starts_with('_') || !Name.starts_with('.'))) {
      // Heuristic: deeply-indented non-section names are symbols.
      SymEntry SE;
      SE.Name = Name.str();
      SE.Size = Size;
      SE.Section = Current->Name;
      Current->Symbols.push_back(std::move(SE));
    }
    (void)LMA; // not reported
  }

  // Build output JSON.
  json::Array SectionsArr;
  int64_t TotalSize = 0;
  for (const SectionEntry &S : Sections) {
    json::Object Obj;
    Obj["name"] = S.Name;
    Obj["vma"] = static_cast<int64_t>(S.VMA);
    Obj["size"] = static_cast<int64_t>(S.Size);
    Obj["align"] = static_cast<int64_t>(S.Align);
    TotalSize += static_cast<int64_t>(S.Size);
    SectionsArr.push_back(std::move(Obj));
  }

  // Collect all symbols, sort by size descending, take top 50.
  SmallVector<const SymEntry *, 64> AllSyms;
  for (const SectionEntry &S : Sections)
    for (const SymEntry &Sym : S.Symbols)
      AllSyms.push_back(&Sym);

  llvm::sort(AllSyms, [](const SymEntry *A, const SymEntry *B) {
    return A->Size > B->Size;
  });

  json::Array SymsArr;
  for (size_t I = 0, E = std::min<size_t>(AllSyms.size(), 50); I < E; ++I) {
    const SymEntry &Sym = *AllSyms[I];
    SymsArr.push_back(json::Object{
        {"name", Sym.Name},
        {"size", static_cast<int64_t>(Sym.Size)},
        {"section", Sym.Section},
    });
  }

  return json::Object{
      {"format", "lld"},
      {"total_mapped_bytes", TotalSize},
      {"section_count", static_cast<int64_t>(Sections.size())},
      {"sections", std::move(SectionsArr)},
      {"top_symbols_by_size", std::move(SymsArr)},
  };
}

// ---- GNU ld map parser --------------------------------------------------
//
// GNU ld map files have a free-form text layout inside
// "Linker script and memory map".  Each output section starts with a line
// like ".text           0x0000000000001000   0x1234"
// followed by input contributions and symbol assignments.

static json::Value parseGNUld(StringRef Content) {
  // Find the layout section.
  size_t Start = Content.find("Linker script and memory map");
  StringRef Body =
      Start != StringRef::npos ? Content.drop_front(Start) : Content;

  struct SectionEntry {
    std::string Name;
    uint64_t VMA;
    uint64_t Size;
  };

  SmallVector<SectionEntry, 32> Sections;
  SmallVector<SymEntry, 64> Symbols;
  std::string CurrentSection;

  SmallVector<StringRef, 4> Lines;
  Body.split(Lines, '\n');

  for (StringRef Raw : Lines) {
    StringRef Line = Raw.trim();
    if (Line.empty() || Line.starts_with('*') || Line.starts_with('/'))
      continue;

    // Section line: starts with '.' followed by non-whitespace.
    // Pattern: .name     0xADDR    0xSIZE
    if (Line.starts_with('.') && !Line.starts_with(".(")) {
      SmallVector<StringRef, 4> Tokens;
      Line.split(Tokens, ' ', -1, /*KeepEmpty=*/false);
      if (Tokens.size() >= 3) {
        uint64_t Addr, Size;
        if (parseHex(Tokens[1], Addr) && parseHex(Tokens[2], Size) &&
            Size > 0) {
          SectionEntry SE;
          SE.Name = Tokens[0].str();
          SE.VMA = Addr;
          SE.Size = Size;
          Sections.push_back(SE);
          CurrentSection = SE.Name;
          continue;
        }
      }
    }

    // Symbol assignment: "    0xADDR    symbolname"
    if (Line.starts_with("0x")) {
      SmallVector<StringRef, 4> Tokens;
      Line.split(Tokens, ' ', -1, /*KeepEmpty=*/false);
      if (Tokens.size() >= 2) {
        uint64_t Addr;
        if (parseHex(Tokens[0], Addr)) {
          StringRef Sym = Tokens[1];
          if (!Sym.starts_with("0x") && !Sym.empty()) {
            SymEntry SE;
            SE.Name = Sym.str();
            SE.Size = 0; // GNU ld map doesn't always give symbol sizes here
            SE.Section = CurrentSection;
            Symbols.push_back(std::move(SE));
          }
        }
      }
    }
  }

  json::Array SectionsArr;
  int64_t TotalSize = 0;
  for (const SectionEntry &S : Sections) {
    SectionsArr.push_back(json::Object{
        {"name", S.Name},
        {"vma", static_cast<int64_t>(S.VMA)},
        {"size", static_cast<int64_t>(S.Size)},
    });
    TotalSize += static_cast<int64_t>(S.Size);
  }

  // GNU ld symbols generally lack size info — report just the count.
  return json::Object{
      {"format", "gnu_ld"},
      {"total_mapped_bytes", TotalSize},
      {"section_count", static_cast<int64_t>(Sections.size())},
      {"sections", std::move(SectionsArr)},
      {"symbol_count", static_cast<int64_t>(Symbols.size())},
  };
}

// ---- Runner -------------------------------------------------------------

Expected<std::unique_ptr<CapabilityResult>>
LinkerMapAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string MapPath = findMapFile(Context);
  if (MapPath.empty())
    return makeUnavailableResult(CapID, UnitID,
                                 "no linker map file found");

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(MapPath);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "cannot read map file: %s",
                             MapPath.c_str());

  StringRef Content = (*BufOrErr)->getBuffer();
  MapFormat Fmt = detectFormat(Content);

  json::Value Parsed = Fmt == MapFormat::LLD ? parseLLD(Content)
                       : Fmt == MapFormat::GNUld
                           ? parseGNUld(Content)
                           : json::Object{{"format", "unknown"}};

  json::Object *PObj = Parsed.getAsObject();
  if (PObj)
    (*PObj)["map_path"] = MapPath;

  return std::make_unique<JSONCapabilityResult>(std::move(Parsed));
}

} // namespace llvm::advisor
