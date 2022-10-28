//===-- LVBinaryReader.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVBinaryReader class, which is used to describe a
// binary reader.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVBINARYREADER_H
#define LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVBINARYREADER_H

#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"

namespace llvm {
namespace logicalview {

constexpr bool UpdateHighAddress = false;

// Logical scope, Section address, Section index, IsComdat.
struct LVSymbolTableEntry final {
  LVScope *Scope = nullptr;
  LVAddress Address = 0;
  LVSectionIndex SectionIndex = 0;
  bool IsComdat = false;
  LVSymbolTableEntry() = default;
  LVSymbolTableEntry(LVScope *Scope, LVAddress Address,
                     LVSectionIndex SectionIndex, bool IsComdat)
      : Scope(Scope), Address(Address), SectionIndex(SectionIndex),
        IsComdat(IsComdat) {}
};

// Function names extracted from the object symbol table.
class LVSymbolTable final {
  using LVSymbolNames = std::map<std::string, LVSymbolTableEntry>;
  LVSymbolNames SymbolNames;

public:
  LVSymbolTable() = default;

  void add(StringRef Name, LVScope *Function, LVSectionIndex SectionIndex = 0);
  void add(StringRef Name, LVAddress Address, LVSectionIndex SectionIndex,
           bool IsComdat);
  LVSectionIndex update(LVScope *Function);

  const LVSymbolTableEntry &getEntry(StringRef Name);
  LVAddress getAddress(StringRef Name);
  LVSectionIndex getIndex(StringRef Name);
  bool getIsComdat(StringRef Name);

  void print(raw_ostream &OS);
};

class LVBinaryReader : public LVReader {
  // Function names extracted from the object symbol table.
  LVSymbolTable SymbolTable;

  // Instruction lines for a logical scope. These instructions are fetched
  // during its merge with the debug lines.
  LVDoubleMap<LVSectionIndex, LVScope *, LVLines *> ScopeInstructions;

  // Links the scope with its first assembler address line.
  LVDoubleMap<LVSectionIndex, LVAddress, LVScope *> AssemblerMappings;

  // Mapping from virtual address to section.
  // The virtual address refers to the address where the section is loaded.
  using LVSectionAddresses = std::map<LVSectionIndex, object::SectionRef>;
  LVSectionAddresses SectionAddresses;

  void addSectionAddress(const object::SectionRef &Section) {
    if (SectionAddresses.find(Section.getAddress()) == SectionAddresses.end())
      SectionAddresses.emplace(Section.getAddress(), Section);
  }

  // Scopes with ranges for current compile unit. It is used to find a line
  // giving its exact or closest address. To support comdat functions, all
  // addresses for the same section are recorded in the same map.
  using LVSectionRanges = std::map<LVSectionIndex, LVRange *>;
  LVSectionRanges SectionRanges;

  // Image base and virtual address for Executable file.
  uint64_t ImageBaseAddress = 0;
  uint64_t VirtualAddress = 0;

  // Object sections with machine code.
  using LVSections = std::map<LVSectionIndex, object::SectionRef>;
  LVSections Sections;

protected:
  // It contains the LVLineDebug elements representing the logical lines for
  // the current compile unit, created by parsing the debug line section.
  LVLines CULines;

  std::unique_ptr<const MCRegisterInfo> MRI;
  std::unique_ptr<const MCAsmInfo> MAI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<const MCDisassembler> MD;
  std::unique_ptr<MCContext> MC;
  std::unique_ptr<MCInstPrinter> MIP;

  // Loads all info for the architecture of the provided object file.
  Error loadGenericTargetInfo(StringRef TheTriple, StringRef TheFeatures);

  virtual void mapRangeAddress(const object::ObjectFile &Obj) {}
  virtual void mapRangeAddress(const object::ObjectFile &Obj,
                               const object::SectionRef &Section,
                               bool IsComdat) {}

  // Create a mapping from virtual address to section.
  void mapVirtualAddress(const object::ObjectFile &Obj);
  void mapVirtualAddress(const object::COFFObjectFile &COFFObj);

  Expected<std::pair<LVSectionIndex, object::SectionRef>>
  getSection(LVScope *Scope, LVAddress Address, LVSectionIndex SectionIndex);

  void addSectionRange(LVSectionIndex SectionIndex, LVScope *Scope);
  void addSectionRange(LVSectionIndex SectionIndex, LVScope *Scope,
                       LVAddress LowerAddress, LVAddress UpperAddress);
  LVRange *getSectionRanges(LVSectionIndex SectionIndex);

  Error createInstructions();
  Error createInstructions(LVScope *Function, LVSectionIndex SectionIndex);
  Error createInstructions(LVScope *Function, LVSectionIndex SectionIndex,
                           const LVNameInfo &NameInfo);

  void processLines(LVLines *DebugLines, LVSectionIndex SectionIndex);
  void processLines(LVLines *DebugLines, LVSectionIndex SectionIndex,
                    LVScope *Function);

public:
  LVBinaryReader() = delete;
  LVBinaryReader(StringRef Filename, StringRef FileFormatName, ScopedPrinter &W,
                 LVBinaryType BinaryType)
      : LVReader(Filename, FileFormatName, W, BinaryType) {}
  LVBinaryReader(const LVBinaryReader &) = delete;
  LVBinaryReader &operator=(const LVBinaryReader &) = delete;
  virtual ~LVBinaryReader();

  void addToSymbolTable(StringRef Name, LVScope *Function,
                        LVSectionIndex SectionIndex = 0);
  void addToSymbolTable(StringRef Name, LVAddress Address,
                        LVSectionIndex SectionIndex, bool IsComdat);
  LVSectionIndex updateSymbolTable(LVScope *Function);

  const LVSymbolTableEntry &getSymbolTableEntry(StringRef Name);
  LVAddress getSymbolTableAddress(StringRef Name);
  LVSectionIndex getSymbolTableIndex(StringRef Name);
  bool getSymbolTableIsComdat(StringRef Name);

  LVSectionIndex getSectionIndex(LVScope *Scope) override {
    return Scope ? getSymbolTableIndex(Scope->getLinkageName())
                 : DotTextSectionIndex;
  }

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const { print(dbgs()); }
#endif
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVBINARYREADER_H
