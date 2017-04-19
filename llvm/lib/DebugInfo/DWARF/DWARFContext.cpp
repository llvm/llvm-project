//===- DWARFContext.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAranges.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugMacro.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugPubTable.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFGdbIndex.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Object/Decompressor.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace dwarf;
using namespace object;

#define DEBUG_TYPE "dwarf"

typedef DWARFDebugLine::LineTable DWARFLineTable;
typedef DILineInfoSpecifier::FileLineInfoKind FileLineInfoKind;
typedef DILineInfoSpecifier::FunctionNameKind FunctionNameKind;

static void dumpAccelSection(raw_ostream &OS, StringRef Name,
                             const DWARFSection& Section, StringRef StringSection,
                             bool LittleEndian) {
  DataExtractor AccelSection(Section.Data, LittleEndian, 0);
  DataExtractor StrData(StringSection, LittleEndian, 0);
  OS << "\n." << Name << " contents:\n";
  DWARFAcceleratorTable Accel(AccelSection, StrData, Section.Relocs);
  if (!Accel.extract())
    return;
  Accel.dump(OS);
}

void DWARFContext::dump(raw_ostream &OS, DIDumpType DumpType, bool DumpEH,
                        bool SummarizeTypes) {
  if (DumpType == DIDT_All || DumpType == DIDT_Abbrev) {
    OS << ".debug_abbrev contents:\n";
    getDebugAbbrev()->dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_AbbrevDwo)
    if (const DWARFDebugAbbrev *D = getDebugAbbrevDWO()) {
      OS << "\n.debug_abbrev.dwo contents:\n";
      D->dump(OS);
    }

  if (DumpType == DIDT_All || DumpType == DIDT_Info) {
    OS << "\n.debug_info contents:\n";
    for (const auto &CU : compile_units())
      CU->dump(OS);
  }

  if ((DumpType == DIDT_All || DumpType == DIDT_InfoDwo) &&
      getNumDWOCompileUnits()) {
    OS << "\n.debug_info.dwo contents:\n";
    for (const auto &DWOCU : dwo_compile_units())
      DWOCU->dump(OS);
  }

  if ((DumpType == DIDT_All || DumpType == DIDT_Types) && getNumTypeUnits()) {
    OS << "\n.debug_types contents:\n";
    for (const auto &TUS : type_unit_sections())
      for (const auto &TU : TUS)
        TU->dump(OS, SummarizeTypes);
  }

  if ((DumpType == DIDT_All || DumpType == DIDT_TypesDwo) &&
      getNumDWOTypeUnits()) {
    OS << "\n.debug_types.dwo contents:\n";
    for (const auto &DWOTUS : dwo_type_unit_sections())
      for (const auto &DWOTU : DWOTUS)
        DWOTU->dump(OS, SummarizeTypes);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Loc) {
    OS << "\n.debug_loc contents:\n";
    getDebugLoc()->dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_LocDwo) {
    OS << "\n.debug_loc.dwo contents:\n";
    getDebugLocDWO()->dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Frames) {
    OS << "\n.debug_frame contents:\n";
    getDebugFrame()->dump(OS);
    if (DumpEH) {
      OS << "\n.eh_frame contents:\n";
      getEHFrame()->dump(OS);
    }
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Macro) {
    OS << "\n.debug_macinfo contents:\n";
    getDebugMacro()->dump(OS);
  }

  uint32_t offset = 0;
  if (DumpType == DIDT_All || DumpType == DIDT_Aranges) {
    OS << "\n.debug_aranges contents:\n";
    DataExtractor arangesData(getARangeSection(), isLittleEndian(), 0);
    DWARFDebugArangeSet set;
    while (set.extract(arangesData, &offset))
      set.dump(OS);
  }

  uint8_t savedAddressByteSize = 0;
  if (DumpType == DIDT_All || DumpType == DIDT_Line) {
    OS << "\n.debug_line contents:\n";
    for (const auto &CU : compile_units()) {
      savedAddressByteSize = CU->getAddressByteSize();
      auto CUDIE = CU->getUnitDIE();
      if (!CUDIE)
        continue;
      if (auto StmtOffset = toSectionOffset(CUDIE.find(DW_AT_stmt_list))) {
        DataExtractor lineData(getLineSection().Data, isLittleEndian(),
                               savedAddressByteSize);
        DWARFDebugLine::LineTable LineTable;
        uint32_t Offset = *StmtOffset;
        LineTable.parse(lineData, &getLineSection().Relocs, &Offset);
        LineTable.dump(OS);
      }
    }
  }

  if (DumpType == DIDT_All || DumpType == DIDT_CUIndex) {
    OS << "\n.debug_cu_index contents:\n";
    getCUIndex().dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_TUIndex) {
    OS << "\n.debug_tu_index contents:\n";
    getTUIndex().dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_LineDwo) {
    OS << "\n.debug_line.dwo contents:\n";
    unsigned stmtOffset = 0;
    DataExtractor lineData(getLineDWOSection().Data, isLittleEndian(),
                           savedAddressByteSize);
    DWARFDebugLine::LineTable LineTable;
    while (LineTable.Prologue.parse(lineData, &stmtOffset)) {
      LineTable.dump(OS);
      LineTable.clear();
    }
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Str) {
    OS << "\n.debug_str contents:\n";
    DataExtractor strData(getStringSection(), isLittleEndian(), 0);
    offset = 0;
    uint32_t strOffset = 0;
    while (const char *s = strData.getCStr(&offset)) {
      OS << format("0x%8.8x: \"%s\"\n", strOffset, s);
      strOffset = offset;
    }
  }

  if ((DumpType == DIDT_All || DumpType == DIDT_StrDwo) &&
      !getStringDWOSection().empty()) {
    OS << "\n.debug_str.dwo contents:\n";
    DataExtractor strDWOData(getStringDWOSection(), isLittleEndian(), 0);
    offset = 0;
    uint32_t strDWOOffset = 0;
    while (const char *s = strDWOData.getCStr(&offset)) {
      OS << format("0x%8.8x: \"%s\"\n", strDWOOffset, s);
      strDWOOffset = offset;
    }
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Ranges) {
    OS << "\n.debug_ranges contents:\n";
    // In fact, different compile units may have different address byte
    // sizes, but for simplicity we just use the address byte size of the last
    // compile unit (there is no easy and fast way to associate address range
    // list and the compile unit it describes).
    DataExtractor rangesData(getRangeSection(), isLittleEndian(),
                             savedAddressByteSize);
    offset = 0;
    DWARFDebugRangeList rangeList;
    while (rangeList.extract(rangesData, &offset))
      rangeList.dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_Pubnames)
    DWARFDebugPubTable(getPubNamesSection(), isLittleEndian(), false)
        .dump("debug_pubnames", OS);

  if (DumpType == DIDT_All || DumpType == DIDT_Pubtypes)
    DWARFDebugPubTable(getPubTypesSection(), isLittleEndian(), false)
        .dump("debug_pubtypes", OS);

  if (DumpType == DIDT_All || DumpType == DIDT_GnuPubnames)
    DWARFDebugPubTable(getGnuPubNamesSection(), isLittleEndian(),
                       true /* GnuStyle */)
        .dump("debug_gnu_pubnames", OS);

  if (DumpType == DIDT_All || DumpType == DIDT_GnuPubtypes)
    DWARFDebugPubTable(getGnuPubTypesSection(), isLittleEndian(),
                       true /* GnuStyle */)
        .dump("debug_gnu_pubtypes", OS);

  if ((DumpType == DIDT_All || DumpType == DIDT_StrOffsetsDwo) &&
      !getStringOffsetDWOSection().empty()) {
    OS << "\n.debug_str_offsets.dwo contents:\n";
    DataExtractor strOffsetExt(getStringOffsetDWOSection(), isLittleEndian(),
                               0);
    offset = 0;
    uint64_t size = getStringOffsetDWOSection().size();
    while (offset < size) {
      OS << format("0x%8.8x: ", offset);
      OS << format("%8.8x\n", strOffsetExt.getU32(&offset));
    }
  }

  if ((DumpType == DIDT_All || DumpType == DIDT_GdbIndex) &&
      !getGdbIndexSection().empty()) {
    OS << "\n.gnu_index contents:\n";
    getGdbIndex().dump(OS);
  }

  if (DumpType == DIDT_All || DumpType == DIDT_AppleNames)
    dumpAccelSection(OS, "apple_names", getAppleNamesSection(),
                     getStringSection(), isLittleEndian());

  if (DumpType == DIDT_All || DumpType == DIDT_AppleTypes)
    dumpAccelSection(OS, "apple_types", getAppleTypesSection(),
                     getStringSection(), isLittleEndian());

  if (DumpType == DIDT_All || DumpType == DIDT_AppleNamespaces)
    dumpAccelSection(OS, "apple_namespaces", getAppleNamespacesSection(),
                     getStringSection(), isLittleEndian());

  if (DumpType == DIDT_All || DumpType == DIDT_AppleObjC)
    dumpAccelSection(OS, "apple_objc", getAppleObjCSection(),
                     getStringSection(), isLittleEndian());
}

const DWARFUnitIndex &DWARFContext::getCUIndex() {
  if (CUIndex)
    return *CUIndex;

  DataExtractor CUIndexData(getCUIndexSection(), isLittleEndian(), 0);

  CUIndex = llvm::make_unique<DWARFUnitIndex>(DW_SECT_INFO);
  CUIndex->parse(CUIndexData);
  return *CUIndex;
}

const DWARFUnitIndex &DWARFContext::getTUIndex() {
  if (TUIndex)
    return *TUIndex;

  DataExtractor TUIndexData(getTUIndexSection(), isLittleEndian(), 0);

  TUIndex = llvm::make_unique<DWARFUnitIndex>(DW_SECT_TYPES);
  TUIndex->parse(TUIndexData);
  return *TUIndex;
}

DWARFGdbIndex &DWARFContext::getGdbIndex() {
  if (GdbIndex)
    return *GdbIndex;

  DataExtractor GdbIndexData(getGdbIndexSection(), true /*LE*/, 0);
  GdbIndex = llvm::make_unique<DWARFGdbIndex>();
  GdbIndex->parse(GdbIndexData);
  return *GdbIndex;
}

const DWARFDebugAbbrev *DWARFContext::getDebugAbbrev() {
  if (Abbrev)
    return Abbrev.get();

  DataExtractor abbrData(getAbbrevSection(), isLittleEndian(), 0);

  Abbrev.reset(new DWARFDebugAbbrev());
  Abbrev->extract(abbrData);
  return Abbrev.get();
}

const DWARFDebugAbbrev *DWARFContext::getDebugAbbrevDWO() {
  if (AbbrevDWO)
    return AbbrevDWO.get();

  DataExtractor abbrData(getAbbrevDWOSection(), isLittleEndian(), 0);
  AbbrevDWO.reset(new DWARFDebugAbbrev());
  AbbrevDWO->extract(abbrData);
  return AbbrevDWO.get();
}

const DWARFDebugLoc *DWARFContext::getDebugLoc() {
  if (Loc)
    return Loc.get();

  DataExtractor LocData(getLocSection().Data, isLittleEndian(), 0);
  Loc.reset(new DWARFDebugLoc(getLocSection().Relocs));
  // assume all compile units have the same address byte size
  if (getNumCompileUnits())
    Loc->parse(LocData, getCompileUnitAtIndex(0)->getAddressByteSize());
  return Loc.get();
}

const DWARFDebugLocDWO *DWARFContext::getDebugLocDWO() {
  if (LocDWO)
    return LocDWO.get();

  DataExtractor LocData(getLocDWOSection().Data, isLittleEndian(), 0);
  LocDWO.reset(new DWARFDebugLocDWO());
  LocDWO->parse(LocData);
  return LocDWO.get();
}

const DWARFDebugAranges *DWARFContext::getDebugAranges() {
  if (Aranges)
    return Aranges.get();

  Aranges.reset(new DWARFDebugAranges());
  Aranges->generate(this);
  return Aranges.get();
}

const DWARFDebugFrame *DWARFContext::getDebugFrame() {
  if (DebugFrame)
    return DebugFrame.get();

  // There's a "bug" in the DWARFv3 standard with respect to the target address
  // size within debug frame sections. While DWARF is supposed to be independent
  // of its container, FDEs have fields with size being "target address size",
  // which isn't specified in DWARF in general. It's only specified for CUs, but
  // .eh_frame can appear without a .debug_info section. Follow the example of
  // other tools (libdwarf) and extract this from the container (ObjectFile
  // provides this information). This problem is fixed in DWARFv4
  // See this dwarf-discuss discussion for more details:
  // http://lists.dwarfstd.org/htdig.cgi/dwarf-discuss-dwarfstd.org/2011-December/001173.html
  DataExtractor debugFrameData(getDebugFrameSection(), isLittleEndian(),
                               getAddressSize());
  DebugFrame.reset(new DWARFDebugFrame(false /* IsEH */));
  DebugFrame->parse(debugFrameData);
  return DebugFrame.get();
}

const DWARFDebugFrame *DWARFContext::getEHFrame() {
  if (EHFrame)
    return EHFrame.get();

  DataExtractor debugFrameData(getEHFrameSection(), isLittleEndian(),
                               getAddressSize());
  DebugFrame.reset(new DWARFDebugFrame(true /* IsEH */));
  DebugFrame->parse(debugFrameData);
  return DebugFrame.get();
}

const DWARFDebugMacro *DWARFContext::getDebugMacro() {
  if (Macro)
    return Macro.get();

  DataExtractor MacinfoData(getMacinfoSection(), isLittleEndian(), 0);
  Macro.reset(new DWARFDebugMacro());
  Macro->parse(MacinfoData);
  return Macro.get();
}

const DWARFLineTable *
DWARFContext::getLineTableForUnit(DWARFUnit *U) {
  if (!Line)
    Line.reset(new DWARFDebugLine(&getLineSection().Relocs));

  auto UnitDIE = U->getUnitDIE();
  if (!UnitDIE)
    return nullptr;

  auto Offset = toSectionOffset(UnitDIE.find(DW_AT_stmt_list));
  if (!Offset)
    return nullptr; // No line table for this compile unit.

  uint32_t stmtOffset = *Offset + U->getLineTableOffset();
  // See if the line table is cached.
  if (const DWARFLineTable *lt = Line->getLineTable(stmtOffset))
    return lt;

  // We have to parse it first.
  DataExtractor lineData(U->getLineSection(), isLittleEndian(),
                         U->getAddressByteSize());
  return Line->getOrParseLineTable(lineData, stmtOffset);
}

void DWARFContext::parseCompileUnits() {
  CUs.parse(*this, getInfoSection());
}

void DWARFContext::parseTypeUnits() {
  if (!TUs.empty())
    return;
  for (const auto &I : getTypesSections()) {
    TUs.emplace_back();
    TUs.back().parse(*this, I.second);
  }
}

void DWARFContext::parseDWOCompileUnits() {
  DWOCUs.parseDWO(*this, getInfoDWOSection());
}

void DWARFContext::parseDWOTypeUnits() {
  if (!DWOTUs.empty())
    return;
  for (const auto &I : getTypesDWOSections()) {
    DWOTUs.emplace_back();
    DWOTUs.back().parseDWO(*this, I.second);
  }
}

DWARFCompileUnit *DWARFContext::getCompileUnitForOffset(uint32_t Offset) {
  parseCompileUnits();
  return CUs.getUnitForOffset(Offset);
}

DWARFCompileUnit *DWARFContext::getCompileUnitForAddress(uint64_t Address) {
  // First, get the offset of the compile unit.
  uint32_t CUOffset = getDebugAranges()->findAddress(Address);
  // Retrieve the compile unit.
  return getCompileUnitForOffset(CUOffset);
}

static bool getFunctionNameAndStartLineForAddress(DWARFCompileUnit *CU,
                                                  uint64_t Address,
                                                  FunctionNameKind Kind,
                                                  std::string &FunctionName,
                                                  uint32_t &StartLine) {
  // The address may correspond to instruction in some inlined function,
  // so we have to build the chain of inlined functions and take the
  // name of the topmost function in it.
  SmallVector<DWARFDie, 4> InlinedChain;
  CU->getInlinedChainForAddress(Address, InlinedChain);
  if (InlinedChain.empty())
    return false;

  const DWARFDie &DIE = InlinedChain[0];
  bool FoundResult = false;
  const char *Name = nullptr;
  if (Kind != FunctionNameKind::None && (Name = DIE.getSubroutineName(Kind))) {
    FunctionName = Name;
    FoundResult = true;
  }
  if (auto DeclLineResult = DIE.getDeclLine()) {
    StartLine = DeclLineResult;
    FoundResult = true;
  }

  return FoundResult;
}

DILineInfo DWARFContext::getLineInfoForAddress(uint64_t Address,
                                               DILineInfoSpecifier Spec) {
  DILineInfo Result;

  DWARFCompileUnit *CU = getCompileUnitForAddress(Address);
  if (!CU)
    return Result;
  getFunctionNameAndStartLineForAddress(CU, Address, Spec.FNKind,
                                        Result.FunctionName,
                                        Result.StartLine);
  if (Spec.FLIKind != FileLineInfoKind::None) {
    if (const DWARFLineTable *LineTable = getLineTableForUnit(CU))
      LineTable->getFileLineInfoForAddress(Address, CU->getCompilationDir(),
                                           Spec.FLIKind, Result);
  }
  return Result;
}

DILineInfoTable
DWARFContext::getLineInfoForAddressRange(uint64_t Address, uint64_t Size,
                                         DILineInfoSpecifier Spec) {
  DILineInfoTable  Lines;
  DWARFCompileUnit *CU = getCompileUnitForAddress(Address);
  if (!CU)
    return Lines;

  std::string FunctionName = "<invalid>";
  uint32_t StartLine = 0;
  getFunctionNameAndStartLineForAddress(CU, Address, Spec.FNKind, FunctionName,
                                        StartLine);

  // If the Specifier says we don't need FileLineInfo, just
  // return the top-most function at the starting address.
  if (Spec.FLIKind == FileLineInfoKind::None) {
    DILineInfo Result;
    Result.FunctionName = FunctionName;
    Result.StartLine = StartLine;
    Lines.push_back(std::make_pair(Address, Result));
    return Lines;
  }

  const DWARFLineTable *LineTable = getLineTableForUnit(CU);

  // Get the index of row we're looking for in the line table.
  std::vector<uint32_t> RowVector;
  if (!LineTable->lookupAddressRange(Address, Size, RowVector))
    return Lines;

  for (uint32_t RowIndex : RowVector) {
    // Take file number and line/column from the row.
    const DWARFDebugLine::Row &Row = LineTable->Rows[RowIndex];
    DILineInfo Result;
    LineTable->getFileNameByIndex(Row.File, CU->getCompilationDir(),
                                  Spec.FLIKind, Result.FileName);
    Result.FunctionName = FunctionName;
    Result.Line = Row.Line;
    Result.Column = Row.Column;
    Result.StartLine = StartLine;
    Lines.push_back(std::make_pair(Row.Address, Result));
  }

  return Lines;
}

DIInliningInfo
DWARFContext::getInliningInfoForAddress(uint64_t Address,
                                        DILineInfoSpecifier Spec) {
  DIInliningInfo InliningInfo;

  DWARFCompileUnit *CU = getCompileUnitForAddress(Address);
  if (!CU)
    return InliningInfo;

  const DWARFLineTable *LineTable = nullptr;
  SmallVector<DWARFDie, 4> InlinedChain;
  CU->getInlinedChainForAddress(Address, InlinedChain);
  if (InlinedChain.size() == 0) {
    // If there is no DIE for address (e.g. it is in unavailable .dwo file),
    // try to at least get file/line info from symbol table.
    if (Spec.FLIKind != FileLineInfoKind::None) {
      DILineInfo Frame;
      LineTable = getLineTableForUnit(CU);
      if (LineTable &&
          LineTable->getFileLineInfoForAddress(Address, CU->getCompilationDir(),
                                               Spec.FLIKind, Frame))
        InliningInfo.addFrame(Frame);
    }
    return InliningInfo;
  }

  uint32_t CallFile = 0, CallLine = 0, CallColumn = 0, CallDiscriminator = 0;
  for (uint32_t i = 0, n = InlinedChain.size(); i != n; i++) {
    DWARFDie &FunctionDIE = InlinedChain[i];
    DILineInfo Frame;
    // Get function name if necessary.
    if (const char *Name = FunctionDIE.getSubroutineName(Spec.FNKind))
      Frame.FunctionName = Name;
    if (auto DeclLineResult = FunctionDIE.getDeclLine())
      Frame.StartLine = DeclLineResult;
    if (Spec.FLIKind != FileLineInfoKind::None) {
      if (i == 0) {
        // For the topmost frame, initialize the line table of this
        // compile unit and fetch file/line info from it.
        LineTable = getLineTableForUnit(CU);
        // For the topmost routine, get file/line info from line table.
        if (LineTable)
          LineTable->getFileLineInfoForAddress(Address, CU->getCompilationDir(),
                                               Spec.FLIKind, Frame);
      } else {
        // Otherwise, use call file, call line and call column from
        // previous DIE in inlined chain.
        if (LineTable)
          LineTable->getFileNameByIndex(CallFile, CU->getCompilationDir(),
                                        Spec.FLIKind, Frame.FileName);
        Frame.Line = CallLine;
        Frame.Column = CallColumn;
        Frame.Discriminator = CallDiscriminator;
      }
      // Get call file/line/column of a current DIE.
      if (i + 1 < n) {
        FunctionDIE.getCallerFrame(CallFile, CallLine, CallColumn,
                                   CallDiscriminator);
      }
    }
    InliningInfo.addFrame(Frame);
  }
  return InliningInfo;
}

static Error createError(const Twine &Reason, llvm::Error E) {
  return make_error<StringError>(Reason + toString(std::move(E)),
                                 inconvertibleErrorCode());
}

/// Returns the address of symbol relocation used against. Used for futher
/// relocations computation. Symbol's section load address is taken in account if
/// LoadedObjectInfo interface is provided.
static Expected<uint64_t> getSymbolAddress(const object::ObjectFile &Obj,
                                           const RelocationRef &Reloc,
                                           const LoadedObjectInfo *L) {
  uint64_t Ret = 0;
  object::section_iterator RSec = Obj.section_end();
  object::symbol_iterator Sym = Reloc.getSymbol();

  // First calculate the address of the symbol or section as it appears
  // in the object file
  if (Sym != Obj.symbol_end()) {
    Expected<uint64_t> SymAddrOrErr = Sym->getAddress();
    if (!SymAddrOrErr)
      return createError("error: failed to compute symbol address: ",
                         SymAddrOrErr.takeError());

    // Also remember what section this symbol is in for later
    auto SectOrErr = Sym->getSection();
    if (!SectOrErr)
      return createError("error: failed to get symbol section: ",
                         SectOrErr.takeError());

    RSec = *SectOrErr;
    Ret = *SymAddrOrErr;
  } else if (auto *MObj = dyn_cast<MachOObjectFile>(&Obj)) {
    RSec = MObj->getRelocationSection(Reloc.getRawDataRefImpl());
    Ret = RSec->getAddress();
  }

  // If we are given load addresses for the sections, we need to adjust:
  // SymAddr = (Address of Symbol Or Section in File) -
  //           (Address of Section in File) +
  //           (Load Address of Section)
  // RSec is now either the section being targeted or the section
  // containing the symbol being targeted. In either case,
  // we need to perform the same computation.
  if (L && RSec != Obj.section_end())
    if (uint64_t SectionLoadAddress = L->getSectionLoadAddress(*RSec))
      Ret += SectionLoadAddress - RSec->getAddress();
  return Ret;
}

static bool isRelocScattered(const object::ObjectFile &Obj,
                             const RelocationRef &Reloc) {
  const MachOObjectFile *MachObj = dyn_cast<MachOObjectFile>(&Obj);
  if (!MachObj)
    return false;
  // MachO also has relocations that point to sections and
  // scattered relocations.
  auto RelocInfo = MachObj->getRelocation(Reloc.getRawDataRefImpl());
  return MachObj->isRelocationScattered(RelocInfo);
}

DWARFContextInMemory::DWARFContextInMemory(const object::ObjectFile &Obj,
    const LoadedObjectInfo *L)
    : IsLittleEndian(Obj.isLittleEndian()),
      AddressSize(Obj.getBytesInAddress()) {
  for (const SectionRef &Section : Obj.sections()) {
    StringRef name;
    Section.getName(name);
    // Skip BSS and Virtual sections, they aren't interesting.
    bool IsBSS = Section.isBSS();
    if (IsBSS)
      continue;
    bool IsVirtual = Section.isVirtual();
    if (IsVirtual)
      continue;
    StringRef data;

    section_iterator RelocatedSection = Section.getRelocatedSection();
    // Try to obtain an already relocated version of this section.
    // Else use the unrelocated section from the object file. We'll have to
    // apply relocations ourselves later.
    if (!L || !L->getLoadedSectionContents(*RelocatedSection,data))
      Section.getContents(data);

    if (Decompressor::isCompressed(Section)) {
      Expected<Decompressor> Decompressor =
          Decompressor::create(name, data, IsLittleEndian, AddressSize == 8);
      if (!Decompressor)
        continue;
      SmallString<32> Out;
      if (auto Err = Decompressor->decompress(Out))
        continue;
      UncompressedSections.emplace_back(std::move(Out));
      data = UncompressedSections.back();
    }

    // Compressed sections names in GNU style starts from ".z",
    // at this point section is decompressed and we drop compression prefix.
    name = name.substr(
        name.find_first_not_of("._z")); // Skip ".", "z" and "_" prefixes.

    if (StringRef *SectionData = MapSectionToMember(name)) {
      *SectionData = data;
      if (name == "debug_ranges") {
        // FIXME: Use the other dwo range section when we emit it.
        RangeDWOSection = data;
      }
    } else if (name == "debug_types") {
      // Find debug_types data by section rather than name as there are
      // multiple, comdat grouped, debug_types sections.
      TypesSections[Section].Data = data;
    } else if (name == "debug_types.dwo") {
      TypesDWOSections[Section].Data = data;
    }

    if (RelocatedSection == Obj.section_end())
      continue;

    StringRef RelSecName;
    StringRef RelSecData;
    RelocatedSection->getName(RelSecName);

    // If the section we're relocating was relocated already by the JIT,
    // then we used the relocated version above, so we do not need to process
    // relocations for it now.
    if (L && L->getLoadedSectionContents(*RelocatedSection,RelSecData))
      continue;

    // In Mach-o files, the relocations do not need to be applied if
    // there is no load offset to apply. The value read at the
    // relocation point already factors in the section address
    // (actually applying the relocations will produce wrong results
    // as the section address will be added twice).
    if (!L && isa<MachOObjectFile>(&Obj))
      continue;

    RelSecName = RelSecName.substr(
        RelSecName.find_first_not_of("._")); // Skip . and _ prefixes.

    // TODO: Add support for relocations in other sections as needed.
    // Record relocations for the debug_info and debug_line sections.
    RelocAddrMap *Map = StringSwitch<RelocAddrMap*>(RelSecName)
        .Case("debug_info", &InfoSection.Relocs)
        .Case("debug_loc", &LocSection.Relocs)
        .Case("debug_info.dwo", &InfoDWOSection.Relocs)
        .Case("debug_line", &LineSection.Relocs)
        .Case("apple_names", &AppleNamesSection.Relocs)
        .Case("apple_types", &AppleTypesSection.Relocs)
        .Case("apple_namespaces", &AppleNamespacesSection.Relocs)
        .Case("apple_namespac", &AppleNamespacesSection.Relocs)
        .Case("apple_objc", &AppleObjCSection.Relocs)
        .Default(nullptr);
    if (!Map) {
      // Find debug_types relocs by section rather than name as there are
      // multiple, comdat grouped, debug_types sections.
      if (RelSecName == "debug_types")
        Map = &TypesSections[*RelocatedSection].Relocs;
      else if (RelSecName == "debug_types.dwo")
        Map = &TypesDWOSections[*RelocatedSection].Relocs;
      else
        continue;
    }

    if (Section.relocation_begin() != Section.relocation_end()) {
      uint64_t SectionSize = RelocatedSection->getSize();
      for (const RelocationRef &Reloc : Section.relocations()) {
        // FIXME: it's not clear how to correctly handle scattered
        // relocations.
        if (isRelocScattered(Obj, Reloc))
          continue;

        Expected<uint64_t> SymAddrOrErr = getSymbolAddress(Obj, Reloc, L);
        if (!SymAddrOrErr) {
          errs() << toString(SymAddrOrErr.takeError()) << '\n';
          continue;
        }

        object::RelocVisitor V(Obj);
        object::RelocToApply R(V.visit(Reloc.getType(), Reloc, *SymAddrOrErr));
        if (V.error()) {
          SmallString<32> Name;
          Reloc.getTypeName(Name);
          errs() << "error: failed to compute relocation: "
                 << Name << "\n";
          continue;
        }
        uint64_t Address = Reloc.getOffset();
        if (Address + R.Width > SectionSize) {
          errs() << "error: " << R.Width << "-byte relocation starting "
                 << Address << " bytes into section " << name << " which is "
                 << SectionSize << " bytes long.\n";
          continue;
        }
        if (R.Width > 8) {
          errs() << "error: can't handle a relocation of more than 8 bytes at "
                    "a time.\n";
          continue;
        }
        DEBUG(dbgs() << "Writing " << format("%p", R.Value)
                     << " at " << format("%p", Address)
                     << " with width " << format("%d", R.Width)
                     << "\n");
        Map->insert(std::make_pair(Address, std::make_pair(R.Width, R.Value)));
      }
    }
  }
}

DWARFContextInMemory::DWARFContextInMemory(
    const StringMap<std::unique_ptr<MemoryBuffer>> &Sections, uint8_t AddrSize,
    bool isLittleEndian)
    : IsLittleEndian(isLittleEndian), AddressSize(AddrSize) {
  for (const auto &SecIt : Sections) {
    if (StringRef *SectionData = MapSectionToMember(SecIt.first()))
      *SectionData = SecIt.second->getBuffer();
  }
}

StringRef *DWARFContextInMemory::MapSectionToMember(StringRef Name) {
  return StringSwitch<StringRef *>(Name)
      .Case("debug_info", &InfoSection.Data)
      .Case("debug_abbrev", &AbbrevSection)
      .Case("debug_loc", &LocSection.Data)
      .Case("debug_line", &LineSection.Data)
      .Case("debug_aranges", &ARangeSection)
      .Case("debug_frame", &DebugFrameSection)
      .Case("eh_frame", &EHFrameSection)
      .Case("debug_str", &StringSection)
      .Case("debug_ranges", &RangeSection)
      .Case("debug_macinfo", &MacinfoSection)
      .Case("debug_pubnames", &PubNamesSection)
      .Case("debug_pubtypes", &PubTypesSection)
      .Case("debug_gnu_pubnames", &GnuPubNamesSection)
      .Case("debug_gnu_pubtypes", &GnuPubTypesSection)
      .Case("debug_info.dwo", &InfoDWOSection.Data)
      .Case("debug_abbrev.dwo", &AbbrevDWOSection)
      .Case("debug_loc.dwo", &LocDWOSection.Data)
      .Case("debug_line.dwo", &LineDWOSection.Data)
      .Case("debug_str.dwo", &StringDWOSection)
      .Case("debug_str_offsets.dwo", &StringOffsetDWOSection)
      .Case("debug_addr", &AddrSection)
      .Case("apple_names", &AppleNamesSection.Data)
      .Case("apple_types", &AppleTypesSection.Data)
      .Case("apple_namespaces", &AppleNamespacesSection.Data)
      .Case("apple_namespac", &AppleNamespacesSection.Data)
      .Case("apple_objc", &AppleObjCSection.Data)
      .Case("debug_cu_index", &CUIndexSection)
      .Case("debug_tu_index", &TUIndexSection)
      .Case("gdb_index", &GdbIndexSection)
      // Any more debug info sections go here.
      .Default(nullptr);
}

void DWARFContextInMemory::anchor() {}
