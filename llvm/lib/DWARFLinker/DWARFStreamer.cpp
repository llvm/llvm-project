//===- DwarfStreamer.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/DWARFStreamer.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/NonRelocatableStringpool.h"
#include "llvm/DWARFLinker/DWARFLinkerCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

bool DwarfStreamer::init(Triple TheTriple) {
  std::string ErrorStr;
  std::string TripleName;
  StringRef Context = "dwarf streamer init";

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TripleName, TheTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context), false;
  TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context),
           false;

  MCTargetOptions MCOptions = mc::InitMCTargetOptionsFromFlags();
  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context), false;

  MOFI.reset(new MCObjectFileInfo);
  MC.reset(new MCContext(MAI.get(), MRI.get(), MOFI.get()));
  MOFI->InitMCObjectFileInfo(TheTriple, /*PIC*/ false, *MC);

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context), false;

  MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions);
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context), false;

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context), false;

  MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, *MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context), false;

  switch (OutFileType) {
  case OutputFileType::Assembly: {
    MIP = TheTarget->createMCInstPrinter(TheTriple, MAI->getAssemblerDialect(),
                                         *MAI, *MII, *MRI);
    MS = TheTarget->createAsmStreamer(
        *MC, std::make_unique<formatted_raw_ostream>(OutFile), true, true, MIP,
        std::unique_ptr<MCCodeEmitter>(MCE), std::unique_ptr<MCAsmBackend>(MAB),
        true);
    break;
  }
  case OutputFileType::Object: {
    MS = TheTarget->createMCObjectStreamer(
        TheTriple, *MC, std::unique_ptr<MCAsmBackend>(MAB),
        MAB->createObjectWriter(OutFile), std::unique_ptr<MCCodeEmitter>(MCE),
        *MSTI, MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ false);
    break;
  }
  }

  if (!MS)
    return error("no object streamer for target " + TripleName, Context), false;

  // Finally create the AsmPrinter we'll use to emit the DIEs.
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          None));
  if (!TM)
    return error("no target machine for target " + TripleName, Context), false;

  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return error("no asm printer for target " + TripleName, Context), false;

  RangesSectionSize = 0;
  LocSectionSize = 0;
  LineSectionSize = 0;
  FrameSectionSize = 0;
  DebugInfoSectionSize = 0;

  return true;
}

void DwarfStreamer::finish() { MS->Finish(); }

void DwarfStreamer::switchToDebugInfoSection(unsigned DwarfVersion) {
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  MC->setDwarfVersion(DwarfVersion);
}

/// Emit the compilation unit header for \p Unit in the debug_info section.
///
/// A Dwarf 4 section header is encoded as:
///  uint32_t   Unit length (omitting this field)
///  uint16_t   Version
///  uint32_t   Abbreviation table offset
///  uint8_t    Address size
/// Leading to a total of 11 bytes.
///
/// A Dwarf 5 section header is encoded as:
///  uint32_t   Unit length (omitting this field)
///  uint16_t   Version
///  uint8_t    Unit type
///  uint8_t    Address size
///  uint32_t   Abbreviation table offset
/// Leading to a total of 12 bytes.
void DwarfStreamer::emitCompileUnitHeader(CompileUnit &Unit,
                                          unsigned DwarfVersion) {
  switchToDebugInfoSection(DwarfVersion);

  /// The start of the unit within its section.
  Unit.setLabelBegin(Asm->createTempSymbol("cu_begin"));
  Asm->OutStreamer->emitLabel(Unit.getLabelBegin());

  // Emit size of content not including length itself. The size has already
  // been computed in CompileUnit::computeOffsets(). Subtract 4 to that size to
  // account for the length field.
  Asm->emitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset() - 4);
  Asm->emitInt16(DwarfVersion);

  if (DwarfVersion >= 5) {
    Asm->emitInt8(dwarf::DW_UT_compile);
    Asm->emitInt8(Unit.getOrigUnit().getAddressByteSize());
    // We share one abbreviations table across all units so it's always at the
    // start of the section.
    Asm->emitInt32(0);
    DebugInfoSectionSize += 12;
  } else {
    // We share one abbreviations table across all units so it's always at the
    // start of the section.
    Asm->emitInt32(0);
    Asm->emitInt8(Unit.getOrigUnit().getAddressByteSize());
    DebugInfoSectionSize += 11;
  }

  // Remember this CU.
  EmittedUnits.push_back({Unit.getUniqueID(), Unit.getLabelBegin()});
}

/// Emit the \p Abbrevs array as the shared abbreviation table
/// for the linked Dwarf file.
void DwarfStreamer::emitAbbrevs(
    const std::vector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
    unsigned DwarfVersion) {
  MS->SwitchSection(MOFI->getDwarfAbbrevSection());
  MC->setDwarfVersion(DwarfVersion);
  Asm->emitDwarfAbbrevs(Abbrevs);
}

/// Recursively emit the DIE tree rooted at \p Die.
void DwarfStreamer::emitDIE(DIE &Die) {
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  Asm->emitDwarfDIE(Die);
  DebugInfoSectionSize += Die.getSize();
}

/// Emit contents of section SecName From Obj.
void DwarfStreamer::emitSectionContents(StringRef SecData, StringRef SecName) {
  MCSection *Section =
      StringSwitch<MCSection *>(SecName)
          .Case("debug_line", MC->getObjectFileInfo()->getDwarfLineSection())
          .Case("debug_loc", MC->getObjectFileInfo()->getDwarfLocSection())
          .Case("debug_ranges",
                MC->getObjectFileInfo()->getDwarfRangesSection())
          .Case("debug_frame", MC->getObjectFileInfo()->getDwarfFrameSection())
          .Case("debug_aranges",
                MC->getObjectFileInfo()->getDwarfARangesSection())
          .Default(nullptr);

  if (Section) {
    MS->SwitchSection(Section);

    MS->emitBytes(SecData);
  }
}

/// Emit DIE containing warnings.
void DwarfStreamer::emitPaperTrailWarningsDie(DIE &Die) {
  switchToDebugInfoSection(/* Version */ 2);
  auto &Asm = getAsmPrinter();
  Asm.emitInt32(11 + Die.getSize() - 4);
  Asm.emitInt16(2);
  Asm.emitInt32(0);
  Asm.emitInt8(MOFI->getTargetTriple().isArch64Bit() ? 8 : 4);
  DebugInfoSectionSize += 11;
  emitDIE(Die);
}

/// Emit the debug_str section stored in \p Pool.
void DwarfStreamer::emitStrings(const NonRelocatableStringpool &Pool) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfStrSection());
  std::vector<DwarfStringPoolEntryRef> Entries = Pool.getEntriesForEmission();
  for (auto Entry : Entries) {
    // Emit the string itself.
    Asm->OutStreamer->emitBytes(Entry.getString());
    // Emit a null terminator.
    Asm->emitInt8(0);
  }

#if 0
  if (DwarfVersion >= 5) {
    // Emit an empty string offset section.
    Asm->OutStreamer->SwitchSection(MOFI->getDwarfStrOffSection());
    Asm->emitDwarfUnitLength(4, "Length of String Offsets Set");
    Asm->emitInt16(DwarfVersion);
    Asm->emitInt16(0);
  }
#endif
}

void DwarfStreamer::emitDebugNames(
    AccelTable<DWARF5AccelTableStaticData> &Table) {
  if (EmittedUnits.empty())
    return;

  // Build up data structures needed to emit this section.
  std::vector<MCSymbol *> CompUnits;
  DenseMap<unsigned, size_t> UniqueIdToCuMap;
  unsigned Id = 0;
  for (auto &CU : EmittedUnits) {
    CompUnits.push_back(CU.LabelBegin);
    // We might be omitting CUs, so we need to remap them.
    UniqueIdToCuMap[CU.ID] = Id++;
  }

  Asm->OutStreamer->SwitchSection(MOFI->getDwarfDebugNamesSection());
  emitDWARF5AccelTable(
      Asm.get(), Table, CompUnits,
      [&UniqueIdToCuMap](const DWARF5AccelTableStaticData &Entry) {
        return UniqueIdToCuMap[Entry.getCUIndex()];
      });
}

void DwarfStreamer::emitAppleNamespaces(
    AccelTable<AppleAccelTableStaticOffsetData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelNamespaceSection());
  auto *SectionBegin = Asm->createTempSymbol("namespac_begin");
  Asm->OutStreamer->emitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "namespac", SectionBegin);
}

void DwarfStreamer::emitAppleNames(
    AccelTable<AppleAccelTableStaticOffsetData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelNamesSection());
  auto *SectionBegin = Asm->createTempSymbol("names_begin");
  Asm->OutStreamer->emitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "names", SectionBegin);
}

void DwarfStreamer::emitAppleObjc(
    AccelTable<AppleAccelTableStaticOffsetData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelObjCSection());
  auto *SectionBegin = Asm->createTempSymbol("objc_begin");
  Asm->OutStreamer->emitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "objc", SectionBegin);
}

void DwarfStreamer::emitAppleTypes(
    AccelTable<AppleAccelTableStaticTypeData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelTypesSection());
  auto *SectionBegin = Asm->createTempSymbol("types_begin");
  Asm->OutStreamer->emitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "types", SectionBegin);
}

/// Emit the swift_ast section stored in \p Buffers.
void DwarfStreamer::emitSwiftAST(StringRef Buffer) {
  MCSection *SwiftASTSection = MOFI->getDwarfSwiftASTSection();
  SwiftASTSection->setAlignment(Align(32));
  MS->SwitchSection(SwiftASTSection);
  MS->emitBytes(Buffer);
}

/// Emit the debug_range section contents for \p FuncRange by
/// translating the original \p Entries. The debug_range section
/// format is totally trivial, consisting just of pairs of address
/// sized addresses describing the ranges.
void DwarfStreamer::emitRangesEntries(
    int64_t UnitPcOffset, uint64_t OrigLowPc,
    const FunctionIntervals::const_iterator &FuncRange,
    const std::vector<DWARFDebugRangeList::RangeListEntry> &Entries,
    unsigned AddressSize) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());

  // Offset each range by the right amount.
  int64_t PcOffset = Entries.empty() ? 0 : FuncRange.value() + UnitPcOffset;
  for (const auto &Range : Entries) {
    if (Range.isBaseAddressSelectionEntry(AddressSize)) {
      warn("unsupported base address selection operation",
           "emitting debug_ranges");
      break;
    }
    // Do not emit empty ranges.
    if (Range.StartAddress == Range.EndAddress)
      continue;

    // All range entries should lie in the function range.
    if (!(Range.StartAddress + OrigLowPc >= FuncRange.start() &&
          Range.EndAddress + OrigLowPc <= FuncRange.stop()))
      warn("inconsistent range data.", "emitting debug_ranges");
    MS->emitIntValue(Range.StartAddress + PcOffset, AddressSize);
    MS->emitIntValue(Range.EndAddress + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->emitIntValue(0, AddressSize);
  MS->emitIntValue(0, AddressSize);
  RangesSectionSize += 2 * AddressSize;
}

/// Emit the debug_aranges contribution of a unit and
/// if \p DoDebugRanges is true the debug_range contents for a
/// compile_unit level DW_AT_ranges attribute (Which are basically the
/// same thing with a different base address).
/// Just aggregate all the ranges gathered inside that unit.
void DwarfStreamer::emitUnitRangesEntries(CompileUnit &Unit,
                                          bool DoDebugRanges) {
  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  // Gather the ranges in a vector, so that we can simplify them. The
  // IntervalMap will have coalesced the non-linked ranges, but here
  // we want to coalesce the linked addresses.
  std::vector<std::pair<uint64_t, uint64_t>> Ranges;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  for (auto Range = FunctionRanges.begin(), End = FunctionRanges.end();
       Range != End; ++Range)
    Ranges.push_back(std::make_pair(Range.start() + Range.value(),
                                    Range.stop() + Range.value()));

  // The object addresses where sorted, but again, the linked
  // addresses might end up in a different order.
  llvm::sort(Ranges);

  if (!Ranges.empty()) {
    MS->SwitchSection(MC->getObjectFileInfo()->getDwarfARangesSection());

    MCSymbol *BeginLabel = Asm->createTempSymbol("Barange");
    MCSymbol *EndLabel = Asm->createTempSymbol("Earange");

    unsigned HeaderSize =
        sizeof(int32_t) + // Size of contents (w/o this field
        sizeof(int16_t) + // DWARF ARange version number
        sizeof(int32_t) + // Offset of CU in the .debug_info section
        sizeof(int8_t) +  // Pointer Size (in bytes)
        sizeof(int8_t);   // Segment Size (in bytes)

    unsigned TupleSize = AddressSize * 2;
    unsigned Padding = offsetToAlignment(HeaderSize, Align(TupleSize));

    Asm->emitLabelDifference(EndLabel, BeginLabel, 4); // Arange length
    Asm->OutStreamer->emitLabel(BeginLabel);
    Asm->emitInt16(dwarf::DW_ARANGES_VERSION); // Version number
    Asm->emitInt32(Unit.getStartOffset());     // Corresponding unit's offset
    Asm->emitInt8(AddressSize);                // Address size
    Asm->emitInt8(0);                          // Segment size

    Asm->OutStreamer->emitFill(Padding, 0x0);

    for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End;
         ++Range) {
      uint64_t RangeStart = Range->first;
      MS->emitIntValue(RangeStart, AddressSize);
      while ((Range + 1) != End && Range->second == (Range + 1)->first)
        ++Range;
      MS->emitIntValue(Range->second - RangeStart, AddressSize);
    }

    // Emit terminator
    Asm->OutStreamer->emitIntValue(0, AddressSize);
    Asm->OutStreamer->emitIntValue(0, AddressSize);
    Asm->OutStreamer->emitLabel(EndLabel);
  }

  if (!DoDebugRanges)
    return;

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());
  // Offset each range by the right amount.
  int64_t PcOffset = -Unit.getLowPc();
  // Emit coalesced ranges.
  for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End; ++Range) {
    MS->emitIntValue(Range->first + PcOffset, AddressSize);
    while (Range + 1 != End && Range->second == (Range + 1)->first)
      ++Range;
    MS->emitIntValue(Range->second + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->emitIntValue(0, AddressSize);
  MS->emitIntValue(0, AddressSize);
  RangesSectionSize += 2 * AddressSize;
}

/// Emit location lists for \p Unit and update attributes to point to the new
/// entries.
void DwarfStreamer::emitLocationsForUnit(
    const CompileUnit &Unit, DWARFContext &Dwarf,
    std::function<void(StringRef, SmallVectorImpl<uint8_t> &)> ProcessExpr) {
  const auto &Attributes = Unit.getLocationAttributes();

  if (Attributes.empty())
    return;

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLocSection());

  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  uint64_t BaseAddressMarker = (AddressSize == 8)
                                   ? std::numeric_limits<uint64_t>::max()
                                   : std::numeric_limits<uint32_t>::max();
  const DWARFSection &InputSec = Dwarf.getDWARFObj().getLocSection();
  DataExtractor Data(InputSec.Data, Dwarf.isLittleEndian(), AddressSize);
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  auto OrigUnitDie = OrigUnit.getUnitDIE(false);
  int64_t UnitPcOffset = 0;
  if (auto OrigLowPc = dwarf::toAddress(OrigUnitDie.find(dwarf::DW_AT_low_pc)))
    UnitPcOffset = int64_t(*OrigLowPc) - Unit.getLowPc();

  SmallVector<uint8_t, 32> Buffer;
  for (const auto &Attr : Attributes) {
    uint64_t Offset = Attr.first.get();
    Attr.first.set(LocSectionSize);
    // This is the quantity to add to the old location address to get
    // the correct address for the new one.
    int64_t LocPcOffset = Attr.second + UnitPcOffset;
    while (Data.isValidOffset(Offset)) {
      uint64_t Low = Data.getUnsigned(&Offset, AddressSize);
      uint64_t High = Data.getUnsigned(&Offset, AddressSize);
      LocSectionSize += 2 * AddressSize;
      // End of list entry.
      if (Low == 0 && High == 0) {
        Asm->OutStreamer->emitIntValue(0, AddressSize);
        Asm->OutStreamer->emitIntValue(0, AddressSize);
        break;
      }
      // Base address selection entry.
      if (Low == BaseAddressMarker) {
        Asm->OutStreamer->emitIntValue(BaseAddressMarker, AddressSize);
        Asm->OutStreamer->emitIntValue(High + Attr.second, AddressSize);
        LocPcOffset = 0;
        continue;
      }
      // Location list entry.
      Asm->OutStreamer->emitIntValue(Low + LocPcOffset, AddressSize);
      Asm->OutStreamer->emitIntValue(High + LocPcOffset, AddressSize);
      uint64_t Length = Data.getU16(&Offset);
      Asm->OutStreamer->emitIntValue(Length, 2);
      // Copy the bytes into to the buffer, process them, emit them.
      Buffer.reserve(Length);
      Buffer.resize(0);
      StringRef Input = InputSec.Data.substr(Offset, Length);
      ProcessExpr(Input, Buffer);
      Asm->OutStreamer->emitBytes(
          StringRef((const char *)Buffer.data(), Length));
      Offset += Length;
      LocSectionSize += Length + 2;
    }
  }
}

void DwarfStreamer::emitLineTableForUnit(MCDwarfLineTableParams Params,
                                         StringRef PrologueBytes,
                                         unsigned MinInstLength,
                                         std::vector<DWARFDebugLine::Row> &Rows,
                                         unsigned PointerSize) {
  // Switch to the section where the table will be emitted into.
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLineSection());
  MCSymbol *LineStartSym = MC->createTempSymbol();
  MCSymbol *LineEndSym = MC->createTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  Asm->emitLabelDifference(LineEndSym, LineStartSym, 4);
  Asm->OutStreamer->emitLabel(LineStartSym);
  // Copy Prologue.
  MS->emitBytes(PrologueBytes);
  LineSectionSize += PrologueBytes.size() + 4;

  SmallString<128> EncodingBuffer;
  raw_svector_ostream EncodingOS(EncodingBuffer);

  if (Rows.empty()) {
    // We only have the dummy entry, dsymutil emits an entry with a 0
    // address in that case.
    MCDwarfLineAddr::Encode(*MC, Params, std::numeric_limits<int64_t>::max(), 0,
                            EncodingOS);
    MS->emitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    MS->emitLabel(LineEndSym);
    return;
  }

  // Line table state machine fields
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned IsStatement = 1;
  unsigned Isa = 0;
  uint64_t Address = -1ULL;

  unsigned RowsSinceLastSequence = 0;

  for (unsigned Idx = 0; Idx < Rows.size(); ++Idx) {
    auto &Row = Rows[Idx];

    int64_t AddressDelta;
    if (Address == -1ULL) {
      MS->emitIntValue(dwarf::DW_LNS_extended_op, 1);
      MS->emitULEB128IntValue(PointerSize + 1);
      MS->emitIntValue(dwarf::DW_LNE_set_address, 1);
      MS->emitIntValue(Row.Address.Address, PointerSize);
      LineSectionSize += 2 + PointerSize + getULEB128Size(PointerSize + 1);
      AddressDelta = 0;
    } else {
      AddressDelta = (Row.Address.Address - Address) / MinInstLength;
    }

    // FIXME: code copied and transformed from MCDwarf.cpp::EmitDwarfLineTable.
    // We should find a way to share this code, but the current compatibility
    // requirement with classic dsymutil makes it hard. Revisit that once this
    // requirement is dropped.

    if (FileNum != Row.File) {
      FileNum = Row.File;
      MS->emitIntValue(dwarf::DW_LNS_set_file, 1);
      MS->emitULEB128IntValue(FileNum);
      LineSectionSize += 1 + getULEB128Size(FileNum);
    }
    if (Column != Row.Column) {
      Column = Row.Column;
      MS->emitIntValue(dwarf::DW_LNS_set_column, 1);
      MS->emitULEB128IntValue(Column);
      LineSectionSize += 1 + getULEB128Size(Column);
    }

    // FIXME: We should handle the discriminator here, but dsymutil doesn't
    // consider it, thus ignore it for now.

    if (Isa != Row.Isa) {
      Isa = Row.Isa;
      MS->emitIntValue(dwarf::DW_LNS_set_isa, 1);
      MS->emitULEB128IntValue(Isa);
      LineSectionSize += 1 + getULEB128Size(Isa);
    }
    if (IsStatement != Row.IsStmt) {
      IsStatement = Row.IsStmt;
      MS->emitIntValue(dwarf::DW_LNS_negate_stmt, 1);
      LineSectionSize += 1;
    }
    if (Row.BasicBlock) {
      MS->emitIntValue(dwarf::DW_LNS_set_basic_block, 1);
      LineSectionSize += 1;
    }

    if (Row.PrologueEnd) {
      MS->emitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
      LineSectionSize += 1;
    }

    if (Row.EpilogueBegin) {
      MS->emitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);
      LineSectionSize += 1;
    }

    int64_t LineDelta = int64_t(Row.Line) - LastLine;
    if (!Row.EndSequence) {
      MCDwarfLineAddr::Encode(*MC, Params, LineDelta, AddressDelta, EncodingOS);
      MS->emitBytes(EncodingOS.str());
      LineSectionSize += EncodingBuffer.size();
      EncodingBuffer.resize(0);
      Address = Row.Address.Address;
      LastLine = Row.Line;
      RowsSinceLastSequence++;
    } else {
      if (LineDelta) {
        MS->emitIntValue(dwarf::DW_LNS_advance_line, 1);
        MS->emitSLEB128IntValue(LineDelta);
        LineSectionSize += 1 + getSLEB128Size(LineDelta);
      }
      if (AddressDelta) {
        MS->emitIntValue(dwarf::DW_LNS_advance_pc, 1);
        MS->emitULEB128IntValue(AddressDelta);
        LineSectionSize += 1 + getULEB128Size(AddressDelta);
      }
      MCDwarfLineAddr::Encode(*MC, Params, std::numeric_limits<int64_t>::max(),
                              0, EncodingOS);
      MS->emitBytes(EncodingOS.str());
      LineSectionSize += EncodingBuffer.size();
      EncodingBuffer.resize(0);
      Address = -1ULL;
      LastLine = FileNum = IsStatement = 1;
      RowsSinceLastSequence = Column = Isa = 0;
    }
  }

  if (RowsSinceLastSequence) {
    MCDwarfLineAddr::Encode(*MC, Params, std::numeric_limits<int64_t>::max(), 0,
                            EncodingOS);
    MS->emitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    EncodingBuffer.resize(0);
  }

  MS->emitLabel(LineEndSym);
}

/// Copy the debug_line over to the updated binary while unobfuscating the file
/// names and directories.
void DwarfStreamer::translateLineTable(DataExtractor Data, uint64_t Offset) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLineSection());
  StringRef Contents = Data.getData();

  // We have to deconstruct the line table header, because it contains to
  // length fields that will need to be updated when we change the length of
  // the files and directories in there.
  unsigned UnitLength = Data.getU32(&Offset);
  uint64_t UnitEnd = Offset + UnitLength;
  MCSymbol *BeginLabel = MC->createTempSymbol();
  MCSymbol *EndLabel = MC->createTempSymbol();
  unsigned Version = Data.getU16(&Offset);

  if (Version > 5) {
    warn("Unsupported line table version: dropping contents and not "
         "unobfsucating line table.");
    return;
  }

  Asm->emitLabelDifference(EndLabel, BeginLabel, 4);
  Asm->OutStreamer->emitLabel(BeginLabel);
  Asm->emitInt16(Version);
  LineSectionSize += 6;

  MCSymbol *HeaderBeginLabel = MC->createTempSymbol();
  MCSymbol *HeaderEndLabel = MC->createTempSymbol();
  Asm->emitLabelDifference(HeaderEndLabel, HeaderBeginLabel, 4);
  Asm->OutStreamer->emitLabel(HeaderBeginLabel);
  Offset += 4;
  LineSectionSize += 4;

  uint64_t AfterHeaderLengthOffset = Offset;
  // Skip to the directories.
  Offset += (Version >= 4) ? 5 : 4;
  unsigned OpcodeBase = Data.getU8(&Offset);
  Offset += OpcodeBase - 1;
  Asm->OutStreamer->emitBytes(Contents.slice(AfterHeaderLengthOffset, Offset));
  LineSectionSize += Offset - AfterHeaderLengthOffset;

  // Offset points to the first directory.
  while (const char *Dir = Data.getCStr(&Offset)) {
    if (Dir[0] == 0)
      break;

    StringRef Translated = Translator(Dir);
    Asm->OutStreamer->emitBytes(Translated);
    Asm->emitInt8(0);
    LineSectionSize += Translated.size() + 1;
  }
  Asm->emitInt8(0);
  LineSectionSize += 1;

  while (const char *File = Data.getCStr(&Offset)) {
    if (File[0] == 0)
      break;

    StringRef Translated = Translator(File);
    Asm->OutStreamer->emitBytes(Translated);
    Asm->emitInt8(0);
    LineSectionSize += Translated.size() + 1;

    uint64_t OffsetBeforeLEBs = Offset;
    Asm->emitULEB128(Data.getULEB128(&Offset));
    Asm->emitULEB128(Data.getULEB128(&Offset));
    Asm->emitULEB128(Data.getULEB128(&Offset));
    LineSectionSize += Offset - OffsetBeforeLEBs;
  }
  Asm->emitInt8(0);
  LineSectionSize += 1;

  Asm->OutStreamer->emitLabel(HeaderEndLabel);

  // Copy the actual line table program over.
  Asm->OutStreamer->emitBytes(Contents.slice(Offset, UnitEnd));
  LineSectionSize += UnitEnd - Offset;

  Asm->OutStreamer->emitLabel(EndLabel);
  Offset = UnitEnd;
}

/// Emit the pubnames or pubtypes section contribution for \p
/// Unit into \p Sec. The data is provided in \p Names.
void DwarfStreamer::emitPubSectionForUnit(
    MCSection *Sec, StringRef SecName, const CompileUnit &Unit,
    const std::vector<CompileUnit::AccelInfo> &Names) {
  if (Names.empty())
    return;

  // Start the dwarf pubnames section.
  Asm->OutStreamer->SwitchSection(Sec);
  MCSymbol *BeginLabel = Asm->createTempSymbol("pub" + SecName + "_begin");
  MCSymbol *EndLabel = Asm->createTempSymbol("pub" + SecName + "_end");

  bool HeaderEmitted = false;
  // Emit the pubnames for this compilation unit.
  for (const auto &Name : Names) {
    if (Name.SkipPubSection)
      continue;

    if (!HeaderEmitted) {
      // Emit the header.
      Asm->emitLabelDifference(EndLabel, BeginLabel, 4); // Length
      Asm->OutStreamer->emitLabel(BeginLabel);
      Asm->emitInt16(dwarf::DW_PUBNAMES_VERSION); // Version
      Asm->emitInt32(Unit.getStartOffset());      // Unit offset
      Asm->emitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset()); // Size
      HeaderEmitted = true;
    }
    Asm->emitInt32(Name.Die->getOffset());

    // Emit the string itself.
    Asm->OutStreamer->emitBytes(Name.Name.getString());
    // Emit a null terminator.
    Asm->emitInt8(0);
  }

  if (!HeaderEmitted)
    return;
  Asm->emitInt32(0); // End marker.
  Asm->OutStreamer->emitLabel(EndLabel);
}

/// Emit .debug_pubnames for \p Unit.
void DwarfStreamer::emitPubNamesForUnit(const CompileUnit &Unit) {
  emitPubSectionForUnit(MC->getObjectFileInfo()->getDwarfPubNamesSection(),
                        "names", Unit, Unit.getPubnames());
}

/// Emit .debug_pubtypes for \p Unit.
void DwarfStreamer::emitPubTypesForUnit(const CompileUnit &Unit) {
  emitPubSectionForUnit(MC->getObjectFileInfo()->getDwarfPubTypesSection(),
                        "types", Unit, Unit.getPubtypes());
}

/// Emit a CIE into the debug_frame section.
void DwarfStreamer::emitCIE(StringRef CIEBytes) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfFrameSection());

  MS->emitBytes(CIEBytes);
  FrameSectionSize += CIEBytes.size();
}

/// Emit a FDE into the debug_frame section. \p FDEBytes
/// contains the FDE data without the length, CIE offset and address
/// which will be replaced with the parameter values.
void DwarfStreamer::emitFDE(uint32_t CIEOffset, uint32_t AddrSize,
                            uint32_t Address, StringRef FDEBytes) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfFrameSection());

  MS->emitIntValue(FDEBytes.size() + 4 + AddrSize, 4);
  MS->emitIntValue(CIEOffset, 4);
  MS->emitIntValue(Address, AddrSize);
  MS->emitBytes(FDEBytes);
  FrameSectionSize += FDEBytes.size() + 8 + AddrSize;
}

} // namespace llvm
