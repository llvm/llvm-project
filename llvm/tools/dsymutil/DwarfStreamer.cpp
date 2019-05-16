//===- tools/dsymutil/DwarfStreamer.cpp - Dwarf Streamer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfStreamer.h"
#include "CompileUnit.h"
#include "LinkUtils.h"
#include "MachOUtils.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.inc"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
namespace dsymutil {

/// Retrieve the section named \a SecName in \a Obj.
///
/// To accommodate for platform discrepancies, the name passed should be
/// (for example) 'debug_info' to match either '__debug_info' or '.debug_info'.
/// This function will strip the initial platform-specific characters.
static Optional<object::SectionRef>
getSectionByName(const object::ObjectFile &Obj, StringRef SecName) {
  for (const object::SectionRef &Section : Obj.sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    SectionName = SectionName.substr(SectionName.find_first_not_of("._"));
    if (SectionName != SecName)
      continue;
    return Section;
  }
  return None;
}

bool DwarfStreamer::init(Triple TheTriple) {
  std::string ErrorStr;
  std::string TripleName;
  StringRef Context = "dwarf streamer init";

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TripleName, TheTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  MOFI.reset(new MCObjectFileInfo);
  MC.reset(new MCContext(MAI.get(), MRI.get(), MOFI.get()));
  MOFI->InitMCObjectFileInfo(TheTriple, /*PIC*/ false, *MC);

  MSTI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCTargetOptions MCOptions = InitMCTargetOptionsFromFlags();
  MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions);
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, *MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  switch (Options.FileType) {
  case OutputFileType::Assembly: {
    MIP = TheTarget->createMCInstPrinter(TheTriple, MAI->getAssemblerDialect(),
                                         *MAI, *MII, *MRI);
    MS = TheTarget->createAsmStreamer(
        *MC, llvm::make_unique<formatted_raw_ostream>(OutFile), true, true, MIP,
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
    return error("no object streamer for target " + TripleName, Context);

  // Finally create the AsmPrinter we'll use to emit the DIEs.
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          None));
  if (!TM)
    return error("no target machine for target " + TripleName, Context);

  Asm.reset(TheTarget->createAsmPrinter(*TM, std::unique_ptr<MCStreamer>(MS)));
  if (!Asm)
    return error("no asm printer for target " + TripleName, Context);

  RangesSectionSize = 0;
  LocSectionSize = 0;
  LineSectionSize = 0;
  FrameSectionSize = 0;

  return true;
}

bool DwarfStreamer::finish(const DebugMap &DM, SymbolMapTranslator &T) {
  bool Result = true;
  if (DM.getTriple().isOSDarwin() && !DM.getBinaryPath().empty() &&
      Options.FileType == OutputFileType::Object)
    Result = MachOUtils::generateDsymCompanion(DM, T, *MS, OutFile);
  else
    MS->Finish();
  return Result;
}

void DwarfStreamer::switchToDebugInfoSection(unsigned DwarfVersion) {
  MS->SwitchSection(MOFI->getDwarfInfoSection());
  MC->setDwarfVersion(DwarfVersion);
}

/// Emit the compilation unit header for \p Unit in the debug_info section.
///
/// A Dwarf section header is encoded as:
///  uint32_t   Unit length (omitting this field)
///  uint16_t   Version
///  uint32_t   Abbreviation table offset
///  uint8_t    Address size
///
/// Leading to a total of 11 bytes.
void DwarfStreamer::emitCompileUnitHeader(CompileUnit &Unit) {
  unsigned Version = Unit.getOrigUnit().getVersion();
  switchToDebugInfoSection(Version);

  /// The start of the unit within its section.
  Unit.setLabelBegin(Asm->createTempSymbol("cu_begin"));
  Asm->OutStreamer->EmitLabel(Unit.getLabelBegin());

  // Emit size of content not including length itself. The size has already
  // been computed in CompileUnit::computeOffsets(). Subtract 4 to that size to
  // account for the length field.
  Asm->emitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset() - 4);
  Asm->emitInt16(Version);

  // We share one abbreviations table across all units so it's always at the
  // start of the section.
  Asm->emitInt32(0);
  Asm->emitInt8(Unit.getOrigUnit().getAddressByteSize());

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
}

/// Emit the debug_str section stored in \p Pool.
void DwarfStreamer::emitStrings(const NonRelocatableStringpool &Pool) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfStrSection());
  std::vector<DwarfStringPoolEntryRef> Entries = Pool.getEntriesForEmission();
  for (auto Entry : Entries) {
    // Emit the string itself.
    Asm->OutStreamer->EmitBytes(Entry.getString());
    // Emit a null terminator.
    Asm->emitInt8(0);
  }
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
  Asm->OutStreamer->EmitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "namespac", SectionBegin);
}

void DwarfStreamer::emitAppleNames(
    AccelTable<AppleAccelTableStaticOffsetData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelNamesSection());
  auto *SectionBegin = Asm->createTempSymbol("names_begin");
  Asm->OutStreamer->EmitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "names", SectionBegin);
}

void DwarfStreamer::emitAppleObjc(
    AccelTable<AppleAccelTableStaticOffsetData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelObjCSection());
  auto *SectionBegin = Asm->createTempSymbol("objc_begin");
  Asm->OutStreamer->EmitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "objc", SectionBegin);
}

void DwarfStreamer::emitAppleTypes(
    AccelTable<AppleAccelTableStaticTypeData> &Table) {
  Asm->OutStreamer->SwitchSection(MOFI->getDwarfAccelTypesSection());
  auto *SectionBegin = Asm->createTempSymbol("types_begin");
  Asm->OutStreamer->EmitLabel(SectionBegin);
  emitAppleAccelTable(Asm.get(), Table, "types", SectionBegin);
}

/// Emit the swift_ast section stored in \p Buffers.
void DwarfStreamer::emitSwiftAST(StringRef Buffer) {
  MCSection *SwiftASTSection = MOFI->getDwarfSwiftASTSection();
  SwiftASTSection->setAlignment(1 << 5);
  MS->SwitchSection(SwiftASTSection);
  MS->EmitBytes(Buffer);
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
    MS->EmitIntValue(Range.StartAddress + PcOffset, AddressSize);
    MS->EmitIntValue(Range.EndAddress + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->EmitIntValue(0, AddressSize);
  MS->EmitIntValue(0, AddressSize);
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
    unsigned Padding = OffsetToAlignment(HeaderSize, TupleSize);

    Asm->EmitLabelDifference(EndLabel, BeginLabel, 4); // Arange length
    Asm->OutStreamer->EmitLabel(BeginLabel);
    Asm->emitInt16(dwarf::DW_ARANGES_VERSION); // Version number
    Asm->emitInt32(Unit.getStartOffset());     // Corresponding unit's offset
    Asm->emitInt8(AddressSize);                // Address size
    Asm->emitInt8(0);                          // Segment size

    Asm->OutStreamer->emitFill(Padding, 0x0);

    for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End;
         ++Range) {
      uint64_t RangeStart = Range->first;
      MS->EmitIntValue(RangeStart, AddressSize);
      while ((Range + 1) != End && Range->second == (Range + 1)->first)
        ++Range;
      MS->EmitIntValue(Range->second - RangeStart, AddressSize);
    }

    // Emit terminator
    Asm->OutStreamer->EmitIntValue(0, AddressSize);
    Asm->OutStreamer->EmitIntValue(0, AddressSize);
    Asm->OutStreamer->EmitLabel(EndLabel);
  }

  if (!DoDebugRanges)
    return;

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());
  // Offset each range by the right amount.
  int64_t PcOffset = -Unit.getLowPc();
  // Emit coalesced ranges.
  for (auto Range = Ranges.begin(), End = Ranges.end(); Range != End; ++Range) {
    MS->EmitIntValue(Range->first + PcOffset, AddressSize);
    while (Range + 1 != End && Range->second == (Range + 1)->first)
      ++Range;
    MS->EmitIntValue(Range->second + PcOffset, AddressSize);
    RangesSectionSize += 2 * AddressSize;
  }

  // Add the terminator entry.
  MS->EmitIntValue(0, AddressSize);
  MS->EmitIntValue(0, AddressSize);
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
  const DWARFSection &InputSec = Dwarf.getDWARFObj().getLocSection();
  DataExtractor Data(InputSec.Data, Dwarf.isLittleEndian(), AddressSize);
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  auto OrigUnitDie = OrigUnit.getUnitDIE(false);
  int64_t UnitPcOffset = 0;
  if (auto OrigLowPc = dwarf::toAddress(OrigUnitDie.find(dwarf::DW_AT_low_pc)))
    UnitPcOffset = int64_t(*OrigLowPc) - Unit.getLowPc();

  SmallVector<uint8_t, 32> Buffer;
  for (const auto &Attr : Attributes) {
    uint32_t Offset = Attr.first.get();
    Attr.first.set(LocSectionSize);
    // This is the quantity to add to the old location address to get
    // the correct address for the new one.
    int64_t LocPcOffset = Attr.second + UnitPcOffset;
    while (Data.isValidOffset(Offset)) {
      uint64_t Low = Data.getUnsigned(&Offset, AddressSize);
      uint64_t High = Data.getUnsigned(&Offset, AddressSize);
      LocSectionSize += 2 * AddressSize;
      if (Low == 0 && High == 0) {
        Asm->OutStreamer->EmitIntValue(0, AddressSize);
        Asm->OutStreamer->EmitIntValue(0, AddressSize);
        break;
      }
      Asm->OutStreamer->EmitIntValue(Low + LocPcOffset, AddressSize);
      Asm->OutStreamer->EmitIntValue(High + LocPcOffset, AddressSize);
      uint64_t Length = Data.getU16(&Offset);
      Asm->OutStreamer->EmitIntValue(Length, 2);
      // Copy the bytes into to the buffer, process them, emit them.
      Buffer.reserve(Length);
      Buffer.resize(0);
      StringRef Input = InputSec.Data.substr(Offset, Length);
      ProcessExpr(Input, Buffer);
      Asm->OutStreamer->EmitBytes(
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
  Asm->EmitLabelDifference(LineEndSym, LineStartSym, 4);
  Asm->OutStreamer->EmitLabel(LineStartSym);
  // Copy Prologue.
  MS->EmitBytes(PrologueBytes);
  LineSectionSize += PrologueBytes.size() + 4;

  SmallString<128> EncodingBuffer;
  raw_svector_ostream EncodingOS(EncodingBuffer);

  if (Rows.empty()) {
    // We only have the dummy entry, dsymutil emits an entry with a 0
    // address in that case.
    MCDwarfLineAddr::Encode(*MC, Params, std::numeric_limits<int64_t>::max(), 0,
                            EncodingOS);
    MS->EmitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    MS->EmitLabel(LineEndSym);
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
      MS->EmitIntValue(dwarf::DW_LNS_extended_op, 1);
      MS->EmitULEB128IntValue(PointerSize + 1);
      MS->EmitIntValue(dwarf::DW_LNE_set_address, 1);
      MS->EmitIntValue(Row.Address.Address, PointerSize);
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
      MS->EmitIntValue(dwarf::DW_LNS_set_file, 1);
      MS->EmitULEB128IntValue(FileNum);
      LineSectionSize += 1 + getULEB128Size(FileNum);
    }
    if (Column != Row.Column) {
      Column = Row.Column;
      MS->EmitIntValue(dwarf::DW_LNS_set_column, 1);
      MS->EmitULEB128IntValue(Column);
      LineSectionSize += 1 + getULEB128Size(Column);
    }

    // FIXME: We should handle the discriminator here, but dsymutil doesn't
    // consider it, thus ignore it for now.

    if (Isa != Row.Isa) {
      Isa = Row.Isa;
      MS->EmitIntValue(dwarf::DW_LNS_set_isa, 1);
      MS->EmitULEB128IntValue(Isa);
      LineSectionSize += 1 + getULEB128Size(Isa);
    }
    if (IsStatement != Row.IsStmt) {
      IsStatement = Row.IsStmt;
      MS->EmitIntValue(dwarf::DW_LNS_negate_stmt, 1);
      LineSectionSize += 1;
    }
    if (Row.BasicBlock) {
      MS->EmitIntValue(dwarf::DW_LNS_set_basic_block, 1);
      LineSectionSize += 1;
    }

    if (Row.PrologueEnd) {
      MS->EmitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
      LineSectionSize += 1;
    }

    if (Row.EpilogueBegin) {
      MS->EmitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);
      LineSectionSize += 1;
    }

    int64_t LineDelta = int64_t(Row.Line) - LastLine;
    if (!Row.EndSequence) {
      MCDwarfLineAddr::Encode(*MC, Params, LineDelta, AddressDelta, EncodingOS);
      MS->EmitBytes(EncodingOS.str());
      LineSectionSize += EncodingBuffer.size();
      EncodingBuffer.resize(0);
      Address = Row.Address.Address;
      LastLine = Row.Line;
      RowsSinceLastSequence++;
    } else {
      if (LineDelta) {
        MS->EmitIntValue(dwarf::DW_LNS_advance_line, 1);
        MS->EmitSLEB128IntValue(LineDelta);
        LineSectionSize += 1 + getSLEB128Size(LineDelta);
      }
      if (AddressDelta) {
        MS->EmitIntValue(dwarf::DW_LNS_advance_pc, 1);
        MS->EmitULEB128IntValue(AddressDelta);
        LineSectionSize += 1 + getULEB128Size(AddressDelta);
      }
      MCDwarfLineAddr::Encode(*MC, Params, std::numeric_limits<int64_t>::max(),
                              0, EncodingOS);
      MS->EmitBytes(EncodingOS.str());
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
    MS->EmitBytes(EncodingOS.str());
    LineSectionSize += EncodingBuffer.size();
    EncodingBuffer.resize(0);
  }

  MS->EmitLabel(LineEndSym);
}

/// Copy the debug_line over to the updated binary while unobfuscating the file
/// names and directories.
void DwarfStreamer::translateLineTable(DataExtractor Data, uint32_t Offset,
                                       LinkOptions &Options) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLineSection());
  StringRef Contents = Data.getData();

  // We have to deconstruct the line table header, because it contains to
  // length fields that will need to be updated when we change the length of
  // the files and directories in there.
  unsigned UnitLength = Data.getU32(&Offset);
  unsigned UnitEnd = Offset + UnitLength;
  MCSymbol *BeginLabel = MC->createTempSymbol();
  MCSymbol *EndLabel = MC->createTempSymbol();
  unsigned Version = Data.getU16(&Offset);

  if (Version > 5) {
    warn("Unsupported line table version: dropping contents and not "
         "unobfsucating line table.");
    return;
  }

  Asm->EmitLabelDifference(EndLabel, BeginLabel, 4);
  Asm->OutStreamer->EmitLabel(BeginLabel);
  Asm->emitInt16(Version);
  LineSectionSize += 6;

  MCSymbol *HeaderBeginLabel = MC->createTempSymbol();
  MCSymbol *HeaderEndLabel = MC->createTempSymbol();
  Asm->EmitLabelDifference(HeaderEndLabel, HeaderBeginLabel, 4);
  Asm->OutStreamer->EmitLabel(HeaderBeginLabel);
  Offset += 4;
  LineSectionSize += 4;

  uint32_t AfterHeaderLengthOffset = Offset;
  // Skip to the directories.
  Offset += (Version >= 4) ? 5 : 4;
  unsigned OpcodeBase = Data.getU8(&Offset);
  Offset += OpcodeBase - 1;
  Asm->OutStreamer->EmitBytes(Contents.slice(AfterHeaderLengthOffset, Offset));
  LineSectionSize += Offset - AfterHeaderLengthOffset;

  // Offset points to the first directory.
  while (const char *Dir = Data.getCStr(&Offset)) {
    if (Dir[0] == 0)
      break;

    StringRef Translated = Options.Translator(Dir);
    Asm->OutStreamer->EmitBytes(Translated);
    Asm->emitInt8(0);
    LineSectionSize += Translated.size() + 1;
  }
  Asm->emitInt8(0);
  LineSectionSize += 1;

  while (const char *File = Data.getCStr(&Offset)) {
    if (File[0] == 0)
      break;

    StringRef Translated = Options.Translator(File);
    Asm->OutStreamer->EmitBytes(Translated);
    Asm->emitInt8(0);
    LineSectionSize += Translated.size() + 1;

    uint32_t OffsetBeforeLEBs = Offset;
    Asm->EmitULEB128(Data.getULEB128(&Offset));
    Asm->EmitULEB128(Data.getULEB128(&Offset));
    Asm->EmitULEB128(Data.getULEB128(&Offset));
    LineSectionSize += Offset - OffsetBeforeLEBs;
  }
  Asm->emitInt8(0);
  LineSectionSize += 1;

  Asm->OutStreamer->EmitLabel(HeaderEndLabel);

  // Copy the actual line table program over.
  Asm->OutStreamer->EmitBytes(Contents.slice(Offset, UnitEnd));
  LineSectionSize += UnitEnd - Offset;

  Asm->OutStreamer->EmitLabel(EndLabel);
  Offset = UnitEnd;
}

static void emitSectionContents(const object::ObjectFile &Obj,
                                StringRef SecName, MCStreamer *MS) {
  if (auto Sec = getSectionByName(Obj, SecName)) {
    if (Expected<StringRef> E = Sec->getContents())
      MS->EmitBytes(*E);
    else
      consumeError(E.takeError());
  }
}

void DwarfStreamer::copyInvariantDebugSection(const object::ObjectFile &Obj) {
  if (!Options.Translator) {
    MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLineSection());
    emitSectionContents(Obj, "debug_line", MS);
  }

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfLocSection());
  emitSectionContents(Obj, "debug_loc", MS);

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfRangesSection());
  emitSectionContents(Obj, "debug_ranges", MS);

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfFrameSection());
  emitSectionContents(Obj, "debug_frame", MS);

  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfARangesSection());
  emitSectionContents(Obj, "debug_aranges", MS);
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
      Asm->EmitLabelDifference(EndLabel, BeginLabel, 4); // Length
      Asm->OutStreamer->EmitLabel(BeginLabel);
      Asm->emitInt16(dwarf::DW_PUBNAMES_VERSION); // Version
      Asm->emitInt32(Unit.getStartOffset());      // Unit offset
      Asm->emitInt32(Unit.getNextUnitOffset() - Unit.getStartOffset()); // Size
      HeaderEmitted = true;
    }
    Asm->emitInt32(Name.Die->getOffset());

    // Emit the string itself.
    Asm->OutStreamer->EmitBytes(Name.Name.getString());
    // Emit a null terminator.
    Asm->emitInt8(0);
  }

  if (!HeaderEmitted)
    return;
  Asm->emitInt32(0); // End marker.
  Asm->OutStreamer->EmitLabel(EndLabel);
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

  MS->EmitBytes(CIEBytes);
  FrameSectionSize += CIEBytes.size();
}

/// Emit a FDE into the debug_frame section. \p FDEBytes
/// contains the FDE data without the length, CIE offset and address
/// which will be replaced with the parameter values.
void DwarfStreamer::emitFDE(uint32_t CIEOffset, uint32_t AddrSize,
                            uint32_t Address, StringRef FDEBytes) {
  MS->SwitchSection(MC->getObjectFileInfo()->getDwarfFrameSection());

  MS->EmitIntValue(FDEBytes.size() + 4 + AddrSize, 4);
  MS->EmitIntValue(CIEOffset, 4);
  MS->EmitIntValue(Address, AddrSize);
  MS->EmitBytes(FDEBytes);
  FrameSectionSize += FDEBytes.size() + 8 + AddrSize;
}

} // namespace dsymutil
} // namespace llvm
