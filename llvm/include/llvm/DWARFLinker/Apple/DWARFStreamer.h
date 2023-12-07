//===- DwarfStreamer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_APPLE_DWARFSTREAMER_H
#define LLVM_DWARFLINKER_APPLE_DWARFSTREAMER_H

#include "llvm/BinaryFormat/Swift.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DWARFLinker/Apple/DWARFLinker.h"
#include "llvm/DWARFLinker/IndexedValuesMap.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
template <typename DataT> class AccelTable;

///   User of DwarfStreamer should call initialization code
///   for AsmPrinter:
///
///   InitializeAllTargetInfos();
///   InitializeAllTargetMCs();
///   InitializeAllTargets();
///   InitializeAllAsmPrinters();

class MCCodeEmitter;
class DWARFDebugMacro;

namespace dwarflinker {

/// The Dwarf streaming logic.
///
/// All interactions with the MC layer that is used to build the debug
/// information binary representation are handled in this class.
class DwarfStreamer {
public:
  DwarfStreamer(DWARFLinkerBase::OutputFileType OutFileType,
                raw_pwrite_stream &OutFile,
                std::function<StringRef(StringRef Input)> Translator,
                DWARFLinkerBase::MessageHandlerTy Warning)
      : OutFile(OutFile), OutFileType(OutFileType), Translator(Translator),
        WarningHandler(Warning) {}

  static Expected<DwarfStreamer *>
  createStreamer(const Triple &TheTriple,
                 DWARFLinkerBase::OutputFileType FileType,
                 raw_pwrite_stream &OutFile,
                 std::function<StringRef(StringRef Input)> Translator,
                 DWARFLinkerBase::MessageHandlerTy Warning);

  Error init(Triple TheTriple, StringRef Swift5ReflectionSegmentName);

  const Triple &getTargetTriple() { return TheTargetTriple; }

  /// Dump the file to the disk.
  void finish();

  AsmPrinter &getAsmPrinter() const { return *Asm; }

  /// Set the current output section to debug_info and change
  /// the MC Dwarf version to \p DwarfVersion.
  void switchToDebugInfoSection(unsigned DwarfVersion);

  /// Emit the compilation unit header for \p Unit in the
  /// debug_info section.
  ///
  /// As a side effect, this also switches the current Dwarf version
  /// of the MC layer to the one of U.getOrigUnit().
  void emitCompileUnitHeader(CompileUnit &Unit, unsigned DwarfVersion);

  /// Recursively emit the DIE tree rooted at \p Die.
  void emitDIE(DIE &Die);

  /// Emit the abbreviation table \p Abbrevs to the debug_abbrev section.
  void emitAbbrevs(const std::vector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
                   unsigned DwarfVersion);

  /// Emit contents of section SecName From Obj.
  void emitSectionContents(StringRef SecData, StringRef SecName);

  /// Emit the string table described by \p Pool into .debug_str table.
  void emitStrings(const NonRelocatableStringpool &Pool);

  /// Emit the debug string offset table described by \p StringOffsets into the
  /// .debug_str_offsets table.
  void emitStringOffsets(const SmallVector<uint64_t> &StringOffset,
                         uint16_t TargetDWARFVersion);

  /// Emit the string table described by \p Pool into .debug_line_str table.
  void emitLineStrings(const NonRelocatableStringpool &Pool);

  /// Emit the swift_ast section stored in \p Buffer.
  void emitSwiftAST(StringRef Buffer);

  /// Emit the swift reflection section stored in \p Buffer.
  void emitSwiftReflectionSection(
      llvm::binaryformat::Swift5ReflectionSectionKind ReflSectionKind,
      StringRef Buffer, uint32_t Alignment, uint32_t Size);

  /// Emit debug ranges(.debug_ranges, .debug_rnglists) header.
  MCSymbol *emitDwarfDebugRangeListHeader(const CompileUnit &Unit);

  /// Emit debug ranges(.debug_ranges, .debug_rnglists) fragment.
  void emitDwarfDebugRangeListFragment(const CompileUnit &Unit,
                                       const AddressRanges &LinkedRanges,
                                       PatchLocation Patch,
                                       IndexedValuesMap<uint64_t> &AddrPool);

  /// Emit debug ranges(.debug_ranges, .debug_rnglists) footer.
  void emitDwarfDebugRangeListFooter(const CompileUnit &Unit,
                                     MCSymbol *EndLabel);

  /// Emit debug locations(.debug_loc, .debug_loclists) header.
  MCSymbol *emitDwarfDebugLocListHeader(const CompileUnit &Unit);

  /// Emit .debug_addr header.
  MCSymbol *emitDwarfDebugAddrsHeader(const CompileUnit &Unit);

  /// Emit the addresses described by \p Addrs into .debug_addr table.
  void emitDwarfDebugAddrs(const SmallVector<uint64_t> &Addrs,
                           uint8_t AddrSize);

  /// Emit .debug_addr footer.
  void emitDwarfDebugAddrsFooter(const CompileUnit &Unit, MCSymbol *EndLabel);

  /// Emit debug ranges(.debug_loc, .debug_loclists) fragment.
  void emitDwarfDebugLocListFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      PatchLocation Patch, IndexedValuesMap<uint64_t> &AddrPool);

  /// Emit debug ranges(.debug_loc, .debug_loclists) footer.
  void emitDwarfDebugLocListFooter(const CompileUnit &Unit, MCSymbol *EndLabel);

  /// Emit .debug_aranges entries for \p Unit
  void emitDwarfDebugArangesTable(const CompileUnit &Unit,
                                  const AddressRanges &LinkedRanges);

  uint64_t getRangesSectionSize() const { return RangesSectionSize; }

  uint64_t getRngListsSectionSize() const { return RngListsSectionSize; }

  /// Emit .debug_line table entry for specified \p LineTable
  void emitLineTableForUnit(const DWARFDebugLine::LineTable &LineTable,
                            const CompileUnit &Unit,
                            OffsetsStringPool &DebugStrPool,
                            OffsetsStringPool &DebugLineStrPool);

  uint64_t getLineSectionSize() const { return LineSectionSize; }

  /// Emit the .debug_pubnames contribution for \p Unit.
  void emitPubNamesForUnit(const CompileUnit &Unit);

  /// Emit the .debug_pubtypes contribution for \p Unit.
  void emitPubTypesForUnit(const CompileUnit &Unit);

  /// Emit a CIE.
  void emitCIE(StringRef CIEBytes);

  /// Emit an FDE with data \p Bytes.
  void emitFDE(uint32_t CIEOffset, uint32_t AddreSize, uint64_t Address,
               StringRef Bytes);

  /// Emit DWARF debug names.
  void emitDebugNames(DWARF5AccelTable &Table);

  /// Emit Apple namespaces accelerator table.
  void emitAppleNamespaces(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emit Apple names accelerator table.
  void emitAppleNames(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emit Apple Objective-C accelerator table.
  void emitAppleObjc(AccelTable<AppleAccelTableStaticOffsetData> &Table);

  /// Emit Apple type accelerator table.
  void emitAppleTypes(AccelTable<AppleAccelTableStaticTypeData> &Table);

  uint64_t getFrameSectionSize() const { return FrameSectionSize; }

  uint64_t getDebugInfoSectionSize() const { return DebugInfoSectionSize; }

  uint64_t getDebugMacInfoSectionSize() const { return MacInfoSectionSize; }

  uint64_t getDebugMacroSectionSize() const { return MacroSectionSize; }

  uint64_t getLocListsSectionSize() const { return LocListsSectionSize; }

  uint64_t getDebugAddrSectionSize() const { return AddrSectionSize; }

  void emitMacroTables(DWARFContext *Context,
                       const Offset2UnitMap &UnitMacroMap,
                       OffsetsStringPool &StringPool);

private:
  inline void warn(const Twine &Warning, StringRef Context = "") {
    if (WarningHandler)
      WarningHandler(Warning, Context, nullptr);
  }

  MCSection *switchSection(StringRef SecName);

  void emitMacroTableImpl(const DWARFDebugMacro *MacroTable,
                          const Offset2UnitMap &UnitMacroMap,
                          OffsetsStringPool &StringPool, uint64_t &OutOffset);

  /// Emit piece of .debug_ranges for \p LinkedRanges.
  void emitDwarfDebugRangesTableFragment(const CompileUnit &Unit,
                                         const AddressRanges &LinkedRanges,
                                         PatchLocation Patch);

  /// Emit piece of .debug_rnglists for \p LinkedRanges.
  void emitDwarfDebugRngListsTableFragment(
      const CompileUnit &Unit, const AddressRanges &LinkedRanges,
      PatchLocation Patch, IndexedValuesMap<uint64_t> &AddrPool);

  /// Emit piece of .debug_loc for \p LinkedRanges.
  void emitDwarfDebugLocTableFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      PatchLocation Patch);

  /// Emit piece of .debug_loclists for \p LinkedRanges.
  void emitDwarfDebugLocListsTableFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      PatchLocation Patch, IndexedValuesMap<uint64_t> &AddrPool);

  /// \defgroup Line table emission
  /// @{
  void emitLineTablePrologue(const DWARFDebugLine::Prologue &P,
                             OffsetsStringPool &DebugStrPool,
                             OffsetsStringPool &DebugLineStrPool);
  void emitLineTableString(const DWARFDebugLine::Prologue &P,
                           const DWARFFormValue &String,
                           OffsetsStringPool &DebugStrPool,
                           OffsetsStringPool &DebugLineStrPool);
  void emitLineTableProloguePayload(const DWARFDebugLine::Prologue &P,
                                    OffsetsStringPool &DebugStrPool,
                                    OffsetsStringPool &DebugLineStrPool);
  void emitLineTablePrologueV2IncludeAndFileTable(
      const DWARFDebugLine::Prologue &P, OffsetsStringPool &DebugStrPool,
      OffsetsStringPool &DebugLineStrPool);
  void emitLineTablePrologueV5IncludeAndFileTable(
      const DWARFDebugLine::Prologue &P, OffsetsStringPool &DebugStrPool,
      OffsetsStringPool &DebugLineStrPool);
  void emitLineTableRows(const DWARFDebugLine::LineTable &LineTable,
                         MCSymbol *LineEndSym, unsigned AddressByteSize);
  void emitIntOffset(uint64_t Offset, dwarf::DwarfFormat Format,
                     uint64_t &SectionSize);
  void emitLabelDifference(const MCSymbol *Hi, const MCSymbol *Lo,
                           dwarf::DwarfFormat Format, uint64_t &SectionSize);
  /// @}

  /// \defgroup MCObjects MC layer objects constructed by the streamer
  /// @{
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCContext> MC;
  MCAsmBackend *MAB; // Owned by MCStreamer
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCSubtargetInfo> MSTI;
  MCInstPrinter *MIP; // Owned by AsmPrinter
  MCCodeEmitter *MCE; // Owned by MCStreamer
  MCStreamer *MS;     // Owned by AsmPrinter
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<AsmPrinter> Asm;
  /// @}

  /// The output file we stream the linked Dwarf to.
  raw_pwrite_stream &OutFile;
  DWARFLinkerBase::OutputFileType OutFileType =
      DWARFLinkerBase::OutputFileType::Object;
  std::function<StringRef(StringRef Input)> Translator;

  Triple TheTargetTriple;

  uint64_t RangesSectionSize = 0;
  uint64_t RngListsSectionSize = 0;
  uint64_t LocSectionSize = 0;
  uint64_t LocListsSectionSize = 0;
  uint64_t LineSectionSize = 0;
  uint64_t FrameSectionSize = 0;
  uint64_t DebugInfoSectionSize = 0;
  uint64_t MacInfoSectionSize = 0;
  uint64_t MacroSectionSize = 0;
  uint64_t AddrSectionSize = 0;
  uint64_t StrOffsetSectionSize = 0;

  /// Keep track of emitted CUs and their Unique ID.
  struct EmittedUnit {
    unsigned ID;
    MCSymbol *LabelBegin;
  };
  std::vector<EmittedUnit> EmittedUnits;

  /// Emit the pubnames or pubtypes section contribution for \p
  /// Unit into \p Sec. The data is provided in \p Names.
  void emitPubSectionForUnit(MCSection *Sec, StringRef Name,
                             const CompileUnit &Unit,
                             const std::vector<CompileUnit::AccelInfo> &Names);

  DWARFLinkerBase::MessageHandlerTy WarningHandler = nullptr;
};

} // end namespace dwarflinker
} // end namespace llvm

#endif // LLVM_DWARFLINKER_APPLE_DWARFSTREAMER_H
