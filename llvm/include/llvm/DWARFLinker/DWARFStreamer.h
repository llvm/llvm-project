//===- DwarfStreamer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFSTREAMER_H
#define LLVM_DWARFLINKER_DWARFSTREAMER_H

#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DWARFLinker/DWARFLinker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

enum class OutputFileType {
  Object,
  Assembly,
};

///   User of DwarfStreamer should call initialization code
///   for AsmPrinter:
///
///   InitializeAllTargetInfos();
///   InitializeAllTargetMCs();
///   InitializeAllTargets();
///   InitializeAllAsmPrinters();

class MCCodeEmitter;

/// The Dwarf streaming logic.
///
/// All interactions with the MC layer that is used to build the debug
/// information binary representation are handled in this class.
class DwarfStreamer : public DwarfEmitter {
public:
  DwarfStreamer(OutputFileType OutFileType, raw_pwrite_stream &OutFile,
                std::function<StringRef(StringRef Input)> Translator,
                bool Minimize, messageHandler Error, messageHandler Warning)
      : OutFile(OutFile), OutFileType(OutFileType), Translator(Translator),
        Minimize(Minimize), ErrorHandler(Error), WarningHandler(Warning) {}

  bool init(Triple TheTriple);

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
  void emitCompileUnitHeader(CompileUnit &Unit, unsigned DwarfVersion) override;

  /// Recursively emit the DIE tree rooted at \p Die.
  void emitDIE(DIE &Die) override;

  /// Emit the abbreviation table \p Abbrevs to the debug_abbrev section.
  void emitAbbrevs(const std::vector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
                   unsigned DwarfVersion) override;

  /// Emit DIE containing warnings.
  void emitPaperTrailWarningsDie(DIE &Die) override;

  /// Emit contents of section SecName From Obj.
  void emitSectionContents(StringRef SecData, StringRef SecName) override;

  /// Emit the string table described by \p Pool.
  void emitStrings(const NonRelocatableStringpool &Pool) override;

  /// Emit the swift_ast section stored in \p Buffer.
  void emitSwiftAST(StringRef Buffer);

  /// Emit debug_ranges for \p FuncRange by translating the
  /// original \p Entries.
  void emitRangesEntries(
      int64_t UnitPcOffset, uint64_t OrigLowPc,
      const FunctionIntervals::const_iterator &FuncRange,
      const std::vector<DWARFDebugRangeList::RangeListEntry> &Entries,
      unsigned AddressSize) override;

  /// Emit debug_aranges entries for \p Unit and if \p DoRangesSection is true,
  /// also emit the debug_ranges entries for the DW_TAG_compile_unit's
  /// DW_AT_ranges attribute.
  void emitUnitRangesEntries(CompileUnit &Unit, bool DoRangesSection) override;

  uint64_t getRangesSectionSize() const override { return RangesSectionSize; }

  /// Emit the debug_loc contribution for \p Unit by copying the entries from
  /// \p Dwarf and offsetting them. Update the location attributes to point to
  /// the new entries.
  void emitLocationsForUnit(
      const CompileUnit &Unit, DWARFContext &Dwarf,
      std::function<void(StringRef, SmallVectorImpl<uint8_t> &)> ProcessExpr)
      override;

  /// Emit the line table described in \p Rows into the debug_line section.
  void emitLineTableForUnit(MCDwarfLineTableParams Params,
                            StringRef PrologueBytes, unsigned MinInstLength,
                            std::vector<DWARFDebugLine::Row> &Rows,
                            unsigned AdddressSize) override;

  /// Copy the debug_line over to the updated binary while unobfuscating the
  /// file names and directories.
  void translateLineTable(DataExtractor LineData, uint64_t Offset) override;

  uint64_t getLineSectionSize() const override { return LineSectionSize; }

  /// Emit the .debug_pubnames contribution for \p Unit.
  void emitPubNamesForUnit(const CompileUnit &Unit) override;

  /// Emit the .debug_pubtypes contribution for \p Unit.
  void emitPubTypesForUnit(const CompileUnit &Unit) override;

  /// Emit a CIE.
  void emitCIE(StringRef CIEBytes) override;

  /// Emit an FDE with data \p Bytes.
  void emitFDE(uint32_t CIEOffset, uint32_t AddreSize, uint32_t Address,
               StringRef Bytes) override;

  /// Emit DWARF debug names.
  void emitDebugNames(AccelTable<DWARF5AccelTableStaticData> &Table) override;

  /// Emit Apple namespaces accelerator table.
  void emitAppleNamespaces(
      AccelTable<AppleAccelTableStaticOffsetData> &Table) override;

  /// Emit Apple names accelerator table.
  void
  emitAppleNames(AccelTable<AppleAccelTableStaticOffsetData> &Table) override;

  /// Emit Apple Objective-C accelerator table.
  void
  emitAppleObjc(AccelTable<AppleAccelTableStaticOffsetData> &Table) override;

  /// Emit Apple type accelerator table.
  void
  emitAppleTypes(AccelTable<AppleAccelTableStaticTypeData> &Table) override;

  uint64_t getFrameSectionSize() const override { return FrameSectionSize; }

  uint64_t getDebugInfoSectionSize() const override {
    return DebugInfoSectionSize;
  }

private:
  inline void error(const Twine &Error, StringRef Context = "") {
    if (ErrorHandler)
      ErrorHandler(Error, Context, nullptr);
  }

  inline void warn(const Twine &Warning, StringRef Context = "") {
    if (WarningHandler)
      WarningHandler(Warning, Context, nullptr);
  }

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
  OutputFileType OutFileType = OutputFileType::Object;
  std::function<StringRef(StringRef Input)> Translator;
  bool Minimize = true;

  uint64_t RangesSectionSize = 0;
  uint64_t LocSectionSize = 0;
  uint64_t LineSectionSize = 0;
  uint64_t FrameSectionSize = 0;
  uint64_t DebugInfoSectionSize = 0;

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

  messageHandler ErrorHandler = nullptr;
  messageHandler WarningHandler = nullptr;
};

} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFSTREAMER_H
