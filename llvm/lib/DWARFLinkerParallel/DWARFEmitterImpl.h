//===- DwarfEmitterImpl.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H

#include "DWARFLinkerCompileUnit.h"
#include "llvm/BinaryFormat/Swift.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/DWARFLinkerParallel/StringTable.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

///   User of DwarfEmitterImpl should call initialization code
///   for AsmPrinter:
///
///   InitializeAllTargetInfos();
///   InitializeAllTargetMCs();
///   InitializeAllTargets();
///   InitializeAllAsmPrinters();

template <typename DataT> class AccelTable;
class MCCodeEmitter;
class DWARFDebugMacro;

namespace dwarflinker_parallel {

struct UnitStartSymbol {
  unsigned UnitID = 0;
  MCSymbol *Symbol = 0;
};
using UnitStartSymbolsTy = SmallVector<UnitStartSymbol>;
using Offset2UnitMapTy = DenseMap<uint64_t, CompileUnit *>;

struct RangeAttrPatch;
struct LocAttrPatch;

/// The Dwarf emission logic.
///
/// All interactions with the MC layer that is used to build the debug
/// information binary representation are handled in this class.
class DwarfEmitterImpl : public ExtraDwarfEmitter {
public:
  DwarfEmitterImpl(DWARFLinker::OutputFileType OutFileType,
                   raw_pwrite_stream &OutFile,
                   std::function<StringRef(StringRef Input)> Translator,
                   DWARFLinker::MessageHandlerTy Warning)
      : OutFile(OutFile), OutFileType(OutFileType), Translator(Translator),
        WarningHandler(Warning) {}

  Error init(Triple TheTriple, StringRef Swift5ReflectionSegmentName);

  /// Dump the file to the disk.
  void finish() override { MS->finish(); }

  AsmPrinter &getAsmPrinter() const override { return *Asm; }

  /// Set the current output section to debug_info and change
  /// the MC Dwarf version to \p DwarfVersion.
  void switchToDebugInfoSection(unsigned DwarfVersion) {}

  /// Emit the swift_ast section stored in \p Buffer.
  void emitSwiftAST(StringRef Buffer) override {}

  /// Emit the swift reflection section stored in \p Buffer.
  void emitSwiftReflectionSection(
      llvm::binaryformat::Swift5ReflectionSectionKind ReflSectionKind,
      StringRef Buffer, uint32_t Alignment, uint32_t Size) override {}

  void emitPaperTrailWarningsDie(DIE &Die) {}

  void emitSectionContents(StringRef SecData, StringRef SecName) override {}

  MCSymbol *emitTempSym(StringRef SecName, StringRef SymName) override {
    return nullptr;
  }

  void emitAbbrevs(const SmallVector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
                   unsigned DwarfVersion) {}

  void emitStrings(const StringTable &Strings) {}

  void emitLineStrings(const StringTable &Strings) {}

  void emitDebugNames(AccelTable<DWARF5AccelTableStaticData> &,
                      UnitStartSymbolsTy &UnitOffsets) {}

  void emitAppleNamespaces(AccelTable<AppleAccelTableStaticOffsetData> &) {}

  void emitAppleNames(AccelTable<AppleAccelTableStaticOffsetData> &) {}

  void emitAppleObjc(AccelTable<AppleAccelTableStaticOffsetData> &) {}

  void emitAppleTypes(AccelTable<AppleAccelTableStaticTypeData> &) {}

  MCSymbol *emitDwarfDebugRangeListHeader(const CompileUnit &Unit) {
    return nullptr;
  }

  void emitDwarfDebugRangeListFragment(const CompileUnit &Unit,
                                       const AddressRanges &LinkedRanges,
                                       RangeAttrPatch &Patch) {}

  void emitDwarfDebugRangeListFooter(const CompileUnit &Unit,
                                     MCSymbol *EndLabel) {}

  MCSymbol *emitDwarfDebugLocListHeader(const CompileUnit &Unit) {
    return nullptr;
  }

  void emitDwarfDebugLocListFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      LocAttrPatch &Patch) {}

  void emitDwarfDebugLocListFooter(const CompileUnit &Unit,
                                   MCSymbol *EndLabel) {}

  void emitDwarfDebugArangesTable(const CompileUnit &Unit,
                                  const AddressRanges &LinkedRanges) {}

  void translateLineTable(DataExtractor LineData, uint64_t Offset) {}

  void emitLineTableForUnit(MCDwarfLineTableParams Params,
                            StringRef PrologueBytes, unsigned MinInstLength,
                            std::vector<DWARFDebugLine::Row> &Rows,
                            unsigned AdddressSize) {}

  void emitLineTableForUnit(const DWARFDebugLine::LineTable &LineTable,
                            const CompileUnit &Unit, const StringTable &Strings,
                            const StringTable &LineTableStrings) {}

  void emitPubNamesForUnit(const CompileUnit &Unit) {}

  void emitPubTypesForUnit(const CompileUnit &Unit) {}

  void emitCIE(StringRef CIEBytes) {}

  void emitFDE(uint32_t CIEOffset, uint32_t AddreSize, uint64_t Address,
               StringRef Bytes) {}

  void emitCompileUnitHeader(CompileUnit &Unit, unsigned DwarfVersion) {}

  void emitDIE(DIE &Die) {}

  void emitMacroTables(DWARFContext *Context,
                       const Offset2UnitMapTy &UnitMacroMap,
                       StringTable &Strings) {}

  /// Returns size of generated .debug_line section.
  uint64_t getDebugLineSectionSize() const { return LineSectionSize; }

  /// Returns size of generated .debug_frame section.
  uint64_t getDebugFrameSectionSize() const { return FrameSectionSize; }

  /// Returns size of generated .debug_ranges section.
  uint64_t getDebugRangesSectionSize() const { return RangesSectionSize; }

  /// Returns size of generated .debug_rnglists section.
  uint64_t getDebugRngListsSectionSize() const { return RngListsSectionSize; }

  /// Returns size of generated .debug_info section.
  uint64_t getDebugInfoSectionSize() const { return DebugInfoSectionSize; }

  /// Returns size of generated .debug_macinfo section.
  uint64_t getDebugMacInfoSectionSize() const { return MacInfoSectionSize; }

  /// Returns size of generated .debug_macro section.
  uint64_t getDebugMacroSectionSize() const { return MacroSectionSize; }

  /// Returns size of generated .debug_loc section.
  uint64_t getDebugLocSectionSize() const { return LocSectionSize; }

  /// Returns size of generated .debug_loclists section.
  uint64_t getDebugLocListsSectionSize() const { return LocListsSectionSize; }

private:
  inline void warn(const Twine &Warning, StringRef Context = "") {
    if (WarningHandler)
      WarningHandler(Warning, Context, nullptr);
  }

  void emitMacroTableImpl(const DWARFDebugMacro *MacroTable,
                          const Offset2UnitMapTy &UnitMacroMap,
                          StringPool &StringPool, uint64_t &OutOffset) {}

  /// Emit piece of .debug_ranges for \p LinkedRanges.
  void emitDwarfDebugRangesTableFragment(const CompileUnit &Unit,
                                         const AddressRanges &LinkedRanges,
                                         RangeAttrPatch &Patch) {}

  /// Emit piece of .debug_rnglists for \p LinkedRanges.
  void emitDwarfDebugRngListsTableFragment(const CompileUnit &Unit,
                                           const AddressRanges &LinkedRanges,
                                           RangeAttrPatch &Patch) {}

  /// Emit piece of .debug_loc for \p LinkedRanges.
  void emitDwarfDebugLocTableFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      LocAttrPatch &Patch) {}

  /// Emit piece of .debug_loclists for \p LinkedRanges.
  void emitDwarfDebugLocListsTableFragment(
      const CompileUnit &Unit,
      const DWARFLocationExpressionsVector &LinkedLocationExpression,
      LocAttrPatch &Patch) {}

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
  DWARFLinker::OutputFileType OutFileType = DWARFLinker::OutputFileType::Object;
  std::function<StringRef(StringRef Input)> Translator;

  uint64_t RangesSectionSize = 0;
  uint64_t RngListsSectionSize = 0;
  uint64_t LocSectionSize = 0;
  uint64_t LocListsSectionSize = 0;
  uint64_t LineSectionSize = 0;
  uint64_t FrameSectionSize = 0;
  uint64_t DebugInfoSectionSize = 0;
  uint64_t MacInfoSectionSize = 0;
  uint64_t MacroSectionSize = 0;

  /// Keep track of emitted CUs and their Unique ID.
  struct EmittedUnit {
    unsigned ID;
    MCSymbol *LabelBegin;
  };
  std::vector<EmittedUnit> EmittedUnitsTy;

  /// Emit the pubnames or pubtypes section contribution for \p
  /// Unit into \p Sec. The data is provided in \p Names.
  void emitPubSectionForUnit(MCSection *Sec, StringRef Name,
                             const CompileUnit &Unit,
                             const std::vector<CompileUnit::AccelInfo> &Names);

  DWARFLinker::MessageHandlerTy WarningHandler = nullptr;
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFEMITTERIMPL_H
