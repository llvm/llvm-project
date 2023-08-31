//===- DWARFLinkerCompileUnit.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H

#include "DWARFLinkerUnit.h"
#include "IndexedValuesMap.h"
#include "llvm/DWARFLinkerParallel/DWARFFile.h"
#include <optional>

namespace llvm {
namespace dwarflinker_parallel {

using OffsetToUnitTy = function_ref<CompileUnit *(uint64_t Offset)>;

/// Stores all information related to a compile unit, be it in its original
/// instance of the object file or its brand new cloned and generated DIE tree.
class CompileUnit : public DwarfUnit {
public:
  /// The stages of new compile unit processing.
  enum class Stage : uint8_t {
    /// Created, linked with input DWARF file.
    CreatedNotLoaded = 0,

    /// Input DWARF is loaded.
    Loaded,

    /// Input DWARF is analysed(DIEs pointing to the real code section are
    /// discovered, type names are assigned if ODR is requested).
    LivenessAnalysisDone,

    /// Output DWARF is generated.
    Cloned,

    /// Offsets inside patch records are updated.
    PatchesUpdated,

    /// Resources(Input DWARF, Output DWARF tree) are released.
    Cleaned,
  };

  CompileUnit(LinkingGlobalData &GlobalData, unsigned ID,
              StringRef ClangModuleName, DWARFFile &File,
              OffsetToUnitTy UnitFromOffset, dwarf::FormParams Format,
              support::endianness Endianess)
      : DwarfUnit(GlobalData, ID, ClangModuleName), File(File),
        getUnitFromOffset(UnitFromOffset), Stage(Stage::CreatedNotLoaded) {
    UnitName = File.FileName;
    setOutputFormat(Format, Endianess);
  }

  CompileUnit(LinkingGlobalData &GlobalData, DWARFUnit &OrigUnit, unsigned ID,
              StringRef ClangModuleName, DWARFFile &File,
              OffsetToUnitTy UnitFromOffset, dwarf::FormParams Format,
              support::endianness Endianess)
      : DwarfUnit(GlobalData, ID, ClangModuleName), File(File),
        OrigUnit(&OrigUnit), getUnitFromOffset(UnitFromOffset),
        Stage(Stage::CreatedNotLoaded) {
    DWARFDie CUDie = OrigUnit.getUnitDIE();
    if (!CUDie)
      return;

    setOutputFormat(Format, Endianess);

    Language = dwarf::toUnsigned(CUDie.find(dwarf::DW_AT_language), 0);
    if (const char *CUName = CUDie.getName(DINameKind::ShortName))
      UnitName = CUName;
    else
      UnitName = File.FileName;
    SysRoot = dwarf::toStringRef(CUDie.find(dwarf::DW_AT_LLVM_sysroot)).str();
  }

  /// Returns stage of overall processing.
  Stage getStage() const { return Stage; }

  /// Set stage of overall processing.
  void setStage(Stage Stage) { this->Stage = Stage; }

  /// Loads unit line table.
  void loadLineTable();

  /// Returns name of the file for the \p FileIdx
  /// from the unit`s line table.
  StringEntry *getFileName(unsigned FileIdx, StringPool &GlobalStrings);

  /// Returns DWARFFile containing this compile unit.
  const DWARFFile &getContaingFile() const { return File; }

  /// Load DIEs of input compilation unit. \returns true if input DIEs
  /// successfully loaded.
  bool loadInputDIEs();

  /// Reset compile units data(results of liveness analysis, clonning)
  /// if current stage greater than Stage::Loaded. We need to reset data
  /// as we are going to repeat stages.
  void maybeResetToLoadedStage();

  /// Collect references to parseable Swift interfaces in imported
  /// DW_TAG_module blocks.
  void analyzeImportedModule(const DWARFDebugInfoEntry *DieEntry);

  /// Navigate DWARF tree and set die properties.
  void analyzeDWARFStructure() {
    analyzeDWARFStructureRec(getUnitDIE().getDebugInfoEntry(), false, false);
  }

  /// Cleanup unneeded resources after compile unit is cloned.
  void cleanupDataAfterClonning();

  /// After cloning stage the output DIEs offsets are deallocated.
  /// This method copies output offsets for referenced DIEs into DIEs patches.
  void updateDieRefPatchesWithClonedOffsets();

  /// Kinds of placement for the output die.
  enum DieOutputPlacement : uint8_t {
    NotSet = 0,

    /// Corresponding DIE goes to the type table only.
    /// NOTE: Not used yet.
    TypeTable = 1,

    /// Corresponding DIE goes to the plain dwarf only.
    PlainDwarf = 2,

    /// Corresponding DIE goes to type table and to plain dwarf.
    /// NOTE: Not used yet.
    Both = 3,

    /// Corresponding DIE needs to examine parent to determine
    /// the point of placement.
    /// NOTE: Not used yet.
    Parent = 4
  };

  /// Information gathered about source DIEs.
  struct DIEInfo {
    DIEInfo() = default;
    DIEInfo(const DIEInfo &Other) { Flags = Other.Flags.load(); }
    DIEInfo &operator=(const DIEInfo &Other) {
      Flags = Other.Flags.load();
      return *this;
    }

    /// Data member keeping various flags.
    std::atomic<uint16_t> Flags = {0};

    /// \returns Placement kind for the corresponding die.
    DieOutputPlacement getPlacement() const {
      return DieOutputPlacement(Flags & 0x7);
    }

    /// Sets Placement kind for the corresponding die.
    void setPlacement(DieOutputPlacement Placement) {
      auto InputData = Flags.load();
      while (!Flags.compare_exchange_weak(InputData,
                                          ((InputData & ~0x7) | Placement))) {
      }
    }

    /// Unsets Placement kind for the corresponding die.
    void unsetPlacement() {
      auto InputData = Flags.load();
      while (!Flags.compare_exchange_weak(InputData, (InputData & ~0x7))) {
      }
    }

    /// Sets Placement kind for the corresponding die.
    bool setPlacementIfUnset(DieOutputPlacement Placement) {
      auto InputData = Flags.load();
      if ((InputData & 0x7) == NotSet)
        if (Flags.compare_exchange_weak(InputData, (InputData | Placement)))
          return true;

      return false;
    }

#define SINGLE_FLAG_METHODS_SET(Name, Value)                                   \
  bool get##Name() const { return Flags & Value; }                             \
  void set##Name() {                                                           \
    auto InputData = Flags.load();                                             \
    while (!Flags.compare_exchange_weak(InputData, InputData | Value)) {       \
    }                                                                          \
  }                                                                            \
  void unset##Name() {                                                         \
    auto InputData = Flags.load();                                             \
    while (!Flags.compare_exchange_weak(InputData, InputData & ~Value)) {      \
    }                                                                          \
  }

    /// DIE is a part of the linked output.
    SINGLE_FLAG_METHODS_SET(Keep, 0x08)

    /// DIE has children which are part of the linked output.
    SINGLE_FLAG_METHODS_SET(KeepChildren, 0x10)

    /// DIE is referenced by other DIE.
    SINGLE_FLAG_METHODS_SET(ReferrencedBy, 0x20)

    /// DIE is in module scope.
    SINGLE_FLAG_METHODS_SET(IsInMouduleScope, 0x40)

    /// DIE is in function scope.
    SINGLE_FLAG_METHODS_SET(IsInFunctionScope, 0x80)

    void unsetFlagsWhichSetDuringLiveAnalysis() {
      auto InputData = Flags.load();
      while (!Flags.compare_exchange_weak(
          InputData, InputData & ~(0x7 | 0x8 | 0x10 | 0x20))) {
      }
    }

    /// Erase all flags.
    void eraseData() { Flags = 0; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    LLVM_DUMP_METHOD void dump();
#endif
  };

  /// \defgroup Group of functions returning DIE info.
  ///
  /// @{

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  DIEInfo &getDIEInfo(unsigned Idx) { return DieInfoArray[Idx]; }

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  const DIEInfo &getDIEInfo(unsigned Idx) const { return DieInfoArray[Idx]; }

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  DIEInfo &getDIEInfo(const DWARFDebugInfoEntry *Entry) {
    return DieInfoArray[getOrigUnit().getDIEIndex(Entry)];
  }

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  const DIEInfo &getDIEInfo(const DWARFDebugInfoEntry *Entry) const {
    return DieInfoArray[getOrigUnit().getDIEIndex(Entry)];
  }

  /// \p Die
  /// \returns PlainDieInfo descriptor.
  DIEInfo &getDIEInfo(const DWARFDie &Die) {
    return DieInfoArray[getOrigUnit().getDIEIndex(Die)];
  }

  /// \p Die
  /// \returns PlainDieInfo descriptor.
  const DIEInfo &getDIEInfo(const DWARFDie &Die) const {
    return DieInfoArray[getOrigUnit().getDIEIndex(Die)];
  }

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  uint64_t getDieOutOffset(uint32_t Idx) {
    return reinterpret_cast<std::atomic<uint64_t> *>(&OutDieOffsetArray[Idx])
        ->load();
  }

  /// \p Idx index of the DIE.
  /// \returns DieInfo descriptor.
  void rememberDieOutOffset(uint32_t Idx, uint64_t Offset) {
    reinterpret_cast<std::atomic<uint64_t> *>(&OutDieOffsetArray[Idx])
        ->store(Offset);
  }

  /// @}

  /// Returns value of DW_AT_low_pc attribute.
  std::optional<uint64_t> getLowPc() const { return LowPc; }

  /// Returns value of DW_AT_high_pc attribute.
  uint64_t getHighPc() const { return HighPc; }

  /// Returns true if there is a label corresponding to the specified \p Addr.
  bool hasLabelAt(uint64_t Addr) const { return Labels.count(Addr); }

  /// Add the low_pc of a label that is relocated by applying
  /// offset \p PCOffset.
  void addLabelLowPc(uint64_t LabelLowPc, int64_t PcOffset);

  /// Resolve the DIE attribute reference that has been extracted in \p
  /// RefValue. The resulting DIE might be in another CompileUnit.
  /// \returns referenced die and corresponding compilation unit.
  ///          compilation unit is null if reference could not be resolved.
  std::optional<std::pair<CompileUnit *, uint32_t>>
  resolveDIEReference(const DWARFFormValue &RefValue);
  /// @}

  /// Add a function range [\p LowPC, \p HighPC) that is relocated by applying
  /// offset \p PCOffset.
  void addFunctionRange(uint64_t LowPC, uint64_t HighPC, int64_t PCOffset);

  /// Returns function ranges of this unit.
  const RangesTy &getFunctionRanges() const { return Ranges; }

  /// Clone and emit this compilation unit.
  Error cloneAndEmit(std::optional<Triple> TargetTriple);

  /// Clone and emit debug locations(.debug_loc/.debug_loclists).
  Error cloneAndEmitDebugLocations();

  /// Clone and emit ranges.
  Error cloneAndEmitRanges();

  /// Clone and emit debug macros(.debug_macinfo/.debug_macro).
  Error cloneAndEmitDebugMacro();

  // Clone input DIE entry.
  DIE *cloneDIE(const DWARFDebugInfoEntry *InputDieEntry, uint64_t OutOffset,
                std::optional<int64_t> FuncAddressAdjustment,
                std::optional<int64_t> VarAddressAdjustment,
                BumpPtrAllocator &Allocator);

  // Clone and emit line table.
  Error cloneAndEmitLineTable(Triple &TargetTriple);

  /// Clone attribute location axpression.
  void cloneDieAttrExpression(const DWARFExpression &InputExpression,
                              SmallVectorImpl<uint8_t> &OutputExpression,
                              SectionDescriptor &Section,
                              std::optional<int64_t> VarAddressAdjustment,
                              OffsetsPtrVector &PatchesOffsets);

  /// Returns index(inside .debug_addr) of an address.
  uint64_t getDebugAddrIndex(uint64_t Addr) {
    return DebugAddrIndexMap.getValueIndex(Addr);
  }

  /// Returns index(inside .debug_str_offsets) of specified string.
  uint64_t getDebugStrIndex(const StringEntry *String) {
    return DebugStringIndexMap.getValueIndex(String);
  }

  /// \defgroup Helper methods to access OrigUnit.
  ///
  /// @{

  /// Returns paired compile unit from input DWARF.
  DWARFUnit &getOrigUnit() const {
    assert(OrigUnit != nullptr);
    return *OrigUnit;
  }

  const DWARFDebugInfoEntry *
  getFirstChildEntry(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getFirstChildEntry(Die);
  }

  const DWARFDebugInfoEntry *
  getSiblingEntry(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getSiblingEntry(Die);
  }

  DWARFDie getParent(const DWARFDebugInfoEntry *Die) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getParent(Die);
  }

  DWARFDie getDIEAtIndex(unsigned Index) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEAtIndex(Index);
  }

  const DWARFDebugInfoEntry *getDebugInfoEntry(unsigned Index) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDebugInfoEntry(Index);
  }

  DWARFDie getUnitDIE(bool ExtractUnitDIEOnly = true) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getUnitDIE(ExtractUnitDIEOnly);
  }

  DWARFDie getDIE(const DWARFDebugInfoEntry *Die) {
    assert(OrigUnit != nullptr);
    return DWARFDie(OrigUnit, Die);
  }

  uint32_t getDIEIndex(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEIndex(Die);
  }

  uint32_t getDIEIndex(const DWARFDie &Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEIndex(Die);
  }

  std::optional<DWARFFormValue> find(uint32_t DieIdx,
                                     ArrayRef<dwarf::Attribute> Attrs) const {
    assert(OrigUnit != nullptr);
    return find(OrigUnit->getDebugInfoEntry(DieIdx), Attrs);
  }

  std::optional<DWARFFormValue> find(const DWARFDebugInfoEntry *Die,
                                     ArrayRef<dwarf::Attribute> Attrs) const {
    if (!Die)
      return std::nullopt;
    auto AbbrevDecl = Die->getAbbreviationDeclarationPtr();
    if (AbbrevDecl) {
      for (auto Attr : Attrs) {
        if (auto Value = AbbrevDecl->getAttributeValue(Die->getOffset(), Attr,
                                                       *OrigUnit))
          return Value;
      }
    }
    return std::nullopt;
  }

  std::optional<uint32_t> getDIEIndexForOffset(uint64_t Offset) {
    return OrigUnit->getDIEIndexForOffset(Offset);
  }

  /// @}

  /// \defgroup Methods used for reporting warnings and errors:
  ///
  /// @{

  void warn(const Twine &Warning, const DWARFDie *DIE = nullptr) {
    GlobalData.warn(Warning, getUnitName(), DIE);
  }

  void warn(Error Warning, const DWARFDie *DIE = nullptr) {
    handleAllErrors(std::move(Warning), [&](ErrorInfoBase &Info) {
      GlobalData.warn(Info.message(), getUnitName(), DIE);
    });
  }

  void warn(const Twine &Warning, const DWARFDebugInfoEntry *DieEntry) {
    if (DieEntry != nullptr) {
      DWARFDie DIE(&getOrigUnit(), DieEntry);
      GlobalData.warn(Warning, getUnitName(), &DIE);
      return;
    }

    GlobalData.warn(Warning, getUnitName());
  }

  void error(const Twine &Err, const DWARFDie *DIE = nullptr) {
    GlobalData.warn(Err, getUnitName(), DIE);
  }

  void error(Error Err, const DWARFDie *DIE = nullptr) {
    handleAllErrors(std::move(Err), [&](ErrorInfoBase &Info) {
      GlobalData.error(Info.message(), getUnitName(), DIE);
    });
  }

  /// @}

private:
  /// Navigate DWARF tree recursively and set die properties.
  void analyzeDWARFStructureRec(const DWARFDebugInfoEntry *DieEntry,
                                bool IsInModule, bool IsInFunction);

  struct LinkedLocationExpressionsWithOffsetPatches {
    DWARFLocationExpression Expression;
    OffsetsPtrVector Patches;
  };
  using LinkedLocationExpressionsVector =
      SmallVector<LinkedLocationExpressionsWithOffsetPatches>;

  /// Emit debug locations.
  void emitLocations(DebugSectionKind LocationSectionKind);

  /// Emit location list header.
  uint64_t emitLocListHeader(SectionDescriptor &OutLocationSection);

  /// Emit location list fragment.
  uint64_t emitLocListFragment(
      const LinkedLocationExpressionsVector &LinkedLocationExpression,
      SectionDescriptor &OutLocationSection);

  /// Emit the .debug_addr section fragment for current unit.
  Error emitDebugAddrSection();

  /// Emit the .debug_str_offsets section for current unit.
  Error emitDebugStringOffsetSection();

  /// Emit .debug_aranges.
  void emitAranges(AddressRanges &LinkedFunctionRanges);

  /// Clone and emit .debug_ranges/.debug_rnglists.
  void cloneAndEmitRangeList(DebugSectionKind RngSectionKind,
                             AddressRanges &LinkedFunctionRanges);

  /// Emit range list header.
  uint64_t emitRangeListHeader(SectionDescriptor &OutRangeSection);

  /// Emit range list fragment.
  void emitRangeListFragment(const AddressRanges &LinkedRanges,
                             SectionDescriptor &OutRangeSection);

  /// Insert the new line info sequence \p Seq into the current
  /// set of already linked line info \p Rows.
  void insertLineSequence(std::vector<DWARFDebugLine::Row> &Seq,
                          std::vector<DWARFDebugLine::Row> &Rows);

  /// Emits body for both macro sections.
  void emitMacroTableImpl(const DWARFDebugMacro *MacroTable,
                          uint64_t OffsetToMacroTable, bool hasDWARFv5Header);

  /// DWARFFile containing this compile unit.
  DWARFFile &File;

  /// Pointer to the paired compile unit from the input DWARF.
  DWARFUnit *OrigUnit = nullptr;

  /// Line table for this unit.
  const DWARFDebugLine::LineTable *LineTablePtr = nullptr;

  /// Cached resolved paths from the line table.
  /// The key is <UniqueUnitID, FileIdx>.
  using ResolvedPathsMap = DenseMap<unsigned, StringEntry *>;
  ResolvedPathsMap ResolvedFullPaths;
  StringMap<StringEntry *> ResolvedParentPaths;

  /// This field instructs compile unit to store DIE name with stripped
  /// template parameters into the accelerator table.
  bool CanStripTemplateName = false;

  /// Maps an address into the index inside .debug_addr section.
  IndexedValuesMap<uint64_t> DebugAddrIndexMap;

  /// Maps a string into the index inside .debug_str_offsets section.
  IndexedValuesMap<const StringEntry *> DebugStringIndexMap;

  /// \defgroup Data Members accessed asinchroniously.
  ///
  /// @{
  OffsetToUnitTy getUnitFromOffset;

  std::optional<uint64_t> LowPc;
  uint64_t HighPc = 0;

  /// The ranges in that map are the PC ranges for functions in this unit,
  /// associated with the PC offset to apply to the addresses to get
  /// the linked address.
  RangesTy Ranges;
  std::mutex RangesMutex;

  /// The DW_AT_low_pc of each DW_TAG_label.
  SmallDenseMap<uint64_t, uint64_t, 1> Labels;
  std::mutex LabelsMutex;

  /// This field keeps current stage of overall compile unit processing.
  std::atomic<Stage> Stage;

  /// DIE info indexed by DIE index.
  SmallVector<DIEInfo> DieInfoArray;
  SmallVector<uint64_t> OutDieOffsetArray;
  /// @}
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H
