//=== DWARFLinkerCompileUnit.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerCompileUnit.h"
#include "DIEAttributeCloner.h"
#include "DIEGenerator.h"
#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugMacro.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::dwarflinker_parallel;

void CompileUnit::loadLineTable() {
  LineTablePtr = File.Dwarf->getLineTableForUnit(&getOrigUnit());
}

void CompileUnit::maybeResetToLoadedStage() {
  // Nothing to reset if stage is less than "Loaded".
  if (getStage() < Stage::Loaded)
    return;

  // Note: We need to do erasing for "Loaded" stage because
  // if live analysys failed then we will have "Loaded" stage
  // with marking from "LivenessAnalysisDone" stage partially
  // done. That marking should be cleared.

  for (DIEInfo &Info : DieInfoArray)
    Info.unsetFlagsWhichSetDuringLiveAnalysis();

  LowPc = std::nullopt;
  HighPc = 0;
  Labels.clear();
  Ranges.clear();

  if (getStage() < Stage::Cloned) {
    setStage(Stage::Loaded);
    return;
  }

  AcceleratorRecords.erase();
  AbbreviationsSet.clear();
  Abbreviations.clear();
  OutUnitDIE = nullptr;
  DebugAddrIndexMap.clear();

  for (uint64_t &Offset : OutDieOffsetArray)
    Offset = 0;
  eraseSections();

  setStage(Stage::CreatedNotLoaded);
}

bool CompileUnit::loadInputDIEs() {
  DWARFDie InputUnitDIE = getUnitDIE(false);
  if (!InputUnitDIE)
    return false;

  // load input dies, resize Info structures array.
  DieInfoArray.resize(getOrigUnit().getNumDIEs());
  OutDieOffsetArray.resize(getOrigUnit().getNumDIEs(), 0);
  return true;
}

void CompileUnit::analyzeDWARFStructureRec(const DWARFDebugInfoEntry *DieEntry,
                                           bool IsInModule, bool IsInFunction) {
  for (const DWARFDebugInfoEntry *CurChild = getFirstChildEntry(DieEntry);
       CurChild && CurChild->getAbbreviationDeclarationPtr();
       CurChild = getSiblingEntry(CurChild)) {
    CompileUnit::DIEInfo &ChildInfo = getDIEInfo(CurChild);

    if (IsInModule)
      ChildInfo.setIsInMouduleScope();
    if (IsInFunction)
      ChildInfo.setIsInFunctionScope();

    switch (CurChild->getTag()) {
    case dwarf::DW_TAG_module:
      ChildInfo.setIsInMouduleScope();
      if (DieEntry->getTag() == dwarf::DW_TAG_compile_unit &&
          dwarf::toString(find(CurChild, dwarf::DW_AT_name), "") !=
              getClangModuleName())
        analyzeImportedModule(CurChild);
      break;
    case dwarf::DW_TAG_subprogram:
      ChildInfo.setIsInFunctionScope();
      break;
    default:
      break;
    }

    if (IsInModule)
      ChildInfo.setIsInMouduleScope();
    if (IsInFunction)
      ChildInfo.setIsInFunctionScope();

    if (CurChild->hasChildren())
      analyzeDWARFStructureRec(CurChild, ChildInfo.getIsInMouduleScope(),
                               ChildInfo.getIsInFunctionScope());
  }
}

StringEntry *CompileUnit::getFileName(unsigned FileIdx,
                                      StringPool &GlobalStrings) {
  if (LineTablePtr) {
    if (LineTablePtr->hasFileAtIndex(FileIdx)) {
      // Cache the resolved paths based on the index in the line table,
      // because calling realpath is expensive.
      ResolvedPathsMap::const_iterator It = ResolvedFullPaths.find(FileIdx);
      if (It == ResolvedFullPaths.end()) {
        std::string OrigFileName;
        bool FoundFileName = LineTablePtr->getFileNameByIndex(
            FileIdx, getOrigUnit().getCompilationDir(),
            DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
            OrigFileName);
        (void)FoundFileName;
        assert(FoundFileName && "Must get file name from line table");

        // Second level of caching, this time based on the file's parent
        // path.
        StringRef FileName = sys::path::filename(OrigFileName);
        StringRef ParentPath = sys::path::parent_path(OrigFileName);

        // If the ParentPath has not yet been resolved, resolve and cache it for
        // future look-ups.
        StringMap<StringEntry *>::iterator ParentIt =
            ResolvedParentPaths.find(ParentPath);
        if (ParentIt == ResolvedParentPaths.end()) {
          SmallString<256> RealPath;
          sys::fs::real_path(ParentPath, RealPath);
          ParentIt =
              ResolvedParentPaths
                  .insert({ParentPath, GlobalStrings.insert(RealPath).first})
                  .first;
        }

        // Join the file name again with the resolved path.
        SmallString<256> ResolvedPath(ParentIt->second->first());
        sys::path::append(ResolvedPath, FileName);

        It = ResolvedFullPaths
                 .insert(std::make_pair(
                     FileIdx, GlobalStrings.insert(ResolvedPath).first))
                 .first;
      }

      return It->second;
    }
  }

  return nullptr;
}

void CompileUnit::cleanupDataAfterClonning() {
  AbbreviationsSet.clear();
  ResolvedFullPaths.shrink_and_clear();
  ResolvedParentPaths.clear();
  DieInfoArray = SmallVector<DIEInfo>();
  OutDieOffsetArray = SmallVector<uint64_t>();
  getOrigUnit().clear();
}

/// Collect references to parseable Swift interfaces in imported
/// DW_TAG_module blocks.
void CompileUnit::analyzeImportedModule(const DWARFDebugInfoEntry *DieEntry) {
  if (getLanguage() != dwarf::DW_LANG_Swift)
    return;

  if (!GlobalData.getOptions().ParseableSwiftInterfaces)
    return;

  StringRef Path =
      dwarf::toStringRef(find(DieEntry, dwarf::DW_AT_LLVM_include_path));
  if (!Path.endswith(".swiftinterface"))
    return;
  // Don't track interfaces that are part of the SDK.
  StringRef SysRoot =
      dwarf::toStringRef(find(DieEntry, dwarf::DW_AT_LLVM_sysroot));
  if (SysRoot.empty())
    SysRoot = getSysRoot();
  if (!SysRoot.empty() && Path.startswith(SysRoot))
    return;
  if (std::optional<DWARFFormValue> Val = find(DieEntry, dwarf::DW_AT_name)) {
    Expected<const char *> Name = Val->getAsCString();
    if (!Name) {
      warn(Name.takeError());
      return;
    }

    auto &Entry = (*GlobalData.getOptions().ParseableSwiftInterfaces)[*Name];
    // The prepend path is applied later when copying.
    SmallString<128> ResolvedPath;
    if (sys::path::is_relative(Path))
      sys::path::append(
          ResolvedPath,
          dwarf::toString(getUnitDIE().find(dwarf::DW_AT_comp_dir), ""));
    sys::path::append(ResolvedPath, Path);
    if (!Entry.empty() && Entry != ResolvedPath) {
      DWARFDie Die = getDIE(DieEntry);
      warn(Twine("conflicting parseable interfaces for Swift Module ") + *Name +
               ": " + Entry + " and " + Path + ".",
           &Die);
    }
    Entry = std::string(ResolvedPath.str());
  }
}

void CompileUnit::updateDieRefPatchesWithClonedOffsets() {
  if (std::optional<SectionDescriptor *> DebugInfoSection =
          tryGetSectionDescriptor(DebugSectionKind::DebugInfo)) {

    (*DebugInfoSection)
        ->ListDebugDieRefPatch.forEach([](DebugDieRefPatch &Patch) {
          /// Replace stored DIE indexes with DIE output offsets.
          Patch.RefDieIdxOrClonedOffset =
              Patch.RefCU.getPointer()->getDieOutOffset(
                  Patch.RefDieIdxOrClonedOffset);
        });

    (*DebugInfoSection)
        ->ListDebugULEB128DieRefPatch.forEach(
            [](DebugULEB128DieRefPatch &Patch) {
              /// Replace stored DIE indexes with DIE output offsets.
              Patch.RefDieIdxOrClonedOffset =
                  Patch.RefCU.getPointer()->getDieOutOffset(
                      Patch.RefDieIdxOrClonedOffset);
            });
  }

  if (std::optional<SectionDescriptor *> DebugLocSection =
          tryGetSectionDescriptor(DebugSectionKind::DebugLoc)) {
    (*DebugLocSection)
        ->ListDebugULEB128DieRefPatch.forEach(
            [](DebugULEB128DieRefPatch &Patch) {
              /// Replace stored DIE indexes with DIE output offsets.
              Patch.RefDieIdxOrClonedOffset =
                  Patch.RefCU.getPointer()->getDieOutOffset(
                      Patch.RefDieIdxOrClonedOffset);
            });
  }

  if (std::optional<SectionDescriptor *> DebugLocListsSection =
          tryGetSectionDescriptor(DebugSectionKind::DebugLocLists)) {
    (*DebugLocListsSection)
        ->ListDebugULEB128DieRefPatch.forEach(
            [](DebugULEB128DieRefPatch &Patch) {
              /// Replace stored DIE indexes with DIE output offsets.
              Patch.RefDieIdxOrClonedOffset =
                  Patch.RefCU.getPointer()->getDieOutOffset(
                      Patch.RefDieIdxOrClonedOffset);
            });
  }
}

std::optional<std::pair<CompileUnit *, uint32_t>>
CompileUnit::resolveDIEReference(
    const DWARFFormValue &RefValue,
    ResolveInterCUReferencesMode CanResolveInterCUReferences) {
  if (std::optional<DWARFFormValue::UnitOffset> Ref =
          *RefValue.getAsRelativeReference()) {
    if (Ref->Unit != nullptr) {
      // Referenced DIE is in current compile unit.

      if (std::optional<uint32_t> RefDieIdx =
              getDIEIndexForOffset(Ref->Unit->getOffset() + Ref->Offset))
        return std::make_pair(this, *RefDieIdx);
    }

    if (CompileUnit *RefCU = getUnitFromOffset(Ref->Offset)) {
      if (RefCU->getUniqueID() == getUniqueID()) {
        // Referenced DIE is in current compile unit.
        if (std::optional<uint32_t> RefDieIdx =
                getDIEIndexForOffset(Ref->Offset))
          return std::make_pair(this, *RefDieIdx);
      } else if (CanResolveInterCUReferences) {
        // Referenced DIE is in other compile unit.

        // Check whether DIEs are loaded for that compile unit.
        enum Stage ReferredCUStage = RefCU->getStage();
        if (ReferredCUStage < Stage::Loaded || ReferredCUStage > Stage::Cloned)
          return std::make_pair(RefCU, 0);

        if (std::optional<uint32_t> RefDieIdx =
                RefCU->getDIEIndexForOffset(Ref->Offset))
          return std::make_pair(RefCU, *RefDieIdx);
      } else
        return std::make_pair(RefCU, 0);
    }
  }

  return std::nullopt;
}

void CompileUnit::addFunctionRange(uint64_t FuncLowPc, uint64_t FuncHighPc,
                                   int64_t PcOffset) {
  std::lock_guard<std::mutex> Guard(RangesMutex);

  Ranges.insert({FuncLowPc, FuncHighPc}, PcOffset);
  if (LowPc)
    LowPc = std::min(*LowPc, FuncLowPc + PcOffset);
  else
    LowPc = FuncLowPc + PcOffset;
  this->HighPc = std::max(HighPc, FuncHighPc + PcOffset);
}

void CompileUnit::addLabelLowPc(uint64_t LabelLowPc, int64_t PcOffset) {
  std::lock_guard<std::mutex> Guard(LabelsMutex);
  Labels.insert({LabelLowPc, PcOffset});
}

Error CompileUnit::cloneAndEmitDebugLocations() {
  if (getGlobalData().getOptions().UpdateIndexTablesOnly)
    return Error::success();

  if (getOrigUnit().getVersion() < 5) {
    emitLocations(DebugSectionKind::DebugLoc);
    return Error::success();
  }

  emitLocations(DebugSectionKind::DebugLocLists);
  return Error::success();
}

void CompileUnit::emitLocations(DebugSectionKind LocationSectionKind) {
  SectionDescriptor &DebugInfoSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo);

  if (!DebugInfoSection.ListDebugLocPatch.empty()) {
    SectionDescriptor &OutLocationSection =
        getOrCreateSectionDescriptor(LocationSectionKind);
    DWARFUnit &OrigUnit = getOrigUnit();

    uint64_t OffsetAfterUnitLength = emitLocListHeader(OutLocationSection);

    DebugInfoSection.ListDebugLocPatch.forEach([&](DebugLocPatch &Patch) {
      // Get location expressions vector corresponding to the current
      // attribute from the source DWARF.
      uint64_t InputDebugLocSectionOffset = DebugInfoSection.getIntVal(
          Patch.PatchOffset,
          DebugInfoSection.getFormParams().getDwarfOffsetByteSize());
      Expected<DWARFLocationExpressionsVector> OriginalLocations =
          OrigUnit.findLoclistFromOffset(InputDebugLocSectionOffset);

      if (!OriginalLocations) {
        warn(OriginalLocations.takeError());
        return;
      }

      LinkedLocationExpressionsVector LinkedLocationExpressions;
      for (DWARFLocationExpression &CurExpression : *OriginalLocations) {
        LinkedLocationExpressionsWithOffsetPatches LinkedExpression;

        if (CurExpression.Range) {
          // Relocate address range.
          LinkedExpression.Expression.Range = {
              CurExpression.Range->LowPC + Patch.AddrAdjustmentValue,
              CurExpression.Range->HighPC + Patch.AddrAdjustmentValue};
        }

        DataExtractor Data(CurExpression.Expr, OrigUnit.isLittleEndian(),
                           OrigUnit.getAddressByteSize());

        DWARFExpression InputExpression(Data, OrigUnit.getAddressByteSize(),
                                        OrigUnit.getFormParams().Format);
        cloneDieAttrExpression(InputExpression,
                               LinkedExpression.Expression.Expr,
                               OutLocationSection, Patch.AddrAdjustmentValue,
                               LinkedExpression.Patches);

        LinkedLocationExpressions.push_back({LinkedExpression});
      }

      // Emit locations list table fragment corresponding to the CurLocAttr.
      DebugInfoSection.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset,
                             OutLocationSection.OS.tell());
      emitLocListFragment(LinkedLocationExpressions, OutLocationSection);
    });

    if (OffsetAfterUnitLength > 0) {
      assert(OffsetAfterUnitLength -
                 OutLocationSection.getFormParams().getDwarfOffsetByteSize() <
             OffsetAfterUnitLength);
      OutLocationSection.apply(
          OffsetAfterUnitLength -
              OutLocationSection.getFormParams().getDwarfOffsetByteSize(),
          dwarf::DW_FORM_sec_offset,
          OutLocationSection.OS.tell() - OffsetAfterUnitLength);
    }
  }
}

/// Emit debug locations(.debug_loc, .debug_loclists) header.
uint64_t CompileUnit::emitLocListHeader(SectionDescriptor &OutLocationSection) {
  if (getOrigUnit().getVersion() < 5)
    return 0;

  // unit_length.
  OutLocationSection.emitUnitLength(0xBADDEF);
  uint64_t OffsetAfterUnitLength = OutLocationSection.OS.tell();

  // Version.
  OutLocationSection.emitIntVal(5, 2);

  // Address size.
  OutLocationSection.emitIntVal(OutLocationSection.getFormParams().AddrSize, 1);

  // Seg_size
  OutLocationSection.emitIntVal(0, 1);

  // Offset entry count
  OutLocationSection.emitIntVal(0, 4);

  return OffsetAfterUnitLength;
}

/// Emit debug locations(.debug_loc, .debug_loclists) fragment.
uint64_t CompileUnit::emitLocListFragment(
    const LinkedLocationExpressionsVector &LinkedLocationExpression,
    SectionDescriptor &OutLocationSection) {
  uint64_t OffsetBeforeLocationExpression = 0;

  if (getOrigUnit().getVersion() < 5) {
    uint64_t BaseAddress = 0;
    if (std::optional<uint64_t> LowPC = getLowPc())
      BaseAddress = *LowPC;

    for (const LinkedLocationExpressionsWithOffsetPatches &LocExpression :
         LinkedLocationExpression) {
      if (LocExpression.Expression.Range) {
        OutLocationSection.emitIntVal(
            LocExpression.Expression.Range->LowPC - BaseAddress,
            OutLocationSection.getFormParams().AddrSize);
        OutLocationSection.emitIntVal(
            LocExpression.Expression.Range->HighPC - BaseAddress,
            OutLocationSection.getFormParams().AddrSize);
      }

      OutLocationSection.emitIntVal(LocExpression.Expression.Expr.size(), 2);
      OffsetBeforeLocationExpression = OutLocationSection.OS.tell();
      for (uint64_t *OffsetPtr : LocExpression.Patches)
        *OffsetPtr += OffsetBeforeLocationExpression;

      OutLocationSection.OS
          << StringRef((const char *)LocExpression.Expression.Expr.data(),
                       LocExpression.Expression.Expr.size());
    }

    // Emit the terminator entry.
    OutLocationSection.emitIntVal(0,
                                  OutLocationSection.getFormParams().AddrSize);
    OutLocationSection.emitIntVal(0,
                                  OutLocationSection.getFormParams().AddrSize);
    return OffsetBeforeLocationExpression;
  }

  std::optional<uint64_t> BaseAddress;
  for (const LinkedLocationExpressionsWithOffsetPatches &LocExpression :
       LinkedLocationExpression) {
    if (LocExpression.Expression.Range) {
      // Check whether base address is set. If it is not set yet
      // then set current base address and emit base address selection entry.
      if (!BaseAddress) {
        BaseAddress = LocExpression.Expression.Range->LowPC;

        // Emit base address.
        OutLocationSection.emitIntVal(dwarf::DW_LLE_base_addressx, 1);
        encodeULEB128(DebugAddrIndexMap.getValueIndex(*BaseAddress),
                      OutLocationSection.OS);
      }

      // Emit type of entry.
      OutLocationSection.emitIntVal(dwarf::DW_LLE_offset_pair, 1);

      // Emit start offset relative to base address.
      encodeULEB128(LocExpression.Expression.Range->LowPC - *BaseAddress,
                    OutLocationSection.OS);

      // Emit end offset relative to base address.
      encodeULEB128(LocExpression.Expression.Range->HighPC - *BaseAddress,
                    OutLocationSection.OS);
    } else
      // Emit type of entry.
      OutLocationSection.emitIntVal(dwarf::DW_LLE_default_location, 1);

    encodeULEB128(LocExpression.Expression.Expr.size(), OutLocationSection.OS);
    OffsetBeforeLocationExpression = OutLocationSection.OS.tell();
    for (uint64_t *OffsetPtr : LocExpression.Patches)
      *OffsetPtr += OffsetBeforeLocationExpression;

    OutLocationSection.OS << StringRef(
        (const char *)LocExpression.Expression.Expr.data(),
        LocExpression.Expression.Expr.size());
  }

  // Emit the terminator entry.
  OutLocationSection.emitIntVal(dwarf::DW_LLE_end_of_list, 1);
  return OffsetBeforeLocationExpression;
}

Error CompileUnit::emitDebugAddrSection() {
  if (GlobalData.getOptions().UpdateIndexTablesOnly)
    return Error::success();

  if (getVersion() < 5)
    return Error::success();

  if (DebugAddrIndexMap.empty())
    return Error::success();

  SectionDescriptor &OutAddrSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugAddr);

  // Emit section header.

  //   Emit length.
  OutAddrSection.emitUnitLength(0xBADDEF);
  uint64_t OffsetAfterSectionLength = OutAddrSection.OS.tell();

  //   Emit version.
  OutAddrSection.emitIntVal(5, 2);

  //   Emit address size.
  OutAddrSection.emitIntVal(getFormParams().AddrSize, 1);

  //   Emit segment size.
  OutAddrSection.emitIntVal(0, 1);

  // Emit addresses.
  for (uint64_t AddrValue : DebugAddrIndexMap.getValues())
    OutAddrSection.emitIntVal(AddrValue, getFormParams().AddrSize);

  // Patch section length.
  OutAddrSection.apply(
      OffsetAfterSectionLength -
          OutAddrSection.getFormParams().getDwarfOffsetByteSize(),
      dwarf::DW_FORM_sec_offset,
      OutAddrSection.OS.tell() - OffsetAfterSectionLength);

  return Error::success();
}

Error CompileUnit::emitDebugStringOffsetSection() {
  if (getVersion() < 5)
    return Error::success();

  if (DebugStringIndexMap.empty())
    return Error::success();

  SectionDescriptor &OutDebugStrOffsetsSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugStrOffsets);

  // Emit section header.

  //   Emit length.
  OutDebugStrOffsetsSection.emitUnitLength(0xBADDEF);
  uint64_t OffsetAfterSectionLength = OutDebugStrOffsetsSection.OS.tell();

  //   Emit version.
  OutDebugStrOffsetsSection.emitIntVal(5, 2);

  //   Emit padding.
  OutDebugStrOffsetsSection.emitIntVal(0, 2);

  //   Emit index to offset map.
  for (const StringEntry *String : DebugStringIndexMap.getValues()) {
    // Note patch for string offset value.
    OutDebugStrOffsetsSection.notePatch(
        DebugStrPatch{{OutDebugStrOffsetsSection.OS.tell()}, String});

    // Emit placeholder for offset value.
    OutDebugStrOffsetsSection.emitOffset(0xBADDEF);
  }

  // Patch section length.
  OutDebugStrOffsetsSection.apply(
      OffsetAfterSectionLength -
          OutDebugStrOffsetsSection.getFormParams().getDwarfOffsetByteSize(),
      dwarf::DW_FORM_sec_offset,
      OutDebugStrOffsetsSection.OS.tell() - OffsetAfterSectionLength);

  return Error::success();
}

Error CompileUnit::cloneAndEmitRanges() {
  if (getGlobalData().getOptions().UpdateIndexTablesOnly)
    return Error::success();

  // Build set of linked address ranges for unit function ranges.
  AddressRanges LinkedFunctionRanges;
  for (const AddressRangeValuePair &Range : getFunctionRanges())
    LinkedFunctionRanges.insert(
        {Range.Range.start() + Range.Value, Range.Range.end() + Range.Value});

  emitAranges(LinkedFunctionRanges);

  if (getOrigUnit().getVersion() < 5) {
    cloneAndEmitRangeList(DebugSectionKind::DebugRange, LinkedFunctionRanges);
    return Error::success();
  }

  cloneAndEmitRangeList(DebugSectionKind::DebugRngLists, LinkedFunctionRanges);
  return Error::success();
}

void CompileUnit::cloneAndEmitRangeList(DebugSectionKind RngSectionKind,
                                        AddressRanges &LinkedFunctionRanges) {
  SectionDescriptor &DebugInfoSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo);
  SectionDescriptor &OutRangeSection =
      getOrCreateSectionDescriptor(RngSectionKind);

  if (!DebugInfoSection.ListDebugRangePatch.empty()) {
    std::optional<AddressRangeValuePair> CachedRange;
    uint64_t OffsetAfterUnitLength = emitRangeListHeader(OutRangeSection);

    DebugRangePatch *CompileUnitRangePtr = nullptr;
    DebugInfoSection.ListDebugRangePatch.forEach([&](DebugRangePatch &Patch) {
      if (Patch.IsCompileUnitRanges) {
        CompileUnitRangePtr = &Patch;
      } else {
        // Get ranges from the source DWARF corresponding to the current
        // attribute.
        AddressRanges LinkedRanges;
        uint64_t InputDebugRangesSectionOffset = DebugInfoSection.getIntVal(
            Patch.PatchOffset,
            DebugInfoSection.getFormParams().getDwarfOffsetByteSize());
        if (Expected<DWARFAddressRangesVector> InputRanges =
                getOrigUnit().findRnglistFromOffset(
                    InputDebugRangesSectionOffset)) {
          // Apply relocation adjustment.
          for (const auto &Range : *InputRanges) {
            if (!CachedRange || !CachedRange->Range.contains(Range.LowPC))
              CachedRange =
                  getFunctionRanges().getRangeThatContains(Range.LowPC);

            // All range entries should lie in the function range.
            if (!CachedRange) {
              warn("inconsistent range data.");
              continue;
            }

            // Store range for emiting.
            LinkedRanges.insert({Range.LowPC + CachedRange->Value,
                                 Range.HighPC + CachedRange->Value});
          }
        } else {
          llvm::consumeError(InputRanges.takeError());
          warn("invalid range list ignored.");
        }

        // Emit linked ranges.
        DebugInfoSection.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset,
                               OutRangeSection.OS.tell());
        emitRangeListFragment(LinkedRanges, OutRangeSection);
      }
    });

    if (CompileUnitRangePtr != nullptr) {
      // Emit compile unit ranges last to be binary compatible with classic
      // dsymutil.
      DebugInfoSection.apply(CompileUnitRangePtr->PatchOffset,
                             dwarf::DW_FORM_sec_offset,
                             OutRangeSection.OS.tell());
      emitRangeListFragment(LinkedFunctionRanges, OutRangeSection);
    }

    if (OffsetAfterUnitLength > 0) {
      assert(OffsetAfterUnitLength -
                 OutRangeSection.getFormParams().getDwarfOffsetByteSize() <
             OffsetAfterUnitLength);
      OutRangeSection.apply(
          OffsetAfterUnitLength -
              OutRangeSection.getFormParams().getDwarfOffsetByteSize(),
          dwarf::DW_FORM_sec_offset,
          OutRangeSection.OS.tell() - OffsetAfterUnitLength);
    }
  }
}

uint64_t CompileUnit::emitRangeListHeader(SectionDescriptor &OutRangeSection) {
  if (OutRangeSection.getFormParams().Version < 5)
    return 0;

  // unit_length.
  OutRangeSection.emitUnitLength(0xBADDEF);
  uint64_t OffsetAfterUnitLength = OutRangeSection.OS.tell();

  // Version.
  OutRangeSection.emitIntVal(5, 2);

  // Address size.
  OutRangeSection.emitIntVal(OutRangeSection.getFormParams().AddrSize, 1);

  // Seg_size
  OutRangeSection.emitIntVal(0, 1);

  // Offset entry count
  OutRangeSection.emitIntVal(0, 4);

  return OffsetAfterUnitLength;
}

void CompileUnit::emitRangeListFragment(const AddressRanges &LinkedRanges,
                                        SectionDescriptor &OutRangeSection) {
  if (OutRangeSection.getFormParams().Version < 5) {
    // Emit ranges.
    uint64_t BaseAddress = 0;
    if (std::optional<uint64_t> LowPC = getLowPc())
      BaseAddress = *LowPC;

    for (const AddressRange &Range : LinkedRanges) {
      OutRangeSection.emitIntVal(Range.start() - BaseAddress,
                                 OutRangeSection.getFormParams().AddrSize);
      OutRangeSection.emitIntVal(Range.end() - BaseAddress,
                                 OutRangeSection.getFormParams().AddrSize);
    }

    // Add the terminator entry.
    OutRangeSection.emitIntVal(0, OutRangeSection.getFormParams().AddrSize);
    OutRangeSection.emitIntVal(0, OutRangeSection.getFormParams().AddrSize);
    return;
  }

  std::optional<uint64_t> BaseAddress;
  for (const AddressRange &Range : LinkedRanges) {
    if (!BaseAddress) {
      BaseAddress = Range.start();

      // Emit base address.
      OutRangeSection.emitIntVal(dwarf::DW_RLE_base_addressx, 1);
      encodeULEB128(getDebugAddrIndex(*BaseAddress), OutRangeSection.OS);
    }

    // Emit type of entry.
    OutRangeSection.emitIntVal(dwarf::DW_RLE_offset_pair, 1);

    // Emit start offset relative to base address.
    encodeULEB128(Range.start() - *BaseAddress, OutRangeSection.OS);

    // Emit end offset relative to base address.
    encodeULEB128(Range.end() - *BaseAddress, OutRangeSection.OS);
  }

  // Emit the terminator entry.
  OutRangeSection.emitIntVal(dwarf::DW_RLE_end_of_list, 1);
}

void CompileUnit::emitAranges(AddressRanges &LinkedFunctionRanges) {
  if (LinkedFunctionRanges.empty())
    return;

  SectionDescriptor &DebugInfoSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo);
  SectionDescriptor &OutArangesSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugARanges);

  // Emit Header.
  unsigned HeaderSize =
      sizeof(int32_t) + // Size of contents (w/o this field
      sizeof(int16_t) + // DWARF ARange version number
      sizeof(int32_t) + // Offset of CU in the .debug_info section
      sizeof(int8_t) +  // Pointer Size (in bytes)
      sizeof(int8_t);   // Segment Size (in bytes)

  unsigned TupleSize = OutArangesSection.getFormParams().AddrSize * 2;
  unsigned Padding = offsetToAlignment(HeaderSize, Align(TupleSize));

  OutArangesSection.emitOffset(0xBADDEF); // Aranges length
  uint64_t OffsetAfterArangesLengthField = OutArangesSection.OS.tell();

  OutArangesSection.emitIntVal(dwarf::DW_ARANGES_VERSION, 2); // Version number
  OutArangesSection.notePatch(
      DebugOffsetPatch{OutArangesSection.OS.tell(), &DebugInfoSection});
  OutArangesSection.emitOffset(0xBADDEF); // Corresponding unit's offset
  OutArangesSection.emitIntVal(OutArangesSection.getFormParams().AddrSize,
                               1);    // Address size
  OutArangesSection.emitIntVal(0, 1); // Segment size

  for (size_t Idx = 0; Idx < Padding; Idx++)
    OutArangesSection.emitIntVal(0, 1); // Padding

  // Emit linked ranges.
  for (const AddressRange &Range : LinkedFunctionRanges) {
    OutArangesSection.emitIntVal(Range.start(),
                                 OutArangesSection.getFormParams().AddrSize);
    OutArangesSection.emitIntVal(Range.end() - Range.start(),
                                 OutArangesSection.getFormParams().AddrSize);
  }

  // Emit terminator.
  OutArangesSection.emitIntVal(0, OutArangesSection.getFormParams().AddrSize);
  OutArangesSection.emitIntVal(0, OutArangesSection.getFormParams().AddrSize);

  uint64_t OffsetAfterArangesEnd = OutArangesSection.OS.tell();

  // Update Aranges lentgh.
  OutArangesSection.apply(
      OffsetAfterArangesLengthField -
          OutArangesSection.getFormParams().getDwarfOffsetByteSize(),
      dwarf::DW_FORM_sec_offset,
      OffsetAfterArangesEnd - OffsetAfterArangesLengthField);
}

Error CompileUnit::cloneAndEmitDebugMacro() {
  if (getOutUnitDIE() == nullptr)
    return Error::success();

  DWARFUnit &OrigUnit = getOrigUnit();
  DWARFDie OrigUnitDie = OrigUnit.getUnitDIE();

  // Check for .debug_macro table.
  if (std::optional<uint64_t> MacroAttr =
          dwarf::toSectionOffset(OrigUnitDie.find(dwarf::DW_AT_macros))) {
    if (const DWARFDebugMacro *Table =
            getContaingFile().Dwarf->getDebugMacro()) {
      emitMacroTableImpl(Table, *MacroAttr, true);
    }
  }

  // Check for .debug_macinfo table.
  if (std::optional<uint64_t> MacroAttr =
          dwarf::toSectionOffset(OrigUnitDie.find(dwarf::DW_AT_macro_info))) {
    if (const DWARFDebugMacro *Table =
            getContaingFile().Dwarf->getDebugMacinfo()) {
      emitMacroTableImpl(Table, *MacroAttr, false);
    }
  }

  return Error::success();
}

void CompileUnit::emitMacroTableImpl(const DWARFDebugMacro *MacroTable,
                                     uint64_t OffsetToMacroTable,
                                     bool hasDWARFv5Header) {
  SectionDescriptor &OutSection =
      hasDWARFv5Header
          ? getOrCreateSectionDescriptor(DebugSectionKind::DebugMacro)
          : getOrCreateSectionDescriptor(DebugSectionKind::DebugMacinfo);

  bool DefAttributeIsReported = false;
  bool UndefAttributeIsReported = false;
  bool ImportAttributeIsReported = false;

  for (const DWARFDebugMacro::MacroList &List : MacroTable->MacroLists) {
    if (OffsetToMacroTable == List.Offset) {
      // Write DWARFv5 header.
      if (hasDWARFv5Header) {
        // Write header version.
        OutSection.emitIntVal(List.Header.Version, sizeof(List.Header.Version));

        uint8_t Flags = List.Header.Flags;

        // Check for OPCODE_OPERANDS_TABLE.
        if (Flags &
            DWARFDebugMacro::HeaderFlagMask::MACRO_OPCODE_OPERANDS_TABLE) {
          Flags &=
              ~DWARFDebugMacro::HeaderFlagMask::MACRO_OPCODE_OPERANDS_TABLE;
          warn("opcode_operands_table is not supported yet.");
        }

        // Check for DEBUG_LINE_OFFSET.
        std::optional<uint64_t> StmtListOffset;
        if (Flags & DWARFDebugMacro::HeaderFlagMask::MACRO_DEBUG_LINE_OFFSET) {
          // Get offset to the line table from the cloned compile unit.
          for (auto &V : getOutUnitDIE()->values()) {
            if (V.getAttribute() == dwarf::DW_AT_stmt_list) {
              StmtListOffset = V.getDIEInteger().getValue();
              break;
            }
          }

          if (!StmtListOffset) {
            Flags &= ~DWARFDebugMacro::HeaderFlagMask::MACRO_DEBUG_LINE_OFFSET;
            warn("couldn`t find line table for macro table.");
          }
        }

        // Write flags.
        OutSection.emitIntVal(Flags, sizeof(Flags));

        // Write offset to line table.
        if (StmtListOffset) {
          OutSection.notePatch(DebugOffsetPatch{
              OutSection.OS.tell(),
              &getOrCreateSectionDescriptor(DebugSectionKind::DebugLine)});
          // TODO: check that List.Header.getOffsetByteSize() and
          // DebugOffsetPatch agree on size.
          OutSection.emitIntVal(0xBADDEF, List.Header.getOffsetByteSize());
        }
      }

      // Write macro entries.
      for (const DWARFDebugMacro::Entry &MacroEntry : List.Macros) {
        if (MacroEntry.Type == 0) {
          encodeULEB128(MacroEntry.Type, OutSection.OS);
          continue;
        }

        uint8_t MacroType = MacroEntry.Type;
        switch (MacroType) {
        default: {
          bool HasVendorSpecificExtension =
              (!hasDWARFv5Header &&
               MacroType == dwarf::DW_MACINFO_vendor_ext) ||
              (hasDWARFv5Header && (MacroType >= dwarf::DW_MACRO_lo_user &&
                                    MacroType <= dwarf::DW_MACRO_hi_user));

          if (HasVendorSpecificExtension) {
            // Write macinfo type.
            OutSection.emitIntVal(MacroType, 1);

            // Write vendor extension constant.
            encodeULEB128(MacroEntry.ExtConstant, OutSection.OS);

            // Write vendor extension string.
            OutSection.emitString(dwarf::DW_FORM_string, MacroEntry.ExtStr);
          } else
            warn("unknown macro type. skip.");
        } break;
        // debug_macro and debug_macinfo share some common encodings.
        // DW_MACRO_define     == DW_MACINFO_define
        // DW_MACRO_undef      == DW_MACINFO_undef
        // DW_MACRO_start_file == DW_MACINFO_start_file
        // DW_MACRO_end_file   == DW_MACINFO_end_file
        // For readibility/uniformity we are using DW_MACRO_*.
        case dwarf::DW_MACRO_define:
        case dwarf::DW_MACRO_undef: {
          // Write macinfo type.
          OutSection.emitIntVal(MacroType, 1);

          // Write source line.
          encodeULEB128(MacroEntry.Line, OutSection.OS);

          // Write macro string.
          OutSection.emitString(dwarf::DW_FORM_string, MacroEntry.MacroStr);
        } break;
        case dwarf::DW_MACRO_define_strp:
        case dwarf::DW_MACRO_undef_strp:
        case dwarf::DW_MACRO_define_strx:
        case dwarf::DW_MACRO_undef_strx: {
          // DW_MACRO_*_strx forms are not supported currently.
          // Convert to *_strp.
          switch (MacroType) {
          case dwarf::DW_MACRO_define_strx: {
            MacroType = dwarf::DW_MACRO_define_strp;
            if (!DefAttributeIsReported) {
              warn("DW_MACRO_define_strx unsupported yet. Convert to "
                   "DW_MACRO_define_strp.");
              DefAttributeIsReported = true;
            }
          } break;
          case dwarf::DW_MACRO_undef_strx: {
            MacroType = dwarf::DW_MACRO_undef_strp;
            if (!UndefAttributeIsReported) {
              warn("DW_MACRO_undef_strx unsupported yet. Convert to "
                   "DW_MACRO_undef_strp.");
              UndefAttributeIsReported = true;
            }
          } break;
          default:
            // Nothing to do.
            break;
          }

          // Write macinfo type.
          OutSection.emitIntVal(MacroType, 1);

          // Write source line.
          encodeULEB128(MacroEntry.Line, OutSection.OS);

          // Write macro string.
          OutSection.emitString(dwarf::DW_FORM_strp, MacroEntry.MacroStr);
          break;
        }
        case dwarf::DW_MACRO_start_file: {
          // Write macinfo type.
          OutSection.emitIntVal(MacroType, 1);
          // Write source line.
          encodeULEB128(MacroEntry.Line, OutSection.OS);
          // Write source file id.
          encodeULEB128(MacroEntry.File, OutSection.OS);
        } break;
        case dwarf::DW_MACRO_end_file: {
          // Write macinfo type.
          OutSection.emitIntVal(MacroType, 1);
        } break;
        case dwarf::DW_MACRO_import:
        case dwarf::DW_MACRO_import_sup: {
          if (!ImportAttributeIsReported) {
            warn("DW_MACRO_import and DW_MACRO_import_sup are unsupported "
                 "yet. remove.");
            ImportAttributeIsReported = true;
          }
        } break;
        }
      }

      return;
    }
  }
}

void CompileUnit::cloneDieAttrExpression(
    const DWARFExpression &InputExpression,
    SmallVectorImpl<uint8_t> &OutputExpression, SectionDescriptor &Section,
    std::optional<int64_t> VarAddressAdjustment,
    OffsetsPtrVector &PatchesOffsets) {
  using Encoding = DWARFExpression::Operation::Encoding;

  DWARFUnit &OrigUnit = getOrigUnit();
  uint8_t OrigAddressByteSize = OrigUnit.getAddressByteSize();

  uint64_t OpOffset = 0;
  for (auto &Op : InputExpression) {
    auto Desc = Op.getDescription();
    // DW_OP_const_type is variable-length and has 3
    // operands. Thus far we only support 2.
    if ((Desc.Op.size() == 2 && Desc.Op[0] == Encoding::BaseTypeRef) ||
        (Desc.Op.size() == 2 && Desc.Op[1] == Encoding::BaseTypeRef &&
         Desc.Op[0] != Encoding::Size1))
      warn("unsupported DW_OP encoding.");

    if ((Desc.Op.size() == 1 && Desc.Op[0] == Encoding::BaseTypeRef) ||
        (Desc.Op.size() == 2 && Desc.Op[1] == Encoding::BaseTypeRef &&
         Desc.Op[0] == Encoding::Size1)) {
      // This code assumes that the other non-typeref operand fits into 1 byte.
      assert(OpOffset < Op.getEndOffset());
      uint32_t ULEBsize = Op.getEndOffset() - OpOffset - 1;
      assert(ULEBsize <= 16);

      // Copy over the operation.
      assert(!Op.getSubCode() && "SubOps not yet supported");
      OutputExpression.push_back(Op.getCode());
      uint64_t RefOffset;
      if (Desc.Op.size() == 1) {
        RefOffset = Op.getRawOperand(0);
      } else {
        OutputExpression.push_back(Op.getRawOperand(0));
        RefOffset = Op.getRawOperand(1);
      }
      uint8_t ULEB[16];
      uint32_t Offset = 0;
      unsigned RealSize = 0;
      // Look up the base type. For DW_OP_convert, the operand may be 0 to
      // instead indicate the generic type. The same holds for
      // DW_OP_reinterpret, which is currently not supported.
      if (RefOffset > 0 || Op.getCode() != dwarf::DW_OP_convert) {
        RefOffset += OrigUnit.getOffset();
        uint32_t RefDieIdx = 0;
        if (std::optional<uint32_t> Idx =
                OrigUnit.getDIEIndexForOffset(RefOffset))
          RefDieIdx = *Idx;

        // Use fixed size for ULEB128 data, since we need to update that size
        // later with the proper offsets. Use 5 for DWARF32, 9 for DWARF64.
        ULEBsize = getFormParams().getDwarfOffsetByteSize() + 1;

        RealSize = encodeULEB128(0xBADDEF, ULEB, ULEBsize);

        Section.notePatchWithOffsetUpdate(
            DebugULEB128DieRefPatch(OutputExpression.size(), this, this,
                                    RefDieIdx),
            PatchesOffsets);
      } else
        RealSize = encodeULEB128(Offset, ULEB, ULEBsize);

      if (RealSize > ULEBsize) {
        // Emit the generic type as a fallback.
        RealSize = encodeULEB128(0, ULEB, ULEBsize);
        warn("base type ref doesn't fit.");
      }
      assert(RealSize == ULEBsize && "padding failed");
      ArrayRef<uint8_t> ULEBbytes(ULEB, ULEBsize);
      OutputExpression.append(ULEBbytes.begin(), ULEBbytes.end());
    } else if (!getGlobalData().getOptions().UpdateIndexTablesOnly &&
               Op.getCode() == dwarf::DW_OP_addrx) {
      if (std::optional<object::SectionedAddress> SA =
              OrigUnit.getAddrOffsetSectionItem(Op.getRawOperand(0))) {
        // DWARFLinker does not use addrx forms since it generates relocated
        // addresses. Replace DW_OP_addrx with DW_OP_addr here.
        // Argument of DW_OP_addrx should be relocated here as it is not
        // processed by applyValidRelocs.
        OutputExpression.push_back(dwarf::DW_OP_addr);
        uint64_t LinkedAddress =
            SA->Address + (VarAddressAdjustment ? *VarAddressAdjustment : 0);
        if (getEndianness() != llvm::endianness::native)
          sys::swapByteOrder(LinkedAddress);
        ArrayRef<uint8_t> AddressBytes(
            reinterpret_cast<const uint8_t *>(&LinkedAddress),
            OrigAddressByteSize);
        OutputExpression.append(AddressBytes.begin(), AddressBytes.end());
      } else
        warn("cann't read DW_OP_addrx operand.");
    } else if (!getGlobalData().getOptions().UpdateIndexTablesOnly &&
               Op.getCode() == dwarf::DW_OP_constx) {
      if (std::optional<object::SectionedAddress> SA =
              OrigUnit.getAddrOffsetSectionItem(Op.getRawOperand(0))) {
        // DWARFLinker does not use constx forms since it generates relocated
        // addresses. Replace DW_OP_constx with DW_OP_const[*]u here.
        // Argument of DW_OP_constx should be relocated here as it is not
        // processed by applyValidRelocs.
        std::optional<uint8_t> OutOperandKind;
        switch (OrigAddressByteSize) {
        case 2:
          OutOperandKind = dwarf::DW_OP_const2u;
          break;
        case 4:
          OutOperandKind = dwarf::DW_OP_const4u;
          break;
        case 8:
          OutOperandKind = dwarf::DW_OP_const8u;
          break;
        default:
          warn(
              formatv(("unsupported address size: {0}."), OrigAddressByteSize));
          break;
        }

        if (OutOperandKind) {
          OutputExpression.push_back(*OutOperandKind);
          uint64_t LinkedAddress =
              SA->Address + (VarAddressAdjustment ? *VarAddressAdjustment : 0);
          if (getEndianness() != llvm::endianness::native)
            sys::swapByteOrder(LinkedAddress);
          ArrayRef<uint8_t> AddressBytes(
              reinterpret_cast<const uint8_t *>(&LinkedAddress),
              OrigAddressByteSize);
          OutputExpression.append(AddressBytes.begin(), AddressBytes.end());
        }
      } else
        warn("cann't read DW_OP_constx operand.");
    } else {
      // Copy over everything else unmodified.
      StringRef Bytes =
          InputExpression.getData().slice(OpOffset, Op.getEndOffset());
      OutputExpression.append(Bytes.begin(), Bytes.end());
    }
    OpOffset = Op.getEndOffset();
  }
}

Error CompileUnit::cloneAndEmit(std::optional<Triple> TargetTriple) {
  BumpPtrAllocator Allocator;

  DWARFDie OrigUnitDIE = getOrigUnit().getUnitDIE();
  if (!OrigUnitDIE.isValid())
    return Error::success();

  // Clone input DIE entry recursively.
  DIE *OutCUDie =
      cloneDIE(OrigUnitDIE.getDebugInfoEntry(), getDebugInfoHeaderSize(),
               std::nullopt, std::nullopt, Allocator);
  setOutUnitDIE(OutCUDie);

  if (getGlobalData().getOptions().NoOutput || (OutCUDie == nullptr))
    return Error::success();

  assert(TargetTriple.has_value());
  if (Error Err = cloneAndEmitLineTable(*TargetTriple))
    return Err;

  if (Error Err = cloneAndEmitDebugMacro())
    return Err;

  if (Error Err = emitDebugInfo(*TargetTriple))
    return Err;

  // ASSUMPTION: .debug_info section should already be emitted at this point.
  // cloneAndEmitRanges & cloneAndEmitDebugLocations use .debug_info section
  // data.

  if (Error Err = cloneAndEmitRanges())
    return Err;

  if (Error Err = cloneAndEmitDebugLocations())
    return Err;

  if (Error Err = emitDebugAddrSection())
    return Err;

  // Generate Pub accelerator tables.
  if (llvm::is_contained(GlobalData.getOptions().AccelTables,
                         DWARFLinker::AccelTableKind::Pub))
    emitPubAccelerators();

  if (Error Err = emitDebugStringOffsetSection())
    return Err;

  return emitAbbreviations();
}

bool needToClone(CompileUnit::DIEInfo &Info) {
  return Info.getKeep() || Info.getKeepChildren();
}

DIE *CompileUnit::cloneDIE(const DWARFDebugInfoEntry *InputDieEntry,
                           uint64_t OutOffset,
                           std::optional<int64_t> FuncAddressAdjustment,
                           std::optional<int64_t> VarAddressAdjustment,
                           BumpPtrAllocator &Allocator) {
  uint32_t InputDieIdx = getDIEIndex(InputDieEntry);
  CompileUnit::DIEInfo &Info = getDIEInfo(InputDieIdx);

  if (!needToClone(Info))
    return nullptr;

  bool HasLocationExpressionAddress = false;
  if (InputDieEntry->getTag() == dwarf::DW_TAG_subprogram) {
    // Get relocation adjustment value for the current function.
    FuncAddressAdjustment =
        getContaingFile().Addresses->getSubprogramRelocAdjustment(
            getDIE(InputDieEntry));
  } else if (InputDieEntry->getTag() == dwarf::DW_TAG_variable) {
    // Get relocation adjustment value for the current variable.
    std::pair<bool, std::optional<int64_t>> LocExprAddrAndRelocAdjustment =
        getContaingFile().Addresses->getVariableRelocAdjustment(
            getDIE(InputDieEntry));

    HasLocationExpressionAddress = LocExprAddrAndRelocAdjustment.first;
    if (LocExprAddrAndRelocAdjustment.first &&
        LocExprAddrAndRelocAdjustment.second)
      VarAddressAdjustment = *LocExprAddrAndRelocAdjustment.second;
  }

  DIEGenerator DIEGenerator(Allocator, *this);
  DIE *ClonedDIE = DIEGenerator.createDIE(InputDieEntry->getTag(), OutOffset);
  rememberDieOutOffset(InputDieIdx, OutOffset);

  // Clone Attributes.
  DIEAttributeCloner AttributesCloner(
      ClonedDIE, *this, InputDieEntry, DIEGenerator, FuncAddressAdjustment,
      VarAddressAdjustment, HasLocationExpressionAddress);
  AttributesCloner.clone();

  // Remember accelerator info.
  rememberAcceleratorEntries(InputDieEntry, OutOffset,
                             AttributesCloner.AttrInfo);

  bool HasChildrenToClone = Info.getKeepChildren();
  OutOffset = AttributesCloner.finalizeAbbreviations(HasChildrenToClone);

  if (HasChildrenToClone) {
    // Recursively clone children.
    for (const DWARFDebugInfoEntry *CurChild =
             getFirstChildEntry(InputDieEntry);
         CurChild && CurChild->getAbbreviationDeclarationPtr();
         CurChild = getSiblingEntry(CurChild)) {
      if (DIE *ClonedChild =
              cloneDIE(CurChild, OutOffset, FuncAddressAdjustment,
                       VarAddressAdjustment, Allocator)) {
        OutOffset = ClonedChild->getOffset() + ClonedChild->getSize();
        DIEGenerator.addChild(ClonedChild);
      }
    }

    // Account for the end of children marker.
    OutOffset += sizeof(int8_t);
  }

  // Update our size.
  ClonedDIE->setSize(OutOffset - ClonedDIE->getOffset());
  return ClonedDIE;
}

Error CompileUnit::cloneAndEmitLineTable(Triple &TargetTriple) {
  const DWARFDebugLine::LineTable *InputLineTable =
      getContaingFile().Dwarf->getLineTableForUnit(&getOrigUnit());
  if (InputLineTable == nullptr) {
    warn("cann't load line table.");
    return Error::success();
  }

  DWARFDebugLine::LineTable OutLineTable;

  // Set Line Table header.
  OutLineTable.Prologue = InputLineTable->Prologue;
  OutLineTable.Prologue.FormParams.AddrSize = getFormParams().AddrSize;

  // Set Line Table Rows.
  if (getGlobalData().getOptions().UpdateIndexTablesOnly) {
    OutLineTable.Rows = InputLineTable->Rows;
    // If all the line table contains is a DW_LNE_end_sequence, clear the line
    // table rows, it will be inserted again in the DWARFStreamer.
    if (OutLineTable.Rows.size() == 1 && OutLineTable.Rows[0].EndSequence)
      OutLineTable.Rows.clear();

    OutLineTable.Sequences = InputLineTable->Sequences;
  } else {
    // This vector is the output line table.
    std::vector<DWARFDebugLine::Row> NewRows;
    NewRows.reserve(InputLineTable->Rows.size());

    // Current sequence of rows being extracted, before being inserted
    // in NewRows.
    std::vector<DWARFDebugLine::Row> Seq;

    const auto &FunctionRanges = getFunctionRanges();
    std::optional<AddressRangeValuePair> CurrRange;

    // FIXME: This logic is meant to generate exactly the same output as
    // Darwin's classic dsymutil. There is a nicer way to implement this
    // by simply putting all the relocated line info in NewRows and simply
    // sorting NewRows before passing it to emitLineTableForUnit. This
    // should be correct as sequences for a function should stay
    // together in the sorted output. There are a few corner cases that
    // look suspicious though, and that required to implement the logic
    // this way. Revisit that once initial validation is finished.

    // Iterate over the object file line info and extract the sequences
    // that correspond to linked functions.
    for (DWARFDebugLine::Row Row : InputLineTable->Rows) {
      // Check whether we stepped out of the range. The range is
      // half-open, but consider accept the end address of the range if
      // it is marked as end_sequence in the input (because in that
      // case, the relocation offset is accurate and that entry won't
      // serve as the start of another function).
      if (!CurrRange || !CurrRange->Range.contains(Row.Address.Address)) {
        // We just stepped out of a known range. Insert a end_sequence
        // corresponding to the end of the range.
        uint64_t StopAddress =
            CurrRange ? CurrRange->Range.end() + CurrRange->Value : -1ULL;
        CurrRange = FunctionRanges.getRangeThatContains(Row.Address.Address);
        if (StopAddress != -1ULL && !Seq.empty()) {
          // Insert end sequence row with the computed end address, but
          // the same line as the previous one.
          auto NextLine = Seq.back();
          NextLine.Address.Address = StopAddress;
          NextLine.EndSequence = 1;
          NextLine.PrologueEnd = 0;
          NextLine.BasicBlock = 0;
          NextLine.EpilogueBegin = 0;
          Seq.push_back(NextLine);
          insertLineSequence(Seq, NewRows);
        }

        if (!CurrRange)
          continue;
      }

      // Ignore empty sequences.
      if (Row.EndSequence && Seq.empty())
        continue;

      // Relocate row address and add it to the current sequence.
      Row.Address.Address += CurrRange->Value;
      Seq.emplace_back(Row);

      if (Row.EndSequence)
        insertLineSequence(Seq, NewRows);
    }

    OutLineTable.Rows = std::move(NewRows);
  }

  return emitDebugLine(TargetTriple, OutLineTable);
}

void CompileUnit::insertLineSequence(std::vector<DWARFDebugLine::Row> &Seq,
                                     std::vector<DWARFDebugLine::Row> &Rows) {
  if (Seq.empty())
    return;

  if (!Rows.empty() && Rows.back().Address < Seq.front().Address) {
    llvm::append_range(Rows, Seq);
    Seq.clear();
    return;
  }

  object::SectionedAddress Front = Seq.front().Address;
  auto InsertPoint = partition_point(
      Rows, [=](const DWARFDebugLine::Row &O) { return O.Address < Front; });

  // FIXME: this only removes the unneeded end_sequence if the
  // sequences have been inserted in order. Using a global sort like
  // described in cloneAndEmitLineTable() and delaying the end_sequene
  // elimination to DebugLineEmitter::emit() we can get rid of all of them.
  if (InsertPoint != Rows.end() && InsertPoint->Address == Front &&
      InsertPoint->EndSequence) {
    *InsertPoint = Seq.front();
    Rows.insert(InsertPoint + 1, Seq.begin() + 1, Seq.end());
  } else {
    Rows.insert(InsertPoint, Seq.begin(), Seq.end());
  }

  Seq.clear();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CompileUnit::DIEInfo::dump() {
  llvm::errs() << "{\n";
  llvm::errs() << "  Placement: ";
  switch (getPlacement()) {
  case NotSet:
    llvm::errs() << "NotSet\n";
    break;
  case TypeTable:
    llvm::errs() << "TypeTable\n";
    break;
  case PlainDwarf:
    llvm::errs() << "PlainDwarf\n";
    break;
  case Both:
    llvm::errs() << "Both\n";
    break;
  case Parent:
    llvm::errs() << "Parent\n";
    break;
  }

  llvm::errs() << "  Keep: " << getKeep();
  llvm::errs() << "  KeepChildren: " << getKeepChildren();
  llvm::errs() << "  ReferrencedBy: " << getReferrencedBy();
  llvm::errs() << "  IsInMouduleScope: " << getIsInMouduleScope();
  llvm::errs() << "  IsInFunctionScope: " << getIsInFunctionScope();
  llvm::errs() << "}\n";
}
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

static uint32_t hashFullyQualifiedName(CompileUnit *InputCU, DWARFDie &InputDIE,
                                       int ChildRecurseDepth = 0) {
  const char *Name = nullptr;
  CompileUnit *CU = InputCU;
  std::optional<DWARFFormValue> RefVal;

  // Usually name`s depth does not exceed 3. Set maximal depth
  // to 1000 here, to avoid infinite loop in case incorrect input
  // DWARF.
  size_t MaxNameDepth = 1000;
  size_t CurNameDepth = 0;

  while (CurNameDepth++ < MaxNameDepth) {
    if (const char *CurrentName = InputDIE.getName(DINameKind::ShortName))
      Name = CurrentName;

    if (!(RefVal = InputDIE.find(dwarf::DW_AT_specification)) &&
        !(RefVal = InputDIE.find(dwarf::DW_AT_abstract_origin)))
      break;

    if (!RefVal->isFormClass(DWARFFormValue::FC_Reference))
      break;

    std::optional<std::pair<CompileUnit *, uint32_t>> RefDie =
        CU->resolveDIEReference(*RefVal, ResolveInterCUReferencesMode::Resolve);
    if (!RefDie)
      break;

    assert(RefDie->second != 0);

    CU = RefDie->first;
    InputDIE = RefDie->first->getDIEAtIndex(RefDie->second);
  }

  if (!Name && InputDIE.getTag() == dwarf::DW_TAG_namespace)
    Name = "(anonymous namespace)";

  DWARFDie ParentDie = InputDIE.getParent();
  if (!ParentDie.isValid() || ParentDie.getTag() == dwarf::DW_TAG_compile_unit)
    return djbHash(Name ? Name : "", djbHash(ChildRecurseDepth ? "" : "::"));

  return djbHash(
      (Name ? Name : ""),
      djbHash((Name ? "::" : ""),
              hashFullyQualifiedName(CU, ParentDie, ++ChildRecurseDepth)));
}

void CompileUnit::rememberAcceleratorEntries(
    const DWARFDebugInfoEntry *InputDieEntry, uint64_t OutOffset,
    AttributesInfo &AttrInfo) {
  if (GlobalData.getOptions().AccelTables.empty())
    return;

  DWARFDie InputDIE = getDIE(InputDieEntry);

  // Look for short name recursively if short name is not known yet.
  if (AttrInfo.Name == nullptr)
    if (const char *ShortName = InputDIE.getShortName())
      AttrInfo.Name = getGlobalData().getStringPool().insert(ShortName).first;

  switch (InputDieEntry->getTag()) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_string_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_set_type:
  case dwarf::DW_TAG_subrange_type:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_file_type:
  case dwarf::DW_TAG_namelist:
  case dwarf::DW_TAG_packed_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_atomic_type:
  case dwarf::DW_TAG_interface_type:
  case dwarf::DW_TAG_unspecified_type:
  case dwarf::DW_TAG_shared_type:
  case dwarf::DW_TAG_immutable_type:
  case dwarf::DW_TAG_rvalue_reference_type: {
    if (!AttrInfo.IsDeclaration && AttrInfo.Name != nullptr &&
        !AttrInfo.Name->getKey().empty()) {
      uint32_t Hash = hashFullyQualifiedName(this, InputDIE);

      uint64_t RuntimeLang =
          dwarf::toUnsigned(InputDIE.find(dwarf::DW_AT_APPLE_runtime_class))
              .value_or(0);

      bool ObjCClassIsImplementation =
          (RuntimeLang == dwarf::DW_LANG_ObjC ||
           RuntimeLang == dwarf::DW_LANG_ObjC_plus_plus) &&
          dwarf::toUnsigned(
              InputDIE.find(dwarf::DW_AT_APPLE_objc_complete_type))
              .value_or(0);

      rememberTypeForAccelerators(AttrInfo.Name, OutOffset,
                                  InputDieEntry->getTag(), Hash,
                                  ObjCClassIsImplementation);
    }
  } break;
  case dwarf::DW_TAG_namespace: {
    if (AttrInfo.Name == nullptr)
      AttrInfo.Name =
          getGlobalData().getStringPool().insert("(anonymous namespace)").first;

    rememberNamespaceForAccelerators(AttrInfo.Name, OutOffset,
                                     InputDieEntry->getTag());
  } break;
  case dwarf::DW_TAG_imported_declaration: {
    if (AttrInfo.Name != nullptr)
      rememberNamespaceForAccelerators(AttrInfo.Name, OutOffset,
                                       InputDieEntry->getTag());
  } break;
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_lexical_block: {
    // Nothing to do.
  } break;
  default:
    if (AttrInfo.HasLiveAddress || AttrInfo.HasRanges) {
      if (AttrInfo.Name != nullptr)
        rememberNameForAccelerators(
            AttrInfo.Name, OutOffset, InputDieEntry->getTag(),
            InputDieEntry->getTag() == dwarf::DW_TAG_inlined_subroutine);

      // Look for mangled name recursively if mangled name is not known yet.
      if (AttrInfo.MangledName == nullptr)
        if (const char *LinkageName = InputDIE.getLinkageName())
          AttrInfo.MangledName =
              getGlobalData().getStringPool().insert(LinkageName).first;

      if (AttrInfo.MangledName != nullptr &&
          AttrInfo.MangledName != AttrInfo.Name)
        rememberNameForAccelerators(
            AttrInfo.MangledName, OutOffset, InputDieEntry->getTag(),
            InputDieEntry->getTag() == dwarf::DW_TAG_inlined_subroutine);

      // Strip template parameters from the short name.
      if (AttrInfo.Name != nullptr && AttrInfo.MangledName != AttrInfo.Name &&
          (InputDieEntry->getTag() != dwarf::DW_TAG_inlined_subroutine)) {
        if (std::optional<StringRef> Name =
                StripTemplateParameters(AttrInfo.Name->getKey())) {
          StringEntry *NameWithoutTemplateParams =
              getGlobalData().getStringPool().insert(*Name).first;

          rememberNameForAccelerators(NameWithoutTemplateParams, OutOffset,
                                      InputDieEntry->getTag(), true);
        }
      }

      if (AttrInfo.Name)
        rememberObjCAccelerator(InputDieEntry, OutOffset, AttrInfo);
    }
    break;
  }
}

void CompileUnit::rememberObjCAccelerator(
    const DWARFDebugInfoEntry *InputDieEntry, uint64_t OutOffset,
    AttributesInfo &AttrInfo) {
  std::optional<ObjCSelectorNames> Names =
      getObjCNamesIfSelector(AttrInfo.Name->getKey());
  if (!Names)
    return;

  StringEntry *Selector =
      getGlobalData().getStringPool().insert(Names->Selector).first;
  rememberNameForAccelerators(Selector, OutOffset, InputDieEntry->getTag(),
                              true);
  StringEntry *ClassName =
      getGlobalData().getStringPool().insert(Names->ClassName).first;
  rememberObjCNameForAccelerators(ClassName, OutOffset,
                                  InputDieEntry->getTag());
  if (Names->ClassNameNoCategory) {
    StringEntry *ClassNameNoCategory = getGlobalData()
                                           .getStringPool()
                                           .insert(*Names->ClassNameNoCategory)
                                           .first;
    rememberObjCNameForAccelerators(ClassNameNoCategory, OutOffset,
                                    InputDieEntry->getTag());
  }
  if (Names->MethodNameNoCategory) {
    StringEntry *MethodNameNoCategory =
        getGlobalData()
            .getStringPool()
            .insert(*Names->MethodNameNoCategory)
            .first;
    rememberNameForAccelerators(MethodNameNoCategory, OutOffset,
                                InputDieEntry->getTag(), true);
  }
}
