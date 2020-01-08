//===- DWARFLinkerCompileUnit.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFLINKERCOMPILEUNIT_H
#define LLVM_DWARFLINKER_DWARFLINKERCOMPILEUNIT_H

#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class DeclContext;

template <typename KeyT, typename ValT>
using HalfOpenIntervalMap =
    IntervalMap<KeyT, ValT, IntervalMapImpl::NodeSizer<KeyT, ValT>::LeafSize,
                IntervalMapHalfOpenInfo<KeyT>>;

using FunctionIntervals = HalfOpenIntervalMap<uint64_t, int64_t>;

// FIXME: Delete this structure.
struct PatchLocation {
  DIE::value_iterator I;

  PatchLocation() = default;
  PatchLocation(DIE::value_iterator I) : I(I) {}

  void set(uint64_t New) const {
    assert(I);
    const auto &Old = *I;
    assert(Old.getType() == DIEValue::isInteger);
    *I = DIEValue(Old.getAttribute(), Old.getForm(), DIEInteger(New));
  }

  uint64_t get() const {
    assert(I);
    return I->getDIEInteger().getValue();
  }
};

/// Stores all information relating to a compile unit, be it in its original
/// instance in the object file to its brand new cloned and generated DIE tree.
class CompileUnit {
public:
  /// Information gathered about a DIE in the object file.
  struct DIEInfo {
    /// Address offset to apply to the described entity.
    int64_t AddrAdjust;

    /// ODR Declaration context.
    DeclContext *Ctxt;

    /// Cloned version of that DIE.
    DIE *Clone;

    /// The index of this DIE's parent.
    uint32_t ParentIdx;

    /// Is the DIE part of the linked output?
    bool Keep : 1;

    /// Was this DIE's entity found in the map?
    bool InDebugMap : 1;

    /// Is this a pure forward declaration we can strip?
    bool Prune : 1;

    /// Does DIE transitively refer an incomplete decl?
    bool Incomplete : 1;
  };

  CompileUnit(DWARFUnit &OrigUnit, unsigned ID, bool CanUseODR,
              StringRef ClangModuleName)
      : OrigUnit(OrigUnit), ID(ID), Ranges(RangeAlloc),
        ClangModuleName(ClangModuleName) {
    Info.resize(OrigUnit.getNumDIEs());

    auto CUDie = OrigUnit.getUnitDIE(false);
    if (!CUDie) {
      HasODR = false;
      return;
    }
    if (auto Lang = dwarf::toUnsigned(CUDie.find(dwarf::DW_AT_language)))
      HasODR = CanUseODR && (*Lang == dwarf::DW_LANG_C_plus_plus ||
                             *Lang == dwarf::DW_LANG_C_plus_plus_03 ||
                             *Lang == dwarf::DW_LANG_C_plus_plus_11 ||
                             *Lang == dwarf::DW_LANG_C_plus_plus_14 ||
                             *Lang == dwarf::DW_LANG_ObjC_plus_plus);
    else
      HasODR = false;
  }

  DWARFUnit &getOrigUnit() const { return OrigUnit; }

  unsigned getUniqueID() const { return ID; }

  void createOutputDIE() {
    NewUnit.emplace(OrigUnit.getVersion(), OrigUnit.getAddressByteSize(),
                    OrigUnit.getUnitDIE().getTag());
  }

  DIE *getOutputUnitDIE() const {
    if (NewUnit)
      return &const_cast<BasicDIEUnit &>(*NewUnit).getUnitDie();
    return nullptr;
  }

  bool hasODR() const { return HasODR; }
  bool isClangModule() const { return !ClangModuleName.empty(); }
  uint16_t getLanguage();

  const std::string &getClangModuleName() const { return ClangModuleName; }

  DIEInfo &getInfo(unsigned Idx) { return Info[Idx]; }
  const DIEInfo &getInfo(unsigned Idx) const { return Info[Idx]; }

  uint64_t getStartOffset() const { return StartOffset; }
  uint64_t getNextUnitOffset() const { return NextUnitOffset; }
  void setStartOffset(uint64_t DebugInfoSize) { StartOffset = DebugInfoSize; }

  uint64_t getLowPc() const { return LowPc; }
  uint64_t getHighPc() const { return HighPc; }
  bool hasLabelAt(uint64_t Addr) const { return Labels.count(Addr); }

  Optional<PatchLocation> getUnitRangesAttribute() const {
    return UnitRangeAttribute;
  }

  const FunctionIntervals &getFunctionRanges() const { return Ranges; }

  const std::vector<PatchLocation> &getRangesAttributes() const {
    return RangeAttributes;
  }

  const std::vector<std::pair<PatchLocation, int64_t>> &
  getLocationAttributes() const {
    return LocationAttributes;
  }

  void setHasInterestingContent() { HasInterestingContent = true; }
  bool hasInterestingContent() { return HasInterestingContent; }

  /// Mark every DIE in this unit as kept. This function also
  /// marks variables as InDebugMap so that they appear in the
  /// reconstructed accelerator tables.
  void markEverythingAsKept();

  /// Compute the end offset for this unit. Must be called after the CU's DIEs
  /// have been cloned.  \returns the next unit offset (which is also the
  /// current debug_info section size).
  uint64_t computeNextUnitOffset();

  /// Keep track of a forward reference to DIE \p Die in \p RefUnit by \p
  /// Attr. The attribute should be fixed up later to point to the absolute
  /// offset of \p Die in the debug_info section or to the canonical offset of
  /// \p Ctxt if it is non-null.
  void noteForwardReference(DIE *Die, const CompileUnit *RefUnit,
                            DeclContext *Ctxt, PatchLocation Attr);

  /// Apply all fixups recorded by noteForwardReference().
  void fixupForwardReferences();

  /// Add the low_pc of a label that is relocated by applying
  /// offset \p PCOffset.
  void addLabelLowPc(uint64_t LabelLowPc, int64_t PcOffset);

  /// Add a function range [\p LowPC, \p HighPC) that is relocated by applying
  /// offset \p PCOffset.
  void addFunctionRange(uint64_t LowPC, uint64_t HighPC, int64_t PCOffset);

  /// Keep track of a DW_AT_range attribute that we will need to patch up later.
  void noteRangeAttribute(const DIE &Die, PatchLocation Attr);

  /// Keep track of a location attribute pointing to a location list in the
  /// debug_loc section.
  void noteLocationAttribute(PatchLocation Attr, int64_t PcOffset);

  /// Add a name accelerator entry for \a Die with \a Name.
  void addNamespaceAccelerator(const DIE *Die, DwarfStringPoolEntryRef Name);

  /// Add a name accelerator entry for \a Die with \a Name.
  void addNameAccelerator(const DIE *Die, DwarfStringPoolEntryRef Name,
                          bool SkipPubnamesSection = false);

  /// Add various accelerator entries for \p Die with \p Name which is stored
  /// in the string table at \p Offset. \p Name must be an Objective-C
  /// selector.
  void addObjCAccelerator(const DIE *Die, DwarfStringPoolEntryRef Name,
                          bool SkipPubnamesSection = false);

  /// Add a type accelerator entry for \p Die with \p Name which is stored in
  /// the string table at \p Offset.
  void addTypeAccelerator(const DIE *Die, DwarfStringPoolEntryRef Name,
                          bool ObjcClassImplementation,
                          uint32_t QualifiedNameHash);

  struct AccelInfo {
    /// Name of the entry.
    DwarfStringPoolEntryRef Name;

    /// DIE this entry describes.
    const DIE *Die;

    /// Hash of the fully qualified name.
    uint32_t QualifiedNameHash;

    /// Emit this entry only in the apple_* sections.
    bool SkipPubSection;

    /// Is this an ObjC class implementation?
    bool ObjcClassImplementation;

    AccelInfo(DwarfStringPoolEntryRef Name, const DIE *Die,
              bool SkipPubSection = false)
        : Name(Name), Die(Die), SkipPubSection(SkipPubSection) {}

    AccelInfo(DwarfStringPoolEntryRef Name, const DIE *Die,
              uint32_t QualifiedNameHash, bool ObjCClassIsImplementation)
        : Name(Name), Die(Die), QualifiedNameHash(QualifiedNameHash),
          SkipPubSection(false),
          ObjcClassImplementation(ObjCClassIsImplementation) {}
  };

  const std::vector<AccelInfo> &getPubnames() const { return Pubnames; }
  const std::vector<AccelInfo> &getPubtypes() const { return Pubtypes; }
  const std::vector<AccelInfo> &getNamespaces() const { return Namespaces; }
  const std::vector<AccelInfo> &getObjC() const { return ObjC; }

  /// Get the full path for file \a FileNum in the line table
  StringRef getResolvedPath(unsigned FileNum) {
    if (FileNum >= ResolvedPaths.size())
      return StringRef();
    return ResolvedPaths[FileNum];
  }

  /// Set the fully resolved path for the line-table's file \a FileNum
  /// to \a Path.
  void setResolvedPath(unsigned FileNum, StringRef Path) {
    if (ResolvedPaths.size() <= FileNum)
      ResolvedPaths.resize(FileNum + 1);
    ResolvedPaths[FileNum] = Path;
  }

  MCSymbol *getLabelBegin() { return LabelBegin; }
  void setLabelBegin(MCSymbol *S) { LabelBegin = S; }

private:
  DWARFUnit &OrigUnit;
  unsigned ID;
  std::vector<DIEInfo> Info; ///< DIE info indexed by DIE index.
  Optional<BasicDIEUnit> NewUnit;
  MCSymbol *LabelBegin = nullptr;

  uint64_t StartOffset;
  uint64_t NextUnitOffset;

  uint64_t LowPc = std::numeric_limits<uint64_t>::max();
  uint64_t HighPc = 0;

  /// A list of attributes to fixup with the absolute offset of
  /// a DIE in the debug_info section.
  ///
  /// The offsets for the attributes in this array couldn't be set while
  /// cloning because for cross-cu forward references the target DIE's offset
  /// isn't known you emit the reference attribute.
  std::vector<
      std::tuple<DIE *, const CompileUnit *, DeclContext *, PatchLocation>>
      ForwardDIEReferences;

  FunctionIntervals::Allocator RangeAlloc;

  /// The ranges in that interval map are the PC ranges for
  /// functions in this unit, associated with the PC offset to apply
  /// to the addresses to get the linked address.
  FunctionIntervals Ranges;

  /// The DW_AT_low_pc of each DW_TAG_label.
  SmallDenseMap<uint64_t, uint64_t, 1> Labels;

  /// DW_AT_ranges attributes to patch after we have gathered
  /// all the unit's function addresses.
  /// @{
  std::vector<PatchLocation> RangeAttributes;
  Optional<PatchLocation> UnitRangeAttribute;
  /// @}

  /// Location attributes that need to be transferred from the
  /// original debug_loc section to the liked one. They are stored
  /// along with the PC offset that is to be applied to their
  /// function's address.
  std::vector<std::pair<PatchLocation, int64_t>> LocationAttributes;

  /// Accelerator entries for the unit, both for the pub*
  /// sections and the apple* ones.
  /// @{
  std::vector<AccelInfo> Pubnames;
  std::vector<AccelInfo> Pubtypes;
  std::vector<AccelInfo> Namespaces;
  std::vector<AccelInfo> ObjC;
  /// @}

  /// Cached resolved paths from the line table.
  /// Note, the StringRefs here point in to the intern (uniquing) string pool.
  /// This means that a StringRef returned here doesn't need to then be uniqued
  /// for the purposes of getting a unique address for each string.
  std::vector<StringRef> ResolvedPaths;

  /// Is this unit subject to the ODR rule?
  bool HasODR;

  /// Did a DIE actually contain a valid reloc?
  bool HasInterestingContent;

  /// The DW_AT_language of this unit.
  uint16_t Language = 0;

  /// If this is a Clang module, this holds the module's name.
  std::string ClangModuleName;
};

} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFLINKERCOMPILEUNIT_H
