//===- OutputSections.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H
#define LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H

#include "ArrayList.h"
#include "StringEntryToDwarfStringPoolEntryMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>

namespace llvm {
namespace dwarflinker_parallel {

/// List of tracked debug tables.
enum class DebugSectionKind : uint8_t {
  DebugInfo = 0,
  DebugLine,
  DebugFrame,
  DebugRange,
  DebugRngLists,
  DebugLoc,
  DebugLocLists,
  DebugARanges,
  DebugAbbrev,
  DebugMacinfo,
  DebugMacro,
  DebugAddr,
  DebugStr,
  DebugLineStr,
  DebugStrOffsets,
  DebugPubNames,
  DebugPubTypes,
  DebugNames,
  AppleNames,
  AppleNamespaces,
  AppleObjC,
  AppleTypes,
  NumberOfEnumEntries // must be last
};
constexpr static size_t SectionKindsNum =
    static_cast<size_t>(DebugSectionKind::NumberOfEnumEntries);

/// Recognise the table name and match it with the DebugSectionKind.
std::optional<DebugSectionKind> parseDebugTableName(StringRef Name);

/// Return the name of the section.
const StringLiteral &getSectionName(DebugSectionKind SectionKind);

/// There are fields(sizes, offsets) which should be updated after
/// sections are generated. To remember offsets and related data
/// the descendants of SectionPatch structure should be used.

struct SectionPatch {
  uint64_t PatchOffset = 0;
};

/// This structure is used to update strings offsets into .debug_str.
struct DebugStrPatch : SectionPatch {
  const StringEntry *String = nullptr;
};

/// This structure is used to update strings offsets into .debug_line_str.
struct DebugLineStrPatch : SectionPatch {
  const StringEntry *String = nullptr;
};

/// This structure is used to update range list offset into
/// .debug_ranges/.debug_rnglists.
struct DebugRangePatch : SectionPatch {
  /// Indicates patch which points to immediate compile unit's attribute.
  bool IsCompileUnitRanges = false;
};

/// This structure is used to update location list offset into
/// .debug_loc/.debug_loclists.
struct DebugLocPatch : SectionPatch {
  int64_t AddrAdjustmentValue = 0;
};

/// This structure is used to update offset with start of another section.
struct SectionDescriptor;
struct DebugOffsetPatch : SectionPatch {
  DebugOffsetPatch(uint64_t PatchOffset, SectionDescriptor *SectionPtr,
                   bool AddLocalValue = false)
      : SectionPatch({PatchOffset}), SectionPtr(SectionPtr, AddLocalValue) {}

  PointerIntPair<SectionDescriptor *, 1> SectionPtr;
};

/// This structure is used to update reference to the DIE.
struct DebugDieRefPatch : SectionPatch {
  DebugDieRefPatch(uint64_t PatchOffset, CompileUnit *SrcCU, CompileUnit *RefCU,
                   uint32_t RefIdx);

  PointerIntPair<CompileUnit *, 1> RefCU;
  uint64_t RefDieIdxOrClonedOffset;
};

/// This structure is used to update reference to the DIE of ULEB128 form.
struct DebugULEB128DieRefPatch : SectionPatch {
  DebugULEB128DieRefPatch(uint64_t PatchOffset, CompileUnit *SrcCU,
                          CompileUnit *RefCU, uint32_t RefIdx);

  PointerIntPair<CompileUnit *, 1> RefCU;
  uint64_t RefDieIdxOrClonedOffset;
};

/// Type for section data.
using OutSectionDataTy = SmallString<0>;

/// Type for list of pointers to patches offsets.
using OffsetsPtrVector = SmallVector<uint64_t *>;

class OutputSections;

/// This structure is used to keep data of the concrete section.
/// Like data bits, list of patches, format.
struct SectionDescriptor {
  friend OutputSections;

  SectionDescriptor(DebugSectionKind SectionKind, LinkingGlobalData &GlobalData,
                    dwarf::FormParams Format, llvm::endianness Endianess)
      : OS(Contents), GlobalData(GlobalData), SectionKind(SectionKind),
        Format(Format), Endianess(Endianess) {
    ListDebugStrPatch.setAllocator(&GlobalData.getAllocator());
    ListDebugLineStrPatch.setAllocator(&GlobalData.getAllocator());
    ListDebugRangePatch.setAllocator(&GlobalData.getAllocator());
    ListDebugLocPatch.setAllocator(&GlobalData.getAllocator());
    ListDebugDieRefPatch.setAllocator(&GlobalData.getAllocator());
    ListDebugULEB128DieRefPatch.setAllocator(&GlobalData.getAllocator());
    ListDebugOffsetPatch.setAllocator(&GlobalData.getAllocator());
  }

  /// Erase whole section contents(data bits, list of patches).
  void clearAllSectionData();

  /// Erase only section output data bits.
  void clearSectionContent();

  /// When objects(f.e. compile units) are glued into the single file,
  /// the debug sections corresponding to the concrete object are assigned
  /// with offsets inside the whole file. This field keeps offset
  /// to the debug section, corresponding to this object.
  uint64_t StartOffset = 0;

  /// Stream which stores data to the Contents.
  raw_svector_ostream OS;

  /// Section patches.
#define ADD_PATCHES_LIST(T)                                                    \
  T &notePatch(const T &Patch) { return List##T.add(Patch); }                  \
  ArrayList<T> List##T;

  ADD_PATCHES_LIST(DebugStrPatch)
  ADD_PATCHES_LIST(DebugLineStrPatch)
  ADD_PATCHES_LIST(DebugRangePatch)
  ADD_PATCHES_LIST(DebugLocPatch)
  ADD_PATCHES_LIST(DebugDieRefPatch)
  ADD_PATCHES_LIST(DebugULEB128DieRefPatch)
  ADD_PATCHES_LIST(DebugOffsetPatch)

  /// Offsets to some fields are not known at the moment of noting patch.
  /// In that case we remember pointers to patch offset to update them later.
  template <typename T>
  void notePatchWithOffsetUpdate(const T &Patch,
                                 OffsetsPtrVector &PatchesOffsetsList) {
    PatchesOffsetsList.emplace_back(&notePatch(Patch).PatchOffset);
  }

  /// Some sections are emitted using AsmPrinter. In that case "Contents"
  /// member of SectionDescriptor contains elf file. This method searches
  /// for section data inside elf file and remember offset to it.
  void setSizesForSectionCreatedByAsmPrinter();

  /// Returns section content.
  StringRef getContents() {
    if (SectionOffsetInsideAsmPrinterOutputStart == 0)
      return StringRef(Contents.data(), Contents.size());

    return Contents.slice(SectionOffsetInsideAsmPrinterOutputStart,
                          SectionOffsetInsideAsmPrinterOutputEnd);
  }

  /// Emit unit length into the current section contents.
  void emitUnitLength(uint64_t Length) {
    maybeEmitDwarf64Mark();
    emitIntVal(Length, getFormParams().getDwarfOffsetByteSize());
  }

  /// Emit DWARF64 mark into the current section contents.
  void maybeEmitDwarf64Mark() {
    if (getFormParams().Format != dwarf::DWARF64)
      return;
    emitIntVal(dwarf::DW_LENGTH_DWARF64, 4);
  }

  /// Emit specified offset value into the current section contents.
  void emitOffset(uint64_t Val) {
    emitIntVal(Val, getFormParams().getDwarfOffsetByteSize());
  }

  /// Emit specified integer value into the current section contents.
  void emitIntVal(uint64_t Val, unsigned Size);

  /// Emit specified string value into the current section contents.
  void emitString(dwarf::Form StringForm, const char *StringVal);

  /// Emit specified inplace string value into the current section contents.
  void emitInplaceString(StringRef String) {
    OS << GlobalData.translateString(String);
    emitIntVal(0, 1);
  }

  /// Emit string placeholder into the current section contents.
  void emitStringPlaceholder() {
    // emit bad offset which should be updated later.
    emitOffset(0xBADDEF);
  }

  /// Write specified \p Value of \p AttrForm to the \p PatchOffset.
  void apply(uint64_t PatchOffset, dwarf::Form AttrForm, uint64_t Val);

  /// Returns section kind.
  DebugSectionKind getKind() { return SectionKind; }

  /// Returns section name.
  const StringLiteral &getName() const { return getSectionName(SectionKind); }

  /// Returns endianess used by section.
  llvm::endianness getEndianess() const { return Endianess; }

  /// Returns FormParams used by section.
  dwarf::FormParams getFormParams() const { return Format; }

  /// Returns integer value of \p Size located by specified \p PatchOffset.
  uint64_t getIntVal(uint64_t PatchOffset, unsigned Size);

protected:
  /// Writes integer value \p Val of \p Size by specified \p PatchOffset.
  void applyIntVal(uint64_t PatchOffset, uint64_t Val, unsigned Size);

  /// Writes integer value \p Val of ULEB128 format by specified \p PatchOffset.
  void applyULEB128(uint64_t PatchOffset, uint64_t Val);

  /// Writes integer value \p Val of SLEB128 format by specified \p PatchOffset.
  void applySLEB128(uint64_t PatchOffset, uint64_t Val);

  /// Sets output format.
  void setOutputFormat(dwarf::FormParams Format, llvm::endianness Endianess) {
    this->Format = Format;
    this->Endianess = Endianess;
  }

  LinkingGlobalData &GlobalData;

  /// The section kind.
  DebugSectionKind SectionKind = DebugSectionKind::NumberOfEnumEntries;

  /// Section data bits.
  OutSectionDataTy Contents;

  /// Some sections are generated using AsmPrinter. The real section data
  /// located inside elf file in that case. Following fields points to the
  /// real section content inside elf file.
  size_t SectionOffsetInsideAsmPrinterOutputStart = 0;
  size_t SectionOffsetInsideAsmPrinterOutputEnd = 0;

  /// Output format.
  dwarf::FormParams Format = {4, 4, dwarf::DWARF32};
  llvm::endianness Endianess = llvm::endianness::little;
};

/// This class keeps contents and offsets to the debug sections. Any objects
/// which is supposed to be emitted into the debug sections should use this
/// class to track debug sections offsets and keep sections data.
class OutputSections {
public:
  OutputSections(LinkingGlobalData &GlobalData) : GlobalData(GlobalData) {}

  /// Sets output format for all keeping sections.
  void setOutputFormat(dwarf::FormParams Format, llvm::endianness Endianness) {
    this->Format = Format;
    this->Endianness = Endianness;
  }

  /// Returns descriptor for the specified section of \p SectionKind.
  /// The descriptor should already be created. The llvm_unreachable
  /// would be raised if it is not.
  const SectionDescriptor &
  getSectionDescriptor(DebugSectionKind SectionKind) const {
    SectionsSetTy::const_iterator It = SectionDescriptors.find(SectionKind);

    if (It == SectionDescriptors.end())
      llvm_unreachable(
          formatv("Section {0} does not exist", getSectionName(SectionKind))
              .str()
              .c_str());

    return It->second;
  }

  /// Returns descriptor for the specified section of \p SectionKind.
  /// The descriptor should already be created. The llvm_unreachable
  /// would be raised if it is not.
  SectionDescriptor &getSectionDescriptor(DebugSectionKind SectionKind) {
    SectionsSetTy::iterator It = SectionDescriptors.find(SectionKind);

    if (It == SectionDescriptors.end())
      llvm_unreachable(
          formatv("Section {0} does not exist", getSectionName(SectionKind))
              .str()
              .c_str());

    return It->second;
  }

  /// Returns descriptor for the specified section of \p SectionKind.
  /// Returns std::nullopt if section descriptor is not created yet.
  std::optional<const SectionDescriptor *>
  tryGetSectionDescriptor(DebugSectionKind SectionKind) const {
    SectionsSetTy::const_iterator It = SectionDescriptors.find(SectionKind);

    if (It == SectionDescriptors.end())
      return std::nullopt;

    return &It->second;
  }

  /// Returns descriptor for the specified section of \p SectionKind.
  /// Returns std::nullopt if section descriptor is not created yet.
  std::optional<SectionDescriptor *>
  tryGetSectionDescriptor(DebugSectionKind SectionKind) {
    SectionsSetTy::iterator It = SectionDescriptors.find(SectionKind);

    if (It == SectionDescriptors.end())
      return std::nullopt;

    return &It->second;
  }

  /// Returns descriptor for the specified section of \p SectionKind.
  /// If descriptor does not exist then creates it.
  SectionDescriptor &
  getOrCreateSectionDescriptor(DebugSectionKind SectionKind) {
    return SectionDescriptors
        .try_emplace(SectionKind, SectionKind, GlobalData, Format, Endianness)
        .first->second;
  }

  /// Erases data of all sections.
  void eraseSections() {
    for (auto &Section : SectionDescriptors)
      Section.second.clearAllSectionData();
  }

  /// Enumerate all sections and call \p Handler for each.
  void forEach(function_ref<void(SectionDescriptor &)> Handler) {
    for (auto &Section : SectionDescriptors)
      Handler(Section.second);
  }

  /// Enumerate all sections, for each section set current offset
  /// (kept by \p SectionSizesAccumulator), update current offset with section
  /// length.
  void assignSectionsOffsetAndAccumulateSize(
      std::array<uint64_t, SectionKindsNum> &SectionSizesAccumulator) {
    for (auto &Section : SectionDescriptors) {
      Section.second.StartOffset = SectionSizesAccumulator[static_cast<uint8_t>(
          Section.second.getKind())];
      SectionSizesAccumulator[static_cast<uint8_t>(Section.second.getKind())] +=
          Section.second.getContents().size();
    }
  }

  /// Enumerate all sections, for each section apply all section patches.
  void applyPatches(SectionDescriptor &Section,
                    StringEntryToDwarfStringPoolEntryMap &DebugStrStrings,
                    StringEntryToDwarfStringPoolEntryMap &DebugLineStrStrings);

  /// Endiannes for the sections.
  llvm::endianness getEndianness() const { return Endianness; }

  /// Return DWARF version.
  uint16_t getVersion() const { return Format.Version; }

  /// Return size of header of debug_info table.
  uint16_t getDebugInfoHeaderSize() const {
    return Format.Version >= 5 ? 12 : 11;
  }

  /// Return size of header of debug_ table.
  uint16_t getDebugAddrHeaderSize() const {
    assert(Format.Version >= 5);
    return Format.Format == dwarf::DwarfFormat::DWARF32 ? 8 : 16;
  }

  /// Return size of header of debug_str_offsets table.
  uint16_t getDebugStrOffsetsHeaderSize() const {
    assert(Format.Version >= 5);
    return Format.Format == dwarf::DwarfFormat::DWARF32 ? 8 : 16;
  }

  /// Return size of address.
  const dwarf::FormParams &getFormParams() const { return Format; }

protected:
  LinkingGlobalData &GlobalData;

  /// Format for sections.
  dwarf::FormParams Format = {4, 4, dwarf::DWARF32};

  /// Endiannes for sections.
  llvm::endianness Endianness = llvm::endianness::native;

  /// All keeping sections.
  using SectionsSetTy = std::map<DebugSectionKind, SectionDescriptor>;
  SectionsSetTy SectionDescriptors;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H
