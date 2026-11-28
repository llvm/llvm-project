//===- CVPacks.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The records and debug subsections that a COFF object packed against a
// CodeView pack carries. Reading and writing packs, and packing and restoring
// objects, is in the DebugInfoCVPack library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CVPACKS_H
#define LLVM_DEBUGINFO_CODEVIEW_CVPACKS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

inline constexpr uint16_t FormatVersion = 1;
inline constexpr uint32_t MaxMapBodySize = 0xfef4;

/// The content id of a file: its BLAKE3-256 hash. A pack is named by its
/// content id, and a packed object records that of its original bytes. A
/// struct rather than a std::array, so that its YAML form does not apply to
/// every array of 32 bytes.
struct PackId {
  std::array<uint8_t, 32> Bytes{};

  constexpr PackId() = default;
  /*implicit*/ constexpr PackId(const std::array<uint8_t, 32> &Bytes)
      : Bytes(Bytes) {}

  static constexpr size_t size() { return 32; }
  const uint8_t *data() const { return Bytes.data(); }
  uint8_t *data() { return Bytes.data(); }
  auto begin() const { return Bytes.begin(); }
  auto end() const { return Bytes.end(); }
  auto begin() { return Bytes.begin(); }
  auto end() { return Bytes.end(); }
  uint8_t operator[](size_t I) const { return Bytes[I]; }
  uint8_t &operator[](size_t I) { return Bytes[I]; }

  friend bool operator==(const PackId &A, const PackId &B) {
    return A.Bytes == B.Bytes;
  }
  friend bool operator!=(const PackId &A, const PackId &B) {
    return A.Bytes != B.Bytes;
  }
  friend bool operator<(const PackId &A, const PackId &B) {
    return A.Bytes < B.Bytes;
  }
};

/// Whether a .debug$T section, signature included, is the type stream of a
/// packed object: COFF::DEBUG_CVPACK_SECTION_MAGIC, then LF_CVPACK_LIST.
LLVM_ABI bool isPackedTypeStream(ArrayRef<uint8_t> Section);

/// Whether the signature of a CodeView debug section is that of the type
/// stream of a packed object. Readers accept it in place of
/// COFF::DEBUG_SECTION_MAGIC: the records that follow are the same.
inline bool isPackedTypeStreamMagic(StringRef SectionName, uint32_t Magic) {
  return SectionName == ".debug$T" && Magic == COFF::DEBUG_CVPACK_SECTION_MAGIC;
}

/// A packed object carries this directive in its .drectve, so that a linker
/// that does not know CodeView packs fails with an error that names the
/// symbol, instead of misreading the object's debug info. LLD ignores it.
inline constexpr StringLiteral PackedObjectSentinel =
    "__cvpack_object_needs_lld_link_or_cvpack_materialize";
inline constexpr StringLiteral PackedObjectDirective =
    " /INCLUDE:__cvpack_object_needs_lld_link_or_cvpack_materialize";

/// Call Callback with the offset of each 4-byte type index field of a type
/// record, or of a symbol record if IsSymbol, in record order, and with the
/// kind of index it holds. Offsets count from the record prefix. Fails for a
/// symbol that type-index discovery does not know, or a field past the end of
/// the record.
LLVM_ABI Error
forEachTypeIndex(ArrayRef<uint8_t> Record, bool IsSymbol,
                 function_ref<Error(uint32_t Offset, TiRefKind Kind)> Callback);

/// Debug subsection kinds used to move the debug info of COMDAT functions to
/// a pack. They carry the ignore bit, so that readers that do not know them
/// skip them.
inline constexpr uint32_t PackFilesSubsectionKind = 0x80000f10;
inline constexpr uint32_t PackFunctionsSubsectionKind = 0x80000f11;
inline constexpr uint32_t FunctionRefSubsectionKind = 0x80000f12;
/// The size of a COMDAT .debug$S section whose content moved to a pack: the
/// magic, a subsection header and a FunctionRef.
inline constexpr size_t FunctionRefSectionSize = 24;

/// A source file referenced by the function debug info of a pack, identified
/// as in an object's file checksums subsection.
struct PackFileEntry {
  StringRef Name;
  uint8_t ChecksumKind = 0;
  ArrayRef<uint8_t> Checksum;
};

/// The content of a COMDAT .debug$S section whose debug info moved to a pack:
/// function Ordinal of the object's pack. Slot is reserved for objects that
/// refer to several packs, and must be 0 in format version 1.
struct FunctionRef {
  uint32_t Slot = 0;
  uint32_t Ordinal = 0;
  uint32_t OriginalSize = 0;
};

LLVM_ABI std::optional<FunctionRef>
decodeFunctionRef(ArrayRef<uint8_t> SectionData);
LLVM_ABI std::vector<uint8_t> encodeFunctionRef(const FunctionRef &Ref);

/// The bodies of the subsections above, without their subsection header. A
/// files or functions body is a count followed by 4-byte aligned entries.
LLVM_ABI Expected<FunctionRef> decodeFunctionRefBody(ArrayRef<uint8_t> Body);
LLVM_ABI std::vector<uint8_t> encodeFunctionRefBody(const FunctionRef &Ref);
LLVM_ABI Expected<std::vector<PackFileEntry>>
decodePackFiles(ArrayRef<uint8_t> Body);
LLVM_ABI Expected<std::vector<uint8_t>>
encodePackFiles(ArrayRef<PackFileEntry> Files);
LLVM_ABI Expected<std::vector<ArrayRef<uint8_t>>>
decodePackFunctions(ArrayRef<uint8_t> Body);
/// Append the body of a functions subsection to Out.
LLVM_ABI void encodePackFunctions(ArrayRef<ArrayRef<uint8_t>> Functions,
                                  std::vector<uint8_t> &Out);

/// One range of a type or symbol map. Slot is reserved for objects that refer
/// to several packs, and must be 0 in format version 1.
struct RangeMapEntry {
  uint64_t Skip = 0;
  uint32_t Slot = 0;
  uint32_t Count = 0;
  int64_t FirstDelta = 0;
};

struct RangeMap {
  std::vector<RangeMapEntry> Entries;
};

struct RangeMapSegment {
  bool IsPackReference = false;
  uint64_t Count = 0;
  uint64_t FirstIndex = 0;
};

enum class OriginEditKind : uint16_t {
  TypeStream = 1,
  SymbolStream = 2,
  DebugH = 3,
  /// PackedObjectDirective inserted at OriginalOffset of an existing
  /// .drectve section.
  Directive = 4,
  /// A .drectve section holding only PackedObjectDirective, added as the last
  /// section of an object that had none.
  AddedDirectiveSection = 5,
};

struct RestorationFragment {
  uint32_t Offset = 0;
  std::vector<uint8_t> Bytes;
};

struct OriginEdit {
  uint32_t SectionIndex = 0;
  OriginEditKind Kind = OriginEditKind::TypeStream;
  uint64_t OriginalOffset = 0;
  uint32_t OriginalLength = 0;
  uint32_t PackedLength = 0;
  std::vector<RestorationFragment> RestorationFragments;
};

struct Origin {
  uint64_t OriginalFileSize = 0;
  PackId OriginalFileHash{};
  std::vector<uint8_t> DebugHHeader;
  std::vector<OriginEdit> Edits;
};

/// LF_CVPACK_LIST, the first type record of a packed object: the pack it
/// refers to. The record holds a count and the ids, and format version 1
/// requires exactly one id.
LLVM_ABI Expected<PackId> decodePackList(CVType Record);
LLVM_ABI Expected<Origin> decodeOrigin(CVType Record);
LLVM_ABI Expected<RangeMap> decodeTypeMap(ArrayRef<CVType> Records);
LLVM_ABI Expected<RangeMap> decodeSymbolMap(ArrayRef<CVSymbol> Records);

LLVM_ABI Expected<std::vector<uint8_t>> encodePackList(const PackId &Id);
LLVM_ABI Expected<std::vector<uint8_t>> encodeOrigin(const Origin &Data);
LLVM_ABI Expected<std::vector<std::vector<uint8_t>>>
encodeTypeMap(const RangeMap &Map);
LLVM_ABI Expected<std::vector<std::vector<uint8_t>>>
encodeSymbolMap(const RangeMap &Map);

/// Expand a map into inline and pack-reference segments. PackEnd is the
/// exclusive end index of the pack (0x1000 plus the type count, or the symbol
/// count). InlineCount is the total number of inline records in the stream.
LLVM_ABI Expected<std::vector<RangeMapSegment>>
expandRangeMap(const RangeMap &Map, uint64_t PackEnd, uint64_t InlineCount,
               uint64_t FirstPackIndex);

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVPACKS_H
