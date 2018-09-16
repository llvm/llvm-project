//===- DWARFListTable.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFLISTTABLE_H
#define LLVM_DEBUGINFO_DWARFLISTTABLE_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <vector>

namespace llvm {

class DWARFContext;

/// A base class for DWARF list entries, such as range or location list
/// entries.
struct DWARFListEntryBase {
  /// The offset at which the entry is located in the section.
  uint32_t Offset;
  /// The DWARF encoding (DW_RLE_* or DW_LLE_*).
  uint8_t EntryKind;
  /// The index of the section this entry belongs to.
  uint64_t SectionIndex;
};

/// A base class for lists of entries that are extracted from a particular
/// section, such as range lists or location lists.
template <typename ListEntryType> class DWARFListType {
public:
  using EntryType = ListEntryType;
  using ListEntries = std::vector<EntryType>;

protected:
  ListEntries Entries;

public:
  const ListEntries &getEntries() const { return Entries; }
  bool empty() const {
    return Entries.empty() || Entries.begin()->isEndOfList();
  }
  void clear() { Entries.clear(); }
  uint32_t getOffset() const {
    if (Entries.empty())
      return 0;
    return Entries.begin()->Offset;
  }

  /// Extract a list. The caller must pass the correct DWARF version.
  /// The end-of-list entry is retained as the last element of the vector of
  /// entries.
  Error extract(DWARFDataExtractor Data, uint32_t HeaderOffset, uint32_t End,
                uint16_t Version, uint32_t *OffsetPtr, StringRef SectionName,
                StringRef ListStringName, bool isDWO = false);
  void dump(raw_ostream &OS, DWARFContext *C, uint8_t AddressSize,
            uint64_t BaseAddress, unsigned Indent, uint16_t Version,
            size_t MaxEncodingStringLength = 0,
            DIDumpOptions DumpOpts = {}) const;
};

/// A class representing the header of a list table such as the range list
/// table in the .debug_rnglists section.
class DWARFListTableHeader {
  struct Header {
    /// The total length of the entries for this table, not including the length
    /// field itself.
    uint32_t Length = 0;
    /// The DWARF version number.
    uint16_t Version;
    /// The size in bytes of an address on the target architecture. For
    /// segmented addressing, this is the size of the offset portion of the
    /// address.
    uint8_t AddrSize;
    /// The size in bytes of a segment selector on the target architecture.
    /// If the target system uses a flat address space, this value is 0.
    uint8_t SegSize = 0;
    /// The number of offsets that follow the header before the range lists.
    uint32_t OffsetEntryCount = 0;
  };

  Header HeaderData;
  /// The offset table, which contains offsets to the individual list entries.
  /// It is used by forms such as DW_FORM_rnglistx.
  /// FIXME: Generate the table and use the appropriate forms.
  std::vector<uint32_t> Offsets;
  /// The table's format, either DWARF32 or DWARF64.
  dwarf::DwarfFormat Format = dwarf::DwarfFormat::DWARF32;
  /// The offset at which the header (and hence the table) is located within
  /// its section.
  uint32_t HeaderOffset = 0;
  /// The name of the section the list is located in.
  StringRef SectionName;
  /// A characterization of the list for dumping purposes, e.g. "range" or
  /// "location".
  StringRef ListTypeString;

public:
  DWARFListTableHeader(StringRef SectionName, StringRef ListTypeString)
      : SectionName(SectionName), ListTypeString(ListTypeString) {}

  void clear() {
    HeaderData = {};
    Offsets.clear();
  }
  uint32_t getHeaderOffset() const { return HeaderOffset; }

  uint8_t getAddrSize() const { return HeaderData.AddrSize; }
  void setAddrSize(uint8_t AddrSize) { HeaderData.AddrSize = AddrSize; }

  uint32_t getLength() const { return HeaderData.Length; }
  void setLength(uint32_t Length) { HeaderData.Length = Length; }

  uint16_t getVersion() const { return HeaderData.Version; }
  void setVersion(uint16_t Version) { HeaderData.Version = Version; }

  uint8_t getSegSize() const { return HeaderData.SegSize; }
  uint32_t getOffsetEntryCount() const { return HeaderData.OffsetEntryCount; }

  StringRef getSectionName() const { return SectionName; }
  StringRef getListTypeString() const { return ListTypeString; }
  dwarf::DwarfFormat getFormat() const { return Format; }

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts = {}) const;
  Optional<uint32_t> getOffsetEntry(uint32_t Index) const {
    if (Index < Offsets.size())
      return Offsets[Index];
    return None;
  }

  /// Extract the table header and the array of offsets.
  Error extract(DWARFDataExtractor Data, uint32_t *OffsetPtr);

  /// Returns the length of the table, including the length field, or 0 if the
  /// length has not been determined (e.g. because the table has not yet been
  /// parsed, or there was a problem in parsing). In fake tables, such as for
  /// DWARF v4 and earlier, there is no header, so the length simply reflects
  /// the size of the section.
  uint32_t getTableLength() const;
};

/// A class representing a table of lists as specified in the DWARF v5
/// standard for location lists and range lists. The table consists of a header
/// followed by an array of offsets into a DWARF section, followed by zero or
/// more list entries. The list entries are kept in a map where the keys are
/// the lists' section offsets.
template <typename DWARFListType> class DWARFListTableBase {
  DWARFListTableHeader Header;
  /// A mapping between file offsets and lists. It is used to find a particular
  /// list based on an offset (obtained from DW_AT_ranges, for example).
  std::map<uint32_t, DWARFListType> ListMap;
  DWARFContext *C;
  /// True if this list is located in a split-DWARF (dwo or dwp) file.
  bool isDWO;
  /// This string is displayed as a heading before the list is dumped
  /// (e.g. "ranges:").
  StringRef HeaderString;
  /// A function returning the encoding string for a given list entry encoding,
  /// e.g. "DW_RLE_start_end".
  std::function<StringRef(unsigned)> EncodingString;

protected:
  DWARFListTableBase(DWARFContext *C, StringRef SectionName, bool isDWO,
                     StringRef HeaderString, StringRef ListTypeString,
                     std::function<StringRef(unsigned)> EncodingString)
      : Header(SectionName, ListTypeString), C(C), isDWO(isDWO),
        HeaderString(HeaderString), EncodingString(EncodingString) {}

public:
  void clear() {
    Header.clear();
    ListMap.clear();
  }
  /// Extract the table header and the array of offsets.
  Error extractHeaderAndOffsets(DWARFDataExtractor Data, uint32_t *OffsetPtr) {
    return Header.extract(Data, OffsetPtr);
  }

  /// Initialize the table header to explicit values. This is used for DWARF v4
  /// and earlier since there is no header that can be extracted from a section.
  void setHeaderData(uint32_t Length, uint16_t Version, uint8_t AddrSize) {
    assert(Header.getSegSize() == 0 &&
           "Unexpected segsize in list table header.");
    assert(Header.getOffsetEntryCount() == 0 &&
           "Unexpected offset entry count in list table header.");
    Header.setLength(Length);
    Header.setVersion(Version);
    Header.setAddrSize(AddrSize);
  }

  /// Extract an entire table, including all list entries.
  Error extract(DWARFDataExtractor Data, uint16_t Version, uint32_t *OffsetPtr);
  /// Look up a list based on a given offset. Extract it and enter it into the
  /// list map if necessary.
  Expected<DWARFListType> findList(DWARFDataExtractor Data, uint32_t Offset);

  uint32_t getHeaderOffset() const { return Header.getHeaderOffset(); }
  uint8_t getAddrSize() const { return Header.getAddrSize(); }
  StringRef getListTypeString() const { return Header.getListTypeString(); }

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts = {}) const;

  /// Return the contents of the offset entry designated by a given index.
  Optional<uint32_t> getOffsetEntry(uint32_t Index) const {
    return Header.getOffsetEntry(Index);
  }
  /// Return the size of the table header including the length but not including
  /// the offsets. This is dependent on the table format, which is unambiguously
  /// derived from parsing the table.
  uint8_t getHeaderSize() const {
    switch (Header.getFormat()) {
    case dwarf::DwarfFormat::DWARF32:
      return 12;
    case dwarf::DwarfFormat::DWARF64:
      return 20;
    }
    llvm_unreachable("Invalid DWARF format (expected DWARF32 or DWARF64");
  }

  uint16_t getVersion() const { return Header.getVersion(); }
  uint32_t getLength() const { return Header.getTableLength(); }
};

template <typename DWARFListType>
Error DWARFListTableBase<DWARFListType>::extract(DWARFDataExtractor Data,
                                                 uint16_t Version,
                                                 uint32_t *OffsetPtr) {
  assert(Version > 0 && "DWARF version required and not given.");
  clear();
  // For DWARF v4 and earlier, we cannot extract a table header, so we
  // initialize it explicitly.
  if (Version < 5)
    setHeaderData(Data.size(), Version, Data.getAddressSize());
  else if (Error E = extractHeaderAndOffsets(Data, OffsetPtr))
    return E;

  Data.setAddressSize(Header.getAddrSize());
  uint32_t End = getHeaderOffset() + getLength();
  // Extract all lists.
  while (*OffsetPtr < End) {
    DWARFListType CurrentList;
    uint32_t Off = *OffsetPtr;
    if (Error E = CurrentList.extract(
            Data, getHeaderOffset(), End, Header.getVersion(), OffsetPtr,
            Header.getSectionName(), Header.getListTypeString(), isDWO)) {
      *OffsetPtr = End;
      return E;
    }
    ListMap[Off] = CurrentList;
  }

  assert(*OffsetPtr == End &&
         "mismatch between expected length of table and length "
         "of extracted data");
  return Error::success();
}

template <typename ListEntryType>
Error DWARFListType<ListEntryType>::extract(
    DWARFDataExtractor Data, uint32_t HeaderOffset, uint32_t End,
    uint16_t Version, uint32_t *OffsetPtr, StringRef SectionName,
    StringRef ListTypeString, bool isDWO) {
  if (*OffsetPtr < HeaderOffset || *OffsetPtr >= End)
    return createStringError(errc::invalid_argument,
                       "invalid %s list offset 0x%" PRIx32,
                       ListTypeString.data(), *OffsetPtr);
  Entries.clear();
  while (*OffsetPtr < End) {
    ListEntryType Entry;
    if (Error E =
            Entry.extract(Data, End, Version, SectionName, OffsetPtr, isDWO))
      return E;
    Entries.push_back(Entry);
    if (Entry.isEndOfList())
      return Error::success();
  }
  return createStringError(errc::illegal_byte_sequence,
                     "no end of list marker detected at end of %s table "
                     "starting at offset 0x%" PRIx32,
                     SectionName.data(), HeaderOffset);
}

template <typename ListEntryType>
void DWARFListType<ListEntryType>::dump(raw_ostream &OS, DWARFContext *C,
                                        uint8_t AddressSize,
                                        uint64_t BaseAddress, unsigned Indent,
                                        uint16_t Version,
                                        size_t MaxEncodingStringLength,
                                        DIDumpOptions DumpOpts) const {
  uint64_t CurrentBase = BaseAddress;
  for (const auto &Entry : Entries)
    Entry.dump(OS, C, AddressSize, CurrentBase, Indent, Version,
               MaxEncodingStringLength, DumpOpts);
}

template <typename DWARFListType>
void DWARFListTableBase<DWARFListType>::dump(raw_ostream &OS,
                                             DIDumpOptions DumpOpts) const {
  size_t MaxEncodingStringLength = 0;
  // Don't dump the fake table header we create for DWARF v4 and earlier.
  if (Header.getVersion() > 4) {
    Header.dump(OS, DumpOpts);
    OS << HeaderString << '\n';
    // Determine the length of the longest encoding string we have in the table,
    // so we can align the output properly. We only need this in verbose mode.
    if (DumpOpts.Verbose)
      for (const auto &List : ListMap)
        for (const auto &Entry : List.second.getEntries())
          MaxEncodingStringLength = std::max(
              MaxEncodingStringLength, EncodingString(Entry.EntryKind).size());
  }

  uint64_t CurrentBase = 0;
  for (const auto &List : ListMap) {
    List.second.dump(OS, C, getAddrSize(), CurrentBase, 0, Header.getVersion(),
                     MaxEncodingStringLength, DumpOpts);
  }
}

template <typename DWARFListType>
Expected<DWARFListType>
DWARFListTableBase<DWARFListType>::findList(DWARFDataExtractor Data,
                                            uint32_t Offset) {
  auto Entry = ListMap.find(Offset);
  if (Entry != ListMap.end())
    return Entry->second;

  // Extract the list from the section and enter it into the list map.
  DWARFListType List;
  uint32_t End = getHeaderOffset() + getLength();
  uint32_t StartingOffset = Offset;
  if (Error E = List.extract(Data, getHeaderOffset(), End, Header.getVersion(),
                             &Offset, Header.getSectionName(),
                             Header.getListTypeString(), isDWO))
    return std::move(E);
  ListMap[StartingOffset] = List;
  return List;
}

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFLISTTABLE_H
