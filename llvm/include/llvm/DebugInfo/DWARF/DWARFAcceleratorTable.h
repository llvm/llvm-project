//===- DWARFAcceleratorTable.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
#define LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include <cstdint>
#include <utility>

namespace llvm {

class raw_ostream;
class ScopedPrinter;

/// The accelerator tables are designed to allow efficient random access
/// (using a symbol name as a key) into debug info by providing an index of the
/// debug info DIEs. This class implements the common functionality of Apple and
/// DWARF 5 accelerator tables.
/// TODO: Generalize the rest of the AppleAcceleratorTable interface and move it
/// to this class.
class DWARFAcceleratorTable {
protected:
  DWARFDataExtractor AccelSection;
  DataExtractor StringSection;

public:
  DWARFAcceleratorTable(const DWARFDataExtractor &AccelSection,
                        DataExtractor StringSection)
      : AccelSection(AccelSection), StringSection(StringSection) {}
  virtual ~DWARFAcceleratorTable();

  virtual llvm::Error extract() = 0;
  virtual void dump(raw_ostream &OS) const = 0;

  DWARFAcceleratorTable(const DWARFAcceleratorTable &) = delete;
  void operator=(const DWARFAcceleratorTable &) = delete;
};

/// This implements the Apple accelerator table format, a precursor of the
/// DWARF 5 accelerator table format.
class AppleAcceleratorTable : public DWARFAcceleratorTable {
  struct Header {
    uint32_t Magic;
    uint16_t Version;
    uint16_t HashFunction;
    uint32_t NumBuckets;
    uint32_t NumHashes;
    uint32_t HeaderDataLength;
  };

  struct HeaderData {
    using AtomType = uint16_t;
    using Form = dwarf::Form;

    uint32_t DIEOffsetBase;
    SmallVector<std::pair<AtomType, Form>, 3> Atoms;
  };

  struct Header Hdr;
  struct HeaderData HdrData;
  bool IsValid = false;

public:
  /// An iterator for the entries associated with one key. Each entry can have
  /// multiple DWARFFormValues.
  class ValueIterator : public std::iterator<std::input_iterator_tag,
                                            ArrayRef<DWARFFormValue>> {
    const AppleAcceleratorTable *AccelTable = nullptr;
    SmallVector<DWARFFormValue, 3> AtomForms; ///< The decoded data entry.

    unsigned DataOffset = 0; ///< Offset into the section.
    unsigned Data = 0; ///< Current data entry.
    unsigned NumData = 0; ///< Number of data entries.

    /// Advance the iterator.
    void Next();
  public:
    /// Construct a new iterator for the entries at \p DataOffset.
    ValueIterator(const AppleAcceleratorTable &AccelTable, unsigned DataOffset);
    /// End marker.
    ValueIterator() = default;

    const ArrayRef<DWARFFormValue> operator*() const {
      return AtomForms;
    }
    ValueIterator &operator++() { Next(); return *this; }
    ValueIterator operator++(int) {
      ValueIterator I = *this;
      Next();
      return I;
    }
    friend bool operator==(const ValueIterator &A, const ValueIterator &B) {
      return A.NumData == B.NumData && A.DataOffset == B.DataOffset;
    }
    friend bool operator!=(const ValueIterator &A, const ValueIterator &B) {
      return !(A == B);
    }
  };

  AppleAcceleratorTable(const DWARFDataExtractor &AccelSection,
                        DataExtractor StringSection)
      : DWARFAcceleratorTable(AccelSection, StringSection) {}

  llvm::Error extract() override;
  uint32_t getNumBuckets();
  uint32_t getNumHashes();
  uint32_t getSizeHdr();
  uint32_t getHeaderDataLength();
  ArrayRef<std::pair<HeaderData::AtomType, HeaderData::Form>> getAtomsDesc();
  bool validateForms();

  /// Return information related to the DWARF DIE we're looking for when
  /// performing a lookup by name.
  ///
  /// \param HashDataOffset an offset into the hash data table
  /// \returns <DieOffset, DieTag>
  /// DieOffset is the offset into the .debug_info section for the DIE
  /// related to the input hash data offset.
  /// DieTag is the tag of the DIE
  std::pair<uint32_t, dwarf::Tag> readAtoms(uint32_t &HashDataOffset);
  void dump(raw_ostream &OS) const override;

  /// Look up all entries in the accelerator table matching \c Key.
  iterator_range<ValueIterator> equal_range(StringRef Key) const;
};

/// .debug_names section consists of one or more units. Each unit starts with a
/// header, which is followed by a list of compilation units, local and foreign
/// type units.
///
/// These may be followed by an (optional) hash lookup table, which consists of
/// an array of buckets and hashes similar to the apple tables above. The only
/// difference is that the hashes array is 1-based, and consequently an empty
/// bucket is denoted by 0 and not UINT32_MAX.
///
/// Next is the name table, which consists of an array of names and array of
/// entry offsets. This is different from the apple tables, which store names
/// next to the actual entries.
///
/// The structure of the entries is described by an abbreviations table, which
/// comes after the name table. Unlike the apple tables, which have a uniform
/// entry structure described in the header, each .debug_names entry may have
/// different index attributes (DW_IDX_???) attached to it.
///
/// The last segment consists of a list of entries, which is a 0-terminated list
/// referenced by the name table and interpreted with the help of the
/// abbreviation table.
class DWARFDebugNames : public DWARFAcceleratorTable {
public:
  /// Dwarf 5 Name Index header.
  struct Header {
    uint32_t UnitLength;
    uint16_t Version;
    uint16_t Padding;
    uint32_t CompUnitCount;
    uint32_t LocalTypeUnitCount;
    uint32_t ForeignTypeUnitCount;
    uint32_t BucketCount;
    uint32_t NameCount;
    uint32_t AbbrevTableSize;
    uint32_t AugmentationStringSize;
    SmallString<8> AugmentationString;

    Error extract(const DWARFDataExtractor &AS, uint32_t *Offset);
    void dump(ScopedPrinter &W) const;
  };

  /// Index attribute and its encoding.
  struct AttributeEncoding {
    dwarf::Index Index;
    dwarf::Form Form;

    constexpr AttributeEncoding(dwarf::Index Index, dwarf::Form Form)
        : Index(Index), Form(Form) {}

    friend bool operator==(const AttributeEncoding &LHS,
                           const AttributeEncoding &RHS) {
      return LHS.Index == RHS.Index && LHS.Form == RHS.Form;
    }
  };

  /// Abbreviation describing the encoding of Name Index entries.
  struct Abbrev {
    uint32_t Code;  ///< Abbreviation code
    dwarf::Tag Tag; ///< Dwarf Tag of the described entity.
    std::vector<AttributeEncoding> Attributes; ///< List of index attributes.

    Abbrev(uint32_t Code, dwarf::Tag Tag,
           std::vector<AttributeEncoding> Attributes)
        : Code(Code), Tag(Tag), Attributes(std::move(Attributes)) {}

    void dump(ScopedPrinter &W) const;
  };

  /// A single entry in the Name Index.
  struct Entry {
    const Abbrev &Abbr;

    /// Values of the index attributes described by Abbr.
    std::vector<DWARFFormValue> Values;

    Entry(const Abbrev &Abbr);

    void dump(ScopedPrinter &W) const;
  };

private:
  /// Error returned by NameIndex::getEntry to report it has reached the end of
  /// the entry list.
  class SentinelError : public ErrorInfo<SentinelError> {
  public:
    static char ID;

    void log(raw_ostream &OS) const override { OS << "Sentinel"; }
    std::error_code convertToErrorCode() const override;
  };

  /// DenseMapInfo for struct Abbrev.
  struct AbbrevMapInfo {
    static Abbrev getEmptyKey();
    static Abbrev getTombstoneKey();
    static unsigned getHashValue(uint32_t Code) {
      return DenseMapInfo<uint32_t>::getHashValue(Code);
    }
    static unsigned getHashValue(const Abbrev &Abbr) {
      return getHashValue(Abbr.Code);
    }
    static bool isEqual(uint32_t LHS, const Abbrev &RHS) {
      return LHS == RHS.Code;
    }
    static bool isEqual(const Abbrev &LHS, const Abbrev &RHS) {
      return LHS.Code == RHS.Code;
    }
  };

  /// A single entry in the Name Table (Dwarf 5 sect. 6.1.1.4.6) of the Name
  /// Index.
  struct NameTableEntry {
    uint32_t StringOffset; ///< Offset of the name of the described entities.
    uint32_t EntryOffset;  ///< Offset of the first Entry in the list.
  };

public:
  /// Represents a single accelerator table within the Dwarf 5 .debug_names
  /// section.
  class NameIndex {
    DenseSet<Abbrev, AbbrevMapInfo> Abbrevs;
    struct Header Hdr;
    const DWARFDebugNames &Section;

    // Base of the whole unit and of various important tables, as offsets from
    // the start of the section.
    uint32_t Base;
    uint32_t CUsBase;
    uint32_t BucketsBase;
    uint32_t HashesBase;
    uint32_t StringOffsetsBase;
    uint32_t EntryOffsetsBase;
    uint32_t EntriesBase;

    /// Reads offset of compilation unit CU. CU is 0-based.
    uint32_t getCUOffset(uint32_t CU) const;

    /// Reads offset of local type unit TU, TU is 0-based.
    uint32_t getLocalTUOffset(uint32_t TU) const;

    /// Reads signature of foreign type unit TU. TU is 0-based.
    uint64_t getForeignTUOffset(uint32_t TU) const;

    /// Reads an entry in the Bucket Array for the given Bucket. The returned
    /// value is a (1-based) index into the Names, StringOffsets and
    /// EntryOffsets arrays. The input Bucket index is 0-based.
    uint32_t getBucketArrayEntry(uint32_t Bucket) const;

    /// Reads an entry in the Hash Array for the given Index. The input Index
    /// is 1-based.
    uint32_t getHashArrayEntry(uint32_t Index) const;

    /// Reads an entry in the Name Table for the given Index. The Name Table
    /// consists of two arrays -- String Offsets and Entry Offsets. The returned
    /// offsets are relative to the starts of respective sections. Input Index
    /// is 1-based.
    NameTableEntry getNameTableEntry(uint32_t Index) const;

    Expected<Entry> getEntry(uint32_t *Offset) const;

    void dumpCUs(ScopedPrinter &W) const;
    void dumpLocalTUs(ScopedPrinter &W) const;
    void dumpForeignTUs(ScopedPrinter &W) const;
    void dumpAbbreviations(ScopedPrinter &W) const;
    bool dumpEntry(ScopedPrinter &W, uint32_t *Offset) const;
    void dumpName(ScopedPrinter &W, uint32_t Index,
                  Optional<uint32_t> Hash) const;
    void dumpBucket(ScopedPrinter &W, uint32_t Bucket) const;

    Expected<AttributeEncoding> extractAttributeEncoding(uint32_t *Offset);

    Expected<std::vector<AttributeEncoding>>
    extractAttributeEncodings(uint32_t *Offset);

    Expected<Abbrev> extractAbbrev(uint32_t *Offset);

  public:
    NameIndex(const DWARFDebugNames &Section, uint32_t Base)
        : Section(Section), Base(Base) {}

    llvm::Error extract();
    uint32_t getNextUnitOffset() const { return Base + 4 + Hdr.UnitLength; }
    void dump(ScopedPrinter &W) const;
  };

private:
  std::vector<NameIndex> NameIndices;

public:
  DWARFDebugNames(const DWARFDataExtractor &AccelSection,
                  DataExtractor StringSection)
      : DWARFAcceleratorTable(AccelSection, StringSection) {}

  llvm::Error extract() override;
  void dump(raw_ostream &OS) const override;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
