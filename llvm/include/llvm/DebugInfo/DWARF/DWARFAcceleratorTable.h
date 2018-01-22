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

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include <cstdint>
#include <utility>

namespace llvm {

class raw_ostream;

/// This implements the Apple accelerator table format, a precursor of the
/// DWARF 5 accelerator table format.
/// TODO: Factor out a common base class for both formats.
class AppleAcceleratorTable {
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
  DWARFDataExtractor AccelSection;
  DataExtractor StringSection;
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
      : AccelSection(AccelSection), StringSection(StringSection) {}

  llvm::Error extract();
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
  void dump(raw_ostream &OS) const;

  /// Look up all entries in the accelerator table matching \c Key.
  iterator_range<ValueIterator> equal_range(StringRef Key) const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
