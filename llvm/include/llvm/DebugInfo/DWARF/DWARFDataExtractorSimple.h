//===- DWARFDataExtractorSimple.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Non relocating, low-level dwarf-data extractor. Suitable for use from
// libraries that cannot have build-time dependencies on relocation providers.

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTORSIMPLE_H
#define LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTORSIMPLE_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

/// A DataExtractor suitable use for parsing dwarf from memory with minimal
/// dwarf context--no sections and no objects.  getRelocated* functions
/// return raw values.

class DWARFDataExtractorSimple : public DataExtractor {

public:
  DWARFDataExtractorSimple(StringRef Data, bool IsLittleEndian,
                           uint8_t AddressSize)
      : DataExtractor(Data, IsLittleEndian, AddressSize) {}
  DWARFDataExtractorSimple(ArrayRef<uint8_t> Data, bool IsLittleEndian,
                           uint8_t AddressSize)
      : DataExtractor(
            StringRef(reinterpret_cast<const char *>(Data.data()), Data.size()),
            IsLittleEndian, AddressSize) {}

  /// Truncating constructor
  DWARFDataExtractorSimple(const DWARFDataExtractorSimple &Other, size_t Length)
      : DataExtractor(Other.getData().substr(0, Length), Other.isLittleEndian(),
                      Other.getAddressSize()) {}

  virtual ~DWARFDataExtractorSimple() = default;

  /// Extracts the DWARF "initial length" field, which can either be a 32-bit
  /// value smaller than 0xfffffff0, or the value 0xffffffff followed by a
  /// 64-bit length. Returns the actual length, and the DWARF format which is
  /// encoded in the field. In case of errors, it returns {0, DWARF32} and
  /// leaves the offset unchanged.
  std::pair<uint64_t, dwarf::DwarfFormat>
  getInitialLength(uint64_t *Off, Error *Err = nullptr) const;

  std::pair<uint64_t, dwarf::DwarfFormat> getInitialLength(Cursor &C) const {
    return getInitialLength(&getOffset(C), &getError(C));
  }

  /// Extracts a value and returns it unrelocated. Named such to implement the
  /// required interface.
  virtual uint64_t getRelocatedValue(uint32_t Size, uint64_t *Off,
                                     uint64_t *SectionIndex = nullptr,
                                     Error *Err = nullptr) const {
    assert(SectionIndex == nullptr &&
           "DWARFDATAExtractorSimple cannot take section indices.");
    return getUnsigned(Off, Size, Err);
  }
  virtual uint64_t getRelocatedValue(Cursor &C, uint32_t Size,
                                     uint64_t *SectionIndex = nullptr) const {
    return getRelocatedValue(Size, &getOffset(C), SectionIndex, &getError(C));
  }

  /// Extracts an address-sized value.
  uint64_t getRelocatedAddress(uint64_t *Off, uint64_t *SecIx = nullptr) const {
    return getRelocatedValue(getAddressSize(), Off, SecIx);
  }
  uint64_t getRelocatedAddress(Cursor &C, uint64_t *SecIx = nullptr) const {
    return getRelocatedValue(getAddressSize(), &getOffset(C), SecIx,
                             &getError(C));
  }

  /// Extracts a DWARF-encoded pointer in \p Offset using \p Encoding.
  /// There is a DWARF encoding that uses a PC-relative adjustment.
  /// For these values, \p AbsPosOffset is used to fix them, which should
  /// reflect the absolute address of this pointer.
  std::optional<uint64_t> getEncodedPointer(uint64_t *Offset, uint8_t Encoding,
                                            uint64_t AbsPosOffset = 0) const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTOR_H
