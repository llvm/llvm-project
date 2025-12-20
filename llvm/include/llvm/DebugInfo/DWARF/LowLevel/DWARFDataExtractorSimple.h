//===- DWARFDataExtractorSimple.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFDATAEXTRACTORSIMPLE_H
#define LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFDATAEXTRACTORSIMPLE_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

/// A DataExtractor suitable use for parsing dwarf from memory.  Clients use
/// Relocator::getRelocatedValueImpl to relocate values as appropriate.

template <typename Relocator>
class DWARFDataExtractorBase : public DataExtractor {

public:
  DWARFDataExtractorBase(StringRef Data, bool IsLittleEndian,
                         uint8_t AddressSize)
      : DataExtractor(Data, IsLittleEndian, AddressSize) {}
  DWARFDataExtractorBase(ArrayRef<uint8_t> Data, bool IsLittleEndian,
                         uint8_t AddressSize)
      : DataExtractor(
            StringRef(reinterpret_cast<const char *>(Data.data()), Data.size()),
            IsLittleEndian, AddressSize) {}

  /// Truncating constructor
  DWARFDataExtractorBase(const DWARFDataExtractorBase &Other, size_t Length)
      : DataExtractor(Other.getData().substr(0, Length), Other.isLittleEndian(),
                      Other.getAddressSize()) {}

  /// Extracts a value and returns it as adjusted by the Relocator
  uint64_t getRelocatedValue(uint32_t Size, uint64_t *Off,
                             uint64_t *SectionIndex = nullptr,
                             Error *Err = nullptr) const {
    return static_cast<const Relocator *>(this)->getRelocatedValueImpl(
        Size, Off, SectionIndex, Err);
  }

  uint64_t getRelocatedValue(Cursor &C, uint32_t Size,
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

  /// Extracts the DWARF "initial length" field, which can either be a 32-bit
  /// value smaller than 0xfffffff0, or the value 0xffffffff followed by a
  /// 64-bit length. Returns the actual length, and the DWARF format which is
  /// encoded in the field. In case of errors, it returns {0, DWARF32} and
  /// leaves the offset unchanged.
  std::pair<uint64_t, dwarf::DwarfFormat>
  getInitialLength(uint64_t *Off, Error *Err = nullptr) const {
    ErrorAsOutParameter ErrAsOut(Err);
    if (Err && *Err)
      return {0, dwarf::DWARF32};

    Cursor C(*Off);
    uint64_t Length = getRelocatedValue(C, 4);
    dwarf::DwarfFormat Format = dwarf::DWARF32;
    if (Length == dwarf::DW_LENGTH_DWARF64) {
      Length = getRelocatedValue(C, 8);
      Format = dwarf::DWARF64;
    } else if (Length >= dwarf::DW_LENGTH_lo_reserved) {
      cantFail(C.takeError());
      if (Err)
        *Err = createStringError(
            std::errc::invalid_argument,
            "unsupported reserved unit length of value 0x%8.8" PRIx64, Length);
      return {0, dwarf::DWARF32};
    }

    if (C) {
      *Off = C.tell();
      return {Length, Format};
    }
    if (Err)
      *Err = C.takeError();
    else
      consumeError(C.takeError());
    return {0, dwarf::DWARF32};
  }

  std::pair<uint64_t, dwarf::DwarfFormat> getInitialLength(Cursor &C) const {
    return getInitialLength(&getOffset(C), &getError(C));
  }

  /// Extracts a DWARF-encoded pointer in \p Offset using \p Encoding.
  /// There is a DWARF encoding that uses a PC-relative adjustment.
  /// For these values, \p AbsPosOffset is used to fix them, which should
  /// reflect the absolute address of this pointer.
  std::optional<uint64_t> getEncodedPointer(uint64_t *Offset, uint8_t Encoding,
                                            uint64_t PCRelOffset) const {
    if (Encoding == dwarf::DW_EH_PE_omit)
      return std::nullopt;

    uint64_t Result = 0;
    uint64_t OldOffset = *Offset;
    // First get value
    switch (Encoding & 0x0F) {
    case dwarf::DW_EH_PE_absptr:
      switch (getAddressSize()) {
      case 2:
      case 4:
      case 8:
        Result = getUnsigned(Offset, getAddressSize());
        break;
      default:
        return std::nullopt;
      }
      break;
    case dwarf::DW_EH_PE_uleb128:
      Result = getULEB128(Offset);
      break;
    case dwarf::DW_EH_PE_sleb128:
      Result = getSLEB128(Offset);
      break;
    case dwarf::DW_EH_PE_udata2:
      Result = getUnsigned(Offset, 2);
      break;
    case dwarf::DW_EH_PE_udata4:
      Result = getUnsigned(Offset, 4);
      break;
    case dwarf::DW_EH_PE_udata8:
      Result = getUnsigned(Offset, 8);
      break;
    case dwarf::DW_EH_PE_sdata2:
      Result = getSigned(Offset, 2);
      break;
    case dwarf::DW_EH_PE_sdata4:
      Result = SignExtend64<32>(getRelocatedValue(4, Offset));
      break;
    case dwarf::DW_EH_PE_sdata8:
      Result = getRelocatedValue(8, Offset);
      break;
    default:
      return std::nullopt;
    }
    // Then add relative offset, if required
    switch (Encoding & 0x70) {
    case dwarf::DW_EH_PE_absptr:
      // do nothing
      break;
    case dwarf::DW_EH_PE_pcrel:
      Result += PCRelOffset;
      break;
    case dwarf::DW_EH_PE_datarel:
    case dwarf::DW_EH_PE_textrel:
    case dwarf::DW_EH_PE_funcrel:
    case dwarf::DW_EH_PE_aligned:
    default:
      *Offset = OldOffset;
      return std::nullopt;
    }

    return Result;
  }
};

// Non relocating, low-level dwarf-data extractor. Suitable for use from
// libraries that cannot have build-time dependencies on relocation providers.

class DWARFDataExtractorSimple
    : public DWARFDataExtractorBase<DWARFDataExtractorSimple> {
public:
  using DWARFDataExtractorBase::DWARFDataExtractorBase;

  LLVM_ABI uint64_t getRelocatedValueImpl(uint32_t Size, uint64_t *Off,
                                          uint64_t *SectionIndex = nullptr,
                                          Error *Err = nullptr) const {
    assert(SectionIndex == nullptr &&
           "DWARFDATAExtractorSimple cannot take section indices.");
    return getUnsigned(Off, Size, Err);
  }
};

} // end namespace llvm
#endif // LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFDATAEXTRACTORSIMPLE_H
