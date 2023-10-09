//===- MC/MCCASDebugV1.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_CAS_MCCASDEBUGV1_H
#define LLVM_MC_CAS_MCCASDEBUGV1_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"

namespace llvm {
namespace mccasformats {
namespace v1 {

constexpr unsigned Dwarf4HeaderSize32Bit = 11;
constexpr unsigned Dwarf5HeaderSize32Bit = 12;

/// Returns true if the values associated with a combination of Form and Attr
/// are not expected to deduplicate.
bool doesntDedup(dwarf::Form Form, dwarf::Attribute Attr);

/// Reads data from `CUData[CUOffset]`, interpreting it as a value encoded as
/// `Form`, and returns the number of bytes taken by the encoded value.
Expected<uint64_t> getFormSize(dwarf::Form Form, dwarf::FormParams FP,
                               StringRef CUData, uint64_t CUOffset,
                               bool IsLittleEndian, uint8_t AddressSize);

/// A special value to indicate the end of a sequence of sibling DIEs.
inline uint16_t getEndOfDIESiblingsMarker() { return 0; }

/// A special value to indicate that a DIE has been placed in a separate CAS
/// block.
inline uint16_t getDIEInAnotherBlockMarker() { return 1; }

/// Converts an index into the abbreviation table into a value that can be
/// written to the CAS.
inline uint64_t encodeAbbrevIndex(uint64_t Index) { return Index + 2; }

/// Decodes a value written to the CAS as an index into the abbreviation set's
/// list of CAS references.
inline uint64_t decodeAbbrevIndexAsAbbrevSetIdx(uint64_t Index) {
  assert(Index != getEndOfDIESiblingsMarker() &&
         Index != getDIEInAnotherBlockMarker());
  return Index - 2;
}

/// Decodes a value written to the CAS as an index into DWARF's abbreviation
/// table.
inline uint64_t decodeAbbrevIndexAsDwarfAbbrevIdx(uint64_t Index) {
  assert(Index != getEndOfDIESiblingsMarker() &&
         Index != getDIEInAnotherBlockMarker());
  // Dwarf 5: Section 7.5.3:
  // the abbreviation code 0 is reserved for null debugging information entries.
  return Index - 1;
}

/// A special value to indicate the end of a sequence of attributes. This must
/// not conflict with any dwarf::Attribute.
inline uint16_t getEndOfAttributesMarker() { return 0; }

/// Helper class to write (possibly LEB encoded) data to a stream backed by a
/// SmallVector.
struct DataWriter {
  DataWriter() : DataStream(Data) {}

  /// Write ULEB128(V) to the data stream.
  void writeULEB128(uint64_t V) { encodeULEB128(V, DataStream); }

  /// Write SLEB128(V) to the data stream.
  void writeSLEB128(int64_t V) { encodeSLEB128(V, DataStream); }

  /// Write V to the data stream.
  void writeByte(uint8_t V) { DataStream << V; }

  /// Write the contents of V to the data stream.
  void writeData(ArrayRef<char> V) { DataStream.write(V.data(), V.size()); }

protected:
  SmallVector<char> Data;
  raw_svector_ostream DataStream;
};

/// Use a more efficient format for storing 4-byte wide form data.
uint64_t convertFourByteFormDataToULEB(ArrayRef<char> FormData,
                                       DataWriter &Writer);

// Helper class to write a DIE's abbreviation contents to a buffer.
struct AbbrevEntryWriter : DataWriter {
  void writeAbbrevEntry(DWARFDie DIE);
};

struct AbbrevEntryReader {
  AbbrevEntryReader(StringRef Data)
      : DataStream(Data, support::endianness::little) {}
  Expected<dwarf::Tag> readTag();
  Expected<bool> readHasChildren();

  Expected<dwarf::Attribute> readAttr();
  Expected<dwarf::Form> readForm();

private:
  BinaryStreamReader DataStream;
};

/// Given a sequence of AbbrevEntries, as written by an AbbrevEntryWriter,
/// reconstruct the abbreviation section into OS according to the DWARF
/// specification.
/// Returns the number of bytes written to OS.
uint64_t reconstructAbbrevSection(raw_ostream &OS,
                                  ArrayRef<StringRef> AbbrevEntries,
                                  uint64_t &MaxDIEAbbrevCount);
} // namespace v1
} // namespace mccasformats
} // namespace llvm

#endif // LLVM_MC_CAS_MCCASDEBUGV1_H
