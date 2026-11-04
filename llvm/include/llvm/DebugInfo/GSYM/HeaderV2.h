//===- HeaderV2.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_HEADERV2_H
#define LLVM_DEBUGINFO_GSYM_HEADERV2_H

#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace llvm {
class raw_ostream;

namespace gsym {
class FileWriter;
class GsymDataExtractor;

/// Encoding format for the string table.
enum class StringTableEncoding : uint8_t {
  /// A list of NULL-terminated strings (same as V1). The first string at
  /// offset zero must be the empty C string.
  Default = 0,
};

/// The GSYM V2 header.
///
/// The GSYM V2 header is found at the start of a stand alone GSYM file, or as
/// the first bytes in a section when GSYM is contained in a section of an
/// executable file (ELF, mach-o, COFF).
///
/// The header structure is encoded exactly as it appears in the structure
/// definition with no gaps between members. Alignment should not change from
/// system to system as the members are laid out so that they will align the
/// same on different architectures.
///
/// When endianness of the system loading a GSYM file matches, the file can
/// be mmap'ed in and a pointer to the header can be cast to the first bytes
/// of the file (stand alone GSYM file) or section data (GSYM in a section).
/// When endianness is swapped, HeaderV2::decode() is used to read the header.
///
/// The V2 file layout is:
///   [HeaderV2 - 20 bytes fixed]
///   [GlobalData entries - array of 20-byte entries, terminated by EndOfList]
///   [Data sections at arbitrary file offsets, zero-padded for alignment]
///
/// Each GlobalData entry (see GlobalData.h) describes a section by its type,
/// file offset, and file size. The sections can appear in any order in the
/// file since each GlobalData entry contains an offset from the first byte of
/// the GSYM header. The GlobalData array is terminated by an entry with type
/// EndOfList and all other fields set to zero. See GlobalInfoType (in
/// GlobalData.h) for all section types.
struct HeaderV2 {
  /// The magic bytes should be set to GSYM_MAGIC. This helps detect if a file
  /// is a GSYM file by scanning the first 4 bytes of a file or section.
  /// This value might appear byte swapped when endianness is swapped.
  uint32_t Magic;
  /// The version number determines how the header is decoded. As version
  /// numbers increase, "Magic" and "Version" members should always appear at
  /// offset zero and 4 respectively to ensure clients figure out if they can
  /// parse the format.
  uint16_t Version;
  /// The size in bytes of each address offset in the address offsets table.
  uint8_t AddrOffSize;
  /// String table encoding. Allows for future encoding for string table.
  StringTableEncoding StrTableEncoding;
  /// The 64 bit base address that all address offsets in the address offsets
  /// table are relative to. Storing a full 64 bit address allows our address
  /// offsets table to be smaller on disk.
  uint64_t BaseAddress;
  /// The number of addresses stored in the address offsets table and the
  /// address info offsets table.
  uint32_t NumAddresses;

  /// Return the version of this header.
  static constexpr uint32_t getVersion() { return 2; }

  /// Return the on-disk encoded size of the header in bytes.
  /// This may differ from sizeof(HeaderV2) due to struct padding at the end.
  static constexpr uint64_t getEncodedSize() { return 20; }

  /// Return the size in bytes of address info offsets.
  static constexpr uint8_t getAddressInfoOffsetSize() { return 8; }

  /// Return the size in bytes of string table offsets.
  static constexpr uint8_t getStringOffsetSize() { return 8; }

  /// Check if a header is valid and return an error if anything is wrong.
  ///
  /// This function can be used prior to encoding a header to ensure it is
  /// valid, or after decoding a header to ensure it is valid and supported.
  ///
  /// Check a correctly byte swapped header for errors:
  ///   - check magic value
  ///   - check that version number is supported
  ///   - check that the address offset size is supported
  ///   - check that the string table encoding is supported
  ///
  /// \returns An error if anything is wrong in the header, or Error::success()
  /// if there are no errors.
  LLVM_ABI llvm::Error checkForError() const;

  /// Decode an object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from. This object must
  /// have the data for the object starting at offset zero. The data
  /// can contain more data than needed.
  ///
  /// \returns A HeaderV2 or an error describing the issue that was
  /// encountered during decoding.
  LLVM_ABI static llvm::Expected<HeaderV2> decode(GsymDataExtractor &Data);

  /// Encode this object into FileWriter stream.
  ///
  /// \param O The binary stream to write the data to at the current file
  /// position.
  ///
  /// \returns An error object that indicates success or failure of the
  /// encoding process.
  LLVM_ABI llvm::Error encode(FileWriter &O) const;
};

LLVM_ABI bool operator==(const HeaderV2 &LHS, const HeaderV2 &RHS);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const llvm::gsym::HeaderV2 &H);

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_HEADERV2_H
