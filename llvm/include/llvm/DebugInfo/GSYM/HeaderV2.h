//===- HeaderV2.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_HEADERV2_H
#define LLVM_DEBUGINFO_GSYM_HEADERV2_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace llvm {
class raw_ostream;
class DataExtractor;

namespace gsym {
class FileWriter;

constexpr uint32_t GSYM_MAGIC = 0x4753594d; // 'GSYM'
constexpr uint32_t GSYM_CIGAM = 0x4d595347; // 'MYSG'
constexpr uint32_t GSYM_VERSION_2 = 2;

/// The GSYM V2 header.
///
/// The GSYM V2 header is found at the start of a stand alone GSYM file, or as
/// the first bytes in a section when GSYM is contained in a section of an
/// executable file (ELF, mach-o, COFF).
///
/// The V2 file format consists of the following GSYM sections in order:
///   - Header (this struct, 40 bytes)
///   - GlobalData (a list of GlobalData, each point to one of the following GSYM sections)
///   - Followed by all the sections mentioned in the GlobalData list at the specified file offsets and sizes, with padding of zeros for alignment.
///
/// The header structure is encoded exactly as it appears in the structure definition
/// with no gaps between members. Alignment should not change from system to
/// system as the members are laid out so that they will align the same
/// on different architectures.
///
/// When endianness of the system loading a GSYM file matches, the file can
/// be mmap'ed in and a pointer to the header can be cast to the first bytes
/// of the file (stand alone GSYM file) or section data (GSYM in a section).
/// When endianness is swapped, the HeaderV2::decode() function should be used
/// to decode the header.
struct HeaderV2 {
  /// The magic bytes should be set to GSYM_MAGIC. This helps detect if a file
  /// is a GSYM file by scanning the first 4 bytes of a file or section.
  /// This value might appear byte swapped when endianness is swapped.
  uint32_t Magic;
  /// The version number determines how the header is decoded. As version numbers increase,
  /// "Magic" and "Version" members should always appear at offset zero and 4
  /// respectively to ensure clients figure out if they can parse the format.
  uint16_t Version;
  /// Padding for alignment to keep all the "size" fields together. Must be set to zero.
  uint16_t Padding;
  /// The 64 bit base address that all address offsets in the address offsets
  /// table are relative to. Storing a full 64 bit address allows our address
  /// offsets table to be smaller on disk.
  uint64_t BaseAddress;
  /// The number of addresses stored in the address offsets table and the
  /// address info offsets table.
  uint32_t NumAddresses;
  /// The size in bytes of each address offset in the address offsets table.
  uint8_t AddrOffSize;
  /// The size in bytes of each entry in the address info offsets table.
  /// These offsets point into GlobalData.
  uint8_t AddrInfoOffSize;
  /// The size in bytes of each string table reference (strp) in FunctionInfo
  /// and other data structures within GlobalData.
  uint8_t StrpSize;
  /// Padding for alignment. Must be set to zero.
  uint8_t Padding2;
  /// The starting point of the global data. This is a list of GlobalData objects, with the last one being the
  /// GlobalInfoType::EndOfList. Each of the GlobalData objects point to a section in the GSYM, e.g. address FunctionInfos, UUID, string table, and any other future sections.
  uint8_t GlobalData[0];

  /// Check if a header is valid and return an error if anything is wrong.
  ///
  /// This function can be used prior to encoding a header to ensure it is
  /// valid, or after decoding a header to ensure it is valid and supported.
  ///
  /// Check a correctly byte swapped header for errors:
  ///   - check magic value
  ///   - check that version number is supported
  ///   - check that the address offset size is supported
  ///   - check that the address info offset size is supported
  ///   - check that the strp size is supported
  ///   - check that padding fields are zero
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
  LLVM_ABI static llvm::Expected<HeaderV2> decode(DataExtractor &Data);

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
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const llvm::gsym::HeaderV2 &H);

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_HEADERV2_H
