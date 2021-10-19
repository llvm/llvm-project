//===- GsymCreator.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H
#define LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/Range.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace llvm {

namespace gsym {
class FileWriter;

/// GsymCreator is used to emit GSYM data to a stand alone file or section
/// within a file.
///
/// The GsymCreator is designed to be used in 3 stages:
/// - Create FunctionInfo objects and add them
/// - Finalize the GsymCreator object
/// - Save to file or section
///
/// The first stage involves creating FunctionInfo objects from another source
/// of information like compiler debug info metadata, DWARF or Breakpad files.
/// Any strings in the FunctionInfo or contained information, like InlineInfo
/// or LineTable objects, should get the string table offsets by calling
/// GsymCreator::insertString(...). Any file indexes that are needed should be
/// obtained by calling GsymCreator::insertFile(...). All of the function calls
/// in GsymCreator are thread safe. This allows multiple threads to create and
/// add FunctionInfo objects while parsing debug information.
///
/// Once all of the FunctionInfo objects have been added, the
/// GsymCreator::finalize(...) must be called prior to saving. This function
/// will sort the FunctionInfo objects, finalize the string table, and do any
/// other passes on the information needed to prepare the information to be
/// saved.
///
/// Once the object has been finalized, it can be saved to a file or section.
///
/// ENCODING
///
/// GSYM files are designed to be memory mapped into a process as shared, read
/// only data, and used as is.
///
/// The GSYM file format when in a stand alone file consists of:
///   - Header
///   - Address Table
///   - Function Info Offsets
///   - File Table
///   - String Table
///   - Function Info Data
///
/// HEADER
///
/// The header is fully described in "llvm/DebugInfo/GSYM/Header.h".
///
/// ADDRESS TABLE
///
/// The address table immediately follows the header in the file and consists
/// of Header.NumAddresses address offsets. These offsets are sorted and can be
/// binary searched for efficient lookups. Addresses in the address table are
/// stored as offsets from a 64 bit base address found in Header.BaseAddress.
/// This allows the address table to contain 8, 16, or 32 offsets. This allows
/// the address table to not require full 64 bit addresses for each address.
/// The resulting GSYM size is smaller and causes fewer pages to be touched
/// during address lookups when the address table is smaller. The size of the
/// address offsets in the address table is specified in the header in
/// Header.AddrOffSize. The first offset in the address table is aligned to
/// Header.AddrOffSize alignment to ensure efficient access when loaded into
/// memory.
///
/// FUNCTION INFO OFFSETS TABLE
///
/// The function info offsets table immediately follows the address table and
/// consists of Header.NumAddresses 32 bit file offsets: one for each address
/// in the address table. This data is aligned to a 4 byte boundary. The
/// offsets in this table are the relative offsets from the start offset of the
/// GSYM header and point to the function info data for each address in the
/// address table. Keeping this data separate from the address table helps to
/// reduce the number of pages that are touched when address lookups occur on a
/// GSYM file.
///
/// FILE TABLE
///
/// The file table immediately follows the function info offsets table. The
/// encoding of the FileTable is:
///
/// struct FileTable {
///   uint32_t Count;
///   FileEntry Files[];
/// };
///
/// The file table starts with a 32 bit count of the number of files that are
/// used in all of the function info, followed by that number of FileEntry
/// structures. The file table is aligned to a 4 byte boundary, Each file in
/// the file table is represented with a FileEntry structure.
/// See "llvm/DebugInfo/GSYM/FileEntry.h" for details.
///
/// STRING TABLE
///
/// The string table follows the file table in stand alone GSYM files and
/// contains all strings for everything contained in the GSYM file. Any string
/// data should be added to the string table and any references to strings
/// inside GSYM information must be stored as 32 bit string table offsets into
/// this string table. The string table always starts with an empty string at
/// offset zero and is followed by any strings needed by the GSYM information.
/// The start of the string table is not aligned to any boundary.
///
/// FUNCTION INFO DATA
///
/// The function info data is the payload that contains information about the
/// address that is being looked up. It contains all of the encoded
/// FunctionInfo objects. Each encoded FunctionInfo's data is pointed to by an
/// entry in the Function Info Offsets Table. For details on the exact encoding
/// of FunctionInfo objects, see "llvm/DebugInfo/GSYM/FunctionInfo.h".
class GsymCreator {
  // Private member variables require Mutex protections
  mutable std::mutex Mutex;
  std::vector<FunctionInfo> Funcs;
  StringTableBuilder StrTab;
  StringSet<> StringStorage;
  DenseMap<llvm::gsym::FileEntry, uint32_t> FileEntryToIndex;
  std::vector<llvm::gsym::FileEntry> Files;
  std::vector<uint8_t> UUID;
  Optional<AddressRanges> ValidTextRanges;
  AddressRanges Ranges;
  llvm::Optional<uint64_t> BaseAddress;
  bool Finalized = false;
  bool Quiet;

public:
  GsymCreator(bool Quiet = false);

  /// Save a GSYM file to a stand alone file.
  ///
  /// \param Path The file path to save the GSYM file to.
  /// \param ByteOrder The endianness to use when saving the file.
  /// \returns An error object that indicates success or failure of the save.
  llvm::Error save(StringRef Path, llvm::support::endianness ByteOrder) const;

  /// Encode a GSYM into the file writer stream at the current position.
  ///
  /// \param O The stream to save the binary data to
  /// \returns An error object that indicates success or failure of the save.
  llvm::Error encode(FileWriter &O) const;

  /// Insert a string into the GSYM string table.
  ///
  /// All strings used by GSYM files must be uniqued by adding them to this
  /// string pool and using the returned offset for any string values.
  ///
  /// \param S The string to insert into the string table.
  /// \param Copy If true, then make a backing copy of the string. If false,
  ///             the string is owned by another object that will stay around
  ///             long enough for the GsymCreator to save the GSYM file.
  /// \returns The unique 32 bit offset into the string table.
  uint32_t insertString(StringRef S, bool Copy = true);

  /// Insert a file into this GSYM creator.
  ///
  /// Inserts a file by adding a FileEntry into the "Files" member variable if
  /// the file has not already been added. The file path is split into
  /// directory and filename which are both added to the string table. This
  /// allows paths to be stored efficiently by reusing the directories that are
  /// common between multiple files.
  ///
  /// \param   Path The path to the file to insert.
  /// \param   Style The path style for the "Path" parameter.
  /// \returns The unique file index for the inserted file.
  uint32_t insertFile(StringRef Path,
                      sys::path::Style Style = sys::path::Style::native);

  /// Add a function info to this GSYM creator.
  ///
  /// All information in the FunctionInfo object must use the
  /// GsymCreator::insertString(...) function when creating string table
  /// offsets for names and other strings.
  ///
  /// \param   FI The function info object to emplace into our functions list.
  void addFunctionInfo(FunctionInfo &&FI);

  /// Finalize the data in the GSYM creator prior to saving the data out.
  ///
  /// Finalize must be called after all FunctionInfo objects have been added
  /// and before GsymCreator::save() is called.
  ///
  /// \param  OS Output stream to report duplicate function infos, overlapping
  ///         function infos, and function infos that were merged or removed.
  /// \returns An error object that indicates success or failure of the
  ///          finalize.
  llvm::Error finalize(llvm::raw_ostream &OS);

  /// Set the UUID value.
  ///
  /// \param UUIDBytes The new UUID bytes.
  void setUUID(llvm::ArrayRef<uint8_t> UUIDBytes) {
    UUID.assign(UUIDBytes.begin(), UUIDBytes.end());
  }

  /// Thread safe iteration over all function infos.
  ///
  /// \param  Callback A callback function that will get called with each
  ///         FunctionInfo. If the callback returns false, stop iterating.
  void forEachFunctionInfo(
      std::function<bool(FunctionInfo &)> const &Callback);

  /// Thread safe const iteration over all function infos.
  ///
  /// \param  Callback A callback function that will get called with each
  ///         FunctionInfo. If the callback returns false, stop iterating.
  void forEachFunctionInfo(
      std::function<bool(const FunctionInfo &)> const &Callback) const;

  /// Get the current number of FunctionInfo objects contained in this
  /// object.
  size_t getNumFunctionInfos() const;

  /// Check if an address has already been added as a function info.
  ///
  /// FunctionInfo data can come from many sources: debug info, symbol tables,
  /// exception information, and more. Symbol tables should be added after
  /// debug info and can use this function to see if a symbol's start address
  /// has already been added to the GsymReader. Calling this before adding
  /// a function info from a source other than debug info avoids clients adding
  /// many redundant FunctionInfo objects from many sources only for them to be
  /// removed during the finalize() call.
  bool hasFunctionInfoForAddress(uint64_t Addr) const;

  /// Set valid .text address ranges that all functions must be contained in.
  void SetValidTextRanges(AddressRanges &TextRanges) {
    ValidTextRanges = TextRanges;
  }

  /// Get the valid text ranges.
  const Optional<AddressRanges> GetValidTextRanges() const {
    return ValidTextRanges;
  }

  /// Check if an address is a valid code address.
  ///
  /// Any functions whose addresses do not exist within these function bounds
  /// will not be converted into the final GSYM. This allows the object file
  /// to figure out the valid file address ranges of all the code sections
  /// and ensure we don't add invalid functions to the final output. Many
  /// linkers have issues when dead stripping functions from DWARF debug info
  /// where they set the DW_AT_low_pc to zero, but newer DWARF has the
  /// DW_AT_high_pc as an offset from the DW_AT_low_pc and these size
  /// attributes have no relocations that can be applied. This results in DWARF
  /// where many functions have an DW_AT_low_pc of zero and a valid offset size
  /// for DW_AT_high_pc. If we extract all valid ranges from an object file
  /// that are marked with executable permissions, we can properly ensure that
  /// these functions are removed.
  ///
  /// \param Addr An address to check.
  ///
  /// \returns True if the address is in the valid text ranges or if no valid
  ///          text ranges have been set, false otherwise.
  bool IsValidTextAddress(uint64_t Addr) const;

  /// Set the base address to use for the GSYM file.
  ///
  /// Setting the base address to use for the GSYM file. Object files typically
  /// get loaded from a base address when the OS loads them into memory. Using
  /// GSYM files for symbolication becomes easier if the base address in the
  /// GSYM header is the same address as it allows addresses to be easily slid
  /// and allows symbolication without needing to find the original base
  /// address in the original object file.
  ///
  /// \param  Addr The address to use as the base address of the GSYM file
  ///              when it is saved to disk.
  void setBaseAddress(uint64_t Addr) {
    BaseAddress = Addr;
  }

  /// Whether the transformation should be quiet, i.e. not output warnings.
  bool isQuiet() const { return Quiet; }
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H
