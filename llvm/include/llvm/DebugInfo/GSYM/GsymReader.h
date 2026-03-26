//===- GsymReader.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADER_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/LineEntry.h"
#include "llvm/DebugInfo/GSYM/StringTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"
#include <inttypes.h>
#include <memory>
#include <stdint.h>
#include <vector>

namespace llvm {
class MemoryBuffer;
class raw_ostream;

namespace gsym {

/// GsymReader is an abstract interface for reading GSYM data.
///
/// This interface provides the methods needed by FunctionInfo::lookup and
/// InlineInfo::lookup to resolve strings and files during symbolication.
/// Both GsymReaderV1 and GsymReaderV2 implement this interface.
class GsymReader {
public:
  virtual ~GsymReader() = default;

  /// Get a string from the string table.
  virtual StringRef getString(uint32_t Offset) const = 0;

  /// Get a file entry for the supplied file index.
  virtual std::optional<FileEntry> getFile(uint32_t Index) const = 0;

  /// Get the full function info for an address.
  virtual llvm::Expected<FunctionInfo> getFunctionInfo(uint64_t Addr) const = 0;

  /// Get the full function info given an address index.
  virtual llvm::Expected<FunctionInfo>
  getFunctionInfoAtIndex(uint64_t AddrIdx) const = 0;

  /// Lookup an address in the GSYM.
  virtual llvm::Expected<LookupResult>
  lookup(uint64_t Addr,
         std::optional<DataExtractor> *MergedFuncsData = nullptr) const = 0;

  /// Lookup all merged functions for a given address.
  virtual llvm::Expected<std::vector<LookupResult>>
  lookupAll(uint64_t Addr) const = 0;

  /// Get the number of addresses in this GSYM file.
  virtual uint32_t getNumAddresses() const = 0;

  /// Gets an address from the address table.
  virtual std::optional<uint64_t> getAddress(size_t Index) const = 0;

  /// Dump the entire GSYM data contained in this object.
  virtual void dump(raw_ostream &OS) = 0;

  /// Dump a FunctionInfo object.
  virtual void dump(raw_ostream &OS, const FunctionInfo &FI,
                    uint32_t Indent = 0) = 0;

  /// Dump a MergedFunctionsInfo object.
  virtual void dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) = 0;

  /// Dump a CallSiteInfo object.
  virtual void dump(raw_ostream &OS, const CallSiteInfo &CSI) = 0;

  /// Dump a CallSiteInfoCollection object.
  virtual void dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                    uint32_t Indent = 0) = 0;

  /// Dump a LineTable object.
  virtual void dump(raw_ostream &OS, const LineTable &LT,
                    uint32_t Indent = 0) = 0;

  /// Dump a InlineInfo object.
  virtual void dump(raw_ostream &OS, const InlineInfo &II,
                    uint32_t Indent = 0) = 0;

  /// Dump a FileEntry object.
  virtual void dump(raw_ostream &OS, std::optional<FileEntry> FE) = 0;
};

/// GsymReaderV1 is used to read GSYM V1 data from a file or buffer.
///
/// This class is optimized for very quick lookups when the endianness matches
/// the host system. The Header, address table, address info offsets, and file
/// table is designed to be mmap'ed as read only into memory and used without
/// any parsing needed. If the endianness doesn't match, we swap these objects
/// and tables into GsymReaderV1::SwappedData and then point our header and
/// ArrayRefs to this swapped internal data.
///
/// GsymReaderV1 objects must use one of the static functions to create an
/// instance: GsymReaderV1::openFile(...) and GsymReaderV1::copyBuffer(...).

class GsymReaderV1 : public GsymReader {
  GsymReaderV1(std::unique_ptr<MemoryBuffer> Buffer);
  llvm::Error parse();

  std::unique_ptr<MemoryBuffer> MemBuffer;
  StringRef GsymBytes;
  llvm::endianness Endian;
  const Header *Hdr = nullptr;
  ArrayRef<uint8_t> AddrOffsets;
  ArrayRef<uint32_t> AddrInfoOffsets;
  ArrayRef<FileEntry> Files;
  StringTable StrTab;
  /// When the GSYM file's endianness doesn't match the host system then
  /// we must decode all data structures that need to be swapped into
  /// local storage and set point the ArrayRef objects above to these swapped
  /// copies.
  struct SwappedData {
    Header Hdr;
    std::vector<uint8_t> AddrOffsets;
    std::vector<uint32_t> AddrInfoOffsets;
    std::vector<FileEntry> Files;
  };
  std::unique_ptr<SwappedData> Swap;

public:
  LLVM_ABI GsymReaderV1(GsymReaderV1 &&RHS);
  LLVM_ABI ~GsymReaderV1() override;

  /// Construct a GsymReaderV1 from a file on disk.
  ///
  /// \param Path The file path the GSYM file to read.
  /// \returns An expected GsymReaderV1 that contains the object or an error
  /// object that indicates reason for failing to read the GSYM.
  LLVM_ABI static llvm::Expected<GsymReaderV1> openFile(StringRef Path);

  /// Construct a GsymReaderV1 from a buffer.
  ///
  /// \param Bytes A set of bytes that will be copied and owned by the
  /// returned object on success.
  /// \returns An expected GsymReaderV1 that contains the object or an error
  /// object that indicates reason for failing to read the GSYM.
  LLVM_ABI static llvm::Expected<GsymReaderV1> copyBuffer(StringRef Bytes);

  /// Access the GSYM header.
  /// \returns A native endian version of the GSYM header.
  LLVM_ABI const Header &getHeader() const;

  /// Get the full function info for an address.
  ///
  /// This should be called when a client will store a copy of the complete
  /// FunctionInfo for a given address. For one off lookups, use the lookup()
  /// function below.
  ///
  /// Symbolication server processes might want to parse the entire function
  /// info for a given address and cache it if the process stays around to
  /// service many symbolication addresses, like for parsing profiling
  /// information.
  ///
  /// \param Addr A virtual address from the orignal object file to lookup.
  ///
  /// \returns An expected FunctionInfo that contains the function info object
  /// or an error object that indicates reason for failing to lookup the
  /// address.
  LLVM_ABI llvm::Expected<FunctionInfo>
  getFunctionInfo(uint64_t Addr) const override;

  /// Get the full function info given an address index.
  ///
  /// \param AddrIdx A address index for an address in the address table.
  ///
  /// \returns An expected FunctionInfo that contains the function info object
  /// or an error object that indicates reason for failing get the function
  /// info object.
  LLVM_ABI llvm::Expected<FunctionInfo>
  getFunctionInfoAtIndex(uint64_t AddrIdx) const override;

  /// Lookup an address in the a GSYM.
  ///
  /// Lookup just the information needed for a specific address \a Addr. This
  /// function is faster that calling getFunctionInfo() as it will only return
  /// information that pertains to \a Addr and allows the parsing to skip any
  /// extra information encoded for other addresses. For example the line table
  /// parsing can stop when a matching LineEntry has been fouhnd, and the
  /// InlineInfo can stop parsing early once a match has been found and also
  /// skip information that doesn't match. This avoids memory allocations and
  /// is much faster for lookups.
  ///
  /// \param Addr A virtual address from the orignal object file to lookup.
  ///
  /// \param MergedFuncsData A pointer to an optional DataExtractor that, if
  /// non-null, will be set to the raw data of the MergedFunctionInfo, if
  /// present.
  ///
  /// \returns An expected LookupResult that contains only the information
  /// needed for the current address, or an error object that indicates reason
  /// for failing to lookup the address.
  LLVM_ABI llvm::Expected<LookupResult>
  lookup(uint64_t Addr,
         std::optional<DataExtractor> *MergedFuncsData = nullptr) const override;

  /// Lookup all merged functions for a given address.
  ///
  /// This function performs a lookup for the specified address and then
  /// retrieves additional LookupResults from any merged functions associated
  /// with the primary LookupResult.
  ///
  /// \param Addr The address to lookup.
  ///
  /// \returns A vector of LookupResult objects, where the first element is the
  /// primary result, followed by results for any merged functions
  LLVM_ABI llvm::Expected<std::vector<LookupResult>>
  lookupAll(uint64_t Addr) const override;

  /// Get a string from the string table.
  ///
  /// \param Offset The string table offset for the string to retrieve.
  /// \returns The string from the strin table.
  StringRef getString(uint32_t Offset) const override { return StrTab[Offset]; }

  /// Get the a file entry for the suppplied file index.
  ///
  /// Used to convert any file indexes in the FunctionInfo data back into
  /// files. This function can be used for iteration, but is more commonly used
  /// for random access when doing lookups.
  ///
  /// \param Index An index into the file table.
  /// \returns An optional FileInfo that will be valid if the file index is
  /// valid, or std::nullopt if the file index is out of bounds,
  std::optional<FileEntry> getFile(uint32_t Index) const override {
    if (Index < Files.size())
      return Files[Index];
    return std::nullopt;
  }

  /// Dump the entire Gsym data contained in this object.
  ///
  /// \param  OS The output stream to dump to.
  LLVM_ABI void dump(raw_ostream &OS) override;

  /// Dump a FunctionInfo object.
  ///
  /// This function will convert any string table indexes and file indexes
  /// into human readable format.
  ///
  /// \param  OS The output stream to dump to.
  ///
  /// \param FI The object to dump.
  ///
  /// \param Indent The indentation as number of spaces. Used when dumping as an
  /// item within MergedFunctionsInfo.
  LLVM_ABI void dump(raw_ostream &OS, const FunctionInfo &FI,
                     uint32_t Indent = 0) override;

  /// Dump a MergedFunctionsInfo object.
  ///
  /// This function will dump a MergedFunctionsInfo object - basically by
  /// dumping the contained FunctionInfo objects with indentation.
  ///
  /// \param  OS The output stream to dump to.
  ///
  /// \param MFI The object to dump.
  LLVM_ABI void dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) override;

  /// Dump a CallSiteInfo object.
  ///
  /// This function will output the details of a CallSiteInfo object in a
  /// human-readable format.
  ///
  /// \param OS The output stream to dump to.
  ///
  /// \param CSI The CallSiteInfo object to dump.
  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfo &CSI) override;

  /// Dump a CallSiteInfoCollection object.
  ///
  /// This function will iterate over a collection of CallSiteInfo objects and
  /// dump each one.
  ///
  /// \param OS The output stream to dump to.
  ///
  /// \param CSIC The CallSiteInfoCollection object to dump.
  ///
  /// \param Indent The indentation as number of spaces. Used when dumping as an
  /// item from within MergedFunctionsInfo.
  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                     uint32_t Indent = 0) override;

  /// Dump a LineTable object.
  ///
  /// This function will convert any string table indexes and file indexes
  /// into human readable format.
  ///
  ///
  /// \param  OS The output stream to dump to.
  ///
  /// \param LT The object to dump.
  ///
  /// \param Indent The indentation as number of spaces. Used when dumping as an
  /// item from within MergedFunctionsInfo.
  LLVM_ABI void dump(raw_ostream &OS, const LineTable &LT,
                     uint32_t Indent = 0) override;

  /// Dump a InlineInfo object.
  ///
  /// This function will convert any string table indexes and file indexes
  /// into human readable format.
  ///
  /// \param  OS The output stream to dump to.
  ///
  /// \param II The object to dump.
  ///
  /// \param Indent The indentation as number of spaces. Used for recurive
  /// dumping.
  LLVM_ABI void dump(raw_ostream &OS, const InlineInfo &II,
                     uint32_t Indent = 0) override;

  /// Dump a FileEntry object.
  ///
  /// This function will convert any string table indexes into human readable
  /// format.
  ///
  /// \param  OS The output stream to dump to.
  ///
  /// \param FE The object to dump.
  LLVM_ABI void dump(raw_ostream &OS, std::optional<FileEntry> FE) override;

  /// Get the number of addresses in this Gsym file.
  uint32_t getNumAddresses() const override {
    return Hdr->NumAddresses;
  }

  /// Gets an address from the address table.
  ///
  /// Addresses are stored as offsets frrom the gsym::Header::BaseAddress.
  ///
  /// \param Index A index into the address table.
  /// \returns A resolved virtual address for adddress in the address table
  /// or std::nullopt if Index is out of bounds.
  LLVM_ABI std::optional<uint64_t> getAddress(size_t Index) const override;

protected:

  /// Get an appropriate address info offsets array.
  ///
  /// The address table in the GSYM file is stored as array of 1, 2, 4 or 8
  /// byte offsets from the The gsym::Header::BaseAddress. The table is stored
  /// internally as a array of bytes that are in the correct endianness. When
  /// we access this table we must get an array that matches those sizes. This
  /// templatized helper function is used when accessing address offsets in the
  /// AddrOffsets member variable.
  ///
  /// \returns An ArrayRef of an appropriate address offset size.
  template <class T> ArrayRef<T>
  getAddrOffsets() const {
    return ArrayRef<T>(reinterpret_cast<const T *>(AddrOffsets.data()),
                       AddrOffsets.size()/sizeof(T));
  }

  /// Get an appropriate address from the address table.
  ///
  /// The address table in the GSYM file is stored as array of 1, 2, 4 or 8
  /// byte address offsets from the The gsym::Header::BaseAddress. The table is
  /// stored internally as a array of bytes that are in the correct endianness.
  /// In order to extract an address from the address table we must access the
  /// address offset using the correct size and then add it to the BaseAddress
  /// in the header.
  ///
  /// \param Index An index into the AddrOffsets array.
  /// \returns An virtual address that matches the original object file for the
  /// address as the specified index, or std::nullopt if Index is out of bounds.
  template <class T>
  std::optional<uint64_t> addressForIndex(size_t Index) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    if (Index < AIO.size())
      return AIO[Index] + Hdr->BaseAddress;
    return std::nullopt;
  }
  /// Lookup an address offset in the AddrOffsets table.
  ///
  /// Given an address offset, look it up using a binary search of the
  /// AddrOffsets table.
  ///
  /// \param AddrOffset An address offset, that has already been computed by
  /// subtracting the gsym::Header::BaseAddress.
  /// \returns The matching address offset index. This index will be used to
  /// extract the FunctionInfo data's offset from the AddrInfoOffsets array.
  template <class T>
  std::optional<uint64_t>
  getAddressOffsetIndex(const uint64_t AddrOffset) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    const auto Begin = AIO.begin();
    const auto End = AIO.end();
    auto Iter = std::lower_bound(Begin, End, AddrOffset);
    // Watch for addresses that fall between the gsym::Header::BaseAddress and
    // the first address offset.
    if (Iter == Begin && AddrOffset < *Begin)
      return std::nullopt;
    if (Iter == End || AddrOffset < *Iter)
      --Iter;

    // GSYM files have sorted function infos with the most information (line
    // table and/or inline info) first in the array of function infos, so
    // always backup as much as possible as long as the address offset is the
    // same as the previous entry.
    while (Iter != Begin) {
      auto Prev = Iter - 1;
      if (*Prev == *Iter)
        Iter = Prev;
      else
        break;
    }

    return std::distance(Begin, Iter);
  }

  /// Create a GSYM from a memory buffer.
  ///
  /// Called by both openFile() and copyBuffer(), this function does all of the
  /// work of parsing the GSYM file and returning an error.
  ///
  /// \param MemBuffer A memory buffer that will transfer ownership into the
  /// GsymReaderV1.
  /// \returns An expected GsymReaderV1 that contains the object or an error
  /// object that indicates reason for failing to read the GSYM.
  LLVM_ABI static llvm::Expected<llvm::gsym::GsymReaderV1>
  create(std::unique_ptr<MemoryBuffer> &MemBuffer);

  /// Given an address, find the address index.
  ///
  /// Binary search the address table and find the matching address index.
  ///
  /// \param Addr A virtual address that matches the original object file
  /// to lookup.
  /// \returns An index into the address table. This index can be used to
  /// extract the FunctionInfo data's offset from the AddrInfoOffsets array.
  /// Returns an error if the address isn't in the GSYM with details of why.
  LLVM_ABI Expected<uint64_t> getAddressIndex(const uint64_t Addr) const;

  /// Given an address index, get the offset for the FunctionInfo.
  ///
  /// Looking up an address is done by finding the corresponding address
  /// index for the address. This index is then used to get the offset of the
  /// FunctionInfo data that we will decode using this function.
  ///
  /// \param Index An index into the address table.
  /// \returns An optional GSYM data offset for the offset of the FunctionInfo
  /// that needs to be decoded.
  LLVM_ABI std::optional<uint64_t> getAddressInfoOffset(size_t Index) const;

  /// Given an address, find the correct function info data and function
  /// address.
  ///
  /// Binary search the address table and find the matching address info
  /// and make sure that the function info contains the address. GSYM allows
  /// functions to overlap, and the most debug info is contained in the first
  /// entries due to the sorting when GSYM files are created. We can have
  /// multiple function info that start at the same address only if their
  /// address range doesn't match. So find the first entry that matches \a Addr
  /// and iterate forward until we find one that contains the address.
  ///
  /// \param[in] Addr A virtual address that matches the original object file
  /// to lookup.
  ///
  /// \param[out] FuncStartAddr A virtual address that is the base address of
  /// the function that is used for decoding the FunctionInfo.
  ///
  /// \returns An valid data extractor on success, or an error if we fail to
  /// find the address in a function info or corrrectly decode the data
  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataForAddress(uint64_t Addr, uint64_t &FuncStartAddr) const;

  /// Get the function data and address given an address index.
  ///
  /// \param AddrIdx A address index from the address table.
  ///
  /// \returns An expected FunctionInfo that contains the function info object
  /// or an error object that indicates reason for failing to lookup the
  /// address.
  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataAtIndex(uint64_t AddrIdx, uint64_t &FuncStartAddr) const;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADER_H
