//===- GsymReaderV1.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/GsymReader.h"
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

  LLVM_ABI llvm::Expected<FunctionInfo>
  getFunctionInfo(uint64_t Addr) const override;

  LLVM_ABI llvm::Expected<FunctionInfo>
  getFunctionInfoAtIndex(uint64_t AddrIdx) const override;

  LLVM_ABI llvm::Expected<LookupResult>
  lookup(uint64_t Addr,
         std::optional<DataExtractor> *MergedFuncsData = nullptr) const override;

  LLVM_ABI llvm::Expected<std::vector<LookupResult>>
  lookupAll(uint64_t Addr) const override;

  StringRef getString(uint32_t Offset) const override { return StrTab[Offset]; }

  std::optional<FileEntry> getFile(uint32_t Index) const override {
    if (Index < Files.size())
      return Files[Index];
    return std::nullopt;
  }

  LLVM_ABI void dump(raw_ostream &OS) override;

  LLVM_ABI void dump(raw_ostream &OS, const FunctionInfo &FI,
                     uint32_t Indent = 0) override;

  LLVM_ABI void dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) override;

  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfo &CSI) override;

  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                     uint32_t Indent = 0) override;

  LLVM_ABI void dump(raw_ostream &OS, const LineTable &LT,
                     uint32_t Indent = 0) override;

  LLVM_ABI void dump(raw_ostream &OS, const InlineInfo &II,
                     uint32_t Indent = 0) override;

  LLVM_ABI void dump(raw_ostream &OS, std::optional<FileEntry> FE) override;

  uint32_t getNumAddresses() const override {
    return Hdr->NumAddresses;
  }

  LLVM_ABI std::optional<uint64_t> getAddress(size_t Index) const override;

protected:

  template <class T> ArrayRef<T>
  getAddrOffsets() const {
    return ArrayRef<T>(reinterpret_cast<const T *>(AddrOffsets.data()),
                       AddrOffsets.size()/sizeof(T));
  }

  template <class T>
  std::optional<uint64_t> addressForIndex(size_t Index) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    if (Index < AIO.size())
      return AIO[Index] + Hdr->BaseAddress;
    return std::nullopt;
  }

  template <class T>
  std::optional<uint64_t>
  getAddressOffsetIndex(const uint64_t AddrOffset) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    const auto Begin = AIO.begin();
    const auto End = AIO.end();
    auto Iter = std::lower_bound(Begin, End, AddrOffset);
    if (Iter == Begin && AddrOffset < *Begin)
      return std::nullopt;
    if (Iter == End || AddrOffset < *Iter)
      --Iter;

    while (Iter != Begin) {
      auto Prev = Iter - 1;
      if (*Prev == *Iter)
        Iter = Prev;
      else
        break;
    }

    return std::distance(Begin, Iter);
  }

  LLVM_ABI static llvm::Expected<llvm::gsym::GsymReaderV1>
  create(std::unique_ptr<MemoryBuffer> &MemBuffer);

  LLVM_ABI Expected<uint64_t> getAddressIndex(const uint64_t Addr) const;

  LLVM_ABI std::optional<uint64_t> getAddressInfoOffset(size_t Index) const;

  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataForAddress(uint64_t Addr, uint64_t &FuncStartAddr) const;

  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataAtIndex(uint64_t AddrIdx, uint64_t &FuncStartAddr) const;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
