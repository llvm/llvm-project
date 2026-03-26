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

/// GsymReader is the base class for reading GSYM data.
///
/// This class contains all shared state and logic for V1 and V2 readers.
/// Subclasses implement version-specific parsing (parse()) and header access.
class GsymReader {
protected:
  std::unique_ptr<MemoryBuffer> MemBuffer;
  llvm::endianness Endian;
  ArrayRef<uint8_t> AddrOffsets;
  ArrayRef<uint32_t> AddrInfoOffsets;
  ArrayRef<FileEntry> Files;
  StringTable StrTab;

  // Cached header values, populated by subclass parse().
  uint64_t CachedBaseAddress = 0;
  uint32_t CachedNumAddresses = 0;
  uint8_t CachedAddrOffSize = 0;

  LLVM_ABI GsymReader(std::unique_ptr<MemoryBuffer> Buffer);

  template <class T> ArrayRef<T>
  getAddrOffsets() const {
    return ArrayRef<T>(reinterpret_cast<const T *>(AddrOffsets.data()),
                       AddrOffsets.size()/sizeof(T));
  }

  template <class T>
  std::optional<uint64_t> addressForIndex(size_t Index) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    if (Index < AIO.size())
      return AIO[Index] + CachedBaseAddress;
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

  LLVM_ABI Expected<uint64_t> getAddressIndex(const uint64_t Addr) const;
  LLVM_ABI std::optional<uint64_t> getAddressInfoOffset(size_t Index) const;
  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataForAddress(uint64_t Addr, uint64_t &FuncStartAddr) const;
  LLVM_ABI llvm::Expected<llvm::DataExtractor>
  getFunctionInfoDataAtIndex(uint64_t AddrIdx, uint64_t &FuncStartAddr) const;

public:
  LLVM_ABI GsymReader(GsymReader &&RHS);
  virtual ~GsymReader() = default;

  /// Open a GSYM file, auto-detecting the format version.
  LLVM_ABI static llvm::Expected<std::unique_ptr<GsymReader>>
  openFile(StringRef Path);

  /// Construct a GsymReader from a buffer, auto-detecting the format version.
  LLVM_ABI static llvm::Expected<std::unique_ptr<GsymReader>>
  copyBuffer(StringRef Bytes);

  StringRef getString(uint32_t Offset) const { return StrTab[Offset]; }

  std::optional<FileEntry> getFile(uint32_t Index) const {
    if (Index < Files.size())
      return Files[Index];
    return std::nullopt;
  }

  uint32_t getNumAddresses() const { return CachedNumAddresses; }

  LLVM_ABI llvm::Expected<FunctionInfo> getFunctionInfo(uint64_t Addr) const;
  LLVM_ABI llvm::Expected<FunctionInfo>
  getFunctionInfoAtIndex(uint64_t AddrIdx) const;

  LLVM_ABI llvm::Expected<LookupResult>
  lookup(uint64_t Addr,
         std::optional<DataExtractor> *MergedFuncsData = nullptr) const;

  LLVM_ABI llvm::Expected<std::vector<LookupResult>>
  lookupAll(uint64_t Addr) const;

  LLVM_ABI std::optional<uint64_t> getAddress(size_t Index) const;

  /// Dump the entire GSYM data. Version-specific (header format differs).
  virtual void dump(raw_ostream &OS) = 0;

  LLVM_ABI void dump(raw_ostream &OS, const FunctionInfo &FI,
                     uint32_t Indent = 0);
  LLVM_ABI void dump(raw_ostream &OS, const MergedFunctionsInfo &MFI);
  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfo &CSI);
  LLVM_ABI void dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                     uint32_t Indent = 0);
  LLVM_ABI void dump(raw_ostream &OS, const LineTable &LT,
                     uint32_t Indent = 0);
  LLVM_ABI void dump(raw_ostream &OS, const InlineInfo &II,
                     uint32_t Indent = 0);
  LLVM_ABI void dump(raw_ostream &OS, std::optional<FileEntry> FE);
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADER_H
