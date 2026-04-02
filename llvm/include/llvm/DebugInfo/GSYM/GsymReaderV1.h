//===- GsymReaderV1.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H

#include "llvm/DebugInfo/GSYM/GsymReader.h"
#include "llvm/DebugInfo/GSYM/Header.h"

namespace llvm {
class MemoryBuffer;

namespace gsym {

/// GsymReaderV1 reads GSYM V1 data from a file or buffer.
class GsymReaderV1 : public GsymReader {
  GsymReaderV1(std::unique_ptr<MemoryBuffer> Buffer);
  llvm::Error parse();

  const Header *Hdr = nullptr;
  ArrayRef<FileEntry> Files;
  ArrayRef<uint32_t> AddrInfoOffsets;
  struct SwappedData {
    Header Hdr;
    std::vector<uint8_t> AddrOffsets;
    std::vector<uint32_t> AddrInfoOffsets;
    std::vector<FileEntry> Files;
  };
  std::unique_ptr<SwappedData> Swap;

  /// Get the address for a given index using direct typed array access.
  /// This is the fast path for V1 — reinterpret_cast to a typed array and
  /// index directly, avoiding DataExtractor construction per call.
  template <class T>
  std::optional<uint64_t> addressForIndex(size_t Index) const {
    ArrayRef<T> AIO = getAddrOffsets<T>();
    if (Index < AIO.size())
      return static_cast<uint64_t>(AIO[Index]) + Hdr->BaseAddress;
    return std::nullopt;
  }

  /// Get the address offset index for a given address offset using typed
  /// array binary search (power-of-two sizes only).
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

  LLVM_ABI static llvm::Expected<GsymReaderV1>
  create(std::unique_ptr<MemoryBuffer> &MemBuffer);

public:
  LLVM_ABI GsymReaderV1(GsymReaderV1 &&RHS);
  LLVM_ABI ~GsymReaderV1() override;

  LLVM_ABI static llvm::Expected<GsymReaderV1> openFile(StringRef Path);
  LLVM_ABI static llvm::Expected<GsymReaderV1> copyBuffer(StringRef Bytes);

  LLVM_ABI const Header &getHeader() const;

  // Header accessors
  uint64_t getBaseAddress() const override { return getHeader().BaseAddress; }
  uint64_t getNumAddresses() const override { return getHeader().NumAddresses; }
  uint64_t getAddressOffsetByteSize() const override {
    return getHeader().AddrOffSize;
  }
  uint64_t getAddressInfoOffsetByteSize() const override { return 4; }
  uint64_t getStringOffsetByteSize() const override { return 4; }

  LLVM_ABI std::optional<uint64_t> getAddress(size_t Index) const override;

  std::optional<FileEntry> getFile(uint32_t Index) const override {
    if (Index < Files.size())
      return Files[Index];
    return std::nullopt;
  }

  LLVM_ABI Expected<uint64_t>
  getAddressIndex(const uint64_t Addr) const override;

  // GlobalData accessors
  uint64_t getAddressInfoOffset(size_t Index) const override;

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
