//===- GsymReaderV2.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/GsymReader.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"

#include <map>

namespace llvm {
class MemoryBuffer;

namespace gsym {

/// GsymReaderV2 reads GSYM V2 data from a file or buffer.
class GsymReaderV2 : public GsymReader {
  GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer);
  llvm::Error parse();

  const HeaderV2 *Hdr = nullptr;
  /// Parsed GlobalData section descriptors, keyed by type.
  std::map<GlobalInfoType, GlobalData> GlobalDataSections;
  /// DataExtractor for on-demand addr info offset decoding. Contains exactly
  /// the AddrInfoOffsets section data.
  DataExtractor AddrInfoOffsetsData;
  /// DataExtractor for on-demand file table decoding. Contains exactly the
  /// file entry data (does not include the NumFiles uint32_t).
  DataExtractor FileData;
  struct SwappedData {
    HeaderV2 Hdr;
    std::vector<uint8_t> AddrOffsets;
  };
  std::unique_ptr<SwappedData> Swap;

  /// A random access iterator over a byte array of fixed-size unsigned
  /// integers. Supports any element byte size from 1 to 8, enabling
  /// std::lower_bound() to binary search address offset tables without
  /// requiring power-of-two element sizes.
  class AddrOffsetIterator {
    ArrayRef<uint8_t> Data;
    uint8_t ByteSize;
    size_t Index;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = uint64_t;
    using difference_type = int64_t;
    using pointer = const uint64_t *;
    using reference = uint64_t;

    AddrOffsetIterator(ArrayRef<uint8_t> Data, uint8_t ByteSize, size_t Index)
        : Data(Data), ByteSize(ByteSize), Index(Index) {}

    uint64_t operator*() const {
      return getUnsigned(Data, ByteSize, Index).value_or(0);
    }
    uint64_t operator[](difference_type N) const {
      return getUnsigned(Data, ByteSize, Index + N).value_or(0);
    }
    AddrOffsetIterator &operator++() {
      ++Index;
      return *this;
    }
    AddrOffsetIterator operator++(int) {
      auto T = *this;
      ++Index;
      return T;
    }
    AddrOffsetIterator &operator--() {
      --Index;
      return *this;
    }
    AddrOffsetIterator operator--(int) {
      auto T = *this;
      --Index;
      return T;
    }
    AddrOffsetIterator &operator+=(difference_type N) {
      Index += N;
      return *this;
    }
    AddrOffsetIterator &operator-=(difference_type N) {
      Index -= N;
      return *this;
    }
    AddrOffsetIterator operator+(difference_type N) const {
      return AddrOffsetIterator(Data, ByteSize, Index + N);
    }
    AddrOffsetIterator operator-(difference_type N) const {
      return AddrOffsetIterator(Data, ByteSize, Index - N);
    }
    difference_type operator-(const AddrOffsetIterator &RHS) const {
      return static_cast<difference_type>(Index) -
             static_cast<difference_type>(RHS.Index);
    }
    bool operator==(const AddrOffsetIterator &RHS) const {
      return Index == RHS.Index;
    }
    bool operator!=(const AddrOffsetIterator &RHS) const {
      return Index != RHS.Index;
    }
    bool operator<(const AddrOffsetIterator &RHS) const {
      return Index < RHS.Index;
    }
    bool operator>(const AddrOffsetIterator &RHS) const {
      return Index > RHS.Index;
    }
    bool operator<=(const AddrOffsetIterator &RHS) const {
      return Index <= RHS.Index;
    }
    bool operator>=(const AddrOffsetIterator &RHS) const {
      return Index >= RHS.Index;
    }

    size_t getIndex() const { return Index; }
  };

  LLVM_ABI static llvm::Expected<GsymReaderV2>
  create(std::unique_ptr<MemoryBuffer> &MemBuffer);

public:
  LLVM_ABI GsymReaderV2(GsymReaderV2 &&RHS);
  LLVM_ABI ~GsymReaderV2() override;

  LLVM_ABI static llvm::Expected<GsymReaderV2> openFile(StringRef Path);
  LLVM_ABI static llvm::Expected<GsymReaderV2> copyBuffer(StringRef Bytes);

  LLVM_ABI const HeaderV2 &getHeader() const;

  // Header accessors
  uint64_t getBaseAddress() const override { return getHeader().BaseAddress; }
  uint64_t getNumAddresses() const override { return getHeader().NumAddresses; }
  uint64_t getAddressOffsetByteSize() const override {
    return getHeader().AddrOffSize;
  }
  uint64_t getAddressInfoOffsetByteSize() const override {
    return getHeader().AddrInfoOffSize;
  }
  uint64_t getStringOffsetByteSize() const override {
    return getHeader().StrpSize;
  }

  LLVM_ABI std::optional<uint64_t> getAddress(size_t Index) const override;

  LLVM_ABI Expected<uint64_t>
  getAddressIndex(const uint64_t Addr) const override;

  std::optional<FileEntry> getFile(uint32_t Index) const override;

  // GlobalData accessors
  uint64_t getAddressInfoOffset(size_t Index) const override;

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
