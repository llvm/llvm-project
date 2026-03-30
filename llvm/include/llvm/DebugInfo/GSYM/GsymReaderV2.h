//===- GsymReaderV2.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H

#include "llvm/DebugInfo/GSYM/GsymReader.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"

namespace llvm {
class MemoryBuffer;

namespace gsym {

/// GsymReaderV2 reads GSYM V2 data from a file or buffer.
class GsymReaderV2 : public GsymReader {
  GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer);
  llvm::Error parse();

  const HeaderV2 *Hdr = nullptr;
  struct SwappedData {
    HeaderV2 Hdr;
    std::vector<uint8_t> AddrOffsets;
    std::vector<uint32_t> AddrInfoOffsets;
    std::vector<FileEntry> Files;
  };
  std::unique_ptr<SwappedData> Swap;

  LLVM_ABI static llvm::Expected<GsymReaderV2>
  create(std::unique_ptr<MemoryBuffer> &MemBuffer);

public:
  LLVM_ABI GsymReaderV2(GsymReaderV2 &&RHS);
  LLVM_ABI ~GsymReaderV2() override;

  LLVM_ABI static llvm::Expected<GsymReaderV2> openFile(StringRef Path);
  LLVM_ABI static llvm::Expected<GsymReaderV2> copyBuffer(StringRef Bytes);

  LLVM_ABI const HeaderV2 &getHeader() const;

  uint64_t getBaseAddress() const override { return getHeader().BaseAddress; }
  uint64_t getNumAddresses() const override { return getHeader().NumAddresses; }
  uint64_t getAddressOffsetByteSize() const override { return getHeader().AddrOffSize; }
  uint64_t getAddressInfoOffsetByteSize() const override { return getHeader().AddrInfoOffSize; }
  uint64_t getStringOffsetByteSize() const override { return getHeader().StrpSize; }

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
