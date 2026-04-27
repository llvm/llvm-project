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

/// GsymReaderV2 reads GSYM V2 data from a buffer.
class GsymReaderV2 : public GsymReader {
  friend class GsymReader;
  const HeaderV2 *Hdr = nullptr;
  std::unique_ptr<HeaderV2> SwappedHdr;

protected:
  GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer, llvm::endianness Endian);
  llvm::Error parseHeaderAndGlobalDataEntries() override;

public:
  GsymReaderV2(GsymReaderV2 &&RHS) = default;
  ~GsymReaderV2() override = default;

  // Header accessors
  uint16_t getVersion() const override { return HeaderV2::getVersion(); }
  uint64_t getBaseAddress() const override { return Hdr->BaseAddress; }
  uint64_t getNumAddresses() const override { return Hdr->NumAddresses; }
  uint8_t getAddressOffsetSize() const override { return Hdr->AddrOffSize; }
  uint8_t getAddressInfoOffsetSize() const override {
    return HeaderV2::getAddressInfoOffsetSize();
  }
  uint8_t getStringOffsetSize() const override {
    return HeaderV2::getStringOffsetSize();
  }

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
