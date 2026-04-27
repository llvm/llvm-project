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

/// GsymReaderV1 reads GSYM V1 data from a buffer.
class GsymReaderV1 : public GsymReader {
  friend class GsymReader;
  const Header *Hdr = nullptr;
  std::unique_ptr<Header> SwappedHdr;

protected:
  GsymReaderV1(std::unique_ptr<MemoryBuffer> Buffer, llvm::endianness Endian);
  llvm::Error parseHeaderAndGlobalDataEntries() override;

public:
  GsymReaderV1(GsymReaderV1 &&RHS) = default;
  ~GsymReaderV1() override = default;

  // Header accessors
  uint16_t getVersion() const override { return Header::getVersion(); }
  uint64_t getBaseAddress() const override { return Hdr->BaseAddress; }
  uint64_t getNumAddresses() const override { return Hdr->NumAddresses; }
  uint8_t getAddressOffsetSize() const override { return Hdr->AddrOffSize; }
  uint8_t getAddressInfoOffsetSize() const override {
    return Header::getAddressInfoOffsetSize();
  }
  uint8_t getStringOffsetSize() const override {
    return Header::getStringOffsetSize();
  }

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
