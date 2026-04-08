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
  friend class GsymReader;
  const HeaderV2 *Hdr = nullptr;
  std::unique_ptr<HeaderV2> SwappedHdr;

protected:
  GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer, llvm::endianness Endian);
  llvm::Error parseHeaderAndGlobalDataDirectory() override;

public:
  LLVM_ABI GsymReaderV2(GsymReaderV2 &&RHS);
  LLVM_ABI ~GsymReaderV2() override;

  LLVM_ABI const HeaderV2 &getHeader() const;

  // Header accessors
  uint64_t getBaseAddress() const override { return getHeader().BaseAddress; }
  uint64_t getNumAddresses() const override { return getHeader().NumAddresses; }
  uint8_t getAddressOffsetSize() const override {
    return getHeader().AddrOffSize;
  }
  uint8_t getAddressInfoOffsetSize() const override { return 8; }
  uint8_t getStringOffsetSize() const override { return 8; }

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
