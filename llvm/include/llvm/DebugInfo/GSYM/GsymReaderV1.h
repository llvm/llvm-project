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
  std::unique_ptr<Header> SwappedHdr;

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
  uint8_t getAddressOffsetSize() const override {
    return getHeader().AddrOffSize;
  }
  uint8_t getAddressInfoOffsetSize() const override { return 4; }
  uint8_t getStringOffsetSize() const override { return 4; }

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV1_H
