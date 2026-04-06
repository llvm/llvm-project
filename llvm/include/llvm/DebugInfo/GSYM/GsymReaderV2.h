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
  std::unique_ptr<HeaderV2> SwappedHdr;
  /// Parsed GlobalData section descriptors, keyed by type.
  std::map<GlobalInfoType, GlobalData> GlobalDataSections;

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
  uint64_t getAddressInfoOffsetByteSize() const override { return 8; }
  uint64_t getStringOffsetByteSize() const override { return 8; }

  // GlobalData accessors
  uint64_t getAddressInfoOffset(size_t Index) const override;

  using GsymReader::dump;
  LLVM_ABI void dump(raw_ostream &OS) override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADERV2_H
