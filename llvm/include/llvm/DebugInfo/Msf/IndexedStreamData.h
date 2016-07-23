//===- IndexedStreamData.h - Standard Msf Stream Data -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_INDEXEDSTREAMDATA_H
#define LLVM_DEBUGINFO_MSF_INDEXEDSTREAMDATA_H

#include "llvm/DebugInfo/Msf/IMsfStreamData.h"

namespace llvm {
namespace msf {
class IMsfFile;

class IndexedStreamData : public IMsfStreamData {
public:
  IndexedStreamData(uint32_t StreamIdx, const IMsfFile &File);
  virtual ~IndexedStreamData() {}

  uint32_t getLength() override;
  ArrayRef<support::ulittle32_t> getStreamBlocks() override;

private:
  uint32_t StreamIdx;
  const IMsfFile &File;
};
} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_INDEXEDSTREAMDATA_H
