//===- IndexedStreamData.h - Standard PDB Stream Data -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_INDEXEDSTREAMDATA_H
#define LLVM_DEBUGINFO_PDB_RAW_INDEXEDSTREAMDATA_H

#include "llvm/DebugInfo/PDB/Raw/IPDBStreamData.h"

namespace llvm {
namespace pdb {
class IPDBFile;

class IndexedStreamData : public IPDBStreamData {
public:
  IndexedStreamData(uint32_t StreamIdx, const IPDBFile &File);
  virtual ~IndexedStreamData() {}

  uint32_t getLength() override;
  ArrayRef<support::ulittle32_t> getStreamBlocks() override;

private:
  uint32_t StreamIdx;
  const IPDBFile &File;
};
}
}

#endif
