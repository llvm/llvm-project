//===- MappedBlockStream.h - Reads stream data from a PDBFile ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H

#include "llvm/DebugInfo/PDB/Raw/StreamInterface.h"

#include <stdint.h>

#include <system_error>
#include <vector>

namespace llvm {
namespace pdb {
class PDBFile;

class MappedBlockStream : public StreamInterface {
public:
  MappedBlockStream(uint32_t StreamIdx, const PDBFile &File);

  std::error_code readBytes(uint32_t Offset,
                            MutableArrayRef<uint8_t> Buffer) const override;
  uint32_t getLength() const override { return StreamLength; }

private:
  uint32_t StreamLength;
  std::vector<uint32_t> BlockList;
  const PDBFile &Pdb;
};
}
}

#endif
