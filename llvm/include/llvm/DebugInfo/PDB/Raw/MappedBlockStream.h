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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace pdb {

class PDBFile;

class MappedBlockStream : public codeview::StreamInterface {
public:
  MappedBlockStream(uint32_t StreamIdx, const PDBFile &File);

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override;

  uint32_t getLength() const override { return StreamLength; }

private:
  Error readBytes(uint32_t Offset, MutableArrayRef<uint8_t> Buffer) const;
  bool tryReadContiguously(uint32_t Offset, uint32_t Size,
                           ArrayRef<uint8_t> &Buffer) const;

  uint32_t StreamLength;
  std::vector<uint32_t> BlockList;
  mutable llvm::BumpPtrAllocator Pool;
  mutable DenseMap<uint32_t, uint8_t *> CacheMap;
  const PDBFile &Pdb;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_MAPPEDBLOCKSTREAM_H
