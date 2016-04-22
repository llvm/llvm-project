//===- PDBStream.h - Low level interface to a PDB stream --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class MemoryBufferRef;
class PDBFile;

class PDBStream {
public:
  PDBStream(uint32_t StreamIdx, const PDBFile &File);

  std::error_code readInteger(uint32_t &Dest);
  std::error_code readZeroString(std::string &Dest);
  std::error_code readBytes(void *Dest, uint32_t Length);

  void setOffset(uint32_t Off);
  uint32_t getOffset() const;
  uint32_t getLength() const;

  template <typename T> std::error_code readObject(T *Dest) {
    return readBytes(reinterpret_cast<void *>(Dest), sizeof(T));
  }

private:
  uint32_t Offset;

  uint32_t StreamLength;
  std::vector<uint32_t> BlockList;
  const PDBFile &Pdb;
};
}

#endif
