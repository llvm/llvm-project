//===- GlobalsStream.h - PDB Index of Symbols by Name ------ ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_GLOBALS_STREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_GLOBALS_STREAM_H

#include "llvm/DebugInfo/MSF/BinaryStreamArray.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class DbiStream;
class PDBFile;

class GlobalsStream {
public:
  explicit GlobalsStream(std::unique_ptr<msf::MappedBlockStream> Stream);
  ~GlobalsStream();
  Error commit();
  msf::FixedStreamArray<support::ulittle32_t> getHashBuckets() const {
    return HashBuckets;
  }
  uint32_t getNumBuckets() const { return NumBuckets; }
  Error reload();

private:
  msf::FixedStreamArray<support::ulittle32_t> HashBuckets;
  msf::FixedStreamArray<PSHashRecord> HashRecords;
  uint32_t NumBuckets;
  std::unique_ptr<msf::MappedBlockStream> Stream;
};
}
}

#endif
