//===- PublicsStreamBuilder.h - PDB Publics Stream Creation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBPUBLICSTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBPUBLICSTREAMBUILDER_H

#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace msf {
class MSFBuilder;
}
namespace pdb {
class PublicsStream;
struct PublicsStreamHeader;

class PublicsStreamBuilder {
public:
  explicit PublicsStreamBuilder(msf::MSFBuilder &Msf);
  ~PublicsStreamBuilder();

  PublicsStreamBuilder(const PublicsStreamBuilder &) = delete;
  PublicsStreamBuilder &operator=(const PublicsStreamBuilder &) = delete;

  Error finalizeMsfLayout();
  uint32_t calculateSerializedLength() const;

  Error commit(BinaryStreamWriter &PublicsWriter);

  uint32_t getStreamIndex() const { return StreamIdx; }
  uint32_t getRecordStreamIdx() const { return RecordStreamIdx; }

private:
  uint32_t StreamIdx = kInvalidStreamIndex;
  uint32_t RecordStreamIdx = kInvalidStreamIndex;
  std::vector<PSHashRecord> HashRecords;
  msf::MSFBuilder &Msf;
};
} // namespace pdb
} // namespace llvm

#endif
