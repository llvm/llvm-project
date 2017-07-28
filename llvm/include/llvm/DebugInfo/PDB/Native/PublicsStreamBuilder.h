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

#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {

template <> struct BinaryItemTraits<codeview::CVSymbol> {
  static size_t length(const codeview::CVSymbol &Item) {
    return Item.RecordData.size();
  }
  static ArrayRef<uint8_t> bytes(const codeview::CVSymbol &Item) {
    return Item.RecordData;
  }
};

namespace msf {
class MSFBuilder;
}
namespace pdb {
class PublicsStream;
struct PublicsStreamHeader;

struct GSIHashTableBuilder {
  void addSymbols(ArrayRef<codeview::CVSymbol> Symbols);

  std::vector<PSHashRecord> HashRecords;
  std::array<support::ulittle32_t, (IPHR_HASH + 32) / 32> HashBitmap;
  std::vector<support::ulittle32_t> HashBuckets;
};

class PublicsStreamBuilder {
public:
  explicit PublicsStreamBuilder(msf::MSFBuilder &Msf);
  ~PublicsStreamBuilder();

  PublicsStreamBuilder(const PublicsStreamBuilder &) = delete;
  PublicsStreamBuilder &operator=(const PublicsStreamBuilder &) = delete;

  Error finalizeMsfLayout();
  uint32_t calculateSerializedLength() const;

  Error commit(BinaryStreamWriter &PublicsWriter,
               BinaryStreamWriter &RecWriter);

  uint32_t getStreamIndex() const { return StreamIdx; }
  uint32_t getRecordStreamIdx() const { return RecordStreamIdx; }

  void addPublicSymbol(const codeview::PublicSym32 &Pub);

private:
  uint32_t StreamIdx = kInvalidStreamIndex;
  uint32_t RecordStreamIdx = kInvalidStreamIndex;
  std::unique_ptr<GSIHashTableBuilder> Table;
  std::vector<codeview::CVSymbol> Publics;
  msf::MSFBuilder &Msf;
};
} // namespace pdb
} // namespace llvm

#endif
