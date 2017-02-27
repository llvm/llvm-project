//===- TpiStreamBuilder.h - PDB Tpi Stream Creation -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBTPISTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBTPISTREAMBUILDER_H

#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryItemStream.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace llvm {
class BinaryByteStream;
class BinaryStreamRef;
class WritableBinaryStream;

namespace codeview {
class TypeRecord;
}

template <> struct BinaryItemTraits<llvm::codeview::CVType> {
  static size_t length(const codeview::CVType &Item) { return Item.length(); }
  static ArrayRef<uint8_t> bytes(const codeview::CVType &Item) {
    return Item.data();
  }
};

namespace msf {
class MSFBuilder;
struct MSFLayout;
}
namespace pdb {
class PDBFile;
class TpiStream;
struct TpiStreamHeader;

class TpiStreamBuilder {
public:
  explicit TpiStreamBuilder(msf::MSFBuilder &Msf, uint32_t StreamIdx);
  ~TpiStreamBuilder();

  TpiStreamBuilder(const TpiStreamBuilder &) = delete;
  TpiStreamBuilder &operator=(const TpiStreamBuilder &) = delete;

  void setVersionHeader(PdbRaw_TpiVer Version);
  void addTypeRecord(const codeview::CVType &Record);

  Error finalizeMsfLayout();

  Error commit(const msf::MSFLayout &Layout, WritableBinaryStreamRef Buffer);

  uint32_t calculateSerializedLength();

private:
  uint32_t calculateHashBufferSize() const;
  Error finalize();

  msf::MSFBuilder &Msf;
  BumpPtrAllocator &Allocator;

  Optional<PdbRaw_TpiVer> VerHeader;
  std::vector<codeview::CVType> TypeRecords;
  BinaryItemStream<codeview::CVType> TypeRecordStream;
  uint32_t HashStreamIndex = kInvalidStreamIndex;
  std::unique_ptr<BinaryByteStream> HashValueStream;

  const TpiStreamHeader *Header;
  uint32_t Idx;
};
}
}

#endif
