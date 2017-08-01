//===- GlobalsStreamBuilder.h - PDB Globals Stream Creation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBGLOBALSTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBGLOBALSTREAMBUILDER_H

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
class GlobalsStream;

class GlobalsStreamBuilder {
public:
  explicit GlobalsStreamBuilder(msf::MSFBuilder &Msf);
  ~GlobalsStreamBuilder();

  GlobalsStreamBuilder(const GlobalsStreamBuilder &) = delete;
  GlobalsStreamBuilder &operator=(const GlobalsStreamBuilder &) = delete;

  Error finalizeMsfLayout();
  uint32_t calculateSerializedLength() const;

  Error commit(BinaryStreamWriter &PublicsWriter);

  uint32_t getStreamIndex() const { return StreamIdx; }

private:
  uint32_t StreamIdx = kInvalidStreamIndex;
  msf::MSFBuilder &Msf;
};
} // namespace pdb
} // namespace llvm

#endif
