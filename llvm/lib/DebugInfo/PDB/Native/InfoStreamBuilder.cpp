//===- InfoStreamBuilder.cpp - PDB Info Stream Creation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NamedStreamMap.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

InfoStreamBuilder::InfoStreamBuilder(msf::MSFBuilder &Msf,
                                     NamedStreamMap &NamedStreams)
    : Msf(Msf), Ver(PdbRaw_ImplVer::PdbImplVC70), Sig(-1), Age(0),
      NamedStreams(NamedStreams) {}

void InfoStreamBuilder::setVersion(PdbRaw_ImplVer V) { Ver = V; }

void InfoStreamBuilder::setSignature(uint32_t S) { Sig = S; }

void InfoStreamBuilder::setAge(uint32_t A) { Age = A; }

void InfoStreamBuilder::setGuid(PDB_UniqueId G) { Guid = G; }

Error InfoStreamBuilder::finalizeMsfLayout() {
  uint32_t Length = sizeof(InfoStreamHeader) + NamedStreams.finalize();
  if (auto EC = Msf.setStreamSize(StreamPDB, Length))
    return EC;
  return Error::success();
}

Error InfoStreamBuilder::commit(const msf::MSFLayout &Layout,
                                const msf::WritableStream &Buffer) const {
  auto InfoS =
      WritableMappedBlockStream::createIndexedStream(Layout, Buffer, StreamPDB);
  StreamWriter Writer(*InfoS);

  InfoStreamHeader H;
  H.Age = Age;
  H.Signature = Sig;
  H.Version = Ver;
  H.Guid = Guid;
  if (auto EC = Writer.writeObject(H))
    return EC;

  return NamedStreams.commit(Writer);
}
