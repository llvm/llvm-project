//===- InfoStreamBuilder.cpp - PDB Info Stream Creation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"

#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/RawTypes.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

InfoStreamBuilder::InfoStreamBuilder()
    : Ver(PdbRaw_ImplVer::PdbImplVC70), Sig(-1), Age(0) {}

void InfoStreamBuilder::setVersion(PdbRaw_ImplVer V) { Ver = V; }

void InfoStreamBuilder::setSignature(uint32_t S) { Sig = S; }

void InfoStreamBuilder::setAge(uint32_t A) { Age = A; }

void InfoStreamBuilder::setGuid(PDB_UniqueId G) { Guid = G; }

NameMapBuilder &InfoStreamBuilder::getNamedStreamsBuilder() {
  return NamedStreams;
}

uint32_t InfoStreamBuilder::calculateSerializedLength() const {
  return sizeof(InfoStreamHeader) + NamedStreams.calculateSerializedLength();
}

Expected<std::unique_ptr<InfoStream>>
InfoStreamBuilder::build(PDBFile &File, const msf::WritableStream &Buffer) {
  auto StreamData = MappedBlockStream::createIndexedStream(File.getMsfLayout(),
                                                           Buffer, StreamPDB);
  auto Info = llvm::make_unique<InfoStream>(std::move(StreamData));
  Info->Version = Ver;
  Info->Signature = Sig;
  Info->Age = Age;
  Info->Guid = Guid;
  auto NS = NamedStreams.build();
  if (!NS)
    return NS.takeError();
  Info->NamedStreams = **NS;
  return std::move(Info);
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
