//===- InfoStreamBuilder.cpp - PDB Info Stream Creation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"

#include "llvm/DebugInfo/CodeView/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

InfoStreamBuilder::InfoStreamBuilder() {}

void InfoStreamBuilder::setVersion(PdbRaw_ImplVer V) { Ver = V; }

void InfoStreamBuilder::setSignature(uint32_t S) { Sig = S; }

void InfoStreamBuilder::setAge(uint32_t A) { Age = A; }

void InfoStreamBuilder::setGuid(PDB_UniqueId G) { Guid = G; }

NameMapBuilder &InfoStreamBuilder::getNamedStreamsBuilder() {
  return NamedStreams;
}

uint32_t InfoStreamBuilder::calculateSerializedLength() const {
  return sizeof(InfoStream::HeaderInfo) +
         NamedStreams.calculateSerializedLength();
}

Expected<std::unique_ptr<InfoStream>> InfoStreamBuilder::build(PDBFile &File) {
  if (!Ver.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing PDB Stream Version");
  if (!Sig.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing PDB Stream Signature");
  if (!Age.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing PDB Stream Age");
  if (!Guid.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing PDB Stream Guid");

  auto InfoS = MappedBlockStream::createIndexedStream(StreamPDB, File);
  if (!InfoS)
    return InfoS.takeError();
  auto Info = llvm::make_unique<InfoStream>(std::move(*InfoS));
  Info->Version = *Ver;
  Info->Signature = *Sig;
  Info->Age = *Age;
  Info->Guid = *Guid;
  auto NS = NamedStreams.build();
  if (!NS)
    return NS.takeError();
  Info->NamedStreams = **NS;
  return std::move(Info);
}
