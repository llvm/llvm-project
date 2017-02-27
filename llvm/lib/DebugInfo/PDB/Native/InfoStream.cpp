//===- InfoStream.cpp - PDB Info Stream (Stream 1) Access -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

InfoStream::InfoStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

Error InfoStream::reload() {
  BinaryStreamReader Reader(*Stream);

  const InfoStreamHeader *H;
  if (auto EC = Reader.readObject(H))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "PDB Stream does not contain a header."));

  switch (H->Version) {
  case PdbImplVC70:
  case PdbImplVC80:
  case PdbImplVC110:
  case PdbImplVC140:
    break;
  default:
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported PDB stream version.");
  }

  Version = H->Version;
  Signature = H->Signature;
  Age = H->Age;
  Guid = H->Guid;

  return NamedStreams.load(Reader);
}

uint32_t InfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.get(Name, Result))
    return 0;
  return Result;
}

iterator_range<StringMapConstIterator<uint32_t>>
InfoStream::named_streams() const {
  return NamedStreams.entries();
}

PdbRaw_ImplVer InfoStream::getVersion() const {
  return static_cast<PdbRaw_ImplVer>(Version);
}

uint32_t InfoStream::getSignature() const { return Signature; }

uint32_t InfoStream::getAge() const { return Age; }

PDB_UniqueId InfoStream::getGuid() const { return Guid; }

const NamedStreamMap &InfoStream::getNamedStreams() const {
  return NamedStreams;
}
