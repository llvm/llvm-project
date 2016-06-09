//===- InfoStream.cpp - PDB Info Stream (Stream 1) Access -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/IndexedStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::pdb;

InfoStream::InfoStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

Error InfoStream::reload() {
  codeview::StreamReader Reader(*Stream);

  struct Header {
    support::ulittle32_t Version;
    support::ulittle32_t Signature;
    support::ulittle32_t Age;
    PDB_UniqueId Guid;
  };

  const Header *H;
  if (auto EC = Reader.readObject(H))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "PDB Stream does not contain a header."));

  if (H->Version < PdbRaw_ImplVer::PdbImplVC70)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported PDB stream version.");

  Version = H->Version;
  Signature = H->Signature;
  Age = H->Age;
  Guid = H->Guid;

  return NamedStreams.load(Reader);
}

uint32_t InfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.tryGetValue(Name, Result))
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
