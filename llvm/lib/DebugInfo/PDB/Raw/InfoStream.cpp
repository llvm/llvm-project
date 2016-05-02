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
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;

InfoStream::InfoStream(PDBFile &File) : Pdb(File), Stream(StreamPDB, File) {}

std::error_code InfoStream::reload() {
  StreamReader Reader(Stream);

  struct Header {
    support::ulittle32_t Version;
    support::ulittle32_t Signature;
    support::ulittle32_t Age;
    PDB_UniqueId Guid;
  };

  Header H;
  Reader.readObject(&H);

  if (H.Version < PdbRaw_ImplVer::PdbImplVC70)
    return std::make_error_code(std::errc::not_supported);

  Version = H.Version;
  Signature = H.Signature;
  Age = H.Age;
  Guid = H.Guid;

  NamedStreams.load(Reader);

  return std::error_code();
}

uint32_t InfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.tryGetValue(Name, Result))
    return 0;
  return Result;
}

PdbRaw_ImplVer InfoStream::getVersion() const {
  return static_cast<PdbRaw_ImplVer>(Version);
}

uint32_t InfoStream::getSignature() const { return Signature; }

uint32_t InfoStream::getAge() const { return Age; }

PDB_UniqueId InfoStream::getGuid() const { return Guid; }
