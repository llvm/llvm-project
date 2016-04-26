//===- PDBInfoStream.cpp - PDB Info Stream (Stream 1) Access ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBInfoStream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

PDBInfoStream::PDBInfoStream(PDBFile &File) : Pdb(File), Stream1(1, File) {}

std::error_code PDBInfoStream::reload() {
  Stream1.setOffset(0);
  support::ulittle32_t Value;

  Stream1.readObject(&Value);
  Version = Value;
  if (Version < PdbRaw_ImplVer::PdbImplVC70)
    return std::make_error_code(std::errc::not_supported);

  Stream1.readObject(&Value);
  Signature = Value;

  Stream1.readObject(&Value);
  Age = Value;

  Stream1.readObject(&Guid);
  NamedStreams.load(Stream1);

  return std::error_code();
}

uint32_t PDBInfoStream::getNamedStreamIndex(llvm::StringRef Name) const {
  uint32_t Result;
  if (!NamedStreams.tryGetValue(Name, Result))
    return 0;
  return Result;
}

PdbRaw_ImplVer PDBInfoStream::getVersion() const {
  return static_cast<PdbRaw_ImplVer>(Version);
}

uint32_t PDBInfoStream::getSignature() const { return Signature; }

uint32_t PDBInfoStream::getAge() const { return Age; }

PDB_UniqueId PDBInfoStream::getGuid() const { return Guid; }
