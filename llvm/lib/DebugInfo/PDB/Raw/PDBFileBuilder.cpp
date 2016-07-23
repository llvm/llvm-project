//===- PDBFileBuilder.cpp - PDB File Creation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBFileBuilder.h"

#include "llvm/ADT/BitVector.h"

#include "llvm/DebugInfo/Msf/MsfBuilder.h"
#include "llvm/DebugInfo/Msf/StreamInterface.h"
#include "llvm/DebugInfo/Msf/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::support;

PDBFileBuilder::PDBFileBuilder(BumpPtrAllocator &Allocator)
    : Allocator(Allocator) {}

Error PDBFileBuilder::initialize(const msf::SuperBlock &Super) {
  auto ExpectedMsf =
      MsfBuilder::create(Allocator, Super.BlockSize, Super.NumBlocks);
  if (!ExpectedMsf)
    return ExpectedMsf.takeError();

  auto &MsfResult = *ExpectedMsf;
  if (auto EC = MsfResult.setBlockMapAddr(Super.BlockMapAddr))
    return EC;
  Msf = llvm::make_unique<MsfBuilder>(std::move(MsfResult));
  Msf->setFreePageMap(Super.FreeBlockMapBlock);
  Msf->setUnknown1(Super.Unknown1);
  return Error::success();
}

MsfBuilder &PDBFileBuilder::getMsfBuilder() { return *Msf; }

InfoStreamBuilder &PDBFileBuilder::getInfoBuilder() {
  if (!Info)
    Info = llvm::make_unique<InfoStreamBuilder>();
  return *Info;
}

DbiStreamBuilder &PDBFileBuilder::getDbiBuilder() {
  if (!Dbi)
    Dbi = llvm::make_unique<DbiStreamBuilder>(Allocator);
  return *Dbi;
}

Expected<std::unique_ptr<PDBFile>>
PDBFileBuilder::build(std::unique_ptr<msf::StreamInterface> PdbFileBuffer) {
  if (Info) {
    uint32_t Length = Info->calculateSerializedLength();
    if (auto EC = Msf->setStreamSize(StreamPDB, Length))
      return std::move(EC);
  }
  if (Dbi) {
    uint32_t Length = Dbi->calculateSerializedLength();
    if (auto EC = Msf->setStreamSize(StreamDBI, Length))
      return std::move(EC);
  }

  auto ExpectedLayout = Msf->build();
  if (!ExpectedLayout)
    return ExpectedLayout.takeError();

  auto File = llvm::make_unique<PDBFile>(std::move(PdbFileBuffer), Allocator);
  File->MsfLayout = *ExpectedLayout;

  if (Info) {
    auto ExpectedInfo = Info->build(*File);
    if (!ExpectedInfo)
      return ExpectedInfo.takeError();
    File->Info = std::move(*ExpectedInfo);
  }

  if (Dbi) {
    auto ExpectedDbi = Dbi->build(*File);
    if (!ExpectedDbi)
      return ExpectedDbi.takeError();
    File->Dbi = std::move(*ExpectedDbi);
  }

  if (File->Info && File->Dbi && File->Info->getAge() != File->Dbi->getAge())
    return llvm::make_error<RawError>(
        raw_error_code::corrupt_file,
        "PDB Stream Age doesn't match Dbi Stream Age!");

  return std::move(File);
}
