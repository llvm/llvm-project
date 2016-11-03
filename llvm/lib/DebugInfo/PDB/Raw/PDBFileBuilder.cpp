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

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStreamBuilder.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::support;

PDBFileBuilder::PDBFileBuilder(BumpPtrAllocator &Allocator)
    : Allocator(Allocator) {}

Error PDBFileBuilder::initialize(uint32_t BlockSize) {
  auto ExpectedMsf = MSFBuilder::create(Allocator, BlockSize);
  if (!ExpectedMsf)
    return ExpectedMsf.takeError();
  Msf = llvm::make_unique<MSFBuilder>(std::move(*ExpectedMsf));
  return Error::success();
}

MSFBuilder &PDBFileBuilder::getMsfBuilder() { return *Msf; }

InfoStreamBuilder &PDBFileBuilder::getInfoBuilder() {
  if (!Info)
    Info = llvm::make_unique<InfoStreamBuilder>(*Msf);
  return *Info;
}

DbiStreamBuilder &PDBFileBuilder::getDbiBuilder() {
  if (!Dbi)
    Dbi = llvm::make_unique<DbiStreamBuilder>(*Msf);
  return *Dbi;
}

TpiStreamBuilder &PDBFileBuilder::getTpiBuilder() {
  if (!Tpi)
    Tpi = llvm::make_unique<TpiStreamBuilder>(*Msf, StreamTPI);
  return *Tpi;
}

TpiStreamBuilder &PDBFileBuilder::getIpiBuilder() {
  if (!Ipi)
    Ipi = llvm::make_unique<TpiStreamBuilder>(*Msf, StreamIPI);
  return *Ipi;
}

Expected<msf::MSFLayout> PDBFileBuilder::finalizeMsfLayout() const {
  if (Info) {
    if (auto EC = Info->finalizeMsfLayout())
      return std::move(EC);
  }
  if (Dbi) {
    if (auto EC = Dbi->finalizeMsfLayout())
      return std::move(EC);
  }
  if (Tpi) {
    if (auto EC = Tpi->finalizeMsfLayout())
      return std::move(EC);
  }
  if (Ipi) {
    if (auto EC = Ipi->finalizeMsfLayout())
      return std::move(EC);
  }

  return Msf->build();
}

Expected<std::unique_ptr<PDBFile>>
PDBFileBuilder::build(std::unique_ptr<msf::WritableStream> PdbFileBuffer) {
  auto ExpectedLayout = finalizeMsfLayout();
  if (!ExpectedLayout)
    return ExpectedLayout.takeError();

  auto File = llvm::make_unique<PDBFile>(std::move(PdbFileBuffer), Allocator);
  File->ContainerLayout = *ExpectedLayout;

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

  if (Tpi) {
    auto ExpectedTpi = Tpi->build(*File);
    if (!ExpectedTpi)
      return ExpectedTpi.takeError();
    File->Tpi = std::move(*ExpectedTpi);
  }

  if (Ipi) {
    auto ExpectedIpi = Ipi->build(*File);
    if (!ExpectedIpi)
      return ExpectedIpi.takeError();
    File->Ipi = std::move(*ExpectedIpi);
  }

  if (File->Info && File->Dbi && File->Info->getAge() != File->Dbi->getAge())
    return llvm::make_error<RawError>(
        raw_error_code::corrupt_file,
        "PDB Stream Age doesn't match Dbi Stream Age!");

  return std::move(File);
}

Error PDBFileBuilder::commit(StringRef Filename) {
  auto ExpectedLayout = finalizeMsfLayout();
  if (!ExpectedLayout)
    return ExpectedLayout.takeError();
  auto &Layout = *ExpectedLayout;

  uint64_t Filesize = Layout.SB->BlockSize * Layout.SB->NumBlocks;
  auto OutFileOrError = FileOutputBuffer::create(Filename, Filesize);
  if (OutFileOrError.getError())
    return llvm::make_error<pdb::GenericError>(generic_error_code::invalid_path,
                                               Filename);
  FileBufferByteStream Buffer(std::move(*OutFileOrError));
  StreamWriter Writer(Buffer);

  if (auto EC = Writer.writeObject(*Layout.SB))
    return EC;
  uint32_t BlockMapOffset =
      msf::blockToOffset(Layout.SB->BlockMapAddr, Layout.SB->BlockSize);
  Writer.setOffset(BlockMapOffset);
  if (auto EC = Writer.writeArray(Layout.DirectoryBlocks))
    return EC;

  auto DirStream =
      WritableMappedBlockStream::createDirectoryStream(Layout, Buffer);
  StreamWriter DW(*DirStream);
  if (auto EC =
          DW.writeInteger(static_cast<uint32_t>(Layout.StreamSizes.size())))
    return EC;

  if (auto EC = DW.writeArray(Layout.StreamSizes))
    return EC;

  for (const auto &Blocks : Layout.StreamMap) {
    if (auto EC = DW.writeArray(Blocks))
      return EC;
  }

  if (Info) {
    if (auto EC = Info->commit(Layout, Buffer))
      return EC;
  }

  if (Dbi) {
    if (auto EC = Dbi->commit(Layout, Buffer))
      return EC;
  }

  if (Tpi) {
    if (auto EC = Tpi->commit(Layout, Buffer))
      return EC;
  }

  if (Ipi) {
    if (auto EC = Ipi->commit(Layout, Buffer))
      return EC;
  }

  return Buffer.commit();
}
