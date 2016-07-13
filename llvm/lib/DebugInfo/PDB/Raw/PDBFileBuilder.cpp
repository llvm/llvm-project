//===- PDBFileBuilder.cpp - PDB File Creation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBFileBuilder.h"

#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/DebugInfo/CodeView/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

PDBFileBuilder::PDBFileBuilder(
    std::unique_ptr<codeview::StreamInterface> PdbFileBuffer)
    : File(llvm::make_unique<PDBFile>(std::move(PdbFileBuffer))) {}

Error PDBFileBuilder::setSuperBlock(const PDBFile::SuperBlock &B) {
  auto SB = static_cast<PDBFile::SuperBlock *>(
      File->Allocator.Allocate(sizeof(PDBFile::SuperBlock),
                               llvm::AlignOf<PDBFile::SuperBlock>::Alignment));
  ::memcpy(SB, &B, sizeof(PDBFile::SuperBlock));
  return File->setSuperBlock(SB);
}

void PDBFileBuilder::setStreamSizes(ArrayRef<support::ulittle32_t> S) {
  File->StreamSizes = S;
}

void PDBFileBuilder::setDirectoryBlocks(ArrayRef<support::ulittle32_t> D) {
  File->DirectoryBlocks = D;
}

void PDBFileBuilder::setStreamMap(
    const std::vector<ArrayRef<support::ulittle32_t>> &S) {
  File->StreamMap = S;
}

Error PDBFileBuilder::generateSimpleStreamMap() {
  if (File->StreamSizes.empty())
    return Error::success();

  static std::vector<std::vector<support::ulittle32_t>> StaticMap;
  File->StreamMap.clear();
  StaticMap.clear();

  // Figure out how many blocks are needed for all streams, and set the first
  // used block to the highest block so that we can write the rest of the
  // blocks contiguously.
  uint32_t TotalFileBlocks = File->getBlockCount();
  std::vector<support::ulittle32_t> ReservedBlocks;
  ReservedBlocks.push_back(support::ulittle32_t(0));
  ReservedBlocks.push_back(File->SB->BlockMapAddr);
  ReservedBlocks.insert(ReservedBlocks.end(), File->DirectoryBlocks.begin(),
                        File->DirectoryBlocks.end());

  uint32_t BlocksNeeded = 0;
  for (auto Size : File->StreamSizes)
    BlocksNeeded += File->bytesToBlocks(Size, File->getBlockSize());

  support::ulittle32_t NextBlock(TotalFileBlocks - BlocksNeeded -
                                 ReservedBlocks.size());

  StaticMap.resize(File->StreamSizes.size());
  for (uint32_t S = 0; S < File->StreamSizes.size(); ++S) {
    uint32_t Size = File->StreamSizes[S];
    uint32_t NumBlocks = File->bytesToBlocks(Size, File->getBlockSize());
    auto &ThisStream = StaticMap[S];
    for (uint32_t I = 0; I < NumBlocks;) {
      NextBlock += 1;
      if (std::find(ReservedBlocks.begin(), ReservedBlocks.end(), NextBlock) !=
          ReservedBlocks.end())
        continue;

      ++I;
      assert(NextBlock < File->getBlockCount());
      ThisStream.push_back(NextBlock);
    }
    File->StreamMap.push_back(ThisStream);
  }
  return Error::success();
}

InfoStreamBuilder &PDBFileBuilder::getInfoBuilder() {
  if (!Info)
    Info = llvm::make_unique<InfoStreamBuilder>(*File);
  return *Info;
}

DbiStreamBuilder &PDBFileBuilder::getDbiBuilder() {
  if (!Dbi)
    Dbi = llvm::make_unique<DbiStreamBuilder>(*File);
  return *Dbi;
}

Expected<std::unique_ptr<PDBFile>> PDBFileBuilder::build() {
  if (Info) {
    auto ExpectedInfo = Info->build();
    if (!ExpectedInfo)
      return ExpectedInfo.takeError();
    File->Info = std::move(*ExpectedInfo);
  }

  if (Dbi) {
    auto ExpectedDbi = Dbi->build();
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
