//===- YAMLOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YAMLOutputStyle.h"

#include "PdbYaml.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"

using namespace llvm;
using namespace llvm::pdb;

YAMLOutputStyle::YAMLOutputStyle(PDBFile &File) : File(File), Out(outs()) {}

Error YAMLOutputStyle::dumpFileHeaders() {
  if (!opts::DumpHeaders)
    return Error::success();

  yaml::MsfHeaders Headers;
  Headers.BlockCount = File.getBlockCount();
  Headers.BlockMapIndex = File.getBlockMapIndex();
  Headers.BlockMapOffset = File.getBlockMapOffset();
  Headers.BlockSize = File.getBlockSize();
  auto Blocks = File.getDirectoryBlockArray();
  Headers.DirectoryBlocks.assign(Blocks.begin(), Blocks.end());
  Headers.NumDirectoryBlocks = File.getNumDirectoryBlocks();
  Headers.NumDirectoryBytes = File.getNumDirectoryBytes();
  Headers.NumStreams = File.getNumStreams();
  Headers.Unknown0 = File.getUnknown0();
  Headers.Unknown1 = File.getUnknown1();

  Obj.Headers.emplace(Headers);

  return Error::success();
}

Error YAMLOutputStyle::dumpStreamSummary() {
  if (!opts::DumpStreamSummary)
    return Error::success();

  std::vector<yaml::StreamSizeEntry> Sizes;
  for (uint32_t I = 0; I < File.getNumStreams(); ++I) {
    yaml::StreamSizeEntry Entry;
    Entry.Size = File.getStreamByteSize(I);
    Sizes.push_back(Entry);
  }
  Obj.StreamSizes.emplace(Sizes);
  return Error::success();
}

Error YAMLOutputStyle::dumpStreamBlocks() {
  if (!opts::DumpStreamBlocks)
    return Error::success();

  std::vector<yaml::StreamMapEntry> Blocks;
  for (uint32_t I = 0; I < File.getNumStreams(); ++I) {
    yaml::StreamMapEntry Entry;
    auto BlockList = File.getStreamBlockList(I);
    Entry.Blocks.assign(BlockList.begin(), BlockList.end());
    Blocks.push_back(Entry);
  }
  Obj.StreamMap.emplace(Blocks);

  return Error::success();
}

Error YAMLOutputStyle::dumpStreamData() {
  uint32_t StreamCount = File.getNumStreams();
  StringRef DumpStreamStr = opts::DumpStreamDataIdx;
  uint32_t DumpStreamNum;
  if (DumpStreamStr.getAsInteger(/*Radix=*/0U, DumpStreamNum) ||
      DumpStreamNum >= StreamCount)
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpInfoStream() {
  if (!opts::DumpHeaders)
    return Error::success();
  return Error::success();
}

Error YAMLOutputStyle::dumpNamedStream() {
  if (opts::DumpStreamDataName.empty())
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpTpiStream(uint32_t StreamIdx) {
  return Error::success();
}

Error YAMLOutputStyle::dumpDbiStream() { return Error::success(); }

Error YAMLOutputStyle::dumpSectionContribs() {
  if (!opts::DumpSectionContribs)
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpSectionMap() {
  if (!opts::DumpSectionMap)
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpPublicsStream() {
  if (!opts::DumpPublics)
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpSectionHeaders() {
  if (!opts::DumpSectionHeaders)
    return Error::success();

  return Error::success();
}

Error YAMLOutputStyle::dumpFpoStream() {
  if (!opts::DumpFpo)
    return Error::success();

  return Error::success();
}

void YAMLOutputStyle::flush() {
  Out << Obj;
  outs().flush();
}
