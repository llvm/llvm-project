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
  Obj.Headers.SuperBlock.NumBlocks = File.getBlockCount();
  Obj.Headers.SuperBlock.BlockMapAddr = File.getBlockMapIndex();
  Obj.Headers.BlockMapOffset = File.getBlockMapOffset();
  Obj.Headers.SuperBlock.BlockSize = File.getBlockSize();
  auto Blocks = File.getDirectoryBlockArray();
  Obj.Headers.DirectoryBlocks.assign(Blocks.begin(), Blocks.end());
  Obj.Headers.NumDirectoryBlocks = File.getNumDirectoryBlocks();
  Obj.Headers.SuperBlock.NumDirectoryBytes = File.getNumDirectoryBytes();
  Obj.Headers.NumStreams = File.getNumStreams();
  Obj.Headers.SuperBlock.Unknown0 = File.getUnknown0();
  Obj.Headers.SuperBlock.Unknown1 = File.getUnknown1();
  Obj.Headers.FileSize = File.getFileSize();

  return Error::success();
}

Error YAMLOutputStyle::dumpStreamSummary() {
  if (!opts::DumpStreamSummary)
    return Error::success();

  Obj.StreamSizes = File.getStreamSizes();
  return Error::success();
}

Error YAMLOutputStyle::dumpStreamBlocks() {
  if (!opts::DumpStreamBlocks)
    return Error::success();

  auto StreamMap = File.getStreamMap();
  Obj.StreamMap.emplace();
  for (auto &Stream : StreamMap) {
    pdb::yaml::StreamBlockList BlockList;
    BlockList.Blocks = Stream;
    Obj.StreamMap->push_back(BlockList);
  }

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
