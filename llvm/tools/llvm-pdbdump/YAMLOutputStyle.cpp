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

#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"

using namespace llvm;
using namespace llvm::pdb;

YAMLOutputStyle::YAMLOutputStyle(PDBFile &File) : File(File), Out(outs()) {}

Error YAMLOutputStyle::dump() {
  if (opts::pdb2yaml::StreamDirectory)
    opts::pdb2yaml::StreamMetadata = true;

  if (auto EC = dumpFileHeaders())
    return EC;

  if (auto EC = dumpStreamMetadata())
    return EC;

  if (auto EC = dumpStreamDirectory())
    return EC;

  if (auto EC = dumpPDBStream())
    return EC;

  if (auto EC = dumpDbiStream())
    return EC;

  flush();
  return Error::success();
}

Error YAMLOutputStyle::dumpFileHeaders() {
  if (opts::pdb2yaml::NoFileHeaders)
    return Error::success();

  yaml::MsfHeaders Headers;
  Obj.Headers.emplace();
  Obj.Headers->SuperBlock.NumBlocks = File.getBlockCount();
  Obj.Headers->SuperBlock.BlockMapAddr = File.getBlockMapIndex();
  Obj.Headers->SuperBlock.BlockSize = File.getBlockSize();
  auto Blocks = File.getDirectoryBlockArray();
  Obj.Headers->DirectoryBlocks.assign(Blocks.begin(), Blocks.end());
  Obj.Headers->NumDirectoryBlocks = File.getNumDirectoryBlocks();
  Obj.Headers->SuperBlock.NumDirectoryBytes = File.getNumDirectoryBytes();
  Obj.Headers->NumStreams =
      opts::pdb2yaml::StreamMetadata ? File.getNumStreams() : 0;
  Obj.Headers->SuperBlock.FreeBlockMapBlock = File.getFreeBlockMapBlock();
  Obj.Headers->SuperBlock.Unknown1 = File.getUnknown1();
  Obj.Headers->FileSize = File.getFileSize();

  return Error::success();
}

Error YAMLOutputStyle::dumpStreamMetadata() {
  if (!opts::pdb2yaml::StreamMetadata)
    return Error::success();

  Obj.StreamSizes.emplace();
  Obj.StreamSizes->assign(File.getStreamSizes().begin(),
                          File.getStreamSizes().end());
  return Error::success();
}

Error YAMLOutputStyle::dumpStreamDirectory() {
  if (!opts::pdb2yaml::StreamDirectory)
    return Error::success();

  auto StreamMap = File.getStreamMap();
  Obj.StreamMap.emplace();
  for (auto &Stream : StreamMap) {
    pdb::yaml::StreamBlockList BlockList;
    BlockList.Blocks.assign(Stream.begin(), Stream.end());
    Obj.StreamMap->push_back(BlockList);
  }

  return Error::success();
}

Error YAMLOutputStyle::dumpPDBStream() {
  if (!opts::pdb2yaml::PdbStream)
    return Error::success();

  auto IS = File.getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  auto &InfoS = IS.get();
  Obj.PdbStream.emplace();
  Obj.PdbStream->Age = InfoS.getAge();
  Obj.PdbStream->Guid = InfoS.getGuid();
  Obj.PdbStream->Signature = InfoS.getSignature();
  Obj.PdbStream->Version = InfoS.getVersion();
  for (auto &NS : InfoS.named_streams()) {
    yaml::NamedStreamMapping Mapping;
    Mapping.StreamName = NS.getKey();
    Mapping.StreamNumber = NS.getValue();
    Obj.PdbStream->NamedStreams.push_back(Mapping);
  }

  return Error::success();
}

Error YAMLOutputStyle::dumpDbiStream() {
  if (!opts::pdb2yaml::DbiStream)
    return Error::success();

  auto DbiS = File.getPDBDbiStream();
  if (!DbiS)
    return DbiS.takeError();

  auto &DS = DbiS.get();
  Obj.DbiStream.emplace();
  Obj.DbiStream->Age = DS.getAge();
  Obj.DbiStream->BuildNumber = DS.getBuildNumber();
  Obj.DbiStream->Flags = DS.getFlags();
  Obj.DbiStream->MachineType = DS.getMachineType();
  Obj.DbiStream->PdbDllRbld = DS.getPdbDllRbld();
  Obj.DbiStream->PdbDllVersion = DS.getPdbDllVersion();
  Obj.DbiStream->VerHeader = DS.getDbiVersion();
  return Error::success();
}

void YAMLOutputStyle::flush() {
  Out << Obj;
  outs().flush();
}
