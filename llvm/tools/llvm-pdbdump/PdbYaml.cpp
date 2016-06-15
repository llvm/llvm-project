//===- PdbYAML.cpp -------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PdbYaml.h"

#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"

using namespace llvm;
using namespace llvm::yaml;
using namespace llvm::pdb;
using namespace llvm::pdb::yaml;

void MappingTraits<PDBFile::SuperBlock>::mapping(IO &IO,
                                                 PDBFile::SuperBlock &SB) {
  if (!IO.outputting()) {
    ::memcpy(SB.MagicBytes, MsfMagic, sizeof(MsfMagic));
  }

  IO.mapRequired("BlockSize", SB.BlockSize);
  IO.mapRequired("Unknown0", SB.Unknown0);
  IO.mapRequired("NumBlocks", SB.NumBlocks);
  IO.mapRequired("NumDirectoryBytes", SB.NumDirectoryBytes);
  IO.mapRequired("Unknown1", SB.Unknown1);
  IO.mapRequired("BlockMapAddr", SB.BlockMapAddr);
}

void MappingTraits<StreamBlockList>::mapping(IO &IO, StreamBlockList &SB) {
  IO.mapRequired("Stream", SB.Blocks);
}

void MappingTraits<MsfHeaders>::mapping(IO &IO, MsfHeaders &Obj) {
  IO.mapRequired("SuperBlock", Obj.SuperBlock);
  IO.mapRequired("NumDirectoryBlocks", Obj.NumDirectoryBlocks);
  IO.mapRequired("BlockMapOffset", Obj.BlockMapOffset);
  IO.mapRequired("DirectoryBlocks", Obj.DirectoryBlocks);
  IO.mapRequired("NumStreams", Obj.NumStreams);
  IO.mapRequired("FileSize", Obj.FileSize);
}

void MappingTraits<PdbObject>::mapping(IO &IO, PdbObject &Obj) {
  IO.mapOptional("MSF", Obj.Headers);
  IO.mapOptional("StreamSizes", Obj.StreamSizes);
  IO.mapOptional("StreamMap", Obj.StreamMap);
}
