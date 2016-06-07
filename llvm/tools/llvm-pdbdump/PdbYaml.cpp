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

void MappingTraits<MsfHeaders>::mapping(IO &IO, MsfHeaders &Obj) {
  IO.mapRequired("BlockSize", Obj.BlockSize);
  IO.mapRequired("Unknown0", Obj.Unknown0);
  IO.mapRequired("NumBlocks", Obj.BlockCount);
  IO.mapRequired("NumDirectoryBytes", Obj.NumDirectoryBytes);
  IO.mapRequired("Unknown1", Obj.Unknown1);
  IO.mapRequired("BlockMapAddr", Obj.BlockMapIndex);
  IO.mapRequired("NumDirectoryBlocks", Obj.NumDirectoryBlocks);
  IO.mapRequired("BlockMapOffset", Obj.BlockMapOffset);
  IO.mapRequired("DirectoryBlocks", Obj.DirectoryBlocks);
  IO.mapRequired("NumStreams", Obj.NumStreams);
}

void MappingTraits<PdbObject>::mapping(IO &IO, PdbObject &Obj) {
  IO.mapOptional("MSF", Obj.Headers);
  IO.mapOptional("StreamSizes", Obj.StreamSizes);
  IO.mapOptional("StreamMap", Obj.StreamMap);
}

void MappingTraits<StreamSizeEntry>::mapping(IO &IO, StreamSizeEntry &Obj) {
  IO.mapRequired("Size", Obj.Size);
}

void MappingTraits<StreamMapEntry>::mapping(IO &IO, StreamMapEntry &Obj) {
  IO.mapRequired("Blocks", Obj.Blocks);
}
