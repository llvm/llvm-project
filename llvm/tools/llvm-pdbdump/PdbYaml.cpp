//===- PdbYAML.cpp -------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PdbYaml.h"

#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"

using namespace llvm;
using namespace llvm::yaml;
using namespace llvm::pdb;
using namespace llvm::pdb::yaml;

namespace llvm {
namespace yaml {
template <> struct ScalarTraits<llvm::pdb::PDB_UniqueId> {
  static void output(const llvm::pdb::PDB_UniqueId &S, void *,
                     llvm::raw_ostream &OS) {
    OS << S;
  }

  static StringRef input(StringRef Scalar, void *Ctx,
                         llvm::pdb::PDB_UniqueId &S) {
    if (Scalar.size() != 38)
      return "GUID strings are 38 characters long";
    if (Scalar[0] != '{' || Scalar[37] != '}')
      return "GUID is not enclosed in {}";
    if (Scalar[9] != '-' || Scalar[14] != '-' || Scalar[19] != '-' ||
        Scalar[24] != '-')
      return "GUID sections are not properly delineated with dashes";

    char *OutBuffer = S.Guid;
    for (auto Iter = Scalar.begin(); Iter != Scalar.end();) {
      if (*Iter == '-' || *Iter == '{' || *Iter == '}') {
        ++Iter;
        continue;
      }
      uint8_t Value = (llvm::hexDigitValue(*Iter) << 4);
      ++Iter;
      Value |= llvm::hexDigitValue(*Iter);
      ++Iter;
      *OutBuffer++ = Value;
    }

    return "";
  }

  static bool mustQuote(StringRef Scalar) { return needsQuotes(Scalar); }
};
}
}

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
  IO.mapOptional("PdbStream", Obj.PdbStream);
}

void MappingTraits<PdbInfoStream>::mapping(IO &IO, PdbInfoStream &Obj) {
  IO.mapRequired("Age", Obj.Age);
  IO.mapRequired("Guid", Obj.Guid);
  IO.mapRequired("Signature", Obj.Signature);
  IO.mapRequired("Version", Obj.Version);
}