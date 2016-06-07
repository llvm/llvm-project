//===- PdbYAML.h ---------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_PDBYAML_H
#define LLVM_TOOLS_LLVMPDBDUMP_PDBYAML_H

#include "OutputStyle.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/YAMLTraits.h"

#include <vector>

namespace llvm {
namespace pdb {
class PDBFile;

namespace yaml {
struct MsfHeaders {
  uint32_t BlockSize;
  uint32_t Unknown0;
  uint32_t BlockCount;
  uint32_t NumDirectoryBytes;
  uint32_t Unknown1;
  uint32_t BlockMapIndex;
  uint32_t NumDirectoryBlocks;
  uint32_t BlockMapOffset;
  std::vector<uint32_t> DirectoryBlocks;
  uint32_t NumStreams;
};

struct StreamSizeEntry {
  uint32_t Size;
};

struct StreamMapEntry {
  std::vector<uint32_t> Blocks;
};

struct PdbObject {
  Optional<MsfHeaders> Headers;
  Optional<std::vector<StreamSizeEntry>> StreamSizes;
  Optional<std::vector<StreamMapEntry>> StreamMap;
};
}
}
}

namespace llvm {
namespace yaml {
template <> struct MappingTraits<pdb::yaml::StreamSizeEntry> {
  static void mapping(IO &IO, pdb::yaml::StreamSizeEntry &Obj);
};

template <> struct MappingTraits<pdb::yaml::StreamMapEntry> {
  static void mapping(IO &IO, pdb::yaml::StreamMapEntry &Obj);
};

template <> struct MappingTraits<pdb::yaml::MsfHeaders> {
  static void mapping(IO &IO, pdb::yaml::MsfHeaders &Obj);
};
template <> struct MappingTraits<pdb::yaml::PdbObject> {
  static void mapping(IO &IO, pdb::yaml::PdbObject &Obj);
};
}
}

LLVM_YAML_IS_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::StreamSizeEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::StreamMapEntry)

#endif // LLVM_TOOLS_LLVMPDBDUMP_PDBYAML_H
