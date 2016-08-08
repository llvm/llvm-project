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
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/YAMLTraits.h"

#include <vector>

namespace llvm {
namespace pdb {

namespace yaml {
struct MSFHeaders {
  msf::SuperBlock SuperBlock;
  uint32_t NumDirectoryBlocks;
  std::vector<uint32_t> DirectoryBlocks;
  uint32_t NumStreams;
  uint32_t FileSize;
};

struct StreamBlockList {
  std::vector<uint32_t> Blocks;
};

struct NamedStreamMapping {
  StringRef StreamName;
  uint32_t StreamNumber;
};

struct PdbInfoStream {
  PdbRaw_ImplVer Version;
  uint32_t Signature;
  uint32_t Age;
  PDB_UniqueId Guid;
  std::vector<NamedStreamMapping> NamedStreams;
};

struct PdbDbiModuleInfo {
  StringRef Obj;
  StringRef Mod;
  std::vector<StringRef> SourceFiles;
};

struct PdbDbiStream {
  PdbRaw_DbiVer VerHeader;
  uint32_t Age;
  uint16_t BuildNumber;
  uint32_t PdbDllVersion;
  uint16_t PdbDllRbld;
  uint16_t Flags;
  PDB_Machine MachineType;

  std::vector<PdbDbiModuleInfo> ModInfos;
};

struct PdbObject {
  Optional<MSFHeaders> Headers;
  Optional<std::vector<uint32_t>> StreamSizes;
  Optional<std::vector<StreamBlockList>> StreamMap;
  Optional<PdbInfoStream> PdbStream;
  Optional<PdbDbiStream> DbiStream;
};
}
}
}

namespace llvm {
namespace yaml {

template <> struct MappingTraits<pdb::yaml::PdbObject> {
  static void mapping(IO &IO, pdb::yaml::PdbObject &Obj);
};

template <> struct MappingTraits<pdb::yaml::MSFHeaders> {
  static void mapping(IO &IO, pdb::yaml::MSFHeaders &Obj);
};

template <> struct MappingTraits<msf::SuperBlock> {
  static void mapping(IO &IO, msf::SuperBlock &SB);
};

template <> struct MappingTraits<pdb::yaml::StreamBlockList> {
  static void mapping(IO &IO, pdb::yaml::StreamBlockList &SB);
};

template <> struct MappingTraits<pdb::yaml::PdbInfoStream> {
  static void mapping(IO &IO, pdb::yaml::PdbInfoStream &Obj);
};

template <> struct MappingTraits<pdb::yaml::PdbDbiStream> {
  static void mapping(IO &IO, pdb::yaml::PdbDbiStream &Obj);
};

template <> struct MappingTraits<pdb::yaml::NamedStreamMapping> {
  static void mapping(IO &IO, pdb::yaml::NamedStreamMapping &Obj);
};

template <> struct MappingTraits<pdb::yaml::PdbDbiModuleInfo> {
  static void mapping(IO &IO, pdb::yaml::PdbDbiModuleInfo &Obj);
};
}
}

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::StringRef)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::NamedStreamMapping)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbDbiModuleInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::StreamBlockList)

#endif // LLVM_TOOLS_LLVMPDBDUMP_PDBYAML_H
