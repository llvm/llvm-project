//===- MachOYAML.h - Mach-O YAMLIO implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares classes for handling the YAML representation
/// of Mach-O.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_MACHOYAML_H
#define LLVM_OBJECTYAML_MACHOYAML_H

#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/MachO.h"

namespace llvm {
namespace MachOYAML {

struct FileHeader {
  llvm::yaml::Hex32 magic;
  llvm::yaml::Hex32 cputype;
  llvm::yaml::Hex32 cpusubtype;
  llvm::yaml::Hex32 filetype;
  uint32_t ncmds;
  uint32_t sizeofcmds;
  llvm::yaml::Hex32 flags;
  llvm::yaml::Hex32 reserved;
};

struct LoadCommand {
  virtual ~LoadCommand();
  MachO::LoadCommandType cmd;
  uint32_t cmdsize;
};

struct Object {
  FileHeader Header;
  std::vector<std::unique_ptr<LoadCommand>> LoadCommands;
};

} // namespace llvm::MachOYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::MachOYAML::LoadCommand>)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<MachOYAML::FileHeader> {
  static void mapping(IO &IO, MachOYAML::FileHeader &FileHeader);
};

template <> struct MappingTraits<MachOYAML::Object> {
  static void mapping(IO &IO, MachOYAML::Object &Object);
};

template <> struct MappingTraits<std::unique_ptr<MachOYAML::LoadCommand>> {
  static void mapping(IO &IO,
                      std::unique_ptr<MachOYAML::LoadCommand> &LoadCommand);
};

#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                 \
  io.enumCase(value, #LCName, MachO::LCName);

template <> struct ScalarEnumerationTraits<MachO::LoadCommandType> {
  static void enumeration(IO &io, MachO::LoadCommandType &value) {
#include "llvm/Support/MachO.def"
  }
};

#undef HANDLE_LOAD_COMMAND

} // namespace llvm::yaml

} // namespace llvm

#endif
