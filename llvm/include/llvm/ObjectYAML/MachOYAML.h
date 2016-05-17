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

struct Section {
  char sectname[16];
  char segname[16];
  llvm::yaml::Hex64 addr;
  uint64_t size;
  llvm::yaml::Hex32 offset;
  uint32_t align;
  llvm::yaml::Hex32 reloff;
  uint32_t nreloc;
  llvm::yaml::Hex32 flags;
  llvm::yaml::Hex32 reserved1;
  llvm::yaml::Hex32 reserved2;
  llvm::yaml::Hex32 reserved3;
};

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
  llvm::MachO::macho_load_command Data;
  std::vector<Section> Sections;
};

struct Object {
  FileHeader Header;
  std::vector<LoadCommand> LoadCommands;
  std::vector<Section> Sections;
};

} // namespace llvm::MachOYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::MachOYAML::LoadCommand)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::MachOYAML::Section)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<MachOYAML::FileHeader> {
  static void mapping(IO &IO, MachOYAML::FileHeader &FileHeader);
};

template <> struct MappingTraits<MachOYAML::Object> {
  static void mapping(IO &IO, MachOYAML::Object &Object);
};

template <> struct MappingTraits<MachOYAML::LoadCommand> {
  static void mapping(IO &IO, MachOYAML::LoadCommand &LoadCommand);
};

template <> struct MappingTraits<MachOYAML::Section> {
  static void mapping(IO &IO, MachOYAML::Section &Section);
};

#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  io.enumCase(value, #LCName, MachO::LCName);

template <> struct ScalarEnumerationTraits<MachO::LoadCommandType> {
  static void enumeration(IO &io, MachO::LoadCommandType &value) {
#include "llvm/Support/MachO.def"
  }
};

// This trait is used for 16-byte chars in Mach structures used for strings
typedef char char_16[16];

template <> struct ScalarTraits<char_16> {
  static void output(const char_16 &Val, void *, llvm::raw_ostream &Out);

  static StringRef input(StringRef Scalar, void *, char_16 &Val);
  static bool mustQuote(StringRef S);
};

// This trait is used for UUIDs. It reads and writes them matching otool's
// formatting style.
typedef uint8_t uuid_t[16];

template <> struct ScalarTraits<uuid_t> {
  static void output(const uuid_t &Val, void *, llvm::raw_ostream &Out);

  static StringRef input(StringRef Scalar, void *, uuid_t &Val);
  static bool mustQuote(StringRef S);
};

// Load Command struct mapping traits

#define LOAD_COMMAND_STRUCT(LCStruct)                                          \
  template <> struct MappingTraits<MachO::LCStruct> {                          \
    static void mapping(IO &IO, MachO::LCStruct &LoadCommand);                 \
  };

#include "llvm/Support/MachO.def"

// Extra structures used by load commands
template <> struct MappingTraits<MachO::dylib> {
  static void mapping(IO &IO, MachO::dylib &LoadCommand);
};

template <> struct MappingTraits<MachO::fvmlib> {
  static void mapping(IO &IO, MachO::fvmlib &LoadCommand);
};

template <> struct MappingTraits<MachO::section> {
  static void mapping(IO &IO, MachO::section &LoadCommand);
};

template <> struct MappingTraits<MachO::section_64> {
  static void mapping(IO &IO, MachO::section_64 &LoadCommand);
};

} // namespace llvm::yaml

} // namespace llvm

#endif
