//===- DXContainerYAML.h - DXContainer YAMLIO implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares classes for handling the YAML representation
/// of DXContainer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_DXCONTAINERYAML_H
#define LLVM_OBJECTYAML_DXCONTAINERYAML_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace DXContainerYAML {

struct VersionTuple {
  uint16_t Major;
  uint16_t Minor;
};

// The optional header fields are required in the binary and will be populated
// when reading from binary, but can be omitted in the YAML text because the
// emitter can calculate them.
struct FileHeader {
  std::vector<llvm::yaml::Hex8> Hash;
  VersionTuple Version;
  std::optional<uint32_t> FileSize;
  uint32_t PartCount;
  std::optional<std::vector<uint32_t>> PartOffsets;
};

struct DXILProgram {
  uint8_t MajorVersion;
  uint8_t MinorVersion;
  uint16_t ShaderKind;
  std::optional<uint32_t> Size;
  uint16_t DXILMajorVersion;
  uint16_t DXILMinorVersion;
  std::optional<uint32_t> DXILOffset;
  std::optional<uint32_t> DXILSize;
  std::optional<std::vector<llvm::yaml::Hex8>> DXIL;
};

#define SHADER_FLAG(Num, Val, Str) bool Val = false;
struct ShaderFlags {
  ShaderFlags() = default;
  ShaderFlags(uint64_t FlagData);
  uint64_t getEncodedFlags();
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

struct ShaderHash {
  ShaderHash() = default;
  ShaderHash(const dxbc::ShaderHash &Data);

  bool IncludesSource;
  std::vector<llvm::yaml::Hex8> Digest;
};

struct Part {
  Part() = default;
  Part(std::string N, uint32_t S) : Name(N), Size(S) {}
  std::string Name;
  uint32_t Size;
  std::optional<DXILProgram> Program;
  std::optional<ShaderFlags> Flags;
  std::optional<ShaderHash> Hash;
};

struct Object {
  FileHeader Header;
  std::vector<Part> Parts;
};

} // namespace DXContainerYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::Part)
namespace llvm {

class raw_ostream;

namespace yaml {

template <> struct MappingTraits<DXContainerYAML::VersionTuple> {
  static void mapping(IO &IO, DXContainerYAML::VersionTuple &Version);
};

template <> struct MappingTraits<DXContainerYAML::FileHeader> {
  static void mapping(IO &IO, DXContainerYAML::FileHeader &Header);
};

template <> struct MappingTraits<DXContainerYAML::DXILProgram> {
  static void mapping(IO &IO, DXContainerYAML::DXILProgram &Program);
};

template <> struct MappingTraits<DXContainerYAML::ShaderFlags> {
  static void mapping(IO &IO, DXContainerYAML::ShaderFlags &Flags);
};

template <> struct MappingTraits<DXContainerYAML::ShaderHash> {
  static void mapping(IO &IO, DXContainerYAML::ShaderHash &Hash);
};

template <> struct MappingTraits<DXContainerYAML::Part> {
  static void mapping(IO &IO, DXContainerYAML::Part &Version);
};

template <> struct MappingTraits<DXContainerYAML::Object> {
  static void mapping(IO &IO, DXContainerYAML::Object &Obj);
};

} // namespace yaml

} // namespace llvm

#endif // LLVM_OBJECTYAML_DXCONTAINERYAML_H
