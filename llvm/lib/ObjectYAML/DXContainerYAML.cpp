//===- DXContainerYAML.cpp - DXContainer YAMLIO implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of
// DXContainerYAML.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DXContainerYAML.h"
#include "llvm/BinaryFormat/DXContainer.h"

namespace llvm {

// This assert is duplicated here to leave a breadcrumb of the places that need
// to be updated if flags grow past 64-bits.
static_assert((uint64_t)dxbc::FeatureFlags::NextUnusedBit <= 1ull << 63,
              "Shader flag bits exceed enum size.");

DXContainerYAML::ShaderFlags::ShaderFlags(uint64_t FlagData) {
#define SHADER_FLAG(Num, Val, Str)                                             \
  Val = (FlagData & (uint64_t)dxbc::FeatureFlags::Val) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

uint64_t DXContainerYAML::ShaderFlags::getEncodedFlags() {
  uint64_t Flag = 0;
#define SHADER_FLAG(Num, Val, Str)                                             \
  if (Val)                                                                     \
    Flag |= (uint64_t)dxbc::FeatureFlags::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
}

namespace yaml {

void MappingTraits<DXContainerYAML::VersionTuple>::mapping(
    IO &IO, DXContainerYAML::VersionTuple &Version) {
  IO.mapRequired("Major", Version.Major);
  IO.mapRequired("Minor", Version.Minor);
}

void MappingTraits<DXContainerYAML::FileHeader>::mapping(
    IO &IO, DXContainerYAML::FileHeader &Header) {
  IO.mapRequired("Hash", Header.Hash);
  IO.mapRequired("Version", Header.Version);
  IO.mapOptional("FileSize", Header.FileSize);
  IO.mapRequired("PartCount", Header.PartCount);
  IO.mapOptional("PartOffsets", Header.PartOffsets);
}

void MappingTraits<DXContainerYAML::DXILProgram>::mapping(
    IO &IO, DXContainerYAML::DXILProgram &Program) {
  IO.mapRequired("MajorVersion", Program.MajorVersion);
  IO.mapRequired("MinorVersion", Program.MinorVersion);
  IO.mapRequired("ShaderKind", Program.ShaderKind);
  IO.mapOptional("Size", Program.Size);
  IO.mapRequired("DXILMajorVersion", Program.DXILMajorVersion);
  IO.mapRequired("DXILMinorVersion", Program.DXILMinorVersion);
  IO.mapOptional("DXILSize", Program.DXILSize);
  IO.mapOptional("DXIL", Program.DXIL);
}

void MappingTraits<DXContainerYAML::ShaderFlags>::mapping(
    IO &IO, DXContainerYAML::ShaderFlags &Flags) {
#define SHADER_FLAG(Num, Val, Str) IO.mapRequired(#Val, Flags.Val);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<DXContainerYAML::Part>::mapping(IO &IO,
                                                   DXContainerYAML::Part &P) {
  IO.mapRequired("Name", P.Name);
  IO.mapRequired("Size", P.Size);
  IO.mapOptional("Program", P.Program);
  IO.mapOptional("Flags", P.Flags);
}

void MappingTraits<DXContainerYAML::Object>::mapping(
    IO &IO, DXContainerYAML::Object &Obj) {
  IO.mapTag("!dxcontainer", true);
  IO.mapRequired("Header", Obj.Header);
  IO.mapRequired("Parts", Obj.Parts);
}

} // namespace yaml
} // namespace llvm
