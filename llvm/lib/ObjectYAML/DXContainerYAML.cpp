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

namespace llvm {
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

void MappingTraits<DXContainerYAML::Part>::mapping(IO &IO,
                                                   DXContainerYAML::Part &P) {
  IO.mapRequired("Name", P.Name);
  IO.mapRequired("Size", P.Size);
  IO.mapOptional("Program", P.Program);
}

void MappingTraits<DXContainerYAML::Object>::mapping(
    IO &IO, DXContainerYAML::Object &Obj) {
  IO.mapTag("!dxcontainer", true);
  IO.mapRequired("Header", Obj.Header);
  IO.mapRequired("Parts", Obj.Parts);
}

} // namespace yaml
} // namespace llvm
