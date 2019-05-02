//===-- XCOFFYAML.cpp - XCOFF YAMLIO implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of XCOFF.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/XCOFFYAML.h"
#include <string.h>

namespace llvm {
namespace XCOFFYAML {

Object::Object() { memset(&Header, 0, sizeof(Header)); }

} // namespace XCOFFYAML

namespace yaml {

void MappingTraits<XCOFFYAML::FileHeader>::mapping(
    IO &IO, XCOFFYAML::FileHeader &FileHdr) {
  IO.mapRequired("MagicNumber", FileHdr.Magic);
  IO.mapRequired("NumberOfSections", FileHdr.NumberOfSections);
  IO.mapRequired("CreationTime", FileHdr.TimeStamp);
  IO.mapRequired("OffsetToSymbolTable", FileHdr.SymbolTableOffset);
  IO.mapRequired("EntriesInSymbolTable", FileHdr.NumberOfSymTableEntries);
  IO.mapRequired("AuxiliaryHeaderSize", FileHdr.AuxHeaderSize);
  IO.mapRequired("Flags", FileHdr.Flags);
}

void MappingTraits<XCOFFYAML::Object>::mapping(IO &IO, XCOFFYAML::Object &Obj) {
  IO.mapTag("!XCOFF", true);
  IO.mapRequired("FileHeader", Obj.Header);
}

} // namespace yaml
} // namespace llvm
