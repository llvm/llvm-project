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
#include "llvm/BinaryFormat/XCOFF.h"
#include <string.h>

namespace llvm {
namespace XCOFFYAML {

Object::Object() { memset(&Header, 0, sizeof(Header)); }

} // namespace XCOFFYAML

namespace yaml {

void ScalarEnumerationTraits<XCOFF::StorageClass>::enumeration(
    IO &IO, XCOFF::StorageClass &Value) {
#define ECase(X) IO.enumCase(Value, #X, XCOFF::X)
  ECase(C_NULL);
  ECase(C_AUTO);
  ECase(C_EXT);
  ECase(C_STAT);
  ECase(C_REG);
  ECase(C_EXTDEF);
  ECase(C_LABEL);
  ECase(C_ULABEL);
  ECase(C_MOS);
  ECase(C_ARG);
  ECase(C_STRTAG);
  ECase(C_MOU);
  ECase(C_UNTAG);
  ECase(C_TPDEF);
  ECase(C_USTATIC);
  ECase(C_ENTAG);
  ECase(C_MOE);
  ECase(C_REGPARM);
  ECase(C_FIELD);
  ECase(C_BLOCK);
  ECase(C_FCN);
  ECase(C_EOS);
  ECase(C_FILE);
  ECase(C_LINE);
  ECase(C_ALIAS);
  ECase(C_HIDDEN);
  ECase(C_HIDEXT);
  ECase(C_BINCL);
  ECase(C_EINCL);
  ECase(C_INFO);
  ECase(C_WEAKEXT);
  ECase(C_DWARF);
  ECase(C_GSYM);
  ECase(C_LSYM);
  ECase(C_PSYM);
  ECase(C_RSYM);
  ECase(C_RPSYM);
  ECase(C_STSYM);
  ECase(C_TCSYM);
  ECase(C_BCOMM);
  ECase(C_ECOML);
  ECase(C_ECOMM);
  ECase(C_DECL);
  ECase(C_ENTRY);
  ECase(C_FUN);
  ECase(C_BSTAT);
  ECase(C_ESTAT);
  ECase(C_GTLS);
  ECase(C_STTLS);
  ECase(C_EFCN);
#undef ECase
}

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

void MappingTraits<XCOFFYAML::Symbol>::mapping(IO &IO, XCOFFYAML::Symbol &S) {
  IO.mapRequired("Name", S.SymbolName);
  IO.mapRequired("Value", S.Value);
  IO.mapRequired("Section", S.SectionName);
  IO.mapRequired("Type", S.Type);
  IO.mapRequired("StorageClass", S.StorageClass);
  IO.mapRequired("NumberOfAuxEntries", S.NumberOfAuxEntries);
}

void MappingTraits<XCOFFYAML::Object>::mapping(IO &IO, XCOFFYAML::Object &Obj) {
  IO.mapTag("!XCOFF", true);
  IO.mapRequired("FileHeader", Obj.Header);
  IO.mapRequired("Symbols", Obj.Symbols);
}

} // namespace yaml
} // namespace llvm
