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

void ScalarBitSetTraits<XCOFF::SectionTypeFlags>::bitset(
    IO &IO, XCOFF::SectionTypeFlags &Value) {
#define ECase(X) IO.bitSetCase(Value, #X, XCOFF::X)
  ECase(STYP_PAD);
  ECase(STYP_DWARF);
  ECase(STYP_TEXT);
  ECase(STYP_DATA);
  ECase(STYP_BSS);
  ECase(STYP_EXCEPT);
  ECase(STYP_INFO);
  ECase(STYP_TDATA);
  ECase(STYP_TBSS);
  ECase(STYP_LOADER);
  ECase(STYP_DEBUG);
  ECase(STYP_TYPCHK);
  ECase(STYP_OVRFLO);
#undef ECase
}

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

struct NSectionFlags {
  NSectionFlags(IO &) : Flags(XCOFF::SectionTypeFlags(0)) {}
  NSectionFlags(IO &, uint32_t C) : Flags(XCOFF::SectionTypeFlags(C)) {}

  uint32_t denormalize(IO &) { return Flags; }

  XCOFF::SectionTypeFlags Flags;
};

void MappingTraits<XCOFFYAML::FileHeader>::mapping(
    IO &IO, XCOFFYAML::FileHeader &FileHdr) {
  IO.mapOptional("MagicNumber", FileHdr.Magic);
  IO.mapOptional("NumberOfSections", FileHdr.NumberOfSections);
  IO.mapOptional("CreationTime", FileHdr.TimeStamp);
  IO.mapOptional("OffsetToSymbolTable", FileHdr.SymbolTableOffset);
  IO.mapOptional("EntriesInSymbolTable", FileHdr.NumberOfSymTableEntries);
  IO.mapOptional("AuxiliaryHeaderSize", FileHdr.AuxHeaderSize);
  IO.mapOptional("Flags", FileHdr.Flags);
}

void MappingTraits<XCOFFYAML::Relocation>::mapping(IO &IO,
                                                   XCOFFYAML::Relocation &R) {
  IO.mapOptional("Address", R.VirtualAddress);
  IO.mapOptional("Symbol", R.SymbolIndex);
  IO.mapOptional("Info", R.Info);
  IO.mapOptional("Type", R.Type);
}

void MappingTraits<XCOFFYAML::Section>::mapping(IO &IO,
                                                XCOFFYAML::Section &Sec) {
  MappingNormalization<NSectionFlags, uint32_t> NC(IO, Sec.Flags);
  IO.mapOptional("Name", Sec.SectionName);
  IO.mapOptional("Address", Sec.Address);
  IO.mapOptional("Size", Sec.Size);
  IO.mapOptional("FileOffsetToData", Sec.FileOffsetToData);
  IO.mapOptional("FileOffsetToRelocations", Sec.FileOffsetToRelocations);
  IO.mapOptional("FileOffsetToLineNumbers", Sec.FileOffsetToLineNumbers);
  IO.mapOptional("NumberOfRelocations", Sec.NumberOfRelocations);
  IO.mapOptional("NumberOfLineNumbers", Sec.NumberOfLineNumbers);
  IO.mapOptional("Flags", NC->Flags);
  IO.mapOptional("SectionData", Sec.SectionData);
  IO.mapOptional("Relocations", Sec.Relocations);
}

void MappingTraits<XCOFFYAML::Symbol>::mapping(IO &IO, XCOFFYAML::Symbol &S) {
  IO.mapOptional("Name", S.SymbolName);
  IO.mapOptional("Value", S.Value);
  IO.mapOptional("Section", S.SectionName);
  IO.mapOptional("SectionIndex", S.SectionIndex);
  IO.mapOptional("Type", S.Type);
  IO.mapOptional("StorageClass", S.StorageClass);
  IO.mapOptional("NumberOfAuxEntries", S.NumberOfAuxEntries);
}

void MappingTraits<XCOFFYAML::StringTable>::mapping(IO &IO, XCOFFYAML::StringTable &Str) {
  IO.mapOptional("ContentSize", Str.ContentSize);
  IO.mapOptional("Length", Str.Length);
  IO.mapOptional("Strings", Str.Strings);
  IO.mapOptional("RawContent", Str.RawContent);
}

void MappingTraits<XCOFFYAML::Object>::mapping(IO &IO, XCOFFYAML::Object &Obj) {
  IO.mapTag("!XCOFF", true);
  IO.mapRequired("FileHeader", Obj.Header);
  IO.mapOptional("Sections", Obj.Sections);
  IO.mapOptional("Symbols", Obj.Symbols);
  IO.mapOptional("StringTable", Obj.StrTbl);
}

} // namespace yaml
} // namespace llvm
