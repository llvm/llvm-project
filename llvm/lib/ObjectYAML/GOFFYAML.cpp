//===-- GOFFYAML.cpp - GOFF YAMLIO implementation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of GOFF.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/GOFFYAML.h"
#include "llvm/BinaryFormat/GOFF.h"

namespace llvm {

namespace yaml {

void ScalarEnumerationTraits<GOFFYAML::GOFF_AMODE>::enumeration(
    IO &IO, GOFFYAML::GOFF_AMODE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::ESD_##X)
  ECase(AMODE_None);
  ECase(AMODE_24);
  ECase(AMODE_31);
  ECase(AMODE_ANY);
  ECase(AMODE_64);
  ECase(AMODE_MIN);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_TXTRECORDSTYLE>::enumeration(
    IO &IO, GOFFYAML::GOFF_TXTRECORDSTYLE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::TXT_##X)
  ECase(TS_Byte);
  ECase(TS_Structured);
  ECase(TS_Unstructured);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ENDFLAGS>::enumeration(
    IO &IO, GOFFYAML::GOFF_ENDFLAGS &Value) {
#define ECase(X) IO.enumCase(Value, #X, unsigned(GOFF::END_##X) << 6)
  ECase(EPR_None);
  ECase(EPR_EsdidOffset);
  ECase(EPR_ExternalName);
  ECase(EPR_Reserved);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void MappingTraits<GOFFYAML::ModuleHeader>::mapping(
    IO &IO, GOFFYAML::ModuleHeader &ModHdr) {
  IO.mapOptional("ArchitectureLevel", ModHdr.ArchitectureLevel, 0);
  IO.mapOptional("PropertiesLength", ModHdr.PropertiesLength, 0);
  IO.mapOptional("Properties", ModHdr.Properties);
}

void MappingTraits<GOFFYAML::Text>::mapping(IO &IO, GOFFYAML::Text &Txt) {
  IO.mapOptional("TextStyle", Txt.Style, GOFF::TXT_TS_Byte);
  IO.mapOptional("ESDID", Txt.ESDID, 0);
  IO.mapOptional("Offset", Txt.Offset, 0);
  IO.mapOptional("TrueLength", Txt.TrueLength, 0);
  IO.mapOptional("TextEncoding", Txt.Encoding, 0);
  IO.mapOptional("DataLength", Txt.DataLength, 0);
  IO.mapOptional("Data", Txt.Data);
}

void MappingTraits<GOFFYAML::EndOfModule>::mapping(IO &IO,
                                                   GOFFYAML::EndOfModule &End) {
  IO.mapOptional("Flags", End.Flags, 0);
  IO.mapOptional("AMODE", End.AMODE, 0);
  IO.mapOptional("RecordCount", End.RecordCount, 0);
  IO.mapOptional("ESDID", End.ESDID, 0);
  IO.mapOptional("Offset", End.Offset, 0);
  IO.mapOptional("NameLength", End.NameLength, 0);
  IO.mapOptional("EntryName", End.EntryName);
}

void CustomMappingTraits<GOFFYAML::RecordPtr>::inputOne(
    IO &IO, StringRef Key, GOFFYAML::RecordPtr &Elem) {
  if (Key == "ModuleHeader") {
    GOFFYAML::ModuleHeader ModHdr;
    IO.mapRequired("ModuleHeader", ModHdr);
    Elem = std::make_unique<GOFFYAML::ModuleHeader>(std::move(ModHdr));
  } else if (Key == "Text") {
    GOFFYAML::Text Txt;
    IO.mapRequired("Text", Txt);
    Elem = std::make_unique<GOFFYAML::Text>(std::move(Txt));
  } else if (Key == "End") {
    GOFFYAML::EndOfModule End;
    IO.mapRequired("End", End);
    Elem = std::make_unique<GOFFYAML::EndOfModule>(std::move(End));
  } else if (Key == "RelocationDirectory" || Key == "Symbol" || Key == "Length")
    IO.setError(Twine("not yet implemented ").concat(Key));
  else
    IO.setError(Twine("unknown record type name ").concat(Key));
}

void CustomMappingTraits<GOFFYAML::RecordPtr>::output(
    IO &IO, GOFFYAML::RecordPtr &Elem) {
  switch (Elem->getKind()) {
  case GOFFYAML::RecordBase::Kind::ModuleHeader:
    IO.mapRequired("ModuleHeader",
                   *static_cast<GOFFYAML::ModuleHeader *>(Elem.get()));
    break;
  case GOFFYAML::RecordBase::Kind::Text:
    IO.mapRequired("Text", *static_cast<GOFFYAML::Text *>(Elem.get()));
    break;
  case GOFFYAML::RecordBase::Kind::EndOfModule:
    IO.mapRequired("End", *static_cast<GOFFYAML::EndOfModule *>(Elem.get()));
    break;
  case GOFFYAML::RecordBase::Kind::RelocationDirectory:
  case GOFFYAML::RecordBase::Kind::Symbol:
  case GOFFYAML::RecordBase::Kind::DeferredLength:
    llvm_unreachable("not yet implemented");
  }
}

void MappingTraits<GOFFYAML::Object>::mapping(IO &IO, GOFFYAML::Object &Obj) {
  IO.mapTag("!GOFF", true);
  EmptyContext Context;
  yamlize(IO, Obj.Records, false, Context);
}

} // namespace yaml
} // namespace llvm
