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

namespace llvm {
namespace GOFFYAML {

Object::Object() = default;

} // namespace GOFFYAML

namespace yaml {

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDSYMBOLTYPE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDSYMBOLTYPE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_ST_SectionDefinition);
  ECase(ESD_ST_ElementDefinition);
  ECase(ESD_ST_LabelDefinition);
  ECase(ESD_ST_PartReference);
  ECase(ESD_ST_ExternalReference);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDNAMESPACEID>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDNAMESPACEID &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_NS_ProgramManagementBinder);
  ECase(ESD_NS_NormalName);
  ECase(ESD_NS_PseudoRegister);
  ECase(ESD_NS_Parts);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDAMODE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDAMODE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_AMODE_None);
  ECase(ESD_AMODE_24);
  ECase(ESD_AMODE_31);
  ECase(ESD_AMODE_ANY);
  ECase(ESD_AMODE_64);
  ECase(ESD_AMODE_MIN);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDRMODE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDRMODE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_RMODE_None);
  ECase(ESD_RMODE_24);
  ECase(ESD_RMODE_31);
  ECase(ESD_RMODE_64);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDTEXTSTYLE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDTEXTSTYLE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_TS_ByteOriented);
  ECase(ESD_TS_Structured);
  ECase(ESD_TS_Unstructured);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDBINDINGALGORITHM>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDBINDINGALGORITHM &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_BA_Concatenate);
  ECase(ESD_BA_Merge);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDTASKINGBEHAVIOR>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDTASKINGBEHAVIOR &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_TA_Unspecified);
  ECase(ESD_TA_NonReus);
  ECase(ESD_TA_Reus);
  ECase(ESD_TA_Rent);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDEXECUTABLE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDEXECUTABLE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_EXE_Unspecified);
  ECase(ESD_EXE_DATA);
  ECase(ESD_EXE_CODE);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDLINKAGETYPE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDLINKAGETYPE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_LT_OS);
  ECase(ESD_LT_XPLink);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDBINDINGSTRENGTH>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDBINDINGSTRENGTH &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_BST_Strong);
  ECase(ESD_BST_Weak);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDLOADINGBEHAVIOR>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDLOADINGBEHAVIOR &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_LB_Initial);
  ECase(ESD_LB_Deferred);
  ECase(ESD_LB_NoLoad);
  ECase(ESD_LB_Reserved);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDBINDINGSCOPE>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDBINDINGSCOPE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_BSC_Unspecified);
  ECase(ESD_BSC_Section);
  ECase(ESD_BSC_Module);
  ECase(ESD_BSC_Library);
  ECase(ESD_BSC_ImportExport);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_ESDALIGNMENT>::enumeration(
    IO &IO, GOFFYAML::GOFF_ESDALIGNMENT &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(ESD_ALIGN_Byte);
  ECase(ESD_ALIGN_Halfword);
  ECase(ESD_ALIGN_Fullword);
  ECase(ESD_ALIGN_Doubleword);
  ECase(ESD_ALIGN_Quadword);
  ECase(ESD_ALIGN_32byte);
  ECase(ESD_ALIGN_64byte);
  ECase(ESD_ALIGN_128byte);
  ECase(ESD_ALIGN_256byte);
  ECase(ESD_ALIGN_512byte);
  ECase(ESD_ALIGN_1024byte);
  ECase(ESD_ALIGN_2Kpage);
  ECase(ESD_ALIGN_4Kpage);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void MappingTraits<GOFFYAML::FileHeader>::mapping(
    IO &IO, GOFFYAML::FileHeader &FileHdr) {
  IO.mapOptional("TargetEnvironment", FileHdr.TargetEnvironment, 0);
  IO.mapOptional("TargetOperatingSystem", FileHdr.TargetOperatingSystem, 0);
  IO.mapOptional("CCSID", FileHdr.CCSID, 0);
  IO.mapOptional("CharacterSetName", FileHdr.CharacterSetName, "");
  IO.mapOptional("LanguageProductIdentifier", FileHdr.LanguageProductIdentifier,
                 "");
  IO.mapOptional("ArchitectureLevel", FileHdr.ArchitectureLevel, 1);
  IO.mapOptional("InternalCCSID", FileHdr.InternalCCSID);
  IO.mapOptional("TargetSoftwareEnvironment",
                 FileHdr.TargetSoftwareEnvironment);
}

void MappingTraits<GOFFYAML::ESDRecord>::mapping(IO &IO,
                                                 GOFFYAML::ESDRecord &Record) {
  IO.mapRequired("SymbolType", Record.SymbolType);
  IO.mapRequired("ESDID", Record.ESDID);
  IO.mapOptional("ParentESDID", Record.ParentESDID);
  IO.mapOptional("Offset", Record.Offset);
  IO.mapOptional("Length", Record.Length);
  IO.mapOptional("NameSpace", Record.NameSpace);
  IO.mapOptional("Amode", Record.Amode);
  IO.mapOptional("Rmode", Record.Rmode);
  IO.mapOptional("TextStyle", Record.TextStyle);
  IO.mapOptional("BindingAlgorithm", Record.BindingAlgorithm);
  IO.mapOptional("TaskingBehavior", Record.TaskingBehavior);
  IO.mapOptional("ReadOnly", Record.ReadOnly);
  IO.mapOptional("Executable", Record.Executable);
  IO.mapOptional("BindingStrength", Record.BindingStrength);
  IO.mapOptional("LoadingBehavior", Record.LoadingBehavior);
  IO.mapOptional("IndirectReference", Record.IndirectReference);
  IO.mapOptional("BindingScope", Record.BindingScope);
  IO.mapOptional("LinkageType", Record.LinkageType);
  IO.mapOptional("Alignment", Record.Alignment);
  IO.mapOptional("FillByteValue", Record.FillByteValue);
  IO.mapOptional("ADAESDID", Record.ADAESDID);
  IO.mapOptional("SortPriority", Record.SortPriority);
  IO.mapOptional("Name", Record.Name, std::string(""));
}

void MappingTraits<GOFFYAML::Object>::mapping(IO &IO, GOFFYAML::Object &Obj) {
  IO.mapTag("!GOFF", true);
  IO.mapRequired("FileHeader", Obj.Header);
  IO.mapOptional("ESDRecords", Obj.ESDRecords);
}

} // namespace yaml
} // namespace llvm
