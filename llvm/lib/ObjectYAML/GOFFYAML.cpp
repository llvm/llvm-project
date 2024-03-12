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
#include <string.h>

namespace llvm {
namespace GOFFYAML {

Object::Object() {}

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

void ScalarBitSetTraits<GOFFYAML::GOFF_ESDFlags>::bitset(
    IO &IO, GOFFYAML::GOFF_ESDFlags &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, GOFF::X)
#define BCaseMask(X, M) IO.maskedBitSetCase(Value, #X, GOFF::X, GOFF::M)
  BCase(ESD_FillByteValuePresent);
  BCase(ESD_SymbolDisplayFlag);
  BCase(ESD_SymbolRenamingFlag);
  BCase(ESD_RemovableClass);
  BCaseMask(ESD_RQ_0, ESD_Mask_RQW);
  BCaseMask(ESD_RQ_1, ESD_Mask_RQW);
  BCaseMask(ESD_RQ_2, ESD_Mask_RQW);
  BCaseMask(ESD_RQ_3, ESD_Mask_RQW);
#undef BCase
#undef BCaseMask
}

void ScalarEnumerationTraits<GOFFYAML::GOFF_TEXTRECORDSTYLE>::enumeration(
    IO &IO, GOFFYAML::GOFF_TEXTRECORDSTYLE &Value) {
#define ECase(X) IO.enumCase(Value, #X, GOFF::X)
  ECase(TXT_RS_Byte);
  ECase(TXT_RS_Structured);
  ECase(TXT_RS_Unstructured);
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

void ScalarBitSetTraits<GOFFYAML::GOFF_BAFLAGS>::bitset(
    IO &IO, GOFFYAML::GOFF_BAFLAGS &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, GOFF::X)
#define BCaseMask(X, M) IO.maskedBitSetCase(Value, #X, GOFF::X, GOFF::M)
  BCase(ESD_BA_Movable);
  BCase(ESD_BA_ReadOnly);
  BCase(ESD_BA_NoPrime);
  BCase(ESD_BA_COMMON);
  BCase(ESD_BA_Indirect);
#undef BCase
#undef BCaseMask
}

void MappingTraits<GOFFYAML::Symbol>::mapping(IO &IO, GOFFYAML::Symbol &Sym) {
  IO.mapRequired("Name", Sym.Name);
  IO.mapRequired("Type", Sym.Type);
  IO.mapRequired("ID", Sym.ID);
  IO.mapOptional("OwnerID", Sym.OwnerID, 0);
  IO.mapOptional("Address", Sym.Address, 0);
  IO.mapOptional("Length", Sym.Length, 0);
  IO.mapOptional("ExtAttrID", Sym.ExtAttrID, 0);
  IO.mapOptional("ExtAttrOffset", Sym.ExtAttrOffset, 0);
  IO.mapRequired("NameSpace", Sym.NameSpace);
  IO.mapOptional("Flags", Sym.Flags, GOFFYAML::GOFF_ESDFlags(0));
  IO.mapOptional("FillByteValue", Sym.FillByteValue, 0);
  IO.mapOptional("PSectID", Sym.PSectID, 0);
  IO.mapOptional("Priority", Sym.Priority, 0);
  IO.mapOptional("Signature", Sym.Signature, 0);
  IO.mapOptional("Amode", Sym.Amode, GOFF::ESD_AMODE_None);
  IO.mapOptional("Rmode", Sym.Rmode, GOFF::ESD_RMODE_None);
  IO.mapOptional("TextStyle", Sym.TextStyle, GOFF::ESD_TS_ByteOriented);
  IO.mapOptional("BindingAlgorithm", Sym.BindingAlgorithm,
                 GOFF::ESD_BA_Concatenate);
  IO.mapOptional("TaskingBehavior", Sym.TaskingBehavior,
                 GOFF::ESD_TA_Unspecified);
  IO.mapOptional("Executable", Sym.Executable, GOFF::ESD_EXE_Unspecified);
  IO.mapOptional("LinkageType", Sym.LinkageType, GOFF::ESD_LT_OS);
  IO.mapOptional("BindingStrength", Sym.BindingStrength, GOFF::ESD_BST_Strong);
  IO.mapOptional("LoadingBehavior", Sym.LoadingBehavior, GOFF::ESD_LB_Initial);
  IO.mapOptional("BindingScope", Sym.BindingScope, GOFF::ESD_BSC_Unspecified);
  IO.mapOptional("Alignment", Sym.Alignment, GOFF::ESD_ALIGN_Byte);
  IO.mapOptional("BAFlags", Sym.BAFlags, 0);
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

void CustomMappingTraits<GOFFYAML::RecordPtr>::inputOne(
    IO &IO, StringRef Key, GOFFYAML::RecordPtr &Elem) {
  if (Key == "Symbol") {
    GOFFYAML::Symbol Sym;
    IO.mapRequired("Symbol", Sym);
    Elem = std::make_unique<GOFFYAML::Symbol>(std::move(Sym));
  }
}

void CustomMappingTraits<GOFFYAML::RecordPtr>::output(
    IO &IO, GOFFYAML::RecordPtr &Elem) {
  if (auto *Sym = dyn_cast<GOFFYAML::Symbol>(Elem.get())) {
    IO.mapRequired("Symbol", *Sym);
  } else {
    IO.setError("Unknown record type");
  }
}

void MappingTraits<GOFFYAML::Object>::mapping(IO &IO, GOFFYAML::Object &Obj) {
  IO.mapTag("!GOFF", true);
  IO.mapRequired("FileHeader", Obj.Header);
  IO.mapRequired("Records", Obj.Records);
}

} // namespace yaml
} // namespace llvm
