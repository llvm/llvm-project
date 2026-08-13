//===- GOFFYAML.h - GOFF YAMLIO implementation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares classes for handling the YAML representation of GOFF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_GOFFYAML_H
#define LLVM_OBJECTYAML_GOFFYAML_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/ObjectYAML/YAML.h"
#include <cstdint>

namespace llvm {

// The structure of the yaml files is not an exact 1:1 match to GOFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace GOFFYAML {

// Enum typedefs for YAML representation
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDSYMBOLTYPE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDNAMESPACEID)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDAMODE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDRMODE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDTEXTSTYLE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDBINDINGALGORITHM)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDTASKINGBEHAVIOR)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDEXECUTABLE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDLINKAGETYPE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDBINDINGSTRENGTH)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDLOADINGBEHAVIOR)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDBINDINGSCOPE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDALIGNMENT)

struct FileHeader {
  uint32_t TargetEnvironment = 0;
  uint32_t TargetOperatingSystem = 0;
  uint16_t CCSID = 0;
  std::string CharacterSetName;
  std::string LanguageProductIdentifier;
  uint32_t ArchitectureLevel = 0;
  std::optional<uint16_t> InternalCCSID;
  std::optional<uint8_t> TargetSoftwareEnvironment;
};

struct ESDRecord {
  GOFF_ESDSYMBOLTYPE SymbolType = 0;
  uint32_t ESDID = 0;
  std::optional<uint32_t> ParentESDID;
  std::optional<uint32_t> Offset;
  std::optional<uint32_t> Length;
  std::optional<GOFF_ESDNAMESPACEID> NameSpace;
  std::optional<GOFF_ESDAMODE> Amode;
  std::optional<GOFF_ESDRMODE> Rmode;
  std::optional<GOFF_ESDTEXTSTYLE> TextStyle;
  std::optional<GOFF_ESDBINDINGALGORITHM> BindingAlgorithm;
  std::optional<GOFF_ESDTASKINGBEHAVIOR> TaskingBehavior;
  std::optional<bool> ReadOnly;
  std::optional<GOFF_ESDEXECUTABLE> Executable;
  std::optional<GOFF_ESDBINDINGSTRENGTH> BindingStrength;
  std::optional<GOFF_ESDLOADINGBEHAVIOR> LoadingBehavior;
  std::optional<bool> IndirectReference;
  std::optional<GOFF_ESDBINDINGSCOPE> BindingScope;
  std::optional<GOFF_ESDLINKAGETYPE> LinkageType;
  std::optional<GOFF_ESDALIGNMENT> Alignment;
  std::optional<uint8_t> FillByteValue;
  std::optional<uint32_t> ADAESDID;
  std::optional<uint32_t> SortPriority;
  std::string Name;
};

struct Object {
  FileHeader Header;
  std::vector<ESDRecord> ESDRecords;
  LLVM_ABI Object();
};
} // end namespace GOFFYAML
} // end namespace llvm

LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDSYMBOLTYPE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDNAMESPACEID)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDAMODE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDRMODE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDTEXTSTYLE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDBINDINGALGORITHM)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDTASKINGBEHAVIOR)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDEXECUTABLE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDLINKAGETYPE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDBINDINGSTRENGTH)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDLOADINGBEHAVIOR)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDBINDINGSCOPE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDALIGNMENT)

LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::FileHeader)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::ESDRecord)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Object)
LLVM_YAML_IS_SEQUENCE_VECTOR(GOFFYAML::ESDRecord)

#endif // LLVM_OBJECTYAML_GOFFYAML_H
