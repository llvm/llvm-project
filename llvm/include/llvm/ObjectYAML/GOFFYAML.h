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
#include <vector>

namespace llvm {
namespace GOFF {

enum ESDFlags {
  ESD_FillByteValuePresent = 1 << 7,
  ESD_SymbolDisplayFlag = 1 << 6,
  ESD_SymbolRenamingFlag = 1 << 5,
  ESD_RemovableClass = 1 << 4
};

enum {
  ESD_Mask_ERST = 0x07,
  ESD_Mask_RQW = 0x07,
  ESD_Mask_TextStyle = 0xf0,
  ESD_Mask_BindingAlgorithm = 0x0f,
};

enum ESDBAFlags {
  ESD_BA_Movable = 0x01,
  ESD_BA_ReadOnly = 0x2,
  ESD_BA_NoPrime = 0x4,
  ESD_BA_COMMON = 0x8,
  ESD_BA_Indirect = 0x10,
};

} // end namespace GOFF

// The structure of the yaml files is not an exact 1:1 match to GOFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace GOFFYAML {

LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDSYMBOLTYPE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDNAMESPACEID)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ESDFlags)
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
LLVM_YAML_STRONG_TYPEDEF(uint64_t, GOFF_BAFLAGS)

LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_TEXTRECORDSTYLE)

struct RecordBase {
  enum RecordBaseKind { RBK_Symbol };

private:
  const RecordBaseKind Kind;

protected:
  RecordBase(RecordBaseKind Kind) : Kind(Kind) {}

public:
  RecordBaseKind getKind() const { return Kind; }
};

using RecordPtr = std::unique_ptr<RecordBase>;

struct Symbol : public RecordBase {
  Symbol() : RecordBase(RBK_Symbol) {}

  StringRef Name;
  GOFF_ESDSYMBOLTYPE Type;
  uint32_t ID;
  uint32_t OwnerID;
  uint32_t Address;
  uint32_t Length;
  uint32_t ExtAttrID;
  uint32_t ExtAttrOffset;
  GOFF_ESDNAMESPACEID NameSpace;
  GOFF_ESDFlags Flags;
  uint8_t FillByteValue;
  uint32_t PSectID;
  uint32_t Priority;
  llvm::yaml::Hex64 Signature;
  GOFF_ESDAMODE Amode;
  GOFF_ESDRMODE Rmode;
  GOFF_ESDTEXTSTYLE TextStyle;
  GOFF_ESDBINDINGALGORITHM BindingAlgorithm;
  GOFF_ESDTASKINGBEHAVIOR TaskingBehavior;
  GOFF_ESDEXECUTABLE Executable;
  GOFF_ESDLINKAGETYPE LinkageType;
  GOFF_ESDBINDINGSTRENGTH BindingStrength;
  GOFF_ESDLOADINGBEHAVIOR LoadingBehavior;
  GOFF_ESDBINDINGSCOPE BindingScope;
  GOFF_ESDALIGNMENT Alignment;
  GOFF_BAFLAGS BAFlags;

  static bool classof(const RecordBase *Rec) {
    return Rec->getKind() == RBK_Symbol;
  }
};

struct FileHeader {
  uint32_t TargetEnvironment = 0;
  uint32_t TargetOperatingSystem = 0;
  uint16_t CCSID = 0;
  StringRef CharacterSetName;
  StringRef LanguageProductIdentifier;
  uint32_t ArchitectureLevel = 0;
  std::optional<uint16_t> InternalCCSID;
  std::optional<uint8_t> TargetSoftwareEnvironment;
};

struct Object {
  FileHeader Header;
  std::vector<RecordPtr> Records;
  Object();
};

} // end namespace GOFFYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(GOFFYAML::RecordPtr)

LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDSYMBOLTYPE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ESDNAMESPACEID)
LLVM_YAML_DECLARE_BITSET_TRAITS(GOFFYAML::GOFF_ESDFlags)
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
LLVM_YAML_DECLARE_BITSET_TRAITS(GOFFYAML::GOFF_BAFLAGS)

LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_TEXTRECORDSTYLE)

LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Symbol)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::FileHeader)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Object)

namespace llvm {
namespace yaml {

template <> struct CustomMappingTraits<GOFFYAML::RecordPtr> {
  static void inputOne(IO &IO, StringRef Key, GOFFYAML::RecordPtr &Elem);
  static void output(IO &IO, GOFFYAML::RecordPtr &Elem);
};

} // end namespace yaml
} // end namespace llvm
#endif // LLVM_OBJECTYAML_GOFFYAML_H
