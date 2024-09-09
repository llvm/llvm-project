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

// The structure of the yaml files is not an exact 1:1 match to GOFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace GOFFYAML {

LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_AMODE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_TXTRECORDSTYLE)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, GOFF_ENDFLAGS)

// The GOFF format uses different kinds of logical records. The format imposes
// some rules on those records (e.g. the module header must come first, no
// forward references to records, etc.). However, to be able to specify invalid
// GOFF files, we treat all records the same way.
struct RecordBase {
  enum class Kind {
    ModuleHeader,
    RelocationDirectory,
    Symbol,
    Text,
    DeferredLength,
    EndOfModule
  };

private:
  const Kind RecordKind;

protected:
  RecordBase(Kind RecordKind) : RecordKind(RecordKind) {}

public:
  Kind getKind() const { return RecordKind; }
};
using RecordPtr = std::unique_ptr<RecordBase>;

struct ModuleHeader : public RecordBase {
  ModuleHeader() : RecordBase(Kind::ModuleHeader) {}

  uint32_t ArchitectureLevel;
  uint16_t PropertiesLength;
  std::optional<yaml::BinaryRef> Properties;

  static bool classof(const RecordBase *S) {
    return S->getKind() == Kind::ModuleHeader;
  }
};

struct Text : public RecordBase {
  Text() : RecordBase(Kind::Text) {}

  GOFF_TXTRECORDSTYLE Style;
  uint32_t ESDID;
  uint32_t Offset;
  uint32_t TrueLength;
  uint16_t Encoding;
  uint16_t DataLength;
  std::optional<yaml::BinaryRef> Data;

  static bool classof(const RecordBase *S) {
    return S->getKind() == Kind::Text;
  }
};

struct EndOfModule : public RecordBase {
  EndOfModule() : RecordBase(Kind::EndOfModule) {}

  GOFF_ENDFLAGS Flags;
  GOFF_AMODE AMODE;
  uint32_t RecordCount;
  uint32_t ESDID;
  uint32_t Offset;
  uint16_t NameLength;
  StringRef EntryName;

  static bool classof(const RecordBase *S) {
    return S->getKind() == Kind::EndOfModule;
  }
};

struct Object {
  // A GOFF file is a sequence of records.
  std::vector<RecordPtr> Records;
};
} // end namespace GOFFYAML
} // end namespace llvm

LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_AMODE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_TXTRECORDSTYLE)
LLVM_YAML_DECLARE_ENUM_TRAITS(GOFFYAML::GOFF_ENDFLAGS)

LLVM_YAML_IS_SEQUENCE_VECTOR(GOFFYAML::RecordPtr)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::RecordPtr)

LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::ModuleHeader)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Text)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::EndOfModule)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Object)

namespace llvm {
namespace yaml {

template <> struct CustomMappingTraits<GOFFYAML::RecordPtr> {
  static void inputOne(IO &IO, StringRef Key, GOFFYAML::RecordPtr &Elem);
  static void output(IO &IO, GOFFYAML::RecordPtr &Elem);
};

} // namespace yaml
} // namespace llvm
#endif // LLVM_OBJECTYAML_GOFFYAML_H
