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

struct RecordBase {
  enum RecordBaseKind { RBK_Relocations, RBK_Symbol, RBK_Section };

private:
  const RecordBaseKind Kind;

protected:
  RecordBase(RecordBaseKind Kind) : Kind(Kind) {}

public:
  RecordBaseKind getKind() const { return Kind; }
};
typedef std::unique_ptr<RecordBase> RecordPtr;

struct FileHeader {
  uint32_t TargetEnvironment;
  uint32_t TargetOperatingSystem;
  uint16_t CCSID;
  StringRef CharacterSetName;
  StringRef LanguageProductIdentifier;
  uint32_t ArchitectureLevel;
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

LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::FileHeader)
LLVM_YAML_DECLARE_MAPPING_TRAITS(GOFFYAML::Object)

namespace llvm {
namespace yaml {

template <> struct CustomMappingTraits<GOFFYAML::RecordPtr> {
  static void inputOne(IO &IO, StringRef Key, GOFFYAML::RecordPtr &Elem) {};
  static void output(IO &IO, GOFFYAML::RecordPtr &Elem) {};
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_GOFFYAML_H
