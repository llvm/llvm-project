//===----- XCOFFYAML.h - XCOFF YAMLIO implementation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares classes for handling the YAML representation of XCOFF.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_OBJECTYAML_XCOFFYAML_H
#define LLVM_OBJECTYAML_XCOFFYAML_H

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/ObjectYAML/YAML.h"
#include <vector>

namespace llvm {
namespace XCOFFYAML {

struct FileHeader {
  llvm::yaml::Hex16 Magic;
  uint16_t NumberOfSections;
  int32_t TimeStamp;
  llvm::yaml::Hex32 SymbolTableOffset; // File offset to symbol table.
  int32_t NumberOfSymTableEntries;
  uint16_t AuxHeaderSize;
  llvm::yaml::Hex16 Flags;
};

struct Symbol {
  StringRef SymbolName;
  llvm::yaml::Hex32 Value; // Symbol value; storage class-dependent.
  StringRef SectionName;
  llvm::yaml::Hex16 Type;
  XCOFF::StorageClass StorageClass;
  uint8_t NumberOfAuxEntries; // Number of auxiliary entries
};

struct Object {
  FileHeader Header;
  std::vector<Symbol> Symbols;
  Object();
};
} // namespace XCOFFYAML
} // namespace llvm
LLVM_YAML_IS_SEQUENCE_VECTOR(XCOFFYAML::Symbol)
namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<XCOFF::StorageClass> {
  static void enumeration(IO &IO, XCOFF::StorageClass &Value);
};

template <> struct MappingTraits<XCOFFYAML::FileHeader> {
  static void mapping(IO &IO, XCOFFYAML::FileHeader &H);
};

template <> struct MappingTraits<XCOFFYAML::Object> {
  static void mapping(IO &IO, XCOFFYAML::Object &Obj);
};

template <> struct MappingTraits<XCOFFYAML::Symbol> {
  static void mapping(IO &IO, XCOFFYAML::Symbol &S);
};

} // namespace yaml
} // namespace llvm

#endif // LLVM_OBJECTYAML_XCOFFYAML_H
