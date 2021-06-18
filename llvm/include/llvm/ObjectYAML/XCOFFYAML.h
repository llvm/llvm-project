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
  llvm::yaml::Hex64 SymbolTableOffset;
  uint32_t NumberOfSymTableEntries;
  uint16_t AuxHeaderSize;
  llvm::yaml::Hex16 Flags;
};

struct Relocation {
  llvm::yaml::Hex64 VirtualAddress;
  llvm::yaml::Hex64 SymbolIndex;
  llvm::yaml::Hex8 Info;
  llvm::yaml::Hex8 Type;
};

struct Section {
  StringRef SectionName;
  llvm::yaml::Hex64 Address;
  llvm::yaml::Hex64 Size;
  llvm::yaml::Hex64 FileOffsetToData;
  llvm::yaml::Hex64 FileOffsetToRelocations;
  llvm::yaml::Hex64 FileOffsetToLineNumbers; // Line number pointer. Not supported yet.
  llvm::yaml::Hex16 NumberOfRelocations;
  llvm::yaml::Hex16 NumberOfLineNumbers; // Line number counts. Not supported yet.
  uint32_t Flags;
  yaml::BinaryRef SectionData;
  std::vector<Relocation> Relocations;
};

struct Symbol {
  StringRef SymbolName;
  llvm::yaml::Hex64 Value; // Symbol value; storage class-dependent.
  StringRef SectionName;
  llvm::yaml::Hex16 Type;
  XCOFF::StorageClass StorageClass;
  uint8_t NumberOfAuxEntries;
};

struct Object {
  FileHeader Header;
  std::vector<Section> Sections;
  std::vector<Symbol> Symbols;
  Object();
};
} // namespace XCOFFYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(XCOFFYAML::Symbol)
LLVM_YAML_IS_SEQUENCE_VECTOR(XCOFFYAML::Relocation)
LLVM_YAML_IS_SEQUENCE_VECTOR(XCOFFYAML::Section)

namespace llvm {
namespace yaml {

template <> struct ScalarBitSetTraits<XCOFF::SectionTypeFlags> {
  static void bitset(IO &IO, XCOFF::SectionTypeFlags &Value);
};

template <> struct ScalarEnumerationTraits<XCOFF::StorageClass> {
  static void enumeration(IO &IO, XCOFF::StorageClass &Value);
};

template <> struct MappingTraits<XCOFFYAML::FileHeader> {
  static void mapping(IO &IO, XCOFFYAML::FileHeader &H);
};


template <> struct MappingTraits<XCOFFYAML::Symbol> {
  static void mapping(IO &IO, XCOFFYAML::Symbol &S);
};

template <> struct MappingTraits<XCOFFYAML::Relocation> {
  static void mapping(IO &IO, XCOFFYAML::Relocation &R);
};

template <> struct MappingTraits<XCOFFYAML::Section> {
  static void mapping(IO &IO, XCOFFYAML::Section &Sec);
};

template <> struct MappingTraits<XCOFFYAML::Object> {
  static void mapping(IO &IO, XCOFFYAML::Object &Obj);
};

} // namespace yaml
} // namespace llvm

#endif // LLVM_OBJECTYAML_XCOFFYAML_H
