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
  int32_t NumberOfSymTableEntries;
  uint16_t AuxHeaderSize;
  llvm::yaml::Hex16 Flags;
};

struct AuxiliaryHeader {
  Optional<llvm::yaml::Hex16> Magic;
  Optional<llvm::yaml::Hex16> Version;
  Optional<llvm::yaml::Hex64> TextStartAddr;
  Optional<llvm::yaml::Hex64> DataStartAddr;
  Optional<llvm::yaml::Hex64> TOCAnchorAddr;
  Optional<uint16_t> SecNumOfEntryPoint;
  Optional<uint16_t> SecNumOfText;
  Optional<uint16_t> SecNumOfData;
  Optional<uint16_t> SecNumOfTOC;
  Optional<uint16_t> SecNumOfLoader;
  Optional<uint16_t> SecNumOfBSS;
  Optional<llvm::yaml::Hex16> MaxAlignOfText;
  Optional<llvm::yaml::Hex16> MaxAlignOfData;
  Optional<llvm::yaml::Hex16> ModuleType;
  Optional<llvm::yaml::Hex8> CpuFlag;
  Optional<llvm::yaml::Hex8> CpuType;
  Optional<llvm::yaml::Hex8> TextPageSize;
  Optional<llvm::yaml::Hex8> DataPageSize;
  Optional<llvm::yaml::Hex8> StackPageSize;
  Optional<llvm::yaml::Hex8> FlagAndTDataAlignment;
  Optional<llvm::yaml::Hex64> TextSize;
  Optional<llvm::yaml::Hex64> InitDataSize;
  Optional<llvm::yaml::Hex64> BssDataSize;
  Optional<llvm::yaml::Hex64> EntryPointAddr;
  Optional<llvm::yaml::Hex64> MaxStackSize;
  Optional<llvm::yaml::Hex64> MaxDataSize;
  Optional<uint16_t> SecNumOfTData;
  Optional<uint16_t> SecNumOfTBSS;
  Optional<llvm::yaml::Hex16> Flag;
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
  Optional<StringRef> SectionName;
  Optional<uint16_t> SectionIndex;
  llvm::yaml::Hex16 Type;
  XCOFF::StorageClass StorageClass;
  uint8_t NumberOfAuxEntries;
};

struct StringTable {
  Optional<uint32_t> ContentSize; // The total size of the string table.
  Optional<uint32_t> Length;      // The value of the length field for the first
                                  // 4 bytes of the table.
  Optional<std::vector<StringRef>> Strings;
  Optional<yaml::BinaryRef> RawContent;
};

struct Object {
  FileHeader Header;
  Optional<AuxiliaryHeader> AuxHeader;
  std::vector<Section> Sections;
  std::vector<Symbol> Symbols;
  StringTable StrTbl;
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

template <> struct MappingTraits<XCOFFYAML::AuxiliaryHeader> {
  static void mapping(IO &IO, XCOFFYAML::AuxiliaryHeader &AuxHdr);
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

template <> struct MappingTraits<XCOFFYAML::StringTable> {
  static void mapping(IO &IO, XCOFFYAML::StringTable &Str);
};

template <> struct MappingTraits<XCOFFYAML::Object> {
  static void mapping(IO &IO, XCOFFYAML::Object &Obj);
};

} // namespace yaml
} // namespace llvm

#endif // LLVM_OBJECTYAML_XCOFFYAML_H
