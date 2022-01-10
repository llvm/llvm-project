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

enum AuxSymbolType : uint8_t {
  AUX_EXCEPT = 255,
  AUX_FCN = 254,
  AUX_SYM = 253,
  AUX_FILE = 252,
  AUX_CSECT = 251,
  AUX_SECT = 250,
  AUX_STAT = 249
};

struct AuxSymbolEnt {
  AuxSymbolType Type;

  explicit AuxSymbolEnt(AuxSymbolType T) : Type(T) {}
  virtual ~AuxSymbolEnt();
};

struct FileAuxEnt : AuxSymbolEnt {
  Optional<StringRef> FileNameOrString;
  Optional<XCOFF::CFileStringType> FileStringType;

  FileAuxEnt() : AuxSymbolEnt(AuxSymbolType::AUX_FILE) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_FILE;
  }
};

struct CsectAuxEnt : AuxSymbolEnt {
  // Only for XCOFF32.
  Optional<uint32_t> SectionOrLength;
  Optional<uint32_t> StabInfoIndex;
  Optional<uint16_t> StabSectNum;
  // Only for XCOFF64.
  Optional<uint32_t> SectionOrLengthLo;
  Optional<uint32_t> SectionOrLengthHi;
  // Common fields for both XCOFF32 and XCOFF64.
  Optional<uint32_t> ParameterHashIndex;
  Optional<uint16_t> TypeChkSectNum;
  Optional<uint8_t> SymbolAlignmentAndType;
  Optional<XCOFF::StorageMappingClass> StorageMappingClass;

  CsectAuxEnt() : AuxSymbolEnt(AuxSymbolType::AUX_CSECT) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_CSECT;
  }
};

struct FunctionAuxEnt : AuxSymbolEnt {
  Optional<uint32_t> OffsetToExceptionTbl; // Only for XCOFF32.
  Optional<uint64_t> PtrToLineNum;
  Optional<uint32_t> SizeOfFunction;
  Optional<int32_t> SymIdxOfNextBeyond;

  FunctionAuxEnt() : AuxSymbolEnt(AuxSymbolType::AUX_FCN) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_FCN;
  }
};

struct ExcpetionAuxEnt : AuxSymbolEnt {
  Optional<uint64_t> OffsetToExceptionTbl;
  Optional<uint32_t> SizeOfFunction;
  Optional<int32_t> SymIdxOfNextBeyond;

  ExcpetionAuxEnt() : AuxSymbolEnt(AuxSymbolType::AUX_EXCEPT) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_EXCEPT;
  }
}; // Only for XCOFF64.

struct BlockAuxEnt : AuxSymbolEnt {
  // Only for XCOFF32.
  Optional<uint16_t> LineNumHi;
  Optional<uint16_t> LineNumLo;
  // Only for XCOFF64.
  Optional<uint32_t> LineNum;

  BlockAuxEnt() : AuxSymbolEnt(AuxSymbolType::AUX_SYM) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_SYM;
  }
};

struct SectAuxEntForDWARF : AuxSymbolEnt {
  Optional<uint32_t> LengthOfSectionPortion;
  Optional<uint32_t> NumberOfRelocEnt;

  SectAuxEntForDWARF() : AuxSymbolEnt(AuxSymbolType::AUX_SECT) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_SECT;
  }
};

struct SectAuxEntForStat : AuxSymbolEnt {
  Optional<uint32_t> SectionLength;
  Optional<uint16_t> NumberOfRelocEnt;
  Optional<uint16_t> NumberOfLineNum;

  SectAuxEntForStat() : AuxSymbolEnt(AuxSymbolType::AUX_STAT) {}
  static bool classof(const AuxSymbolEnt *S) {
    return S->Type == AuxSymbolType::AUX_STAT;
  }
}; // Only for XCOFF32.

struct Symbol {
  StringRef SymbolName;
  llvm::yaml::Hex64 Value; // Symbol value; storage class-dependent.
  Optional<StringRef> SectionName;
  Optional<uint16_t> SectionIndex;
  llvm::yaml::Hex16 Type;
  XCOFF::StorageClass StorageClass;
  Optional<uint8_t> NumberOfAuxEntries;
  std::vector<std::unique_ptr<AuxSymbolEnt>> AuxEntries;
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
LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::XCOFFYAML::AuxSymbolEnt>)

namespace llvm {
namespace yaml {

template <> struct ScalarBitSetTraits<XCOFF::SectionTypeFlags> {
  static void bitset(IO &IO, XCOFF::SectionTypeFlags &Value);
};

template <> struct ScalarEnumerationTraits<XCOFF::StorageClass> {
  static void enumeration(IO &IO, XCOFF::StorageClass &Value);
};

template <> struct ScalarEnumerationTraits<XCOFF::StorageMappingClass> {
  static void enumeration(IO &IO, XCOFF::StorageMappingClass &Value);
};

template <> struct ScalarEnumerationTraits<XCOFF::CFileStringType> {
  static void enumeration(IO &IO, XCOFF::CFileStringType &Type);
};

template <> struct ScalarEnumerationTraits<XCOFFYAML::AuxSymbolType> {
  static void enumeration(IO &IO, XCOFFYAML::AuxSymbolType &Type);
};

template <> struct MappingTraits<XCOFFYAML::FileHeader> {
  static void mapping(IO &IO, XCOFFYAML::FileHeader &H);
};

template <> struct MappingTraits<XCOFFYAML::AuxiliaryHeader> {
  static void mapping(IO &IO, XCOFFYAML::AuxiliaryHeader &AuxHdr);
};

template <> struct MappingTraits<std::unique_ptr<XCOFFYAML::AuxSymbolEnt>> {
  static void mapping(IO &IO, std::unique_ptr<XCOFFYAML::AuxSymbolEnt> &AuxSym);
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
