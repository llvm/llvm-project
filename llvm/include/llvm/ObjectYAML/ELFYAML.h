//===- ELFYAML.h - ELF YAMLIO implementation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares classes for handling the YAML representation
/// of ELF.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_ELFYAML_H
#define LLVM_OBJECTYAML_ELFYAML_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace llvm {
namespace ELFYAML {

// These types are invariant across 32/64-bit ELF, so for simplicity just
// directly give them their exact sizes. We don't need to worry about
// endianness because these are just the types in the YAMLIO structures,
// and are appropriately converted to the necessary endianness when
// reading/generating binary object files.
// The naming of these types is intended to be ELF_PREFIX, where PREFIX is
// the common prefix of the respective constants. E.g. ELF_EM corresponds
// to the `e_machine` constants, like `EM_X86_64`.
// In the future, these would probably be better suited by C++11 enum
// class's with appropriate fixed underlying type.
LLVM_YAML_STRONG_TYPEDEF(uint16_t, ELF_ET)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_PT)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_EM)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFCLASS)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFDATA)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFOSABI)
// Just use 64, since it can hold 32-bit values too.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, ELF_EF)
// Just use 64, since it can hold 32-bit values too.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, ELF_DYNTAG)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_PF)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_SHT)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_REL)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_RSS)
// Just use 64, since it can hold 32-bit values too.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, ELF_SHF)
LLVM_YAML_STRONG_TYPEDEF(uint16_t, ELF_SHN)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_STT)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_STV)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_STO)

LLVM_YAML_STRONG_TYPEDEF(uint8_t, MIPS_AFL_REG)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, MIPS_ABI_FP)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_EXT)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_ASE)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_FLAGS1)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_ISA)

// For now, hardcode 64 bits everywhere that 32 or 64 would be needed
// since 64-bit can hold 32-bit values too.
struct FileHeader {
  ELF_ELFCLASS Class;
  ELF_ELFDATA Data;
  ELF_ELFOSABI OSABI;
  llvm::yaml::Hex8 ABIVersion;
  ELF_ET Type;
  ELF_EM Machine;
  ELF_EF Flags;
  llvm::yaml::Hex64 Entry;
};

struct SectionName {
  StringRef Section;
};

struct ProgramHeader {
  ELF_PT Type;
  ELF_PF Flags;
  llvm::yaml::Hex64 VAddr;
  llvm::yaml::Hex64 PAddr;
  Optional<llvm::yaml::Hex64> Align;
  std::vector<SectionName> Sections;
};

struct Symbol {
  StringRef Name;
  ELF_STT Type;
  StringRef Section;
  Optional<ELF_SHN> Index;
  llvm::yaml::Hex64 Value;
  llvm::yaml::Hex64 Size;
  uint8_t Other;
};

struct LocalGlobalWeakSymbols {
  std::vector<Symbol> Local;
  std::vector<Symbol> Global;
  std::vector<Symbol> Weak;
};

struct SectionOrType {
  StringRef sectionNameOrType;
};

struct DynamicEntry {
  ELF_DYNTAG Tag;
  llvm::yaml::Hex64 Val;
};

struct Section {
  enum class SectionKind {
    Dynamic,
    Group,
    RawContent,
    Relocation,
    NoBits,
    Verdef,
    Verneed,
    Symver,
    MipsABIFlags
  };
  SectionKind Kind;
  StringRef Name;
  ELF_SHT Type;
  ELF_SHF Flags;
  llvm::yaml::Hex64 Address;
  StringRef Link;
  llvm::yaml::Hex64 AddressAlign;
  Optional<llvm::yaml::Hex64> EntSize;

  Section(SectionKind Kind) : Kind(Kind) {}
  virtual ~Section();
};

struct DynamicSection : Section {
  std::vector<DynamicEntry> Entries;
  Optional<yaml::BinaryRef> Content;

  DynamicSection() : Section(SectionKind::Dynamic) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Dynamic;
  }
};

struct RawContentSection : Section {
  yaml::BinaryRef Content;
  llvm::yaml::Hex64 Size;
  llvm::yaml::Hex64 Info;

  RawContentSection() : Section(SectionKind::RawContent) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::RawContent;
  }
};

struct NoBitsSection : Section {
  llvm::yaml::Hex64 Size;

  NoBitsSection() : Section(SectionKind::NoBits) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::NoBits;
  }
};

struct VernauxEntry {
  uint32_t Hash;
  uint16_t Flags;
  uint16_t Other;
  StringRef Name;
};

struct VerneedEntry {
  uint16_t Version;
  StringRef File;
  std::vector<VernauxEntry> AuxV;
};

struct VerneedSection : Section {
  std::vector<VerneedEntry> VerneedV;
  llvm::yaml::Hex64 Info;

  VerneedSection() : Section(SectionKind::Verneed) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Verneed;
  }
};

struct SymverSection : Section {
  std::vector<uint16_t> Entries;

  SymverSection() : Section(SectionKind::Symver) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Symver;
  }
};

struct VerdefEntry {
  uint16_t Version;
  uint16_t Flags;
  uint16_t VersionNdx;
  uint32_t Hash;
  std::vector<StringRef> VerNames;
};

struct VerdefSection : Section {
  std::vector<VerdefEntry> Entries;
  llvm::yaml::Hex64 Info;

  VerdefSection() : Section(SectionKind::Verdef) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Verdef;
  }
};

struct Group : Section {
  // Members of a group contain a flag and a list of section indices
  // that are part of the group.
  std::vector<SectionOrType> Members;
  StringRef Signature; /* Info */

  Group() : Section(SectionKind::Group) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Group;
  }
};

struct Relocation {
  llvm::yaml::Hex64 Offset;
  int64_t Addend;
  ELF_REL Type;
  Optional<StringRef> Symbol;
};

struct RelocationSection : Section {
  std::vector<Relocation> Relocations;
  StringRef RelocatableSec; /* Info */

  RelocationSection() : Section(SectionKind::Relocation) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Relocation;
  }
};

// Represents .MIPS.abiflags section
struct MipsABIFlags : Section {
  llvm::yaml::Hex16 Version;
  MIPS_ISA ISALevel;
  llvm::yaml::Hex8 ISARevision;
  MIPS_AFL_REG GPRSize;
  MIPS_AFL_REG CPR1Size;
  MIPS_AFL_REG CPR2Size;
  MIPS_ABI_FP FpABI;
  MIPS_AFL_EXT ISAExtension;
  MIPS_AFL_ASE ASEs;
  MIPS_AFL_FLAGS1 Flags1;
  llvm::yaml::Hex32 Flags2;

  MipsABIFlags() : Section(SectionKind::MipsABIFlags) {}

  static bool classof(const Section *S) {
    return S->Kind == SectionKind::MipsABIFlags;
  }
};

struct Object {
  FileHeader Header;
  std::vector<ProgramHeader> ProgramHeaders;
  std::vector<std::unique_ptr<Section>> Sections;
  // Although in reality the symbols reside in a section, it is a lot
  // cleaner and nicer if we read them from the YAML as a separate
  // top-level key, which automatically ensures that invariants like there
  // being a single SHT_SYMTAB section are upheld.
  LocalGlobalWeakSymbols Symbols;
  LocalGlobalWeakSymbols DynamicSymbols;
};

} // end namespace ELFYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::DynamicEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::ProgramHeader)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::ELFYAML::Section>)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::Symbol)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::VerdefEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::VernauxEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::VerneedEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::Relocation)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::SectionOrType)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::SectionName)

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ET> {
  static void enumeration(IO &IO, ELFYAML::ELF_ET &Value);
};

template <> struct ScalarEnumerationTraits<ELFYAML::ELF_PT> {
  static void enumeration(IO &IO, ELFYAML::ELF_PT &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_EM> {
  static void enumeration(IO &IO, ELFYAML::ELF_EM &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFCLASS> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFCLASS &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFDATA> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFDATA &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFOSABI> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFOSABI &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::ELF_EF> {
  static void bitset(IO &IO, ELFYAML::ELF_EF &Value);
};

template <> struct ScalarBitSetTraits<ELFYAML::ELF_PF> {
  static void bitset(IO &IO, ELFYAML::ELF_PF &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_SHT> {
  static void enumeration(IO &IO, ELFYAML::ELF_SHT &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::ELF_SHF> {
  static void bitset(IO &IO, ELFYAML::ELF_SHF &Value);
};

template <> struct ScalarEnumerationTraits<ELFYAML::ELF_SHN> {
  static void enumeration(IO &IO, ELFYAML::ELF_SHN &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_STT> {
  static void enumeration(IO &IO, ELFYAML::ELF_STT &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_STV> {
  static void enumeration(IO &IO, ELFYAML::ELF_STV &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::ELF_STO> {
  static void bitset(IO &IO, ELFYAML::ELF_STO &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_REL> {
  static void enumeration(IO &IO, ELFYAML::ELF_REL &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_DYNTAG> {
  static void enumeration(IO &IO, ELFYAML::ELF_DYNTAG &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_RSS> {
  static void enumeration(IO &IO, ELFYAML::ELF_RSS &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::MIPS_AFL_REG> {
  static void enumeration(IO &IO, ELFYAML::MIPS_AFL_REG &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::MIPS_ABI_FP> {
  static void enumeration(IO &IO, ELFYAML::MIPS_ABI_FP &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::MIPS_AFL_EXT> {
  static void enumeration(IO &IO, ELFYAML::MIPS_AFL_EXT &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::MIPS_ISA> {
  static void enumeration(IO &IO, ELFYAML::MIPS_ISA &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::MIPS_AFL_ASE> {
  static void bitset(IO &IO, ELFYAML::MIPS_AFL_ASE &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::MIPS_AFL_FLAGS1> {
  static void bitset(IO &IO, ELFYAML::MIPS_AFL_FLAGS1 &Value);
};

template <>
struct MappingTraits<ELFYAML::FileHeader> {
  static void mapping(IO &IO, ELFYAML::FileHeader &FileHdr);
};

template <> struct MappingTraits<ELFYAML::ProgramHeader> {
  static void mapping(IO &IO, ELFYAML::ProgramHeader &FileHdr);
};

template <>
struct MappingTraits<ELFYAML::Symbol> {
  static void mapping(IO &IO, ELFYAML::Symbol &Symbol);
  static StringRef validate(IO &IO, ELFYAML::Symbol &Symbol);
};

template <>
struct MappingTraits<ELFYAML::LocalGlobalWeakSymbols> {
  static void mapping(IO &IO, ELFYAML::LocalGlobalWeakSymbols &Symbols);
};

template <> struct MappingTraits<ELFYAML::DynamicEntry> {
  static void mapping(IO &IO, ELFYAML::DynamicEntry &Rel);
};

template <> struct MappingTraits<ELFYAML::VerdefEntry> {
  static void mapping(IO &IO, ELFYAML::VerdefEntry &E);
};

template <> struct MappingTraits<ELFYAML::VerneedEntry> {
  static void mapping(IO &IO, ELFYAML::VerneedEntry &E);
};

template <> struct MappingTraits<ELFYAML::VernauxEntry> {
  static void mapping(IO &IO, ELFYAML::VernauxEntry &E);
};

template <> struct MappingTraits<ELFYAML::Relocation> {
  static void mapping(IO &IO, ELFYAML::Relocation &Rel);
};

template <>
struct MappingTraits<std::unique_ptr<ELFYAML::Section>> {
  static void mapping(IO &IO, std::unique_ptr<ELFYAML::Section> &Section);
  static StringRef validate(IO &io, std::unique_ptr<ELFYAML::Section> &Section);
};

template <>
struct MappingTraits<ELFYAML::Object> {
  static void mapping(IO &IO, ELFYAML::Object &Object);
};

template <> struct MappingTraits<ELFYAML::SectionOrType> {
  static void mapping(IO &IO, ELFYAML::SectionOrType &sectionOrType);
};

template <> struct MappingTraits<ELFYAML::SectionName> {
  static void mapping(IO &IO, ELFYAML::SectionName &sectionName);
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_ELFYAML_H
