//===- Object.h -------------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_OBJECT_H
#define LLVM_TOOLS_OBJCOPY_OBJECT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <vector>

namespace llvm {

class FileOutputBuffer;
class SectionBase;
class Segment;

class SectionTableRef {
private:
  ArrayRef<std::unique_ptr<SectionBase>> Sections;

public:
  SectionTableRef(ArrayRef<std::unique_ptr<SectionBase>> Secs)
      : Sections(Secs) {}
  SectionTableRef(const SectionTableRef &) = default;

  SectionBase *getSection(uint16_t Index, Twine ErrMsg);

  template <class T>
  T *getSectionOfType(uint16_t Index, Twine IndexErrMsg, Twine TypeErrMsg);
};

class SectionBase {
public:
  StringRef Name;
  Segment *ParentSegment = nullptr;
  uint64_t HeaderOffset;
  uint64_t OriginalOffset;
  uint32_t Index;

  uint64_t Addr = 0;
  uint64_t Align = 1;
  uint32_t EntrySize = 0;
  uint64_t Flags = 0;
  uint64_t Info = 0;
  uint64_t Link = ELF::SHN_UNDEF;
  uint64_t NameIndex = 0;
  uint64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t Type = ELF::SHT_NULL;

  virtual ~SectionBase() = default;

  virtual void initialize(SectionTableRef SecTable);
  virtual void finalize();
  virtual void removeSectionReferences(const SectionBase *Sec);
  template <class ELFT> void writeHeader(FileOutputBuffer &Out) const;
  virtual void writeSection(FileOutputBuffer &Out) const = 0;
};

class Segment {
private:
  struct SectionCompare {
    bool operator()(const SectionBase *Lhs, const SectionBase *Rhs) const {
      // Some sections might have the same address if one of them is empty. To
      // fix this we can use the lexicographic ordering on ->Addr and the
      // address of the actully stored section.
      if (Lhs->OriginalOffset == Rhs->OriginalOffset)
        return Lhs < Rhs;
      return Lhs->OriginalOffset < Rhs->OriginalOffset;
    }
  };

  std::set<const SectionBase *, SectionCompare> Sections;
  ArrayRef<uint8_t> Contents;

public:
  uint64_t Align;
  uint64_t FileSize;
  uint32_t Flags;
  uint32_t Index;
  uint64_t MemSize;
  uint64_t Offset;
  uint64_t PAddr;
  uint64_t Type;
  uint64_t VAddr;

  uint64_t OriginalOffset;
  Segment *ParentSegment = nullptr;

  Segment(ArrayRef<uint8_t> Data) : Contents(Data) {}

  const SectionBase *firstSection() const {
    if (!Sections.empty())
      return *Sections.begin();
    return nullptr;
  }

  void removeSection(const SectionBase *Sec) { Sections.erase(Sec); }
  void addSection(const SectionBase *Sec) { Sections.insert(Sec); }
  template <class ELFT> void writeHeader(FileOutputBuffer &Out) const;
  void writeSegment(FileOutputBuffer &Out) const;
};

class Section : public SectionBase {
private:
  ArrayRef<uint8_t> Contents;

public:
  Section(ArrayRef<uint8_t> Data) : Contents(Data) {}

  void writeSection(FileOutputBuffer &Out) const override;
};

class OwnedDataSection : public SectionBase {
private:
  std::vector<uint8_t> Data;

public:
  OwnedDataSection(StringRef SecName, ArrayRef<uint8_t> Data)
      : Data(std::begin(Data), std::end(Data)) {
    Name = SecName;
    Type = ELF::SHT_PROGBITS;
    Size = Data.size();
  }
  void writeSection(FileOutputBuffer &Out) const override;
};

// There are two types of string tables that can exist, dynamic and not dynamic.
// In the dynamic case the string table is allocated. Changing a dynamic string
// table would mean altering virtual addresses and thus the memory image. So
// dynamic string tables should not have an interface to modify them or
// reconstruct them. This type lets us reconstruct a string table. To avoid
// this class being used for dynamic string tables (which has happened) the
// classof method checks that the particular instance is not allocated. This
// then agrees with the makeSection method used to construct most sections.
class StringTableSection : public SectionBase {
private:
  StringTableBuilder StrTabBuilder;

public:
  StringTableSection() : StrTabBuilder(StringTableBuilder::ELF) {
    Type = ELF::SHT_STRTAB;
  }

  void addString(StringRef Name);
  uint32_t findIndex(StringRef Name) const;
  void finalize() override;
  void writeSection(FileOutputBuffer &Out) const override;

  static bool classof(const SectionBase *S) {
    if (S->Flags & ELF::SHF_ALLOC)
      return false;
    return S->Type == ELF::SHT_STRTAB;
  }
};

// Symbols have a st_shndx field that normally stores an index but occasionally
// stores a different special value. This enum keeps track of what the st_shndx
// field means. Most of the values are just copies of the special SHN_* values.
// SYMBOL_SIMPLE_INDEX means that the st_shndx is just an index of a section.
enum SymbolShndxType {
  SYMBOL_SIMPLE_INDEX = 0,
  SYMBOL_ABS = ELF::SHN_ABS,
  SYMBOL_COMMON = ELF::SHN_COMMON,
  SYMBOL_HEXAGON_SCOMMON = ELF::SHN_HEXAGON_SCOMMON,
  SYMBOL_HEXAGON_SCOMMON_2 = ELF::SHN_HEXAGON_SCOMMON_2,
  SYMBOL_HEXAGON_SCOMMON_4 = ELF::SHN_HEXAGON_SCOMMON_4,
  SYMBOL_HEXAGON_SCOMMON_8 = ELF::SHN_HEXAGON_SCOMMON_8,
};

struct Symbol {
  uint8_t Binding;
  SectionBase *DefinedIn = nullptr;
  SymbolShndxType ShndxType;
  uint32_t Index;
  StringRef Name;
  uint32_t NameIndex;
  uint64_t Size;
  uint8_t Type;
  uint64_t Value;
  uint8_t Visibility;

  uint16_t getShndx() const;
};

class SymbolTableSection : public SectionBase {
protected:
  std::vector<std::unique_ptr<Symbol>> Symbols;
  StringTableSection *SymbolNames = nullptr;

  using SymPtr = std::unique_ptr<Symbol>;

public:
  void setStrTab(StringTableSection *StrTab) { SymbolNames = StrTab; }
  void addSymbol(StringRef Name, uint8_t Bind, uint8_t Type,
                 SectionBase *DefinedIn, uint64_t Value, uint8_t Visibility,
                 uint16_t Shndx, uint64_t Sz);
  void addSymbolNames();
  const SectionBase *getStrTab() const { return SymbolNames; }
  const Symbol *getSymbolByIndex(uint32_t Index) const;
  void removeSectionReferences(const SectionBase *Sec) override;
  void localize(std::function<bool(const Symbol &)> ToLocalize);
  void initialize(SectionTableRef SecTable) override;
  void finalize() override;

  static bool classof(const SectionBase *S) {
    return S->Type == ELF::SHT_SYMTAB;
  }
};

// Only writeSection depends on the ELF type so we implement it in a subclass.
template <class ELFT> class SymbolTableSectionImpl : public SymbolTableSection {
  void writeSection(FileOutputBuffer &Out) const override;
};

struct Relocation {
  const Symbol *RelocSymbol = nullptr;
  uint64_t Offset;
  uint64_t Addend;
  uint32_t Type;
};

// All relocation sections denote relocations to apply to another section.
// However, some relocation sections use a dynamic symbol table and others use
// a regular symbol table. Because the types of the two symbol tables differ in
// our system (because they should behave differently) we can't uniformly
// represent all relocations with the same base class if we expose an interface
// that mentions the symbol table type. So we split the two base types into two
// different classes, one which handles the section the relocation is applied to
// and another which handles the symbol table type. The symbol table type is
// taken as a type parameter to the class (see RelocSectionWithSymtabBase).
class RelocationSectionBase : public SectionBase {
protected:
  SectionBase *SecToApplyRel = nullptr;

public:
  const SectionBase *getSection() const { return SecToApplyRel; }
  void setSection(SectionBase *Sec) { SecToApplyRel = Sec; }

  static bool classof(const SectionBase *S) {
    return S->Type == ELF::SHT_REL || S->Type == ELF::SHT_RELA;
  }
};

// Takes the symbol table type to use as a parameter so that we can deduplicate
// that code between the two symbol table types.
template <class SymTabType>
class RelocSectionWithSymtabBase : public RelocationSectionBase {
private:
  SymTabType *Symbols = nullptr;

protected:
  RelocSectionWithSymtabBase() = default;

public:
  void setSymTab(SymTabType *StrTab) { Symbols = StrTab; }
  void removeSectionReferences(const SectionBase *Sec) override;
  void initialize(SectionTableRef SecTable) override;
  void finalize() override;
};

template <class ELFT>
class RelocationSection
    : public RelocSectionWithSymtabBase<SymbolTableSection> {
private:
  using Elf_Rel = typename ELFT::Rel;
  using Elf_Rela = typename ELFT::Rela;

  std::vector<Relocation> Relocations;

  template <class T> void writeRel(T *Buf) const;

public:
  void addRelocation(Relocation Rel) { Relocations.push_back(Rel); }
  void writeSection(FileOutputBuffer &Out) const override;

  static bool classof(const SectionBase *S) {
    if (S->Flags & ELF::SHF_ALLOC)
      return false;
    return S->Type == ELF::SHT_REL || S->Type == ELF::SHT_RELA;
  }
};

class SectionWithStrTab : public Section {
private:
  const SectionBase *StrTab = nullptr;

public:
  SectionWithStrTab(ArrayRef<uint8_t> Data) : Section(Data) {}

  void setStrTab(const SectionBase *StringTable) { StrTab = StringTable; }
  void removeSectionReferences(const SectionBase *Sec) override;
  void initialize(SectionTableRef SecTable) override;
  void finalize() override;
  static bool classof(const SectionBase *S);
};

class DynamicSymbolTableSection : public SectionWithStrTab {
public:
  DynamicSymbolTableSection(ArrayRef<uint8_t> Data) : SectionWithStrTab(Data) {}

  static bool classof(const SectionBase *S) {
    return S->Type == ELF::SHT_DYNSYM;
  }
};

class DynamicSection : public SectionWithStrTab {
public:
  DynamicSection(ArrayRef<uint8_t> Data) : SectionWithStrTab(Data) {}

  static bool classof(const SectionBase *S) {
    return S->Type == ELF::SHT_DYNAMIC;
  }
};

class DynamicRelocationSection
    : public RelocSectionWithSymtabBase<DynamicSymbolTableSection> {
private:
  ArrayRef<uint8_t> Contents;

public:
  DynamicRelocationSection(ArrayRef<uint8_t> Data) : Contents(Data) {}

  void writeSection(FileOutputBuffer &Out) const override;

  static bool classof(const SectionBase *S) {
    if (!(S->Flags & ELF::SHF_ALLOC))
      return false;
    return S->Type == ELF::SHT_REL || S->Type == ELF::SHT_RELA;
  }
};

template <class ELFT> class Object {
private:
  using SecPtr = std::unique_ptr<SectionBase>;
  using SegPtr = std::unique_ptr<Segment>;

  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Ehdr = typename ELFT::Ehdr;
  using Elf_Phdr = typename ELFT::Phdr;

  void initSymbolTable(const object::ELFFile<ELFT> &ElfFile,
                       SymbolTableSection *SymTab, SectionTableRef SecTable);
  SecPtr makeSection(const object::ELFFile<ELFT> &ElfFile,
                     const Elf_Shdr &Shdr);
  void readProgramHeaders(const object::ELFFile<ELFT> &ElfFile);
  SectionTableRef readSectionHeaders(const object::ELFFile<ELFT> &ElfFile);

protected:
  StringTableSection *SectionNames = nullptr;
  SymbolTableSection *SymbolTable = nullptr;
  std::vector<SecPtr> Sections;
  std::vector<SegPtr> Segments;

  void writeHeader(FileOutputBuffer &Out) const;
  void writeProgramHeaders(FileOutputBuffer &Out) const;
  void writeSectionData(FileOutputBuffer &Out) const;
  void writeSectionHeaders(FileOutputBuffer &Out) const;

public:
  uint8_t Ident[16];
  uint64_t Entry;
  uint64_t SHOffset;
  uint32_t Type;
  uint32_t Machine;
  uint32_t Version;
  uint32_t Flags;
  bool WriteSectionHeaders = true;

  Object(const object::ELFObjectFile<ELFT> &Obj);
  virtual ~Object() = default;

  SymbolTableSection *getSymTab() const { return SymbolTable; }
  const SectionBase *getSectionHeaderStrTab() const { return SectionNames; }
  void removeSections(std::function<bool(const SectionBase &)> ToRemove);
  void addSection(StringRef SecName, ArrayRef<uint8_t> Data);
  virtual size_t totalSize() const = 0;
  virtual void finalize() = 0;
  virtual void write(FileOutputBuffer &Out) const = 0;
};

template <class ELFT> class ELFObject : public Object<ELFT> {
private:
  using SecPtr = std::unique_ptr<SectionBase>;
  using SegPtr = std::unique_ptr<Segment>;

  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Ehdr = typename ELFT::Ehdr;
  using Elf_Phdr = typename ELFT::Phdr;

  void sortSections();
  void assignOffsets();

public:
  ELFObject(const object::ELFObjectFile<ELFT> &Obj) : Object<ELFT>(Obj) {}

  void finalize() override;
  size_t totalSize() const override;
  void write(FileOutputBuffer &Out) const override;
};

template <class ELFT> class BinaryObject : public Object<ELFT> {
private:
  using SecPtr = std::unique_ptr<SectionBase>;
  using SegPtr = std::unique_ptr<Segment>;

  uint64_t TotalSize;

public:
  BinaryObject(const object::ELFObjectFile<ELFT> &Obj) : Object<ELFT>(Obj) {}

  void finalize() override;
  size_t totalSize() const override;
  void write(FileOutputBuffer &Out) const override;
};

} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_OBJECT_H
