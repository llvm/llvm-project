//===- Object.h -------------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_OBJECT_H
#define LLVM_OBJCOPY_OBJECT_H

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/FileOutputBuffer.h"

#include <memory>
#include <set>

class Segment;

class SectionBase {
public:
  llvm::StringRef Name;
  Segment *ParentSegment = nullptr;
  uint64_t HeaderOffset;
  uint64_t OriginalOffset;
  uint32_t Index;

  uint64_t Addr = 0;
  uint64_t Align = 1;
  uint32_t EntrySize = 0;
  uint64_t Flags = 0;
  uint64_t Info = 0;
  uint64_t Link = llvm::ELF::SHN_UNDEF;
  uint64_t NameIndex = 0;
  uint64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t Type = llvm::ELF::SHT_NULL;

  virtual ~SectionBase() {}
  virtual void finalize();
  template <class ELFT> void writeHeader(llvm::FileOutputBuffer &Out) const;
  virtual void writeSection(llvm::FileOutputBuffer &Out) const = 0;
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
  llvm::ArrayRef<uint8_t> Contents;

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

  Segment(llvm::ArrayRef<uint8_t> Data) : Contents(Data) {}
  void finalize();
  const SectionBase *firstSection() const {
    if (!Sections.empty())
      return *Sections.begin();
    return nullptr;
  }
  void addSection(const SectionBase *sec) { Sections.insert(sec); }
  template <class ELFT> void writeHeader(llvm::FileOutputBuffer &Out) const;
  void writeSegment(llvm::FileOutputBuffer &Out) const;
};

class Section : public SectionBase {
private:
  llvm::ArrayRef<uint8_t> Contents;

public:
  Section(llvm::ArrayRef<uint8_t> Data) : Contents(Data) {}
  void writeSection(llvm::FileOutputBuffer &Out) const override;
};

// This is just a wraper around a StringTableBuilder that implements SectionBase
class StringTableSection : public SectionBase {
private:
  llvm::StringTableBuilder StrTabBuilder;

public:
  StringTableSection() : StrTabBuilder(llvm::StringTableBuilder::ELF) {
    Type = llvm::ELF::SHT_STRTAB;
  }

  void addString(llvm::StringRef Name);
  uint32_t findIndex(llvm::StringRef Name) const;
  void finalize() override;
  void writeSection(llvm::FileOutputBuffer &Out) const override;
  static bool classof(const SectionBase *S) {
    return S->Type == llvm::ELF::SHT_STRTAB;
  }
};

template <class ELFT> class Object {
private:
  typedef std::unique_ptr<SectionBase> SecPtr;
  typedef std::unique_ptr<Segment> SegPtr;

  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;

  SecPtr makeSection(const llvm::object::ELFFile<ELFT> &ElfFile,
                     const Elf_Shdr &Shdr);
  void readProgramHeaders(const llvm::object::ELFFile<ELFT> &ElfFile);
  void readSectionHeaders(const llvm::object::ELFFile<ELFT> &ElfFile);

protected:
  StringTableSection *SectionNames;
  std::vector<SecPtr> Sections;
  std::vector<SegPtr> Segments;

  void writeHeader(llvm::FileOutputBuffer &Out) const;
  void writeProgramHeaders(llvm::FileOutputBuffer &Out) const;
  void writeSectionData(llvm::FileOutputBuffer &Out) const;
  void writeSectionHeaders(llvm::FileOutputBuffer &Out) const;

public:
  uint8_t Ident[16];
  uint64_t Entry;
  uint64_t SHOffset;
  uint32_t Type;
  uint32_t Machine;
  uint32_t Version;
  uint32_t Flags;

  Object(const llvm::object::ELFObjectFile<ELFT> &Obj);
  virtual size_t totalSize() const = 0;
  virtual void finalize() = 0;
  virtual void write(llvm::FileOutputBuffer &Out) const = 0;
  virtual ~Object() = default;
};

template <class ELFT> class ELFObject : public Object<ELFT> {
private:
  typedef std::unique_ptr<SectionBase> SecPtr;
  typedef std::unique_ptr<Segment> SegPtr;

  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;

  void sortSections();
  void assignOffsets();

public:
  ELFObject(const llvm::object::ELFObjectFile<ELFT> &Obj) : Object<ELFT>(Obj) {}
  void finalize() override;
  size_t totalSize() const override;
  void write(llvm::FileOutputBuffer &Out) const override;
};

template <class ELFT> class BinaryObject : public Object<ELFT> {
private:
  typedef std::unique_ptr<SectionBase> SecPtr;
  typedef std::unique_ptr<Segment> SegPtr;

  uint64_t TotalSize;

public:
  BinaryObject(const llvm::object::ELFObjectFile<ELFT> &Obj)
      : Object<ELFT>(Obj) {}
  void finalize() override;
  size_t totalSize() const override;
  void write(llvm::FileOutputBuffer &Out) const override;
};
#endif
