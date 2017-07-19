//===- Object.cpp -----------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Object.h"
#include "llvm-objcopy.h"

using namespace llvm;
using namespace object;
using namespace ELF;

template <class ELFT> void Segment::writeHeader(FileOutputBuffer &Out) const {
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Phdr Elf_Phdr;

  uint8_t *Buf = Out.getBufferStart();
  Buf += sizeof(Elf_Ehdr) + Index * sizeof(Elf_Phdr);
  Elf_Phdr &Phdr = *reinterpret_cast<Elf_Phdr *>(Buf);
  Phdr.p_type = Type;
  Phdr.p_flags = Flags;
  Phdr.p_offset = Offset;
  Phdr.p_vaddr = VAddr;
  Phdr.p_paddr = PAddr;
  Phdr.p_filesz = FileSize;
  Phdr.p_memsz = MemSize;
  Phdr.p_align = Align;
}

void Segment::finalize() {
  auto FirstSec = firstSection();
  if (FirstSec) {
    // It is possible for a gap to be at the begining of a segment. Because of
    // this we need to compute the new offset based on how large this gap was
    // in the source file. Section layout should have already ensured that this
    // space is not used for something else.
    uint64_t OriginalOffset = Offset;
    Offset = FirstSec->Offset - (FirstSec->OriginalOffset - OriginalOffset);
  }
}

void SectionBase::finalize() {}

template <class ELFT>
void SectionBase::writeHeader(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart();
  Buf += HeaderOffset;
  typename ELFT::Shdr &Shdr = *reinterpret_cast<typename ELFT::Shdr *>(Buf);
  Shdr.sh_name = NameIndex;
  Shdr.sh_type = Type;
  Shdr.sh_flags = Flags;
  Shdr.sh_addr = Addr;
  Shdr.sh_offset = Offset;
  Shdr.sh_size = Size;
  Shdr.sh_link = Link;
  Shdr.sh_info = Info;
  Shdr.sh_addralign = Align;
  Shdr.sh_entsize = EntrySize;
}

void Section::writeSection(FileOutputBuffer &Out) const {
  if (Type == SHT_NOBITS)
    return;
  uint8_t *Buf = Out.getBufferStart() + Offset;
  std::copy(std::begin(Contents), std::end(Contents), Buf);
}

void StringTableSection::addString(StringRef Name) {
  StrTabBuilder.add(Name);
  Size = StrTabBuilder.getSize();
}

uint32_t StringTableSection::findIndex(StringRef Name) const {
  return StrTabBuilder.getOffset(Name);
}

void StringTableSection::finalize() { StrTabBuilder.finalize(); }

void StringTableSection::writeSection(FileOutputBuffer &Out) const {
  StrTabBuilder.write(Out.getBufferStart() + Offset);
}

// Returns true IFF a section is wholly inside the range of a segment
static bool sectionWithinSegment(const SectionBase &Section,
                                 const Segment &Segment) {
  // If a section is empty it should be treated like it has a size of 1. This is
  // to clarify the case when an empty section lies on a boundary between two
  // segments and ensures that the section "belongs" to the second segment and
  // not the first.
  uint64_t SecSize = Section.Size ? Section.Size : 1;
  return Segment.Offset <= Section.OriginalOffset &&
         Segment.Offset + Segment.FileSize >= Section.OriginalOffset + SecSize;
}

template <class ELFT>
void Object<ELFT>::readProgramHeaders(const ELFFile<ELFT> &ElfFile) {
  uint32_t Index = 0;
  for (const auto &Phdr : unwrapOrError(ElfFile.program_headers())) {
    Segments.emplace_back(make_unique<Segment>());
    Segment &Seg = *Segments.back();
    Seg.Type = Phdr.p_type;
    Seg.Flags = Phdr.p_flags;
    Seg.Offset = Phdr.p_offset;
    Seg.VAddr = Phdr.p_vaddr;
    Seg.PAddr = Phdr.p_paddr;
    Seg.FileSize = Phdr.p_filesz;
    Seg.MemSize = Phdr.p_memsz;
    Seg.Align = Phdr.p_align;
    Seg.Index = Index++;
    for (auto &Section : Sections) {
      if (sectionWithinSegment(*Section, Seg)) {
        Seg.addSection(&*Section);
        if (!Section->ParentSegment ||
            Section->ParentSegment->Offset > Seg.Offset) {
          Section->ParentSegment = &Seg;
        }
      }
    }
  }
}

template <class ELFT>
void Object<ELFT>::readSectionHeaders(const ELFFile<ELFT> &ElfFile) {
  uint32_t Index = 0;
  for (const auto &Shdr : unwrapOrError(ElfFile.sections())) {
    if (Index == 0) {
      ++Index;
      continue;
    }
    if (Shdr.sh_type == SHT_STRTAB)
      continue;
    ArrayRef<uint8_t> Data;
    if (Shdr.sh_type != SHT_NOBITS)
      Data = unwrapOrError(ElfFile.getSectionContents(&Shdr));
    SecPtr Sec = make_unique<Section>(Data);
    Sec->Name = unwrapOrError(ElfFile.getSectionName(&Shdr));
    Sec->Type = Shdr.sh_type;
    Sec->Flags = Shdr.sh_flags;
    Sec->Addr = Shdr.sh_addr;
    Sec->Offset = Shdr.sh_offset;
    Sec->OriginalOffset = Shdr.sh_offset;
    Sec->Size = Shdr.sh_size;
    Sec->Link = Shdr.sh_link;
    Sec->Info = Shdr.sh_info;
    Sec->Align = Shdr.sh_addralign;
    Sec->EntrySize = Shdr.sh_entsize;
    Sec->Index = Index++;
    SectionNames->addString(Sec->Name);
    Sections.push_back(std::move(Sec));
  }
}

template <class ELFT> size_t Object<ELFT>::totalSize() const {
  // We already have the section header offset so we can calculate the total
  // size by just adding up the size of each section header.
  return SHOffset + Sections.size() * sizeof(Elf_Shdr) + sizeof(Elf_Shdr);
}

template <class ELFT> Object<ELFT>::Object(const ELFObjectFile<ELFT> &Obj) {
  const auto &ElfFile = *Obj.getELFFile();
  const auto &Ehdr = *ElfFile.getHeader();

  std::copy(Ehdr.e_ident, Ehdr.e_ident + 16, Ident);
  Type = Ehdr.e_type;
  Machine = Ehdr.e_machine;
  Version = Ehdr.e_version;
  Entry = Ehdr.e_entry;
  Flags = Ehdr.e_flags;

  Sections.push_back(make_unique<StringTableSection>());
  SectionNames = dyn_cast<StringTableSection>(Sections.back().get());
  SectionNames->Name = ".shstrtab";
  SectionNames->addString(SectionNames->Name);

  readSectionHeaders(ElfFile);
  readProgramHeaders(ElfFile);
}

template <class ELFT> void Object<ELFT>::sortSections() {
  // Put all sections in offset order. Maintain the ordering as closely as
  // possible while meeting that demand however.
  auto CompareSections = [](const SecPtr &A, const SecPtr &B) {
    return A->OriginalOffset < B->OriginalOffset;
  };
  std::stable_sort(std::begin(Sections), std::end(Sections), CompareSections);
}

template <class ELFT> void Object<ELFT>::assignOffsets() {
  // Decide file offsets and indexes.
  size_t PhdrSize = Segments.size() * sizeof(Elf_Phdr);
  // We can put section data after the ELF header and the program headers.
  uint64_t Offset = sizeof(Elf_Ehdr) + PhdrSize;
  uint64_t Index = 1;
  for (auto &Section : Sections) {
    // The segment can have a different alignment than the section. In the case
    // that there is a parent segment then as long as we satisfy the alignment
    // of the segment it should follow that that the section is aligned.
    if (Section->ParentSegment) {
      auto FirstInSeg = Section->ParentSegment->firstSection();
      if (FirstInSeg == Section.get()) {
        Offset = alignTo(Offset, Section->ParentSegment->Align);
        // There can be gaps at the start of a segment before the first section.
        // So first we assign the alignment of the segment and then assign the
        // location of the section from there
        Section->Offset =
            Offset + Section->OriginalOffset - Section->ParentSegment->Offset;
      }
      // We should respect interstitial gaps of allocated sections. We *must*
      // maintain the memory image so that addresses are preserved. As, with the
      // exception of SHT_NOBITS sections at the end of segments, the memory
      // image is a copy of the file image, we preserve the file image as well.
      // There's a strange case where a thread local SHT_NOBITS can cause the
      // memory image and file image to not be the same. This occurs, on some
      // systems, when a thread local SHT_NOBITS is between two SHT_PROGBITS
      // and the thread local SHT_NOBITS section is at the end of a TLS segment.
      // In this case to faithfully copy the segment file image we must use
      // relative offsets. In any other case this would be the same as using the
      // relative addresses so this should maintian the memory image as desired.
      Offset = FirstInSeg->Offset + Section->OriginalOffset -
               FirstInSeg->OriginalOffset;
    }
    // Alignment should have already been handled by the above if statement if
    // this if this section is in a segment. Technically this shouldn't do
    // anything bad if the alignments of the sections are all correct and the
    // file image isn't corrupted. Still in sticking with the motto "maintain
    // the file image" we should avoid messing up the file image if the
    // alignment disagrees with the file image.
    if (!Section->ParentSegment && Section->Align)
      Offset = alignTo(Offset, Section->Align);
    Section->Offset = Offset;
    Section->Index = Index++;
    if (Section->Type != SHT_NOBITS)
      Offset += Section->Size;
  }
  // 'offset' should now be just after all the section data so we should set the
  // section header table offset to be exactly here. This spot might not be
  // aligned properly however so we should align it as needed. For 32-bit ELF
  // this needs to be 4-byte aligned and on 64-bit it needs to be 8-byte aligned
  // so the size of ELFT::Addr is used to ensure this.
  Offset = alignTo(Offset, sizeof(typename ELFT::Addr));
  SHOffset = Offset;
}

template <class ELFT> void Object<ELFT>::finalize() {
  sortSections();
  assignOffsets();

  // Finalize SectionNames first so that we can assign name indexes.
  SectionNames->finalize();
  // Finally now that all offsets and indexes have been set we can finalize any
  // remaining issues.
  uint64_t Offset = SHOffset + sizeof(Elf_Shdr);
  for (auto &Section : Sections) {
    Section->HeaderOffset = Offset;
    Offset += sizeof(Elf_Shdr);
    Section->NameIndex = SectionNames->findIndex(Section->Name);
    Section->finalize();
  }

  for (auto &Segment : Segments)
    Segment->finalize();
}

template <class ELFT>
void Object<ELFT>::writeHeader(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart();
  Elf_Ehdr &Ehdr = *reinterpret_cast<Elf_Ehdr *>(Buf);
  std::copy(Ident, Ident + 16, Ehdr.e_ident);
  Ehdr.e_type = Type;
  Ehdr.e_machine = Machine;
  Ehdr.e_version = Version;
  Ehdr.e_entry = Entry;
  Ehdr.e_phoff = sizeof(Elf_Ehdr);
  Ehdr.e_shoff = SHOffset;
  Ehdr.e_flags = Flags;
  Ehdr.e_ehsize = sizeof(Elf_Ehdr);
  Ehdr.e_phentsize = sizeof(Elf_Phdr);
  Ehdr.e_phnum = Segments.size();
  Ehdr.e_shentsize = sizeof(Elf_Shdr);
  Ehdr.e_shnum = Sections.size();
  Ehdr.e_shstrndx = SectionNames->Index;
}

template <class ELFT>
void Object<ELFT>::writeProgramHeaders(FileOutputBuffer &Out) const {
  for (auto &Phdr : Segments)
    Phdr->template writeHeader<ELFT>(Out);
}

template <class ELFT>
void Object<ELFT>::writeSectionHeaders(FileOutputBuffer &Out) const {
  uint8_t *Buf = Out.getBufferStart() + SHOffset;
  // This reference serves to write the dummy section header at the begining
  // of the file.
  Elf_Shdr &Shdr = *reinterpret_cast<Elf_Shdr *>(Buf);
  Shdr.sh_name = 0;
  Shdr.sh_type = SHT_NULL;
  Shdr.sh_flags = 0;
  Shdr.sh_addr = 0;
  Shdr.sh_offset = 0;
  Shdr.sh_size = 0;
  Shdr.sh_link = 0;
  Shdr.sh_info = 0;
  Shdr.sh_addralign = 0;
  Shdr.sh_entsize = 0;

  for (auto &Section : Sections)
    Section->template writeHeader<ELFT>(Out);
}

template <class ELFT>
void Object<ELFT>::writeSectionData(FileOutputBuffer &Out) const {
  for (auto &Section : Sections)
    Section->writeSection(Out);
}

template <class ELFT> void Object<ELFT>::write(FileOutputBuffer &Out) {
  writeHeader(Out);
  writeProgramHeaders(Out);
  writeSectionData(Out);
  writeSectionHeaders(Out);
}

template class Object<ELF64LE>;
template class Object<ELF64BE>;
template class Object<ELF32LE>;
template class Object<ELF32BE>;
