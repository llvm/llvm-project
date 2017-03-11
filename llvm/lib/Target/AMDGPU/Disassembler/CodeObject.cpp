//===- CodeObject.cpp - ELF object file implementation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the HSA Code Object file class.
//
//===----------------------------------------------------------------------===//

#include "CodeObject.h"
#include "AMDGPUPTNote.h"

namespace llvm {

using namespace object;

const ELFNote* getNext(const ELFNote &N) {
  return reinterpret_cast<const ELFNote *>(
    N.getDesc().data() + alignTo(N.descsz, ELFNote::ALIGN));
}

Expected<const amd_kernel_code_t *> KernelSym::getAmdKernelCodeT(
  const HSACodeObject *CodeObject) const {
  auto TextOr = CodeObject->getTextSection();
  if (!TextOr) {
    return TextOr.takeError();
  }

  return getAmdKernelCodeT(CodeObject, *TextOr);
}

Expected<const amd_kernel_code_t *> KernelSym::getAmdKernelCodeT(
  const HSACodeObject * CodeObject,
  const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);

  auto ArrayOr = CodeObject->getELFFile()->getSectionContentsAsArray<uint8_t>(Text);
  if (!ArrayOr)
    return ArrayOr.takeError();

  auto SectionOffsetOr = getSectionOffset(CodeObject, Text);
  if (!SectionOffsetOr)
    return SectionOffsetOr.takeError();

  return reinterpret_cast<const amd_kernel_code_t *>((*ArrayOr).data() + *SectionOffsetOr);
}

Expected<uint64_t> KernelSym::getAddress(
  const HSACodeObject *CodeObject) const {
  auto TextOr = CodeObject->getTextSection();
  if (!TextOr) {
    return TextOr.takeError();
  }
  return getAddress(CodeObject, TextOr.get());
}

Expected<uint64_t> KernelSym::getAddress(
  const HSACodeObject *CodeObject,
  const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);
  auto ElfHeader = CodeObject->getELFFile()->getHeader();
  if (ElfHeader->e_type == ELF::ET_REL) {
    return st_value + Text->sh_addr;
  }

  return st_value;
}

Expected<uint64_t> KernelSym::getSectionOffset(
  const HSACodeObject *CodeObject) const {
  auto TextOr = CodeObject->getTextSection();
  if (!TextOr) {
    return TextOr.takeError();
  }
  return getSectionOffset(CodeObject, TextOr.get());
}


Expected<uint64_t> KernelSym::getSectionOffset(
  const HSACodeObject *CodeObject,
  const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);

  auto AddressOr = getAddress(CodeObject, Text);
  if (!AddressOr)
    return AddressOr.takeError();

  return *AddressOr - Text->sh_addr;
}

Expected<uint64_t> KernelSym::getCodeOffset(
  const HSACodeObject *CodeObject,
  const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);

  auto SectionOffsetOr = getSectionOffset(CodeObject, Text);
  if (!SectionOffsetOr)
    return SectionOffsetOr.takeError();

  auto KernelCodeTOr = getAmdKernelCodeT(CodeObject, Text);
  if (!KernelCodeTOr)
    return KernelCodeTOr.takeError();

  return *SectionOffsetOr + (*KernelCodeTOr)->kernel_code_entry_byte_offset;
}


Expected<const KernelSym *> KernelSym::asKernelSym(const HSACodeObject::Elf_Sym *Sym) {
  if (Sym->getType() != ELF::STT_AMDGPU_HSA_KERNEL)
    return createError("invalid symbol type");

  return static_cast<const KernelSym *>(Sym);
}

void HSACodeObject::InitMarkers() const {
  auto TextSecOr = getTextSection();
  if (!TextSecOr)
    return;
  auto TextSec = TextSecOr.get();

  KernelMarkers.push_back(TextSec->sh_size);

  for (const auto &Sym : kernels()) {
    auto ExpectedKernel = KernelSym::asKernelSym(getSymbol(Sym.getRawDataRefImpl()));
    if (!ExpectedKernel) {
      consumeError(ExpectedKernel.takeError());
      report_fatal_error("invalid kernel symbol");
    }
    auto Kernel = ExpectedKernel.get();

    auto ExpectedKernelOffset = Kernel->getSectionOffset(this, TextSec);
    if (!ExpectedKernelOffset) {
      consumeError(ExpectedKernelOffset.takeError());
      report_fatal_error("invalid kernel offset");
    }
    KernelMarkers.push_back(*ExpectedKernelOffset);

    auto ExpectedCodeOffset = Kernel->getCodeOffset(this, TextSec);
    if (!ExpectedCodeOffset) {
      consumeError(ExpectedCodeOffset.takeError());
      report_fatal_error("invalid kernel code offset");
    }
    KernelMarkers.push_back(*ExpectedCodeOffset);
  }

  array_pod_sort(KernelMarkers.begin(), KernelMarkers.end());
}

HSACodeObject::note_iterator HSACodeObject::notes_begin() const {
  if (auto NotesOr = getNoteSection()) {
    if (auto ContentsOr = getELFFile()->getSectionContentsAsArray<uint8_t>(*NotesOr))
      return const_varsize_item_iterator<ELFNote>(*ContentsOr);
  }

  return const_varsize_item_iterator<ELFNote>();
}

HSACodeObject::note_iterator HSACodeObject::notes_end() const {
  return const_varsize_item_iterator<ELFNote>();
}

iterator_range<HSACodeObject::note_iterator> HSACodeObject::notes() const {
  return make_range(notes_begin(), notes_end());
}

kernel_sym_iterator HSACodeObject::kernels_begin() const {
  auto TextIdxOr = getTextSectionIdx();
  if (!TextIdxOr)
    return kernels_end();

  auto TextIdx = TextIdxOr.get();
  return kernel_sym_iterator(symbol_begin(), symbol_end(),
    [this, TextIdx](const SymbolRef &Sym)->bool {

      auto ExpectedKernel = KernelSym::asKernelSym(getSymbol(Sym.getRawDataRefImpl()));
      if (!ExpectedKernel) {
        consumeError(ExpectedKernel.takeError());
        return false;
      }

      auto Kernel = ExpectedKernel.get();
      if (Kernel->st_shndx != TextIdx)
        return false;

      return true;
    });
}

kernel_sym_iterator HSACodeObject::kernels_end() const {
  return kernel_sym_iterator(symbol_end(), symbol_end(),
                             [](const SymbolRef&){return true;});
}

iterator_range<kernel_sym_iterator> HSACodeObject::kernels() const {
  return make_range(kernels_begin(), kernels_end());
}

Expected<ArrayRef<uint8_t>> HSACodeObject::getKernelCode(const KernelSym *Kernel) const {
  auto KernelCodeTOr = Kernel->getAmdKernelCodeT(this);
  if (!KernelCodeTOr)
    return KernelCodeTOr.takeError();

  auto TextOr = getTextSection();
  if (!TextOr)
    return TextOr.takeError();

  auto SecBytesOr = getELFFile()->getSectionContentsAsArray<uint8_t>(*TextOr);
  if (!SecBytesOr)
    return SecBytesOr.takeError();

  auto CodeStartOr = Kernel->getCodeOffset(this, *TextOr);
  if (!CodeStartOr)
    return CodeStartOr.takeError();
  uint64_t CodeStart = CodeStartOr.get();

  auto CodeEndI = std::upper_bound(KernelMarkers.begin(),
                                   KernelMarkers.end(),
                                   CodeStart);
  uint64_t CodeEnd = CodeStart;
  if (CodeEndI != KernelMarkers.end())
    CodeEnd = *CodeEndI;
  
  return SecBytesOr->slice(CodeStart, CodeEnd - CodeStart);
}

Expected<const HSACodeObject::Elf_Shdr *>
HSACodeObject::getSectionByName(StringRef Name) const {
  auto ELF = getELFFile();
  auto SectionsOr = ELF->sections();
  if (!SectionsOr)
    return SectionsOr.takeError();
  
  for (const auto &Sec : *SectionsOr) {
    auto SecNameOr = ELF->getSectionName(&Sec);
    if (!SecNameOr) {
      return SecNameOr.takeError();
    } else if (*SecNameOr == Name) {
      return Expected<const Elf_Shdr *>(&Sec);
    }
  }
  return createError("invalid section index");
}

Expected<uint32_t> HSACodeObject::getSectionIdxByName(StringRef Name) const {
  auto ELF = getELFFile();
  uint32_t Idx = 0;
  auto SectionsOr = ELF->sections();
  if (!SectionsOr)
    return SectionsOr.takeError();

  for (const auto &Sec : *SectionsOr) {
    auto SecNameOr = ELF->getSectionName(&Sec);
    if (!SecNameOr) {
      return SecNameOr.takeError();
    } else if (*SecNameOr == Name) {
      return Idx;
    }
    ++Idx;
  }
  return createError("invalid section index");
}

Expected<uint32_t> HSACodeObject::getTextSectionIdx() const {
  if (auto IdxOr = getSectionIdxByName(".text")) {
    auto SecOr = getELFFile()->getSection(*IdxOr);
    if (SecOr || isSectionText(toDRI(*SecOr)))
      return IdxOr;
  }
  return createError("invalid section index");
}

Expected<uint32_t> HSACodeObject::getNoteSectionIdx() const {
  return getSectionIdxByName(AMDGPU::ElfNote::SectionName);
}

Expected<const HSACodeObject::Elf_Shdr *> HSACodeObject::getTextSection() const {
  if (auto IdxOr = getTextSectionIdx())
    return getELFFile()->getSection(*IdxOr);

  return createError("invalid section index");
}

Expected<const HSACodeObject::Elf_Shdr *> HSACodeObject::getNoteSection() const {
  if (auto IdxOr = getNoteSectionIdx())
    return getELFFile()->getSection(*IdxOr);

  return createError("invalid section index");
}

} // namespace llvm
