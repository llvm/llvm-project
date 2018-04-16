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

Expected<uint64_t>
FunctionSym::getAddress(const HSACodeObject *CodeObject) const {
  auto TextOr = CodeObject->getTextSection();
  if (!TextOr) {
    return TextOr.takeError();
  }
  return getAddress(CodeObject, TextOr.get());
}

Expected<uint64_t>
FunctionSym::getAddress(const HSACodeObject *CodeObject,
                        const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);
  auto ElfHeader = CodeObject->getELFFile()->getHeader();
  if (ElfHeader->e_type == ELF::ET_REL) {
    return st_value + Text->sh_addr;
  }

  return st_value;
}

Expected<uint64_t>
FunctionSym::getSectionOffset(const HSACodeObject *CodeObject) const {
  auto TextOr = CodeObject->getTextSection();
  if (!TextOr) {
    return TextOr.takeError();
  }
  return getSectionOffset(CodeObject, TextOr.get());
}

Expected<uint64_t> FunctionSym::getSectionOffset(
    const HSACodeObject *CodeObject,
    const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);

  auto AddressOr = getAddress(CodeObject, Text);
  if (!AddressOr)
    return AddressOr.takeError();

  return *AddressOr - Text->sh_addr;
}

Expected<uint64_t> FunctionSym::getCodeOffset(
    const HSACodeObject *CodeObject,
    const object::ELF64LEObjectFile::Elf_Shdr *Text) const {
  assert(Text);

  auto SectionOffsetOr = getSectionOffset(CodeObject, Text);
  if (!SectionOffsetOr)
    return SectionOffsetOr.takeError();

  return *SectionOffsetOr;
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

Expected<const FunctionSym *>
FunctionSym::asFunctionSym(const HSACodeObject::Elf_Sym *Sym) {
  if (Sym->getType() != ELF::STT_FUNC &&
      Sym->getType() != ELF::STT_AMDGPU_HSA_KERNEL)
    return createError("invalid symbol type");

  return static_cast<const FunctionSym *>(Sym);
}

Expected<const KernelSym *> KernelSym::asKernelSym(const FunctionSym *Sym) {
  if (Sym->getType() != ELF::STT_AMDGPU_HSA_KERNEL)
    return createError("invalid symbol type");

  return static_cast<const KernelSym *>(Sym);
}

void HSACodeObject::InitMarkers() const {
  auto TextSecOr = getTextSection();
  if (!TextSecOr)
    return;
  auto TextSec = TextSecOr.get();

  FunctionMarkers.push_back(TextSec->sh_size);

  for (const auto &Sym : functions()) {
    auto ExpectedFunction =
        FunctionSym::asFunctionSym(getSymbol(Sym.getRawDataRefImpl()));
    if (!ExpectedFunction) {
      consumeError(ExpectedFunction.takeError());
      report_fatal_error("invalid function symbol");
    }
    auto Function = ExpectedFunction.get();

    auto ExpectedSectionOffset = Function->getSectionOffset(this, TextSec);
    if (!ExpectedSectionOffset) {
      consumeError(ExpectedSectionOffset.takeError());
      report_fatal_error("invalid section offset");
    }
    FunctionMarkers.push_back(*ExpectedSectionOffset);

    auto ExpectedKernel = KernelSym::asKernelSym(Function);
    if (ExpectedKernel) {
      auto Kernel = ExpectedKernel.get();

      auto ExpectedCodeOffset = Kernel->getCodeOffset(this, TextSec);
      if (!ExpectedCodeOffset) {
        consumeError(ExpectedCodeOffset.takeError());
        report_fatal_error("invalid kernel code offset");
      }

      FunctionMarkers.push_back(*ExpectedCodeOffset);
    } else {
      consumeError(ExpectedKernel.takeError());
    }
  }

  array_pod_sort(FunctionMarkers.begin(), FunctionMarkers.end());
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

function_sym_iterator HSACodeObject::functions_begin() const {
  auto TextIdxOr = getTextSectionIdx();
  if (!TextIdxOr)
    return functions_end();

  auto TextIdx = TextIdxOr.get();
  return function_sym_iterator(symbol_begin(), symbol_end(),
                               [this, TextIdx](const SymbolRef &Sym) -> bool {
                                 auto ExpectedFunction =
                                     FunctionSym::asFunctionSym(
                                         getSymbol(Sym.getRawDataRefImpl()));
                                 if (!ExpectedFunction) {
                                   consumeError(ExpectedFunction.takeError());
                                   return false;
                                 }
                                 auto Function = ExpectedFunction.get();
                                 if (Function->st_shndx != TextIdx)
                                   return false;
                                 return true;
                               });
}

function_sym_iterator HSACodeObject::functions_end() const {
  return function_sym_iterator(symbol_end(), symbol_end(),
                               [](const SymbolRef &) { return true; });
}

iterator_range<function_sym_iterator> HSACodeObject::functions() const {
  return make_range(functions_begin(), functions_end());
}

Expected<ArrayRef<uint8_t>>
HSACodeObject::getCode(const FunctionSym *Function) const {
  auto TextOr = getTextSection();
  if (!TextOr)
    return TextOr.takeError();

  auto SecBytesOr = getELFFile()->getSectionContentsAsArray<uint8_t>(*TextOr);
  if (!SecBytesOr)
    return SecBytesOr.takeError();

  auto CodeStartOr = Function->getCodeOffset(this, *TextOr);
  if (!CodeStartOr)
    return CodeStartOr.takeError();
  uint64_t CodeStart = CodeStartOr.get();

  auto ExpectedKernel = KernelSym::asKernelSym(Function);
  if (ExpectedKernel) {
    auto Kernel = ExpectedKernel.get();
    auto KernelCodeStartOr = Kernel->getCodeOffset(this, *TextOr);
    if (!KernelCodeStartOr)
      return KernelCodeStartOr.takeError();
    CodeStart = KernelCodeStartOr.get();
  } else {
    consumeError(ExpectedKernel.takeError());
  }

  auto CodeEndI = std::upper_bound(FunctionMarkers.begin(),
                                   FunctionMarkers.end(), CodeStart);
  uint64_t CodeEnd = CodeStart;
  if (CodeEndI != FunctionMarkers.end())
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
