//===- elf.cpp - Code to read ELF data structures for llvm-elf2bin --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-elf2bin.h"

using namespace llvm;
using namespace llvm::object;

template <typename ELFT>
static std::vector<Segment>
get_segments(InputObject &inobj, ELFObjectFile<ELFT> &elfobj, bool physical) {
  std::vector<Segment> segments;

  Expected<ArrayRef<typename ELFT::Phdr>> phdrs_or_err =
      elfobj.getELFFile().program_headers();
  if (!phdrs_or_err) {
    fatal(inobj, "unable to read program header table",
          phdrs_or_err.takeError());
    return segments;
  }

  for (const typename ELFT::Phdr &phdr : *phdrs_or_err) {
    Segment seg;
    seg.fileoffset = phdr.p_offset;
    seg.baseaddr = physical ? phdr.p_paddr : phdr.p_vaddr;
    seg.filesize = phdr.p_filesz;
    seg.memsize = phdr.p_memsz;
    segments.push_back(seg);
  }

  return segments;
}

template <typename ELFT>
static uint64_t get_entry_point(ELFObjectFile<ELFT> &obj) {
  return obj.getELFFile().getHeader().e_entry;
}

std::vector<Segment> InputObject::segments(bool physical) {
  if (auto *specific = dyn_cast<ELF32LEObjectFile>(elf.get()))
    return get_segments(*this, *specific, physical);
  if (auto *specific = dyn_cast<ELF32BEObjectFile>(elf.get()))
    return get_segments(*this, *specific, physical);
  if (auto *specific = dyn_cast<ELF64LEObjectFile>(elf.get()))
    return get_segments(*this, *specific, physical);
  if (auto *specific = dyn_cast<ELF64BEObjectFile>(elf.get()))
    return get_segments(*this, *specific, physical);
  llvm_unreachable("unexpected subclass of ELFOBjectFileBase");
}

uint64_t InputObject::entry_point() {
  if (auto *specific = dyn_cast<ELF32LEObjectFile>(elf.get()))
    return get_entry_point(*specific);
  if (auto *specific = dyn_cast<ELF32BEObjectFile>(elf.get()))
    return get_entry_point(*specific);
  if (auto *specific = dyn_cast<ELF64LEObjectFile>(elf.get()))
    return get_entry_point(*specific);
  if (auto *specific = dyn_cast<ELF64BEObjectFile>(elf.get()))
    return get_entry_point(*specific);
  llvm_unreachable("unexpected subclass of ELFOBjectFileBase");
}
