//===- ELFWriter.cpp - Low-level ELF structure writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWP/ELFWriter.h"
#include "llvm/BinaryFormat/ELF.h"

using namespace llvm;

static void writeWord(support::endian::Writer &W, bool Is64Bit, uint64_t Val) {
  if (Is64Bit)
    W.write<uint64_t>(Val);
  else
    W.write<uint32_t>(Val);
}

void ELF::writeHeader(support::endian::Writer &W, bool Is64Bit, uint8_t OSABI,
                      uint8_t ABIVersion, uint16_t EMachine, uint32_t EFlags,
                      uint64_t SHOff, uint16_t SHNum, uint16_t SHStrNdx) {
  W.OS << ElfMagic;
  W.OS << char(Is64Bit ? ELFCLASS64 : ELFCLASS32);
  W.OS << char(W.Endian == llvm::endianness::little ? ELFDATA2LSB
                                                    : ELFDATA2MSB);
  W.OS << char(EV_CURRENT);
  W.OS << char(OSABI);
  W.OS << char(ABIVersion);
  W.OS.write_zeros(EI_NIDENT - EI_PAD);

  W.write<uint16_t>(ET_REL);
  W.write<uint16_t>(EMachine);
  W.write<uint32_t>(EV_CURRENT);
  writeWord(W, Is64Bit, 0); // e_entry
  writeWord(W, Is64Bit, 0); // e_phoff
  writeWord(W, Is64Bit, SHOff);
  W.write<uint32_t>(EFlags);
  W.write<uint16_t>(Is64Bit ? sizeof(Elf64_Ehdr) : sizeof(Elf32_Ehdr));
  W.write<uint16_t>(0); // e_phentsize
  W.write<uint16_t>(0); // e_phnum
  W.write<uint16_t>(Is64Bit ? sizeof(Elf64_Shdr) : sizeof(Elf32_Shdr));
  W.write<uint16_t>(SHNum);
  W.write<uint16_t>(SHStrNdx);
}

void ELF::writeSectionHeader(support::endian::Writer &W, bool Is64Bit,
                             uint32_t Name, uint32_t Type, uint64_t Flags,
                             uint64_t Address, uint64_t Offset, uint64_t Size,
                             uint32_t Link, uint32_t Info, uint64_t Alignment,
                             uint64_t EntrySize) {
  W.write<uint32_t>(Name);
  W.write<uint32_t>(Type);
  writeWord(W, Is64Bit, Flags);
  writeWord(W, Is64Bit, Address);
  writeWord(W, Is64Bit, Offset);
  writeWord(W, Is64Bit, Size);
  W.write<uint32_t>(Link);
  W.write<uint32_t>(Info);
  writeWord(W, Is64Bit, Alignment);
  writeWord(W, Is64Bit, EntrySize);
}
