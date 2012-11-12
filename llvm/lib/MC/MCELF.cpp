//===- lib/MC/MCELF.cpp - MC ELF ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "MCELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/Support/ELF.h"

namespace llvm {

void MCELF::SetBinding(MCSymbolData &SD, unsigned Binding) {
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STB_Shift);
  SD.setFlags(OtherFlags | (Binding << ELF_STB_Shift));
}

unsigned MCELF::GetBinding(const MCSymbolData &SD) {
  uint32_t Binding = (SD.getFlags() & (0xf << ELF_STB_Shift)) >> ELF_STB_Shift;
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  return Binding;
}

void MCELF::SetType(MCSymbolData &SD, unsigned Type) {
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_FILE || Type == ELF::STT_COMMON ||
         Type == ELF::STT_TLS || Type == ELF::STT_GNU_IFUNC);

  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STT_Shift);
  SD.setFlags(OtherFlags | (Type << ELF_STT_Shift));
}

unsigned MCELF::GetType(const MCSymbolData &SD) {
  uint32_t Type = (SD.getFlags() & (0xf << ELF_STT_Shift)) >> ELF_STT_Shift;
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_FILE || Type == ELF::STT_COMMON ||
         Type == ELF::STT_TLS || Type == ELF::STT_GNU_IFUNC);
  return Type;
}

void MCELF::SetVisibility(MCSymbolData &SD, unsigned Visibility) {
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);

  uint32_t OtherFlags = SD.getFlags() & ~(0x3 << ELF_STV_Shift);
  SD.setFlags(OtherFlags | (Visibility << ELF_STV_Shift));
}

unsigned MCELF::GetVisibility(MCSymbolData &SD) {
  unsigned Visibility =
    (SD.getFlags() & (0x3 << ELF_STV_Shift)) >> ELF_STV_Shift;
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);
  return Visibility;
}

}
