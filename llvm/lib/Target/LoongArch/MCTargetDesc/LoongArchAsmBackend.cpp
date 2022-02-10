//===-- LoongArchAsmBackend.cpp - LoongArch Assembler Backend -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LoongArchAsmBackend class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

#define DEBUG_TYPE "loongarch-asmbackend"

using namespace llvm;

void LoongArchAsmBackend::applyFixup(const MCAssembler &Asm,
                                     const MCFixup &Fixup,
                                     const MCValue &Target,
                                     MutableArrayRef<char> Data, uint64_t Value,
                                     bool IsResolved,
                                     const MCSubtargetInfo *STI) const {
  // TODO: Apply the Value for given Fixup into the provided data fragment.
  return;
}

bool LoongArchAsmBackend::shouldForceRelocation(const MCAssembler &Asm,
                                                const MCFixup &Fixup,
                                                const MCValue &Target) {
  // TODO: Determine which relocation require special processing at linking
  // time.
  return false;
}

bool LoongArchAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                       const MCSubtargetInfo *STI) const {
  // Check for byte count not multiple of instruction word size
  if (Count % 4 != 0)
    return false;

  // The nop on LoongArch is andi r0, r0, 0.
  for (; Count >= 4; Count -= 4)
    support::endian::write<uint32_t>(OS, 0x03400000, support::little);

  return true;
}

std::unique_ptr<MCObjectTargetWriter>
LoongArchAsmBackend::createObjectTargetWriter() const {
  return createLoongArchELFObjectWriter(OSABI, Is64Bit);
}

MCAsmBackend *llvm::createLoongArchAsmBackend(const Target &T,
                                              const MCSubtargetInfo &STI,
                                              const MCRegisterInfo &MRI,
                                              const MCTargetOptions &Options) {
  const Triple &TT = STI.getTargetTriple();
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  return new LoongArchAsmBackend(STI, OSABI, TT.isArch64Bit());
}
