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
#include "LoongArchFixupKinds.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

#define DEBUG_TYPE "loongarch-asmbackend"

using namespace llvm;

Optional<MCFixupKind> LoongArchAsmBackend::getFixupKind(StringRef Name) const {
  if (STI.getTargetTriple().isOSBinFormatELF()) {
    auto Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(X, Y) .Case(#X, Y)
#include "llvm/BinaryFormat/ELFRelocs/LoongArch.def"
#undef ELF_RELOC
                    .Case("BFD_RELOC_NONE", ELF::R_LARCH_NONE)
                    .Case("BFD_RELOC_32", ELF::R_LARCH_32)
                    .Case("BFD_RELOC_64", ELF::R_LARCH_64)
                    .Default(-1u);
    if (Type != -1u)
      return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  }
  return None;
}

const MCFixupKindInfo &
LoongArchAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[] = {
      // This table *must* be in the order that the fixup_* kinds are defined in
      // LoongArchFixupKinds.h.
      //
      // {name, offset, bits, flags}
      {"fixup_loongarch_b16", 10, 16, MCFixupKindInfo::FKF_IsPCRel},
      {"fixup_loongarch_b21", 0, 26, MCFixupKindInfo::FKF_IsPCRel},
      {"fixup_loongarch_b26", 0, 26, MCFixupKindInfo::FKF_IsPCRel},
      {"fixup_loongarch_abs_hi20", 5, 20, 0},
      {"fixup_loongarch_abs_lo12", 10, 12, 0},
      {"fixup_loongarch_abs64_lo20", 5, 20, 0},
      {"fixup_loongarch_abs64_hi12", 10, 12, 0},
      {"fixup_loongarch_tls_le_hi20", 5, 20, 0},
      {"fixup_loongarch_tls_le_lo12", 10, 12, 0},
      {"fixup_loongarch_tls_le64_lo20", 5, 20, 0},
      {"fixup_loongarch_tls_le64_hi12", 10, 12, 0},
      // TODO: Add more fixup kinds.
  };

  static_assert((std::size(Infos)) == LoongArch::NumTargetFixupKinds,
                "Not all fixup kinds added to Infos array");

  // Fixup kinds from .reloc directive are like R_LARCH_NONE. They
  // do not require any extra processing.
  if (Kind >= FirstLiteralRelocationKind)
    return MCAsmBackend::getFixupKindInfo(FK_NONE);

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}

static void reportOutOfRangeError(MCContext &Ctx, SMLoc Loc, unsigned N) {
  Ctx.reportError(Loc, "fixup value out of range [" + Twine(llvm::minIntN(N)) +
                           ", " + Twine(llvm::maxIntN(N)) + "]");
}

static uint64_t adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext &Ctx) {
  switch (Fixup.getTargetKind()) {
  default:
    llvm_unreachable("Unknown fixup kind");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
    return Value;
  case LoongArch::fixup_loongarch_b16: {
    if (!isInt<18>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 18);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return (Value >> 2) & 0xffff;
  }
  case LoongArch::fixup_loongarch_b21: {
    if (!isInt<23>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 23);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return ((Value & 0x3fffc) << 8) | ((Value >> 18) & 0x1f);
  }
  case LoongArch::fixup_loongarch_b26: {
    if (!isInt<28>(Value))
      reportOutOfRangeError(Ctx, Fixup.getLoc(), 28);
    if (Value % 4)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned");
    return ((Value & 0x3fffc) << 8) | ((Value >> 18) & 0x3ff);
  }
  case LoongArch::fixup_loongarch_abs_hi20:
  case LoongArch::fixup_loongarch_tls_le_hi20:
    return (Value >> 12) & 0xfffff;
  case LoongArch::fixup_loongarch_abs_lo12:
  case LoongArch::fixup_loongarch_tls_le_lo12:
    return Value & 0xfff;
  case LoongArch::fixup_loongarch_abs64_lo20:
  case LoongArch::fixup_loongarch_tls_le64_lo20:
    return (Value >> 32) & 0xfffff;
  case LoongArch::fixup_loongarch_abs64_hi12:
  case LoongArch::fixup_loongarch_tls_le64_hi12:
    return (Value >> 52) & 0xfff;
  }
}

void LoongArchAsmBackend::applyFixup(const MCAssembler &Asm,
                                     const MCFixup &Fixup,
                                     const MCValue &Target,
                                     MutableArrayRef<char> Data, uint64_t Value,
                                     bool IsResolved,
                                     const MCSubtargetInfo *STI) const {
  if (!Value)
    return; // Doesn't change encoding.

  MCFixupKind Kind = Fixup.getKind();
  if (Kind >= FirstLiteralRelocationKind)
    return;
  MCFixupKindInfo Info = getFixupKindInfo(Kind);
  MCContext &Ctx = Asm.getContext();

  // Apply any target-specific value adjustments.
  Value = adjustFixupValue(Fixup, Value, Ctx);

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned Offset = Fixup.getOffset();
  unsigned NumBytes = alignTo(Info.TargetSize + Info.TargetOffset, 8) / 8;

  assert(Offset + NumBytes <= Data.size() && "Invalid fixup offset!");
  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  for (unsigned I = 0; I != NumBytes; ++I) {
    Data[Offset + I] |= uint8_t((Value >> (I * 8)) & 0xff);
  }
}

bool LoongArchAsmBackend::shouldForceRelocation(const MCAssembler &Asm,
                                                const MCFixup &Fixup,
                                                const MCValue &Target) {
  if (Fixup.getKind() >= FirstLiteralRelocationKind)
    return true;
  switch (Fixup.getTargetKind()) {
  default:
    return false;
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
    return !Target.isAbsolute();
  }
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
