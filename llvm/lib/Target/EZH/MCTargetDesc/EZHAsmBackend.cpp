//===-- EZHAsmBackend.cpp - EZH Assembler Backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHFixupKinds.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Prepare value for the target space
static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
    return Value;
  case EZH::FIXUP_EZH_8_PCREL:
    // PC evaluates to Loc + 8. The Value passed in is Target - Loc.
    // We need (Target - (Loc + 8)) = Target - Loc - 8 = Value - 8.
    // The hardware expects a positive unsigned offset in words.
    // Since literal pool is always placed after the instruction, Value should
    // be > 8.
    return ((Value - 8) >> 2) & 0xFF;
  case EZH::FIXUP_EZH_21:
  case EZH::FIXUP_EZH_21_F:
  case EZH::FIXUP_EZH_25:
    // GOTOs and GOSUBs in EZH use absolute word addressing, not PC-relative.
    return Value >> 2;
  case EZH::FIXUP_EZH_32:
  case EZH::FIXUP_EZH_HI16:
  case EZH::FIXUP_EZH_LO16:
    return Value;
  default:
    llvm_unreachable("Unknown fixup kind!");
  }
}

namespace {
class EZHAsmBackend : public MCAsmBackend {
  Triple::OSType OSType;

public:
  EZHAsmBackend(const Target &T, Triple::OSType OST)
      : MCAsmBackend(llvm::endianness::little), OSType(OST) {}

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  uint8_t *Data, uint64_t Value, bool IsResolved) override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  bool shouldForceRelocation(const MCFixup &Fixup) const {
    switch (static_cast<unsigned>(Fixup.getKind())) {
    case EZH::FIXUP_EZH_21:
    case EZH::FIXUP_EZH_21_F:
    case EZH::FIXUP_EZH_25:
      // GOTOs and GOSUBs are absolute word addresses.
      // We must force the linker to process them to handle section offsets.
      return true;
    default:
      return false;
    }
  }
};

bool EZHAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                 const MCSubtargetInfo *STI) const {
  if ((Count % 4) != 0)
    return false;

  for (uint64_t i = 0; i < Count; i += 4)
    OS.write("\x15\0\0\0", 4);

  return true;
}

void EZHAsmBackend::applyFixup(const MCFragment &F, const MCFixup &Fixup,
                               const MCValue &Target, uint8_t *Data,
                               uint64_t Value, bool IsResolved) {
  if (shouldForceRelocation(Fixup))
    IsResolved = false;

  maybeAddReloc(F, Fixup, Target, Value, IsResolved);

  MCFixupKind Kind = Fixup.getKind();
  if (mc::isRelocation(Kind))
    return;
  Value = adjustFixupValue(static_cast<unsigned>(Kind), Value);
  if (!Value)
    return; // This value doesn't change the encoding

  // Read 32-bit little-endian instruction
  uint32_t CurVal = llvm::support::endian::read32le(Data);

  uint64_t Mask =
      (static_cast<uint64_t>(-1) >> (64 - getFixupKindInfo(Kind).TargetSize));
  CurVal |= (Value & Mask) << getFixupKindInfo(Kind).TargetOffset;

  // Write 32-bit little-endian instruction back
  llvm::support::endian::write32le(Data, CurVal);
}

std::unique_ptr<MCObjectTargetWriter>
EZHAsmBackend::createObjectTargetWriter() const {
  return createEZHELFObjectWriter(MCELFObjectTargetWriter::getOSABI(OSType));
}

MCFixupKindInfo EZHAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  static const MCFixupKindInfo Infos[EZH::NumTargetFixupKinds] = {
      // This table *must* be in same the order of fixup_* kinds in
      // EZHFixupKinds.h.
      // Note: The number of bits indicated here are assumed to be contiguous.
      //   This does not hold true for LANAI_21 and LANAI_21_F which are applied
      //   to bits 0x7cffff and 0x7cfffc, respectively. Since the 'bits' counts
      //   here are used only for cosmetic purposes, we set the size to 16 bits
      //   for these 21-bit relocation as llvm/lib/MC/MCAsmStreamer.cpp checks
      //   no bits are set in the fixup range.
      //
      // name          offset bits flags
      {"FIXUP_EZH_NONE", 0, 32, 0},  {"FIXUP_EZH_21", 11, 21, 0},
      {"FIXUP_EZH_21_F", 11, 21, 0}, {"FIXUP_EZH_25", 5, 27, 0},
      {"FIXUP_EZH_32", 0, 32, 0},    {"FIXUP_EZH_HI16", 16, 16, 0},
      {"FIXUP_EZH_LO16", 0, 16, 0},  {"FIXUP_EZH_8_PCREL", 24, 8, 0}};

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < EZH::NumTargetFixupKinds &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}

std::optional<MCFixupKind> EZHAsmBackend::getFixupKind(StringRef Name) const {
  unsigned Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(NAME, ID) .Case(#NAME, ID)
#include "llvm/BinaryFormat/ELFRelocs/EZH.def"
#undef ELF_RELOC
                      .Case("BFD_RELOC_NONE", ELF::R_EZH_NONE)
                      .Case("BFD_RELOC_32", ELF::R_EZH_32)
                      .Default(-1u);
  if (Type != -1u)
    return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  return std::nullopt;
}

} // namespace

MCAsmBackend *llvm::createEZHAsmBackend(const Target &T,
                                        const MCSubtargetInfo &STI,
                                        const MCRegisterInfo & /*MRI*/,
                                        const MCTargetOptions & /*Options*/) {
  const Triple &TT = STI.getTargetTriple();
  if (!TT.isOSBinFormatELF())
    llvm_unreachable("OS not supported");

  return new EZHAsmBackend(T, TT.getOS());
}
