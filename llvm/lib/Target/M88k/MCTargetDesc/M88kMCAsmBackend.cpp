//===-- M88kMCAsmBackend.cpp - M88k assembler backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/M88kMCFixups.h"
#include "MCTargetDesc/M88kMCTargetDesc.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

namespace {
// Value is a fully-resolved relocation value: Symbol + Addend [- Pivot].
// Return the bits that should be installed in a relocation field for
// fixup kind Kind.
uint64_t extractBitsForFixup(MCFixupKind Kind, uint64_t Value,
                             const MCFixup &Fixup, MCContext &Ctx) {
  if (Kind < FirstTargetFixupKind)
    return Value;

  auto checkFixupInRange = [&](int64_t Min, int64_t Max) -> bool {
    int64_t SVal = int64_t(Value);
    if (SVal < Min || SVal > Max) {
      Ctx.reportError(Fixup.getLoc(), "operand out of range (" + Twine(SVal) +
                                          " not between " + Twine(Min) +
                                          " and " + Twine(Max) + ")");
      return false;
    }
    return true;
  };

  auto handlePCRelFixupValue = [&](unsigned W) -> uint64_t {
    if (Value % 4 != 0)
      Ctx.reportError(Fixup.getLoc(), "Non-even PC relative offset.");
    if (!checkFixupInRange(minIntN(W) * 2, maxIntN(W) * 2))
      return 0;
    return (int64_t)Value >> 2;
  };

  switch (unsigned(Kind)) {
  case M88k::FK_88K_DISP16:
    return handlePCRelFixupValue(16);
  case M88k::FK_88K_DISP26:
    return handlePCRelFixupValue(26);

  case M88k::FK_88K_HI:
  case M88k::FK_88K_LO:
    if (!checkFixupInRange(0, maxUIntN(16)))
      return 0;
    return Value;

  case M88k::FK_88K_NONE:
    return 0;
  }

  llvm_unreachable("Unknown fixup kind!");
}

class M88kMCAsmBackend : public MCAsmBackend {
  uint8_t OSABI;

public:
  M88kMCAsmBackend(uint8_t osABI) : MCAsmBackend(support::big), OSABI(osABI) {}

  // Override MCAsmBackend
  unsigned getNumFixupKinds() const override;
  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
  Optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;
  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *Fragment,
                            const MCAsmLayout &Layout) const override;
  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target) override;
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createM88kObjectWriter(OSABI);
  }
};
} // end anonymous namespace

unsigned M88kMCAsmBackend::getNumFixupKinds() const {
  return M88k::NumTargetFixupKinds;
}

Optional<MCFixupKind> M88kMCAsmBackend::getFixupKind(StringRef Name) const {
  unsigned Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(X, Y) .Case(#X, Y)
#include "llvm/BinaryFormat/ELFRelocs/M88k.def"
#undef ELF_RELOC
                      .Default(-1u);
  if (Type != -1u)
    return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  return None;
}

const MCFixupKindInfo &
M88kMCAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  static const MCFixupKindInfo Infos[M88k::NumTargetFixupKinds] = {
      // This table *must* be in same the order of fixup_* kinds in
      // M88kMCFixups.h.
      // name    offset bits flags
      {"FK_88K_NONE", 0, 32, 0},
      {"FK_88K_DISP16", 16, 16, 0},
      {"FK_88K_DISP26", 6, 26, 0},
      {"FK_88K_HI", 16, 16, 0},
      {"FK_88K_LO", 16, 16, 0}};

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}

void M88kMCAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                  const MCValue &Target,
                                  MutableArrayRef<char> Data, uint64_t Value,
                                  bool IsResolved,
                                  const MCSubtargetInfo *STI) const {
  MCFixupKind Kind = Fixup.getKind();
  if (Kind >= FirstLiteralRelocationKind)
    return;
  unsigned Offset = Fixup.getOffset();
  unsigned BitSize = getFixupKindInfo(Kind).TargetSize;
  unsigned Size = (BitSize + 7) / 8;

  assert(Offset + Size <= Data.size() && "Invalid fixup offset!");

  // Big-endian insertion of Size bytes.
  Value = extractBitsForFixup(Kind, Value, Fixup, Asm.getContext());
  if (BitSize < 64)
    Value &= ((uint64_t)1 << BitSize) - 1;
  unsigned ShiftValue = (Size * 8) - 8;
  for (unsigned I = 0; I != Size; ++I) {
    Data[Offset + I] |= uint8_t(Value >> ShiftValue);
    ShiftValue -= 8;
  }
}

bool M88kMCAsmBackend::mayNeedRelaxation(const MCInst &Inst,
                                         const MCSubtargetInfo &STI) const {
  return false;
}

bool M88kMCAsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                            uint64_t Value,
                                            const MCRelaxableFragment *Fragment,
                                            const MCAsmLayout &Layout) const {
  return false;
}

bool M88kMCAsmBackend::shouldForceRelocation(const MCAssembler &Asm,
                                             const MCFixup &Fixup,
                                             const MCValue &Target) {
  unsigned Kind = Fixup.getKind();
  return Kind == M88k::FK_88K_DISP16 || Kind == M88k::FK_88K_DISP26;
}

bool M88kMCAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                    const MCSubtargetInfo *STI) const {
  if ((Count % 4) != 0)
    return false;

  for (uint64_t I = 0; I != Count; ++I)
    OS << "\xf4\x00\x58\x00"; // or %r0,%r0,%r0

  return true;
}

MCAsmBackend *llvm::createM88kMCAsmBackend(const Target &T,
                                           const MCSubtargetInfo &STI,
                                           const MCRegisterInfo &MRI,
                                           const MCTargetOptions &Options) {
  uint8_t OSABI =
      MCELFObjectTargetWriter::getOSABI(STI.getTargetTriple().getOS());
  return new M88kMCAsmBackend(OSABI);
}
