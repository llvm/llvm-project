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
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

namespace {
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

#if LLVM_VERSION_MAJOR > 13
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;
#else
  bool writeNopData(raw_ostream &OS, uint64_t Count) const override;
#endif

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
  /*
  Value = adjustFixupValue(static_cast<unsigned>(Kind), Value);

  if (!Value)
    return; // This value doesn't change the encoding
  */

  // Where in the object and where the number of bytes that need fixing up.
  unsigned Offset = Fixup.getOffset();
  unsigned NumBytes = (getFixupKindInfo(Kind).TargetSize + 7) / 8;
  unsigned FullSize = 4;

  // Grab current value, if any, from bits.
  uint64_t CurVal = 0;

  // Load instruction and apply value
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned Idx = (FullSize - 1 - i);
    CurVal |= static_cast<uint64_t>(static_cast<uint8_t>(Data[Offset + Idx]))
              << (i * 8);
  }

  uint64_t Mask =
      (static_cast<uint64_t>(-1) >> (64 - getFixupKindInfo(Kind).TargetSize));
  CurVal |= Value & Mask;

  // Write out the fixed up bytes back to the code/data bits.
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned Idx = (FullSize - 1 - i);
    Data[Offset + Idx] = static_cast<uint8_t>((CurVal >> (i * 8)) & 0xff);
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

#if LLVM_VERSION_MAJOR > 13
bool M88kMCAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                    const MCSubtargetInfo *STI) const {
#else
bool M88kMCAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count) const {
#endif
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
