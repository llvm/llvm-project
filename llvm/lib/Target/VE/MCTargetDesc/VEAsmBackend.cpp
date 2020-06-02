//===-- VEAsmBackend.cpp - VE Assembler Backend ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/VEFixupKinds.h"
#include "MCTargetDesc/VEMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

static uint64_t adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  }
}

namespace {
class VEAsmBackend : public MCAsmBackend {
protected:
  const Target &TheTarget;

public:
  VEAsmBackend(const Target &T) : MCAsmBackend(support::little), TheTarget(T) {}

  unsigned getNumFixupKinds() const override { return VE::NumTargetFixupKinds; }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override {
    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    // FIXME.
    llvm_unreachable("getFixupKindInfo() unimplemented");
  }

  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target) override {
    // FIXME.
    return false;
  }

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override {
    // FIXME.
    return false;
  }

  /// fixupNeedsRelaxation - Target specific predicate for whether a given
  /// fixup requires the associated instruction to be relaxed.
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    // FIXME.
    llvm_unreachable("fixupNeedsRelaxation() unimplemented");
    return false;
  }
  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override {
    // FIXME.
    llvm_unreachable("relaxInstruction() unimplemented");
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count) const override {
    if ((Count % 8) != 0)
      return false;

    for (uint64_t i = 0; i < Count; i += 8)
      support::endian::write<uint64_t>(OS, 0x7900000000000000ULL,
                                       support::little);

    return true;
  }
};

class ELFVEAsmBackend : public VEAsmBackend {
  Triple::OSType OSType;

public:
  ELFVEAsmBackend(const Target &T, Triple::OSType OSType)
      : VEAsmBackend(T), OSType(OSType) {}

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override {
    Value = adjustFixupValue(Fixup.getKind(), Value);
    if (!Value)
      return; // Doesn't change encoding.

    // FIXME.
    llvm_unreachable("applyFixup() unimplemented");
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(OSType);
    return createVEELFObjectWriter(OSABI);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createVEAsmBackend(const Target &T,
                                       const MCSubtargetInfo &STI,
                                       const MCRegisterInfo &MRI,
                                       const MCTargetOptions &Options) {
  return new ELFVEAsmBackend(T, STI.getTargetTriple().getOS());
}
