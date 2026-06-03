//===-- SparcAsmBackend.cpp - Sparc Assembler Backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

namespace {
class SuperHAsmBackend : public MCAsmBackend {
public:
  SuperHAsmBackend(const MCSubtargetInfo &STI)
      : MCAsmBackend(STI.getTargetTriple().isLittleEndian()
                         ? llvm::endianness::little
                         : llvm::endianness::big) {}

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;
  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;
  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  uint8_t *Data, uint64_t Value, bool IsResolved) override;

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override {

    // If the count is not 4-byte aligned, we must be writing data into the
    // text section (otherwise we have unaligned instructions, and thus have
    // far bigger problems), so just write zeros instead.
    OS.write_zeros(Count % 2);
    return true;
  }
};

class ELFSuperHAsmBackend : public SuperHAsmBackend {
  Triple::OSType OSType;

public:
  ELFSuperHAsmBackend(const MCSubtargetInfo &STI, Triple::OSType OSType)
      : SuperHAsmBackend(STI), OSType(OSType) {}

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(OSType);
    return createSuperHELFObjectWriter(OSABI);
  }
};
} // end anonymous namespace

std::optional<MCFixupKind> SuperHAsmBackend::getFixupKind(StringRef Name) const {
  return std::nullopt;
}

MCFixupKindInfo SuperHAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  return {"", 0, 2, 0};
}

void SuperHAsmBackend::applyFixup(const MCFragment &F, const MCFixup &Fixup,
                                 const MCValue &Target, uint8_t *Data,
                                 uint64_t Value, bool IsResolved) {

}

MCAsmBackend *llvm::createSuperHAsmBackend(const Target &T,
                                          const MCSubtargetInfo &STI,
                                          const MCRegisterInfo &MRI,
                                          const MCTargetOptions &Options) {
  return new ELFSuperHAsmBackend(STI, STI.getTargetTriple().getOS());
}
