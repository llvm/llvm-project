//===- RISCVWinCOFFObjectWriter.cpp-----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"

using namespace llvm;

namespace {

class RISCVWinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  RISCVWinCOFFObjectWriter(bool Is64Bit);

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsCrossSection,
                        const MCAsmBackend &MAB) const override;
};

} // namespace

RISCVWinCOFFObjectWriter::RISCVWinCOFFObjectWriter(bool Is64Bit)
    : MCWinCOFFObjectTargetWriter(Is64Bit ? COFF::IMAGE_FILE_MACHINE_RISCV64
                                          : COFF::IMAGE_FILE_MACHINE_RISCV32) {}

unsigned RISCVWinCOFFObjectWriter::getRelocType(MCContext &Ctx,
                                                const MCValue &Target,
                                                const MCFixup &Fixup,
                                                bool IsCrossSection,
                                                const MCAsmBackend &MAB) const {
  unsigned FixupKind = Fixup.getKind();

  switch (FixupKind) {
  default:
    Ctx.reportError(Fixup.getLoc(), "unsupported relocation type");
    return COFF::IMAGE_REL_BASED_RISCV_HI20; // FIXME
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createRISCVWinCOFFObjectWriter(bool Is64Bit) {
  return std::make_unique<RISCVWinCOFFObjectWriter>(Is64Bit);
}
