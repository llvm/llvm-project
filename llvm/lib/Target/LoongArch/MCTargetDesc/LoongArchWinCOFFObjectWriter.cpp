//===- LoongArchWinCOFFObjectWriter.cpp -----------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/LoongArchFixupKinds.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"

using namespace llvm;

namespace {

class LoongArchWinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  LoongArchWinCOFFObjectWriter(bool Is64Bit);

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsCrossSection,
                        const MCAsmBackend &MAB) const override;
};

} // end anonymous namespace

LoongArchWinCOFFObjectWriter::LoongArchWinCOFFObjectWriter(bool Is64Bit)
    : MCWinCOFFObjectTargetWriter(COFF::IMAGE_FILE_MACHINE_LOONGARCH64) {}

unsigned LoongArchWinCOFFObjectWriter::getRelocType(
    MCContext &Ctx, const MCValue &Target, const MCFixup &Fixup,
    bool IsCrossSection, const MCAsmBackend &MAB) const {
  // UEFI TODO: convert fixup to coff relocation
  return Fixup.getKind();
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createLoongArchWinCOFFObjectWriter(bool Is64Bit) {
  return std::make_unique<LoongArchWinCOFFObjectWriter>(Is64Bit);
}
