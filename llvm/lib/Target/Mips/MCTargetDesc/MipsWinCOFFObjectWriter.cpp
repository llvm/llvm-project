//===- MipsWinCOFFObjectWriter.cpp------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"

using namespace llvm;

namespace {

class MipsWinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  MipsWinCOFFObjectWriter();

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsCrossSection,
                        const MCAsmBackend &MAB) const override;
};

} // end anonymous namespace

MipsWinCOFFObjectWriter::MipsWinCOFFObjectWriter()
    : MCWinCOFFObjectTargetWriter(COFF::IMAGE_FILE_MACHINE_R4000) {}

unsigned MipsWinCOFFObjectWriter::getRelocType(MCContext &Ctx,
                                               const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsCrossSection,
                                               const MCAsmBackend &MAB) const {
  unsigned FixupKind = Fixup.getKind();

  switch (FixupKind) {
  case FK_Data_4:
    return COFF::IMAGE_REL_MIPS_REFWORD;
  case FK_SecRel_2:
    return COFF::IMAGE_REL_MIPS_SECTION;
  case FK_SecRel_4:
    return COFF::IMAGE_REL_MIPS_SECREL;
  case Mips::fixup_Mips_26:
    return COFF::IMAGE_REL_MIPS_JMPADDR;
  case Mips::fixup_Mips_HI16:
    return COFF::IMAGE_REL_MIPS_REFHI;
  case Mips::fixup_Mips_LO16:
    return COFF::IMAGE_REL_MIPS_REFLO;
  default:
    Ctx.reportError(Fixup.getLoc(), "unsupported relocation type");
    return COFF::IMAGE_REL_MIPS_REFWORD;
  }
}

std::unique_ptr<MCObjectTargetWriter> llvm::createMipsWinCOFFObjectWriter() {
  return std::make_unique<MipsWinCOFFObjectWriter>();
}
