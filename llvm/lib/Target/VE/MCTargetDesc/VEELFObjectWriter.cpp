//===-- VEELFObjectWriter.cpp - VE ELF Writer -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VEFixupKinds.h"
#include "VEMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class VEELFObjectWriter : public MCELFObjectTargetWriter {
public:
  VEELFObjectWriter(uint8_t OSABI)
      : MCELFObjectTargetWriter(/* Is64Bit */ true, OSABI, ELF::EM_VE,
                                /* HasRelocationAddend */ true) {}

  ~VEELFObjectWriter() override {}

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;

  bool needsRelocateWithSymbol(const MCSymbol &Sym,
                               unsigned Type) const override;
};
} // namespace

unsigned VEELFObjectWriter::getRelocType(MCContext &Ctx, const MCValue &Target,
                                         const MCFixup &Fixup,
                                         bool IsPCRel) const {
  // FIXME: implements.
  return ELF::R_VE_NONE;
}

bool VEELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                unsigned Type) const {
  // FIXME: implements.
  return false;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createVEELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<VEELFObjectWriter>(OSABI);
}
