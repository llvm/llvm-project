//===-- Nios2ELFObjectWriter.cpp - Nios2 ELF Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Nios2FixupKinds.h"
#include "MCTargetDesc/Nios2MCExpr.h"
#include "MCTargetDesc/Nios2MCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"

using namespace llvm;

namespace {
class Nios2ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  Nios2ELFObjectWriter(uint8_t OSABI)
      : MCELFObjectTargetWriter(false, OSABI, ELF::EM_ALTERA_NIOS2, false) {}

  ~Nios2ELFObjectWriter() override;

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
} // namespace

Nios2ELFObjectWriter::~Nios2ELFObjectWriter() {}

unsigned Nios2ELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
  return 0;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createNios2ELFObjectWriter(uint8_t OSABI) {
  return llvm::make_unique<Nios2ELFObjectWriter>(OSABI);
}
