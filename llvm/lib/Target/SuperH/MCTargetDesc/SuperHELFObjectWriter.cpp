//===-- SuperHELFObjectWriter.cpp - SuperH ELF Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class SuperHELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    SuperHELFObjectWriter(uint8_t OSABI)
        : MCELFObjectTargetWriter(
              false, OSABI,
              ELF::EM_SH,
              /*HasRelocationAddend*/ true) {}

    ~SuperHELFObjectWriter() override = default;

  protected:
    unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                          bool IsPCRel) const override;

    bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override;
  };
}

unsigned SuperHELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  return ELF::R_SH_NONE;
}

bool SuperHELFObjectWriter::needsRelocateWithSymbol(const MCValue &,
                                                   unsigned Type) const {
  
  return false;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createSuperHELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<SuperHELFObjectWriter>(OSABI);
}
