//===-- M88kMCObjectWriter.cpp - M88k ELF writer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/M88kMCFixups.h"
#include "MCTargetDesc/M88kMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

namespace {

class M88kObjectWriter : public MCELFObjectTargetWriter {
public:
  M88kObjectWriter(uint8_t OSABI);
  ~M88kObjectWriter() override = default;

protected:
  // Override MCELFObjectTargetWriter.
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};

} // end anonymous namespace

M88kObjectWriter::M88kObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit_=*/false, OSABI, ELF::EM_88K,
                              /*HasRelocationAddend_=*/true) {}

unsigned M88kObjectWriter::getRelocType(MCContext &Ctx, const MCValue &Target,
                                        const MCFixup &Fixup,
                                        bool IsPCRel) const {
  unsigned Type;
  unsigned Kind = static_cast<unsigned>(Fixup.getKind());
  switch (Kind) {
  case M88k::FK_88K_HI:
    Type = ELF::R_88K_16H;
    break;
  case M88k::FK_88K_LO:
    Type = ELF::R_88K_16L;
    break;
  case M88k::FK_88K_DISP16:
    Type = ELF::R_88K_DISP16;
    break;
  case M88k::FK_88K_DISP26:
    Type = ELF::R_88K_DISP26;
    break;
  case M88k::FK_88K_NONE:
    Type = ELF::R_88K_NONE;
    break;

  case FK_Data_4:
    Type = ELF::R_88K_32;
    break;

  default:
    llvm_unreachable("Invalid fixup kind!");
  }
  return Type;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createM88kObjectWriter(uint8_t OSABI) {
  return std::make_unique<M88kObjectWriter>(OSABI);
}
