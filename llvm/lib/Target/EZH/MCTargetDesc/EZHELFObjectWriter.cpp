//===-- EZHELFObjectWriter.cpp - EZH ELF Writer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/EZHBaseInfo.h"
#include "MCTargetDesc/EZHFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

class EZHELFObjectWriter : public MCELFObjectTargetWriter {
public:
  explicit EZHELFObjectWriter(uint8_t OSABI);

  ~EZHELFObjectWriter() override = default;

protected:
  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
  bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override;
};

} // end anonymous namespace

EZHELFObjectWriter::EZHELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit_=*/false, OSABI, ELF::EM_EZH,
                              /*HasRelocationAddend_=*/true) {}

unsigned EZHELFObjectWriter::getRelocType(const MCFixup &Fixup, const MCValue &,
                                          bool) const {
  unsigned Type;
  unsigned Kind = static_cast<unsigned>(Fixup.getKind());
  switch (Kind) {
  case EZH::FIXUP_EZH_21:
    Type = ELF::R_EZH_21;
    break;
  case EZH::FIXUP_EZH_21_F:
    Type = ELF::R_EZH_21_F;
    break;
  case EZH::FIXUP_EZH_25:
    Type = ELF::R_EZH_25;
    break;
  case EZH::FIXUP_EZH_32:
  case FK_Data_4:
    Type = ELF::R_EZH_32;
    break;
  case EZH::FIXUP_EZH_HI16:
    Type = ELF::R_EZH_HI16;
    break;
  case EZH::FIXUP_EZH_LO16:
    Type = ELF::R_EZH_LO16;
    break;
  case EZH::FIXUP_EZH_NONE:
    Type = ELF::R_EZH_NONE;
    break;
  case EZH::FIXUP_EZH_8_PCREL:
    Type = ELF::R_EZH_NONE;
    break;

  default:
    llvm_unreachable("Invalid fixup kind!");
  }
  return Type;
}

bool EZHELFObjectWriter::needsRelocateWithSymbol(const MCValue &,
                                                 unsigned Type) const {
  switch (Type) {
  case ELF::R_EZH_21:
  case ELF::R_EZH_21_F:
  case ELF::R_EZH_25:
  case ELF::R_EZH_HI16:
    return true;
  case ELF::R_EZH_32:
    return false;
  default:
    return false;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createEZHELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<EZHELFObjectWriter>(OSABI);
}
