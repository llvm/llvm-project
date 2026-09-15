//===-- SuperHELFObjectWriter.cpp - SuperH ELF Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "MCTargetDesc/SuperHBaseInfo.h"
#include "MCTargetDesc/SuperHFixupKinds.h"
#include "SuperHMCAsmInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "sh-elf-objwriter"

namespace llvm {
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

    bool needsRelocateWithSymbol(const MCValue &Val, unsigned Type) const override;
  };
}

unsigned SuperHELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  auto Spec = Target.getSpecifier();
  switch ((unsigned)Fixup.getKind()) { 
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_8:
    return ELF::R_SH_NONE;

  case FK_Data_4: {
    switch (Spec) {
    case SH::S_None:
      return ELF::R_SH_DIR32;
    case SH::S_PCREL:
      return ELF::R_SH_REL32;
    default:
      return ELF::R_SH_NONE;
    }
  }
  
  case SH::fixup_pcrel4_by2:
  case SH::fixup_pcrel4_by4:
  case SH::fixup_pcrel8_by4:
  case SH::fixup_pcrel8_4by2:
  case SH::fixup_pcrel12_4by2: {
    switch (Spec) {
    case SH::S_None:
      return ELF::R_SH_DIR32;
    case SH::S_PCREL:
      return ELF::R_SH_REL32;
    default:
      return ELF::R_SH_NONE;
    }
  }

  case SH::fixup_32:
    return ELF::R_SH_DIR32;

  default:
    llvm_unreachable("invalid fixup kind!");
  }
}

bool SuperHELFObjectWriter::needsRelocateWithSymbol(const MCValue &Val,
                                                   unsigned Type) const {

  return false;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createSuperHELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<SuperHELFObjectWriter>(OSABI);
}
