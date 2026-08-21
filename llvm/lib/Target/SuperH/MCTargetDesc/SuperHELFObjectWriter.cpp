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
  case FK_Data_4:
  case FK_Data_8:
    return ELF::R_SH_NONE;

  case SH::fixup_got32:
    return ELF::R_SH_GOT32;

  case SH::fixup_got_low16:
    return ELF::R_SH_GOT_LOW16;

  case SH::fixup_got_medlow16:
    return ELF::R_SH_GOT_MEDLOW16;

  case SH::fixup_got_medhi16:
    return ELF::R_SH_GOT_MEDHI16;;

  case SH::fixup_got_hi16:
    return ELF::R_SH_GOT_HI16;

  case SH::fixup_plt32:
    return ELF::R_SH_PLT32;

  case SH::fixup_plt_low16:
    return ELF::R_SH_PLT_LOW16;

  case SH::fixup_plt_medlow16:
    return ELF::R_SH_PLT_MEDLOW16;

  case SH::fixup_plt_medhi16:
    return ELF::R_SH_PLT_MEDHI16;;

  case SH::fixup_plt_hi16:
    return ELF::R_SH_PLT_HI16;

  case SH::fixup_gotplt32:
    return ELF::R_SH_GOTPLT32;

  case SH::fixup_gotplt_low16:
    return ELF::R_SH_GOTPLT_LOW16;

  case SH::fixup_gotplt_medlow16:
    return ELF::R_SH_GOTPLT_MEDLOW16;

  case SH::fixup_gotplt_medhi16:
    return ELF::R_SH_GOTPLT_MEDHI16;;

  case SH::fixup_gotplt_hi16:
    return ELF::R_SH_GOTPLT_HI16;

  case SH::fixup_gotoff:
    return ELF::R_SH_GOTOFF;

  case SH::fixup_gotoff_low16:
    return ELF::R_SH_GOTOFF_LOW16;

  case SH::fixup_gotoff_medlow16:
    return ELF::R_SH_GOTOFF_MEDLOW16;

  case SH::fixup_gotoff_medhi16:
    return ELF::R_SH_GOTOFF_MEDHI16;;

  case SH::fixup_gotoff_hi16:
    return ELF::R_SH_GOTOFF_HI16;

  case SH::fixup_gotpc:
    return ELF::R_SH_GOTPC;

  case SH::fixup_gotpc_low16:
    return ELF::R_SH_GOTPC_LOW16;

  case SH::fixup_gotpc_medlow16:
    return ELF::R_SH_GOTPC_MEDLOW16;

  case SH::fixup_gotpc_medhi16:
    return ELF::R_SH_GOTPC_MEDHI16;;

  case SH::fixup_gotpc_hi16:
    return ELF::R_SH_GOTPC_HI16;

  case SH::fixup_copy:
    return ELF::R_SH_COPY;

  case SH::fixup_copy64:
    return ELF::R_SH_COPY64;

  case SH::fixup_glob_dat:
    return ELF::R_SH_GLOB_DAT;

  case SH::fixup_glob_dat64:
    return ELF::R_SH_GLOB_DAT64;

  case SH::fixup_jump_slot:
    return ELF::R_SH_JMP_SLOT;

  case SH::fixup_jump_slot64:
    return ELF::R_SH_JMP_SLOT64;

  case SH::fixup_relative:
    return ELF::R_SH_RELATIVE;

  case SH::fixup_relative64:
    return ELF::R_SH_RELATIVE64;

  case SH::fixup_dir32:
    return ELF::R_SH_DIR32;

  case SH::fixup_rel32:
    return ELF::R_SH_REL32;

  case SH::fixup_64:
    return ELF::R_SH_64;

  case SH::fixup_64_pcrel:
    return ELF::R_SH_64_PCREL;

  default:
    llvm_unreachable("invalid fixup kind!");
  }
}

bool SuperHELFObjectWriter::needsRelocateWithSymbol(const MCValue &Val,
                                                   unsigned Type) const {
  return true;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createSuperHELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<SuperHELFObjectWriter>(OSABI);
}
