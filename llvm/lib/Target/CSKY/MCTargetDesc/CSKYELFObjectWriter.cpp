//===-- CSKYELFObjectWriter.cpp - CSKY ELF Writer -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSKYFixupKinds.h"
#include "CSKYMCAsmInfo.h"
#include "CSKYMCTargetDesc.h"
#include "MCTargetDesc/CSKYMCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbolELF.h"

#define DEBUG_TYPE "csky-elf-object-writer"

using namespace llvm;

namespace {

class CSKYELFObjectWriter : public MCELFObjectTargetWriter {
public:
  CSKYELFObjectWriter(uint8_t OSABI = 0)
      : MCELFObjectTargetWriter(false, OSABI, ELF::EM_CSKY, true){};
  ~CSKYELFObjectWriter() {}

  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
  bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override;
};

} // namespace

unsigned CSKYELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                           const MCValue &Target,
                                           bool IsPCRel) const {
  const MCExpr *Expr = Fixup.getValue();
  // Determine the type of the relocation
  auto Kind = Fixup.getKind();
  uint8_t Modifier = Target.getSpecifier();

  switch (Target.getSpecifier()) {
  case CSKY::S_TLSIE:
  case CSKY::S_TLSLE:
  case CSKY::S_TLSGD:
  case CSKY::S_TLSLDM:
  case CSKY::S_TLSLDO:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  default:
    break;
  }

  if (IsPCRel) {
    switch (Kind) {
    default:
      LLVM_DEBUG(dbgs() << "Unknown Kind1  = " << Kind);
      reportError(Fixup.getLoc(), "Unsupported relocation type");
      return ELF::R_CKCORE_NONE;
    case FK_Data_4:
      return ELF::R_CKCORE_PCREL32;
    case CSKY::fixup_csky_pcrel_uimm16_scale4:
      return ELF::R_CKCORE_PCREL_IMM16_4;
    case CSKY::fixup_csky_pcrel_uimm8_scale4:
      return ELF::R_CKCORE_PCREL_IMM8_4;
    case CSKY::fixup_csky_pcrel_imm26_scale2:
      return ELF::R_CKCORE_PCREL_IMM26_2;
    case CSKY::fixup_csky_pcrel_imm18_scale2:
      return ELF::R_CKCORE_PCREL_IMM18_2;
    case CSKY::fixup_csky_pcrel_imm16_scale2:
      return ELF::R_CKCORE_PCREL_IMM16_2;
    case CSKY::fixup_csky_pcrel_imm10_scale2:
      return ELF::R_CKCORE_PCREL_IMM10_2;
    case CSKY::fixup_csky_pcrel_uimm7_scale4:
      return ELF::R_CKCORE_PCREL_IMM7_4;
    }
  }

  switch (Kind) {
  default:
    LLVM_DEBUG(dbgs() << "Unknown Kind2  = " << Kind);
    reportError(Fixup.getLoc(), "Unsupported relocation type");
    return ELF::R_CKCORE_NONE;
  case FK_Data_1:
    reportError(Fixup.getLoc(), "1-byte data relocations not supported");
    return ELF::R_CKCORE_NONE;
  case FK_Data_2:
    reportError(Fixup.getLoc(), "2-byte data relocations not supported");
    return ELF::R_CKCORE_NONE;
  case FK_Data_4:
    if (Expr->getKind() == MCExpr::Specifier) {
      auto TK = cast<MCSpecifierExpr>(Expr)->getSpecifier();
      if (TK == CSKY::S_ADDR)
        return ELF::R_CKCORE_ADDR32;
      if (TK == CSKY::S_GOT)
        return ELF::R_CKCORE_GOT32;
      if (TK == CSKY::S_GOTOFF)
        return ELF::R_CKCORE_GOTOFF;
      if (TK == CSKY::S_PLT)
        return ELF::R_CKCORE_PLT32;
      if (TK == CSKY::S_TLSIE)
        return ELF::R_CKCORE_TLS_IE32;
      if (TK == CSKY::S_TLSLE)
        return ELF::R_CKCORE_TLS_LE32;
      if (TK == CSKY::S_TLSGD)
        return ELF::R_CKCORE_TLS_GD32;
      if (TK == CSKY::S_TLSLDM)
        return ELF::R_CKCORE_TLS_LDM32;
      if (TK == CSKY::S_TLSLDO)
        return ELF::R_CKCORE_TLS_LDO32;
      if (TK == CSKY::S_GOTPC)
        return ELF::R_CKCORE_GOTPC;
      if (TK == CSKY::S_None)
        return ELF::R_CKCORE_ADDR32;

      LLVM_DEBUG(dbgs() << "Unknown FK_Data_4 TK  = " << TK);
      reportError(Fixup.getLoc(), "unknown target FK_Data_4");
    } else {
      switch (Modifier) {
      default:
        reportError(Fixup.getLoc(), "invalid fixup for 4-byte data relocation");
        return ELF::R_CKCORE_NONE;
      case CSKY::S_GOT:
        return ELF::R_CKCORE_GOT32;
      case CSKY::S_GOTOFF:
        return ELF::R_CKCORE_GOTOFF;
      case CSKY::S_PLT:
        return ELF::R_CKCORE_PLT32;
      case CSKY::S_TLSGD:
        return ELF::R_CKCORE_TLS_GD32;
      case CSKY::S_TLSLDM:
        return ELF::R_CKCORE_TLS_LDM32;
      case CSKY::S_TPOFF:
        return ELF::R_CKCORE_TLS_LE32;
      case CSKY::S_None:
        return ELF::R_CKCORE_ADDR32;
      }
    }
    return ELF::R_CKCORE_NONE;
  case FK_Data_8:
    reportError(Fixup.getLoc(), "8-byte data relocations not supported");
    return ELF::R_CKCORE_NONE;
  case CSKY::fixup_csky_addr32:
    return ELF::R_CKCORE_ADDR32;
  case CSKY::fixup_csky_addr_hi16:
    return ELF::R_CKCORE_ADDR_HI16;
  case CSKY::fixup_csky_addr_lo16:
    return ELF::R_CKCORE_ADDR_LO16;
  case CSKY::fixup_csky_doffset_imm18:
    return ELF::R_CKCORE_DOFFSET_IMM18;
  case CSKY::fixup_csky_doffset_imm18_scale2:
    return ELF::R_CKCORE_DOFFSET_IMM18_2;
  case CSKY::fixup_csky_doffset_imm18_scale4:
    return ELF::R_CKCORE_DOFFSET_IMM18_4;
  case CSKY::fixup_csky_got_imm18_scale4:
    return ELF::R_CKCORE_GOT_IMM18_4;
  case CSKY::fixup_csky_plt_imm18_scale4:
    return ELF::R_CKCORE_PLT_IMM18_4;
  }
}

bool CSKYELFObjectWriter::needsRelocateWithSymbol(const MCValue &V,
                                                  unsigned Type) const {
  switch (V.getSpecifier()) {
  case CSKY::S_PLT:
  case CSKY::S_GOT:
    return true;
  default:
    return false;
  }
}

std::unique_ptr<MCObjectTargetWriter> llvm::createCSKYELFObjectWriter() {
  return std::make_unique<CSKYELFObjectWriter>();
}
