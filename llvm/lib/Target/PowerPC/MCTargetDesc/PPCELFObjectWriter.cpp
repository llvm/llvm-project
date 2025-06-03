//===-- PPCELFObjectWriter.cpp - PPC ELF Writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCFixupKinds.h"
#include "MCTargetDesc/PPCMCExpr.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class PPCELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI);

  protected:
    unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                          bool IsPCRel) const override;

    bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override;
  };
}

PPCELFObjectWriter::PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI)
  : MCELFObjectTargetWriter(Is64Bit, OSABI,
                            Is64Bit ?  ELF::EM_PPC64 : ELF::EM_PPC,
                            /*HasRelocationAddend*/ true) {}

unsigned PPCELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                          const MCValue &Target,
                                          bool IsPCRel) const {
  SMLoc Loc = Fixup.getValue()->getLoc();
  auto Spec = static_cast<PPCMCExpr::Specifier>(Target.getSpecifier());
  switch (Spec) {
  case PPCMCExpr::VK_DTPMOD:
  case PPCMCExpr::VK_DTPREL:
  case PPCMCExpr::VK_DTPREL_HA:
  case PPCMCExpr::VK_DTPREL_HI:
  case PPCMCExpr::VK_DTPREL_HIGH:
  case PPCMCExpr::VK_DTPREL_HIGHA:
  case PPCMCExpr::VK_DTPREL_HIGHER:
  case PPCMCExpr::VK_DTPREL_HIGHERA:
  case PPCMCExpr::VK_DTPREL_HIGHEST:
  case PPCMCExpr::VK_DTPREL_HIGHESTA:
  case PPCMCExpr::VK_DTPREL_LO:
  case PPCMCExpr::VK_GOT_DTPREL:
  case PPCMCExpr::VK_GOT_DTPREL_HA:
  case PPCMCExpr::VK_GOT_DTPREL_HI:
  case PPCMCExpr::VK_GOT_DTPREL_LO:
  case PPCMCExpr::VK_GOT_TLSGD:
  case PPCMCExpr::VK_GOT_TLSGD_HA:
  case PPCMCExpr::VK_GOT_TLSGD_HI:
  case PPCMCExpr::VK_GOT_TLSGD_LO:
  case PPCMCExpr::VK_GOT_TLSGD_PCREL:
  case PPCMCExpr::VK_GOT_TLSLD:
  case PPCMCExpr::VK_GOT_TLSLD_HA:
  case PPCMCExpr::VK_GOT_TLSLD_HI:
  case PPCMCExpr::VK_GOT_TLSLD_LO:
  case PPCMCExpr::VK_GOT_TPREL:
  case PPCMCExpr::VK_GOT_TPREL_HA:
  case PPCMCExpr::VK_GOT_TPREL_HI:
  case PPCMCExpr::VK_GOT_TPREL_LO:
  case PPCMCExpr::VK_GOT_TPREL_PCREL:
  case PPCMCExpr::VK_TLS:
  case PPCMCExpr::VK_TLSGD:
  case PPCMCExpr::VK_TLSLD:
  case PPCMCExpr::VK_TLS_PCREL:
  case PPCMCExpr::VK_TPREL:
  case PPCMCExpr::VK_TPREL_HA:
  case PPCMCExpr::VK_TPREL_HI:
  case PPCMCExpr::VK_TPREL_HIGH:
  case PPCMCExpr::VK_TPREL_HIGHA:
  case PPCMCExpr::VK_TPREL_HIGHER:
  case PPCMCExpr::VK_TPREL_HIGHERA:
  case PPCMCExpr::VK_TPREL_HIGHEST:
  case PPCMCExpr::VK_TPREL_HIGHESTA:
  case PPCMCExpr::VK_TPREL_LO:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  default:
    break;
  }

  // determine the type of the relocation
  unsigned Type = 0;
  if (IsPCRel) {
    switch (Fixup.getTargetKind()) {
    default:
      llvm_unreachable("Unimplemented");
    case PPC::fixup_ppc_br24:
    case PPC::fixup_ppc_br24abs:
    case PPC::fixup_ppc_br24_notoc:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_None:
        Type = ELF::R_PPC_REL24;
        break;
      case PPCMCExpr::VK_PLT:
        Type = ELF::R_PPC_PLTREL24;
        break;
      case PPCMCExpr::VK_LOCAL:
        Type = ELF::R_PPC_LOCAL24PC;
        break;
      case PPCMCExpr::VK_NOTOC:
        Type = ELF::R_PPC64_REL24_NOTOC;
        break;
      }
      break;
    case PPC::fixup_ppc_brcond14:
    case PPC::fixup_ppc_brcond14abs:
      Type = ELF::R_PPC_REL14;
      break;
    case PPC::fixup_ppc_half16:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        return ELF::R_PPC_NONE;
      case PPCMCExpr::VK_None:
        return ELF::R_PPC_REL16;
      case PPCMCExpr::VK_LO:
        return ELF::R_PPC_REL16_LO;
      case PPCMCExpr::VK_HI:
        return ELF::R_PPC_REL16_HI;
      case PPCMCExpr::VK_HA:
        return ELF::R_PPC_REL16_HA;
      }
      break;
    case PPC::fixup_ppc_half16ds:
    case PPC::fixup_ppc_half16dq:
      reportError(Loc, "unsupported relocation type");
      break;
    case PPC::fixup_ppc_pcrel34:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_PCREL:
        Type = ELF::R_PPC64_PCREL34;
        break;
      case PPCMCExpr::VK_GOT_PCREL:
        Type = ELF::R_PPC64_GOT_PCREL34;
        break;
      case PPCMCExpr::VK_GOT_TLSGD_PCREL:
        Type = ELF::R_PPC64_GOT_TLSGD_PCREL34;
        break;
      case PPCMCExpr::VK_GOT_TLSLD_PCREL:
        Type = ELF::R_PPC64_GOT_TLSLD_PCREL34;
        break;
      case PPCMCExpr::VK_GOT_TPREL_PCREL:
        Type = ELF::R_PPC64_GOT_TPREL_PCREL34;
        break;
      }
      break;
    case FK_Data_4:
      Type = ELF::R_PPC_REL32;
      break;
    case FK_Data_8:
      Type = ELF::R_PPC64_REL64;
      break;
    }
  } else {
    switch (Fixup.getTargetKind()) {
      default: llvm_unreachable("invalid fixup kind!");
    case PPC::fixup_ppc_br24abs:
      Type = ELF::R_PPC_ADDR24;
      break;
    case PPC::fixup_ppc_brcond14abs:
      Type = ELF::R_PPC_ADDR14; // XXX: or BRNTAKEN?_
      break;
    case PPC::fixup_ppc_half16:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_LO:
        return ELF::R_PPC_ADDR16_LO;
      case PPCMCExpr::VK_HI:
        return ELF::R_PPC_ADDR16_HI;
      case PPCMCExpr::VK_HA:
        return ELF::R_PPC_ADDR16_HA;
      case PPCMCExpr::VK_HIGH:
        return ELF::R_PPC64_ADDR16_HIGH;
      case PPCMCExpr::VK_HIGHA:
        return ELF::R_PPC64_ADDR16_HIGHA;
      case PPCMCExpr::VK_HIGHER:
        return ELF::R_PPC64_ADDR16_HIGHER;
      case PPCMCExpr::VK_HIGHERA:
        return ELF::R_PPC64_ADDR16_HIGHERA;
      case PPCMCExpr::VK_HIGHEST:
        return ELF::R_PPC64_ADDR16_HIGHEST;
      case PPCMCExpr::VK_HIGHESTA:
        return ELF::R_PPC64_ADDR16_HIGHESTA;

      case PPCMCExpr::VK_None:
        Type = ELF::R_PPC_ADDR16;
        break;
      case PPCMCExpr::VK_GOT:
        Type = ELF::R_PPC_GOT16;
        break;
      case PPCMCExpr::VK_GOT_LO:
        Type = ELF::R_PPC_GOT16_LO;
        break;
      case PPCMCExpr::VK_GOT_HI:
        Type = ELF::R_PPC_GOT16_HI;
        break;
      case PPCMCExpr::VK_GOT_HA:
        Type = ELF::R_PPC_GOT16_HA;
        break;
      case PPCMCExpr::VK_TOC:
        Type = ELF::R_PPC64_TOC16;
        break;
      case PPCMCExpr::VK_TOC_LO:
        Type = ELF::R_PPC64_TOC16_LO;
        break;
      case PPCMCExpr::VK_TOC_HI:
        Type = ELF::R_PPC64_TOC16_HI;
        break;
      case PPCMCExpr::VK_TOC_HA:
        Type = ELF::R_PPC64_TOC16_HA;
        break;
      case PPCMCExpr::VK_TPREL:
        Type = ELF::R_PPC_TPREL16;
        break;
      case PPCMCExpr::VK_TPREL_LO:
        Type = ELF::R_PPC_TPREL16_LO;
        break;
      case PPCMCExpr::VK_TPREL_HI:
        Type = ELF::R_PPC_TPREL16_HI;
        break;
      case PPCMCExpr::VK_TPREL_HA:
        Type = ELF::R_PPC_TPREL16_HA;
        break;
      case PPCMCExpr::VK_TPREL_HIGH:
        Type = ELF::R_PPC64_TPREL16_HIGH;
        break;
      case PPCMCExpr::VK_TPREL_HIGHA:
        Type = ELF::R_PPC64_TPREL16_HIGHA;
        break;
      case PPCMCExpr::VK_TPREL_HIGHER:
        Type = ELF::R_PPC64_TPREL16_HIGHER;
        break;
      case PPCMCExpr::VK_TPREL_HIGHERA:
        Type = ELF::R_PPC64_TPREL16_HIGHERA;
        break;
      case PPCMCExpr::VK_TPREL_HIGHEST:
        Type = ELF::R_PPC64_TPREL16_HIGHEST;
        break;
      case PPCMCExpr::VK_TPREL_HIGHESTA:
        Type = ELF::R_PPC64_TPREL16_HIGHESTA;
        break;
      case PPCMCExpr::VK_DTPREL:
        Type = ELF::R_PPC64_DTPREL16;
        break;
      case PPCMCExpr::VK_DTPREL_LO:
        Type = ELF::R_PPC64_DTPREL16_LO;
        break;
      case PPCMCExpr::VK_DTPREL_HI:
        Type = ELF::R_PPC64_DTPREL16_HI;
        break;
      case PPCMCExpr::VK_DTPREL_HA:
        Type = ELF::R_PPC64_DTPREL16_HA;
        break;
      case PPCMCExpr::VK_DTPREL_HIGH:
        Type = ELF::R_PPC64_DTPREL16_HIGH;
        break;
      case PPCMCExpr::VK_DTPREL_HIGHA:
        Type = ELF::R_PPC64_DTPREL16_HIGHA;
        break;
      case PPCMCExpr::VK_DTPREL_HIGHER:
        Type = ELF::R_PPC64_DTPREL16_HIGHER;
        break;
      case PPCMCExpr::VK_DTPREL_HIGHERA:
        Type = ELF::R_PPC64_DTPREL16_HIGHERA;
        break;
      case PPCMCExpr::VK_DTPREL_HIGHEST:
        Type = ELF::R_PPC64_DTPREL16_HIGHEST;
        break;
      case PPCMCExpr::VK_DTPREL_HIGHESTA:
        Type = ELF::R_PPC64_DTPREL16_HIGHESTA;
        break;
      case PPCMCExpr::VK_GOT_TLSGD:
        if (is64Bit())
          Type = ELF::R_PPC64_GOT_TLSGD16;
        else
          Type = ELF::R_PPC_GOT_TLSGD16;
        break;
      case PPCMCExpr::VK_GOT_TLSGD_LO:
        Type = ELF::R_PPC64_GOT_TLSGD16_LO;
        break;
      case PPCMCExpr::VK_GOT_TLSGD_HI:
        Type = ELF::R_PPC64_GOT_TLSGD16_HI;
        break;
      case PPCMCExpr::VK_GOT_TLSGD_HA:
        Type = ELF::R_PPC64_GOT_TLSGD16_HA;
        break;
      case PPCMCExpr::VK_GOT_TLSLD:
        if (is64Bit())
          Type = ELF::R_PPC64_GOT_TLSLD16;
        else
          Type = ELF::R_PPC_GOT_TLSLD16;
        break;
      case PPCMCExpr::VK_GOT_TLSLD_LO:
        Type = ELF::R_PPC64_GOT_TLSLD16_LO;
        break;
      case PPCMCExpr::VK_GOT_TLSLD_HI:
        Type = ELF::R_PPC64_GOT_TLSLD16_HI;
        break;
      case PPCMCExpr::VK_GOT_TLSLD_HA:
        Type = ELF::R_PPC64_GOT_TLSLD16_HA;
        break;
      case PPCMCExpr::VK_GOT_TPREL:
        /* We don't have R_PPC64_GOT_TPREL16, but since GOT offsets
           are always 4-aligned, we can use R_PPC64_GOT_TPREL16_DS.  */
        Type = ELF::R_PPC64_GOT_TPREL16_DS;
        break;
      case PPCMCExpr::VK_GOT_TPREL_LO:
        /* We don't have R_PPC64_GOT_TPREL16_LO, but since GOT offsets
           are always 4-aligned, we can use R_PPC64_GOT_TPREL16_LO_DS.  */
        Type = ELF::R_PPC64_GOT_TPREL16_LO_DS;
        break;
      case PPCMCExpr::VK_GOT_TPREL_HI:
        Type = ELF::R_PPC64_GOT_TPREL16_HI;
        break;
      case PPCMCExpr::VK_GOT_DTPREL:
        /* We don't have R_PPC64_GOT_DTPREL16, but since GOT offsets
           are always 4-aligned, we can use R_PPC64_GOT_DTPREL16_DS.  */
        Type = ELF::R_PPC64_GOT_DTPREL16_DS;
        break;
      case PPCMCExpr::VK_GOT_DTPREL_LO:
        /* We don't have R_PPC64_GOT_DTPREL16_LO, but since GOT offsets
           are always 4-aligned, we can use R_PPC64_GOT_DTPREL16_LO_DS.  */
        Type = ELF::R_PPC64_GOT_DTPREL16_LO_DS;
        break;
      case PPCMCExpr::VK_GOT_TPREL_HA:
        Type = ELF::R_PPC64_GOT_TPREL16_HA;
        break;
      case PPCMCExpr::VK_GOT_DTPREL_HI:
        Type = ELF::R_PPC64_GOT_DTPREL16_HI;
        break;
      case PPCMCExpr::VK_GOT_DTPREL_HA:
        Type = ELF::R_PPC64_GOT_DTPREL16_HA;
        break;
      }
      break;
    case PPC::fixup_ppc_half16ds:
    case PPC::fixup_ppc_half16dq:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_LO:
        return ELF::R_PPC64_ADDR16_LO_DS;
      case PPCMCExpr::VK_None:
        Type = ELF::R_PPC64_ADDR16_DS;
        break;
      case PPCMCExpr::VK_GOT:
        Type = ELF::R_PPC64_GOT16_DS;
        break;
      case PPCMCExpr::VK_GOT_LO:
        Type = ELF::R_PPC64_GOT16_LO_DS;
        break;
      case PPCMCExpr::VK_TOC:
        Type = ELF::R_PPC64_TOC16_DS;
        break;
      case PPCMCExpr::VK_TOC_LO:
        Type = ELF::R_PPC64_TOC16_LO_DS;
        break;
      case PPCMCExpr::VK_TPREL:
        Type = ELF::R_PPC64_TPREL16_DS;
        break;
      case PPCMCExpr::VK_TPREL_LO:
        Type = ELF::R_PPC64_TPREL16_LO_DS;
        break;
      case PPCMCExpr::VK_DTPREL:
        Type = ELF::R_PPC64_DTPREL16_DS;
        break;
      case PPCMCExpr::VK_DTPREL_LO:
        Type = ELF::R_PPC64_DTPREL16_LO_DS;
        break;
      case PPCMCExpr::VK_GOT_TPREL:
        Type = ELF::R_PPC64_GOT_TPREL16_DS;
        break;
      case PPCMCExpr::VK_GOT_TPREL_LO:
        Type = ELF::R_PPC64_GOT_TPREL16_LO_DS;
        break;
      case PPCMCExpr::VK_GOT_DTPREL:
        Type = ELF::R_PPC64_GOT_DTPREL16_DS;
        break;
      case PPCMCExpr::VK_GOT_DTPREL_LO:
        Type = ELF::R_PPC64_GOT_DTPREL16_LO_DS;
        break;
      }
      break;
    case PPC::fixup_ppc_nofixup:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_TLSGD:
        if (is64Bit())
          Type = ELF::R_PPC64_TLSGD;
        else
          Type = ELF::R_PPC_TLSGD;
        break;
      case PPCMCExpr::VK_TLSLD:
        if (is64Bit())
          Type = ELF::R_PPC64_TLSLD;
        else
          Type = ELF::R_PPC_TLSLD;
        break;
      case PPCMCExpr::VK_TLS:
        if (is64Bit())
          Type = ELF::R_PPC64_TLS;
        else
          Type = ELF::R_PPC_TLS;
        break;
      case PPCMCExpr::VK_TLS_PCREL:
        Type = ELF::R_PPC64_TLS;
        break;
      }
      break;
    case PPC::fixup_ppc_imm34:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_DTPREL:
        Type = ELF::R_PPC64_DTPREL34;
        break;
      case PPCMCExpr::VK_TPREL:
        Type = ELF::R_PPC64_TPREL34;
        break;
      }
      break;
    case FK_Data_8:
      switch (Spec) {
      default:
        reportError(Loc, "unsupported relocation type");
        break;
      case PPCMCExpr::VK_TOCBASE:
        Type = ELF::R_PPC64_TOC;
        break;
      case PPCMCExpr::VK_None:
        Type = ELF::R_PPC64_ADDR64;
        break;
      case PPCMCExpr::VK_DTPMOD:
        Type = ELF::R_PPC64_DTPMOD64;
        break;
      case PPCMCExpr::VK_TPREL:
        Type = ELF::R_PPC64_TPREL64;
        break;
      case PPCMCExpr::VK_DTPREL:
        Type = ELF::R_PPC64_DTPREL64;
        break;
      }
      break;
    case FK_Data_4:
      switch (Spec) {
      case PPCMCExpr::VK_DTPREL:
        Type = ELF::R_PPC_DTPREL32;
        break;
      default:
        Type = ELF::R_PPC_ADDR32;
      }
      break;
    case FK_Data_2:
      Type = ELF::R_PPC_ADDR16;
      break;
    }
  }
  return Type;
}

bool PPCELFObjectWriter::needsRelocateWithSymbol(const MCValue &V,
                                                 unsigned Type) const {
  switch (Type) {
    default:
      return false;

    case ELF::R_PPC_REL24:
    case ELF::R_PPC64_REL24_NOTOC: {
      // If the target symbol has a local entry point, we must keep the
      // target symbol to preserve that information for the linker.
      // The "other" values are stored in the last 6 bits of the second byte.
      // The traditional defines for STO values assume the full byte and thus
      // the shift to pack it.
      unsigned Other = cast<MCSymbolELF>(V.getAddSym())->getOther() << 2;
      return (Other & ELF::STO_PPC64_LOCAL_MASK) != 0;
    }

    case ELF::R_PPC64_GOT16:
    case ELF::R_PPC64_GOT16_DS:
    case ELF::R_PPC64_GOT16_LO:
    case ELF::R_PPC64_GOT16_LO_DS:
    case ELF::R_PPC64_GOT16_HI:
    case ELF::R_PPC64_GOT16_HA:
      return true;
    }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createPPCELFObjectWriter(bool Is64Bit, uint8_t OSABI) {
  return std::make_unique<PPCELFObjectWriter>(Is64Bit, OSABI);
}
