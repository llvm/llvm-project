//===-- AArch64ELFObjectWriter.cpp - AArch64 ELF Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file handles ELF-specific object emission, converting LLVM's internal
// fixups into the appropriate relocations.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64FixupKinds.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class AArch64ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  AArch64ELFObjectWriter(uint8_t OSABI, bool IsLittleEndian);

  ~AArch64ELFObjectWriter() override;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;

private:
};
}

AArch64ELFObjectWriter::AArch64ELFObjectWriter(uint8_t OSABI,
                                               bool IsLittleEndian)
    : MCELFObjectTargetWriter(/*Is64Bit*/ true, OSABI, ELF::EM_AARCH64,
                              /*HasRelocationAddend*/ true) {}

AArch64ELFObjectWriter::~AArch64ELFObjectWriter() {}

unsigned AArch64ELFObjectWriter::getRelocType(MCContext &Ctx,
                                              const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsPCRel) const {
  AArch64MCExpr::VariantKind RefKind =
      static_cast<AArch64MCExpr::VariantKind>(Target.getRefKind());
  AArch64MCExpr::VariantKind SymLoc = AArch64MCExpr::getSymbolLoc(RefKind);
  bool IsNC = AArch64MCExpr::isNotChecked(RefKind);

  assert((!Target.getSymA() ||
          Target.getSymA()->getKind() == MCSymbolRefExpr::VK_None) &&
         "Should only be expression-level modifiers here");

  assert((!Target.getSymB() ||
          Target.getSymB()->getKind() == MCSymbolRefExpr::VK_None) &&
         "Should only be expression-level modifiers here");

  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    case FK_Data_1:
      Ctx.reportError(Fixup.getLoc(), "1-byte data relocations not supported");
      return ELF::R_AARCH64_NONE;
    case FK_Data_2:
      return ELF::R_AARCH64_PREL16;
    case FK_Data_4:
      return ELF::R_AARCH64_PREL32;
    case FK_Data_8:
      return ELF::R_AARCH64_PREL64;
    case AArch64::fixup_aarch64_pcrel_adr_imm21:
      assert(SymLoc == AArch64MCExpr::VK_NONE && "unexpected ADR relocation");
      return ELF::R_AARCH64_ADR_PREL_LO21;
    case AArch64::fixup_aarch64_pcrel_adrp_imm21:
      if (SymLoc == AArch64MCExpr::VK_ABS && !IsNC)
        return ELF::R_AARCH64_ADR_PREL_PG_HI21;
      if (SymLoc == AArch64MCExpr::VK_GOT && !IsNC)
        return ELF::R_AARCH64_ADR_GOT_PAGE;
      if (SymLoc == AArch64MCExpr::VK_GOTTPREL && !IsNC)
        return ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21;
      if (SymLoc == AArch64MCExpr::VK_TLSDESC && !IsNC)
        return ELF::R_AARCH64_TLSDESC_ADR_PAGE21;
      Ctx.reportError(Fixup.getLoc(),
                      "invalid symbol kind for ADRP relocation");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_pcrel_branch26:
      return ELF::R_AARCH64_JUMP26;
    case AArch64::fixup_aarch64_pcrel_call26:
      return ELF::R_AARCH64_CALL26;
    case AArch64::fixup_aarch64_ldr_pcrel_imm19:
      if (SymLoc == AArch64MCExpr::VK_GOTTPREL)
        return ELF::R_AARCH64_TLSIE_LD_GOTTPREL_PREL19;
      return ELF::R_AARCH64_LD_PREL_LO19;
    case AArch64::fixup_aarch64_pcrel_branch14:
      return ELF::R_AARCH64_TSTBR14;
    case AArch64::fixup_aarch64_pcrel_branch19:
      return ELF::R_AARCH64_CONDBR19;
    default:
      Ctx.reportError(Fixup.getLoc(), "Unsupported pc-relative fixup kind");
      return ELF::R_AARCH64_NONE;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    case FK_Data_1:
      Ctx.reportError(Fixup.getLoc(), "1-byte data relocations not supported");
      return ELF::R_AARCH64_NONE;
    case FK_Data_2:
      return ELF::R_AARCH64_ABS16;
    case FK_Data_4:
      return ELF::R_AARCH64_ABS32;
    case FK_Data_8:
      return ELF::R_AARCH64_ABS64;
    case AArch64::fixup_aarch64_add_imm12:
      if (RefKind == AArch64MCExpr::VK_DTPREL_HI12)
        return ELF::R_AARCH64_TLSLD_ADD_DTPREL_HI12;
      if (RefKind == AArch64MCExpr::VK_TPREL_HI12)
        return ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12;
      if (RefKind == AArch64MCExpr::VK_DTPREL_LO12_NC)
        return ELF::R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC;
      if (RefKind == AArch64MCExpr::VK_DTPREL_LO12)
        return ELF::R_AARCH64_TLSLD_ADD_DTPREL_LO12;
      if (RefKind == AArch64MCExpr::VK_TPREL_LO12_NC)
        return ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC;
      if (RefKind == AArch64MCExpr::VK_TPREL_LO12)
        return ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12;
      if (RefKind == AArch64MCExpr::VK_TLSDESC_LO12)
        return ELF::R_AARCH64_TLSDESC_ADD_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_ADD_ABS_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for add (uimm12) instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_ldst_imm12_scale1:
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_LDST8_ABS_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && !IsNC)
        return ELF::R_AARCH64_TLSLD_LDST8_DTPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && IsNC)
        return ELF::R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_TPREL && !IsNC)
        return ELF::R_AARCH64_TLSLE_LDST8_TPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_TPREL && IsNC)
        return ELF::R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for 8-bit load/store instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_ldst_imm12_scale2:
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_LDST16_ABS_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && !IsNC)
        return ELF::R_AARCH64_TLSLD_LDST16_DTPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && IsNC)
        return ELF::R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_TPREL && !IsNC)
        return ELF::R_AARCH64_TLSLE_LDST16_TPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_TPREL && IsNC)
        return ELF::R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for 16-bit load/store instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_ldst_imm12_scale4:
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_LDST32_ABS_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && !IsNC)
        return ELF::R_AARCH64_TLSLD_LDST32_DTPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && IsNC)
        return ELF::R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_TPREL && !IsNC)
        return ELF::R_AARCH64_TLSLE_LDST32_TPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_TPREL && IsNC)
        return ELF::R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for 32-bit load/store instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_ldst_imm12_scale8:
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_LDST64_ABS_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_GOT && IsNC)
        return ELF::R_AARCH64_LD64_GOT_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && !IsNC)
        return ELF::R_AARCH64_TLSLD_LDST64_DTPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_DTPREL && IsNC)
        return ELF::R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_TPREL && !IsNC)
        return ELF::R_AARCH64_TLSLE_LDST64_TPREL_LO12;
      if (SymLoc == AArch64MCExpr::VK_TPREL && IsNC)
        return ELF::R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_GOTTPREL && IsNC)
        return ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC;
      if (SymLoc == AArch64MCExpr::VK_TLSDESC && IsNC)
        return ELF::R_AARCH64_TLSDESC_LD64_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for 64-bit load/store instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_ldst_imm12_scale16:
      if (SymLoc == AArch64MCExpr::VK_ABS && IsNC)
        return ELF::R_AARCH64_LDST128_ABS_LO12_NC;

      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for 128-bit load/store instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_movw:
      if (RefKind == AArch64MCExpr::VK_ABS_G3)
        return ELF::R_AARCH64_MOVW_UABS_G3;
      if (RefKind == AArch64MCExpr::VK_ABS_G2)
        return ELF::R_AARCH64_MOVW_UABS_G2;
      if (RefKind == AArch64MCExpr::VK_ABS_G2_S)
        return ELF::R_AARCH64_MOVW_SABS_G2;
      if (RefKind == AArch64MCExpr::VK_ABS_G2_NC)
        return ELF::R_AARCH64_MOVW_UABS_G2_NC;
      if (RefKind == AArch64MCExpr::VK_ABS_G1)
        return ELF::R_AARCH64_MOVW_UABS_G1;
      if (RefKind == AArch64MCExpr::VK_ABS_G1_S)
        return ELF::R_AARCH64_MOVW_SABS_G1;
      if (RefKind == AArch64MCExpr::VK_ABS_G1_NC)
        return ELF::R_AARCH64_MOVW_UABS_G1_NC;
      if (RefKind == AArch64MCExpr::VK_ABS_G0)
        return ELF::R_AARCH64_MOVW_UABS_G0;
      if (RefKind == AArch64MCExpr::VK_ABS_G0_S)
        return ELF::R_AARCH64_MOVW_SABS_G0;
      if (RefKind == AArch64MCExpr::VK_ABS_G0_NC)
        return ELF::R_AARCH64_MOVW_UABS_G0_NC;
      if (RefKind == AArch64MCExpr::VK_DTPREL_G2)
        return ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G2;
      if (RefKind == AArch64MCExpr::VK_DTPREL_G1)
        return ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G1;
      if (RefKind == AArch64MCExpr::VK_DTPREL_G1_NC)
        return ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC;
      if (RefKind == AArch64MCExpr::VK_DTPREL_G0)
        return ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G0;
      if (RefKind == AArch64MCExpr::VK_DTPREL_G0_NC)
        return ELF::R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC;
      if (RefKind == AArch64MCExpr::VK_TPREL_G2)
        return ELF::R_AARCH64_TLSLE_MOVW_TPREL_G2;
      if (RefKind == AArch64MCExpr::VK_TPREL_G1)
        return ELF::R_AARCH64_TLSLE_MOVW_TPREL_G1;
      if (RefKind == AArch64MCExpr::VK_TPREL_G1_NC)
        return ELF::R_AARCH64_TLSLE_MOVW_TPREL_G1_NC;
      if (RefKind == AArch64MCExpr::VK_TPREL_G0)
        return ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0;
      if (RefKind == AArch64MCExpr::VK_TPREL_G0_NC)
        return ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC;
      if (RefKind == AArch64MCExpr::VK_GOTTPREL_G1)
        return ELF::R_AARCH64_TLSIE_MOVW_GOTTPREL_G1;
      if (RefKind == AArch64MCExpr::VK_GOTTPREL_G0_NC)
        return ELF::R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC;
      Ctx.reportError(Fixup.getLoc(),
                      "invalid fixup for movz/movk instruction");
      return ELF::R_AARCH64_NONE;
    case AArch64::fixup_aarch64_tlsdesc_call:
      return ELF::R_AARCH64_TLSDESC_CALL;
    default:
      Ctx.reportError(Fixup.getLoc(), "Unknown ELF relocation type");
      return ELF::R_AARCH64_NONE;
    }
  }

  llvm_unreachable("Unimplemented fixup -> relocation");
}

MCObjectWriter *llvm::createAArch64ELFObjectWriter(raw_pwrite_stream &OS,
                                                   uint8_t OSABI,
                                                   bool IsLittleEndian) {
  MCELFObjectTargetWriter *MOTW =
      new AArch64ELFObjectWriter(OSABI, IsLittleEndian);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
