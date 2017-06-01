//===-- ARMELFObjectWriter.cpp - ARM ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "MCTargetDesc/ARMFixupKinds.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  class ARMELFObjectWriter : public MCELFObjectTargetWriter {
    enum { DefaultEABIVersion = 0x05000000U };
    unsigned GetRelocTypeInner(const MCValue &Target,
                               const MCFixup &Fixup,
                               bool IsPCRel) const;


  public:
    ARMELFObjectWriter(uint8_t OSABI);

    ~ARMELFObjectWriter() override;

    unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                          const MCFixup &Fixup, bool IsPCRel) const override;

    bool needsRelocateWithSymbol(const MCSymbol &Sym,
                                 unsigned Type) const override;
  };
}

ARMELFObjectWriter::ARMELFObjectWriter(uint8_t OSABI)
  : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI,
                            ELF::EM_ARM,
                            /*HasRelocationAddend*/ false) {}

ARMELFObjectWriter::~ARMELFObjectWriter() {}

bool ARMELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                 unsigned Type) const {
  // FIXME: This is extremely conservative. This really needs to use a
  // whitelist with a clear explanation for why each realocation needs to
  // point to the symbol, not to the section.
  switch (Type) {
  default:
    return true;

  case ELF::R_ARM_PREL31:
  case ELF::R_ARM_ABS32:
    return false;
  }
}

// Need to examine the Fixup when determining whether to 
// emit the relocation as an explicit symbol or as a section relative
// offset
unsigned ARMELFObjectWriter::getRelocType(MCContext &Ctx, const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel) const {
  return GetRelocTypeInner(Target, Fixup, IsPCRel);
}

unsigned ARMELFObjectWriter::GetRelocTypeInner(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const  {
  MCSymbolRefExpr::VariantKind Modifier = Target.getAccessVariant();

  unsigned Type = 0;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      report_fatal_error("unsupported relocation on symbol");
      return ELF::R_ARM_NONE;
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_REL32;
        break;
      case MCSymbolRefExpr::VK_TLSGD:
        llvm_unreachable("unimplemented");
      case MCSymbolRefExpr::VK_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOT_PREL:
        Type = ELF::R_ARM_GOT_PREL;
        break;
      case MCSymbolRefExpr::VK_ARM_PREL31:
        Type = ELF::R_ARM_PREL31;
        break;
      }
      break;
    case ARM::fixup_arm_blx:
    case ARM::fixup_arm_uncondbl:
      switch (Modifier) {
      case MCSymbolRefExpr::VK_PLT:
        Type = ELF::R_ARM_CALL;
        break;
      case MCSymbolRefExpr::VK_TLSCALL:
        Type = ELF::R_ARM_TLS_CALL;
        break;
      default:
        Type = ELF::R_ARM_CALL;
        break;
      }
      break;
    case ARM::fixup_arm_condbl:
    case ARM::fixup_arm_condbranch:
    case ARM::fixup_arm_uncondbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_t2_condbranch:
      Type = ELF::R_ARM_THM_JUMP19;
      break;
    case ARM::fixup_t2_uncondbranch:
      Type = ELF::R_ARM_THM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
      Type = ELF::R_ARM_MOVT_PREL;
      break;
    case ARM::fixup_arm_movw_lo16:
      Type = ELF::R_ARM_MOVW_PREL_NC;
      break;
    case ARM::fixup_t2_movt_hi16:
      Type = ELF::R_ARM_THM_MOVT_PREL;
      break;
    case ARM::fixup_t2_movw_lo16:
      Type = ELF::R_ARM_THM_MOVW_PREL_NC;
      break;
    case ARM::fixup_arm_thumb_br:
      Type = ELF::R_ARM_THM_JUMP11;
      break;
    case ARM::fixup_arm_thumb_bcc:
      Type = ELF::R_ARM_THM_JUMP8;
      break;
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_blx:
      switch (Modifier) {
      case MCSymbolRefExpr::VK_TLSCALL:
        Type = ELF::R_ARM_THM_TLS_CALL;
        break;
      default:
        Type = ELF::R_ARM_THM_CALL;
        break;
      }
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default:
      report_fatal_error("unsupported relocation on symbol");
      return ELF::R_ARM_NONE;
    case FK_Data_1:
      switch (Modifier) {
      default: llvm_unreachable("unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS8;
        break;
      }
      break;
    case FK_Data_2:
      switch (Modifier) {
      default: llvm_unreachable("unsupported modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS16;
        break;
      }
      break;
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_ARM_NONE:
        Type = ELF::R_ARM_NONE;
        break;
      case MCSymbolRefExpr::VK_GOT:
        Type = ELF::R_ARM_GOT_BREL;
        break;
      case MCSymbolRefExpr::VK_TLSGD:
        Type = ELF::R_ARM_TLS_GD32;
        break;
      case MCSymbolRefExpr::VK_TPOFF:
        Type = ELF::R_ARM_TLS_LE32;
        break;
      case MCSymbolRefExpr::VK_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS32;
        break;
      case MCSymbolRefExpr::VK_GOTOFF:
        Type = ELF::R_ARM_GOTOFF32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOT_PREL:
        Type = ELF::R_ARM_GOT_PREL;
        break;
      case MCSymbolRefExpr::VK_ARM_TARGET1:
        Type = ELF::R_ARM_TARGET1;
        break;
      case MCSymbolRefExpr::VK_ARM_TARGET2:
        Type = ELF::R_ARM_TARGET2;
        break;
      case MCSymbolRefExpr::VK_ARM_PREL31:
        Type = ELF::R_ARM_PREL31;
        break;
      case MCSymbolRefExpr::VK_ARM_SBREL:
        Type = ELF::R_ARM_SBREL32;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSLDO:
        Type = ELF::R_ARM_TLS_LDO32;
        break;
      case MCSymbolRefExpr::VK_TLSCALL:
        Type = ELF::R_ARM_TLS_CALL;
        break;
      case MCSymbolRefExpr::VK_TLSDESC:
        Type = ELF::R_ARM_TLS_GOTDESC;
        break;
      case MCSymbolRefExpr::VK_TLSLDM:
        Type = ELF::R_ARM_TLS_LDM32;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSDESCSEQ:
        Type = ELF::R_ARM_TLS_DESCSEQ;
        break;
      }
      break;
    case ARM::fixup_arm_ldst_pcrel_12:
    case ARM::fixup_arm_pcrel_10:
    case ARM::fixup_arm_adr_pcrel_12:
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_cb:
    case ARM::fixup_arm_thumb_cp:
    case ARM::fixup_arm_thumb_br:
      llvm_unreachable("Unimplemented");
    case ARM::fixup_arm_condbranch:
    case ARM::fixup_arm_uncondbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_MOVT_ABS;
        break;
      case MCSymbolRefExpr::VK_ARM_SBREL:
        Type = ELF:: R_ARM_MOVT_BREL;
        break;
      }
      break;
    case ARM::fixup_arm_movw_lo16:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_MOVW_ABS_NC;
        break;
      case MCSymbolRefExpr::VK_ARM_SBREL:
        Type = ELF:: R_ARM_MOVW_BREL_NC;
        break;
      }
      break;
    case ARM::fixup_t2_movt_hi16:
      Type = ELF::R_ARM_THM_MOVT_ABS;
      break;
    case ARM::fixup_t2_movw_lo16:
      Type = ELF::R_ARM_THM_MOVW_ABS_NC;
      break;
    }
  }

  return Type;
}

MCObjectWriter *llvm::createARMELFObjectWriter(raw_pwrite_stream &OS,
                                               uint8_t OSABI,
                                               bool IsLittleEndian) {
  MCELFObjectTargetWriter *MOTW = new ARMELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
