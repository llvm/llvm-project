//===-- ARMELFObjectWriter.cpp - ARM ELF Writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMFixupKinds.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace llvm;

namespace {

class ARMELFObjectWriter : public MCELFObjectTargetWriter {
  enum { DefaultEABIVersion = 0x05000000U };

public:
  ARMELFObjectWriter(uint8_t OSABI);

  ~ARMELFObjectWriter() override = default;

  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;

  bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym,
                               unsigned Type) const override;
};

} // end anonymous namespace

ARMELFObjectWriter::ARMELFObjectWriter(uint8_t OSABI)
  : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI,
                            ELF::EM_ARM,
                            /*HasRelocationAddend*/ false) {}

bool ARMELFObjectWriter::needsRelocateWithSymbol(const MCValue &,
                                                 const MCSymbol &,
                                                 unsigned Type) const {
  // FIXME: This is extremely conservative. This really needs to use an
  // explicit list with a clear explanation for why each realocation needs to
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
unsigned ARMELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                          const MCValue &Target,
                                          bool IsPCRel) const {
  unsigned Kind = Fixup.getTargetKind();
  uint8_t Specifier = Target.getSpecifier();
  auto CheckFDPIC = [&](uint32_t Type) {
    if (getOSABI() != ELF::ELFOSABI_ARM_FDPIC)
      reportError(Fixup.getLoc(),
                  "relocation " +
                      object::getELFRelocationTypeName(ELF::EM_ARM, Type) +
                      " only supported in FDPIC mode");
    return Type;
  };

  switch (Specifier) {
  case ARMMCExpr::VK_GOTTPOFF:
  case ARMMCExpr::VK_GOTTPOFF_FDPIC:
  case ARMMCExpr::VK_TLSCALL:
  case ARMMCExpr::VK_TLSDESC:
  case ARMMCExpr::VK_TLSGD:
  case ARMMCExpr::VK_TLSGD_FDPIC:
  case ARMMCExpr::VK_TLSLDM:
  case ARMMCExpr::VK_TLSLDM_FDPIC:
  case ARMMCExpr::VK_TLSLDO:
  case ARMMCExpr::VK_TPOFF:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  default:
    break;
  }

  if (IsPCRel) {
    switch (Fixup.getTargetKind()) {
    default:
      reportError(Fixup.getLoc(), "unsupported relocation type");
      return ELF::R_ARM_NONE;
    case FK_Data_4:
      switch (Specifier) {
      default:
        reportError(Fixup.getLoc(),
                    "invalid fixup for 4-byte pc-relative data relocation");
        return ELF::R_ARM_NONE;
      case ARMMCExpr::VK_None: {
        if (const auto *SA = Target.getAddSym()) {
          // For GNU AS compatibility expressions such as
          // _GLOBAL_OFFSET_TABLE_ - label emit a R_ARM_BASE_PREL relocation.
          if (SA->getName() == "_GLOBAL_OFFSET_TABLE_")
            return ELF::R_ARM_BASE_PREL;
        }
        return ELF::R_ARM_REL32;
      }
      case ARMMCExpr::VK_GOTTPOFF:
        return ELF::R_ARM_TLS_IE32;
      case ARMMCExpr::VK_GOT_PREL:
        return ELF::R_ARM_GOT_PREL;
      case ARMMCExpr::VK_PREL31:
        return ELF::R_ARM_PREL31;
      }
    case ARM::fixup_arm_blx:
    case ARM::fixup_arm_uncondbl:
      switch (Specifier) {
      case ARMMCExpr::VK_PLT:
        return ELF::R_ARM_CALL;
      case ARMMCExpr::VK_TLSCALL:
        return ELF::R_ARM_TLS_CALL;
      default:
        return ELF::R_ARM_CALL;
      }
    case ARM::fixup_arm_condbl:
    case ARM::fixup_arm_condbranch:
    case ARM::fixup_arm_uncondbranch:
      return ELF::R_ARM_JUMP24;
    case ARM::fixup_t2_condbranch:
      return ELF::R_ARM_THM_JUMP19;
    case ARM::fixup_t2_uncondbranch:
      return ELF::R_ARM_THM_JUMP24;
    case ARM::fixup_arm_movt_hi16:
      return ELF::R_ARM_MOVT_PREL;
    case ARM::fixup_arm_movw_lo16:
      return ELF::R_ARM_MOVW_PREL_NC;
    case ARM::fixup_t2_movt_hi16:
      return ELF::R_ARM_THM_MOVT_PREL;
    case ARM::fixup_t2_movw_lo16:
      return ELF::R_ARM_THM_MOVW_PREL_NC;
    case ARM::fixup_arm_thumb_upper_8_15:
      return ELF::R_ARM_THM_ALU_ABS_G3;
    case ARM::fixup_arm_thumb_upper_0_7:
      return ELF::R_ARM_THM_ALU_ABS_G2_NC;
    case ARM::fixup_arm_thumb_lower_8_15:
      return ELF::R_ARM_THM_ALU_ABS_G1_NC;
    case ARM::fixup_arm_thumb_lower_0_7:
      return ELF::R_ARM_THM_ALU_ABS_G0_NC;
    case ARM::fixup_arm_thumb_br:
      return ELF::R_ARM_THM_JUMP11;
    case ARM::fixup_arm_thumb_bcc:
      return ELF::R_ARM_THM_JUMP8;
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_blx:
      switch (Specifier) {
      case ARMMCExpr::VK_TLSCALL:
        return ELF::R_ARM_THM_TLS_CALL;
      default:
        return ELF::R_ARM_THM_CALL;
      }
    case ARM::fixup_arm_ldst_pcrel_12:
      return ELF::R_ARM_LDR_PC_G0;
    case ARM::fixup_arm_pcrel_10_unscaled:
      return ELF::R_ARM_LDRS_PC_G0;
    case ARM::fixup_t2_ldst_pcrel_12:
      return ELF::R_ARM_THM_PC12;
    case ARM::fixup_arm_adr_pcrel_12:
      return ELF::R_ARM_ALU_PC_G0;
    case ARM::fixup_thumb_adr_pcrel_10:
      return ELF::R_ARM_THM_PC8;
    case ARM::fixup_t2_adr_pcrel_12:
      return ELF::R_ARM_THM_ALU_PREL_11_0;
    case ARM::fixup_bf_target:
      return ELF::R_ARM_THM_BF16;
    case ARM::fixup_bfc_target:
      return ELF::R_ARM_THM_BF12;
    case ARM::fixup_bfl_target:
      return ELF::R_ARM_THM_BF18;
    }
  }
  switch (Kind) {
  default:
    reportError(Fixup.getLoc(), "unsupported relocation type");
    return ELF::R_ARM_NONE;
  case FK_Data_1:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for 1-byte data relocation");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_ABS8;
    }
  case FK_Data_2:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for 2-byte data relocation");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_ABS16;
    }
  case FK_Data_4:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for 4-byte data relocation");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_ARM_NONE:
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_GOT:
      return ELF::R_ARM_GOT_BREL;
    case ARMMCExpr::VK_TLSGD:
      return ELF::R_ARM_TLS_GD32;
    case ARMMCExpr::VK_TPOFF:
      return ELF::R_ARM_TLS_LE32;
    case ARMMCExpr::VK_GOTTPOFF:
      return ELF::R_ARM_TLS_IE32;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_ABS32;
    case ARMMCExpr::VK_GOTOFF:
      return ELF::R_ARM_GOTOFF32;
    case ARMMCExpr::VK_GOT_PREL:
      return ELF::R_ARM_GOT_PREL;
    case ARMMCExpr::VK_TARGET1:
      return ELF::R_ARM_TARGET1;
    case ARMMCExpr::VK_TARGET2:
      return ELF::R_ARM_TARGET2;
    case ARMMCExpr::VK_PREL31:
      return ELF::R_ARM_PREL31;
    case ARMMCExpr::VK_SBREL:
      return ELF::R_ARM_SBREL32;
    case ARMMCExpr::VK_TLSLDO:
      return ELF::R_ARM_TLS_LDO32;
    case ARMMCExpr::VK_TLSCALL:
      return ELF::R_ARM_TLS_CALL;
    case ARMMCExpr::VK_TLSDESC:
      return ELF::R_ARM_TLS_GOTDESC;
    case ARMMCExpr::VK_TLSLDM:
      return ELF::R_ARM_TLS_LDM32;
    case ARMMCExpr::VK_TLSDESCSEQ:
      return ELF::R_ARM_TLS_DESCSEQ;
    case ARMMCExpr::VK_FUNCDESC:
      return CheckFDPIC(ELF::R_ARM_FUNCDESC);
    case ARMMCExpr::VK_GOTFUNCDESC:
      return CheckFDPIC(ELF::R_ARM_GOTFUNCDESC);
    case ARMMCExpr::VK_GOTOFFFUNCDESC:
      return CheckFDPIC(ELF::R_ARM_GOTOFFFUNCDESC);
    case ARMMCExpr::VK_TLSGD_FDPIC:
      return CheckFDPIC(ELF::R_ARM_TLS_GD32_FDPIC);
    case ARMMCExpr::VK_TLSLDM_FDPIC:
      return CheckFDPIC(ELF::R_ARM_TLS_LDM32_FDPIC);
    case ARMMCExpr::VK_GOTTPOFF_FDPIC:
      return CheckFDPIC(ELF::R_ARM_TLS_IE32_FDPIC);
    }
  case ARM::fixup_arm_condbranch:
  case ARM::fixup_arm_uncondbranch:
    return ELF::R_ARM_JUMP24;
  case ARM::fixup_arm_movt_hi16:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for ARM MOVT instruction");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_MOVT_ABS;
    case ARMMCExpr::VK_SBREL:
      return ELF::R_ARM_MOVT_BREL;
    }
  case ARM::fixup_arm_movw_lo16:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for ARM MOVW instruction");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_MOVW_ABS_NC;
    case ARMMCExpr::VK_SBREL:
      return ELF::R_ARM_MOVW_BREL_NC;
    }
  case ARM::fixup_t2_movt_hi16:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for Thumb MOVT instruction");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_THM_MOVT_ABS;
    case ARMMCExpr::VK_SBREL:
      return ELF::R_ARM_THM_MOVT_BREL;
    }
  case ARM::fixup_t2_movw_lo16:
    switch (Specifier) {
    default:
      reportError(Fixup.getLoc(), "invalid fixup for Thumb MOVW instruction");
      return ELF::R_ARM_NONE;
    case ARMMCExpr::VK_None:
      return ELF::R_ARM_THM_MOVW_ABS_NC;
    case ARMMCExpr::VK_SBREL:
      return ELF::R_ARM_THM_MOVW_BREL_NC;
    }

  case ARM::fixup_arm_thumb_upper_8_15:
    return ELF::R_ARM_THM_ALU_ABS_G3;
  case ARM::fixup_arm_thumb_upper_0_7:
    return ELF::R_ARM_THM_ALU_ABS_G2_NC;
  case ARM::fixup_arm_thumb_lower_8_15:
    return ELF::R_ARM_THM_ALU_ABS_G1_NC;
  case ARM::fixup_arm_thumb_lower_0_7:
    return ELF::R_ARM_THM_ALU_ABS_G0_NC;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createARMELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<ARMELFObjectWriter>(OSABI);
}
