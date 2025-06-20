//===-- RISCVELFObjectWriter.cpp - RISC-V ELF Writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class RISCVELFObjectWriter : public MCELFObjectTargetWriter {
public:
  RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~RISCVELFObjectWriter() override;

  // Return true if the given relocation must be with a symbol rather than
  // section plus offset.
  bool needsRelocateWithSymbol(const MCValue &, unsigned Type) const override {
    // TODO: this is very conservative, update once RISC-V psABI requirements
    //       are clarified.
    return true;
  }

protected:
  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
};
}

RISCVELFObjectWriter::RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_RISCV,
                              /*HasRelocationAddend*/ true) {}

RISCVELFObjectWriter::~RISCVELFObjectWriter() = default;

unsigned RISCVELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  unsigned Kind = Fixup.getTargetKind();
  auto Spec = Target.getSpecifier();
  switch (Spec) {
  case ELF::R_RISCV_TPREL_HI20:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
  case ELF::R_RISCV_TLSDESC_HI20:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  case ELF::R_RISCV_PLT32:
  case ELF::R_RISCV_GOT32_PCREL:
    if (Kind == FK_Data_4)
      break;
    reportError(Fixup.getLoc(), "%" + RISCV::getSpecifierName(Spec) +
                                    " can only be used in a .word directive");
    return ELF::R_RISCV_NONE;
  default:
    break;
  }

  // Extract the relocation type from the fixup kind, after applying STT_TLS as
  // needed.
  if (mc::isRelocation(Fixup.getKind()))
    return Kind;

  if (IsPCRel) {
    switch (Kind) {
    default:
      reportError(Fixup.getLoc(), "unsupported relocation type");
      return ELF::R_RISCV_NONE;
    case FK_Data_4:
      return ELF::R_RISCV_32_PCREL;
    case RISCV::fixup_riscv_pcrel_hi20:
      return ELF::R_RISCV_PCREL_HI20;
    case RISCV::fixup_riscv_pcrel_lo12_i:
      return ELF::R_RISCV_PCREL_LO12_I;
    case RISCV::fixup_riscv_pcrel_lo12_s:
      return ELF::R_RISCV_PCREL_LO12_S;
    case RISCV::fixup_riscv_jal:
      return ELF::R_RISCV_JAL;
    case RISCV::fixup_riscv_branch:
      return ELF::R_RISCV_BRANCH;
    case RISCV::fixup_riscv_rvc_jump:
      return ELF::R_RISCV_RVC_JUMP;
    case RISCV::fixup_riscv_rvc_branch:
      return ELF::R_RISCV_RVC_BRANCH;
    case RISCV::fixup_riscv_call:
      return ELF::R_RISCV_CALL_PLT;
    case RISCV::fixup_riscv_call_plt:
      return ELF::R_RISCV_CALL_PLT;
    case RISCV::fixup_riscv_qc_e_branch:
      return ELF::R_RISCV_QC_E_BRANCH;
    case RISCV::fixup_riscv_qc_e_jump_plt:
      return ELF::R_RISCV_QC_E_JUMP_PLT;
    }
  }

  switch (Kind) {
  default:
    reportError(Fixup.getLoc(), "unsupported relocation type");
    return ELF::R_RISCV_NONE;

  case FK_Data_1:
    reportError(Fixup.getLoc(), "1-byte data relocations not supported");
    return ELF::R_RISCV_NONE;
  case FK_Data_2:
    reportError(Fixup.getLoc(), "2-byte data relocations not supported");
    return ELF::R_RISCV_NONE;
  case FK_Data_4:
    switch (Spec) {
    case ELF::R_RISCV_32_PCREL:
    case ELF::R_RISCV_GOT32_PCREL:
    case ELF::R_RISCV_PLT32:
      return Spec;
    }
    return ELF::R_RISCV_32;
  case FK_Data_8:
    return ELF::R_RISCV_64;
  case RISCV::fixup_riscv_hi20:
    return ELF::R_RISCV_HI20;
  case RISCV::fixup_riscv_lo12_i:
    return ELF::R_RISCV_LO12_I;
  case RISCV::fixup_riscv_lo12_s:
    return ELF::R_RISCV_LO12_S;
  case RISCV::fixup_riscv_qc_e_32:
    return ELF::R_RISCV_QC_E_32;
  case RISCV::fixup_riscv_qc_abs20_u:
    return ELF::R_RISCV_QC_ABS20_U;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createRISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit) {
  return std::make_unique<RISCVELFObjectWriter>(OSABI, Is64Bit);
}
