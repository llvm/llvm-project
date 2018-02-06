//===-- RISCVELFObjectWriter.cpp - RISCV ELF Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class RISCVELFObjectWriter : public MCELFObjectTargetWriter {
public:
  RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~RISCVELFObjectWriter() override;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
}

RISCVELFObjectWriter::RISCVELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_RISCV,
                              /*HasRelocationAddend*/ true) {}

RISCVELFObjectWriter::~RISCVELFObjectWriter() {}

unsigned RISCVELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
  // Determine the type of the relocation
  switch ((unsigned)Fixup.getKind()) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    return ELF::R_RISCV_32;
  case FK_Data_8:
    return ELF::R_RISCV_64;
  case RISCV::fixup_riscv_hi20:
    return ELF::R_RISCV_HI20;
  case RISCV::fixup_riscv_lo12_i:
    return ELF::R_RISCV_LO12_I;
  case RISCV::fixup_riscv_lo12_s:
    return ELF::R_RISCV_LO12_S;
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
  }
}

std::unique_ptr<MCObjectWriter>
llvm::createRISCVELFObjectWriter(raw_pwrite_stream &OS, uint8_t OSABI,
                                 bool Is64Bit) {
  return createELFObjectWriter(
      llvm::make_unique<RISCVELFObjectWriter>(OSABI, Is64Bit), OS,
      /*IsLittleEndian=*/true);
}
