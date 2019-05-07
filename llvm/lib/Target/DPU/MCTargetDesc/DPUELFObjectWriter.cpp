//===-- DPUELFObjectWriter.cpp - DPU ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUELFObjectWriter class.
//
//===----------------------------------------------------------------------===//

#include "DPUELFObjectWriter.h"
#include "DPUFixupKinds.h"
#include "llvm/MC/MCValue.h"

#define DEBUG_TYPE "elfobjectwriter"

namespace llvm {

bool DPUELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                 unsigned Type) const {
  return true;
}

unsigned int DPUELFObjectWriter::getRelocType(MCContext &Ctx,
                                              const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsPCRel) const {
  unsigned Type;
  unsigned Kind = static_cast<unsigned>(Fixup.getKind());
  switch (Kind) {
  case DPU::FIXUP_DPU_NONE:
    Type = ELF::R_DPU_NONE;
    break;
  case MCFixupKind ::FK_Data_1:
    Type = ELF::R_DPU_8;
    break;
  case MCFixupKind ::FK_Data_2:
    Type = ELF::R_DPU_16;
    break;
  case MCFixupKind ::FK_Data_4:
  case DPU::FIXUP_DPU_32:
    Type = ELF::R_DPU_32;
    break;
  case MCFixupKind ::FK_Data_8:
    Type = ELF::R_DPU_64;
    break;
  case DPU::FIXUP_DPU_PC:
    Type = ELF::R_DPU_PC;
    break;
  case DPU::FIXUP_DPU_IMM5:
    Type = ELF::R_DPU_IMM5;
    break;
  case DPU::FIXUP_DPU_IMM8_DMA:
    Type = ELF::R_DPU_IMM8_DMA;
    break;
  case DPU::FIXUP_DPU_IMM24_PC:
    Type = ELF::R_DPU_IMM24_PC;
    break;
  case DPU::FIXUP_DPU_IMM27_PC:
    Type = ELF::R_DPU_IMM27_PC;
    break;
  case DPU::FIXUP_DPU_IMM28_PC_OPC8:
    Type = ELF::R_DPU_IMM28_PC_OPC8;
    break;
  case DPU::FIXUP_DPU_IMM8_STR:
    Type = ELF::R_DPU_IMM8_STR;
    break;
  case DPU::FIXUP_DPU_IMM12_STR:
    Type = ELF::R_DPU_IMM12_STR;
    break;
  case DPU::FIXUP_DPU_IMM16_STR:
    Type = ELF::R_DPU_IMM16_STR;
    break;
  case DPU::FIXUP_DPU_IMM16_ATM:
    Type = ELF::R_DPU_IMM16_ATM;
    break;
  case DPU::FIXUP_DPU_IMM24:
    Type = ELF::R_DPU_IMM24;
    break;
  case DPU::FIXUP_DPU_IMM24_RB:
    Type = ELF::R_DPU_IMM24_RB;
    break;
  case DPU::FIXUP_DPU_IMM27:
    Type = ELF::R_DPU_IMM27;
    break;
  case DPU::FIXUP_DPU_IMM28:
    Type = ELF::R_DPU_IMM28;
    break;
  case DPU::FIXUP_DPU_IMM32:
    Type = ELF::R_DPU_IMM32;
    break;
  case DPU::FIXUP_DPU_IMM32_ZERO_RB:
    Type = ELF::R_DPU_IMM32_ZERO_RB;
    break;
  case DPU::FIXUP_DPU_IMM17_24:
    Type = ELF::R_DPU_IMM17_24;
    break;
  case DPU::FIXUP_DPU_IMM32_DUS_RB:
    Type = ELF::R_DPU_IMM32_DUS_RB;
    break;
  default:
    llvm_unreachable("Invalid fixup kind!");
  }
  return Type;
}

std::unique_ptr<MCObjectTargetWriter> createDPUELFObjectWriter() {
  return llvm::make_unique<DPUELFObjectWriter>();
}

} // namespace llvm
