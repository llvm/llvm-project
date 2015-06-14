//===-- MipsELFObjectWriter.cpp - Mips ELF Writer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include <list>

using namespace llvm;

namespace {
  class MipsELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                        bool _isN64, bool IsLittleEndian);

    virtual ~MipsELFObjectWriter();

    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const override;
    bool needsRelocateWithSymbol(const MCSymbolData &SD,
                                 unsigned Type) const override;
  };
}

MipsELFObjectWriter::MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                                         bool _isN64, bool IsLittleEndian)
  : MCELFObjectTargetWriter(_is64Bit, OSABI, ELF::EM_MIPS,
                            /*HasRelocationAddend*/ (_isN64) ? true : false,
                            /*IsN64*/ _isN64) {}

MipsELFObjectWriter::~MipsELFObjectWriter() {}

unsigned MipsELFObjectWriter::GetRelocType(const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
  // determine the type of the relocation
  unsigned Type = (unsigned)ELF::R_MIPS_NONE;
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    return IsPCRel ? ELF::R_MIPS_PC32 : ELF::R_MIPS_32;
  case FK_Data_8:
    Type = ELF::R_MIPS_64;
    break;
  case FK_GPRel_4:
    if (isN64()) {
      Type = setRType((unsigned)ELF::R_MIPS_GPREL32, Type);
      Type = setRType2((unsigned)ELF::R_MIPS_64, Type);
      Type = setRType3((unsigned)ELF::R_MIPS_NONE, Type);
    }
    else
      Type = ELF::R_MIPS_GPREL32;
    break;
  case Mips::fixup_Mips_GPREL16:
    Type = ELF::R_MIPS_GPREL16;
    break;
  case Mips::fixup_Mips_26:
    Type = ELF::R_MIPS_26;
    break;
  case Mips::fixup_Mips_CALL16:
    Type = ELF::R_MIPS_CALL16;
    break;
  case Mips::fixup_Mips_GOT_Global:
  case Mips::fixup_Mips_GOT_Local:
    Type = ELF::R_MIPS_GOT16;
    break;
  case Mips::fixup_Mips_HI16:
    Type = ELF::R_MIPS_HI16;
    break;
  case Mips::fixup_Mips_LO16:
    Type = ELF::R_MIPS_LO16;
    break;
  case Mips::fixup_Mips_TLSGD:
    Type = ELF::R_MIPS_TLS_GD;
    break;
  case Mips::fixup_Mips_GOTTPREL:
    Type = ELF::R_MIPS_TLS_GOTTPREL;
    break;
  case Mips::fixup_Mips_TPREL_HI:
    Type = ELF::R_MIPS_TLS_TPREL_HI16;
    break;
  case Mips::fixup_Mips_TPREL_LO:
    Type = ELF::R_MIPS_TLS_TPREL_LO16;
    break;
  case Mips::fixup_Mips_TLSLDM:
    Type = ELF::R_MIPS_TLS_LDM;
    break;
  case Mips::fixup_Mips_DTPREL_HI:
    Type = ELF::R_MIPS_TLS_DTPREL_HI16;
    break;
  case Mips::fixup_Mips_DTPREL_LO:
    Type = ELF::R_MIPS_TLS_DTPREL_LO16;
    break;
  case Mips::fixup_Mips_Branch_PCRel:
  case Mips::fixup_Mips_PC16:
    Type = ELF::R_MIPS_PC16;
    break;
  case Mips::fixup_Mips_GOT_PAGE:
    Type = ELF::R_MIPS_GOT_PAGE;
    break;
  case Mips::fixup_Mips_GOT_OFST:
    Type = ELF::R_MIPS_GOT_OFST;
    break;
  case Mips::fixup_Mips_GOT_DISP:
    Type = ELF::R_MIPS_GOT_DISP;
    break;
  case Mips::fixup_Mips_GPOFF_HI:
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_HI16, Type);
    break;
  case Mips::fixup_Mips_GPOFF_LO:
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_LO16, Type);
    break;
  case Mips::fixup_Mips_HIGHER:
    Type = ELF::R_MIPS_HIGHER;
    break;
  case Mips::fixup_Mips_HIGHEST:
    Type = ELF::R_MIPS_HIGHEST;
    break;
  case Mips::fixup_Mips_GOT_HI16:
    Type = ELF::R_MIPS_GOT_HI16;
    break;
  case Mips::fixup_Mips_GOT_LO16:
    Type = ELF::R_MIPS_GOT_LO16;
    break;
  case Mips::fixup_Mips_CALL_HI16:
    Type = ELF::R_MIPS_CALL_HI16;
    break;
  case Mips::fixup_Mips_CALL_LO16:
    Type = ELF::R_MIPS_CALL_LO16;
    break;
  case Mips::fixup_MICROMIPS_26_S1:
    Type = ELF::R_MICROMIPS_26_S1;
    break;
  case Mips::fixup_MICROMIPS_HI16:
    Type = ELF::R_MICROMIPS_HI16;
    break;
  case Mips::fixup_MICROMIPS_LO16:
    Type = ELF::R_MICROMIPS_LO16;
    break;
  case Mips::fixup_MICROMIPS_GOT16:
    Type = ELF::R_MICROMIPS_GOT16;
    break;
  case Mips::fixup_MICROMIPS_PC7_S1:
    Type = ELF::R_MICROMIPS_PC7_S1;
    break;
  case Mips::fixup_MICROMIPS_PC16_S1:
    Type = ELF::R_MICROMIPS_PC16_S1;
    break;
  case Mips::fixup_MICROMIPS_CALL16:
    Type = ELF::R_MICROMIPS_CALL16;
    break;
  case Mips::fixup_MICROMIPS_GOT_DISP:
    Type = ELF::R_MICROMIPS_GOT_DISP;
    break;
  case Mips::fixup_MICROMIPS_GOT_PAGE:
    Type = ELF::R_MICROMIPS_GOT_PAGE;
    break;
  case Mips::fixup_MICROMIPS_GOT_OFST:
    Type = ELF::R_MICROMIPS_GOT_OFST;
    break;
  case Mips::fixup_MICROMIPS_TLS_GD:
    Type = ELF::R_MICROMIPS_TLS_GD;
    break;
  case Mips::fixup_MICROMIPS_TLS_LDM:
    Type = ELF::R_MICROMIPS_TLS_LDM;
    break;
  case Mips::fixup_MICROMIPS_TLS_DTPREL_HI16:
    Type = ELF::R_MICROMIPS_TLS_DTPREL_HI16;
    break;
  case Mips::fixup_MICROMIPS_TLS_DTPREL_LO16:
    Type = ELF::R_MICROMIPS_TLS_DTPREL_LO16;
    break;
  case Mips::fixup_MICROMIPS_TLS_TPREL_HI16:
    Type = ELF::R_MICROMIPS_TLS_TPREL_HI16;
    break;
  case Mips::fixup_MICROMIPS_TLS_TPREL_LO16:
    Type = ELF::R_MICROMIPS_TLS_TPREL_LO16;
    break;
  case Mips::fixup_MIPS_PC19_S2:
    Type = ELF::R_MIPS_PC19_S2;
    break;
  case Mips::fixup_MIPS_PC18_S3:
    Type = ELF::R_MIPS_PC18_S3;
    break;
  case Mips::fixup_MIPS_PC21_S2:
    Type = ELF::R_MIPS_PC21_S2;
    break;
  case Mips::fixup_MIPS_PC26_S2:
    Type = ELF::R_MIPS_PC26_S2;
    break;
  case Mips::fixup_MIPS_PCHI16:
    Type = ELF::R_MIPS_PCHI16;
    break;
  case Mips::fixup_MIPS_PCLO16:
    Type = ELF::R_MIPS_PCLO16;
    break;
  }
  return Type;
}

bool
MipsELFObjectWriter::needsRelocateWithSymbol(const MCSymbolData &SD,
                                             unsigned Type) const {
  // FIXME: This is extremely conservative. This really needs to use a
  // whitelist with a clear explanation for why each realocation needs to
  // point to the symbol, not to the section.
  switch (Type) {
  default:
    return true;

  case ELF::R_MIPS_GOT16:
  case ELF::R_MIPS16_GOT16:
  case ELF::R_MICROMIPS_GOT16:
    llvm_unreachable("Should have been handled already");

  // These relocations might be paired with another relocation. The pairing is
  // done by the static linker by matching the symbol. Since we only see one
  // relocation at a time, we have to force them to relocate with a symbol to
  // avoid ending up with a pair where one points to a section and another
  // points to a symbol.
  case ELF::R_MIPS_HI16:
  case ELF::R_MIPS16_HI16:
  case ELF::R_MICROMIPS_HI16:
  case ELF::R_MIPS_LO16:
  case ELF::R_MIPS16_LO16:
  case ELF::R_MICROMIPS_LO16:
    return true;

  case ELF::R_MIPS_32:
    if (MCELF::getOther(SD) & (ELF::STO_MIPS_MICROMIPS >> 2))
      return true;
    // falltrough
  case ELF::R_MIPS_26:
  case ELF::R_MIPS_64:
  case ELF::R_MIPS_GPREL16:
    return false;
  }
}

MCObjectWriter *llvm::createMipsELFObjectWriter(raw_ostream &OS,
                                                uint8_t OSABI,
                                                bool IsLittleEndian,
                                                bool Is64Bit) {
  MCELFObjectTargetWriter *MOTW = new MipsELFObjectWriter(Is64Bit, OSABI,
                                                (Is64Bit) ? true : false,
                                                IsLittleEndian);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
