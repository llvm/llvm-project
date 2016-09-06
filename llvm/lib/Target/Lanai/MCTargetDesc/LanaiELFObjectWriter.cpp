//===-- LanaiELFObjectWriter.cpp - Lanai ELF Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/LanaiBaseInfo.h"
#include "MCTargetDesc/LanaiFixupKinds.h"
#include "MCTargetDesc/LanaiMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class LanaiELFObjectWriter : public MCELFObjectTargetWriter {
public:
  explicit LanaiELFObjectWriter(uint8_t OSABI);

  ~LanaiELFObjectWriter() override;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
  bool needsRelocateWithSymbol(const MCSymbol &SD,
                               unsigned Type) const override;
};
} // namespace

LanaiELFObjectWriter::LanaiELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit_=*/false, OSABI, ELF::EM_LANAI,
                              /*HasRelocationAddend=*/true) {}

LanaiELFObjectWriter::~LanaiELFObjectWriter() {}

unsigned LanaiELFObjectWriter::getRelocType(MCContext & /*Ctx*/,
                                            const MCValue & /*Target*/,
                                            const MCFixup &Fixup,
                                            bool /*IsPCRel*/) const {
  unsigned Type;
  unsigned Kind = static_cast<unsigned>(Fixup.getKind());
  switch (Kind) {
  case Lanai::FIXUP_LANAI_21:
    Type = ELF::R_LANAI_21;
    break;
  case Lanai::FIXUP_LANAI_21_F:
    Type = ELF::R_LANAI_21_F;
    break;
  case Lanai::FIXUP_LANAI_25:
    Type = ELF::R_LANAI_25;
    break;
  case Lanai::FIXUP_LANAI_32:
  case FK_Data_4:
    Type = ELF::R_LANAI_32;
    break;
  case Lanai::FIXUP_LANAI_HI16:
    Type = ELF::R_LANAI_HI16;
    break;
  case Lanai::FIXUP_LANAI_LO16:
    Type = ELF::R_LANAI_LO16;
    break;
  case Lanai::FIXUP_LANAI_NONE:
    Type = ELF::R_LANAI_NONE;
    break;

  default:
    llvm_unreachable("Invalid fixup kind!");
  }
  return Type;
}

bool LanaiELFObjectWriter::needsRelocateWithSymbol(const MCSymbol & /*SD*/,
                                                   unsigned Type) const {
  switch (Type) {
  case ELF::R_LANAI_21:
  case ELF::R_LANAI_21_F:
  case ELF::R_LANAI_25:
  case ELF::R_LANAI_32:
  case ELF::R_LANAI_HI16:
    return true;
  default:
    return false;
  }
}

MCObjectWriter *llvm::createLanaiELFObjectWriter(raw_pwrite_stream &OS,
                                                 uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new LanaiELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian=*/false);
}
