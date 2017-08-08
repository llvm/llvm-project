//===-- RISCVELFObjectWriter.cpp - RISCV ELF Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
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
                              /*HasRelocationAddend*/ false) {}

RISCVELFObjectWriter::~RISCVELFObjectWriter() {}

unsigned RISCVELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
  llvm_unreachable("invalid fixup kind!");
}

MCObjectWriter *llvm::createRISCVELFObjectWriter(raw_pwrite_stream &OS,
                                                 uint8_t OSABI, bool Is64Bit) {
  MCELFObjectTargetWriter *MOTW = new RISCVELFObjectWriter(OSABI, Is64Bit);
  return createELFObjectWriter(MOTW, OS, /*IsLittleEndian*/ true);
}
