//===-- BPFELFObjectWriter.cpp - BPF ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class BPFELFObjectWriter : public MCELFObjectTargetWriter {
public:
  BPFELFObjectWriter(uint8_t OSABI);

  ~BPFELFObjectWriter() override;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
}

BPFELFObjectWriter::BPFELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit*/ true, OSABI, ELF::EM_NONE,
                              /*HasRelocationAddend*/ false) {}

BPFELFObjectWriter::~BPFELFObjectWriter() {}

unsigned BPFELFObjectWriter::getRelocType(MCContext &Ctx, const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel) const {
  // determine the type of the relocation
  switch ((unsigned)Fixup.getKind()) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_SecRel_8:
    return ELF::R_X86_64_64;
  case FK_SecRel_4:
    return ELF::R_X86_64_PC32;
  case FK_Data_8:
    return IsPCRel ? ELF::R_X86_64_PC64 : ELF::R_X86_64_64;
  case FK_Data_4:
    return IsPCRel ? ELF::R_X86_64_PC32 : ELF::R_X86_64_32;
  }
}

MCObjectWriter *llvm::createBPFELFObjectWriter(raw_pwrite_stream &OS,
                                               uint8_t OSABI, bool IsLittleEndian) {
  MCELFObjectTargetWriter *MOTW = new BPFELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
