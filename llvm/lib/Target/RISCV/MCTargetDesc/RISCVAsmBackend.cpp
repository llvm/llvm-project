//===-- RISCVAsmBackend.cpp - RISCV Assembler Backend ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class RISCVAsmBackend : public MCAsmBackend {
  uint8_t OSABI;
  bool Is64Bit;

public:
  RISCVAsmBackend(uint8_t OSABI, bool Is64Bit)
      : MCAsmBackend(), OSABI(OSABI), Is64Bit(Is64Bit) {}
  ~RISCVAsmBackend() override {}

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved) const override;

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override;

  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    return false;
  }

  unsigned getNumFixupKinds() const override { return 1; }

  bool mayNeedRelaxation(const MCInst &Inst) const override { return false; }

  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override {

    report_fatal_error("RISCVAsmBackend::relaxInstruction() unimplemented");
  }

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override;
};

bool RISCVAsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  // Once support for the compressed instruction set is added, we will be able
  // to conditionally support 16-bit NOPs
  if ((Count % 4) != 0)
    return false;

  // The canonical nop on RISC-V is addi x0, x0, 0
  for (uint64_t i = 0; i < Count; i += 4)
    OW->write32(0x13);

  return true;
}

void RISCVAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                 const MCValue &Target,
                                 MutableArrayRef<char> Data, uint64_t Value,
                                 bool IsResolved) const {
  return;
}

MCObjectWriter *
RISCVAsmBackend::createObjectWriter(raw_pwrite_stream &OS) const {
  return createRISCVELFObjectWriter(OS, OSABI, Is64Bit);
}

} // end anonymous namespace

MCAsmBackend *llvm::createRISCVAsmBackend(const Target &T,
                                          const MCRegisterInfo &MRI,
                                          const Triple &TT, StringRef CPU,
                                          const MCTargetOptions &Options) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  return new RISCVAsmBackend(OSABI, TT.isArch64Bit());
}
