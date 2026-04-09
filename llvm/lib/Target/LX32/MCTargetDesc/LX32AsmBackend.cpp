//===-- LX32AsmBackend.cpp - LX32 Assembler Backend -------------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "LX32MCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class LX32AsmBackend : public MCAsmBackend {
public:
  LX32AsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI)
      : MCAsmBackend(llvm::endianness::little) {}
  ~LX32AsmBackend() override {}

  void applyFixup(const MCFragment &Fragment, const MCFixup &Fixup,
                  const MCValue &Target, uint8_t *Data, uint64_t Value,
                  bool IsResolved) override {
    if (!IsResolved) return;
    uint32_t CurVal = 0;
    CurVal =  (uint32_t)Data[0] | ((uint32_t)Data[1] << 8) |
             ((uint32_t)Data[2] << 16) | ((uint32_t)Data[3] << 24);

    if (Fixup.getKind() == (MCFixupKind)1 /* branch */) {
      uint32_t imm = Value;
      uint32_t bit11 = (imm >> 11) & 1;
      uint32_t bit4_1 = (imm >> 1) & 0xF;
      uint32_t bit10_5 = (imm >> 5) & 0x3F;
      uint32_t bit12 = (imm >> 12) & 1;
      CurVal |= (bit11 << 7) | (bit4_1 << 8) | (bit10_5 << 25) | (bit12 << 31);
    } else if (Fixup.getKind() == (MCFixupKind)2 /* jump */) {
      uint32_t imm = Value;
      uint32_t bit19_12 = (imm >> 12) & 0xFF;
      uint32_t bit11 = (imm >> 11) & 1;
      uint32_t bit10_1 = (imm >> 1) & 0x3FF;
      uint32_t bit20 = (imm >> 20) & 1;
      CurVal |= (bit19_12 << 12) | (bit11 << 20) | (bit10_1 << 21) | (bit20 << 31);
    }

    Data[0] = CurVal & 0xFF;
    Data[1] = (CurVal >> 8) & 0xFF;
    Data[2] = (CurVal >> 16) & 0xFF;
    Data[3] = (CurVal >> 24) & 0xFF;
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override {
    // 4-byte NOPs = addi x0, x0, 0 = 0x00000013
    uint64_t NumNops = Count / 4;
    for (uint64_t i = 0; i != NumNops; ++i)
      OS.write("\x13\x00\x00\x00", 4);

    OS.write_zeros(Count % 4);
    return true;
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createLX32ELFObjectWriter(0);
  }
};
} // end anonymous namespace

MCAsmBackend *llvm::createLX32AsmBackend(const Target &T,
                                         const MCSubtargetInfo &STI,
                                         const MCRegisterInfo &MRI,
                                         const MCTargetOptions &Options) {
  return new LX32AsmBackend(STI, 0); // OSABI 0
}
