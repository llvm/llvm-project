//===-- LX32ELFObjectWriter.cpp - LX32 ELF Writer ---------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
#include "LX32MCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/BinaryFormat/ELF.h"

using namespace llvm;

namespace {
class LX32ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  LX32ELFObjectWriter(uint8_t OSABI)
      : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI, ELF::EM_RISCV,
                              /*HasRelocationAddend*/ true) {}

  ~LX32ELFObjectWriter() override {}

protected:
  unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target, bool IsPCRel) const override {
    if (Fixup.getKind() == (MCFixupKind)1 /* branch */)
      return ELF::R_RISCV_BRANCH;
    if (Fixup.getKind() == (MCFixupKind)2 /* jump */)
      return ELF::R_RISCV_JAL;
    return ELF::R_RISCV_NONE; // R_NONE is always 0
  }
};
} // end anonymous namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createLX32ELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<LX32ELFObjectWriter>(OSABI);
}
