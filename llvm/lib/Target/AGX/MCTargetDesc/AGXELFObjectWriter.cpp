//===-- AGXELFObjectWriter.cpp - AGX ELF writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX objects are ELF64 files whose .text holds raw instructions. Branches
// are resolved by the assembler; there are no AGX relocation types.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"

using namespace llvm;

namespace {
class AGXELFObjectWriter : public MCELFObjectTargetWriter {
public:
  explicit AGXELFObjectWriter(uint8_t OSABI)
      : MCELFObjectTargetWriter(/*Is64Bit=*/true, OSABI, ELF::EM_AGX,
                                /*HasRelocationAddend=*/true) {}

protected:
  unsigned getRelocType(const MCFixup &Fixup, const MCValue &,
                        bool) const override {
    reportError(Fixup.getLoc(), "AGX does not support relocations");
    return 0;
  }
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createAGXELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<AGXELFObjectWriter>(OSABI);
}
