//===-- AGXELFObjectWriter.cpp - AGX ELF writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX objects are ELF64 files whose .text holds raw instructions. Branches
// are resolved by the assembler. Global data that holds addresses of global
// data (.agx.globals) has absolute relocations, which the loader applies once
// it has placed the global buffer:
//
//   R_AGX_ABS64 = 1   S + A, 64 bits
//   R_AGX_ABS32 = 2   S + A, 32 bits
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXFixupKinds.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

namespace {
class AGXELFObjectWriter : public MCELFObjectTargetWriter {
public:
  explicit AGXELFObjectWriter(uint8_t OSABI)
      : MCELFObjectTargetWriter(/*Is64Bit=*/true, OSABI, ELF::EM_AGX,
                                /*HasRelocationAddend=*/true) {}

protected:
  unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                        bool IsPCRel) const override {
    // R_AGX_CALL32 (3): a call's 32-bit offset, from the start of the call
    // instruction (two bytes before the field) to the function: S + A - P + 2
    if (IsPCRel && Fixup.getKind() == AGX::fixup_agx_branch32)
      return 3;
    if (!IsPCRel && !Target.getSubSym()) {
      switch (Fixup.getKind()) {
      case FK_Data_8:
        return 1;
      case FK_Data_4:
        return 2;
      default:
        break;
      }
    }
    reportError(Fixup.getLoc(), "AGX supports only absolute 32- and 64-bit "
                                "relocations in data");
    return 0;
  }
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createAGXELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<AGXELFObjectWriter>(OSABI);
}
