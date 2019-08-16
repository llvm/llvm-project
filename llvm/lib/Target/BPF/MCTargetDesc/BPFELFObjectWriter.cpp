//===-- BPFELFObjectWriter.cpp - BPF ELF Writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace llvm;

namespace {

class BPFELFObjectWriter : public MCELFObjectTargetWriter {
public:
  BPFELFObjectWriter(uint8_t OSABI);
  ~BPFELFObjectWriter() override = default;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};

} // end anonymous namespace

BPFELFObjectWriter::BPFELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit*/ true, OSABI, ELF::EM_BPF,
                              /*HasRelocationAddend*/ false) {}

unsigned BPFELFObjectWriter::getRelocType(MCContext &Ctx, const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel) const {
  // determine the type of the relocation
  switch ((unsigned)Fixup.getKind()) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_SecRel_8:
    return ELF::R_BPF_64_64;
  case FK_PCRel_4:
  case FK_SecRel_4:
    return ELF::R_BPF_64_32;
  case FK_Data_8:
    return ELF::R_BPF_64_64;
  case FK_Data_4:
    if (const MCSymbolRefExpr *A = Target.getSymA()) {
      const MCSymbol &Sym = A->getSymbol();

      if (Sym.isDefined()) {
        MCSection &Section = Sym.getSection();
        const MCSectionELF *SectionELF = dyn_cast<MCSectionELF>(&Section);
        assert(SectionELF && "Null section for reloc symbol");

        unsigned Flags = SectionELF->getFlags();

        if (Sym.isTemporary()) {
          // .BTF.ext generates FK_Data_4 relocations for
          // insn offset by creating temporary labels.
          // The insn offset is within the code section and
          // already been fulfilled by applyFixup(). No
          // further relocation is needed.
          // The reloc symbol should be in text section.
          if ((Flags & ELF::SHF_ALLOC) && (Flags & ELF::SHF_EXECINSTR))
            return ELF::R_BPF_NONE;
        } else {
          // .BTF generates FK_Data_4 relocations for variable
          // offset in DataSec kind. Similar to the above .BTF.ext
          // insn offset, no further relocation is needed.
          // The reloc symbol should be in data section.
          if ((Flags & ELF::SHF_ALLOC) && (Flags & ELF::SHF_WRITE))
            return ELF::R_BPF_NONE;
        }
      }
    }
    return ELF::R_BPF_64_32;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createBPFELFObjectWriter(uint8_t OSABI) {
  return llvm::make_unique<BPFELFObjectWriter>(OSABI);
}
