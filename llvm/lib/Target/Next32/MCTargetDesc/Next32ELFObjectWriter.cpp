//===-- Next32ELFObjectWriter.cpp - Next32 ELF Writer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32FixupKinds.h"
#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace llvm;

namespace {

class Next32ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  Next32ELFObjectWriter(uint8_t OSABI);
  ~Next32ELFObjectWriter() override = default;

  bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym,
                               unsigned Type) const override;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};

} // end anonymous namespace

Next32ELFObjectWriter::Next32ELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit*/ true, OSABI, ELF::EM_NEXT32,
                              /*HasRelocationAddend*/ true) {}

bool Next32ELFObjectWriter::needsRelocateWithSymbol(const MCValue &Val,
                                                    const MCSymbol &Sym,
                                                    unsigned Type) const {
  switch (Type) {
  case ELF::R_NEXT32_SYM_BB_IMM:
  case ELF::R_NEXT32_SYM_FUNCTION:
  case ELF::R_NEXT32_SYM_MEM_64HI:
  case ELF::R_NEXT32_SYM_MEM_64LO:
  case ELF::R_NEXT32_SYM_FUNC_64HI:
  case ELF::R_NEXT32_SYM_FUNC_64LO:
    return true;
  default:
    return false;
  }
}

unsigned Next32ELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
  // determine the type of the relocation
  switch ((unsigned)Fixup.getKind()) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    return ELF::R_NEXT32_ABS32;
  case FK_Data_8:
    return ELF::R_NEXT32_ABS64;
  case Next32::reloc_4byte_sym_function:
    return ELF::R_NEXT32_SYM_FUNCTION;
  case Next32::reloc_4byte_sym_bb_imm:
    return ELF::R_NEXT32_SYM_BB_IMM;
  case Next32::reloc_4byte_mem_high:
    return ELF::R_NEXT32_SYM_MEM_64HI;
  case Next32::reloc_4byte_mem_low:
    return ELF::R_NEXT32_SYM_MEM_64LO;
  case Next32::reloc_4byte_func_high:
    return ELF::R_NEXT32_SYM_FUNC_64HI;
  case Next32::reloc_4byte_func_low:
    return ELF::R_NEXT32_SYM_FUNC_64LO;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createNext32ELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<Next32ELFObjectWriter>(OSABI);
}
