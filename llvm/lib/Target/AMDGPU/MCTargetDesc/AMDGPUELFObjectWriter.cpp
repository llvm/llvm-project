//===- AMDGPUELFObjectWriter.cpp - AMDGPU ELF Writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

class AMDGPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  AMDGPUELFObjectWriter(bool Is64Bit, bool HasRelocationAddend);

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};


} // end anonymous namespace

AMDGPUELFObjectWriter::AMDGPUELFObjectWriter(bool Is64Bit,
                                             bool HasRelocationAddend)
  : MCELFObjectTargetWriter(Is64Bit,
                            ELF::ELFOSABI_AMDGPU_HSA,
                            ELF::EM_AMDGPU,
                            HasRelocationAddend) {}

unsigned AMDGPUELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
  if (const auto *SymA = Target.getSymA()) {
    // SCRATCH_RSRC_DWORD[01] is a special global variable that represents
    // the scratch buffer.
    if (SymA->getSymbol().getName() == "SCRATCH_RSRC_DWORD0")
      return ELF::R_AMDGPU_ABS32_LO;

    if (SymA->getSymbol().getName() == "SCRATCH_RSRC_DWORD1")
      return ELF::R_AMDGPU_ABS32_HI;
  }

  switch (Target.getAccessVariant()) {
  default:
    break;
  case MCSymbolRefExpr::VK_GOTPCREL:
    return ELF::R_AMDGPU_GOTPCREL;
  case MCSymbolRefExpr::VK_AMDGPU_GOTPCREL32_LO:
    return ELF::R_AMDGPU_GOTPCREL32_LO;
  case MCSymbolRefExpr::VK_AMDGPU_GOTPCREL32_HI:
    return ELF::R_AMDGPU_GOTPCREL32_HI;
  case MCSymbolRefExpr::VK_AMDGPU_REL32_LO:
    return ELF::R_AMDGPU_REL32_LO;
  case MCSymbolRefExpr::VK_AMDGPU_REL32_HI:
    return ELF::R_AMDGPU_REL32_HI;
  }

  switch (Fixup.getKind()) {
  default: break;
  case FK_PCRel_4:
    return ELF::R_AMDGPU_REL32;
  case FK_Data_4:
  case FK_SecRel_4:
    return ELF::R_AMDGPU_ABS32;
  case FK_Data_8:
    return ELF::R_AMDGPU_ABS64;
  }

  llvm_unreachable("unhandled relocation type");
}

MCObjectWriter *llvm::createAMDGPUELFObjectWriter(bool Is64Bit,
                                                  bool HasRelocationAddend,
                                                  raw_pwrite_stream &OS) {
  MCELFObjectTargetWriter *MOTW =
      new AMDGPUELFObjectWriter(Is64Bit, HasRelocationAddend);
  return createELFObjectWriter(MOTW, OS, true);
}
