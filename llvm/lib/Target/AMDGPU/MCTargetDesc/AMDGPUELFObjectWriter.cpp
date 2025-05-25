//===- AMDGPUELFObjectWriter.cpp - AMDGPU ELF Writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUFixupKinds.h"
#include "AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

namespace {

class AMDGPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  AMDGPUELFObjectWriter(bool Is64Bit, uint8_t OSABI, bool HasRelocationAddend);

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};


} // end anonymous namespace

AMDGPUELFObjectWriter::AMDGPUELFObjectWriter(bool Is64Bit, uint8_t OSABI,
                                             bool HasRelocationAddend)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_AMDGPU,
                              HasRelocationAddend) {}

unsigned AMDGPUELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
  if (const auto *SymA = Target.getAddSym()) {
    // SCRATCH_RSRC_DWORD[01] is a special global variable that represents
    // the scratch buffer.
    if (SymA->getName() == "SCRATCH_RSRC_DWORD0" ||
        SymA->getName() == "SCRATCH_RSRC_DWORD1")
      return ELF::R_AMDGPU_ABS32_LO;
  }

  switch (AMDGPUMCExpr::Specifier(Target.getSpecifier())) {
  default:
    break;
  case AMDGPUMCExpr::S_GOTPCREL:
    return ELF::R_AMDGPU_GOTPCREL;
  case AMDGPUMCExpr::S_GOTPCREL32_LO:
    return ELF::R_AMDGPU_GOTPCREL32_LO;
  case AMDGPUMCExpr::S_GOTPCREL32_HI:
    return ELF::R_AMDGPU_GOTPCREL32_HI;
  case AMDGPUMCExpr::S_REL32_LO:
    return ELF::R_AMDGPU_REL32_LO;
  case AMDGPUMCExpr::S_REL32_HI:
    return ELF::R_AMDGPU_REL32_HI;
  case AMDGPUMCExpr::S_REL64:
    return ELF::R_AMDGPU_REL64;
  case AMDGPUMCExpr::S_ABS32_LO:
    return ELF::R_AMDGPU_ABS32_LO;
  case AMDGPUMCExpr::S_ABS32_HI:
    return ELF::R_AMDGPU_ABS32_HI;
  }

  MCFixupKind Kind = Fixup.getKind();
  switch (Kind) {
  default: break;
  case FK_PCRel_4:
    return ELF::R_AMDGPU_REL32;
  case FK_Data_4:
  case FK_SecRel_4:
    return IsPCRel ? ELF::R_AMDGPU_REL32 : ELF::R_AMDGPU_ABS32;
  case FK_Data_8:
    return IsPCRel ? ELF::R_AMDGPU_REL64 : ELF::R_AMDGPU_ABS64;
  }

  if (Fixup.getTargetKind() == AMDGPU::fixup_si_sopp_br) {
    const auto *SymA = Target.getAddSym();
    assert(SymA);

    if (SymA->isUndefined()) {
      reportError(Fixup.getLoc(),
                  Twine("undefined label '") + SymA->getName() + "'");
      return ELF::R_AMDGPU_NONE;
    }
    return ELF::R_AMDGPU_REL16;
  }

  llvm_unreachable("unhandled relocation type");
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createAMDGPUELFObjectWriter(bool Is64Bit, uint8_t OSABI,
                                  bool HasRelocationAddend) {
  return std::make_unique<AMDGPUELFObjectWriter>(Is64Bit, OSABI,
                                                 HasRelocationAddend);
}
