//===-- SparcELFObjectWriter.cpp - Sparc ELF Writer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SparcFixupKinds.h"
#include "MCTargetDesc/SparcMCExpr.h"
#include "MCTargetDesc/SparcMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class SparcELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    SparcELFObjectWriter(bool Is64Bit, bool IsV8Plus, uint8_t OSABI)
        : MCELFObjectTargetWriter(
              Is64Bit, OSABI,
              Is64Bit ? ELF::EM_SPARCV9
                      : (IsV8Plus ? ELF::EM_SPARC32PLUS : ELF::EM_SPARC),
              /*HasRelocationAddend*/ true) {}

    ~SparcELFObjectWriter() override = default;

  protected:
    unsigned getRelocType(const MCFixup &Fixup, const MCValue &Target,
                          bool IsPCRel) const override;

    bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym,
                                 unsigned Type) const override;
  };
}

unsigned SparcELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  switch (Target.getSpecifier()) {
  case ELF::R_SPARC_TLS_GD_HI22:
  case ELF::R_SPARC_TLS_GD_LO10:
  case ELF::R_SPARC_TLS_GD_ADD:
  case ELF::R_SPARC_TLS_LDM_HI22:
  case ELF::R_SPARC_TLS_LDM_LO10:
  case ELF::R_SPARC_TLS_LDM_ADD:
  case ELF::R_SPARC_TLS_LDO_HIX22:
  case ELF::R_SPARC_TLS_LDO_LOX10:
  case ELF::R_SPARC_TLS_LDO_ADD:
  case ELF::R_SPARC_TLS_IE_HI22:
  case ELF::R_SPARC_TLS_IE_LO10:
  case ELF::R_SPARC_TLS_IE_LD:
  case ELF::R_SPARC_TLS_IE_LDX:
  case ELF::R_SPARC_TLS_IE_ADD:
  case ELF::R_SPARC_TLS_LE_HIX22:
  case ELF::R_SPARC_TLS_LE_LOX10:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  default:
    break;
  }

  // Extract the relocation type from the fixup kind, after applying STT_TLS as
  // needed.
  unsigned Kind = Fixup.getTargetKind();
  if (mc::isRelocation(Fixup.getKind()))
    return Kind;

  if (const SparcMCExpr *SExpr = dyn_cast<SparcMCExpr>(Fixup.getValue())) {
    if (SExpr->getSpecifier() == ELF::R_SPARC_DISP32)
      return ELF::R_SPARC_DISP32;
  }

  if (IsPCRel) {
    switch (Kind) {
    default:
      llvm_unreachable("Unimplemented fixup -> relocation");
    case FK_Data_1:                  return ELF::R_SPARC_DISP8;
    case FK_Data_2:                  return ELF::R_SPARC_DISP16;
    case FK_Data_4:                  return ELF::R_SPARC_DISP32;
    case FK_Data_8:                  return ELF::R_SPARC_DISP64;
    case Sparc::fixup_sparc_call30:
      if (getContext().getObjectFileInfo()->isPositionIndependent())
        return ELF::R_SPARC_WPLT30;
      return ELF::R_SPARC_WDISP30;
    }
  }

  // clang-format off
  switch(Fixup.getTargetKind()) {
  default:
    llvm_unreachable("Unimplemented fixup -> relocation");
  case FK_NONE:                  return ELF::R_SPARC_NONE;
  case FK_Data_1:                return ELF::R_SPARC_8;
  case FK_Data_2:                return ((Fixup.getOffset() % 2)
                                         ? ELF::R_SPARC_UA16
                                         : ELF::R_SPARC_16);
  case FK_Data_4:                return ((Fixup.getOffset() % 4)
                                         ? ELF::R_SPARC_UA32
                                         : ELF::R_SPARC_32);
  case FK_Data_8:                return ((Fixup.getOffset() % 8)
                                         ? ELF::R_SPARC_UA64
                                         : ELF::R_SPARC_64);
  case Sparc::fixup_sparc_13:
    if (getContext().getObjectFileInfo()->isPositionIndependent())
      return ELF::R_SPARC_GOT13;
    return ELF::R_SPARC_13;
  }
  // clang-format on

  return ELF::R_SPARC_NONE;
}

bool SparcELFObjectWriter::needsRelocateWithSymbol(const MCValue &,
                                                   const MCSymbol &,
                                                   unsigned Type) const {
  switch (Type) {
    default:
      return false;

    // All relocations that use a GOT need a symbol, not an offset, as
    // the offset of the symbol within the section is irrelevant to
    // where the GOT entry is. Don't need to list all the TLS entries,
    // as they're all marked as requiring a symbol anyways.
    case ELF::R_SPARC_GOT10:
    case ELF::R_SPARC_GOT13:
    case ELF::R_SPARC_GOT22:
    case ELF::R_SPARC_GOTDATA_HIX22:
    case ELF::R_SPARC_GOTDATA_LOX10:
    case ELF::R_SPARC_GOTDATA_OP_HIX22:
    case ELF::R_SPARC_GOTDATA_OP_LOX10:
      return true;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createSparcELFObjectWriter(bool Is64Bit, bool IsV8Plus, uint8_t OSABI) {
  return std::make_unique<SparcELFObjectWriter>(Is64Bit, IsV8Plus, OSABI);
}
