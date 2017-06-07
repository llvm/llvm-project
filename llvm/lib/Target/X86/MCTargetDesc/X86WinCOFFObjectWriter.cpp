//===-- X86WinCOFFObjectWriter.cpp - X86 Win COFF Writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

class X86WinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  X86WinCOFFObjectWriter(bool Is64Bit);
  ~X86WinCOFFObjectWriter() override = default;

  unsigned getRelocType(const MCValue &Target, const MCFixup &Fixup,
                        bool IsCrossSection,
                        const MCAsmBackend &MAB) const override;
};

} // end anonymous namespace

X86WinCOFFObjectWriter::X86WinCOFFObjectWriter(bool Is64Bit)
    : MCWinCOFFObjectTargetWriter(Is64Bit ? COFF::IMAGE_FILE_MACHINE_AMD64
                                          : COFF::IMAGE_FILE_MACHINE_I386) {}

unsigned X86WinCOFFObjectWriter::getRelocType(const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsCrossSection,
                                              const MCAsmBackend &MAB) const {
  unsigned FixupKind = IsCrossSection ? FK_PCRel_4 : Fixup.getKind();

  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  if (getMachine() == COFF::IMAGE_FILE_MACHINE_AMD64) {
    switch (FixupKind) {
    case FK_PCRel_4:
    case X86::reloc_riprel_4byte:
    case X86::reloc_riprel_4byte_movq_load:
    case X86::reloc_riprel_4byte_relax:
    case X86::reloc_riprel_4byte_relax_rex:
      return COFF::IMAGE_REL_AMD64_REL32;
    case FK_Data_4:
    case X86::reloc_signed_4byte:
    case X86::reloc_signed_4byte_relax:
      if (Modifier == MCSymbolRefExpr::VK_COFF_IMGREL32)
        return COFF::IMAGE_REL_AMD64_ADDR32NB;
      if (Modifier == MCSymbolRefExpr::VK_SECREL)
        return COFF::IMAGE_REL_AMD64_SECREL;
      return COFF::IMAGE_REL_AMD64_ADDR32;
    case FK_Data_8:
      return COFF::IMAGE_REL_AMD64_ADDR64;
    case FK_SecRel_2:
      return COFF::IMAGE_REL_AMD64_SECTION;
    case FK_SecRel_4:
      return COFF::IMAGE_REL_AMD64_SECREL;
    default:
      llvm_unreachable("unsupported relocation type");
    }
  } else if (getMachine() == COFF::IMAGE_FILE_MACHINE_I386) {
    switch (FixupKind) {
    case FK_PCRel_4:
    case X86::reloc_riprel_4byte:
    case X86::reloc_riprel_4byte_movq_load:
      return COFF::IMAGE_REL_I386_REL32;
    case FK_Data_4:
    case X86::reloc_signed_4byte:
    case X86::reloc_signed_4byte_relax:
      if (Modifier == MCSymbolRefExpr::VK_COFF_IMGREL32)
        return COFF::IMAGE_REL_I386_DIR32NB;
      if (Modifier == MCSymbolRefExpr::VK_SECREL)
        return COFF::IMAGE_REL_AMD64_SECREL;
      return COFF::IMAGE_REL_I386_DIR32;
    case FK_SecRel_2:
      return COFF::IMAGE_REL_I386_SECTION;
    case FK_SecRel_4:
      return COFF::IMAGE_REL_I386_SECREL;
    default:
      llvm_unreachable("unsupported relocation type");
    }
  } else
    llvm_unreachable("Unsupported COFF machine type.");
}

MCObjectWriter *llvm::createX86WinCOFFObjectWriter(raw_pwrite_stream &OS,
                                                   bool Is64Bit) {
  MCWinCOFFObjectTargetWriter *MOTW = new X86WinCOFFObjectWriter(Is64Bit);
  return createWinCOFFObjectWriter(MOTW, OS);
}
