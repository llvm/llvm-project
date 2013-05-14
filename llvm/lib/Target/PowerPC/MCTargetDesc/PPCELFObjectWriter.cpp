//===-- PPCELFObjectWriter.cpp - PPC ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCFixupKinds.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class PPCELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI);

    virtual ~PPCELFObjectWriter();
  protected:
    virtual unsigned getRelocTypeInner(const MCValue &Target,
                                       const MCFixup &Fixup,
                                       bool IsPCRel) const;
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual const MCSymbol *undefinedExplicitRelSym(const MCValue &Target,
                                                    const MCFixup &Fixup,
                                                    bool IsPCRel) const;
    virtual void adjustFixupOffset(const MCFixup &Fixup, uint64_t &RelocOffset);

    virtual void sortRelocs(const MCAssembler &Asm,
                            std::vector<ELFRelocationEntry> &Relocs);
  };

  class PPCELFRelocationEntry : public ELFRelocationEntry {
  public:
    PPCELFRelocationEntry(const ELFRelocationEntry &RE);
    bool operator<(const PPCELFRelocationEntry &RE) const {
      return (RE.r_offset < r_offset ||
              (RE.r_offset == r_offset && RE.Type > Type));
    }
  };
}

PPCELFRelocationEntry::PPCELFRelocationEntry(const ELFRelocationEntry &RE)
  : ELFRelocationEntry(RE.r_offset, RE.Index, RE.Type, RE.Symbol,
                       RE.r_addend, *RE.Fixup) {}

PPCELFObjectWriter::PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI)
  : MCELFObjectTargetWriter(Is64Bit, OSABI,
                            Is64Bit ?  ELF::EM_PPC64 : ELF::EM_PPC,
                            /*HasRelocationAddend*/ true) {}

PPCELFObjectWriter::~PPCELFObjectWriter() {
}

unsigned PPCELFObjectWriter::getRelocTypeInner(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const
{
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  // determine the type of the relocation
  unsigned Type;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_REL24;
      break;
    case PPC::fixup_ppc_brcond14:
      Type = ELF::R_PPC_REL14;
      break;
    case FK_Data_4:
    case FK_PCRel_4:
      Type = ELF::R_PPC_REL32;
      break;
    case FK_Data_8:
    case FK_PCRel_8:
      Type = ELF::R_PPC64_REL64;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_ADDR24;
      break;
    case PPC::fixup_ppc_brcond14:
      Type = ELF::R_PPC_ADDR14; // XXX: or BRNTAKEN?_
      break;
    case PPC::fixup_ppc_ha16:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TPREL16_HA:
        Type = ELF::R_PPC_TPREL16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_DTPREL16_HA:
        Type = ELF::R_PPC64_DTPREL16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GAS_HA16:
      case MCSymbolRefExpr::VK_PPC_DARWIN_HA16:
        Type = ELF::R_PPC_ADDR16_HA;
	break;
      case MCSymbolRefExpr::VK_PPC_TOC16_HA:
        Type = ELF::R_PPC64_TOC16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TPREL16_HA:
        Type = ELF::R_PPC64_GOT_TPREL16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSGD16_HA:
        Type = ELF::R_PPC64_GOT_TLSGD16_HA;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSLD16_HA:
        Type = ELF::R_PPC64_GOT_TLSLD16_HA;
        break;
      }
      break;
    case PPC::fixup_ppc_lo16:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TPREL16_LO:
        Type = ELF::R_PPC_TPREL16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_DTPREL16_LO:
        Type = ELF::R_PPC64_DTPREL16_LO;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC_ADDR16;
        break;
      case MCSymbolRefExpr::VK_PPC_GAS_LO16:
      case MCSymbolRefExpr::VK_PPC_DARWIN_LO16:
        Type = ELF::R_PPC_ADDR16_LO;
	break;
      case MCSymbolRefExpr::VK_PPC_TOC_ENTRY:
        Type = ELF::R_PPC64_TOC16;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC16_LO:
        Type = ELF::R_PPC64_TOC16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSGD16_LO:
        Type = ELF::R_PPC64_GOT_TLSGD16_LO;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TLSLD16_LO:
        Type = ELF::R_PPC64_GOT_TLSLD16_LO;
        break;
      }
      break;
    case PPC::fixup_ppc_lo16_ds:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC64_ADDR16_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_GAS_LO16:
      case MCSymbolRefExpr::VK_PPC_DARWIN_LO16:
        Type = ELF::R_PPC64_ADDR16_LO_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_TOC_ENTRY:
        Type = ELF::R_PPC64_TOC16_DS;
	break;
      case MCSymbolRefExpr::VK_PPC_TOC16_LO:
        Type = ELF::R_PPC64_TOC16_LO_DS;
        break;
      case MCSymbolRefExpr::VK_PPC_GOT_TPREL16_LO:
        Type = ELF::R_PPC64_GOT_TPREL16_LO_DS;
        break;
      }
      break;
    case PPC::fixup_ppc_tlsreg:
      Type = ELF::R_PPC64_TLS;
      break;
    case PPC::fixup_ppc_nofixup:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TLSGD:
        Type = ELF::R_PPC64_TLSGD;
        break;
      case MCSymbolRefExpr::VK_PPC_TLSLD:
        Type = ELF::R_PPC64_TLSLD;
        break;
      }
      break;
    case FK_Data_8:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_PPC_TOC:
        Type = ELF::R_PPC64_TOC;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_PPC64_ADDR64;
	break;
      }
      break;
    case FK_Data_4:
      Type = ELF::R_PPC_ADDR32;
      break;
    case FK_Data_2:
      Type = ELF::R_PPC_ADDR16;
      break;
    }
  }
  return Type;
}

unsigned PPCELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) const {
  return getRelocTypeInner(Target, Fixup, IsPCRel);
}

const MCSymbol *PPCELFObjectWriter::undefinedExplicitRelSym(const MCValue &Target,
                                                            const MCFixup &Fixup,
                                                            bool IsPCRel) const {
  assert(Target.getSymA() && "SymA cannot be 0");
  const MCSymbol &Symbol = Target.getSymA()->getSymbol().AliasedSymbol();

  unsigned RelocType = getRelocTypeInner(Target, Fixup, IsPCRel);

  // The .odp creation emits a relocation against the symbol ".TOC." which
  // create a R_PPC64_TOC relocation. However the relocation symbol name
  // in final object creation should be NULL, since the symbol does not
  // really exist, it is just the reference to TOC base for the current
  // object file.
  bool EmitThisSym = RelocType != ELF::R_PPC64_TOC;

  if (EmitThisSym && !Symbol.isTemporary())
    return &Symbol;
  return NULL;
}

void PPCELFObjectWriter::
adjustFixupOffset(const MCFixup &Fixup, uint64_t &RelocOffset) {
  switch ((unsigned)Fixup.getKind()) {
    case PPC::fixup_ppc_ha16:
    case PPC::fixup_ppc_lo16:
    case PPC::fixup_ppc_lo16_ds:
      RelocOffset += 2;
      break;
    default:
      break;
  }
}

// The standard sorter only sorts on the r_offset field, but PowerPC can
// have multiple relocations at the same offset.  Sort secondarily on the
// relocation type to avoid nondeterminism.
void PPCELFObjectWriter::sortRelocs(const MCAssembler &Asm,
                                    std::vector<ELFRelocationEntry> &Relocs) {

  // Copy to a temporary vector of relocation entries having a different
  // sort function.
  std::vector<PPCELFRelocationEntry> TmpRelocs;
  
  for (std::vector<ELFRelocationEntry>::iterator R = Relocs.begin();
       R != Relocs.end(); ++R) {
    TmpRelocs.push_back(PPCELFRelocationEntry(*R));
  }

  // Sort in place by ascending r_offset and descending r_type.
  array_pod_sort(TmpRelocs.begin(), TmpRelocs.end());

  // Copy back to the original vector.
  unsigned I = 0;
  for (std::vector<PPCELFRelocationEntry>::iterator R = TmpRelocs.begin();
       R != TmpRelocs.end(); ++R, ++I) {
    Relocs[I] = ELFRelocationEntry(R->r_offset, R->Index, R->Type,
                                   R->Symbol, R->r_addend, *R->Fixup);
  }
}


MCObjectWriter *llvm::createPPCELFObjectWriter(raw_ostream &OS,
                                               bool Is64Bit,
                                               uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new PPCELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/false);
}
