//===-- X86ELFObjectWriter.cpp - X86 ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class X86ELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    X86ELFObjectWriter(bool IsELF64, uint8_t OSABI, uint16_t EMachine);

    virtual ~X86ELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
  };
}

X86ELFObjectWriter::X86ELFObjectWriter(bool IsELF64, uint8_t OSABI,
                                       uint16_t EMachine)
  : MCELFObjectTargetWriter(IsELF64, OSABI, EMachine,
                            // Only i386 uses Rel instead of RelA.
                            /*HasRelocationAddend*/ EMachine != ELF::EM_386) {}

X86ELFObjectWriter::~X86ELFObjectWriter()
{}

unsigned X86ELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) const {
  // determine the type of the relocation

  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();
  unsigned Type;
  if (getEMachine() == ELF::EM_X86_64) {
    if (IsPCRel) {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      case FK_Data_8: Type = ELF::R_X86_64_PC64; break;
      case FK_Data_4: Type = ELF::R_X86_64_PC32; break;
      case FK_Data_2: Type = ELF::R_X86_64_PC16; break;

      case FK_PCRel_8:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC64;
        break;
      case X86::reloc_signed_4byte:
      case X86::reloc_riprel_4byte_movq_load:
      case X86::reloc_riprel_4byte:
      case FK_PCRel_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_X86_64_PC32;
          break;
        case MCSymbolRefExpr::VK_PLT:
          Type = ELF::R_X86_64_PLT32;
          break;
        case MCSymbolRefExpr::VK_GOTPCREL:
          Type = ELF::R_X86_64_GOTPCREL;
          break;
        case MCSymbolRefExpr::VK_GOTTPOFF:
          Type = ELF::R_X86_64_GOTTPOFF;
        break;
        case MCSymbolRefExpr::VK_TLSGD:
          Type = ELF::R_X86_64_TLSGD;
          break;
        case MCSymbolRefExpr::VK_TLSLD:
          Type = ELF::R_X86_64_TLSLD;
          break;
        }
        break;
      case FK_PCRel_2:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC16;
        break;
      case FK_PCRel_1:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC8;
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case FK_Data_8: Type = ELF::R_X86_64_64; break;
      case X86::reloc_signed_4byte:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_X86_64_32S;
          break;
        case MCSymbolRefExpr::VK_GOT:
          Type = ELF::R_X86_64_GOT32;
          break;
        case MCSymbolRefExpr::VK_GOTPCREL:
          Type = ELF::R_X86_64_GOTPCREL;
          break;
        case MCSymbolRefExpr::VK_TPOFF:
          Type = ELF::R_X86_64_TPOFF32;
          break;
        case MCSymbolRefExpr::VK_DTPOFF:
          Type = ELF::R_X86_64_DTPOFF32;
          break;
        }
        break;
      case FK_Data_4:
        Type = ELF::R_X86_64_32;
        break;
      case FK_Data_2: Type = ELF::R_X86_64_16; break;
      case FK_PCRel_1:
      case FK_Data_1: Type = ELF::R_X86_64_8; break;
      }
    }
  } else if (getEMachine() == ELF::EM_386) {
    if (IsPCRel) {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      case X86::reloc_global_offset_table:
        Type = ELF::R_386_GOTPC;
        break;

      case X86::reloc_signed_4byte:
      case FK_PCRel_4:
      case FK_Data_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_386_PC32;
          break;
        case MCSymbolRefExpr::VK_PLT:
          Type = ELF::R_386_PLT32;
          break;
        }
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      case X86::reloc_global_offset_table:
        Type = ELF::R_386_GOTPC;
        break;

      // FIXME: Should we avoid selecting reloc_signed_4byte in 32 bit mode
      // instead?
      case X86::reloc_signed_4byte:
      case FK_PCRel_4:
      case FK_Data_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_386_32;
          break;
        case MCSymbolRefExpr::VK_GOT:
          Type = ELF::R_386_GOT32;
          break;
        case MCSymbolRefExpr::VK_GOTOFF:
          Type = ELF::R_386_GOTOFF;
          break;
        case MCSymbolRefExpr::VK_TLSGD:
          Type = ELF::R_386_TLS_GD;
          break;
        case MCSymbolRefExpr::VK_TPOFF:
          Type = ELF::R_386_TLS_LE_32;
          break;
        case MCSymbolRefExpr::VK_INDNTPOFF:
          Type = ELF::R_386_TLS_IE;
          break;
        case MCSymbolRefExpr::VK_NTPOFF:
          Type = ELF::R_386_TLS_LE;
          break;
        case MCSymbolRefExpr::VK_GOTNTPOFF:
          Type = ELF::R_386_TLS_GOTIE;
          break;
        case MCSymbolRefExpr::VK_TLSLDM:
          Type = ELF::R_386_TLS_LDM;
          break;
        case MCSymbolRefExpr::VK_DTPOFF:
          Type = ELF::R_386_TLS_LDO_32;
          break;
        case MCSymbolRefExpr::VK_GOTTPOFF:
          Type = ELF::R_386_TLS_IE_32;
          break;
        }
        break;
      case FK_Data_2: Type = ELF::R_386_16; break;
      case FK_PCRel_1:
      case FK_Data_1: Type = ELF::R_386_8; break;
      }
    }
  } else
    llvm_unreachable("Unsupported ELF machine type.");

  return Type;
}

MCObjectWriter *llvm::createX86ELFObjectWriter(raw_ostream &OS,
                                               bool IsELF64,
                                               uint8_t OSABI,
                                               uint16_t EMachine) {
  MCELFObjectTargetWriter *MOTW =
    new X86ELFObjectWriter(IsELF64, OSABI, EMachine);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/true);
}
