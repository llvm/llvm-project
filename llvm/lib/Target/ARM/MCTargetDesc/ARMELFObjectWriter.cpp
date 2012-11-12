//===-- ARMELFObjectWriter.cpp - ARM ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMFixupKinds.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

namespace {
  class ARMELFObjectWriter : public MCELFObjectTargetWriter {
    enum { DefaultEABIVersion = 0x05000000U };
    unsigned GetRelocTypeInner(const MCValue &Target,
                               const MCFixup &Fixup,
                               bool IsPCRel) const;


  public:
    ARMELFObjectWriter(uint8_t OSABI);

    virtual ~ARMELFObjectWriter();

    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual unsigned getEFlags() const;
    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                   const MCValue &Target,
                                   const MCFragment &F,
                                   const MCFixup &Fixup,
                                   bool IsPCRel) const;
  };
}

ARMELFObjectWriter::ARMELFObjectWriter(uint8_t OSABI)
  : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI,
                            ELF::EM_ARM,
                            /*HasRelocationAddend*/ false) {}

ARMELFObjectWriter::~ARMELFObjectWriter() {}

// FIXME: get the real EABI Version from the Triple.
unsigned ARMELFObjectWriter::getEFlags() const {
  return ELF::EF_ARM_EABIMASK & DefaultEABIVersion;
}

// In ARM, _MergedGlobals and other most symbols get emitted directly.
// I.e. not as an offset to a section symbol.
// This code is an approximation of what ARM/gcc does.

STATISTIC(PCRelCount, "Total number of PIC Relocations");
STATISTIC(NonPCRelCount, "Total number of non-PIC relocations");

const MCSymbol *ARMELFObjectWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                   const MCValue &Target,
                                                   const MCFragment &F,
                                                   const MCFixup &Fixup,
                                                   bool IsPCRel) const {
  const MCSymbol &Symbol = Target.getSymA()->getSymbol().AliasedSymbol();
  bool EmitThisSym = false;

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(Symbol.getSection());
  bool InNormalSection = true;
  unsigned RelocType = 0;
  RelocType = GetRelocTypeInner(Target, Fixup, IsPCRel);

  DEBUG(
      const MCSymbolRefExpr::VariantKind Kind = Target.getSymA()->getKind();
      MCSymbolRefExpr::VariantKind Kind2;
      Kind2 = Target.getSymB() ?  Target.getSymB()->getKind() :
        MCSymbolRefExpr::VK_None;
      dbgs() << "considering symbol "
        << Section.getSectionName() << "/"
        << Symbol.getName() << "/"
        << " Rel:" << (unsigned)RelocType
        << " Kind: " << (int)Kind << "/" << (int)Kind2
        << " Tmp:"
        << Symbol.isAbsolute() << "/" << Symbol.isDefined() << "/"
        << Symbol.isVariable() << "/" << Symbol.isTemporary()
        << " Counts:" << PCRelCount << "/" << NonPCRelCount << "\n");

  if (IsPCRel) { ++PCRelCount;
    switch (RelocType) {
    default:
      // Most relocation types are emitted as explicit symbols
      InNormalSection =
        StringSwitch<bool>(Section.getSectionName())
        .Case(".data.rel.ro.local", false)
        .Case(".data.rel", false)
        .Case(".bss", false)
        .Default(true);
      EmitThisSym = true;
      break;
    case ELF::R_ARM_ABS32:
      // But things get strange with R_ARM_ABS32
      // In this case, most things that go in .rodata show up
      // as section relative relocations
      InNormalSection =
        StringSwitch<bool>(Section.getSectionName())
        .Case(".data.rel.ro.local", false)
        .Case(".data.rel", false)
        .Case(".rodata", false)
        .Case(".bss", false)
        .Default(true);
      EmitThisSym = false;
      break;
    }
  } else {
    NonPCRelCount++;
    InNormalSection =
      StringSwitch<bool>(Section.getSectionName())
      .Case(".data.rel.ro.local", false)
      .Case(".rodata", false)
      .Case(".data.rel", false)
      .Case(".bss", false)
      .Default(true);

    switch (RelocType) {
    default: EmitThisSym = true; break;
    case ELF::R_ARM_ABS32: EmitThisSym = false; break;
    }
  }

  if (EmitThisSym)
    return &Symbol;
  if (! Symbol.isTemporary() && InNormalSection) {
    return &Symbol;
  }
  return NULL;
}

// Need to examine the Fixup when determining whether to 
// emit the relocation as an explicit symbol or as a section relative
// offset
unsigned ARMELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) const {
  return GetRelocTypeInner(Target, Fixup, IsPCRel);
}

unsigned ARMELFObjectWriter::GetRelocTypeInner(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const  {
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  unsigned Type = 0;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("Unimplemented");
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_REL32;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        llvm_unreachable("unimplemented");
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      }
      break;
    case ARM::fixup_arm_blx:
    case ARM::fixup_arm_uncondbl:
      switch (Modifier) {
      case MCSymbolRefExpr::VK_ARM_PLT:
        Type = ELF::R_ARM_PLT32;
        break;
      default:
        Type = ELF::R_ARM_CALL;
        break;
      }
      break;
    case ARM::fixup_arm_condbl:
    case ARM::fixup_arm_condbranch:
    case ARM::fixup_arm_uncondbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_t2_condbranch:
    case ARM::fixup_t2_uncondbranch:
      Type = ELF::R_ARM_THM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
    case ARM::fixup_arm_movt_hi16_pcrel:
      Type = ELF::R_ARM_MOVT_PREL;
      break;
    case ARM::fixup_arm_movw_lo16:
    case ARM::fixup_arm_movw_lo16_pcrel:
      Type = ELF::R_ARM_MOVW_PREL_NC;
      break;
    case ARM::fixup_t2_movt_hi16:
    case ARM::fixup_t2_movt_hi16_pcrel:
      Type = ELF::R_ARM_THM_MOVT_PREL;
      break;
    case ARM::fixup_t2_movw_lo16:
    case ARM::fixup_t2_movw_lo16_pcrel:
      Type = ELF::R_ARM_THM_MOVW_PREL_NC;
      break;
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_blx:
      Type = ELF::R_ARM_THM_CALL;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_ARM_GOT:
        Type = ELF::R_ARM_GOT_BREL;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        Type = ELF::R_ARM_TLS_GD32;
        break;
      case MCSymbolRefExpr::VK_ARM_TPOFF:
        Type = ELF::R_ARM_TLS_LE32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOTOFF:
        Type = ELF::R_ARM_GOTOFF32;
        break;
      case MCSymbolRefExpr::VK_ARM_TARGET1:
        Type = ELF::R_ARM_TARGET1;
        break;
      case MCSymbolRefExpr::VK_ARM_TARGET2:
        Type = ELF::R_ARM_TARGET2;
        break;
      } 
      break;
    case ARM::fixup_arm_ldst_pcrel_12:
    case ARM::fixup_arm_pcrel_10:
    case ARM::fixup_arm_adr_pcrel_12:
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_cb:
    case ARM::fixup_arm_thumb_cp:
    case ARM::fixup_arm_thumb_br:
      llvm_unreachable("Unimplemented");
    case ARM::fixup_arm_condbranch:
    case ARM::fixup_arm_uncondbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
      Type = ELF::R_ARM_MOVT_ABS;
      break;
    case ARM::fixup_arm_movw_lo16:
      Type = ELF::R_ARM_MOVW_ABS_NC;
      break;
    case ARM::fixup_t2_movt_hi16:
      Type = ELF::R_ARM_THM_MOVT_ABS;
      break;
    case ARM::fixup_t2_movw_lo16:
      Type = ELF::R_ARM_THM_MOVW_ABS_NC;
      break;
    }
  }

  return Type;
}

MCObjectWriter *llvm::createARMELFObjectWriter(raw_ostream &OS,
                                               uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new ARMELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/true);
}
