//===-- ARMMachObjectWriter.cpp - ARM Mach Object Writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMFixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCMachOSymbolFlags.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;
using namespace llvm::object;

namespace {
class ARMMachObjectWriter : public MCMachObjectTargetWriter {
  void RecordARMScatteredRelocation(MachObjectWriter *Writer,
                                    const MCAssembler &Asm,
                                    const MCAsmLayout &Layout,
                                    const MCFragment *Fragment,
                                    const MCFixup &Fixup,
                                    MCValue Target,
                                    unsigned Log2Size,
                                    uint64_t &FixedValue);
  void RecordARMScatteredHalfRelocation(MachObjectWriter *Writer,
                                        const MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup, MCValue Target,
                                        uint64_t &FixedValue);

  bool requiresExternRelocation(MachObjectWriter *Writer,
                                const MCAssembler &Asm,
                                const MCFragment &Fragment,
                                unsigned RelocType, const MCSymbolData *SD,
                                uint64_t FixedValue);

public:
  ARMMachObjectWriter(bool Is64Bit, uint32_t CPUType,
                      uint32_t CPUSubtype)
    : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype,
                               /*UseAggressiveSymbolFolding=*/true) {}

  void RecordRelocation(MachObjectWriter *Writer,
                        const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue);
};
}

static bool getARMFixupKindMachOInfo(unsigned Kind, unsigned &RelocType,
                              unsigned &Log2Size) {
  RelocType = unsigned(macho::RIT_Vanilla);
  Log2Size = ~0U;

  switch (Kind) {
  default:
    return false;

  case FK_Data_1:
    Log2Size = llvm::Log2_32(1);
    return true;
  case FK_Data_2:
    Log2Size = llvm::Log2_32(2);
    return true;
  case FK_Data_4:
    Log2Size = llvm::Log2_32(4);
    return true;
  case FK_Data_8:
    Log2Size = llvm::Log2_32(8);
    return true;

    // Handle 24-bit branch kinds.
  case ARM::fixup_arm_ldst_pcrel_12:
  case ARM::fixup_arm_pcrel_10:
  case ARM::fixup_arm_adr_pcrel_12:
  case ARM::fixup_arm_condbranch:
  case ARM::fixup_arm_uncondbranch:
  case ARM::fixup_arm_uncondbl:
  case ARM::fixup_arm_condbl:
  case ARM::fixup_arm_blx:
    RelocType = unsigned(macho::RIT_ARM_Branch24Bit);
    // Report as 'long', even though that is not quite accurate.
    Log2Size = llvm::Log2_32(4);
    return true;

    // Handle Thumb branches.
  case ARM::fixup_arm_thumb_br:
    RelocType = unsigned(macho::RIT_ARM_ThumbBranch22Bit);
    Log2Size = llvm::Log2_32(2);
    return true;

  case ARM::fixup_t2_uncondbranch:
  case ARM::fixup_arm_thumb_bl:
  case ARM::fixup_arm_thumb_blx:
    RelocType = unsigned(macho::RIT_ARM_ThumbBranch22Bit);
    Log2Size = llvm::Log2_32(4);
    return true;

  // For movw/movt r_type relocations they always have a pair following them and
  // the r_length bits are used differently.  The encoding of the r_length is as
  // follows:
  //   low bit of r_length:
  //      0 - :lower16: for movw instructions
  //      1 - :upper16: for movt instructions
  //   high bit of r_length:
  //      0 - arm instructions
  //      1 - thumb instructions
  case ARM::fixup_arm_movt_hi16:
  case ARM::fixup_arm_movt_hi16_pcrel:
    RelocType = unsigned(macho::RIT_ARM_Half);
    Log2Size = 1;
    return true;
  case ARM::fixup_t2_movt_hi16:
  case ARM::fixup_t2_movt_hi16_pcrel:
    RelocType = unsigned(macho::RIT_ARM_Half);
    Log2Size = 3;
    return true;

  case ARM::fixup_arm_movw_lo16:
  case ARM::fixup_arm_movw_lo16_pcrel:
    RelocType = unsigned(macho::RIT_ARM_Half);
    Log2Size = 0;
    return true;
  case ARM::fixup_t2_movw_lo16:
  case ARM::fixup_t2_movw_lo16_pcrel:
    RelocType = unsigned(macho::RIT_ARM_Half);
    Log2Size = 2;
    return true;
  }
}

void ARMMachObjectWriter::
RecordARMScatteredHalfRelocation(MachObjectWriter *Writer,
                                 const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFragment *Fragment,
                                 const MCFixup &Fixup,
                                 MCValue Target,
                                 uint64_t &FixedValue) {
  uint32_t FixupOffset = Layout.getFragmentOffset(Fragment)+Fixup.getOffset();
  unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, Fixup.getKind());
  unsigned Type = macho::RIT_ARM_Half;

  // See <reloc.h>.
  const MCSymbol *A = &Target.getSymA()->getSymbol();
  MCSymbolData *A_SD = &Asm.getSymbolData(*A);

  if (!A_SD->getFragment())
    Asm.getContext().FatalError(Fixup.getLoc(),
                       "symbol '" + A->getName() +
                       "' can not be undefined in a subtraction expression");

  uint32_t Value = Writer->getSymbolAddress(A_SD, Layout);
  uint32_t Value2 = 0;
  uint64_t SecAddr =
    Writer->getSectionAddress(A_SD->getFragment()->getParent());
  FixedValue += SecAddr;

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    MCSymbolData *B_SD = &Asm.getSymbolData(B->getSymbol());

    if (!B_SD->getFragment())
      Asm.getContext().FatalError(Fixup.getLoc(),
                         "symbol '" + B->getSymbol().getName() +
                         "' can not be undefined in a subtraction expression");

    // Select the appropriate difference relocation type.
    Type = macho::RIT_ARM_HalfDifference;
    Value2 = Writer->getSymbolAddress(B_SD, Layout);
    FixedValue -= Writer->getSectionAddress(B_SD->getFragment()->getParent());
  }

  // Relocations are written out in reverse order, so the PAIR comes first.
  // ARM_RELOC_HALF and ARM_RELOC_HALF_SECTDIFF abuse the r_length field:
  //
  // For these two r_type relocations they always have a pair following them and
  // the r_length bits are used differently.  The encoding of the r_length is as
  // follows:
  //   low bit of r_length:
  //      0 - :lower16: for movw instructions
  //      1 - :upper16: for movt instructions
  //   high bit of r_length:
  //      0 - arm instructions
  //      1 - thumb instructions
  // the other half of the relocated expression is in the following pair
  // relocation entry in the low 16 bits of r_address field.
  unsigned ThumbBit = 0;
  unsigned MovtBit = 0;
  switch ((unsigned)Fixup.getKind()) {
  default: break;
  case ARM::fixup_arm_movt_hi16:
  case ARM::fixup_arm_movt_hi16_pcrel:
    MovtBit = 1;
    // The thumb bit shouldn't be set in the 'other-half' bit of the
    // relocation, but it will be set in FixedValue if the base symbol
    // is a thumb function. Clear it out here.
    if (A_SD->getFlags() & SF_ThumbFunc)
      FixedValue &= 0xfffffffe;
    break;
  case ARM::fixup_t2_movt_hi16:
  case ARM::fixup_t2_movt_hi16_pcrel:
    if (A_SD->getFlags() & SF_ThumbFunc)
      FixedValue &= 0xfffffffe;
    MovtBit = 1;
    // Fallthrough
  case ARM::fixup_t2_movw_lo16:
  case ARM::fixup_t2_movw_lo16_pcrel:
    ThumbBit = 1;
    break;
  }

  if (Type == macho::RIT_ARM_HalfDifference) {
    uint32_t OtherHalf = MovtBit
      ? (FixedValue & 0xffff) : ((FixedValue & 0xffff0000) >> 16);

    macho::RelocationEntry MRE;
    MRE.Word0 = ((OtherHalf       <<  0) |
                 (macho::RIT_Pair << 24) |
                 (MovtBit         << 28) |
                 (ThumbBit        << 29) |
                 (IsPCRel         << 30) |
                 macho::RF_Scattered);
    MRE.Word1 = Value2;
    Writer->addRelocation(Fragment->getParent(), MRE);
  }

  macho::RelocationEntry MRE;
  MRE.Word0 = ((FixupOffset <<  0) |
               (Type        << 24) |
               (MovtBit     << 28) |
               (ThumbBit    << 29) |
               (IsPCRel     << 30) |
               macho::RF_Scattered);
  MRE.Word1 = Value;
  Writer->addRelocation(Fragment->getParent(), MRE);
}

void ARMMachObjectWriter::RecordARMScatteredRelocation(MachObjectWriter *Writer,
                                                    const MCAssembler &Asm,
                                                    const MCAsmLayout &Layout,
                                                    const MCFragment *Fragment,
                                                    const MCFixup &Fixup,
                                                    MCValue Target,
                                                    unsigned Log2Size,
                                                    uint64_t &FixedValue) {
  uint32_t FixupOffset = Layout.getFragmentOffset(Fragment)+Fixup.getOffset();
  unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, Fixup.getKind());
  unsigned Type = macho::RIT_Vanilla;

  // See <reloc.h>.
  const MCSymbol *A = &Target.getSymA()->getSymbol();
  MCSymbolData *A_SD = &Asm.getSymbolData(*A);

  if (!A_SD->getFragment())
    Asm.getContext().FatalError(Fixup.getLoc(),
                       "symbol '" + A->getName() +
                       "' can not be undefined in a subtraction expression");

  uint32_t Value = Writer->getSymbolAddress(A_SD, Layout);
  uint64_t SecAddr = Writer->getSectionAddress(A_SD->getFragment()->getParent());
  FixedValue += SecAddr;
  uint32_t Value2 = 0;

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    MCSymbolData *B_SD = &Asm.getSymbolData(B->getSymbol());

    if (!B_SD->getFragment())
      Asm.getContext().FatalError(Fixup.getLoc(),
                         "symbol '" + B->getSymbol().getName() +
                         "' can not be undefined in a subtraction expression");

    // Select the appropriate difference relocation type.
    Type = macho::RIT_Difference;
    Value2 = Writer->getSymbolAddress(B_SD, Layout);
    FixedValue -= Writer->getSectionAddress(B_SD->getFragment()->getParent());
  }

  // Relocations are written out in reverse order, so the PAIR comes first.
  if (Type == macho::RIT_Difference ||
      Type == macho::RIT_Generic_LocalDifference) {
    macho::RelocationEntry MRE;
    MRE.Word0 = ((0         <<  0) |
                 (macho::RIT_Pair  << 24) |
                 (Log2Size  << 28) |
                 (IsPCRel   << 30) |
                 macho::RF_Scattered);
    MRE.Word1 = Value2;
    Writer->addRelocation(Fragment->getParent(), MRE);
  }

  macho::RelocationEntry MRE;
  MRE.Word0 = ((FixupOffset <<  0) |
               (Type        << 24) |
               (Log2Size    << 28) |
               (IsPCRel     << 30) |
               macho::RF_Scattered);
  MRE.Word1 = Value;
  Writer->addRelocation(Fragment->getParent(), MRE);
}

bool ARMMachObjectWriter::requiresExternRelocation(MachObjectWriter *Writer,
                                                   const MCAssembler &Asm,
                                                   const MCFragment &Fragment,
                                                   unsigned RelocType,
                                                   const MCSymbolData *SD,
                                                   uint64_t FixedValue) {
  // Most cases can be identified purely from the symbol.
  if (Writer->doesSymbolRequireExternRelocation(SD))
    return true;
  int64_t Value = (int64_t)FixedValue;  // The displacement is signed.
  int64_t Range;
  switch (RelocType) {
  default:
    return false;
  case macho::RIT_ARM_Branch24Bit:
    // PC pre-adjustment of 8 for these instructions.
    Value -= 8;
    // ARM BL/BLX has a 25-bit offset.
    Range = 0x1ffffff;
    break;
  case macho::RIT_ARM_ThumbBranch22Bit:
    // PC pre-adjustment of 4 for these instructions.
    Value -= 4;
    // Thumb BL/BLX has a 24-bit offset.
    Range = 0xffffff;
  }
  // BL/BLX also use external relocations when an internal relocation
  // would result in the target being out of range. This gives the linker
  // enough information to generate a branch island.
  const MCSectionData &SymSD = Asm.getSectionData(
    SD->getSymbol().getSection());
  Value += Writer->getSectionAddress(&SymSD);
  Value -= Writer->getSectionAddress(Fragment.getParent());
  // If the resultant value would be out of range for an internal relocation,
  // use an external instead.
  if (Value > Range || Value < -(Range + 1))
    return true;
  return false;
}

void ARMMachObjectWriter::RecordRelocation(MachObjectWriter *Writer,
                                           const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
  unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, Fixup.getKind());
  unsigned Log2Size;
  unsigned RelocType = macho::RIT_Vanilla;
  if (!getARMFixupKindMachOInfo(Fixup.getKind(), RelocType, Log2Size))
    // If we failed to get fixup kind info, it's because there's no legal
    // relocation type for the fixup kind. This happens when it's a fixup that's
    // expected to always be resolvable at assembly time and not have any
    // relocations needed.
    Asm.getContext().FatalError(Fixup.getLoc(),
                                "unsupported relocation on symbol");

  // If this is a difference or a defined symbol plus an offset, then we need a
  // scattered relocation entry.  Differences always require scattered
  // relocations.
  if (Target.getSymB()) {
    if (RelocType == macho::RIT_ARM_Half)
      return RecordARMScatteredHalfRelocation(Writer, Asm, Layout, Fragment,
                                              Fixup, Target, FixedValue);
    return RecordARMScatteredRelocation(Writer, Asm, Layout, Fragment, Fixup,
                                        Target, Log2Size, FixedValue);
  }

  // Get the symbol data, if any.
  MCSymbolData *SD = 0;
  if (Target.getSymA())
    SD = &Asm.getSymbolData(Target.getSymA()->getSymbol());

  // FIXME: For other platforms, we need to use scattered relocations for
  // internal relocations with offsets.  If this is an internal relocation with
  // an offset, it also needs a scattered relocation entry.
  //
  // Is this right for ARM?
  uint32_t Offset = Target.getConstant();
  if (IsPCRel && RelocType == macho::RIT_Vanilla)
    Offset += 1 << Log2Size;
  if (Offset && SD && !Writer->doesSymbolRequireExternRelocation(SD))
    return RecordARMScatteredRelocation(Writer, Asm, Layout, Fragment, Fixup,
                                        Target, Log2Size, FixedValue);

  // See <reloc.h>.
  uint32_t FixupOffset = Layout.getFragmentOffset(Fragment)+Fixup.getOffset();
  unsigned Index = 0;
  unsigned IsExtern = 0;
  unsigned Type = 0;

  if (Target.isAbsolute()) { // constant
    // FIXME!
    report_fatal_error("FIXME: relocations to absolute targets "
                       "not yet implemented");
  } else {
    // Resolve constant variables.
    if (SD->getSymbol().isVariable()) {
      int64_t Res;
      if (SD->getSymbol().getVariableValue()->EvaluateAsAbsolute(
            Res, Layout, Writer->getSectionAddressMap())) {
        FixedValue = Res;
        return;
      }
    }

    // Check whether we need an external or internal relocation.
    if (requiresExternRelocation(Writer, Asm, *Fragment, RelocType, SD,
                                 FixedValue)) {
      IsExtern = 1;
      Index = SD->getIndex();

      // For external relocations, make sure to offset the fixup value to
      // compensate for the addend of the symbol address, if it was
      // undefined. This occurs with weak definitions, for example.
      if (!SD->Symbol->isUndefined())
        FixedValue -= Layout.getSymbolOffset(SD);
    } else {
      // The index is the section ordinal (1-based).
      const MCSectionData &SymSD = Asm.getSectionData(
        SD->getSymbol().getSection());
      Index = SymSD.getOrdinal() + 1;
      FixedValue += Writer->getSectionAddress(&SymSD);
    }
    if (IsPCRel)
      FixedValue -= Writer->getSectionAddress(Fragment->getParent());

    // The type is determined by the fixup kind.
    Type = RelocType;
  }

  // struct relocation_info (8 bytes)
  macho::RelocationEntry MRE;
  MRE.Word0 = FixupOffset;
  MRE.Word1 = ((Index     <<  0) |
               (IsPCRel   << 24) |
               (Log2Size  << 25) |
               (IsExtern  << 27) |
               (Type      << 28));

  // Even when it's not a scattered relocation, movw/movt always uses
  // a PAIR relocation.
  if (Type == macho::RIT_ARM_Half) {
    // The other-half value only gets populated for the movt and movw
    // relocation entries.
    uint32_t Value = 0;
    switch ((unsigned)Fixup.getKind()) {
    default: break;
    case ARM::fixup_arm_movw_lo16:
    case ARM::fixup_arm_movw_lo16_pcrel:
    case ARM::fixup_t2_movw_lo16:
    case ARM::fixup_t2_movw_lo16_pcrel:
      Value = (FixedValue >> 16) & 0xffff;
      break;
    case ARM::fixup_arm_movt_hi16:
    case ARM::fixup_arm_movt_hi16_pcrel:
    case ARM::fixup_t2_movt_hi16:
    case ARM::fixup_t2_movt_hi16_pcrel:
      Value = FixedValue & 0xffff;
      break;
    }
    macho::RelocationEntry MREPair;
    MREPair.Word0 = Value;
    MREPair.Word1 = ((0xffffff) |
                     (Log2Size << 25) |
                     (macho::RIT_Pair << 28));

    Writer->addRelocation(Fragment->getParent(), MREPair);
  }

  Writer->addRelocation(Fragment->getParent(), MRE);
}

MCObjectWriter *llvm::createARMMachObjectWriter(raw_ostream &OS,
                                                bool Is64Bit,
                                                uint32_t CPUType,
                                                uint32_t CPUSubtype) {
  return createMachObjectWriter(new ARMMachObjectWriter(Is64Bit,
                                                        CPUType,
                                                        CPUSubtype),
                                OS, /*IsLittleEndian=*/true);
}
