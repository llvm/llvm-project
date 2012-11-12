//===-- X86MachObjectWriter.cpp - X86 Mach-O Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCValue.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Object/MachOFormat.h"

using namespace llvm;
using namespace llvm::object;

namespace {
class X86MachObjectWriter : public MCMachObjectTargetWriter {
  bool RecordScatteredRelocation(MachObjectWriter *Writer,
                                 const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFragment *Fragment,
                                 const MCFixup &Fixup,
                                 MCValue Target,
                                 unsigned Log2Size,
                                 uint64_t &FixedValue);
  void RecordTLVPRelocation(MachObjectWriter *Writer,
                            const MCAssembler &Asm,
                            const MCAsmLayout &Layout,
                            const MCFragment *Fragment,
                            const MCFixup &Fixup,
                            MCValue Target,
                            uint64_t &FixedValue);

  void RecordX86Relocation(MachObjectWriter *Writer,
                              const MCAssembler &Asm,
                              const MCAsmLayout &Layout,
                              const MCFragment *Fragment,
                              const MCFixup &Fixup,
                              MCValue Target,
                              uint64_t &FixedValue);
  void RecordX86_64Relocation(MachObjectWriter *Writer,
                              const MCAssembler &Asm,
                              const MCAsmLayout &Layout,
                              const MCFragment *Fragment,
                              const MCFixup &Fixup,
                              MCValue Target,
                              uint64_t &FixedValue);
public:
  X86MachObjectWriter(bool Is64Bit, uint32_t CPUType,
                      uint32_t CPUSubtype)
    : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype,
                               /*UseAggressiveSymbolFolding=*/Is64Bit) {}

  void RecordRelocation(MachObjectWriter *Writer,
                        const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) {
    if (Writer->is64Bit())
      RecordX86_64Relocation(Writer, Asm, Layout, Fragment, Fixup, Target,
                             FixedValue);
    else
      RecordX86Relocation(Writer, Asm, Layout, Fragment, Fixup, Target,
                          FixedValue);
  }
};
}

static bool isFixupKindRIPRel(unsigned Kind) {
  return Kind == X86::reloc_riprel_4byte ||
    Kind == X86::reloc_riprel_4byte_movq_load;
}

static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_PCRel_1:
  case FK_Data_1: return 0;
  case FK_PCRel_2:
  case FK_Data_2: return 1;
  case FK_PCRel_4:
    // FIXME: Remove these!!!
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case X86::reloc_signed_4byte:
  case FK_Data_4: return 2;
  case FK_Data_8: return 3;
  }
}

void X86MachObjectWriter::RecordX86_64Relocation(MachObjectWriter *Writer,
                                                 const MCAssembler &Asm,
                                                 const MCAsmLayout &Layout,
                                                 const MCFragment *Fragment,
                                                 const MCFixup &Fixup,
                                                 MCValue Target,
                                                 uint64_t &FixedValue) {
  unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, Fixup.getKind());
  unsigned IsRIPRel = isFixupKindRIPRel(Fixup.getKind());
  unsigned Log2Size = getFixupKindLog2Size(Fixup.getKind());

  // See <reloc.h>.
  uint32_t FixupOffset =
    Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
  uint32_t FixupAddress =
    Writer->getFragmentAddress(Fragment, Layout) + Fixup.getOffset();
  int64_t Value = 0;
  unsigned Index = 0;
  unsigned IsExtern = 0;
  unsigned Type = 0;

  Value = Target.getConstant();

  if (IsPCRel) {
    // Compensate for the relocation offset, Darwin x86_64 relocations only have
    // the addend and appear to have attempted to define it to be the actual
    // expression addend without the PCrel bias. However, instructions with data
    // following the relocation are not accommodated for (see comment below
    // regarding SIGNED{1,2,4}), so it isn't exactly that either.
    Value += 1LL << Log2Size;
  }

  if (Target.isAbsolute()) { // constant
    // SymbolNum of 0 indicates the absolute section.
    Type = macho::RIT_X86_64_Unsigned;
    Index = 0;

    // FIXME: I believe this is broken, I don't think the linker can understand
    // it. I think it would require a local relocation, but I'm not sure if that
    // would work either. The official way to get an absolute PCrel relocation
    // is to use an absolute symbol (which we don't support yet).
    if (IsPCRel) {
      IsExtern = 1;
      Type = macho::RIT_X86_64_Branch;
    }
  } else if (Target.getSymB()) { // A - B + constant
    const MCSymbol *A = &Target.getSymA()->getSymbol();
    MCSymbolData &A_SD = Asm.getSymbolData(*A);
    const MCSymbolData *A_Base = Asm.getAtom(&A_SD);

    const MCSymbol *B = &Target.getSymB()->getSymbol();
    MCSymbolData &B_SD = Asm.getSymbolData(*B);
    const MCSymbolData *B_Base = Asm.getAtom(&B_SD);

    // Neither symbol can be modified.
    if (Target.getSymA()->getKind() != MCSymbolRefExpr::VK_None ||
        Target.getSymB()->getKind() != MCSymbolRefExpr::VK_None)
      report_fatal_error("unsupported relocation of modified symbol");

    // We don't support PCrel relocations of differences. Darwin 'as' doesn't
    // implement most of these correctly.
    if (IsPCRel)
      report_fatal_error("unsupported pc-relative relocation of difference");

    // The support for the situation where one or both of the symbols would
    // require a local relocation is handled just like if the symbols were
    // external.  This is certainly used in the case of debug sections where the
    // section has only temporary symbols and thus the symbols don't have base
    // symbols.  This is encoded using the section ordinal and non-extern
    // relocation entries.

    // Darwin 'as' doesn't emit correct relocations for this (it ends up with a
    // single SIGNED relocation); reject it for now.  Except the case where both
    // symbols don't have a base, equal but both NULL.
    if (A_Base == B_Base && A_Base)
      report_fatal_error("unsupported relocation with identical base");

    Value += Writer->getSymbolAddress(&A_SD, Layout) -
      (A_Base == NULL ? 0 : Writer->getSymbolAddress(A_Base, Layout));
    Value -= Writer->getSymbolAddress(&B_SD, Layout) -
      (B_Base == NULL ? 0 : Writer->getSymbolAddress(B_Base, Layout));

    if (A_Base) {
      Index = A_Base->getIndex();
      IsExtern = 1;
    }
    else {
      Index = A_SD.getFragment()->getParent()->getOrdinal() + 1;
      IsExtern = 0;
    }
    Type = macho::RIT_X86_64_Unsigned;

    macho::RelocationEntry MRE;
    MRE.Word0 = FixupOffset;
    MRE.Word1 = ((Index     <<  0) |
                 (IsPCRel   << 24) |
                 (Log2Size  << 25) |
                 (IsExtern  << 27) |
                 (Type      << 28));
    Writer->addRelocation(Fragment->getParent(), MRE);

    if (B_Base) {
      Index = B_Base->getIndex();
      IsExtern = 1;
    }
    else {
      Index = B_SD.getFragment()->getParent()->getOrdinal() + 1;
      IsExtern = 0;
    }
    Type = macho::RIT_X86_64_Subtractor;
  } else {
    const MCSymbol *Symbol = &Target.getSymA()->getSymbol();
    MCSymbolData &SD = Asm.getSymbolData(*Symbol);
    const MCSymbolData *Base = Asm.getAtom(&SD);

    // Relocations inside debug sections always use local relocations when
    // possible. This seems to be done because the debugger doesn't fully
    // understand x86_64 relocation entries, and expects to find values that
    // have already been fixed up.
    if (Symbol->isInSection()) {
      const MCSectionMachO &Section = static_cast<const MCSectionMachO&>(
        Fragment->getParent()->getSection());
      if (Section.hasAttribute(MCSectionMachO::S_ATTR_DEBUG))
        Base = 0;
    }

    // x86_64 almost always uses external relocations, except when there is no
    // symbol to use as a base address (a local symbol with no preceding
    // non-local symbol).
    if (Base) {
      Index = Base->getIndex();
      IsExtern = 1;

      // Add the local offset, if needed.
      if (Base != &SD)
        Value += Layout.getSymbolOffset(&SD) - Layout.getSymbolOffset(Base);
    } else if (Symbol->isInSection() && !Symbol->isVariable()) {
      // The index is the section ordinal (1-based).
      Index = SD.getFragment()->getParent()->getOrdinal() + 1;
      IsExtern = 0;
      Value += Writer->getSymbolAddress(&SD, Layout);

      if (IsPCRel)
        Value -= FixupAddress + (1 << Log2Size);
    } else if (Symbol->isVariable()) {
      const MCExpr *Value = Symbol->getVariableValue();
      int64_t Res;
      bool isAbs = Value->EvaluateAsAbsolute(Res, Layout,
                                             Writer->getSectionAddressMap());
      if (isAbs) {
        FixedValue = Res;
        return;
      } else {
        report_fatal_error("unsupported relocation of variable '" +
                           Symbol->getName() + "'");
      }
    } else {
      report_fatal_error("unsupported relocation of undefined symbol '" +
                         Symbol->getName() + "'");
    }

    MCSymbolRefExpr::VariantKind Modifier = Target.getSymA()->getKind();
    if (IsPCRel) {
      if (IsRIPRel) {
        if (Modifier == MCSymbolRefExpr::VK_GOTPCREL) {
          // x86_64 distinguishes movq foo@GOTPCREL so that the linker can
          // rewrite the movq to an leaq at link time if the symbol ends up in
          // the same linkage unit.
          if (unsigned(Fixup.getKind()) == X86::reloc_riprel_4byte_movq_load)
            Type = macho::RIT_X86_64_GOTLoad;
          else
            Type = macho::RIT_X86_64_GOT;
        }  else if (Modifier == MCSymbolRefExpr::VK_TLVP) {
          Type = macho::RIT_X86_64_TLV;
        }  else if (Modifier != MCSymbolRefExpr::VK_None) {
          report_fatal_error("unsupported symbol modifier in relocation");
        } else {
          Type = macho::RIT_X86_64_Signed;

          // The Darwin x86_64 relocation format has a problem where it cannot
          // encode an address (L<foo> + <constant>) which is outside the atom
          // containing L<foo>. Generally, this shouldn't occur but it does
          // happen when we have a RIPrel instruction with data following the
          // relocation entry (e.g., movb $012, L0(%rip)). Even with the PCrel
          // adjustment Darwin x86_64 uses, the offset is still negative and the
          // linker has no way to recognize this.
          //
          // To work around this, Darwin uses several special relocation types
          // to indicate the offsets. However, the specification or
          // implementation of these seems to also be incomplete; they should
          // adjust the addend as well based on the actual encoded instruction
          // (the additional bias), but instead appear to just look at the final
          // offset.
          switch (-(Target.getConstant() + (1LL << Log2Size))) {
          case 1: Type = macho::RIT_X86_64_Signed1; break;
          case 2: Type = macho::RIT_X86_64_Signed2; break;
          case 4: Type = macho::RIT_X86_64_Signed4; break;
          }
        }
      } else {
        if (Modifier != MCSymbolRefExpr::VK_None)
          report_fatal_error("unsupported symbol modifier in branch "
                             "relocation");

        Type = macho::RIT_X86_64_Branch;
      }
    } else {
      if (Modifier == MCSymbolRefExpr::VK_GOT) {
        Type = macho::RIT_X86_64_GOT;
      } else if (Modifier == MCSymbolRefExpr::VK_GOTPCREL) {
        // GOTPCREL is allowed as a modifier on non-PCrel instructions, in which
        // case all we do is set the PCrel bit in the relocation entry; this is
        // used with exception handling, for example. The source is required to
        // include any necessary offset directly.
        Type = macho::RIT_X86_64_GOT;
        IsPCRel = 1;
      } else if (Modifier == MCSymbolRefExpr::VK_TLVP) {
        report_fatal_error("TLVP symbol modifier should have been rip-rel");
      } else if (Modifier != MCSymbolRefExpr::VK_None)
        report_fatal_error("unsupported symbol modifier in relocation");
      else
        Type = macho::RIT_X86_64_Unsigned;
    }
  }

  // x86_64 always writes custom values into the fixups.
  FixedValue = Value;

  // struct relocation_info (8 bytes)
  macho::RelocationEntry MRE;
  MRE.Word0 = FixupOffset;
  MRE.Word1 = ((Index     <<  0) |
               (IsPCRel   << 24) |
               (Log2Size  << 25) |
               (IsExtern  << 27) |
               (Type      << 28));
  Writer->addRelocation(Fragment->getParent(), MRE);
}

bool X86MachObjectWriter::RecordScatteredRelocation(MachObjectWriter *Writer,
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
    report_fatal_error("symbol '" + A->getName() +
                       "' can not be undefined in a subtraction expression");

  uint32_t Value = Writer->getSymbolAddress(A_SD, Layout);
  uint64_t SecAddr = Writer->getSectionAddress(A_SD->getFragment()->getParent());
  FixedValue += SecAddr;
  uint32_t Value2 = 0;

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    MCSymbolData *B_SD = &Asm.getSymbolData(B->getSymbol());

    if (!B_SD->getFragment())
      report_fatal_error("symbol '" + B->getSymbol().getName() +
                         "' can not be undefined in a subtraction expression");

    // Select the appropriate difference relocation type.
    //
    // Note that there is no longer any semantic difference between these two
    // relocation types from the linkers point of view, this is done solely for
    // pedantic compatibility with 'as'.
    Type = A_SD->isExternal() ? (unsigned)macho::RIT_Difference :
      (unsigned)macho::RIT_Generic_LocalDifference;
    Value2 = Writer->getSymbolAddress(B_SD, Layout);
    FixedValue -= Writer->getSectionAddress(B_SD->getFragment()->getParent());
  }

  // Relocations are written out in reverse order, so the PAIR comes first.
  if (Type == macho::RIT_Difference ||
      Type == macho::RIT_Generic_LocalDifference) {
    // If the offset is too large to fit in a scattered relocation,
    // we're hosed. It's an unfortunate limitation of the MachO format.
    if (FixupOffset > 0xffffff) {
      char Buffer[32];
      format("0x%x", FixupOffset).print(Buffer, sizeof(Buffer));
      Asm.getContext().FatalError(Fixup.getLoc(),
                         Twine("Section too large, can't encode "
                                "r_address (") + Buffer +
                         ") into 24 bits of scattered "
                         "relocation entry.");
      llvm_unreachable("fatal error returned?!");
    }

    macho::RelocationEntry MRE;
    MRE.Word0 = ((0         <<  0) |
                 (macho::RIT_Pair  << 24) |
                 (Log2Size  << 28) |
                 (IsPCRel   << 30) |
                 macho::RF_Scattered);
    MRE.Word1 = Value2;
    Writer->addRelocation(Fragment->getParent(), MRE);
  } else {
    // If the offset is more than 24-bits, it won't fit in a scattered
    // relocation offset field, so we fall back to using a non-scattered
    // relocation. This is a bit risky, as if the offset reaches out of
    // the block and the linker is doing scattered loading on this
    // symbol, things can go badly.
    //
    // Required for 'as' compatibility.
    if (FixupOffset > 0xffffff)
      return false;
  }

  macho::RelocationEntry MRE;
  MRE.Word0 = ((FixupOffset <<  0) |
               (Type        << 24) |
               (Log2Size    << 28) |
               (IsPCRel     << 30) |
               macho::RF_Scattered);
  MRE.Word1 = Value;
  Writer->addRelocation(Fragment->getParent(), MRE);
  return true;
}

void X86MachObjectWriter::RecordTLVPRelocation(MachObjectWriter *Writer,
                                               const MCAssembler &Asm,
                                               const MCAsmLayout &Layout,
                                               const MCFragment *Fragment,
                                               const MCFixup &Fixup,
                                               MCValue Target,
                                               uint64_t &FixedValue) {
  assert(Target.getSymA()->getKind() == MCSymbolRefExpr::VK_TLVP &&
         !is64Bit() &&
         "Should only be called with a 32-bit TLVP relocation!");

  unsigned Log2Size = getFixupKindLog2Size(Fixup.getKind());
  uint32_t Value = Layout.getFragmentOffset(Fragment)+Fixup.getOffset();
  unsigned IsPCRel = 0;

  // Get the symbol data.
  MCSymbolData *SD_A = &Asm.getSymbolData(Target.getSymA()->getSymbol());
  unsigned Index = SD_A->getIndex();

  // We're only going to have a second symbol in pic mode and it'll be a
  // subtraction from the picbase. For 32-bit pic the addend is the difference
  // between the picbase and the next address.  For 32-bit static the addend is
  // zero.
  if (Target.getSymB()) {
    // If this is a subtraction then we're pcrel.
    uint32_t FixupAddress =
      Writer->getFragmentAddress(Fragment, Layout) + Fixup.getOffset();
    MCSymbolData *SD_B = &Asm.getSymbolData(Target.getSymB()->getSymbol());
    IsPCRel = 1;
    FixedValue = (FixupAddress - Writer->getSymbolAddress(SD_B, Layout) +
                  Target.getConstant());
    FixedValue += 1ULL << Log2Size;
  } else {
    FixedValue = 0;
  }

  // struct relocation_info (8 bytes)
  macho::RelocationEntry MRE;
  MRE.Word0 = Value;
  MRE.Word1 = ((Index                  <<  0) |
               (IsPCRel                << 24) |
               (Log2Size               << 25) |
               (1                      << 27) | // Extern
               (macho::RIT_Generic_TLV << 28)); // Type
  Writer->addRelocation(Fragment->getParent(), MRE);
}

void X86MachObjectWriter::RecordX86Relocation(MachObjectWriter *Writer,
                                              const MCAssembler &Asm,
                                              const MCAsmLayout &Layout,
                                              const MCFragment *Fragment,
                                              const MCFixup &Fixup,
                                              MCValue Target,
                                              uint64_t &FixedValue) {
  unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, Fixup.getKind());
  unsigned Log2Size = getFixupKindLog2Size(Fixup.getKind());

  // If this is a 32-bit TLVP reloc it's handled a bit differently.
  if (Target.getSymA() &&
      Target.getSymA()->getKind() == MCSymbolRefExpr::VK_TLVP) {
    RecordTLVPRelocation(Writer, Asm, Layout, Fragment, Fixup, Target,
                         FixedValue);
    return;
  }

  // If this is a difference or a defined symbol plus an offset, then we need a
  // scattered relocation entry. Differences always require scattered
  // relocations.
  if (Target.getSymB()) {
    RecordScatteredRelocation(Writer, Asm, Layout, Fragment, Fixup,
                              Target, Log2Size, FixedValue);
    return;
  }

  // Get the symbol data, if any.
  MCSymbolData *SD = 0;
  if (Target.getSymA())
    SD = &Asm.getSymbolData(Target.getSymA()->getSymbol());

  // If this is an internal relocation with an offset, it also needs a scattered
  // relocation entry.
  uint32_t Offset = Target.getConstant();
  if (IsPCRel)
    Offset += 1 << Log2Size;
  // Try to record the scattered relocation if needed. Fall back to non
  // scattered if necessary (see comments in RecordScatteredRelocation()
  // for details).
  if (Offset && SD && !Writer->doesSymbolRequireExternRelocation(SD) &&
      RecordScatteredRelocation(Writer, Asm, Layout, Fragment, Fixup,
                                Target, Log2Size, FixedValue))
    return;

  // See <reloc.h>.
  uint32_t FixupOffset = Layout.getFragmentOffset(Fragment)+Fixup.getOffset();
  unsigned Index = 0;
  unsigned IsExtern = 0;
  unsigned Type = 0;

  if (Target.isAbsolute()) { // constant
    // SymbolNum of 0 indicates the absolute section.
    //
    // FIXME: Currently, these are never generated (see code below). I cannot
    // find a case where they are actually emitted.
    Type = macho::RIT_Vanilla;
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
    if (Writer->doesSymbolRequireExternRelocation(SD)) {
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

    Type = macho::RIT_Vanilla;
  }

  // struct relocation_info (8 bytes)
  macho::RelocationEntry MRE;
  MRE.Word0 = FixupOffset;
  MRE.Word1 = ((Index     <<  0) |
               (IsPCRel   << 24) |
               (Log2Size  << 25) |
               (IsExtern  << 27) |
               (Type      << 28));
  Writer->addRelocation(Fragment->getParent(), MRE);
}

MCObjectWriter *llvm::createX86MachObjectWriter(raw_ostream &OS,
                                                bool Is64Bit,
                                                uint32_t CPUType,
                                                uint32_t CPUSubtype) {
  return createMachObjectWriter(new X86MachObjectWriter(Is64Bit,
                                                        CPUType,
                                                        CPUSubtype),
                                OS, /*IsLittleEndian=*/true);
}
