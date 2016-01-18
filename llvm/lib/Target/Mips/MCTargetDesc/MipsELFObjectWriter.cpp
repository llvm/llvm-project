//===-- MipsELFObjectWriter.cpp - Mips ELF Writer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include <list>

using namespace llvm;

namespace {
// A helper structure based on ELFRelocationEntry, used for sorting entries in
// the relocation table.
struct MipsRelocationEntry {
  MipsRelocationEntry(const ELFRelocationEntry &R)
      : R(R), SortOffset(R.Offset), HasMatchingHi(false) {}
  const ELFRelocationEntry R;
  // SortOffset equals R.Offset except for the *HI16 relocations, for which it
  // will be set based on the R.Offset of the matching *LO16 relocation.
  int64_t SortOffset;
  // True when this is a *LO16 relocation chosen as a match for a *HI16
  // relocation.
  bool HasMatchingHi;
};

  class MipsELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                        bool _isN64, bool IsLittleEndian);

    ~MipsELFObjectWriter() override;

    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const override;
    bool needsRelocateWithSymbol(const MCSymbol &Sym,
                                 unsigned Type) const override;
    virtual void sortRelocs(const MCAssembler &Asm,
                            std::vector<ELFRelocationEntry> &Relocs) override;
  };
}

MipsELFObjectWriter::MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                                         bool _isN64, bool IsLittleEndian)
    : MCELFObjectTargetWriter(_is64Bit, OSABI, ELF::EM_MIPS,
                              /*HasRelocationAddend*/ _isN64,
                              /*IsN64*/ _isN64) {}

MipsELFObjectWriter::~MipsELFObjectWriter() {}

unsigned MipsELFObjectWriter::GetRelocType(const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
  // Determine the type of the relocation.
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
  case Mips::fixup_Mips_16:
  case FK_Data_2:
    return IsPCRel ? ELF::R_MIPS_PC16 : ELF::R_MIPS_16;
  case Mips::fixup_Mips_32:
  case FK_Data_4:
    return IsPCRel ? ELF::R_MIPS_PC32 : ELF::R_MIPS_32;
  }

  if (IsPCRel) {
    switch (Kind) {
    case Mips::fixup_Mips_Branch_PCRel:
    case Mips::fixup_Mips_PC16:
      return ELF::R_MIPS_PC16;
    case Mips::fixup_MICROMIPS_PC7_S1:
      return ELF::R_MICROMIPS_PC7_S1;
    case Mips::fixup_MICROMIPS_PC10_S1:
      return ELF::R_MICROMIPS_PC10_S1;
    case Mips::fixup_MICROMIPS_PC16_S1:
      return ELF::R_MICROMIPS_PC16_S1;
    case Mips::fixup_MIPS_PC19_S2:
      return ELF::R_MIPS_PC19_S2;
    case Mips::fixup_MIPS_PC18_S3:
      return ELF::R_MIPS_PC18_S3;
    case Mips::fixup_MIPS_PC21_S2:
      return ELF::R_MIPS_PC21_S2;
    case Mips::fixup_MIPS_PC26_S2:
      return ELF::R_MIPS_PC26_S2;
    case Mips::fixup_MIPS_PCHI16:
      return ELF::R_MIPS_PCHI16;
    case Mips::fixup_MIPS_PCLO16:
      return ELF::R_MIPS_PCLO16;
    }

    llvm_unreachable("invalid PC-relative fixup kind!");
  }

  switch (Kind) {
  case Mips::fixup_Mips_64:
  case FK_Data_8:
    return ELF::R_MIPS_64;
  case FK_GPRel_4:
    if (isN64()) {
      unsigned Type = (unsigned)ELF::R_MIPS_NONE;
      Type = setRType((unsigned)ELF::R_MIPS_GPREL32, Type);
      Type = setRType2((unsigned)ELF::R_MIPS_64, Type);
      Type = setRType3((unsigned)ELF::R_MIPS_NONE, Type);
      return Type;
    }
    return ELF::R_MIPS_GPREL32;
  case Mips::fixup_Mips_GPREL16:
    return ELF::R_MIPS_GPREL16;
  case Mips::fixup_Mips_26:
    return ELF::R_MIPS_26;
  case Mips::fixup_Mips_CALL16:
    return ELF::R_MIPS_CALL16;
  case Mips::fixup_Mips_GOT_Global:
  case Mips::fixup_Mips_GOT_Local:
    return ELF::R_MIPS_GOT16;
  case Mips::fixup_Mips_HI16:
    return ELF::R_MIPS_HI16;
  case Mips::fixup_Mips_LO16:
    return ELF::R_MIPS_LO16;
  case Mips::fixup_Mips_TLSGD:
    return ELF::R_MIPS_TLS_GD;
  case Mips::fixup_Mips_GOTTPREL:
    return ELF::R_MIPS_TLS_GOTTPREL;
  case Mips::fixup_Mips_TPREL_HI:
    return ELF::R_MIPS_TLS_TPREL_HI16;
  case Mips::fixup_Mips_TPREL_LO:
    return ELF::R_MIPS_TLS_TPREL_LO16;
  case Mips::fixup_Mips_TLSLDM:
    return ELF::R_MIPS_TLS_LDM;
  case Mips::fixup_Mips_DTPREL_HI:
    return ELF::R_MIPS_TLS_DTPREL_HI16;
  case Mips::fixup_Mips_DTPREL_LO:
    return ELF::R_MIPS_TLS_DTPREL_LO16;
  case Mips::fixup_Mips_GOT_PAGE:
    return ELF::R_MIPS_GOT_PAGE;
  case Mips::fixup_Mips_GOT_OFST:
    return ELF::R_MIPS_GOT_OFST;
  case Mips::fixup_Mips_GOT_DISP:
    return ELF::R_MIPS_GOT_DISP;
  case Mips::fixup_Mips_GPOFF_HI: {
    unsigned Type = (unsigned)ELF::R_MIPS_NONE;
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_HI16, Type);
    return Type;
  }
  case Mips::fixup_Mips_GPOFF_LO: {
    unsigned Type = (unsigned)ELF::R_MIPS_NONE;
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_LO16, Type);
    return Type;
  }
  case Mips::fixup_Mips_HIGHER:
    return ELF::R_MIPS_HIGHER;
  case Mips::fixup_Mips_HIGHEST:
    return ELF::R_MIPS_HIGHEST;
  case Mips::fixup_Mips_GOT_HI16:
    return ELF::R_MIPS_GOT_HI16;
  case Mips::fixup_Mips_GOT_LO16:
    return ELF::R_MIPS_GOT_LO16;
  case Mips::fixup_Mips_CALL_HI16:
    return ELF::R_MIPS_CALL_HI16;
  case Mips::fixup_Mips_CALL_LO16:
    return ELF::R_MIPS_CALL_LO16;
  case Mips::fixup_MICROMIPS_26_S1:
    return ELF::R_MICROMIPS_26_S1;
  case Mips::fixup_MICROMIPS_HI16:
    return ELF::R_MICROMIPS_HI16;
  case Mips::fixup_MICROMIPS_LO16:
    return ELF::R_MICROMIPS_LO16;
  case Mips::fixup_MICROMIPS_GOT16:
    return ELF::R_MICROMIPS_GOT16;
  case Mips::fixup_MICROMIPS_CALL16:
    return ELF::R_MICROMIPS_CALL16;
  case Mips::fixup_MICROMIPS_GOT_DISP:
    return ELF::R_MICROMIPS_GOT_DISP;
  case Mips::fixup_MICROMIPS_GOT_PAGE:
    return ELF::R_MICROMIPS_GOT_PAGE;
  case Mips::fixup_MICROMIPS_GOT_OFST:
    return ELF::R_MICROMIPS_GOT_OFST;
  case Mips::fixup_MICROMIPS_TLS_GD:
    return ELF::R_MICROMIPS_TLS_GD;
  case Mips::fixup_MICROMIPS_TLS_LDM:
    return ELF::R_MICROMIPS_TLS_LDM;
  case Mips::fixup_MICROMIPS_TLS_DTPREL_HI16:
    return ELF::R_MICROMIPS_TLS_DTPREL_HI16;
  case Mips::fixup_MICROMIPS_TLS_DTPREL_LO16:
    return ELF::R_MICROMIPS_TLS_DTPREL_LO16;
  case Mips::fixup_MICROMIPS_TLS_TPREL_HI16:
    return ELF::R_MICROMIPS_TLS_TPREL_HI16;
  case Mips::fixup_MICROMIPS_TLS_TPREL_LO16:
    return ELF::R_MICROMIPS_TLS_TPREL_LO16;
  }

  llvm_unreachable("invalid fixup kind!");
}

// Sort entries by SortOffset in descending order.
// When there are more *HI16 relocs paired with one *LO16 reloc, the 2nd rule
// sorts them in ascending order of R.Offset.
static int cmpRelMips(const MipsRelocationEntry *AP,
                      const MipsRelocationEntry *BP) {
  const MipsRelocationEntry &A = *AP;
  const MipsRelocationEntry &B = *BP;
  if (A.SortOffset != B.SortOffset)
    return B.SortOffset - A.SortOffset;
  if (A.R.Offset != B.R.Offset)
    return A.R.Offset - B.R.Offset;
  if (B.R.Type != A.R.Type)
    return B.R.Type - A.R.Type;
  //llvm_unreachable("ELFRelocs might be unstable!");
  return 0;
}

// For the given Reloc.Type, return the matching relocation type, as in the
// table below.
static unsigned getMatchingLoType(const MCAssembler &Asm,
                                  const ELFRelocationEntry &Reloc) {
  unsigned Type = Reloc.Type;
  if (Type == ELF::R_MIPS_HI16)
    return ELF::R_MIPS_LO16;
  if (Type == ELF::R_MICROMIPS_HI16)
    return ELF::R_MICROMIPS_LO16;
  if (Type == ELF::R_MIPS16_HI16)
    return ELF::R_MIPS16_LO16;

  if (Reloc.Symbol->getBinding() != ELF::STB_LOCAL)
    return ELF::R_MIPS_NONE;

  if (Type == ELF::R_MIPS_GOT16)
    return ELF::R_MIPS_LO16;
  if (Type == ELF::R_MICROMIPS_GOT16)
    return ELF::R_MICROMIPS_LO16;
  if (Type == ELF::R_MIPS16_GOT16)
    return ELF::R_MIPS16_LO16;

  return ELF::R_MIPS_NONE;
}

// Return true if First needs a matching *LO16, its matching *LO16 type equals
// Second's type and both relocations are against the same symbol.
static bool areMatchingHiAndLo(const MCAssembler &Asm,
                               const ELFRelocationEntry &First,
                               const ELFRelocationEntry &Second) {
  return getMatchingLoType(Asm, First) != ELF::R_MIPS_NONE &&
         getMatchingLoType(Asm, First) == Second.Type &&
         First.Symbol && First.Symbol == Second.Symbol;
}

// Return true if MipsRelocs[Index] is a *LO16 preceded by a matching *HI16.
static bool
isPrecededByMatchingHi(const MCAssembler &Asm, uint32_t Index,
                       std::vector<MipsRelocationEntry> &MipsRelocs) {
  return Index < MipsRelocs.size() - 1 &&
         areMatchingHiAndLo(Asm, MipsRelocs[Index + 1].R, MipsRelocs[Index].R);
}

// Return true if MipsRelocs[Index] is a *LO16 not preceded by a matching *HI16
// and not chosen by a *HI16 as a match.
static bool isFreeLo(const MCAssembler &Asm, uint32_t Index,
                     std::vector<MipsRelocationEntry> &MipsRelocs) {
  return Index < MipsRelocs.size() && !MipsRelocs[Index].HasMatchingHi &&
         !isPrecededByMatchingHi(Asm, Index, MipsRelocs);
}

// Lo is chosen as a match for Hi, set their fields accordingly.
// Mips instructions have fixed length of at least two bytes (two for
// micromips/mips16, four for mips32/64), so we can set HI's SortOffset to
// matching LO's Offset minus one to simplify the sorting function.
static void setMatch(MipsRelocationEntry &Hi, MipsRelocationEntry &Lo) {
  Lo.HasMatchingHi = true;
  Hi.SortOffset = Lo.R.Offset - 1;
}

// We sort relocation table entries by offset, except for one additional rule
// required by MIPS ABI: every *HI16 relocation must be immediately followed by
// the corresponding *LO16 relocation. We also support a GNU extension that
// allows more *HI16s paired with one *LO16.
//
// *HI16 relocations and their matching *LO16 are:
//
// +---------------------------------------------+-------------------+
// |               *HI16                         |  matching *LO16   |
// |---------------------------------------------+-------------------|
// |  R_MIPS_HI16, local R_MIPS_GOT16            |    R_MIPS_LO16    |
// |  R_MICROMIPS_HI16, local R_MICROMIPS_GOT16  | R_MICROMIPS_LO16  |
// |  R_MIPS16_HI16, local R_MIPS16_GOT16        |  R_MIPS16_LO16    |
// +---------------------------------------------+-------------------+
//
// (local R_*_GOT16 meaning R_*_GOT16 against the local symbol.)
//
// To handle *HI16 and *LO16 relocations, the linker needs a combined addend
// ("AHL") calculated from both *HI16 ("AHI") and *LO16 ("ALO") relocations:
// AHL = (AHI << 16) + (short)ALO;
//
// We are reusing gnu as sorting algorithm so we are emitting the relocation
// table sorted the same way as gnu as would sort it, for easier comparison of
// the generated .o files.
//
// The logic is:
// search the table (starting from the highest offset and going back to zero)
// for all *HI16 relocations that don't have a matching *LO16.
// For every such HI, find a matching LO with highest offset that isn't already
// matched with another HI. If there are no free LOs, match it with the first
// found (starting from lowest offset).
// When there are more HIs matched with one LO, sort them in descending order by
// offset.
//
// In other words, when searching for a matching LO:
// - don't look for a 'better' match for the HIs that are already followed by a
//   matching LO;
// - prefer LOs without a pair;
// - prefer LOs with higher offset;
void MipsELFObjectWriter::sortRelocs(const MCAssembler &Asm,
                                     std::vector<ELFRelocationEntry> &Relocs) {
  if (Relocs.size() < 2)
    return;

  // The default function sorts entries by Offset in descending order.
  MCELFObjectTargetWriter::sortRelocs(Asm, Relocs);

  // Init MipsRelocs from Relocs.
  std::vector<MipsRelocationEntry> MipsRelocs;
  for (unsigned I = 0, E = Relocs.size(); I != E; ++I)
    MipsRelocs.push_back(MipsRelocationEntry(Relocs[I]));

  // Find a matching LO for all HIs that need it.
  for (int32_t I = 0, E = MipsRelocs.size(); I != E; ++I) {
    if (getMatchingLoType(Asm, MipsRelocs[I].R) == ELF::R_MIPS_NONE ||
        (I > 0 && isPrecededByMatchingHi(Asm, I - 1, MipsRelocs)))
      continue;

    int32_t MatchedLoIndex = -1;

    // Search the list in the ascending order of Offset.
    for (int32_t J = MipsRelocs.size() - 1, N = -1; J != N; --J) {
      // check for a match
      if (areMatchingHiAndLo(Asm, MipsRelocs[I].R, MipsRelocs[J].R) &&
          (MatchedLoIndex == -1 || // first match
           // or we already have a match,
           // but this one is with higher offset and it's free
           (MatchedLoIndex > J && isFreeLo(Asm, J, MipsRelocs))))
        MatchedLoIndex = J;
    }

    if (MatchedLoIndex != -1)
      // We have a match.
      setMatch(MipsRelocs[I], MipsRelocs[MatchedLoIndex]);
  }

  // SortOffsets are calculated, call the sorting function.
  array_pod_sort(MipsRelocs.begin(), MipsRelocs.end(), cmpRelMips);

  // Copy sorted MipsRelocs back to Relocs.
  for (unsigned I = 0, E = MipsRelocs.size(); I != E; ++I)
    Relocs[I] = MipsRelocs[I].R;
}

bool MipsELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                  unsigned Type) const {
  // FIXME: This is extremely conservative. This really needs to use a
  // whitelist with a clear explanation for why each realocation needs to
  // point to the symbol, not to the section.
  switch (Type) {
  default:
    return true;

  case ELF::R_MIPS_GOT16:
  case ELF::R_MIPS16_GOT16:
  case ELF::R_MICROMIPS_GOT16:
    llvm_unreachable("Should have been handled already");

  // These relocations might be paired with another relocation. The pairing is
  // done by the static linker by matching the symbol. Since we only see one
  // relocation at a time, we have to force them to relocate with a symbol to
  // avoid ending up with a pair where one points to a section and another
  // points to a symbol.
  case ELF::R_MIPS_HI16:
  case ELF::R_MIPS16_HI16:
  case ELF::R_MICROMIPS_HI16:
  case ELF::R_MIPS_LO16:
  case ELF::R_MIPS16_LO16:
  case ELF::R_MICROMIPS_LO16:
    return true;

  case ELF::R_MIPS_32:
    if (cast<MCSymbolELF>(Sym).getOther() & ELF::STO_MIPS_MICROMIPS)
      return true;
    // falltrough
  case ELF::R_MIPS_26:
  case ELF::R_MIPS_64:
  case ELF::R_MIPS_GPREL16:
    return false;
  }
}

MCObjectWriter *llvm::createMipsELFObjectWriter(raw_pwrite_stream &OS,
                                                uint8_t OSABI,
                                                bool IsLittleEndian,
                                                bool Is64Bit) {
  MCELFObjectTargetWriter *MOTW =
      new MipsELFObjectWriter(Is64Bit, OSABI, Is64Bit, IsLittleEndian);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
