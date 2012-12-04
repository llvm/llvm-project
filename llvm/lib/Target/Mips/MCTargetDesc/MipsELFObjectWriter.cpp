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
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include <list>

using namespace llvm;

namespace {
  struct RelEntry {
    RelEntry(const ELFRelocationEntry &R, const MCSymbol *S, int64_t O) :
      Reloc(R), Sym(S), Offset(O) {}
    ELFRelocationEntry Reloc;
    const MCSymbol *Sym;
    int64_t Offset;
  };

  typedef std::list<RelEntry> RelLs;
  typedef RelLs::iterator RelLsIter;

  class MipsELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                        bool _isN64, bool IsLittleEndian);

    virtual ~MipsELFObjectWriter();

    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual unsigned getEFlags() const;
    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                           const MCValue &Target,
                                           const MCFragment &F,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const;
    virtual void sortRelocs(const MCAssembler &Asm,
                            std::vector<ELFRelocationEntry> &Relocs);
  };
}

MipsELFObjectWriter::MipsELFObjectWriter(bool _is64Bit, uint8_t OSABI,
                                         bool _isN64, bool IsLittleEndian)
  : MCELFObjectTargetWriter(_is64Bit, OSABI, ELF::EM_MIPS,
                            /*HasRelocationAddend*/ (_isN64) ? true : false,
                            /*IsN64*/ _isN64) {}

MipsELFObjectWriter::~MipsELFObjectWriter() {}

// FIXME: get the real EABI Version from the Subtarget class.
unsigned MipsELFObjectWriter::getEFlags() const {

  // FIXME: We can't tell if we are PIC (dynamic) or CPIC (static)
  unsigned Flag = ELF::EF_MIPS_NOREORDER;

  if (is64Bit())
    Flag |= ELF::EF_MIPS_ARCH_64R2;
  else
    Flag |= ELF::EF_MIPS_ARCH_32R2;
  return Flag;
}

const MCSymbol *MipsELFObjectWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                    const MCValue &Target,
                                                    const MCFragment &F,
                                                    const MCFixup &Fixup,
                                                    bool IsPCRel) const {
  assert(Target.getSymA() && "SymA cannot be 0.");
  const MCSymbol &Sym = Target.getSymA()->getSymbol().AliasedSymbol();

  if (Sym.getSection().getKind().isMergeableCString() ||
      Sym.getSection().getKind().isMergeableConst())
    return &Sym;

  return NULL;
}

unsigned MipsELFObjectWriter::GetRelocType(const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel,
                                           bool IsRelocWithSymbol,
                                           int64_t Addend) const {
  // determine the type of the relocation
  unsigned Type = (unsigned)ELF::R_MIPS_NONE;
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    Type = ELF::R_MIPS_32;
    break;
  case FK_Data_8:
    Type = ELF::R_MIPS_64;
    break;
  case FK_GPRel_4:
    Type = ELF::R_MIPS_GPREL32;
    break;
  case Mips::fixup_Mips_GPREL16:
    Type = ELF::R_MIPS_GPREL16;
    break;
  case Mips::fixup_Mips_26:
    Type = ELF::R_MIPS_26;
    break;
  case Mips::fixup_Mips_CALL16:
    Type = ELF::R_MIPS_CALL16;
    break;
  case Mips::fixup_Mips_GOT_Global:
  case Mips::fixup_Mips_GOT_Local:
    Type = ELF::R_MIPS_GOT16;
    break;
  case Mips::fixup_Mips_HI16:
    Type = ELF::R_MIPS_HI16;
    break;
  case Mips::fixup_Mips_LO16:
    Type = ELF::R_MIPS_LO16;
    break;
  case Mips::fixup_Mips_TLSGD:
    Type = ELF::R_MIPS_TLS_GD;
    break;
  case Mips::fixup_Mips_GOTTPREL:
    Type = ELF::R_MIPS_TLS_GOTTPREL;
    break;
  case Mips::fixup_Mips_TPREL_HI:
    Type = ELF::R_MIPS_TLS_TPREL_HI16;
    break;
  case Mips::fixup_Mips_TPREL_LO:
    Type = ELF::R_MIPS_TLS_TPREL_LO16;
    break;
  case Mips::fixup_Mips_TLSLDM:
    Type = ELF::R_MIPS_TLS_LDM;
    break;
  case Mips::fixup_Mips_DTPREL_HI:
    Type = ELF::R_MIPS_TLS_DTPREL_HI16;
    break;
  case Mips::fixup_Mips_DTPREL_LO:
    Type = ELF::R_MIPS_TLS_DTPREL_LO16;
    break;
  case Mips::fixup_Mips_Branch_PCRel:
  case Mips::fixup_Mips_PC16:
    Type = ELF::R_MIPS_PC16;
    break;
  case Mips::fixup_Mips_GOT_PAGE:
    Type = ELF::R_MIPS_GOT_PAGE;
    break;
  case Mips::fixup_Mips_GOT_OFST:
    Type = ELF::R_MIPS_GOT_OFST;
    break;
  case Mips::fixup_Mips_GOT_DISP:
    Type = ELF::R_MIPS_GOT_DISP;
    break;
  case Mips::fixup_Mips_GPOFF_HI:
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_HI16, Type);
    break;
  case Mips::fixup_Mips_GPOFF_LO:
    Type = setRType((unsigned)ELF::R_MIPS_GPREL16, Type);
    Type = setRType2((unsigned)ELF::R_MIPS_SUB, Type);
    Type = setRType3((unsigned)ELF::R_MIPS_LO16, Type);
    break;
  case Mips::fixup_Mips_HIGHER:
    Type = ELF::R_MIPS_HIGHER;
    break;
  case Mips::fixup_Mips_HIGHEST:
    Type = ELF::R_MIPS_HIGHEST;
    break;
  case Mips::fixup_Mips_GOT_HI16:
    Type = ELF::R_MIPS_GOT_HI16;
    break;
  case Mips::fixup_Mips_GOT_LO16:
    Type = ELF::R_MIPS_GOT_LO16;
    break;
  case Mips::fixup_Mips_CALL_HI16:
    Type = ELF::R_MIPS_CALL_HI16;
    break;
  case Mips::fixup_Mips_CALL_LO16:
    Type = ELF::R_MIPS_CALL_LO16;
    break;
  }
  return Type;
}

// Return true if R is either a GOT16 against a local symbol or HI16.
static bool NeedsMatchingLo(const MCAssembler &Asm, const RelEntry &R) {
  if (!R.Sym)
    return false;

  MCSymbolData &SD = Asm.getSymbolData(R.Sym->AliasedSymbol());

  return ((R.Reloc.Type == ELF::R_MIPS_GOT16) && !SD.isExternal()) ||
    (R.Reloc.Type == ELF::R_MIPS_HI16);
}

static bool HasMatchingLo(const MCAssembler &Asm, RelLsIter I, RelLsIter Last) {
  if (I == Last)
    return false;

  RelLsIter Hi = I++;

  return (I->Reloc.Type == ELF::R_MIPS_LO16) && (Hi->Sym == I->Sym) &&
    (Hi->Offset == I->Offset);
}

static bool HasSameSymbol(const RelEntry &R0, const RelEntry &R1) {
  return R0.Sym == R1.Sym;
}

static int CompareOffset(const RelEntry &R0, const RelEntry &R1) {
  return (R0.Offset > R1.Offset) ? 1 : ((R0.Offset == R1.Offset) ? 0 : -1);
}

void MipsELFObjectWriter::sortRelocs(const MCAssembler &Asm,
                                     std::vector<ELFRelocationEntry> &Relocs) {
  // Call the default function first. Relocations are sorted in descending
  // order of r_offset.
  MCELFObjectTargetWriter::sortRelocs(Asm, Relocs);

  RelLs RelocLs;
  std::vector<RelLsIter> Unmatched;

  // Fill RelocLs. Traverse Relocs backwards so that relocations in RelocLs
  // are in ascending order of r_offset.
  for (std::vector<ELFRelocationEntry>::reverse_iterator R = Relocs.rbegin();
       R != Relocs.rend(); ++R) {
     std::pair<const MCSymbolRefExpr*, int64_t> P =
       MipsGetSymAndOffset(*R->Fixup);
     RelocLs.push_back(RelEntry(*R, P.first ? &P.first->getSymbol() : 0,
                                P.second));
  }

  // Get list of unmatched HI16 and GOT16.
  for (RelLsIter R = RelocLs.begin(); R != RelocLs.end(); ++R)
    if (NeedsMatchingLo(Asm, *R) && !HasMatchingLo(Asm, R, --RelocLs.end()))
      Unmatched.push_back(R);

  // Insert unmatched HI16 and GOT16 immediately before their matching LO16.
  for (std::vector<RelLsIter>::iterator U = Unmatched.begin();
       U != Unmatched.end(); ++U) {
    RelLsIter LoPos = RelocLs.end(), HiPos = *U;
    bool MatchedLo = false;

    for (RelLsIter R = RelocLs.begin(); R != RelocLs.end(); ++R) {
      if ((R->Reloc.Type == ELF::R_MIPS_LO16) && HasSameSymbol(*HiPos, *R) &&
          (CompareOffset(*R, *HiPos) >= 0) &&
          ((LoPos == RelocLs.end()) || ((CompareOffset(*R, *LoPos) < 0)) ||
           (!MatchedLo && !CompareOffset(*R, *LoPos))))
        LoPos = R;

      MatchedLo = NeedsMatchingLo(Asm, *R) &&
        HasMatchingLo(Asm, R, --RelocLs.end());
    }

    // If a matching LoPos was found, move HiPos and insert it before LoPos.
    // Make the offsets of HiPos and LoPos match.
    if (LoPos != RelocLs.end()) {
      HiPos->Offset = LoPos->Offset;
      RelocLs.insert(LoPos, *HiPos);
      RelocLs.erase(HiPos);
    }
  }

  // Put the sorted list back in reverse order.
  assert(Relocs.size() == RelocLs.size());
  unsigned I = RelocLs.size();

  for (RelLsIter R = RelocLs.begin(); R != RelocLs.end(); ++R)
    Relocs[--I] = R->Reloc;
}

MCObjectWriter *llvm::createMipsELFObjectWriter(raw_ostream &OS,
                                                uint8_t OSABI,
                                                bool IsLittleEndian,
                                                bool Is64Bit) {
  MCELFObjectTargetWriter *MOTW = new MipsELFObjectWriter(Is64Bit, OSABI,
                                                (Is64Bit) ? true : false,
                                                IsLittleEndian);
  return createELFObjectWriter(MOTW, OS, IsLittleEndian);
}
