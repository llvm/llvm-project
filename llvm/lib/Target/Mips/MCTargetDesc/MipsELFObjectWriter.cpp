//===-- MipsELFObjectWriter.cpp - Mips ELF Writer -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCExpr.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <list>
#include <utility>

#define DEBUG_TYPE "mips-elf-object-writer"

using namespace llvm;

namespace {

/// Holds additional information needed by the relocation ordering algorithm.
struct MipsRelocationEntry {
  const ELFRelocationEntry R; ///< The relocation.
  bool Matched = false;       ///< Is this relocation part of a match.

  MipsRelocationEntry(const ELFRelocationEntry &R) : R(R) {}
};

class MipsELFObjectWriter : public MCELFObjectTargetWriter {
public:
  MipsELFObjectWriter(uint8_t OSABI, bool HasRelocationAddend, bool Is64);

  ~MipsELFObjectWriter() override = default;

  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
  bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym,
                               unsigned Type) const override;
  void sortRelocs(std::vector<ELFRelocationEntry> &Relocs) override;
};

/// The possible results of the Predicate function used by find_best.
enum FindBestPredicateResult {
  FindBest_NoMatch = 0,  ///< The current element is not a match.
  FindBest_Match,        ///< The current element is a match but better ones are
                         ///  possible.
  FindBest_PerfectMatch, ///< The current element is an unbeatable match.
};

} // end anonymous namespace

/// Copy elements in the range [First, Last) to d1 when the predicate is true or
/// d2 when the predicate is false. This is essentially both std::copy_if and
/// std::remove_copy_if combined into a single pass.
template <class InputIt, class OutputIt1, class OutputIt2, class UnaryPredicate>
static std::pair<OutputIt1, OutputIt2> copy_if_else(InputIt First, InputIt Last,
                                                    OutputIt1 d1, OutputIt2 d2,
                                                    UnaryPredicate Predicate) {
  for (InputIt I = First; I != Last; ++I) {
    if (Predicate(*I)) {
      *d1 = *I;
      d1++;
    } else {
      *d2 = *I;
      d2++;
    }
  }

  return std::make_pair(d1, d2);
}

/// Find the best match in the range [First, Last).
///
/// An element matches when Predicate(X) returns FindBest_Match or
/// FindBest_PerfectMatch. A value of FindBest_PerfectMatch also terminates
/// the search. BetterThan(A, B) is a comparator that returns true when A is a
/// better match than B. The return value is the position of the best match.
///
/// This is similar to std::find_if but finds the best of multiple possible
/// matches.
template <class InputIt, class UnaryPredicate, class Comparator>
static InputIt find_best(InputIt First, InputIt Last, UnaryPredicate Predicate,
                         Comparator BetterThan) {
  InputIt Best = Last;

  for (InputIt I = First; I != Last; ++I) {
    unsigned Matched = Predicate(*I);
    if (Matched != FindBest_NoMatch) {
      if (Best == Last || BetterThan(*I, *Best))
        Best = I;
    }
    if (Matched == FindBest_PerfectMatch)
      break;
  }

  return Best;
}

/// Determine the low relocation that matches the given relocation.
/// If the relocation does not need a low relocation then the return value
/// is ELF::R_MIPS_NONE.
///
/// The relocations that need a matching low part are
/// R_(MIPS|MICROMIPS|MIPS16)_HI16 for all symbols and
/// R_(MIPS|MICROMIPS|MIPS16)_GOT16 for local symbols only.
static unsigned getMatchingLoType(const ELFRelocationEntry &Reloc) {
  unsigned Type = Reloc.Type;
  if (Type == ELF::R_MIPS_HI16)
    return ELF::R_MIPS_LO16;
  if (Type == ELF::R_MICROMIPS_HI16)
    return ELF::R_MICROMIPS_LO16;
  if (Type == ELF::R_MIPS16_HI16)
    return ELF::R_MIPS16_LO16;

  if (Reloc.Symbol && Reloc.Symbol->getBinding() != ELF::STB_LOCAL)
    return ELF::R_MIPS_NONE;

  if (Type == ELF::R_MIPS_GOT16)
    return ELF::R_MIPS_LO16;
  if (Type == ELF::R_MICROMIPS_GOT16)
    return ELF::R_MICROMIPS_LO16;
  if (Type == ELF::R_MIPS16_GOT16)
    return ELF::R_MIPS16_LO16;

  return ELF::R_MIPS_NONE;
}

// Determine whether a relocation X is a low-part and matches the high-part R
// perfectly by symbol and addend.
static bool isMatchingReloc(unsigned MatchingType, const ELFRelocationEntry &R,
                            const ELFRelocationEntry &X) {
  return X.Type == MatchingType && X.Symbol == R.Symbol && X.Addend == R.Addend;
}

MipsELFObjectWriter::MipsELFObjectWriter(uint8_t OSABI,
                                         bool HasRelocationAddend, bool Is64)
    : MCELFObjectTargetWriter(Is64, OSABI, ELF::EM_MIPS, HasRelocationAddend) {}

unsigned MipsELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                           const MCValue &Target,
                                           bool IsPCRel) const {
  // Determine the type of the relocation.
  unsigned Kind = Fixup.getTargetKind();
  switch (Target.getSpecifier()) {
  case MipsMCExpr::MEK_DTPREL:
  case MipsMCExpr::MEK_DTPREL_HI:
  case MipsMCExpr::MEK_DTPREL_LO:
  case MipsMCExpr::MEK_TLSLDM:
  case MipsMCExpr::MEK_TLSGD:
  case MipsMCExpr::MEK_GOTTPREL:
  case MipsMCExpr::MEK_TPREL_HI:
  case MipsMCExpr::MEK_TPREL_LO:
    if (auto *SA = Target.getAddSym())
      cast<MCSymbolELF>(SA)->setType(ELF::STT_TLS);
    break;
  default:
    break;
  }

  switch (Kind) {
  case FK_NONE:
    return ELF::R_MIPS_NONE;
  case FK_Data_1:
    reportError(Fixup.getLoc(), "MIPS does not support one byte relocations");
    return ELF::R_MIPS_NONE;
  case Mips::fixup_Mips_16:
  case FK_Data_2:
    return IsPCRel ? ELF::R_MIPS_PC16 : ELF::R_MIPS_16;
  case Mips::fixup_Mips_32:
  case FK_Data_4:
    return IsPCRel ? ELF::R_MIPS_PC32 : ELF::R_MIPS_32;
  case Mips::fixup_Mips_64:
  case FK_Data_8:
    return IsPCRel
               ? setRTypes(ELF::R_MIPS_PC32, ELF::R_MIPS_64, ELF::R_MIPS_NONE)
               : (unsigned)ELF::R_MIPS_64;
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
    case Mips::fixup_MICROMIPS_PC26_S1:
      return ELF::R_MICROMIPS_PC26_S1;
    case Mips::fixup_MICROMIPS_PC19_S2:
      return ELF::R_MICROMIPS_PC19_S2;
    case Mips::fixup_MICROMIPS_PC18_S3:
      return ELF::R_MICROMIPS_PC18_S3;
    case Mips::fixup_MICROMIPS_PC21_S1:
      return ELF::R_MICROMIPS_PC21_S1;
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
  case Mips::fixup_Mips_DTPREL32:
    return ELF::R_MIPS_TLS_DTPREL32;
  case Mips::fixup_Mips_DTPREL64:
    return ELF::R_MIPS_TLS_DTPREL64;
  case Mips::fixup_Mips_TPREL32:
    return ELF::R_MIPS_TLS_TPREL32;
  case Mips::fixup_Mips_TPREL64:
    return ELF::R_MIPS_TLS_TPREL64;
  case Mips::fixup_Mips_GPREL32:
    return setRTypes(ELF::R_MIPS_GPREL32,
                     is64Bit() ? ELF::R_MIPS_64 : ELF::R_MIPS_NONE,
                     ELF::R_MIPS_NONE);
  case Mips::fixup_Mips_GPREL16:
    return ELF::R_MIPS_GPREL16;
  case Mips::fixup_Mips_26:
    return ELF::R_MIPS_26;
  case Mips::fixup_Mips_CALL16:
    return ELF::R_MIPS_CALL16;
  case Mips::fixup_Mips_GOT:
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
  case Mips::fixup_Mips_GPOFF_HI:
    return setRTypes(ELF::R_MIPS_GPREL16, ELF::R_MIPS_SUB, ELF::R_MIPS_HI16);
  case Mips::fixup_MICROMIPS_GPOFF_HI:
    return setRTypes(ELF::R_MICROMIPS_GPREL16, ELF::R_MICROMIPS_SUB,
                     ELF::R_MICROMIPS_HI16);
  case Mips::fixup_Mips_GPOFF_LO:
    return setRTypes(ELF::R_MIPS_GPREL16, ELF::R_MIPS_SUB, ELF::R_MIPS_LO16);
  case Mips::fixup_MICROMIPS_GPOFF_LO:
    return setRTypes(ELF::R_MICROMIPS_GPREL16, ELF::R_MICROMIPS_SUB,
                     ELF::R_MICROMIPS_LO16);
  case Mips::fixup_Mips_HIGHER:
    return ELF::R_MIPS_HIGHER;
  case Mips::fixup_Mips_HIGHEST:
    return ELF::R_MIPS_HIGHEST;
  case Mips::fixup_Mips_SUB:
    return ELF::R_MIPS_SUB;
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
  case Mips::fixup_MICROMIPS_GOTTPREL:
    return ELF::R_MICROMIPS_TLS_GOTTPREL;
  case Mips::fixup_MICROMIPS_TLS_TPREL_HI16:
    return ELF::R_MICROMIPS_TLS_TPREL_HI16;
  case Mips::fixup_MICROMIPS_TLS_TPREL_LO16:
    return ELF::R_MICROMIPS_TLS_TPREL_LO16;
  case Mips::fixup_MICROMIPS_SUB:
    return ELF::R_MICROMIPS_SUB;
  case Mips::fixup_MICROMIPS_HIGHER:
    return ELF::R_MICROMIPS_HIGHER;
  case Mips::fixup_MICROMIPS_HIGHEST:
    return ELF::R_MICROMIPS_HIGHEST;
  case Mips::fixup_Mips_JALR:
    return ELF::R_MIPS_JALR;
  case Mips::fixup_MICROMIPS_JALR:
    return ELF::R_MICROMIPS_JALR;
  }

  reportError(Fixup.getLoc(), "unsupported relocation type");
  return ELF::R_MIPS_NONE;
}

/// Sort relocation table entries by offset except where another order is
/// required by the MIPS ABI.
///
/// MIPS has a few relocations that have an AHL component in the expression used
/// to evaluate them. This AHL component is an addend with the same number of
/// bits as a symbol value but not all of our ABI's are able to supply a
/// sufficiently sized addend in a single relocation.
///
/// The O32 ABI for example, uses REL relocations which store the addend in the
/// section data. All the relocations with AHL components affect 16-bit fields
/// so the addend for a single relocation is limited to 16-bit. This ABI
/// resolves the limitation by linking relocations (e.g. R_MIPS_HI16 and
/// R_MIPS_LO16) and distributing the addend between the linked relocations. The
/// ABI mandates that such relocations must be next to each other in a
/// particular order (e.g. R_MIPS_HI16 must be immediately followed by a
/// matching R_MIPS_LO16) but the rule is less strict in practice.
///
/// The de facto standard is lenient in the following ways:
/// - 'Immediately following' does not refer to the next relocation entry but
///   the next matching relocation.
/// - There may be multiple high parts relocations for one low part relocation.
/// - There may be multiple low part relocations for one high part relocation.
/// - The AHL addend in each part does not have to be exactly equal as long as
///   the difference does not affect the carry bit from bit 15 into 16. This is
///   to allow, for example, the use of %lo(foo) and %lo(foo+4) when loading
///   both halves of a long long.
///
/// See getMatchingLoType() for a description of which high part relocations
/// match which low part relocations. One particular thing to note is that
/// R_MIPS_GOT16 and similar only have AHL addends if they refer to local
/// symbols.
///
/// It should also be noted that this function is not affected by whether
/// the symbol was kept or rewritten into a section-relative equivalent. We
/// always match using the expressions from the source.
void MipsELFObjectWriter::sortRelocs(std::vector<ELFRelocationEntry> &Relocs) {
  // We do not need to sort the relocation table for RELA relocations which
  // N32/N64 uses as the relocation addend contains the value we require,
  // rather than it being split across a pair of relocations.
  if (hasRelocationAddend())
    return;

  // Sort relocations by the address they are applied to.
  llvm::sort(Relocs,
             [](const ELFRelocationEntry &A, const ELFRelocationEntry &B) {
               return A.Offset < B.Offset;
             });

  // Place relocations in a list for reorder convenience. Hi16 contains the
  // iterators of high-part relocations.
  std::list<MipsRelocationEntry> Sorted;
  SmallVector<std::list<MipsRelocationEntry>::iterator, 0> Hi16;
  for (auto &R : Relocs) {
    Sorted.push_back(R);
    if (getMatchingLoType(R) != ELF::R_MIPS_NONE)
      Hi16.push_back(std::prev(Sorted.end()));
  }

  for (auto I : Hi16) {
    auto &R = I->R;
    unsigned MatchingType = getMatchingLoType(R);
    // If the next relocation is a perfect match, continue;
    if (std::next(I) != Sorted.end() &&
        isMatchingReloc(MatchingType, R, std::next(I)->R))
      continue;
    // Otherwise, find the best matching low-part relocation with the following
    // criteria. It must have the same symbol and its addend is no lower than
    // that of the current high-part.
    //
    // (1) %lo with a smaller offset is preferred.
    // (2) %lo with the same offset that is unmatched is preferred.
    // (3) later %lo is preferred.
    auto Best = Sorted.end();
    for (auto J = Sorted.begin(); J != Sorted.end(); ++J) {
      auto &R1 = J->R;
      if (R1.Type == MatchingType && R.Symbol == R1.Symbol &&
          R.Addend <= R1.Addend &&
          (Best == Sorted.end() || R1.Addend < Best->R.Addend ||
           (!Best->Matched && R1.Addend == Best->R.Addend)))
        Best = J;
    }
    if (Best != Sorted.end() && R.Addend == Best->R.Addend)
      Best->Matched = true;

    // Move the high-part before the low-part, or if not found, the end of the
    // list. The unmatched high-part will lead to a linker warning/error.
    Sorted.splice(Best, Sorted, I);
  }

  assert(Relocs.size() == Sorted.size() && "Some relocs were not consumed");

  // Overwrite the original vector with the sorted elements.
  unsigned CopyTo = 0;
  for (const auto &R : Sorted)
    Relocs[CopyTo++] = R.R;
}

bool MipsELFObjectWriter::needsRelocateWithSymbol(const MCValue &Val,
                                                  const MCSymbol &Sym,
                                                  unsigned Type) const {
  // If it's a compound relocation for N64 then we need the relocation if any
  // sub-relocation needs it.
  if (!isUInt<8>(Type))
    return needsRelocateWithSymbol(Val, Sym, Type & 0xff) ||
           needsRelocateWithSymbol(Val, Sym, (Type >> 8) & 0xff) ||
           needsRelocateWithSymbol(Val, Sym, (Type >> 16) & 0xff);

  switch (Type) {
  default:
    errs() << Type << "\n";
    llvm_unreachable("Unexpected relocation");
    return true;

  // This relocation doesn't affect the section data.
  case ELF::R_MIPS_NONE:
    return false;

  // On REL ABI's (e.g. O32), these relocations form pairs. The pairing is done
  // by the static linker by matching the symbol and offset.
  // We only see one relocation at a time but it's still safe to relocate with
  // the section so long as both relocations make the same decision.
  //
  // Some older linkers may require the symbol for particular cases. Such cases
  // are not supported yet but can be added as required.
  case ELF::R_MIPS_GOT16:
  case ELF::R_MIPS16_GOT16:
  case ELF::R_MICROMIPS_GOT16:
  case ELF::R_MIPS_HIGHER:
  case ELF::R_MIPS_HIGHEST:
  case ELF::R_MIPS_HI16:
  case ELF::R_MIPS16_HI16:
  case ELF::R_MICROMIPS_HI16:
  case ELF::R_MIPS_LO16:
  case ELF::R_MIPS16_LO16:
  case ELF::R_MICROMIPS_LO16:
    // FIXME: It should be safe to return false for the STO_MIPS_MICROMIPS but
    //        we neglect to handle the adjustment to the LSB of the addend that
    //        it causes in applyFixup() and similar.
    if (cast<MCSymbolELF>(Sym).getOther() & ELF::STO_MIPS_MICROMIPS)
      return true;
    return false;

  case ELF::R_MIPS_GOT_PAGE:
  case ELF::R_MICROMIPS_GOT_PAGE:
  case ELF::R_MIPS_GOT_OFST:
  case ELF::R_MICROMIPS_GOT_OFST:
  case ELF::R_MIPS_16:
  case ELF::R_MIPS_32:
  case ELF::R_MIPS_GPREL32:
    if (cast<MCSymbolELF>(Sym).getOther() & ELF::STO_MIPS_MICROMIPS)
      return true;
    [[fallthrough]];
  case ELF::R_MIPS_26:
  case ELF::R_MIPS_64:
  case ELF::R_MIPS_GPREL16:
  case ELF::R_MIPS_PC16:
  case ELF::R_MIPS_SUB:
    return false;

  // FIXME: Many of these relocations should probably return false but this
  //        hasn't been confirmed to be safe yet.
  case ELF::R_MIPS_REL32:
  case ELF::R_MIPS_LITERAL:
  case ELF::R_MIPS_CALL16:
  case ELF::R_MIPS_SHIFT5:
  case ELF::R_MIPS_SHIFT6:
  case ELF::R_MIPS_GOT_DISP:
  case ELF::R_MIPS_GOT_HI16:
  case ELF::R_MIPS_GOT_LO16:
  case ELF::R_MIPS_INSERT_A:
  case ELF::R_MIPS_INSERT_B:
  case ELF::R_MIPS_DELETE:
  case ELF::R_MIPS_CALL_HI16:
  case ELF::R_MIPS_CALL_LO16:
  case ELF::R_MIPS_SCN_DISP:
  case ELF::R_MIPS_REL16:
  case ELF::R_MIPS_ADD_IMMEDIATE:
  case ELF::R_MIPS_PJUMP:
  case ELF::R_MIPS_RELGOT:
  case ELF::R_MIPS_JALR:
  case ELF::R_MIPS_TLS_DTPMOD32:
  case ELF::R_MIPS_TLS_DTPREL32:
  case ELF::R_MIPS_TLS_DTPMOD64:
  case ELF::R_MIPS_TLS_DTPREL64:
  case ELF::R_MIPS_TLS_GD:
  case ELF::R_MIPS_TLS_LDM:
  case ELF::R_MIPS_TLS_DTPREL_HI16:
  case ELF::R_MIPS_TLS_DTPREL_LO16:
  case ELF::R_MIPS_TLS_GOTTPREL:
  case ELF::R_MIPS_TLS_TPREL32:
  case ELF::R_MIPS_TLS_TPREL64:
  case ELF::R_MIPS_TLS_TPREL_HI16:
  case ELF::R_MIPS_TLS_TPREL_LO16:
  case ELF::R_MIPS_GLOB_DAT:
  case ELF::R_MIPS_PC21_S2:
  case ELF::R_MIPS_PC26_S2:
  case ELF::R_MIPS_PC18_S3:
  case ELF::R_MIPS_PC19_S2:
  case ELF::R_MIPS_PCHI16:
  case ELF::R_MIPS_PCLO16:
  case ELF::R_MIPS_COPY:
  case ELF::R_MIPS_JUMP_SLOT:
  case ELF::R_MIPS_NUM:
  case ELF::R_MIPS_PC32:
  case ELF::R_MIPS_EH:
  case ELF::R_MICROMIPS_26_S1:
  case ELF::R_MICROMIPS_GPREL16:
  case ELF::R_MICROMIPS_LITERAL:
  case ELF::R_MICROMIPS_PC7_S1:
  case ELF::R_MICROMIPS_PC10_S1:
  case ELF::R_MICROMIPS_PC16_S1:
  case ELF::R_MICROMIPS_CALL16:
  case ELF::R_MICROMIPS_GOT_DISP:
  case ELF::R_MICROMIPS_GOT_HI16:
  case ELF::R_MICROMIPS_GOT_LO16:
  case ELF::R_MICROMIPS_SUB:
  case ELF::R_MICROMIPS_HIGHER:
  case ELF::R_MICROMIPS_HIGHEST:
  case ELF::R_MICROMIPS_CALL_HI16:
  case ELF::R_MICROMIPS_CALL_LO16:
  case ELF::R_MICROMIPS_SCN_DISP:
  case ELF::R_MICROMIPS_JALR:
  case ELF::R_MICROMIPS_HI0_LO16:
  case ELF::R_MICROMIPS_TLS_GD:
  case ELF::R_MICROMIPS_TLS_LDM:
  case ELF::R_MICROMIPS_TLS_DTPREL_HI16:
  case ELF::R_MICROMIPS_TLS_DTPREL_LO16:
  case ELF::R_MICROMIPS_TLS_GOTTPREL:
  case ELF::R_MICROMIPS_TLS_TPREL_HI16:
  case ELF::R_MICROMIPS_TLS_TPREL_LO16:
  case ELF::R_MICROMIPS_GPREL7_S2:
  case ELF::R_MICROMIPS_PC23_S2:
  case ELF::R_MICROMIPS_PC21_S1:
  case ELF::R_MICROMIPS_PC26_S1:
  case ELF::R_MICROMIPS_PC18_S3:
  case ELF::R_MICROMIPS_PC19_S2:
    return true;

  // FIXME: Many of these should probably return false but MIPS16 isn't
  //        supported by the integrated assembler.
  case ELF::R_MIPS16_26:
  case ELF::R_MIPS16_GPREL:
  case ELF::R_MIPS16_CALL16:
  case ELF::R_MIPS16_TLS_GD:
  case ELF::R_MIPS16_TLS_LDM:
  case ELF::R_MIPS16_TLS_DTPREL_HI16:
  case ELF::R_MIPS16_TLS_DTPREL_LO16:
  case ELF::R_MIPS16_TLS_GOTTPREL:
  case ELF::R_MIPS16_TLS_TPREL_HI16:
  case ELF::R_MIPS16_TLS_TPREL_LO16:
    llvm_unreachable("Unsupported MIPS16 relocation");
    return true;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createMipsELFObjectWriter(const Triple &TT, bool IsN32) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  bool IsN64 = TT.isArch64Bit() && !IsN32;
  bool HasRelocationAddend = TT.isArch64Bit();
  return std::make_unique<MipsELFObjectWriter>(OSABI, HasRelocationAddend,
                                                IsN64);
}
