//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mcexpr"

namespace {
namespace stats {

STATISTIC(MCExprEvaluate, "Number of MCExpr evaluations");

} // end namespace stats
} // end anonymous namespace

// VariantKind printing and formatting utilize MAI. operator<< (dump and some
// target code) specifies MAI as nullptr and should be avoided when MAI is
// needed.
void MCExpr::print(raw_ostream &OS, const MCAsmInfo *MAI, bool InParens) const {
  switch (getKind()) {
  case MCExpr::Target:
    return cast<MCTargetExpr>(this)->printImpl(OS, MAI);
  case MCExpr::Constant: {
    auto Value = cast<MCConstantExpr>(*this).getValue();
    auto PrintInHex = cast<MCConstantExpr>(*this).useHexFormat();
    auto SizeInBytes = cast<MCConstantExpr>(*this).getSizeInBytes();
    if (Value < 0 && MAI && !MAI->supportsSignedData())
      PrintInHex = true;
    if (PrintInHex)
      switch (SizeInBytes) {
      default:
        OS << "0x" << Twine::utohexstr(Value);
        break;
      case 1:
        OS << format("0x%02" PRIx64, Value);
        break;
      case 2:
        OS << format("0x%04" PRIx64, Value);
        break;
      case 4:
        OS << format("0x%08" PRIx64, Value);
        break;
      case 8:
        OS << format("0x%016" PRIx64, Value);
        break;
      }
    else
      OS << Value;
    return;
  }
  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*this);
    const MCSymbol &Sym = SRE.getSymbol();
    // Parenthesize names that start with $ so that they don't look like
    // absolute names.
    bool UseParens = MAI && MAI->useParensForDollarSignNames() && !InParens &&
                     Sym.getName().starts_with('$');

    if (UseParens) {
      OS << '(';
      Sym.print(OS, MAI);
      OS << ')';
    } else
      Sym.print(OS, MAI);

    const MCSymbolRefExpr::VariantKind Kind = SRE.getKind();
    if (Kind != MCSymbolRefExpr::VK_None) {
      if (!MAI) // should only be used by dump()
        OS << "@<variant " << Kind << '>';
      else if (MAI->useParensForSymbolVariant()) // ARM
        OS << '(' << MAI->getVariantKindName(Kind) << ')';
      else
        OS << '@' << MAI->getVariantKindName(Kind);
    }

    return;
  }

  case MCExpr::Unary: {
    const MCUnaryExpr &UE = cast<MCUnaryExpr>(*this);
    switch (UE.getOpcode()) {
    case MCUnaryExpr::LNot:  OS << '!'; break;
    case MCUnaryExpr::Minus: OS << '-'; break;
    case MCUnaryExpr::Not:   OS << '~'; break;
    case MCUnaryExpr::Plus:  OS << '+'; break;
    }
    bool Binary = UE.getSubExpr()->getKind() == MCExpr::Binary;
    if (Binary) OS << "(";
    UE.getSubExpr()->print(OS, MAI);
    if (Binary) OS << ")";
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getLHS()) || isa<MCSymbolRefExpr>(BE.getLHS())) {
      BE.getLHS()->print(OS, MAI);
    } else {
      OS << '(';
      BE.getLHS()->print(OS, MAI);
      OS << ')';
    }

    switch (BE.getOpcode()) {
    case MCBinaryExpr::Add:
      // Print "X-42" instead of "X+-42".
      if (const MCConstantExpr *RHSC = dyn_cast<MCConstantExpr>(BE.getRHS())) {
        if (RHSC->getValue() < 0) {
          OS << RHSC->getValue();
          return;
        }
      }

      OS <<  '+';
      break;
    case MCBinaryExpr::AShr: OS << ">>"; break;
    case MCBinaryExpr::And:  OS <<  '&'; break;
    case MCBinaryExpr::Div:  OS <<  '/'; break;
    case MCBinaryExpr::EQ:   OS << "=="; break;
    case MCBinaryExpr::GT:   OS <<  '>'; break;
    case MCBinaryExpr::GTE:  OS << ">="; break;
    case MCBinaryExpr::LAnd: OS << "&&"; break;
    case MCBinaryExpr::LOr:  OS << "||"; break;
    case MCBinaryExpr::LShr: OS << ">>"; break;
    case MCBinaryExpr::LT:   OS <<  '<'; break;
    case MCBinaryExpr::LTE:  OS << "<="; break;
    case MCBinaryExpr::Mod:  OS <<  '%'; break;
    case MCBinaryExpr::Mul:  OS <<  '*'; break;
    case MCBinaryExpr::NE:   OS << "!="; break;
    case MCBinaryExpr::Or:   OS <<  '|'; break;
    case MCBinaryExpr::OrNot: OS << '!'; break;
    case MCBinaryExpr::Shl:  OS << "<<"; break;
    case MCBinaryExpr::Sub:  OS <<  '-'; break;
    case MCBinaryExpr::Xor:  OS <<  '^'; break;
    }

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getRHS()) || isa<MCSymbolRefExpr>(BE.getRHS())) {
      BE.getRHS()->print(OS, MAI);
    } else {
      OS << '(';
      BE.getRHS()->print(OS, MAI);
      OS << ')';
    }
    return;
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCExpr::dump() const {
  dbgs() << *this;
  dbgs() << '\n';
}
#endif

bool MCExpr::isSymbolUsedInExpression(const MCSymbol *Sym) const {
  switch (getKind()) {
  case MCExpr::Binary: {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(this);
    return BE->getLHS()->isSymbolUsedInExpression(Sym) ||
           BE->getRHS()->isSymbolUsedInExpression(Sym);
  }
  case MCExpr::Target: {
    const MCTargetExpr *TE = static_cast<const MCTargetExpr *>(this);
    return TE->isSymbolUsedInExpression(Sym);
  }
  case MCExpr::Constant:
    return false;
  case MCExpr::SymbolRef: {
    const MCSymbol &S = static_cast<const MCSymbolRefExpr *>(this)->getSymbol();
    if (S.isVariable() && !S.isWeakExternal())
      return S.getVariableValue()->isSymbolUsedInExpression(Sym);
    return &S == Sym;
  }
  case MCExpr::Unary: {
    const MCExpr *SubExpr =
        static_cast<const MCUnaryExpr *>(this)->getSubExpr();
    return SubExpr->isSymbolUsedInExpression(Sym);
  }
  }

  llvm_unreachable("Unknown expr kind!");
}

/* *** */

const MCBinaryExpr *MCBinaryExpr::create(Opcode Opc, const MCExpr *LHS,
                                         const MCExpr *RHS, MCContext &Ctx,
                                         SMLoc Loc) {
  return new (Ctx) MCBinaryExpr(Opc, LHS, RHS, Loc);
}

const MCUnaryExpr *MCUnaryExpr::create(Opcode Opc, const MCExpr *Expr,
                                       MCContext &Ctx, SMLoc Loc) {
  return new (Ctx) MCUnaryExpr(Opc, Expr, Loc);
}

const MCConstantExpr *MCConstantExpr::create(int64_t Value, MCContext &Ctx,
                                             bool PrintInHex,
                                             unsigned SizeInBytes) {
  return new (Ctx) MCConstantExpr(Value, PrintInHex, SizeInBytes);
}

/* *** */

MCSymbolRefExpr::MCSymbolRefExpr(const MCSymbol *Symbol, VariantKind Kind,
                                 const MCAsmInfo *MAI, SMLoc Loc)
    : MCExpr(MCExpr::SymbolRef, Loc,
             encodeSubclassData(Kind, MAI->hasSubsectionsViaSymbols())),
      Symbol(Symbol) {
  assert(Symbol);
}

const MCSymbolRefExpr *MCSymbolRefExpr::create(const MCSymbol *Sym,
                                               VariantKind Kind,
                                               MCContext &Ctx, SMLoc Loc) {
  return new (Ctx) MCSymbolRefExpr(Sym, Kind, Ctx.getAsmInfo(), Loc);
}

const MCSymbolRefExpr *MCSymbolRefExpr::create(StringRef Name, VariantKind Kind,
                                               MCContext &Ctx) {
  return create(Ctx.getOrCreateSymbol(Name), Kind, Ctx);
}

/* *** */

void MCTargetExpr::anchor() {}

/* *** */

bool MCExpr::evaluateAsAbsolute(int64_t &Res) const {
  return evaluateAsAbsolute(Res, nullptr, nullptr, false);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm,
                                const SectionAddrMap &Addrs) const {
  // Setting InSet causes us to absolutize differences across sections and that
  // is what the MachO writer uses Addrs for.
  return evaluateAsAbsolute(Res, &Asm, &Addrs, true);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm) const {
  return evaluateAsAbsolute(Res, &Asm, nullptr, false);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm) const {
  return evaluateAsAbsolute(Res, Asm, nullptr, false);
}

bool MCExpr::evaluateKnownAbsolute(int64_t &Res, const MCAssembler &Asm) const {
  return evaluateAsAbsolute(Res, &Asm, nullptr, true);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm,
                                const SectionAddrMap *Addrs, bool InSet) const {
  MCValue Value;

  // Fast path constants.
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(this)) {
    Res = CE->getValue();
    return true;
  }

  bool IsRelocatable =
      evaluateAsRelocatableImpl(Value, Asm, nullptr, Addrs, InSet);

  // Record the current value.
  Res = Value.getConstant();

  return IsRelocatable && Value.isAbsolute();
}

/// Helper method for \see EvaluateSymbolAdd().
static void AttemptToFoldSymbolOffsetDifference(
    const MCAssembler *Asm, const SectionAddrMap *Addrs, bool InSet,
    const MCSymbolRefExpr *&A, const MCSymbolRefExpr *&B, int64_t &Addend) {
  if (!A || !B)
    return;

  const MCSymbol &SA = A->getSymbol();
  const MCSymbol &SB = B->getSymbol();

  if (SA.isUndefined() || SB.isUndefined())
    return;

  if (!Asm->getWriter().isSymbolRefDifferenceFullyResolved(*Asm, A, B, InSet))
    return;

  auto FinalizeFolding = [&]() {
    // Pointers to Thumb symbols need to have their low-bit set to allow
    // for interworking.
    if (Asm->isThumbFunc(&SA))
      Addend |= 1;

    // Clear the symbol expr pointers to indicate we have folded these
    // operands.
    A = B = nullptr;
  };

  const MCFragment *FA = SA.getFragment();
  const MCFragment *FB = SB.getFragment();
  const MCSection &SecA = *FA->getParent();
  const MCSection &SecB = *FB->getParent();
  if ((&SecA != &SecB) && !Addrs)
    return;

  // When layout is available, we can generally compute the difference using the
  // getSymbolOffset path, which also avoids the possible slow fragment walk.
  // However, linker relaxation may cause incorrect fold of A-B if A and B are
  // separated by a linker-relaxable instruction. If the section contains
  // instructions and InSet is false (not expressions in directive like
  // .size/.fill), disable the fast path.
  bool Layout = Asm->hasLayout();
  if (Layout && (InSet || !SecA.hasInstructions() ||
                 !Asm->getBackend().allowLinkerRelaxation())) {
    // If both symbols are in the same fragment, return the difference of their
    // offsets. canGetFragmentOffset(FA) may be false.
    if (FA == FB && !SA.isVariable() && !SB.isVariable()) {
      Addend += SA.getOffset() - SB.getOffset();
      return FinalizeFolding();
    }

    // Eagerly evaluate when layout is finalized.
    Addend += Asm->getSymbolOffset(A->getSymbol()) -
              Asm->getSymbolOffset(B->getSymbol());
    if (Addrs && (&SecA != &SecB))
      Addend += (Addrs->lookup(&SecA) - Addrs->lookup(&SecB));

    FinalizeFolding();
  } else {
    // When layout is not finalized, our ability to resolve differences between
    // symbols is limited to specific cases where the fragments between two
    // symbols (including the fragments the symbols are defined in) are
    // fixed-size fragments so the difference can be calculated. For example,
    // this is important when the Subtarget is changed and a new MCDataFragment
    // is created in the case of foo: instr; .arch_extension ext; instr .if . -
    // foo.
    if (SA.isVariable() || SB.isVariable())
      return;

    // Try to find a constant displacement from FA to FB, add the displacement
    // between the offset in FA of SA and the offset in FB of SB.
    bool Reverse = false;
    if (FA == FB)
      Reverse = SA.getOffset() < SB.getOffset();
    else
      Reverse = FA->getLayoutOrder() < FB->getLayoutOrder();

    uint64_t SAOffset = SA.getOffset(), SBOffset = SB.getOffset();
    int64_t Displacement = SA.getOffset() - SB.getOffset();
    if (Reverse) {
      std::swap(FA, FB);
      std::swap(SAOffset, SBOffset);
      Displacement *= -1;
    }

    // Track whether B is before a relaxable instruction and whether A is after
    // a relaxable instruction. If SA and SB are separated by a linker-relaxable
    // instruction, the difference cannot be resolved as it may be changed by
    // the linker.
    bool BBeforeRelax = false, AAfterRelax = false;
    for (auto FI = FB; FI; FI = FI->getNext()) {
      auto DF = dyn_cast<MCDataFragment>(FI);
      if (DF && DF->isLinkerRelaxable()) {
        if (&*FI != FB || SBOffset != DF->getContents().size())
          BBeforeRelax = true;
        if (&*FI != FA || SAOffset == DF->getContents().size())
          AAfterRelax = true;
        if (BBeforeRelax && AAfterRelax)
          return;
      }
      if (&*FI == FA) {
        // If FA and FB belong to the same subsection, the loop will find FA and
        // we can resolve the difference.
        Addend += Reverse ? -Displacement : Displacement;
        FinalizeFolding();
        return;
      }

      int64_t Num;
      unsigned Count;
      if (DF) {
        Displacement += DF->getContents().size();
      } else if (auto *AF = dyn_cast<MCAlignFragment>(FI);
                 AF && Layout && AF->hasEmitNops() &&
                 !Asm->getBackend().shouldInsertExtraNopBytesForCodeAlign(
                     *AF, Count)) {
        Displacement += Asm->computeFragmentSize(*AF);
      } else if (auto *FF = dyn_cast<MCFillFragment>(FI);
                 FF && FF->getNumValues().evaluateAsAbsolute(Num)) {
        Displacement += Num * FF->getValueSize();
      } else {
        return;
      }
    }
  }
}

/// Evaluate the result of an add between (conceptually) two MCValues.
///
/// This routine conceptually attempts to construct an MCValue:
///   Result = (Result_A - Result_B + Result_Cst)
/// from two MCValue's LHS and RHS where
///   Result = LHS + RHS
/// and
///   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
///
/// This routine attempts to aggressively fold the operands such that the result
/// is representable in an MCValue, but may not always succeed.
///
/// \returns True on success, false if the result is not representable in an
/// MCValue.

/// NOTE: It is really important to have both the Asm and Layout arguments.
/// They might look redundant, but this function can be used before layout
/// is done (see the object streamer for example) and having the Asm argument
/// lets us avoid relaxations early.
static bool evaluateSymbolicAdd(const MCAssembler *Asm,
                                const SectionAddrMap *Addrs, bool InSet,
                                const MCValue &LHS, const MCValue &RHS,
                                MCValue &Res) {
  // FIXME: This routine (and other evaluation parts) are *incredibly* sloppy
  // about dealing with modifiers. This will ultimately bite us, one day.
  const MCSymbolRefExpr *LHS_A = LHS.getSymA();
  const MCSymbolRefExpr *LHS_B = LHS.getSymB();
  int64_t LHS_Cst = LHS.getConstant();

  const MCSymbolRefExpr *RHS_A = RHS.getSymA();
  const MCSymbolRefExpr *RHS_B = RHS.getSymB();
  int64_t RHS_Cst = RHS.getConstant();

  if (LHS.getRefKind() != RHS.getRefKind())
    return false;

  // Fold the result constant immediately.
  int64_t Result_Cst = LHS_Cst + RHS_Cst;

  // If we have a layout, we can fold resolved differences.
  if (Asm) {
    // First, fold out any differences which are fully resolved. By
    // reassociating terms in
    //   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
    // we have the four possible differences:
    //   (LHS_A - LHS_B),
    //   (LHS_A - RHS_B),
    //   (RHS_A - LHS_B),
    //   (RHS_A - RHS_B).
    // Since we are attempting to be as aggressive as possible about folding, we
    // attempt to evaluate each possible alternative.
    AttemptToFoldSymbolOffsetDifference(Asm, Addrs, InSet, LHS_A, LHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Addrs, InSet, LHS_A, RHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Addrs, InSet, RHS_A, LHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Addrs, InSet, RHS_A, RHS_B,
                                        Result_Cst);
  }

  // We can't represent the addition or subtraction of two symbols.
  if ((LHS_A && RHS_A) || (LHS_B && RHS_B))
    return false;

  // At this point, we have at most one additive symbol and one subtractive
  // symbol -- find them.
  const MCSymbolRefExpr *A = LHS_A ? LHS_A : RHS_A;
  const MCSymbolRefExpr *B = LHS_B ? LHS_B : RHS_B;

  Res = MCValue::get(A, B, Result_Cst);
  return true;
}

bool MCExpr::evaluateAsRelocatable(MCValue &Res, const MCAssembler *Asm,
                                   const MCFixup *Fixup) const {
  return evaluateAsRelocatableImpl(Res, Asm, Fixup, nullptr, false);
}

bool MCExpr::evaluateAsValue(MCValue &Res, const MCAssembler &Asm) const {
  return evaluateAsRelocatableImpl(Res, &Asm, nullptr, nullptr, true);
}

static bool canExpand(const MCSymbol &Sym, bool InSet) {
  if (Sym.isWeakExternal())
    return false;

  const MCExpr *Expr = Sym.getVariableValue();
  const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr);
  if (Inner) {
    if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF)
      return false;
  }

  if (InSet)
    return true;
  return !Sym.isInSection();
}

bool MCExpr::evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                       const MCFixup *Fixup,
                                       const SectionAddrMap *Addrs,
                                       bool InSet) const {
  ++stats::MCExprEvaluate;
  switch (getKind()) {
  case Target:
    return cast<MCTargetExpr>(this)->evaluateAsRelocatableImpl(Res, Asm, Fixup);

  case Constant:
    Res = MCValue::get(cast<MCConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();
    const auto Kind = SRE->getKind();
    bool Layout = Asm && Asm->hasLayout();

    // Evaluate recursively if this is a variable.
    if (Sym.isVariable() && (Kind == MCSymbolRefExpr::VK_None || Layout) &&
        canExpand(Sym, InSet)) {
      bool IsMachO = SRE->hasSubsectionsViaSymbols();
      if (Sym.getVariableValue()->evaluateAsRelocatableImpl(
              Res, Asm, Fixup, Addrs, InSet || IsMachO)) {
        if (Kind != MCSymbolRefExpr::VK_None) {
          if (Res.isAbsolute()) {
            Res = MCValue::get(SRE, nullptr, 0);
            return true;
          }
          // If the reference has a variant kind, we can only handle expressions
          // which evaluate exactly to a single unadorned symbol. Attach the
          // original VariantKind to SymA of the result.
          if (Res.getRefKind() != MCSymbolRefExpr::VK_None || !Res.getSymA() ||
              Res.getSymB() || Res.getConstant())
            return false;
          Res =
              MCValue::get(MCSymbolRefExpr::create(&Res.getSymA()->getSymbol(),
                                                   Kind, Asm->getContext()),
                           Res.getSymB(), Res.getConstant(), Res.getRefKind());
        }
        if (!IsMachO)
          return true;

        const MCSymbolRefExpr *A = Res.getSymA();
        const MCSymbolRefExpr *B = Res.getSymB();
        // FIXME: This is small hack. Given
        // a = b + 4
        // .long a
        // the OS X assembler will completely drop the 4. We should probably
        // include it in the relocation or produce an error if that is not
        // possible.
        // Allow constant expressions.
        if (!A && !B)
          return true;
        // Allows aliases with zero offset.
        if (Res.getConstant() == 0 && (!A || !B))
          return true;
      }
    }

    Res = MCValue::get(SRE, nullptr, 0);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->evaluateAsRelocatableImpl(Value, Asm, Fixup, Addrs,
                                                      InSet))
      return false;

    switch (AUE->getOpcode()) {
    case MCUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case MCUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getSymA() && !Value.getSymB())
        return false;

      // The cast avoids undefined behavior if the constant is INT64_MIN.
      Res = MCValue::get(Value.getSymB(), Value.getSymA(),
                         -(uint64_t)Value.getConstant());
      break;
    case MCUnaryExpr::Not:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(~Value.getConstant());
      break;
    case MCUnaryExpr::Plus:
      Res = Value;
      break;
    }

    return true;
  }

  case Binary: {
    const MCBinaryExpr *ABE = cast<MCBinaryExpr>(this);
    MCValue LHSValue, RHSValue;

    if (!ABE->getLHS()->evaluateAsRelocatableImpl(LHSValue, Asm, Fixup, Addrs,
                                                  InSet) ||
        !ABE->getRHS()->evaluateAsRelocatableImpl(RHSValue, Asm, Fixup, Addrs,
                                                  InSet)) {
      // Check if both are Target Expressions, see if we can compare them.
      if (const MCTargetExpr *L = dyn_cast<MCTargetExpr>(ABE->getLHS())) {
        if (const MCTargetExpr *R = dyn_cast<MCTargetExpr>(ABE->getRHS())) {
          switch (ABE->getOpcode()) {
          case MCBinaryExpr::EQ:
            Res = MCValue::get(L->isEqualTo(R) ? -1 : 0);
            return true;
          case MCBinaryExpr::NE:
            Res = MCValue::get(L->isEqualTo(R) ? 0 : -1);
            return true;
          default:
            break;
          }
        }
      }
      return false;
    }

    // We only support a few operations on non-constant expressions, handle
    // those first.
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (ABE->getOpcode()) {
      default:
        return false;
      case MCBinaryExpr::Sub:
        // Negate RHS and add.
        // The cast avoids undefined behavior if the constant is INT64_MIN.
        return evaluateSymbolicAdd(
            Asm, Addrs, InSet, LHSValue,
            MCValue::get(RHSValue.getSymB(), RHSValue.getSymA(),
                         -(uint64_t)RHSValue.getConstant(),
                         RHSValue.getRefKind()),
            Res);

      case MCBinaryExpr::Add:
        return evaluateSymbolicAdd(
            Asm, Addrs, InSet, LHSValue,
            MCValue::get(RHSValue.getSymA(), RHSValue.getSymB(),
                         RHSValue.getConstant(), RHSValue.getRefKind()),
            Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons differently from
    // Apple as.
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    int64_t Result = 0;
    auto Op = ABE->getOpcode();
    switch (Op) {
    case MCBinaryExpr::AShr: Result = LHS >> RHS; break;
    case MCBinaryExpr::Add:  Result = LHS + RHS; break;
    case MCBinaryExpr::And:  Result = LHS & RHS; break;
    case MCBinaryExpr::Div:
    case MCBinaryExpr::Mod:
      // Handle division by zero. gas just emits a warning and keeps going,
      // we try to be stricter.
      // FIXME: Currently the caller of this function has no way to understand
      // we're bailing out because of 'division by zero'. Therefore, it will
      // emit a 'expected relocatable expression' error. It would be nice to
      // change this code to emit a better diagnostic.
      if (RHS == 0)
        return false;
      if (ABE->getOpcode() == MCBinaryExpr::Div)
        Result = LHS / RHS;
      else
        Result = LHS % RHS;
      break;
    case MCBinaryExpr::EQ:   Result = LHS == RHS; break;
    case MCBinaryExpr::GT:   Result = LHS > RHS; break;
    case MCBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case MCBinaryExpr::LAnd: Result = LHS && RHS; break;
    case MCBinaryExpr::LOr:  Result = LHS || RHS; break;
    case MCBinaryExpr::LShr: Result = uint64_t(LHS) >> uint64_t(RHS); break;
    case MCBinaryExpr::LT:   Result = LHS < RHS; break;
    case MCBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case MCBinaryExpr::Mul:  Result = LHS * RHS; break;
    case MCBinaryExpr::NE:   Result = LHS != RHS; break;
    case MCBinaryExpr::Or:   Result = LHS | RHS; break;
    case MCBinaryExpr::OrNot: Result = LHS | ~RHS; break;
    case MCBinaryExpr::Shl:  Result = uint64_t(LHS) << uint64_t(RHS); break;
    case MCBinaryExpr::Sub:  Result = LHS - RHS; break;
    case MCBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    switch (Op) {
    default:
      Res = MCValue::get(Result);
      break;
    case MCBinaryExpr::EQ:
    case MCBinaryExpr::GT:
    case MCBinaryExpr::GTE:
    case MCBinaryExpr::LT:
    case MCBinaryExpr::LTE:
    case MCBinaryExpr::NE:
      // A comparison operator returns a -1 if true and 0 if false.
      Res = MCValue::get(Result ? -1 : 0);
      break;
    }

    return true;
  }
  }

  llvm_unreachable("Invalid assembly expression kind!");
}

MCFragment *MCExpr::findAssociatedFragment() const {
  switch (getKind()) {
  case Target:
    // We never look through target specific expressions.
    return cast<MCTargetExpr>(this)->findAssociatedFragment();

  case Constant:
    return MCSymbol::AbsolutePseudoFragment;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();
    return Sym.getFragment();
  }

  case Unary:
    return cast<MCUnaryExpr>(this)->getSubExpr()->findAssociatedFragment();

  case Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(this);
    MCFragment *LHS_F = BE->getLHS()->findAssociatedFragment();
    MCFragment *RHS_F = BE->getRHS()->findAssociatedFragment();

    // If either is absolute, return the other.
    if (LHS_F == MCSymbol::AbsolutePseudoFragment)
      return RHS_F;
    if (RHS_F == MCSymbol::AbsolutePseudoFragment)
      return LHS_F;

    // Not always correct, but probably the best we can do without more context.
    if (BE->getOpcode() == MCBinaryExpr::Sub)
      return MCSymbol::AbsolutePseudoFragment;

    // Otherwise, return the first non-null fragment.
    return LHS_F ? LHS_F : RHS_F;
  }
  }

  llvm_unreachable("Invalid assembly expression kind!");
}
