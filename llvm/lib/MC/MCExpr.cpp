//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Statistic.h"
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

static int getPrecedence(MCBinaryExpr::Opcode Op) {
  switch (Op) {
  case MCBinaryExpr::Add:
  case MCBinaryExpr::Sub:
    return 1;
  default:
    return 0;
  }
}

// VariantKind printing and formatting utilize MAI. operator<< (dump and some
// target code) specifies MAI as nullptr and should be avoided when MAI is
// needed.
void MCExpr::print(raw_ostream &OS, const MCAsmInfo *MAI,
                   int SurroundingPrec) const {
  constexpr int MaxPrec = 9;
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
    Sym.print(OS, MAI);

    const MCSymbolRefExpr::VariantKind Kind = SRE.getKind();
    if (Kind != MCSymbolRefExpr::VK_None) {
      if (!MAI) // should only be used by dump()
        OS << "@<variant " << Kind << '>';
      else if (MAI->useParensForSpecifier()) // ARM
        OS << '(' << MAI->getSpecifierName(Kind) << ')';
      else
        OS << '@' << MAI->getSpecifierName(Kind);
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
    UE.getSubExpr()->print(OS, MAI, MaxPrec);
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);
    // We want to avoid redundant parentheses for relocatable expressions like
    // a-b+c.
    //
    // Print '(' if the current operator has lower precedence than the
    // surrounding operator, or if the surrounding operator's precedence is
    // unknown (set to HighPrecedence).
    int Prec = getPrecedence(BE.getOpcode());
    bool Paren = Prec < SurroundingPrec;
    if (Paren)
      OS << '(';
    // Many operators' precedence is different from C. Set the precedence to
    // HighPrecedence for unknown operators.
    int SubPrec = Prec ? Prec : MaxPrec;
    BE.getLHS()->print(OS, MAI, SubPrec);

    switch (BE.getOpcode()) {
    case MCBinaryExpr::Add:
      // Print "X-42" instead of "X+-42".
      if (const MCConstantExpr *RHSC = dyn_cast<MCConstantExpr>(BE.getRHS())) {
        if (RHSC->getValue() < 0) {
          OS << RHSC->getValue();
          if (Paren)
            OS << ')';
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

    BE.getRHS()->print(OS, MAI, SubPrec + 1);
    if (Paren)
      OS << ')';
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
    : MCExpr(MCExpr::SymbolRef, Loc, Kind), Symbol(Symbol) {
  assert(Symbol);
}

const MCSymbolRefExpr *MCSymbolRefExpr::create(const MCSymbol *Sym,
                                               VariantKind Kind,
                                               MCContext &Ctx, SMLoc Loc) {
  return new (Ctx) MCSymbolRefExpr(Sym, Kind, Ctx.getAsmInfo(), Loc);
}

/* *** */

void MCTargetExpr::anchor() {}

/* *** */

bool MCExpr::evaluateAsAbsolute(int64_t &Res) const {
  return evaluateAsAbsolute(Res, nullptr, false);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm) const {
  return evaluateAsAbsolute(Res, &Asm, false);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm) const {
  return evaluateAsAbsolute(Res, Asm, false);
}

bool MCExpr::evaluateKnownAbsolute(int64_t &Res, const MCAssembler &Asm) const {
  return evaluateAsAbsolute(Res, &Asm, true);
}

bool MCExpr::evaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm,
                                bool InSet) const {
  MCValue Value;

  // Fast path constants.
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(this)) {
    Res = CE->getValue();
    return true;
  }

  bool IsRelocatable = evaluateAsRelocatableImpl(Value, Asm, InSet);
  Res = Value.getConstant();
  // Value with RefKind (e.g. %hi(0xdeadbeef) in MIPS) is not considered
  // absolute (the value is unknown at parse time), even if it might be resolved
  // by evaluateFixup.
  return IsRelocatable && Value.isAbsolute() && Value.getSpecifier() == 0;
}

/// Helper method for \see EvaluateSymbolAdd().
static void attemptToFoldSymbolOffsetDifference(const MCAssembler *Asm,
                                                bool InSet, const MCSymbol *&A,
                                                const MCSymbol *&B,
                                                int64_t &Addend) {
  if (!A || !B)
    return;

  const MCSymbol &SA = *A, &SB = *B;
  if (SA.isUndefined() || SB.isUndefined())
    return;
  if (!Asm->getWriter().isSymbolRefDifferenceFullyResolved(SA, SB, InSet))
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
  if (&SecA != &SecB)
    return;

  // When layout is available, we can generally compute the difference using the
  // getSymbolOffset path, which also avoids the possible slow fragment walk.
  // However, linker relaxation may cause incorrect fold of A-B if A and B are
  // separated by a linker-relaxable fragment. If the section contains
  // linker-relaxable instruction and InSet is false (not expressions in
  // directive like .size/.fill), disable the fast path.
  bool Layout = Asm->hasLayout();
  if (Layout && (InSet || !SecA.isLinkerRelaxable())) {
    // If both symbols are in the same fragment, return the difference of their
    // offsets. canGetFragmentOffset(FA) may be false.
    if (FA == FB && !SA.isVariable() && !SB.isVariable()) {
      Addend += SA.getOffset() - SB.getOffset();
      return FinalizeFolding();
    }

    // Eagerly evaluate when layout is finalized.
    Addend += Asm->getSymbolOffset(SA) - Asm->getSymbolOffset(SB);
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
      } else if (auto *RF = dyn_cast<MCRelaxableFragment>(FI);
                 RF && Asm->hasFinalLayout()) {
        // Before finishLayout, a relaxable fragment's size is indeterminate.
        // After layout, during relocation generation, it can be treated as a
        // data fragment.
        Displacement += RF->getContents().size();
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

// Evaluate the sum of two relocatable expressions.
//
//   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
//
// This routine attempts to aggressively fold the operands such that the result
// is representable in an MCValue, but may not always succeed.
//
// LHS_A and RHS_A might have relocation specifiers while LHS_B and RHS_B
// cannot have specifiers.
//
// \returns True on success, false if the result is not representable in an
// MCValue.

// NOTE: This function can be used before layout is done (see the object
// streamer for example) and having the Asm argument lets us avoid relaxations
// early.
bool MCExpr::evaluateSymbolicAdd(const MCAssembler *Asm, bool InSet,
                                 const MCValue &LHS, const MCValue &RHS,
                                 MCValue &Res) {
  const MCSymbol *LHS_A = LHS.getAddSym();
  const MCSymbol *LHS_B = LHS.getSubSym();
  int64_t LHS_Cst = LHS.getConstant();

  const MCSymbol *RHS_A = RHS.getAddSym();
  const MCSymbol *RHS_B = RHS.getSubSym();
  int64_t RHS_Cst = RHS.getConstant();

  // Fold the result constant immediately.
  int64_t Result_Cst = LHS_Cst + RHS_Cst;

  // If we have a layout, we can fold resolved differences.
  if (Asm && !LHS.getSpecifier() && !RHS.getSpecifier()) {
    // While LHS_A-LHS_B and RHS_A-RHS_B from recursive calls have already been
    // folded, reassociating terms in
    //   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
    // might bring more opportunities.
    if (LHS_A && RHS_B) {
      attemptToFoldSymbolOffsetDifference(Asm, InSet, LHS_A, RHS_B, Result_Cst);
    }
    if (RHS_A && LHS_B) {
      attemptToFoldSymbolOffsetDifference(Asm, InSet, RHS_A, LHS_B, Result_Cst);
    }
  }

  // We can't represent the addition or subtraction of two symbols.
  if ((LHS_A && RHS_A) || (LHS_B && RHS_B))
    return false;

  // At this point, we have at most one additive symbol and one subtractive
  // symbol -- find them.
  auto *A = LHS_A ? LHS_A : RHS_A;
  auto *B = LHS_B ? LHS_B : RHS_B;
  auto Spec = LHS.getSpecifier();
  if (!Spec)
    Spec = RHS.getSpecifier();
  Res = MCValue::get(A, B, Result_Cst, Spec);
  return true;
}

bool MCExpr::evaluateAsRelocatable(MCValue &Res, const MCAssembler *Asm) const {
  return evaluateAsRelocatableImpl(Res, Asm, false);
}
bool MCExpr::evaluateAsValue(MCValue &Res, const MCAssembler &Asm) const {
  return evaluateAsRelocatableImpl(Res, &Asm, true);
}
static bool canExpand(const MCSymbol &Sym, bool InSet) {
  if (Sym.isWeakExternal())
    return false;

  Sym.getVariableValue(true);

  if (InSet)
    return true;
  return !Sym.isInSection();
}

bool MCExpr::evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                       bool InSet) const {
  ++stats::MCExprEvaluate;
  switch (getKind()) {
  case Target:
    return cast<MCTargetExpr>(this)->evaluateAsRelocatableImpl(Res, Asm);
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
      bool IsMachO =
          Asm && Asm->getContext().getAsmInfo()->hasSubsectionsViaSymbols();
      if (Sym.getVariableValue()->evaluateAsRelocatableImpl(Res, Asm,
                                                            InSet || IsMachO)) {
        if (Kind != MCSymbolRefExpr::VK_None) {
          if (Res.isAbsolute()) {
            Res = MCValue::get(&Sym, nullptr, 0, Kind);
            return true;
          }
          // If the reference has a variant kind, we can only handle expressions
          // which evaluate exactly to a single unadorned symbol. Attach the
          // original VariantKind to SymA of the result.
          if (Res.getSpecifier() != MCSymbolRefExpr::VK_None ||
              !Res.getAddSym() || Res.getSubSym() || Res.getConstant())
            return false;
          Res.Specifier = Kind;
        }
        if (!IsMachO)
          return true;

        auto *A = Res.getAddSym();
        auto *B = Res.getSubSym();
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

    Res = MCValue::get(&Sym, nullptr, 0, Kind);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->evaluateAsRelocatableImpl(Value, Asm, InSet))
      return false;
    switch (AUE->getOpcode()) {
    case MCUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case MCUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getAddSym() && !Value.getSubSym())
        return false;

      // The cast avoids undefined behavior if the constant is INT64_MIN.
      Res = MCValue::get(Value.getSubSym(), Value.getAddSym(),
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

    if (!ABE->getLHS()->evaluateAsRelocatableImpl(LHSValue, Asm, InSet) ||
        !ABE->getRHS()->evaluateAsRelocatableImpl(RHSValue, Asm, InSet)) {
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
    auto Op = ABE->getOpcode();
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (Op) {
      default:
        return false;
      case MCBinaryExpr::Add:
      case MCBinaryExpr::Sub:
        if (Op == MCBinaryExpr::Sub) {
          std::swap(RHSValue.SymA, RHSValue.SymB);
          RHSValue.Cst = -(uint64_t)RHSValue.Cst;
        }
        if (RHSValue.isAbsolute()) {
          LHSValue.Cst += RHSValue.Cst;
          Res = LHSValue;
          return true;
        }
        if (LHSValue.isAbsolute()) {
          RHSValue.Cst += LHSValue.Cst;
          Res = RHSValue;
          return true;
        }
        if (LHSValue.SymB && LHSValue.Specifier)
          return false;
        if (RHSValue.SymB && RHSValue.Specifier)
          return false;
        return evaluateSymbolicAdd(Asm, InSet, LHSValue, RHSValue, Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons differently from
    // Apple as.
    int64_t Result = 0;
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
