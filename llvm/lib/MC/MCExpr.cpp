//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mcexpr"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
namespace stats {
STATISTIC(MCExprEvaluate, "Number of MCExpr evaluations");
}
}

void MCExpr::print(raw_ostream &OS) const {
  switch (getKind()) {
  case MCExpr::Target:
    return cast<MCTargetExpr>(this)->PrintImpl(OS);
  case MCExpr::Constant:
    OS << cast<MCConstantExpr>(*this).getValue();
    return;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*this);
    const MCSymbol &Sym = SRE.getSymbol();
    // Parenthesize names that start with $ so that they don't look like
    // absolute names.
    bool UseParens = Sym.getName()[0] == '$';

    if (SRE.getKind() == MCSymbolRefExpr::VK_PPC_DARWIN_HA16 ||
        SRE.getKind() == MCSymbolRefExpr::VK_PPC_DARWIN_LO16) {
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());
      UseParens = true;
    }

    if (UseParens)
      OS << '(' << Sym << ')';
    else
      OS << Sym;

    if (SRE.getKind() == MCSymbolRefExpr::VK_ARM_PLT ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TLSGD ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOT ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOTOFF ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TPOFF ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOTTPOFF ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TARGET1 ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TARGET2)
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());
    else if (SRE.getKind() != MCSymbolRefExpr::VK_None &&
             SRE.getKind() != MCSymbolRefExpr::VK_PPC_DARWIN_HA16 &&
             SRE.getKind() != MCSymbolRefExpr::VK_PPC_DARWIN_LO16)
      OS << '@' << MCSymbolRefExpr::getVariantKindName(SRE.getKind());

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
    OS << *UE.getSubExpr();
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getLHS()) || isa<MCSymbolRefExpr>(BE.getLHS())) {
      OS << *BE.getLHS();
    } else {
      OS << '(' << *BE.getLHS() << ')';
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
    case MCBinaryExpr::And:  OS <<  '&'; break;
    case MCBinaryExpr::Div:  OS <<  '/'; break;
    case MCBinaryExpr::EQ:   OS << "=="; break;
    case MCBinaryExpr::GT:   OS <<  '>'; break;
    case MCBinaryExpr::GTE:  OS << ">="; break;
    case MCBinaryExpr::LAnd: OS << "&&"; break;
    case MCBinaryExpr::LOr:  OS << "||"; break;
    case MCBinaryExpr::LT:   OS <<  '<'; break;
    case MCBinaryExpr::LTE:  OS << "<="; break;
    case MCBinaryExpr::Mod:  OS <<  '%'; break;
    case MCBinaryExpr::Mul:  OS <<  '*'; break;
    case MCBinaryExpr::NE:   OS << "!="; break;
    case MCBinaryExpr::Or:   OS <<  '|'; break;
    case MCBinaryExpr::Shl:  OS << "<<"; break;
    case MCBinaryExpr::Shr:  OS << ">>"; break;
    case MCBinaryExpr::Sub:  OS <<  '-'; break;
    case MCBinaryExpr::Xor:  OS <<  '^'; break;
    }

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getRHS()) || isa<MCSymbolRefExpr>(BE.getRHS())) {
      OS << *BE.getRHS();
    } else {
      OS << '(' << *BE.getRHS() << ')';
    }
    return;
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCExpr::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif

/* *** */

const MCBinaryExpr *MCBinaryExpr::Create(Opcode Opc, const MCExpr *LHS,
                                         const MCExpr *RHS, MCContext &Ctx) {
  return new (Ctx) MCBinaryExpr(Opc, LHS, RHS);
}

const MCUnaryExpr *MCUnaryExpr::Create(Opcode Opc, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) MCUnaryExpr(Opc, Expr);
}

const MCConstantExpr *MCConstantExpr::Create(int64_t Value, MCContext &Ctx) {
  return new (Ctx) MCConstantExpr(Value);
}

/* *** */

const MCSymbolRefExpr *MCSymbolRefExpr::Create(const MCSymbol *Sym,
                                               VariantKind Kind,
                                               MCContext &Ctx) {
  return new (Ctx) MCSymbolRefExpr(Sym, Kind);
}

const MCSymbolRefExpr *MCSymbolRefExpr::Create(StringRef Name, VariantKind Kind,
                                               MCContext &Ctx) {
  return Create(Ctx.GetOrCreateSymbol(Name), Kind, Ctx);
}

StringRef MCSymbolRefExpr::getVariantKindName(VariantKind Kind) {
  switch (Kind) {
  case VK_Invalid: return "<<invalid>>";
  case VK_None: return "<<none>>";

  case VK_GOT: return "GOT";
  case VK_GOTOFF: return "GOTOFF";
  case VK_GOTPCREL: return "GOTPCREL";
  case VK_GOTTPOFF: return "GOTTPOFF";
  case VK_INDNTPOFF: return "INDNTPOFF";
  case VK_NTPOFF: return "NTPOFF";
  case VK_GOTNTPOFF: return "GOTNTPOFF";
  case VK_PLT: return "PLT";
  case VK_TLSGD: return "TLSGD";
  case VK_TLSLD: return "TLSLD";
  case VK_TLSLDM: return "TLSLDM";
  case VK_TPOFF: return "TPOFF";
  case VK_DTPOFF: return "DTPOFF";
  case VK_TLVP: return "TLVP";
  case VK_SECREL: return "SECREL";
  case VK_ARM_PLT: return "(PLT)";
  case VK_ARM_GOT: return "(GOT)";
  case VK_ARM_GOTOFF: return "(GOTOFF)";
  case VK_ARM_TPOFF: return "(tpoff)";
  case VK_ARM_GOTTPOFF: return "(gottpoff)";
  case VK_ARM_TLSGD: return "(tlsgd)";
  case VK_ARM_TARGET1: return "(target1)";
  case VK_ARM_TARGET2: return "(target2)";
  case VK_PPC_TOC: return "tocbase";
  case VK_PPC_TOC_ENTRY: return "toc";
  case VK_PPC_DARWIN_HA16: return "ha16";
  case VK_PPC_DARWIN_LO16: return "lo16";
  case VK_PPC_GAS_HA16: return "ha";
  case VK_PPC_GAS_LO16: return "l";
  case VK_PPC_TPREL16_HA: return "tprel@ha";
  case VK_PPC_TPREL16_LO: return "tprel@l";
  case VK_Mips_GPREL: return "GPREL";
  case VK_Mips_GOT_CALL: return "GOT_CALL";
  case VK_Mips_GOT16: return "GOT16";
  case VK_Mips_GOT: return "GOT";
  case VK_Mips_ABS_HI: return "ABS_HI";
  case VK_Mips_ABS_LO: return "ABS_LO";
  case VK_Mips_TLSGD: return "TLSGD";
  case VK_Mips_TLSLDM: return "TLSLDM";
  case VK_Mips_DTPREL_HI: return "DTPREL_HI";
  case VK_Mips_DTPREL_LO: return "DTPREL_LO";
  case VK_Mips_GOTTPREL: return "GOTTPREL";
  case VK_Mips_TPREL_HI: return "TPREL_HI";
  case VK_Mips_TPREL_LO: return "TPREL_LO";
  case VK_Mips_GPOFF_HI: return "GPOFF_HI";
  case VK_Mips_GPOFF_LO: return "GPOFF_LO";
  case VK_Mips_GOT_DISP: return "GOT_DISP";
  case VK_Mips_GOT_PAGE: return "GOT_PAGE";
  case VK_Mips_GOT_OFST: return "GOT_OFST";
  case VK_Mips_HIGHER:   return "HIGHER";
  case VK_Mips_HIGHEST:  return "HIGHEST";
  case VK_Mips_GOT_HI16: return "GOT_HI16";
  case VK_Mips_GOT_LO16: return "GOT_LO16";
  case VK_Mips_CALL_HI16: return "CALL_HI16";
  case VK_Mips_CALL_LO16: return "CALL_LO16";
  }
  llvm_unreachable("Invalid variant kind");
}

MCSymbolRefExpr::VariantKind
MCSymbolRefExpr::getVariantKindForName(StringRef Name) {
  return StringSwitch<VariantKind>(Name)
    .Case("GOT", VK_GOT)
    .Case("got", VK_GOT)
    .Case("GOTOFF", VK_GOTOFF)
    .Case("gotoff", VK_GOTOFF)
    .Case("GOTPCREL", VK_GOTPCREL)
    .Case("gotpcrel", VK_GOTPCREL)
    .Case("GOTTPOFF", VK_GOTTPOFF)
    .Case("gottpoff", VK_GOTTPOFF)
    .Case("INDNTPOFF", VK_INDNTPOFF)
    .Case("indntpoff", VK_INDNTPOFF)
    .Case("NTPOFF", VK_NTPOFF)
    .Case("ntpoff", VK_NTPOFF)
    .Case("GOTNTPOFF", VK_GOTNTPOFF)
    .Case("gotntpoff", VK_GOTNTPOFF)
    .Case("PLT", VK_PLT)
    .Case("plt", VK_PLT)
    .Case("TLSGD", VK_TLSGD)
    .Case("tlsgd", VK_TLSGD)
    .Case("TLSLD", VK_TLSLD)
    .Case("tlsld", VK_TLSLD)
    .Case("TLSLDM", VK_TLSLDM)
    .Case("tlsldm", VK_TLSLDM)
    .Case("TPOFF", VK_TPOFF)
    .Case("tpoff", VK_TPOFF)
    .Case("DTPOFF", VK_DTPOFF)
    .Case("dtpoff", VK_DTPOFF)
    .Case("TLVP", VK_TLVP)
    .Case("tlvp", VK_TLVP)
    .Default(VK_Invalid);
}

/* *** */

void MCTargetExpr::anchor() {}

/* *** */

bool MCExpr::EvaluateAsAbsolute(int64_t &Res) const {
  return EvaluateAsAbsolute(Res, 0, 0, 0);
}

bool MCExpr::EvaluateAsAbsolute(int64_t &Res,
                                const MCAsmLayout &Layout) const {
  return EvaluateAsAbsolute(Res, &Layout.getAssembler(), &Layout, 0);
}

bool MCExpr::EvaluateAsAbsolute(int64_t &Res,
                                const MCAsmLayout &Layout,
                                const SectionAddrMap &Addrs) const {
  return EvaluateAsAbsolute(Res, &Layout.getAssembler(), &Layout, &Addrs);
}

bool MCExpr::EvaluateAsAbsolute(int64_t &Res, const MCAssembler &Asm) const {
  return EvaluateAsAbsolute(Res, &Asm, 0, 0);
}

bool MCExpr::EvaluateAsAbsolute(int64_t &Res, const MCAssembler *Asm,
                                const MCAsmLayout *Layout,
                                const SectionAddrMap *Addrs) const {
  MCValue Value;

  // Fast path constants.
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(this)) {
    Res = CE->getValue();
    return true;
  }

  // FIXME: The use if InSet = Addrs is a hack. Setting InSet causes us
  // absolutize differences across sections and that is what the MachO writer
  // uses Addrs for.
  bool IsRelocatable =
    EvaluateAsRelocatableImpl(Value, Asm, Layout, Addrs, /*InSet*/ Addrs);

  // Record the current value.
  Res = Value.getConstant();

  return IsRelocatable && Value.isAbsolute();
}

/// \brief Helper method for \see EvaluateSymbolAdd().
static void AttemptToFoldSymbolOffsetDifference(const MCAssembler *Asm,
                                                const MCAsmLayout *Layout,
                                                const SectionAddrMap *Addrs,
                                                bool InSet,
                                                const MCSymbolRefExpr *&A,
                                                const MCSymbolRefExpr *&B,
                                                int64_t &Addend) {
  if (!A || !B)
    return;

  const MCSymbol &SA = A->getSymbol();
  const MCSymbol &SB = B->getSymbol();

  if (SA.isUndefined() || SB.isUndefined())
    return;

  if (!Asm->getWriter().IsSymbolRefDifferenceFullyResolved(*Asm, A, B, InSet))
    return;

  MCSymbolData &AD = Asm->getSymbolData(SA);
  MCSymbolData &BD = Asm->getSymbolData(SB);

  if (AD.getFragment() == BD.getFragment()) {
    Addend += (AD.getOffset() - BD.getOffset());

    // Pointers to Thumb symbols need to have their low-bit set to allow
    // for interworking.
    if (Asm->isThumbFunc(&SA))
      Addend |= 1;

    // Clear the symbol expr pointers to indicate we have folded these
    // operands.
    A = B = 0;
    return;
  }

  if (!Layout)
    return;

  const MCSectionData &SecA = *AD.getFragment()->getParent();
  const MCSectionData &SecB = *BD.getFragment()->getParent();

  if ((&SecA != &SecB) && !Addrs)
    return;

  // Eagerly evaluate.
  Addend += (Layout->getSymbolOffset(&Asm->getSymbolData(A->getSymbol())) -
             Layout->getSymbolOffset(&Asm->getSymbolData(B->getSymbol())));
  if (Addrs && (&SecA != &SecB))
    Addend += (Addrs->lookup(&SecA) - Addrs->lookup(&SecB));

  // Pointers to Thumb symbols need to have their low-bit set to allow
  // for interworking.
  if (Asm->isThumbFunc(&SA))
    Addend |= 1;

  // Clear the symbol expr pointers to indicate we have folded these
  // operands.
  A = B = 0;
}

/// \brief Evaluate the result of an add between (conceptually) two MCValues.
///
/// This routine conceptually attempts to construct an MCValue:
///   Result = (Result_A - Result_B + Result_Cst)
/// from two MCValue's LHS and RHS where
///   Result = LHS + RHS
/// and
///   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
///
/// This routine attempts to aggresively fold the operands such that the result
/// is representable in an MCValue, but may not always succeed.
///
/// \returns True on success, false if the result is not representable in an
/// MCValue.

/// NOTE: It is really important to have both the Asm and Layout arguments.
/// They might look redundant, but this function can be used before layout
/// is done (see the object streamer for example) and having the Asm argument
/// lets us avoid relaxations early.
static bool EvaluateSymbolicAdd(const MCAssembler *Asm,
                                const MCAsmLayout *Layout,
                                const SectionAddrMap *Addrs,
                                bool InSet,
                                const MCValue &LHS,const MCSymbolRefExpr *RHS_A,
                                const MCSymbolRefExpr *RHS_B, int64_t RHS_Cst,
                                MCValue &Res) {
  // FIXME: This routine (and other evaluation parts) are *incredibly* sloppy
  // about dealing with modifiers. This will ultimately bite us, one day.
  const MCSymbolRefExpr *LHS_A = LHS.getSymA();
  const MCSymbolRefExpr *LHS_B = LHS.getSymB();
  int64_t LHS_Cst = LHS.getConstant();

  // Fold the result constant immediately.
  int64_t Result_Cst = LHS_Cst + RHS_Cst;

  assert((!Layout || Asm) &&
         "Must have an assembler object if layout is given!");

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
    AttemptToFoldSymbolOffsetDifference(Asm, Layout, Addrs, InSet, LHS_A, LHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Layout, Addrs, InSet, LHS_A, RHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Layout, Addrs, InSet, RHS_A, LHS_B,
                                        Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Asm, Layout, Addrs, InSet, RHS_A, RHS_B,
                                        Result_Cst);
  }

  // We can't represent the addition or subtraction of two symbols.
  if ((LHS_A && RHS_A) || (LHS_B && RHS_B))
    return false;

  // At this point, we have at most one additive symbol and one subtractive
  // symbol -- find them.
  const MCSymbolRefExpr *A = LHS_A ? LHS_A : RHS_A;
  const MCSymbolRefExpr *B = LHS_B ? LHS_B : RHS_B;

  // If we have a negated symbol, then we must have also have a non-negated
  // symbol in order to encode the expression.
  if (B && !A)
    return false;

  Res = MCValue::get(A, B, Result_Cst);
  return true;
}

bool MCExpr::EvaluateAsRelocatable(MCValue &Res,
                                   const MCAsmLayout &Layout) const {
  return EvaluateAsRelocatableImpl(Res, &Layout.getAssembler(), &Layout,
                                   0, false);
}

bool MCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                       const MCAssembler *Asm,
                                       const MCAsmLayout *Layout,
                                       const SectionAddrMap *Addrs,
                                       bool InSet) const {
  ++stats::MCExprEvaluate;

  switch (getKind()) {
  case Target:
    return cast<MCTargetExpr>(this)->EvaluateAsRelocatableImpl(Res, Layout);

  case Constant:
    Res = MCValue::get(cast<MCConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();

    // Evaluate recursively if this is a variable.
    if (Sym.isVariable() && SRE->getKind() == MCSymbolRefExpr::VK_None) {
      bool Ret = Sym.getVariableValue()->EvaluateAsRelocatableImpl(Res, Asm,
                                                                   Layout,
                                                                   Addrs,
                                                                   true);
      // If we failed to simplify this to a constant, let the target
      // handle it.
      if (Ret && !Res.getSymA() && !Res.getSymB())
        return true;
    }

    Res = MCValue::get(SRE, 0, 0);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->EvaluateAsRelocatableImpl(Value, Asm, Layout,
                                                      Addrs, InSet))
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
      Res = MCValue::get(Value.getSymB(), Value.getSymA(),
                         -Value.getConstant());
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

    if (!ABE->getLHS()->EvaluateAsRelocatableImpl(LHSValue, Asm, Layout,
                                                  Addrs, InSet) ||
        !ABE->getRHS()->EvaluateAsRelocatableImpl(RHSValue, Asm, Layout,
                                                  Addrs, InSet))
      return false;

    // We only support a few operations on non-constant expressions, handle
    // those first.
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (ABE->getOpcode()) {
      default:
        return false;
      case MCBinaryExpr::Sub:
        // Negate RHS and add.
        return EvaluateSymbolicAdd(Asm, Layout, Addrs, InSet, LHSValue,
                                   RHSValue.getSymB(), RHSValue.getSymA(),
                                   -RHSValue.getConstant(),
                                   Res);

      case MCBinaryExpr::Add:
        return EvaluateSymbolicAdd(Asm, Layout, Addrs, InSet, LHSValue,
                                   RHSValue.getSymA(), RHSValue.getSymB(),
                                   RHSValue.getConstant(),
                                   Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons and right shifts
    // differently from Apple as.
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    int64_t Result = 0;
    switch (ABE->getOpcode()) {
    case MCBinaryExpr::Add:  Result = LHS + RHS; break;
    case MCBinaryExpr::And:  Result = LHS & RHS; break;
    case MCBinaryExpr::Div:  Result = LHS / RHS; break;
    case MCBinaryExpr::EQ:   Result = LHS == RHS; break;
    case MCBinaryExpr::GT:   Result = LHS > RHS; break;
    case MCBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case MCBinaryExpr::LAnd: Result = LHS && RHS; break;
    case MCBinaryExpr::LOr:  Result = LHS || RHS; break;
    case MCBinaryExpr::LT:   Result = LHS < RHS; break;
    case MCBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case MCBinaryExpr::Mod:  Result = LHS % RHS; break;
    case MCBinaryExpr::Mul:  Result = LHS * RHS; break;
    case MCBinaryExpr::NE:   Result = LHS != RHS; break;
    case MCBinaryExpr::Or:   Result = LHS | RHS; break;
    case MCBinaryExpr::Shl:  Result = LHS << RHS; break;
    case MCBinaryExpr::Shr:  Result = LHS >> RHS; break;
    case MCBinaryExpr::Sub:  Result = LHS - RHS; break;
    case MCBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    Res = MCValue::get(Result);
    return true;
  }
  }

  llvm_unreachable("Invalid assembly expression kind!");
}

const MCSection *MCExpr::FindAssociatedSection() const {
  switch (getKind()) {
  case Target:
    // We never look through target specific expressions.
    return cast<MCTargetExpr>(this)->FindAssociatedSection();

  case Constant:
    return MCSymbol::AbsolutePseudoSection;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();

    if (Sym.isDefined())
      return &Sym.getSection();

    return 0;
  }

  case Unary:
    return cast<MCUnaryExpr>(this)->getSubExpr()->FindAssociatedSection();

  case Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(this);
    const MCSection *LHS_S = BE->getLHS()->FindAssociatedSection();
    const MCSection *RHS_S = BE->getRHS()->FindAssociatedSection();

    // If either section is absolute, return the other.
    if (LHS_S == MCSymbol::AbsolutePseudoSection)
      return RHS_S;
    if (RHS_S == MCSymbol::AbsolutePseudoSection)
      return LHS_S;

    // Otherwise, return the first non-null section.
    return LHS_S ? LHS_S : RHS_S;
  }
  }

  llvm_unreachable("Invalid assembly expression kind!");
}
