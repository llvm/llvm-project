//===--------- SCEVAffinator.cpp  - Create Scops from LLVM IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a SCEV value.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/SCEVAffinator.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "isl/aff.h"
#include "isl/local_space.h"
#include "isl/set.h"
#include "isl/val.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool> IgnoreIntegerWrapping(
    "polly-ignore-integer-wrapping",
    cl::desc("Do not build run-time checks to proof absence of integer "
             "wrapping"),
    cl::Hidden, cl::cat(PollyCategory));

// The maximal number of basic sets we allow during the construction of a
// piecewise affine function. More complex ones will result in very high
// compile time.
static int const MaxDisjunctionsInPwAff = 100;

// The maximal number of bits for which a general expression is modeled
// precisely.
static unsigned const MaxSmallBitWidth = 7;

/// Add the number of basic sets in @p Domain to @p User
static isl_stat addNumBasicSets(__isl_take isl_set *Domain,
                                __isl_take isl_aff *Aff, void *User) {
  auto *NumBasicSets = static_cast<unsigned *>(User);
  *NumBasicSets += isl_set_n_basic_set(Domain);
  isl_set_free(Domain);
  isl_aff_free(Aff);
  return isl_stat_ok;
}

/// Determine if @p PWAC is too complex to continue.
static bool isTooComplex(PWACtx PWAC) {
  unsigned NumBasicSets = 0;
  isl_pw_aff_foreach_piece(PWAC.first.get(), addNumBasicSets, &NumBasicSets);
  if (NumBasicSets <= MaxDisjunctionsInPwAff)
    return false;
  return true;
}

/// Return the flag describing the possible wrapping of @p Expr.
static SCEV::NoWrapFlags getNoWrapFlags(const SCEV *Expr) {
  if (auto *NAry = dyn_cast<SCEVNAryExpr>(Expr))
    return NAry->getNoWrapFlags();
  return SCEV::NoWrapMask;
}

static PWACtx combine(PWACtx PWAC0, PWACtx PWAC1,
                      __isl_give isl_pw_aff *(Fn)(__isl_take isl_pw_aff *,
                                                  __isl_take isl_pw_aff *)) {
  PWAC0.first = isl::manage(Fn(PWAC0.first.release(), PWAC1.first.release()));
  PWAC0.second = PWAC0.second.unite(PWAC1.second);
  return PWAC0;
}

static __isl_give isl_pw_aff *getWidthExpValOnDomain(unsigned Width,
                                                     __isl_take isl_set *Dom) {
  auto *Ctx = isl_set_get_ctx(Dom);
  auto *WidthVal = isl_val_int_from_ui(Ctx, Width);
  auto *ExpVal = isl_val_2exp(WidthVal);
  return isl_pw_aff_val_on_domain(Dom, ExpVal);
}

SCEVAffinator::SCEVAffinator(Scop *S, LoopInfo &LI)
    : S(S), Ctx(S->getIslCtx().get()), SE(*S->getSE()), LI(LI),
      TD(S->getFunction().getParent()->getDataLayout()) {}

Loop *SCEVAffinator::getScope() { return BB ? LI.getLoopFor(BB) : nullptr; }

void SCEVAffinator::interpretAsUnsigned(PWACtx &PWAC, unsigned Width) {
  auto *NonNegDom = isl_pw_aff_nonneg_set(PWAC.first.copy());
  auto *NonNegPWA =
      isl_pw_aff_intersect_domain(PWAC.first.copy(), isl_set_copy(NonNegDom));
  auto *ExpPWA = getWidthExpValOnDomain(Width, isl_set_complement(NonNegDom));
  PWAC.first = isl::manage(isl_pw_aff_union_add(
      NonNegPWA, isl_pw_aff_add(PWAC.first.release(), ExpPWA)));
}

void SCEVAffinator::takeNonNegativeAssumption(
    PWACtx &PWAC, RecordedAssumptionsTy *RecordedAssumptions) {
  this->RecordedAssumptions = RecordedAssumptions;

  auto *NegPWA = isl_pw_aff_neg(PWAC.first.copy());
  auto *NegDom = isl_pw_aff_pos_set(NegPWA);
  PWAC.second =
      isl::manage(isl_set_union(PWAC.second.release(), isl_set_copy(NegDom)));
  auto *Restriction = BB ? NegDom : isl_set_params(NegDom);
  auto DL = BB ? BB->getTerminator()->getDebugLoc() : DebugLoc();
  recordAssumption(RecordedAssumptions, UNSIGNED, isl::manage(Restriction), DL,
                   AS_RESTRICTION, BB);
}

PWACtx SCEVAffinator::getPWACtxFromPWA(isl::pw_aff PWA) {
  return std::make_pair(PWA, isl::set::empty(isl::space(Ctx, 0, NumIterators)));
}

PWACtx SCEVAffinator::getPwAff(const SCEV *Expr, BasicBlock *BB,
                               RecordedAssumptionsTy *RecordedAssumptions) {
  this->BB = BB;
  this->RecordedAssumptions = RecordedAssumptions;

  if (BB) {
    auto *DC = S->getDomainConditions(BB).release();
    NumIterators = isl_set_n_dim(DC);
    isl_set_free(DC);
  } else
    NumIterators = 0;

  return visit(Expr);
}

PWACtx SCEVAffinator::checkForWrapping(const SCEV *Expr, PWACtx PWAC) const {
  // If the SCEV flags do contain NSW (no signed wrap) then PWA already
  // represents Expr in modulo semantic (it is not allowed to overflow), thus we
  // are done. Otherwise, we will compute:
  //   PWA = ((PWA + 2^(n-1)) mod (2 ^ n)) - 2^(n-1)
  // whereas n is the number of bits of the Expr, hence:
  //   n = bitwidth(ExprType)

  if (IgnoreIntegerWrapping || (getNoWrapFlags(Expr) & SCEV::FlagNSW))
    return PWAC;

  isl::pw_aff PWAMod = addModuloSemantic(PWAC.first, Expr->getType());

  isl::set NotEqualSet = PWAC.first.ne_set(PWAMod);
  PWAC.second = PWAC.second.unite(NotEqualSet).coalesce();

  const DebugLoc &Loc = BB ? BB->getTerminator()->getDebugLoc() : DebugLoc();
  if (!BB)
    NotEqualSet = NotEqualSet.params();
  NotEqualSet = NotEqualSet.coalesce();

  if (!NotEqualSet.is_empty())
    recordAssumption(RecordedAssumptions, WRAPPING, NotEqualSet, Loc,
                     AS_RESTRICTION, BB);

  return PWAC;
}

isl::pw_aff SCEVAffinator::addModuloSemantic(isl::pw_aff PWA,
                                             Type *ExprType) const {
  unsigned Width = TD.getTypeSizeInBits(ExprType);

  auto ModVal = isl::val::int_from_ui(Ctx, Width);
  ModVal = ModVal.pow2();

  isl::set Domain = PWA.domain();
  isl::pw_aff AddPW =
      isl::manage(getWidthExpValOnDomain(Width - 1, Domain.release()));

  return PWA.add(AddPW).mod(ModVal).sub(AddPW);
}

bool SCEVAffinator::hasNSWAddRecForLoop(Loop *L) const {
  for (const auto &CachedPair : CachedExpressions) {
    auto *AddRec = dyn_cast<SCEVAddRecExpr>(CachedPair.first.first);
    if (!AddRec)
      continue;
    if (AddRec->getLoop() != L)
      continue;
    if (AddRec->getNoWrapFlags() & SCEV::FlagNSW)
      return true;
  }

  return false;
}

bool SCEVAffinator::computeModuloForExpr(const SCEV *Expr) {
  unsigned Width = TD.getTypeSizeInBits(Expr->getType());
  // We assume nsw expressions never overflow.
  if (auto *NAry = dyn_cast<SCEVNAryExpr>(Expr))
    if (NAry->getNoWrapFlags() & SCEV::FlagNSW)
      return false;
  return Width <= MaxSmallBitWidth;
}

PWACtx SCEVAffinator::visit(const SCEV *Expr) {

  auto Key = std::make_pair(Expr, BB);
  PWACtx PWAC = CachedExpressions[Key];
  if (!PWAC.first.is_null())
    return PWAC;

  auto ConstantAndLeftOverPair = extractConstantFactor(Expr, SE);
  auto *Factor = ConstantAndLeftOverPair.first;
  Expr = ConstantAndLeftOverPair.second;

  auto *Scope = getScope();
  S->addParams(getParamsInAffineExpr(&S->getRegion(), Scope, Expr, SE));

  // In case the scev is a valid parameter, we do not further analyze this
  // expression, but create a new parameter in the isl_pw_aff. This allows us
  // to treat subexpressions that we cannot translate into an piecewise affine
  // expression, as constant parameters of the piecewise affine expression.
  if (isl_id *Id = S->getIdForParam(Expr).release()) {
    isl_space *Space = isl_space_set_alloc(Ctx.get(), 1, NumIterators);
    Space = isl_space_set_dim_id(Space, isl_dim_param, 0, Id);

    isl_set *Domain = isl_set_universe(isl_space_copy(Space));
    isl_aff *Affine = isl_aff_zero_on_domain(isl_local_space_from_space(Space));
    Affine = isl_aff_add_coefficient_si(Affine, isl_dim_param, 0, 1);

    PWAC = getPWACtxFromPWA(isl::manage(isl_pw_aff_alloc(Domain, Affine)));
  } else {
    PWAC = SCEVVisitor<SCEVAffinator, PWACtx>::visit(Expr);
    if (computeModuloForExpr(Expr))
      PWAC.first = addModuloSemantic(PWAC.first, Expr->getType());
    else
      PWAC = checkForWrapping(Expr, PWAC);
  }

  if (!Factor->getType()->isIntegerTy(1)) {
    PWAC = combine(PWAC, visitConstant(Factor), isl_pw_aff_mul);
    if (computeModuloForExpr(Key.first))
      PWAC.first = addModuloSemantic(PWAC.first, Expr->getType());
  }

  // For compile time reasons we need to simplify the PWAC before we cache and
  // return it.
  PWAC.first = PWAC.first.coalesce();
  if (!computeModuloForExpr(Key.first))
    PWAC = checkForWrapping(Key.first, PWAC);

  CachedExpressions[Key] = PWAC;
  return PWAC;
}

PWACtx SCEVAffinator::visitConstant(const SCEVConstant *Expr) {
  ConstantInt *Value = Expr->getValue();
  isl_val *v;

  // LLVM does not define if an integer value is interpreted as a signed or
  // unsigned value. Hence, without further information, it is unknown how
  // this value needs to be converted to GMP. At the moment, we only support
  // signed operations. So we just interpret it as signed. Later, there are
  // two options:
  //
  // 1. We always interpret any value as signed and convert the values on
  //    demand.
  // 2. We pass down the signedness of the calculation and use it to interpret
  //    this constant correctly.
  v = isl_valFromAPInt(Ctx.get(), Value->getValue(), /* isSigned */ true);

  isl_space *Space = isl_space_set_alloc(Ctx.get(), 0, NumIterators);
  isl_local_space *ls = isl_local_space_from_space(Space);
  return getPWACtxFromPWA(
      isl::manage(isl_pw_aff_from_aff(isl_aff_val_on_domain(ls, v))));
}

PWACtx SCEVAffinator::visitVScale(const SCEVVScale *VScale) {
  llvm_unreachable("SCEVVScale not yet supported");
}

PWACtx SCEVAffinator::visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr) {
  return visit(Expr->getOperand(0));
}

PWACtx SCEVAffinator::visitTruncateExpr(const SCEVTruncateExpr *Expr) {
  // Truncate operations are basically modulo operations, thus we can
  // model them that way. However, for large types we assume the operand
  // to fit in the new type size instead of introducing a modulo with a very
  // large constant.

  auto *Op = Expr->getOperand();
  auto OpPWAC = visit(Op);

  unsigned Width = TD.getTypeSizeInBits(Expr->getType());

  if (computeModuloForExpr(Expr))
    return OpPWAC;

  auto *Dom = OpPWAC.first.domain().release();
  auto *ExpPWA = getWidthExpValOnDomain(Width - 1, Dom);
  auto *GreaterDom =
      isl_pw_aff_ge_set(OpPWAC.first.copy(), isl_pw_aff_copy(ExpPWA));
  auto *SmallerDom =
      isl_pw_aff_lt_set(OpPWAC.first.copy(), isl_pw_aff_neg(ExpPWA));
  auto *OutOfBoundsDom = isl_set_union(SmallerDom, GreaterDom);
  OpPWAC.second = OpPWAC.second.unite(isl::manage_copy(OutOfBoundsDom));

  if (!BB) {
    assert(isl_set_dim(OutOfBoundsDom, isl_dim_set) == 0 &&
           "Expected a zero dimensional set for non-basic-block domains");
    OutOfBoundsDom = isl_set_params(OutOfBoundsDom);
  }

  recordAssumption(RecordedAssumptions, UNSIGNED, isl::manage(OutOfBoundsDom),
                   DebugLoc(), AS_RESTRICTION, BB);

  return OpPWAC;
}

PWACtx SCEVAffinator::visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
  // A zero-extended value can be interpreted as a piecewise defined signed
  // value. If the value was non-negative it stays the same, otherwise it
  // is the sum of the original value and 2^n where n is the bit-width of
  // the original (or operand) type. Examples:
  //   zext i8 127 to i32 -> { [127] }
  //   zext i8  -1 to i32 -> { [256 + (-1)] } = { [255] }
  //   zext i8  %v to i32 -> [v] -> { [v] | v >= 0; [256 + v] | v < 0 }
  //
  // However, LLVM/Scalar Evolution uses zero-extend (potentially lead by a
  // truncate) to represent some forms of modulo computation. The left-hand side
  // of the condition in the code below would result in the SCEV
  // "zext i1 <false, +, true>for.body" which is just another description
  // of the C expression "i & 1 != 0" or, equivalently, "i % 2 != 0".
  //
  //   for (i = 0; i < N; i++)
  //     if (i & 1 != 0 /* == i % 2 */)
  //       /* do something */
  //
  // If we do not make the modulo explicit but only use the mechanism described
  // above we will get the very restrictive assumption "N < 3", because for all
  // values of N >= 3 the SCEVAddRecExpr operand of the zero-extend would wrap.
  // Alternatively, we can make the modulo in the operand explicit in the
  // resulting piecewise function and thereby avoid the assumption on N. For the
  // example this would result in the following piecewise affine function:
  // { [i0] -> [(1)] : 2*floor((-1 + i0)/2) = -1 + i0;
  //   [i0] -> [(0)] : 2*floor((i0)/2) = i0 }
  // To this end we can first determine if the (immediate) operand of the
  // zero-extend can wrap and, in case it might, we will use explicit modulo
  // semantic to compute the result instead of emitting non-wrapping
  // assumptions.
  //
  // Note that operands with large bit-widths are less likely to be negative
  // because it would result in a very large access offset or loop bound after
  // the zero-extend. To this end one can optimistically assume the operand to
  // be positive and avoid the piecewise definition if the bit-width is bigger
  // than some threshold (here MaxZextSmallBitWidth).
  //
  // We choose to go with a hybrid solution of all modeling techniques described
  // above. For small bit-widths (up to MaxZextSmallBitWidth) we will model the
  // wrapping explicitly and use a piecewise defined function. However, if the
  // bit-width is bigger than MaxZextSmallBitWidth we will employ overflow
  // assumptions and assume the "former negative" piece will not exist.

  auto *Op = Expr->getOperand();
  auto OpPWAC = visit(Op);

  // If the width is to big we assume the negative part does not occur.
  if (!computeModuloForExpr(Op)) {
    takeNonNegativeAssumption(OpPWAC, RecordedAssumptions);
    return OpPWAC;
  }

  // If the width is small build the piece for the non-negative part and
  // the one for the negative part and unify them.
  unsigned Width = TD.getTypeSizeInBits(Op->getType());
  interpretAsUnsigned(OpPWAC, Width);
  return OpPWAC;
}

PWACtx SCEVAffinator::visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
  // As all values are represented as signed, a sign extension is a noop.
  return visit(Expr->getOperand());
}

PWACtx SCEVAffinator::visitAddExpr(const SCEVAddExpr *Expr) {
  PWACtx Sum = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    Sum = combine(Sum, visit(Expr->getOperand(i)), isl_pw_aff_add);
    if (isTooComplex(Sum))
      return complexityBailout();
  }

  return Sum;
}

PWACtx SCEVAffinator::visitMulExpr(const SCEVMulExpr *Expr) {
  PWACtx Prod = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    Prod = combine(Prod, visit(Expr->getOperand(i)), isl_pw_aff_mul);
    if (isTooComplex(Prod))
      return complexityBailout();
  }

  return Prod;
}

PWACtx SCEVAffinator::visitAddRecExpr(const SCEVAddRecExpr *Expr) {
  assert(Expr->isAffine() && "Only affine AddRecurrences allowed");

  auto Flags = Expr->getNoWrapFlags();

  // Directly generate isl_pw_aff for Expr if 'start' is zero.
  if (Expr->getStart()->isZero()) {
    assert(S->contains(Expr->getLoop()) &&
           "Scop does not contain the loop referenced in this AddRec");

    PWACtx Step = visit(Expr->getOperand(1));
    isl_space *Space = isl_space_set_alloc(Ctx.get(), 0, NumIterators);
    isl_local_space *LocalSpace = isl_local_space_from_space(Space);

    unsigned loopDimension = S->getRelativeLoopDepth(Expr->getLoop());

    isl_aff *LAff = isl_aff_set_coefficient_si(
        isl_aff_zero_on_domain(LocalSpace), isl_dim_in, loopDimension, 1);
    isl_pw_aff *LPwAff = isl_pw_aff_from_aff(LAff);

    Step.first = Step.first.mul(isl::manage(LPwAff));
    return Step;
  }

  // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
  // if 'start' is not zero.
  // TODO: Using the original SCEV no-wrap flags is not always safe, however
  //       as our code generation is reordering the expression anyway it doesn't
  //       really matter.
  const SCEV *ZeroStartExpr =
      SE.getAddRecExpr(SE.getConstant(Expr->getStart()->getType(), 0),
                       Expr->getStepRecurrence(SE), Expr->getLoop(), Flags);

  PWACtx Result = visit(ZeroStartExpr);
  PWACtx Start = visit(Expr->getStart());
  Result = combine(Result, Start, isl_pw_aff_add);
  return Result;
}

PWACtx SCEVAffinator::visitSMaxExpr(const SCEVSMaxExpr *Expr) {
  PWACtx Max = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    Max = combine(Max, visit(Expr->getOperand(i)), isl_pw_aff_max);
    if (isTooComplex(Max))
      return complexityBailout();
  }

  return Max;
}

PWACtx SCEVAffinator::visitSMinExpr(const SCEVSMinExpr *Expr) {
  PWACtx Min = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    Min = combine(Min, visit(Expr->getOperand(i)), isl_pw_aff_min);
    if (isTooComplex(Min))
      return complexityBailout();
  }

  return Min;
}

PWACtx SCEVAffinator::visitUMaxExpr(const SCEVUMaxExpr *Expr) {
  llvm_unreachable("SCEVUMaxExpr not yet supported");
}

PWACtx SCEVAffinator::visitUMinExpr(const SCEVUMinExpr *Expr) {
  llvm_unreachable("SCEVUMinExpr not yet supported");
}

PWACtx
SCEVAffinator::visitSequentialUMinExpr(const SCEVSequentialUMinExpr *Expr) {
  llvm_unreachable("SCEVSequentialUMinExpr not yet supported");
}

PWACtx SCEVAffinator::visitUDivExpr(const SCEVUDivExpr *Expr) {
  // The handling of unsigned division is basically the same as for signed
  // division, except the interpretation of the operands. As the divisor
  // has to be constant in both cases we can simply interpret it as an
  // unsigned value without additional complexity in the representation.
  // For the dividend we could choose from the different representation
  // schemes introduced for zero-extend operations but for now we will
  // simply use an assumption.
  auto *Dividend = Expr->getLHS();
  auto *Divisor = Expr->getRHS();
  assert(isa<SCEVConstant>(Divisor) &&
         "UDiv is no parameter but has a non-constant RHS.");

  auto DividendPWAC = visit(Dividend);
  auto DivisorPWAC = visit(Divisor);

  if (SE.isKnownNegative(Divisor)) {
    // Interpret negative divisors unsigned. This is a special case of the
    // piece-wise defined value described for zero-extends as we already know
    // the actual value of the constant divisor.
    unsigned Width = TD.getTypeSizeInBits(Expr->getType());
    auto *DivisorDom = DivisorPWAC.first.domain().release();
    auto *WidthExpPWA = getWidthExpValOnDomain(Width, DivisorDom);
    DivisorPWAC.first = DivisorPWAC.first.add(isl::manage(WidthExpPWA));
  }

  // TODO: One can represent the dividend as piece-wise function to be more
  //       precise but therefor a heuristic is needed.

  // Assume a non-negative dividend.
  takeNonNegativeAssumption(DividendPWAC, RecordedAssumptions);

  DividendPWAC = combine(DividendPWAC, DivisorPWAC, isl_pw_aff_div);
  DividendPWAC.first = DividendPWAC.first.floor();

  return DividendPWAC;
}

PWACtx SCEVAffinator::visitSDivInstruction(Instruction *SDiv) {
  assert(SDiv->getOpcode() == Instruction::SDiv && "Assumed SDiv instruction!");

  auto *Scope = getScope();
  auto *Divisor = SDiv->getOperand(1);
  auto *DivisorSCEV = SE.getSCEVAtScope(Divisor, Scope);
  auto DivisorPWAC = visit(DivisorSCEV);
  assert(isa<SCEVConstant>(DivisorSCEV) &&
         "SDiv is no parameter but has a non-constant RHS.");

  auto *Dividend = SDiv->getOperand(0);
  auto *DividendSCEV = SE.getSCEVAtScope(Dividend, Scope);
  auto DividendPWAC = visit(DividendSCEV);
  DividendPWAC = combine(DividendPWAC, DivisorPWAC, isl_pw_aff_tdiv_q);
  return DividendPWAC;
}

PWACtx SCEVAffinator::visitSRemInstruction(Instruction *SRem) {
  assert(SRem->getOpcode() == Instruction::SRem && "Assumed SRem instruction!");

  auto *Scope = getScope();
  auto *Divisor = SRem->getOperand(1);
  auto *DivisorSCEV = SE.getSCEVAtScope(Divisor, Scope);
  auto DivisorPWAC = visit(DivisorSCEV);
  assert(isa<ConstantInt>(Divisor) &&
         "SRem is no parameter but has a non-constant RHS.");

  auto *Dividend = SRem->getOperand(0);
  auto *DividendSCEV = SE.getSCEVAtScope(Dividend, Scope);
  auto DividendPWAC = visit(DividendSCEV);
  DividendPWAC = combine(DividendPWAC, DivisorPWAC, isl_pw_aff_tdiv_r);
  return DividendPWAC;
}

PWACtx SCEVAffinator::visitUnknown(const SCEVUnknown *Expr) {
  if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
    switch (I->getOpcode()) {
    case Instruction::IntToPtr:
      return visit(SE.getSCEVAtScope(I->getOperand(0), getScope()));
    case Instruction::SDiv:
      return visitSDivInstruction(I);
    case Instruction::SRem:
      return visitSRemInstruction(I);
    default:
      break; // Fall through.
    }
  }

  if (isa<ConstantPointerNull>(Expr->getValue())) {
    isl::val v{Ctx, 0};
    isl::space Space{Ctx, 0, NumIterators};
    isl::local_space ls{Space};
    return getPWACtxFromPWA(isl::aff(ls, v));
  }

  llvm_unreachable("Unknowns SCEV was neither a parameter, a constant nor a "
                   "valid instruction.");
}

PWACtx SCEVAffinator::complexityBailout() {
  // We hit the complexity limit for affine expressions; invalidate the scop
  // and return a constant zero.
  const DebugLoc &Loc = BB ? BB->getTerminator()->getDebugLoc() : DebugLoc();
  S->invalidate(COMPLEXITY, Loc);
  return visit(SE.getZero(Type::getInt32Ty(S->getFunction().getContext())));
}
