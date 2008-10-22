//===- ScalarEvolution.cpp - Scalar Evolution Analysis ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution analysis
// engine, which is used primarily to analyze expressions involving induction
// variables in loops.
//
// There are several aspects to this library.  First is the representation of
// scalar expressions, which are represented as subclasses of the SCEV class.
// These classes are used to represent certain types of subexpressions that we
// can handle.  These classes are reference counted, managed by the SCEVHandle
// class.  We only create one SCEV of a particular shape, so pointer-comparisons
// for equality are legal.
//
// One important aspect of the SCEV objects is that they are never cyclic, even
// if there is a cycle in the dataflow for an expression (ie, a PHI node).  If
// the PHI node is one of the idioms that we can represent (e.g., a polynomial
// recurrence) then we represent it directly as a recurrence node, otherwise we
// represent it as a SCEVUnknown node.
//
// In addition to being able to represent expressions of various types, we also
// have folders that are used to build the *canonical* representation for a
// particular expression.  These folders are capable of using a variety of
// rewrite rules to simplify the expressions.
//
// Once the folders are defined, we can implement the more interesting
// higher-level code, such as the code that recognizes PHI nodes of various
// types, computes the execution count of a loop, etc.
//
// TODO: We should use these routines and value representations to implement
// dependence analysis!
//
//===----------------------------------------------------------------------===//
//
// There are several good references for the techniques used in this analysis.
//
//  Chains of recurrences -- a method to expedite the evaluation
//  of closed-form functions
//  Olaf Bachmann, Paul S. Wang, Eugene V. Zima
//
//  On computational properties of chains of recurrences
//  Eugene V. Zima
//
//  Symbolic Evaluation of Chains of Recurrences for Loop Optimization
//  Robert A. van Engelen
//
//  Efficient Symbolic Analysis for Optimizing Compilers
//  Robert A. van Engelen
//
//  Using the chains of recurrences algebra for data dependence testing and
//  induction variable substitution
//  MS Thesis, Johnie Birch
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "scalar-evolution"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/Statistic.h"
#include <ostream>
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(NumBruteForceEvaluations,
          "Number of brute force evaluations needed to "
          "calculate high-order polynomial exit values");
STATISTIC(NumArrayLenItCounts,
          "Number of trip counts computed with array length");
STATISTIC(NumTripCountsComputed,
          "Number of loops with predictable loop counts");
STATISTIC(NumTripCountsNotComputed,
          "Number of loops without predictable loop counts");
STATISTIC(NumBruteForceTripCountsComputed,
          "Number of loops with trip counts computed by force");

static cl::opt<unsigned>
MaxBruteForceIterations("scalar-evolution-max-iterations", cl::ReallyHidden,
                        cl::desc("Maximum number of iterations SCEV will "
                                 "symbolically execute a constant derived loop"),
                        cl::init(100));

static RegisterPass<ScalarEvolution>
R("scalar-evolution", "Scalar Evolution Analysis", false, true);
char ScalarEvolution::ID = 0;

//===----------------------------------------------------------------------===//
//                           SCEV class definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Implementation of the SCEV class.
//
SCEV::~SCEV() {}
void SCEV::dump() const {
  print(cerr);
}

uint32_t SCEV::getBitWidth() const {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(getType()))
    return ITy->getBitWidth();
  return 0;
}

bool SCEV::isZero() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isZero();
  return false;
}


SCEVCouldNotCompute::SCEVCouldNotCompute() : SCEV(scCouldNotCompute) {}

bool SCEVCouldNotCompute::isLoopInvariant(const Loop *L) const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return false;
}

const Type *SCEVCouldNotCompute::getType() const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return 0;
}

bool SCEVCouldNotCompute::hasComputableLoopEvolution(const Loop *L) const {
  assert(0 && "Attempt to use a SCEVCouldNotCompute object!");
  return false;
}

SCEVHandle SCEVCouldNotCompute::
replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                  const SCEVHandle &Conc,
                                  ScalarEvolution &SE) const {
  return this;
}

void SCEVCouldNotCompute::print(std::ostream &OS) const {
  OS << "***COULDNOTCOMPUTE***";
}

bool SCEVCouldNotCompute::classof(const SCEV *S) {
  return S->getSCEVType() == scCouldNotCompute;
}


// SCEVConstants - Only allow the creation of one SCEVConstant for any
// particular value.  Don't use a SCEVHandle here, or else the object will
// never be deleted!
static ManagedStatic<std::map<ConstantInt*, SCEVConstant*> > SCEVConstants;


SCEVConstant::~SCEVConstant() {
  SCEVConstants->erase(V);
}

SCEVHandle ScalarEvolution::getConstant(ConstantInt *V) {
  SCEVConstant *&R = (*SCEVConstants)[V];
  if (R == 0) R = new SCEVConstant(V);
  return R;
}

SCEVHandle ScalarEvolution::getConstant(const APInt& Val) {
  return getConstant(ConstantInt::get(Val));
}

const Type *SCEVConstant::getType() const { return V->getType(); }

void SCEVConstant::print(std::ostream &OS) const {
  WriteAsOperand(OS, V, false);
}

// SCEVTruncates - Only allow the creation of one SCEVTruncateExpr for any
// particular input.  Don't use a SCEVHandle here, or else the object will
// never be deleted!
static ManagedStatic<std::map<std::pair<SCEV*, const Type*>, 
                     SCEVTruncateExpr*> > SCEVTruncates;

SCEVTruncateExpr::SCEVTruncateExpr(const SCEVHandle &op, const Type *ty)
  : SCEV(scTruncate), Op(op), Ty(ty) {
  assert(Op->getType()->isInteger() && Ty->isInteger() &&
         "Cannot truncate non-integer value!");
  assert(Op->getType()->getPrimitiveSizeInBits() > Ty->getPrimitiveSizeInBits()
         && "This is not a truncating conversion!");
}

SCEVTruncateExpr::~SCEVTruncateExpr() {
  SCEVTruncates->erase(std::make_pair(Op, Ty));
}

void SCEVTruncateExpr::print(std::ostream &OS) const {
  OS << "(truncate " << *Op << " to " << *Ty << ")";
}

// SCEVZeroExtends - Only allow the creation of one SCEVZeroExtendExpr for any
// particular input.  Don't use a SCEVHandle here, or else the object will never
// be deleted!
static ManagedStatic<std::map<std::pair<SCEV*, const Type*>,
                     SCEVZeroExtendExpr*> > SCEVZeroExtends;

SCEVZeroExtendExpr::SCEVZeroExtendExpr(const SCEVHandle &op, const Type *ty)
  : SCEV(scZeroExtend), Op(op), Ty(ty) {
  assert(Op->getType()->isInteger() && Ty->isInteger() &&
         "Cannot zero extend non-integer value!");
  assert(Op->getType()->getPrimitiveSizeInBits() < Ty->getPrimitiveSizeInBits()
         && "This is not an extending conversion!");
}

SCEVZeroExtendExpr::~SCEVZeroExtendExpr() {
  SCEVZeroExtends->erase(std::make_pair(Op, Ty));
}

void SCEVZeroExtendExpr::print(std::ostream &OS) const {
  OS << "(zeroextend " << *Op << " to " << *Ty << ")";
}

// SCEVSignExtends - Only allow the creation of one SCEVSignExtendExpr for any
// particular input.  Don't use a SCEVHandle here, or else the object will never
// be deleted!
static ManagedStatic<std::map<std::pair<SCEV*, const Type*>,
                     SCEVSignExtendExpr*> > SCEVSignExtends;

SCEVSignExtendExpr::SCEVSignExtendExpr(const SCEVHandle &op, const Type *ty)
  : SCEV(scSignExtend), Op(op), Ty(ty) {
  assert(Op->getType()->isInteger() && Ty->isInteger() &&
         "Cannot sign extend non-integer value!");
  assert(Op->getType()->getPrimitiveSizeInBits() < Ty->getPrimitiveSizeInBits()
         && "This is not an extending conversion!");
}

SCEVSignExtendExpr::~SCEVSignExtendExpr() {
  SCEVSignExtends->erase(std::make_pair(Op, Ty));
}

void SCEVSignExtendExpr::print(std::ostream &OS) const {
  OS << "(signextend " << *Op << " to " << *Ty << ")";
}

// SCEVCommExprs - Only allow the creation of one SCEVCommutativeExpr for any
// particular input.  Don't use a SCEVHandle here, or else the object will never
// be deleted!
static ManagedStatic<std::map<std::pair<unsigned, std::vector<SCEV*> >,
                     SCEVCommutativeExpr*> > SCEVCommExprs;

SCEVCommutativeExpr::~SCEVCommutativeExpr() {
  SCEVCommExprs->erase(std::make_pair(getSCEVType(),
                                      std::vector<SCEV*>(Operands.begin(),
                                                         Operands.end())));
}

void SCEVCommutativeExpr::print(std::ostream &OS) const {
  assert(Operands.size() > 1 && "This plus expr shouldn't exist!");
  const char *OpStr = getOperationStr();
  OS << "(" << *Operands[0];
  for (unsigned i = 1, e = Operands.size(); i != e; ++i)
    OS << OpStr << *Operands[i];
  OS << ")";
}

SCEVHandle SCEVCommutativeExpr::
replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                  const SCEVHandle &Conc,
                                  ScalarEvolution &SE) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    SCEVHandle H =
      getOperand(i)->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
    if (H != getOperand(i)) {
      std::vector<SCEVHandle> NewOps;
      NewOps.reserve(getNumOperands());
      for (unsigned j = 0; j != i; ++j)
        NewOps.push_back(getOperand(j));
      NewOps.push_back(H);
      for (++i; i != e; ++i)
        NewOps.push_back(getOperand(i)->
                         replaceSymbolicValuesWithConcrete(Sym, Conc, SE));

      if (isa<SCEVAddExpr>(this))
        return SE.getAddExpr(NewOps);
      else if (isa<SCEVMulExpr>(this))
        return SE.getMulExpr(NewOps);
      else if (isa<SCEVSMaxExpr>(this))
        return SE.getSMaxExpr(NewOps);
      else if (isa<SCEVUMaxExpr>(this))
        return SE.getUMaxExpr(NewOps);
      else
        assert(0 && "Unknown commutative expr!");
    }
  }
  return this;
}


// SCEVUDivs - Only allow the creation of one SCEVUDivExpr for any particular
// input.  Don't use a SCEVHandle here, or else the object will never be
// deleted!
static ManagedStatic<std::map<std::pair<SCEV*, SCEV*>, 
                     SCEVUDivExpr*> > SCEVUDivs;

SCEVUDivExpr::~SCEVUDivExpr() {
  SCEVUDivs->erase(std::make_pair(LHS, RHS));
}

void SCEVUDivExpr::print(std::ostream &OS) const {
  OS << "(" << *LHS << " /u " << *RHS << ")";
}

const Type *SCEVUDivExpr::getType() const {
  return LHS->getType();
}

// SCEVAddRecExprs - Only allow the creation of one SCEVAddRecExpr for any
// particular input.  Don't use a SCEVHandle here, or else the object will never
// be deleted!
static ManagedStatic<std::map<std::pair<const Loop *, std::vector<SCEV*> >,
                     SCEVAddRecExpr*> > SCEVAddRecExprs;

SCEVAddRecExpr::~SCEVAddRecExpr() {
  SCEVAddRecExprs->erase(std::make_pair(L,
                                        std::vector<SCEV*>(Operands.begin(),
                                                           Operands.end())));
}

SCEVHandle SCEVAddRecExpr::
replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                  const SCEVHandle &Conc,
                                  ScalarEvolution &SE) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    SCEVHandle H =
      getOperand(i)->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
    if (H != getOperand(i)) {
      std::vector<SCEVHandle> NewOps;
      NewOps.reserve(getNumOperands());
      for (unsigned j = 0; j != i; ++j)
        NewOps.push_back(getOperand(j));
      NewOps.push_back(H);
      for (++i; i != e; ++i)
        NewOps.push_back(getOperand(i)->
                         replaceSymbolicValuesWithConcrete(Sym, Conc, SE));

      return SE.getAddRecExpr(NewOps, L);
    }
  }
  return this;
}


bool SCEVAddRecExpr::isLoopInvariant(const Loop *QueryLoop) const {
  // This recurrence is invariant w.r.t to QueryLoop iff QueryLoop doesn't
  // contain L and if the start is invariant.
  return !QueryLoop->contains(L->getHeader()) &&
         getOperand(0)->isLoopInvariant(QueryLoop);
}


void SCEVAddRecExpr::print(std::ostream &OS) const {
  OS << "{" << *Operands[0];
  for (unsigned i = 1, e = Operands.size(); i != e; ++i)
    OS << ",+," << *Operands[i];
  OS << "}<" << L->getHeader()->getName() + ">";
}

// SCEVUnknowns - Only allow the creation of one SCEVUnknown for any particular
// value.  Don't use a SCEVHandle here, or else the object will never be
// deleted!
static ManagedStatic<std::map<Value*, SCEVUnknown*> > SCEVUnknowns;

SCEVUnknown::~SCEVUnknown() { SCEVUnknowns->erase(V); }

bool SCEVUnknown::isLoopInvariant(const Loop *L) const {
  // All non-instruction values are loop invariant.  All instructions are loop
  // invariant if they are not contained in the specified loop.
  if (Instruction *I = dyn_cast<Instruction>(V))
    return !L->contains(I->getParent());
  return true;
}

const Type *SCEVUnknown::getType() const {
  return V->getType();
}

void SCEVUnknown::print(std::ostream &OS) const {
  WriteAsOperand(OS, V, false);
}

//===----------------------------------------------------------------------===//
//                               SCEV Utilities
//===----------------------------------------------------------------------===//

namespace {
  /// SCEVComplexityCompare - Return true if the complexity of the LHS is less
  /// than the complexity of the RHS.  This comparator is used to canonicalize
  /// expressions.
  struct VISIBILITY_HIDDEN SCEVComplexityCompare {
    bool operator()(const SCEV *LHS, const SCEV *RHS) const {
      return LHS->getSCEVType() < RHS->getSCEVType();
    }
  };
}

/// GroupByComplexity - Given a list of SCEV objects, order them by their
/// complexity, and group objects of the same complexity together by value.
/// When this routine is finished, we know that any duplicates in the vector are
/// consecutive and that complexity is monotonically increasing.
///
/// Note that we go take special precautions to ensure that we get determinstic
/// results from this routine.  In other words, we don't want the results of
/// this to depend on where the addresses of various SCEV objects happened to
/// land in memory.
///
static void GroupByComplexity(std::vector<SCEVHandle> &Ops) {
  if (Ops.size() < 2) return;  // Noop
  if (Ops.size() == 2) {
    // This is the common case, which also happens to be trivially simple.
    // Special case it.
    if (SCEVComplexityCompare()(Ops[1], Ops[0]))
      std::swap(Ops[0], Ops[1]);
    return;
  }

  // Do the rough sort by complexity.
  std::sort(Ops.begin(), Ops.end(), SCEVComplexityCompare());

  // Now that we are sorted by complexity, group elements of the same
  // complexity.  Note that this is, at worst, N^2, but the vector is likely to
  // be extremely short in practice.  Note that we take this approach because we
  // do not want to depend on the addresses of the objects we are grouping.
  for (unsigned i = 0, e = Ops.size(); i != e-2; ++i) {
    SCEV *S = Ops[i];
    unsigned Complexity = S->getSCEVType();

    // If there are any objects of the same complexity and same value as this
    // one, group them.
    for (unsigned j = i+1; j != e && Ops[j]->getSCEVType() == Complexity; ++j) {
      if (Ops[j] == S) { // Found a duplicate.
        // Move it to immediately after i'th element.
        std::swap(Ops[i+1], Ops[j]);
        ++i;   // no need to rescan it.
        if (i == e-2) return;  // Done!
      }
    }
  }
}



//===----------------------------------------------------------------------===//
//                      Simple SCEV method implementations
//===----------------------------------------------------------------------===//

/// getIntegerSCEV - Given an integer or FP type, create a constant for the
/// specified signed integer value and return a SCEV for the constant.
SCEVHandle ScalarEvolution::getIntegerSCEV(int Val, const Type *Ty) {
  Constant *C;
  if (Val == 0)
    C = Constant::getNullValue(Ty);
  else if (Ty->isFloatingPoint())
    C = ConstantFP::get(APFloat(Ty==Type::FloatTy ? APFloat::IEEEsingle : 
                                APFloat::IEEEdouble, Val));
  else 
    C = ConstantInt::get(Ty, Val);
  return getUnknown(C);
}

/// getNegativeSCEV - Return a SCEV corresponding to -V = -1*V
///
SCEVHandle ScalarEvolution::getNegativeSCEV(const SCEVHandle &V) {
  if (SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getUnknown(ConstantExpr::getNeg(VC->getValue()));

  return getMulExpr(V, getConstant(ConstantInt::getAllOnesValue(V->getType())));
}

/// getNotSCEV - Return a SCEV corresponding to ~V = -1-V
SCEVHandle ScalarEvolution::getNotSCEV(const SCEVHandle &V) {
  if (SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getUnknown(ConstantExpr::getNot(VC->getValue()));

  SCEVHandle AllOnes = getConstant(ConstantInt::getAllOnesValue(V->getType()));
  return getMinusSCEV(AllOnes, V);
}

/// getMinusSCEV - Return a SCEV corresponding to LHS - RHS.
///
SCEVHandle ScalarEvolution::getMinusSCEV(const SCEVHandle &LHS,
                                         const SCEVHandle &RHS) {
  // X - Y --> X + -Y
  return getAddExpr(LHS, getNegativeSCEV(RHS));
}


/// BinomialCoefficient - Compute BC(It, K).  The result has width W.
// Assume, K > 0.
static SCEVHandle BinomialCoefficient(SCEVHandle It, unsigned K,
                                      ScalarEvolution &SE,
                                      const IntegerType* ResultTy) {
  // Handle the simplest case efficiently.
  if (K == 1)
    return SE.getTruncateOrZeroExtend(It, ResultTy);

  // We are using the following formula for BC(It, K):
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / K!
  //
  // Suppose, W is the bitwidth of the return value.  We must be prepared for
  // overflow.  Hence, we must assure that the result of our computation is
  // equal to the accurate one modulo 2^W.  Unfortunately, division isn't
  // safe in modular arithmetic.
  //
  // However, this code doesn't use exactly that formula; the formula it uses
  // is something like the following, where T is the number of factors of 2 in 
  // K! (i.e. trailing zeros in the binary representation of K!), and ^ is
  // exponentiation:
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / 2^T / (K! / 2^T)
  //
  // This formula is trivially equivalent to the previous formula.  However,
  // this formula can be implemented much more efficiently.  The trick is that
  // K! / 2^T is odd, and exact division by an odd number *is* safe in modular
  // arithmetic.  To do exact division in modular arithmetic, all we have
  // to do is multiply by the inverse.  Therefore, this step can be done at
  // width W.
  // 
  // The next issue is how to safely do the division by 2^T.  The way this
  // is done is by doing the multiplication step at a width of at least W + T
  // bits.  This way, the bottom W+T bits of the product are accurate. Then,
  // when we perform the division by 2^T (which is equivalent to a right shift
  // by T), the bottom W bits are accurate.  Extra bits are okay; they'll get
  // truncated out after the division by 2^T.
  //
  // In comparison to just directly using the first formula, this technique
  // is much more efficient; using the first formula requires W * K bits,
  // but this formula less than W + K bits. Also, the first formula requires
  // a division step, whereas this formula only requires multiplies and shifts.
  //
  // It doesn't matter whether the subtraction step is done in the calculation
  // width or the input iteration count's width; if the subtraction overflows,
  // the result must be zero anyway.  We prefer here to do it in the width of
  // the induction variable because it helps a lot for certain cases; CodeGen
  // isn't smart enough to ignore the overflow, which leads to much less
  // efficient code if the width of the subtraction is wider than the native
  // register width.
  //
  // (It's possible to not widen at all by pulling out factors of 2 before
  // the multiplication; for example, K=2 can be calculated as
  // It/2*(It+(It*INT_MIN/INT_MIN)+-1). However, it requires
  // extra arithmetic, so it's not an obvious win, and it gets
  // much more complicated for K > 3.)

  // Protection from insane SCEVs; this bound is conservative,
  // but it probably doesn't matter.
  if (K > 1000)
    return new SCEVCouldNotCompute();

  unsigned W = ResultTy->getBitWidth();

  // Calculate K! / 2^T and T; we divide out the factors of two before
  // multiplying for calculating K! / 2^T to avoid overflow.
  // Other overflow doesn't matter because we only care about the bottom
  // W bits of the result.
  APInt OddFactorial(W, 1);
  unsigned T = 1;
  for (unsigned i = 3; i <= K; ++i) {
    APInt Mult(W, i);
    unsigned TwoFactors = Mult.countTrailingZeros();
    T += TwoFactors;
    Mult = Mult.lshr(TwoFactors);
    OddFactorial *= Mult;
  }

  // We need at least W + T bits for the multiplication step
  // FIXME: A temporary hack; we round up the bitwidths
  // to the nearest power of 2 to be nice to the code generator.
  unsigned CalculationBits = 1U << Log2_32_Ceil(W + T);
  // FIXME: Temporary hack to avoid generating integers that are too wide.
  // Although, it's not completely clear how to determine how much
  // widening is safe; for example, on X86, we can't really widen
  // beyond 64 because we need to be able to do multiplication
  // that's CalculationBits wide, but on X86-64, we can safely widen up to
  // 128 bits.
  if (CalculationBits > 64)
    return new SCEVCouldNotCompute();

  // Calcuate 2^T, at width T+W.
  APInt DivFactor = APInt(CalculationBits, 1).shl(T);

  // Calculate the multiplicative inverse of K! / 2^T;
  // this multiplication factor will perform the exact division by
  // K! / 2^T.
  APInt Mod = APInt::getSignedMinValue(W+1);
  APInt MultiplyFactor = OddFactorial.zext(W+1);
  MultiplyFactor = MultiplyFactor.multiplicativeInverse(Mod);
  MultiplyFactor = MultiplyFactor.trunc(W);

  // Calculate the product, at width T+W
  const IntegerType *CalculationTy = IntegerType::get(CalculationBits);
  SCEVHandle Dividend = SE.getTruncateOrZeroExtend(It, CalculationTy);
  for (unsigned i = 1; i != K; ++i) {
    SCEVHandle S = SE.getMinusSCEV(It, SE.getIntegerSCEV(i, It->getType()));
    Dividend = SE.getMulExpr(Dividend,
                             SE.getTruncateOrZeroExtend(S, CalculationTy));
  }

  // Divide by 2^T
  SCEVHandle DivResult = SE.getUDivExpr(Dividend, SE.getConstant(DivFactor));

  // Truncate the result, and divide by K! / 2^T.

  return SE.getMulExpr(SE.getConstant(MultiplyFactor),
                       SE.getTruncateOrZeroExtend(DivResult, ResultTy));
}

/// evaluateAtIteration - Return the value of this chain of recurrences at
/// the specified iteration number.  We can evaluate this recurrence by
/// multiplying each element in the chain by the binomial coefficient
/// corresponding to it.  In other words, we can evaluate {A,+,B,+,C,+,D} as:
///
///   A*BC(It, 0) + B*BC(It, 1) + C*BC(It, 2) + D*BC(It, 3)
///
/// where BC(It, k) stands for binomial coefficient.
///
SCEVHandle SCEVAddRecExpr::evaluateAtIteration(SCEVHandle It,
                                               ScalarEvolution &SE) const {
  SCEVHandle Result = getStart();
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    // The computation is correct in the face of overflow provided that the
    // multiplication is performed _after_ the evaluation of the binomial
    // coefficient.
    SCEVHandle Coeff = BinomialCoefficient(It, i, SE,
                                           cast<IntegerType>(getType()));
    if (isa<SCEVCouldNotCompute>(Coeff))
      return Coeff;

    Result = SE.getAddExpr(Result, SE.getMulExpr(getOperand(i), Coeff));
  }
  return Result;
}

//===----------------------------------------------------------------------===//
//                    SCEV Expression folder implementations
//===----------------------------------------------------------------------===//

SCEVHandle ScalarEvolution::getTruncateExpr(const SCEVHandle &Op, const Type *Ty) {
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getUnknown(
        ConstantExpr::getTrunc(SC->getValue(), Ty));

  // If the input value is a chrec scev made out of constants, truncate
  // all of the constants.
  if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Op)) {
    std::vector<SCEVHandle> Operands;
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
      // FIXME: This should allow truncation of other expression types!
      if (isa<SCEVConstant>(AddRec->getOperand(i)))
        Operands.push_back(getTruncateExpr(AddRec->getOperand(i), Ty));
      else
        break;
    if (Operands.size() == AddRec->getNumOperands())
      return getAddRecExpr(Operands, AddRec->getLoop());
  }

  SCEVTruncateExpr *&Result = (*SCEVTruncates)[std::make_pair(Op, Ty)];
  if (Result == 0) Result = new SCEVTruncateExpr(Op, Ty);
  return Result;
}

SCEVHandle ScalarEvolution::getZeroExtendExpr(const SCEVHandle &Op, const Type *Ty) {
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getUnknown(
        ConstantExpr::getZExt(SC->getValue(), Ty));

  // FIXME: If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can zero extend all of the
  // operands (often constants).  This would allow analysis of something like
  // this:  for (unsigned char X = 0; X < 100; ++X) { int Y = X; }

  SCEVZeroExtendExpr *&Result = (*SCEVZeroExtends)[std::make_pair(Op, Ty)];
  if (Result == 0) Result = new SCEVZeroExtendExpr(Op, Ty);
  return Result;
}

SCEVHandle ScalarEvolution::getSignExtendExpr(const SCEVHandle &Op, const Type *Ty) {
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getUnknown(
        ConstantExpr::getSExt(SC->getValue(), Ty));

  // FIXME: If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can sign extend all of the
  // operands (often constants).  This would allow analysis of something like
  // this:  for (signed char X = 0; X < 100; ++X) { int Y = X; }

  SCEVSignExtendExpr *&Result = (*SCEVSignExtends)[std::make_pair(Op, Ty)];
  if (Result == 0) Result = new SCEVSignExtendExpr(Op, Ty);
  return Result;
}

/// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion
/// of the input value to the specified type.  If the type must be
/// extended, it is zero extended.
SCEVHandle ScalarEvolution::getTruncateOrZeroExtend(const SCEVHandle &V,
                                                    const Type *Ty) {
  const Type *SrcTy = V->getType();
  assert(SrcTy->isInteger() && Ty->isInteger() &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (SrcTy->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return V;  // No conversion
  if (SrcTy->getPrimitiveSizeInBits() > Ty->getPrimitiveSizeInBits())
    return getTruncateExpr(V, Ty);
  return getZeroExtendExpr(V, Ty);
}

// get - Get a canonical add expression, or something simpler if possible.
SCEVHandle ScalarEvolution::getAddExpr(std::vector<SCEVHandle> &Ops) {
  assert(!Ops.empty() && "Cannot get empty add!");
  if (Ops.size() == 1) return Ops[0];

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(LHSC->getValue()->getValue() + 
                                           RHSC->getValue()->getValue());
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant zero being added, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isZero()) {
      Ops.erase(Ops.begin());
      --Idx;
    }
  }

  if (Ops.size() == 1) return Ops[0];

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, merge them together into an multiply expression.  Since we sorted the
  // list, these values are required to be adjacent.
  const Type *Ty = Ops[0]->getType();
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X + Y + Y  -->  X + Y*2
      // Found a match, merge the two values into a multiply, and add any
      // remaining values to the result.
      SCEVHandle Two = getIntegerSCEV(2, Ty);
      SCEVHandle Mul = getMulExpr(Ops[i], Two);
      if (Ops.size() == 2)
        return Mul;
      Ops.erase(Ops.begin()+i, Ops.begin()+i+2);
      Ops.push_back(Mul);
      return getAddExpr(Ops);
    }

  // Now we know the first non-constant operand.  Skip past any cast SCEVs.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddExpr)
    ++Idx;

  // If there are add operands they would be next.
  if (Idx < Ops.size()) {
    bool DeletedAdd = false;
    while (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[Idx])) {
      // If we have an add, expand the add operands onto the end of the operands
      // list.
      Ops.insert(Ops.end(), Add->op_begin(), Add->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedAdd = true;
    }

    // If we deleted at least one add, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just aquired.
    if (DeletedAdd)
      return getAddExpr(Ops);
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // If we are adding something to a multiply expression, make sure the
  // something is not already an operand of the multiply.  If so, merge it into
  // the multiply.
  for (; Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx]); ++Idx) {
    SCEVMulExpr *Mul = cast<SCEVMulExpr>(Ops[Idx]);
    for (unsigned MulOp = 0, e = Mul->getNumOperands(); MulOp != e; ++MulOp) {
      SCEV *MulOpSCEV = Mul->getOperand(MulOp);
      for (unsigned AddOp = 0, e = Ops.size(); AddOp != e; ++AddOp)
        if (MulOpSCEV == Ops[AddOp] && !isa<SCEVConstant>(MulOpSCEV)) {
          // Fold W + X + (X * Y * Z)  -->  W + (X * ((Y*Z)+1))
          SCEVHandle InnerMul = Mul->getOperand(MulOp == 0);
          if (Mul->getNumOperands() != 2) {
            // If the multiply has more than two operands, we must get the
            // Y*Z term.
            std::vector<SCEVHandle> MulOps(Mul->op_begin(), Mul->op_end());
            MulOps.erase(MulOps.begin()+MulOp);
            InnerMul = getMulExpr(MulOps);
          }
          SCEVHandle One = getIntegerSCEV(1, Ty);
          SCEVHandle AddOne = getAddExpr(InnerMul, One);
          SCEVHandle OuterMul = getMulExpr(AddOne, Ops[AddOp]);
          if (Ops.size() == 2) return OuterMul;
          if (AddOp < Idx) {
            Ops.erase(Ops.begin()+AddOp);
            Ops.erase(Ops.begin()+Idx-1);
          } else {
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+AddOp-1);
          }
          Ops.push_back(OuterMul);
          return getAddExpr(Ops);
        }

      // Check this multiply against other multiplies being added together.
      for (unsigned OtherMulIdx = Idx+1;
           OtherMulIdx < Ops.size() && isa<SCEVMulExpr>(Ops[OtherMulIdx]);
           ++OtherMulIdx) {
        SCEVMulExpr *OtherMul = cast<SCEVMulExpr>(Ops[OtherMulIdx]);
        // If MulOp occurs in OtherMul, we can fold the two multiplies
        // together.
        for (unsigned OMulOp = 0, e = OtherMul->getNumOperands();
             OMulOp != e; ++OMulOp)
          if (OtherMul->getOperand(OMulOp) == MulOpSCEV) {
            // Fold X + (A*B*C) + (A*D*E) --> X + (A*(B*C+D*E))
            SCEVHandle InnerMul1 = Mul->getOperand(MulOp == 0);
            if (Mul->getNumOperands() != 2) {
              std::vector<SCEVHandle> MulOps(Mul->op_begin(), Mul->op_end());
              MulOps.erase(MulOps.begin()+MulOp);
              InnerMul1 = getMulExpr(MulOps);
            }
            SCEVHandle InnerMul2 = OtherMul->getOperand(OMulOp == 0);
            if (OtherMul->getNumOperands() != 2) {
              std::vector<SCEVHandle> MulOps(OtherMul->op_begin(),
                                             OtherMul->op_end());
              MulOps.erase(MulOps.begin()+OMulOp);
              InnerMul2 = getMulExpr(MulOps);
            }
            SCEVHandle InnerMulSum = getAddExpr(InnerMul1,InnerMul2);
            SCEVHandle OuterMul = getMulExpr(MulOpSCEV, InnerMulSum);
            if (Ops.size() == 2) return OuterMul;
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+OtherMulIdx-1);
            Ops.push_back(OuterMul);
            return getAddExpr(Ops);
          }
      }
    }
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this add and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    std::vector<SCEVHandle> LIOps;
    SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (Ops[i]->isLoopInvariant(AddRec->getLoop())) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI + LI + {Start,+,Step}  -->  NLI + {LI+Start,+,Step}
      LIOps.push_back(AddRec->getStart());

      std::vector<SCEVHandle> AddRecOps(AddRec->op_begin(), AddRec->op_end());
      AddRecOps[0] = getAddExpr(LIOps);

      SCEVHandle NewRec = getAddRecExpr(AddRecOps, AddRec->getLoop());
      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, add the folded AddRec by the non-liv parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getAddExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // added together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);++OtherIdx)
      if (OtherIdx != Idx) {
        SCEVAddRecExpr *OtherAddRec = cast<SCEVAddRecExpr>(Ops[OtherIdx]);
        if (AddRec->getLoop() == OtherAddRec->getLoop()) {
          // Other + {A,+,B} + {C,+,D}  -->  Other + {A+C,+,B+D}
          std::vector<SCEVHandle> NewOps(AddRec->op_begin(), AddRec->op_end());
          for (unsigned i = 0, e = OtherAddRec->getNumOperands(); i != e; ++i) {
            if (i >= NewOps.size()) {
              NewOps.insert(NewOps.end(), OtherAddRec->op_begin()+i,
                            OtherAddRec->op_end());
              break;
            }
            NewOps[i] = getAddExpr(NewOps[i], OtherAddRec->getOperand(i));
          }
          SCEVHandle NewAddRec = getAddRecExpr(NewOps, AddRec->getLoop());

          if (Ops.size() == 2) return NewAddRec;

          Ops.erase(Ops.begin()+Idx);
          Ops.erase(Ops.begin()+OtherIdx-1);
          Ops.push_back(NewAddRec);
          return getAddExpr(Ops);
        }
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an add expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = (*SCEVCommExprs)[std::make_pair(scAddExpr,
                                                                 SCEVOps)];
  if (Result == 0) Result = new SCEVAddExpr(Ops);
  return Result;
}


SCEVHandle ScalarEvolution::getMulExpr(std::vector<SCEVHandle> &Ops) {
  assert(!Ops.empty() && "Cannot get empty mul!");

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {

    // C1*(C2+V) -> C1*C2 + C1*V
    if (Ops.size() == 2)
      if (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1]))
        if (Add->getNumOperands() == 2 &&
            isa<SCEVConstant>(Add->getOperand(0)))
          return getAddExpr(getMulExpr(LHSC, Add->getOperand(0)),
                            getMulExpr(LHSC, Add->getOperand(1)));


    ++Idx;
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(LHSC->getValue()->getValue() * 
                                           RHSC->getValue()->getValue());
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant one being multiplied, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->equalsInt(1)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isZero()) {
      // If we have a multiply of zero, it will always be zero.
      return Ops[0];
    }
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  if (Ops.size() == 1)
    return Ops[0];

  // If there are mul operands inline them all into this expression.
  if (Idx < Ops.size()) {
    bool DeletedMul = false;
    while (SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[Idx])) {
      // If we have an mul, expand the mul operands onto the end of the operands
      // list.
      Ops.insert(Ops.end(), Mul->op_begin(), Mul->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedMul = true;
    }

    // If we deleted at least one mul, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just aquired.
    if (DeletedMul)
      return getMulExpr(Ops);
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this mul and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    std::vector<SCEVHandle> LIOps;
    SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (Ops[i]->isLoopInvariant(AddRec->getLoop())) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI * LI * {Start,+,Step}  -->  NLI * {LI*Start,+,LI*Step}
      std::vector<SCEVHandle> NewOps;
      NewOps.reserve(AddRec->getNumOperands());
      if (LIOps.size() == 1) {
        SCEV *Scale = LIOps[0];
        for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
          NewOps.push_back(getMulExpr(Scale, AddRec->getOperand(i)));
      } else {
        for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
          std::vector<SCEVHandle> MulOps(LIOps);
          MulOps.push_back(AddRec->getOperand(i));
          NewOps.push_back(getMulExpr(MulOps));
        }
      }

      SCEVHandle NewRec = getAddRecExpr(NewOps, AddRec->getLoop());

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, multiply the folded AddRec by the non-liv parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getMulExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // multiplied together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);++OtherIdx)
      if (OtherIdx != Idx) {
        SCEVAddRecExpr *OtherAddRec = cast<SCEVAddRecExpr>(Ops[OtherIdx]);
        if (AddRec->getLoop() == OtherAddRec->getLoop()) {
          // F * G  -->  {A,+,B} * {C,+,D}  -->  {A*C,+,F*D + G*B + B*D}
          SCEVAddRecExpr *F = AddRec, *G = OtherAddRec;
          SCEVHandle NewStart = getMulExpr(F->getStart(),
                                                 G->getStart());
          SCEVHandle B = F->getStepRecurrence(*this);
          SCEVHandle D = G->getStepRecurrence(*this);
          SCEVHandle NewStep = getAddExpr(getMulExpr(F, D),
                                          getMulExpr(G, B),
                                          getMulExpr(B, D));
          SCEVHandle NewAddRec = getAddRecExpr(NewStart, NewStep,
                                               F->getLoop());
          if (Ops.size() == 2) return NewAddRec;

          Ops.erase(Ops.begin()+Idx);
          Ops.erase(Ops.begin()+OtherIdx-1);
          Ops.push_back(NewAddRec);
          return getMulExpr(Ops);
        }
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an mul expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = (*SCEVCommExprs)[std::make_pair(scMulExpr,
                                                                 SCEVOps)];
  if (Result == 0)
    Result = new SCEVMulExpr(Ops);
  return Result;
}

SCEVHandle ScalarEvolution::getUDivExpr(const SCEVHandle &LHS, const SCEVHandle &RHS) {
  if (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
    if (RHSC->getValue()->equalsInt(1))
      return LHS;                            // X udiv 1 --> x

    if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
      Constant *LHSCV = LHSC->getValue();
      Constant *RHSCV = RHSC->getValue();
      return getUnknown(ConstantExpr::getUDiv(LHSCV, RHSCV));
    }
  }

  // FIXME: implement folding of (X*4)/4 when we know X*4 doesn't overflow.

  SCEVUDivExpr *&Result = (*SCEVUDivs)[std::make_pair(LHS, RHS)];
  if (Result == 0) Result = new SCEVUDivExpr(LHS, RHS);
  return Result;
}


/// SCEVAddRecExpr::get - Get a add recurrence expression for the
/// specified loop.  Simplify the expression as much as possible.
SCEVHandle ScalarEvolution::getAddRecExpr(const SCEVHandle &Start,
                               const SCEVHandle &Step, const Loop *L) {
  std::vector<SCEVHandle> Operands;
  Operands.push_back(Start);
  if (SCEVAddRecExpr *StepChrec = dyn_cast<SCEVAddRecExpr>(Step))
    if (StepChrec->getLoop() == L) {
      Operands.insert(Operands.end(), StepChrec->op_begin(),
                      StepChrec->op_end());
      return getAddRecExpr(Operands, L);
    }

  Operands.push_back(Step);
  return getAddRecExpr(Operands, L);
}

/// SCEVAddRecExpr::get - Get a add recurrence expression for the
/// specified loop.  Simplify the expression as much as possible.
SCEVHandle ScalarEvolution::getAddRecExpr(std::vector<SCEVHandle> &Operands,
                               const Loop *L) {
  if (Operands.size() == 1) return Operands[0];

  if (Operands.back()->isZero()) {
    Operands.pop_back();
    return getAddRecExpr(Operands, L);             // {X,+,0}  -->  X
  }

  // Canonicalize nested AddRecs in by nesting them in order of loop depth.
  if (SCEVAddRecExpr *NestedAR = dyn_cast<SCEVAddRecExpr>(Operands[0])) {
    const Loop* NestedLoop = NestedAR->getLoop();
    if (L->getLoopDepth() < NestedLoop->getLoopDepth()) {
      std::vector<SCEVHandle> NestedOperands(NestedAR->op_begin(),
                                             NestedAR->op_end());
      SCEVHandle NestedARHandle(NestedAR);
      Operands[0] = NestedAR->getStart();
      NestedOperands[0] = getAddRecExpr(Operands, L);
      return getAddRecExpr(NestedOperands, NestedLoop);
    }
  }

  SCEVAddRecExpr *&Result =
    (*SCEVAddRecExprs)[std::make_pair(L, std::vector<SCEV*>(Operands.begin(),
                                                            Operands.end()))];
  if (Result == 0) Result = new SCEVAddRecExpr(Operands, L);
  return Result;
}

SCEVHandle ScalarEvolution::getSMaxExpr(const SCEVHandle &LHS,
                                        const SCEVHandle &RHS) {
  std::vector<SCEVHandle> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getSMaxExpr(Ops);
}

SCEVHandle ScalarEvolution::getSMaxExpr(std::vector<SCEVHandle> Ops) {
  assert(!Ops.empty() && "Cannot get empty smax!");
  if (Ops.size() == 1) return Ops[0];

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(
                              APIntOps::smax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant -inf, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(true)) {
      Ops.erase(Ops.begin());
      --Idx;
    }
  }

  if (Ops.size() == 1) return Ops[0];

  // Find the first SMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scSMaxExpr)
    ++Idx;

  // Check to see if one of the operands is an SMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedSMax = false;
    while (SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(Ops[Idx])) {
      Ops.insert(Ops.end(), SMax->op_begin(), SMax->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedSMax = true;
    }

    if (DeletedSMax)
      return getSMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X smax Y smax Y  -->  X smax Y
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced smax down to nothing!");

  // Okay, it looks like we really DO need an smax expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = (*SCEVCommExprs)[std::make_pair(scSMaxExpr,
                                                                 SCEVOps)];
  if (Result == 0) Result = new SCEVSMaxExpr(Ops);
  return Result;
}

SCEVHandle ScalarEvolution::getUMaxExpr(const SCEVHandle &LHS,
                                        const SCEVHandle &RHS) {
  std::vector<SCEVHandle> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getUMaxExpr(Ops);
}

SCEVHandle ScalarEvolution::getUMaxExpr(std::vector<SCEVHandle> Ops) {
  assert(!Ops.empty() && "Cannot get empty umax!");
  if (Ops.size() == 1) return Ops[0];

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(
                              APIntOps::umax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant zero, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(false)) {
      Ops.erase(Ops.begin());
      --Idx;
    }
  }

  if (Ops.size() == 1) return Ops[0];

  // Find the first UMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scUMaxExpr)
    ++Idx;

  // Check to see if one of the operands is a UMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedUMax = false;
    while (SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(Ops[Idx])) {
      Ops.insert(Ops.end(), UMax->op_begin(), UMax->op_end());
      Ops.erase(Ops.begin()+Idx);
      DeletedUMax = true;
    }

    if (DeletedUMax)
      return getUMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X umax Y umax Y  -->  X umax Y
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced umax down to nothing!");

  // Okay, it looks like we really DO need a umax expr.  Check to see if we
  // already have one, otherwise create a new one.
  std::vector<SCEV*> SCEVOps(Ops.begin(), Ops.end());
  SCEVCommutativeExpr *&Result = (*SCEVCommExprs)[std::make_pair(scUMaxExpr,
                                                                 SCEVOps)];
  if (Result == 0) Result = new SCEVUMaxExpr(Ops);
  return Result;
}

SCEVHandle ScalarEvolution::getUnknown(Value *V) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return getConstant(CI);
  SCEVUnknown *&Result = (*SCEVUnknowns)[V];
  if (Result == 0) Result = new SCEVUnknown(V);
  return Result;
}


//===----------------------------------------------------------------------===//
//             ScalarEvolutionsImpl Definition and Implementation
//===----------------------------------------------------------------------===//
//
/// ScalarEvolutionsImpl - This class implements the main driver for the scalar
/// evolution code.
///
namespace {
  struct VISIBILITY_HIDDEN ScalarEvolutionsImpl {
    /// SE - A reference to the public ScalarEvolution object.
    ScalarEvolution &SE;

    /// F - The function we are analyzing.
    ///
    Function &F;

    /// LI - The loop information for the function we are currently analyzing.
    ///
    LoopInfo &LI;

    /// UnknownValue - This SCEV is used to represent unknown trip counts and
    /// things.
    SCEVHandle UnknownValue;

    /// Scalars - This is a cache of the scalars we have analyzed so far.
    ///
    std::map<Value*, SCEVHandle> Scalars;

    /// IterationCounts - Cache the iteration count of the loops for this
    /// function as they are computed.
    std::map<const Loop*, SCEVHandle> IterationCounts;

    /// ConstantEvolutionLoopExitValue - This map contains entries for all of
    /// the PHI instructions that we attempt to compute constant evolutions for.
    /// This allows us to avoid potentially expensive recomputation of these
    /// properties.  An instruction maps to null if we are unable to compute its
    /// exit value.
    std::map<PHINode*, Constant*> ConstantEvolutionLoopExitValue;

  public:
    ScalarEvolutionsImpl(ScalarEvolution &se, Function &f, LoopInfo &li)
      : SE(se), F(f), LI(li), UnknownValue(new SCEVCouldNotCompute()) {}

    /// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
    /// expression and create a new one.
    SCEVHandle getSCEV(Value *V);

    /// hasSCEV - Return true if the SCEV for this value has already been
    /// computed.
    bool hasSCEV(Value *V) const {
      return Scalars.count(V);
    }

    /// setSCEV - Insert the specified SCEV into the map of current SCEVs for
    /// the specified value.
    void setSCEV(Value *V, const SCEVHandle &H) {
      bool isNew = Scalars.insert(std::make_pair(V, H)).second;
      assert(isNew && "This entry already existed!");
    }


    /// getSCEVAtScope - Compute the value of the specified expression within
    /// the indicated loop (which may be null to indicate in no loop).  If the
    /// expression cannot be evaluated, return UnknownValue itself.
    SCEVHandle getSCEVAtScope(SCEV *V, const Loop *L);


    /// hasLoopInvariantIterationCount - Return true if the specified loop has
    /// an analyzable loop-invariant iteration count.
    bool hasLoopInvariantIterationCount(const Loop *L);

    /// getIterationCount - If the specified loop has a predictable iteration
    /// count, return it.  Note that it is not valid to call this method on a
    /// loop without a loop-invariant iteration count.
    SCEVHandle getIterationCount(const Loop *L);

    /// deleteValueFromRecords - This method should be called by the
    /// client before it removes a value from the program, to make sure
    /// that no dangling references are left around.
    void deleteValueFromRecords(Value *V);

  private:
    /// createSCEV - We know that there is no SCEV for the specified value.
    /// Analyze the expression.
    SCEVHandle createSCEV(Value *V);

    /// createNodeForPHI - Provide the special handling we need to analyze PHI
    /// SCEVs.
    SCEVHandle createNodeForPHI(PHINode *PN);

    /// ReplaceSymbolicValueWithConcrete - This looks up the computed SCEV value
    /// for the specified instruction and replaces any references to the
    /// symbolic value SymName with the specified value.  This is used during
    /// PHI resolution.
    void ReplaceSymbolicValueWithConcrete(Instruction *I,
                                          const SCEVHandle &SymName,
                                          const SCEVHandle &NewVal);

    /// ComputeIterationCount - Compute the number of times the specified loop
    /// will iterate.
    SCEVHandle ComputeIterationCount(const Loop *L);

    /// ComputeLoadConstantCompareIterationCount - Given an exit condition of
    /// 'icmp op load X, cst', try to see if we can compute the trip count.
    SCEVHandle ComputeLoadConstantCompareIterationCount(LoadInst *LI,
                                                        Constant *RHS,
                                                        const Loop *L,
                                                        ICmpInst::Predicate p);

    /// ComputeIterationCountExhaustively - If the trip is known to execute a
    /// constant number of times (the condition evolves only from constants),
    /// try to evaluate a few iterations of the loop until we get the exit
    /// condition gets a value of ExitWhen (true or false).  If we cannot
    /// evaluate the trip count of the loop, return UnknownValue.
    SCEVHandle ComputeIterationCountExhaustively(const Loop *L, Value *Cond,
                                                 bool ExitWhen);

    /// HowFarToZero - Return the number of times a backedge comparing the
    /// specified value to zero will execute.  If not computable, return
    /// UnknownValue.
    SCEVHandle HowFarToZero(SCEV *V, const Loop *L);

    /// HowFarToNonZero - Return the number of times a backedge checking the
    /// specified value for nonzero will execute.  If not computable, return
    /// UnknownValue.
    SCEVHandle HowFarToNonZero(SCEV *V, const Loop *L);

    /// HowManyLessThans - Return the number of times a backedge containing the
    /// specified less-than comparison will execute.  If not computable, return
    /// UnknownValue. isSigned specifies whether the less-than is signed.
    SCEVHandle HowManyLessThans(SCEV *LHS, SCEV *RHS, const Loop *L,
                                bool isSigned);

    /// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
    /// (which may not be an immediate predecessor) which has exactly one
    /// successor from which BB is reachable, or null if no such block is
    /// found.
    BasicBlock* getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB);

    /// executesAtLeastOnce - Test whether entry to the loop is protected by
    /// a conditional between LHS and RHS.
    bool executesAtLeastOnce(const Loop *L, bool isSigned, SCEV *LHS, SCEV *RHS);

    /// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
    /// in the header of its containing loop, we know the loop executes a
    /// constant number of times, and the PHI node is just a recurrence
    /// involving constants, fold it.
    Constant *getConstantEvolutionLoopExitValue(PHINode *PN, const APInt& Its,
                                                const Loop *L);
  };
}

//===----------------------------------------------------------------------===//
//            Basic SCEV Analysis and PHI Idiom Recognition Code
//

/// deleteValueFromRecords - This method should be called by the
/// client before it removes an instruction from the program, to make sure
/// that no dangling references are left around.
void ScalarEvolutionsImpl::deleteValueFromRecords(Value *V) {
  SmallVector<Value *, 16> Worklist;

  if (Scalars.erase(V)) {
    if (PHINode *PN = dyn_cast<PHINode>(V))
      ConstantEvolutionLoopExitValue.erase(PN);
    Worklist.push_back(V);
  }

  while (!Worklist.empty()) {
    Value *VV = Worklist.back();
    Worklist.pop_back();

    for (Instruction::use_iterator UI = VV->use_begin(), UE = VV->use_end();
         UI != UE; ++UI) {
      Instruction *Inst = cast<Instruction>(*UI);
      if (Scalars.erase(Inst)) {
        if (PHINode *PN = dyn_cast<PHINode>(VV))
          ConstantEvolutionLoopExitValue.erase(PN);
        Worklist.push_back(Inst);
      }
    }
  }
}


/// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
/// expression and create a new one.
SCEVHandle ScalarEvolutionsImpl::getSCEV(Value *V) {
  assert(V->getType() != Type::VoidTy && "Can't analyze void expressions!");

  std::map<Value*, SCEVHandle>::iterator I = Scalars.find(V);
  if (I != Scalars.end()) return I->second;
  SCEVHandle S = createSCEV(V);
  Scalars.insert(std::make_pair(V, S));
  return S;
}

/// ReplaceSymbolicValueWithConcrete - This looks up the computed SCEV value for
/// the specified instruction and replaces any references to the symbolic value
/// SymName with the specified value.  This is used during PHI resolution.
void ScalarEvolutionsImpl::
ReplaceSymbolicValueWithConcrete(Instruction *I, const SCEVHandle &SymName,
                                 const SCEVHandle &NewVal) {
  std::map<Value*, SCEVHandle>::iterator SI = Scalars.find(I);
  if (SI == Scalars.end()) return;

  SCEVHandle NV =
    SI->second->replaceSymbolicValuesWithConcrete(SymName, NewVal, SE);
  if (NV == SI->second) return;  // No change.

  SI->second = NV;       // Update the scalars map!

  // Any instruction values that use this instruction might also need to be
  // updated!
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI)
    ReplaceSymbolicValueWithConcrete(cast<Instruction>(*UI), SymName, NewVal);
}

/// createNodeForPHI - PHI nodes have two cases.  Either the PHI node exists in
/// a loop header, making it a potential recurrence, or it doesn't.
///
SCEVHandle ScalarEvolutionsImpl::createNodeForPHI(PHINode *PN) {
  if (PN->getNumIncomingValues() == 2)  // The loops have been canonicalized.
    if (const Loop *L = LI.getLoopFor(PN->getParent()))
      if (L->getHeader() == PN->getParent()) {
        // If it lives in the loop header, it has two incoming values, one
        // from outside the loop, and one from inside.
        unsigned IncomingEdge = L->contains(PN->getIncomingBlock(0));
        unsigned BackEdge     = IncomingEdge^1;

        // While we are analyzing this PHI node, handle its value symbolically.
        SCEVHandle SymbolicName = SE.getUnknown(PN);
        assert(Scalars.find(PN) == Scalars.end() &&
               "PHI node already processed?");
        Scalars.insert(std::make_pair(PN, SymbolicName));

        // Using this symbolic name for the PHI, analyze the value coming around
        // the back-edge.
        SCEVHandle BEValue = getSCEV(PN->getIncomingValue(BackEdge));

        // NOTE: If BEValue is loop invariant, we know that the PHI node just
        // has a special value for the first iteration of the loop.

        // If the value coming around the backedge is an add with the symbolic
        // value we just inserted, then we found a simple induction variable!
        if (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(BEValue)) {
          // If there is a single occurrence of the symbolic value, replace it
          // with a recurrence.
          unsigned FoundIndex = Add->getNumOperands();
          for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
            if (Add->getOperand(i) == SymbolicName)
              if (FoundIndex == e) {
                FoundIndex = i;
                break;
              }

          if (FoundIndex != Add->getNumOperands()) {
            // Create an add with everything but the specified operand.
            std::vector<SCEVHandle> Ops;
            for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
              if (i != FoundIndex)
                Ops.push_back(Add->getOperand(i));
            SCEVHandle Accum = SE.getAddExpr(Ops);

            // This is not a valid addrec if the step amount is varying each
            // loop iteration, but is not itself an addrec in this loop.
            if (Accum->isLoopInvariant(L) ||
                (isa<SCEVAddRecExpr>(Accum) &&
                 cast<SCEVAddRecExpr>(Accum)->getLoop() == L)) {
              SCEVHandle StartVal = getSCEV(PN->getIncomingValue(IncomingEdge));
              SCEVHandle PHISCEV  = SE.getAddRecExpr(StartVal, Accum, L);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and update all of the
              // entries for the scalars that use the PHI (except for the PHI
              // itself) to use the new analyzed value instead of the "symbolic"
              // value.
              ReplaceSymbolicValueWithConcrete(PN, SymbolicName, PHISCEV);
              return PHISCEV;
            }
          }
        } else if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(BEValue)) {
          // Otherwise, this could be a loop like this:
          //     i = 0;  for (j = 1; ..; ++j) { ....  i = j; }
          // In this case, j = {1,+,1}  and BEValue is j.
          // Because the other in-value of i (0) fits the evolution of BEValue
          // i really is an addrec evolution.
          if (AddRec->getLoop() == L && AddRec->isAffine()) {
            SCEVHandle StartVal = getSCEV(PN->getIncomingValue(IncomingEdge));

            // If StartVal = j.start - j.stride, we can use StartVal as the
            // initial step of the addrec evolution.
            if (StartVal == SE.getMinusSCEV(AddRec->getOperand(0),
                                            AddRec->getOperand(1))) {
              SCEVHandle PHISCEV = 
                 SE.getAddRecExpr(StartVal, AddRec->getOperand(1), L);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and update all of the
              // entries for the scalars that use the PHI (except for the PHI
              // itself) to use the new analyzed value instead of the "symbolic"
              // value.
              ReplaceSymbolicValueWithConcrete(PN, SymbolicName, PHISCEV);
              return PHISCEV;
            }
          }
        }

        return SymbolicName;
      }

  // If it's not a loop phi, we can't handle it yet.
  return SE.getUnknown(PN);
}

/// GetMinTrailingZeros - Determine the minimum number of zero bits that S is
/// guaranteed to end in (at every loop iteration).  It is, at the same time,
/// the minimum number of times S is divisible by 2.  For example, given {4,+,8}
/// it returns 2.  If S is guaranteed to be 0, it returns the bitwidth of S.
static uint32_t GetMinTrailingZeros(SCEVHandle S) {
  if (SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return C->getValue()->getValue().countTrailingZeros();

  if (SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(S))
    return std::min(GetMinTrailingZeros(T->getOperand()), T->getBitWidth());

  if (SCEVZeroExtendExpr *E = dyn_cast<SCEVZeroExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == E->getOperand()->getBitWidth() ? E->getBitWidth() : OpRes;
  }

  if (SCEVSignExtendExpr *E = dyn_cast<SCEVSignExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == E->getOperand()->getBitWidth() ? E->getBitWidth() : OpRes;
  }

  if (SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S)) {
    // The result is the sum of all operands results.
    uint32_t SumOpRes = GetMinTrailingZeros(M->getOperand(0));
    uint32_t BitWidth = M->getBitWidth();
    for (unsigned i = 1, e = M->getNumOperands();
         SumOpRes != BitWidth && i != e; ++i)
      SumOpRes = std::min(SumOpRes + GetMinTrailingZeros(M->getOperand(i)),
                          BitWidth);
    return SumOpRes;
  }

  if (SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (SCEVSMaxExpr *M = dyn_cast<SCEVSMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  if (SCEVUMaxExpr *M = dyn_cast<SCEVUMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  // SCEVUDivExpr, SCEVUnknown
  return 0;
}

/// createSCEV - We know that there is no SCEV for the specified value.
/// Analyze the expression.
///
SCEVHandle ScalarEvolutionsImpl::createSCEV(Value *V) {
  if (!isa<IntegerType>(V->getType()))
    return SE.getUnknown(V);
    
  unsigned Opcode = Instruction::UserOp1;
  if (Instruction *I = dyn_cast<Instruction>(V))
    Opcode = I->getOpcode();
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    Opcode = CE->getOpcode();
  else
    return SE.getUnknown(V);

  User *U = cast<User>(V);
  switch (Opcode) {
  case Instruction::Add:
    return SE.getAddExpr(getSCEV(U->getOperand(0)),
                         getSCEV(U->getOperand(1)));
  case Instruction::Mul:
    return SE.getMulExpr(getSCEV(U->getOperand(0)),
                         getSCEV(U->getOperand(1)));
  case Instruction::UDiv:
    return SE.getUDivExpr(getSCEV(U->getOperand(0)),
                          getSCEV(U->getOperand(1)));
  case Instruction::Sub:
    return SE.getMinusSCEV(getSCEV(U->getOperand(0)),
                           getSCEV(U->getOperand(1)));
  case Instruction::Or:
    // If the RHS of the Or is a constant, we may have something like:
    // X*4+1 which got turned into X*4|1.  Handle this as an Add so loop
    // optimizations will transparently handle this case.
    //
    // In order for this transformation to be safe, the LHS must be of the
    // form X*(2^n) and the Or constant must be less than 2^n.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      SCEVHandle LHS = getSCEV(U->getOperand(0));
      const APInt &CIVal = CI->getValue();
      if (GetMinTrailingZeros(LHS) >=
          (CIVal.getBitWidth() - CIVal.countLeadingZeros()))
        return SE.getAddExpr(LHS, getSCEV(U->getOperand(1)));
    }
    break;
  case Instruction::Xor:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      // If the RHS of the xor is a signbit, then this is just an add.
      // Instcombine turns add of signbit into xor as a strength reduction step.
      if (CI->getValue().isSignBit())
        return SE.getAddExpr(getSCEV(U->getOperand(0)),
                             getSCEV(U->getOperand(1)));

      // If the RHS of xor is -1, then this is a not operation.
      else if (CI->isAllOnesValue())
        return SE.getNotSCEV(getSCEV(U->getOperand(0)));
    }
    break;

  case Instruction::Shl:
    // Turn shift left of a constant amount into a multiply.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
      Constant *X = ConstantInt::get(
        APInt(BitWidth, 1).shl(SA->getLimitedValue(BitWidth)));
      return SE.getMulExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::LShr:
    // Turn logical shift right of a constant into a unsigned divide.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
      Constant *X = ConstantInt::get(
        APInt(BitWidth, 1).shl(SA->getLimitedValue(BitWidth)));
      return SE.getUDivExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::Trunc:
    return SE.getTruncateExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::ZExt:
    return SE.getZeroExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::SExt:
    return SE.getSignExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::BitCast:
    // BitCasts are no-op casts so we just eliminate the cast.
    if (U->getType()->isInteger() &&
        U->getOperand(0)->getType()->isInteger())
      return getSCEV(U->getOperand(0));
    break;

  case Instruction::PHI:
    return createNodeForPHI(cast<PHINode>(U));

  case Instruction::Select:
    // This could be a smax or umax that was lowered earlier.
    // Try to recover it.
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(U->getOperand(0))) {
      Value *LHS = ICI->getOperand(0);
      Value *RHS = ICI->getOperand(1);
      switch (ICI->getPredicate()) {
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE:
        if (LHS == U->getOperand(1) && RHS == U->getOperand(2))
          return SE.getSMaxExpr(getSCEV(LHS), getSCEV(RHS));
        else if (LHS == U->getOperand(2) && RHS == U->getOperand(1))
          // ~smax(~x, ~y) == smin(x, y).
          return SE.getNotSCEV(SE.getSMaxExpr(
                                   SE.getNotSCEV(getSCEV(LHS)),
                                   SE.getNotSCEV(getSCEV(RHS))));
        break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE:
        if (LHS == U->getOperand(1) && RHS == U->getOperand(2))
          return SE.getUMaxExpr(getSCEV(LHS), getSCEV(RHS));
        else if (LHS == U->getOperand(2) && RHS == U->getOperand(1))
          // ~umax(~x, ~y) == umin(x, y)
          return SE.getNotSCEV(SE.getUMaxExpr(SE.getNotSCEV(getSCEV(LHS)),
                                              SE.getNotSCEV(getSCEV(RHS))));
        break;
      default:
        break;
      }
    }

  default: // We cannot analyze this expression.
    break;
  }

  return SE.getUnknown(V);
}



//===----------------------------------------------------------------------===//
//                   Iteration Count Computation Code
//

/// getIterationCount - If the specified loop has a predictable iteration
/// count, return it.  Note that it is not valid to call this method on a
/// loop without a loop-invariant iteration count.
SCEVHandle ScalarEvolutionsImpl::getIterationCount(const Loop *L) {
  std::map<const Loop*, SCEVHandle>::iterator I = IterationCounts.find(L);
  if (I == IterationCounts.end()) {
    SCEVHandle ItCount = ComputeIterationCount(L);
    I = IterationCounts.insert(std::make_pair(L, ItCount)).first;
    if (ItCount != UnknownValue) {
      assert(ItCount->isLoopInvariant(L) &&
             "Computed trip count isn't loop invariant for loop!");
      ++NumTripCountsComputed;
    } else if (isa<PHINode>(L->getHeader()->begin())) {
      // Only count loops that have phi nodes as not being computable.
      ++NumTripCountsNotComputed;
    }
  }
  return I->second;
}

/// ComputeIterationCount - Compute the number of times the specified loop
/// will iterate.
SCEVHandle ScalarEvolutionsImpl::ComputeIterationCount(const Loop *L) {
  // If the loop has a non-one exit block count, we can't analyze it.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1) return UnknownValue;

  // Okay, there is one exit block.  Try to find the condition that causes the
  // loop to be exited.
  BasicBlock *ExitBlock = ExitBlocks[0];

  BasicBlock *ExitingBlock = 0;
  for (pred_iterator PI = pred_begin(ExitBlock), E = pred_end(ExitBlock);
       PI != E; ++PI)
    if (L->contains(*PI)) {
      if (ExitingBlock == 0)
        ExitingBlock = *PI;
      else
        return UnknownValue;   // More than one block exiting!
    }
  assert(ExitingBlock && "No exits from loop, something is broken!");

  // Okay, we've computed the exiting block.  See what condition causes us to
  // exit.
  //
  // FIXME: we should be able to handle switch instructions (with a single exit)
  BranchInst *ExitBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (ExitBr == 0) return UnknownValue;
  assert(ExitBr->isConditional() && "If unconditional, it can't be in loop!");
  
  // At this point, we know we have a conditional branch that determines whether
  // the loop is exited.  However, we don't know if the branch is executed each
  // time through the loop.  If not, then the execution count of the branch will
  // not be equal to the trip count of the loop.
  //
  // Currently we check for this by checking to see if the Exit branch goes to
  // the loop header.  If so, we know it will always execute the same number of
  // times as the loop.  We also handle the case where the exit block *is* the
  // loop header.  This is common for un-rotated loops.  More extensive analysis
  // could be done to handle more cases here.
  if (ExitBr->getSuccessor(0) != L->getHeader() &&
      ExitBr->getSuccessor(1) != L->getHeader() &&
      ExitBr->getParent() != L->getHeader())
    return UnknownValue;
  
  ICmpInst *ExitCond = dyn_cast<ICmpInst>(ExitBr->getCondition());

  // If it's not an integer comparison then compute it the hard way. 
  // Note that ICmpInst deals with pointer comparisons too so we must check
  // the type of the operand.
  if (ExitCond == 0 || isa<PointerType>(ExitCond->getOperand(0)->getType()))
    return ComputeIterationCountExhaustively(L, ExitBr->getCondition(),
                                          ExitBr->getSuccessor(0) == ExitBlock);

  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Cond;
  if (ExitBr->getSuccessor(1) == ExitBlock)
    Cond = ExitCond->getPredicate();
  else
    Cond = ExitCond->getInversePredicate();

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      SCEVHandle ItCnt =
        ComputeLoadConstantCompareIterationCount(LI, RHS, L, Cond);
      if (!isa<SCEVCouldNotCompute>(ItCnt)) return ItCnt;
    }

  SCEVHandle LHS = getSCEV(ExitCond->getOperand(0));
  SCEVHandle RHS = getSCEV(ExitCond->getOperand(1));

  // Try to evaluate any dependencies out of the loop.
  SCEVHandle Tmp = getSCEVAtScope(LHS, L);
  if (!isa<SCEVCouldNotCompute>(Tmp)) LHS = Tmp;
  Tmp = getSCEVAtScope(RHS, L);
  if (!isa<SCEVCouldNotCompute>(Tmp)) RHS = Tmp;

  // At this point, we would like to compute how many iterations of the 
  // loop the predicate will return true for these inputs.
  if (LHS->isLoopInvariant(L) && !RHS->isLoopInvariant(L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Cond = ICmpInst::getSwappedPredicate(Cond);
  }

  // FIXME: think about handling pointer comparisons!  i.e.:
  // while (P != P+100) ++P;

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the comparison range using the constant of the correct type so
        // that the ConstantRange class knows to do a signed or unsigned
        // comparison.
        ConstantInt *CompVal = RHSC->getValue();
        const Type *RealTy = ExitCond->getOperand(0)->getType();
        CompVal = dyn_cast<ConstantInt>(
          ConstantExpr::getBitCast(CompVal, RealTy));
        if (CompVal) {
          // Form the constant range.
          ConstantRange CompRange(
              ICmpInst::makeConstantRange(Cond, CompVal->getValue()));

          SCEVHandle Ret = AddRec->getNumIterationsInRange(CompRange, SE);
          if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
        }
      }

  switch (Cond) {
  case ICmpInst::ICMP_NE: {                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    SCEVHandle TC = HowFarToZero(SE.getMinusSCEV(LHS, RHS), L);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  case ICmpInst::ICMP_EQ: {
    // Convert to: while (X-Y == 0)           // while (X == Y)
    SCEVHandle TC = HowFarToNonZero(SE.getMinusSCEV(LHS, RHS), L);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  case ICmpInst::ICMP_SLT: {
    SCEVHandle TC = HowManyLessThans(LHS, RHS, L, true);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  case ICmpInst::ICMP_SGT: {
    SCEVHandle TC = HowManyLessThans(SE.getNotSCEV(LHS),
                                     SE.getNotSCEV(RHS), L, true);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  case ICmpInst::ICMP_ULT: {
    SCEVHandle TC = HowManyLessThans(LHS, RHS, L, false);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  case ICmpInst::ICMP_UGT: {
    SCEVHandle TC = HowManyLessThans(SE.getNotSCEV(LHS),
                                     SE.getNotSCEV(RHS), L, false);
    if (!isa<SCEVCouldNotCompute>(TC)) return TC;
    break;
  }
  default:
#if 0
    cerr << "ComputeIterationCount ";
    if (ExitCond->getOperand(0)->getType()->isUnsigned())
      cerr << "[unsigned] ";
    cerr << *LHS << "   "
         << Instruction::getOpcodeName(Instruction::ICmp) 
         << "   " << *RHS << "\n";
#endif
    break;
  }
  return ComputeIterationCountExhaustively(L, ExitCond,
                                       ExitBr->getSuccessor(0) == ExitBlock);
}

static ConstantInt *
EvaluateConstantChrecAtConstant(const SCEVAddRecExpr *AddRec, ConstantInt *C,
                                ScalarEvolution &SE) {
  SCEVHandle InVal = SE.getConstant(C);
  SCEVHandle Val = AddRec->evaluateAtIteration(InVal, SE);
  assert(isa<SCEVConstant>(Val) &&
         "Evaluation of SCEV at constant didn't fold correctly?");
  return cast<SCEVConstant>(Val)->getValue();
}

/// GetAddressedElementFromGlobal - Given a global variable with an initializer
/// and a GEP expression (missing the pointer index) indexing into it, return
/// the addressed element of the initializer or null if the index expression is
/// invalid.
static Constant *
GetAddressedElementFromGlobal(GlobalVariable *GV,
                              const std::vector<ConstantInt*> &Indices) {
  Constant *Init = GV->getInitializer();
  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    uint64_t Idx = Indices[i]->getZExtValue();
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Init)) {
      assert(Idx < CS->getNumOperands() && "Bad struct index!");
      Init = cast<Constant>(CS->getOperand(Idx));
    } else if (ConstantArray *CA = dyn_cast<ConstantArray>(Init)) {
      if (Idx >= CA->getNumOperands()) return 0;  // Bogus program
      Init = cast<Constant>(CA->getOperand(Idx));
    } else if (isa<ConstantAggregateZero>(Init)) {
      if (const StructType *STy = dyn_cast<StructType>(Init->getType())) {
        assert(Idx < STy->getNumElements() && "Bad struct index!");
        Init = Constant::getNullValue(STy->getElementType(Idx));
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Init->getType())) {
        if (Idx >= ATy->getNumElements()) return 0;  // Bogus program
        Init = Constant::getNullValue(ATy->getElementType());
      } else {
        assert(0 && "Unknown constant aggregate type!");
      }
      return 0;
    } else {
      return 0; // Unknown initializer type
    }
  }
  return Init;
}

/// ComputeLoadConstantCompareIterationCount - Given an exit condition of
/// 'icmp op load X, cst', try to see if we can compute the trip count.
SCEVHandle ScalarEvolutionsImpl::
ComputeLoadConstantCompareIterationCount(LoadInst *LI, Constant *RHS,
                                         const Loop *L, 
                                         ICmpInst::Predicate predicate) {
  if (LI->isVolatile()) return UnknownValue;

  // Check to see if the loaded pointer is a getelementptr of a global.
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(LI->getOperand(0));
  if (!GEP) return UnknownValue;

  // Make sure that it is really a constant global we are gepping, with an
  // initializer, and make sure the first IDX is really 0.
  GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer() ||
      GEP->getNumOperands() < 3 || !isa<Constant>(GEP->getOperand(1)) ||
      !cast<Constant>(GEP->getOperand(1))->isNullValue())
    return UnknownValue;

  // Okay, we allow one non-constant index into the GEP instruction.
  Value *VarIdx = 0;
  std::vector<ConstantInt*> Indexes;
  unsigned VarIdxNum = 0;
  for (unsigned i = 2, e = GEP->getNumOperands(); i != e; ++i)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
      Indexes.push_back(CI);
    } else if (!isa<ConstantInt>(GEP->getOperand(i))) {
      if (VarIdx) return UnknownValue;  // Multiple non-constant idx's.
      VarIdx = GEP->getOperand(i);
      VarIdxNum = i-2;
      Indexes.push_back(0);
    }

  // Okay, we know we have a (load (gep GV, 0, X)) comparison with a constant.
  // Check to see if X is a loop variant variable value now.
  SCEVHandle Idx = getSCEV(VarIdx);
  SCEVHandle Tmp = getSCEVAtScope(Idx, L);
  if (!isa<SCEVCouldNotCompute>(Tmp)) Idx = Tmp;

  // We can only recognize very limited forms of loop index expressions, in
  // particular, only affine AddRec's like {C1,+,C2}.
  SCEVAddRecExpr *IdxExpr = dyn_cast<SCEVAddRecExpr>(Idx);
  if (!IdxExpr || !IdxExpr->isAffine() || IdxExpr->isLoopInvariant(L) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(0)) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(1)))
    return UnknownValue;

  unsigned MaxSteps = MaxBruteForceIterations;
  for (unsigned IterationNum = 0; IterationNum != MaxSteps; ++IterationNum) {
    ConstantInt *ItCst =
      ConstantInt::get(IdxExpr->getType(), IterationNum);
    ConstantInt *Val = EvaluateConstantChrecAtConstant(IdxExpr, ItCst, SE);

    // Form the GEP offset.
    Indexes[VarIdxNum] = Val;

    Constant *Result = GetAddressedElementFromGlobal(GV, Indexes);
    if (Result == 0) break;  // Cannot compute!

    // Evaluate the condition for this iteration.
    Result = ConstantExpr::getICmp(predicate, Result, RHS);
    if (!isa<ConstantInt>(Result)) break;  // Couldn't decide for sure
    if (cast<ConstantInt>(Result)->getValue().isMinValue()) {
#if 0
      cerr << "\n***\n*** Computed loop count " << *ItCst
           << "\n*** From global " << *GV << "*** BB: " << *L->getHeader()
           << "***\n";
#endif
      ++NumArrayLenItCounts;
      return SE.getConstant(ItCst);   // Found terminating iteration!
    }
  }
  return UnknownValue;
}


/// CanConstantFold - Return true if we can constant fold an instruction of the
/// specified type, assuming that all operands were constants.
static bool CanConstantFold(const Instruction *I) {
  if (isa<BinaryOperator>(I) || isa<CmpInst>(I) ||
      isa<SelectInst>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I))
    return true;

  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction())
      return canConstantFoldCallTo(F);
  return false;
}

/// getConstantEvolvingPHI - Given an LLVM value and a loop, return a PHI node
/// in the loop that V is derived from.  We allow arbitrary operations along the
/// way, but the operands of an operation must either be constants or a value
/// derived from a constant PHI.  If this expression does not fit with these
/// constraints, return null.
static PHINode *getConstantEvolvingPHI(Value *V, const Loop *L) {
  // If this is not an instruction, or if this is an instruction outside of the
  // loop, it can't be derived from a loop PHI.
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0 || !L->contains(I->getParent())) return 0;

  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (L->getHeader() == I->getParent())
      return PN;
    else
      // We don't currently keep track of the control flow needed to evaluate
      // PHIs, so we cannot handle PHIs inside of loops.
      return 0;
  }

  // If we won't be able to constant fold this expression even if the operands
  // are constants, return early.
  if (!CanConstantFold(I)) return 0;

  // Otherwise, we can evaluate this instruction if all of its operands are
  // constant or derived from a PHI node themselves.
  PHINode *PHI = 0;
  for (unsigned Op = 0, e = I->getNumOperands(); Op != e; ++Op)
    if (!(isa<Constant>(I->getOperand(Op)) ||
          isa<GlobalValue>(I->getOperand(Op)))) {
      PHINode *P = getConstantEvolvingPHI(I->getOperand(Op), L);
      if (P == 0) return 0;  // Not evolving from PHI
      if (PHI == 0)
        PHI = P;
      else if (PHI != P)
        return 0;  // Evolving from multiple different PHIs.
    }

  // This is a expression evolving from a constant PHI!
  return PHI;
}

/// EvaluateExpression - Given an expression that passes the
/// getConstantEvolvingPHI predicate, evaluate its value assuming the PHI node
/// in the loop has the value PHIVal.  If we can't fold this expression for some
/// reason, return null.
static Constant *EvaluateExpression(Value *V, Constant *PHIVal) {
  if (isa<PHINode>(V)) return PHIVal;
  if (Constant *C = dyn_cast<Constant>(V)) return C;
  Instruction *I = cast<Instruction>(V);

  std::vector<Constant*> Operands;
  Operands.resize(I->getNumOperands());

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Operands[i] = EvaluateExpression(I->getOperand(i), PHIVal);
    if (Operands[i] == 0) return 0;
  }

  if (const CmpInst *CI = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(CI->getPredicate(),
                                           &Operands[0], Operands.size());
  else
    return ConstantFoldInstOperands(I->getOpcode(), I->getType(),
                                    &Operands[0], Operands.size());
}

/// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
/// in the header of its containing loop, we know the loop executes a
/// constant number of times, and the PHI node is just a recurrence
/// involving constants, fold it.
Constant *ScalarEvolutionsImpl::
getConstantEvolutionLoopExitValue(PHINode *PN, const APInt& Its, const Loop *L){
  std::map<PHINode*, Constant*>::iterator I =
    ConstantEvolutionLoopExitValue.find(PN);
  if (I != ConstantEvolutionLoopExitValue.end())
    return I->second;

  if (Its.ugt(APInt(Its.getBitWidth(),MaxBruteForceIterations)))
    return ConstantEvolutionLoopExitValue[PN] = 0;  // Not going to evaluate it.

  Constant *&RetVal = ConstantEvolutionLoopExitValue[PN];

  // Since the loop is canonicalized, the PHI node must have two entries.  One
  // entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  Constant *StartCST =
    dyn_cast<Constant>(PN->getIncomingValue(!SecondIsBackedge));
  if (StartCST == 0)
    return RetVal = 0;  // Must be a constant.

  Value *BEValue = PN->getIncomingValue(SecondIsBackedge);
  PHINode *PN2 = getConstantEvolvingPHI(BEValue, L);
  if (PN2 != PN)
    return RetVal = 0;  // Not derived from same PHI.

  // Execute the loop symbolically to determine the exit value.
  if (Its.getActiveBits() >= 32)
    return RetVal = 0; // More than 2^32-1 iterations?? Not doing it!

  unsigned NumIterations = Its.getZExtValue(); // must be in range
  unsigned IterationNum = 0;
  for (Constant *PHIVal = StartCST; ; ++IterationNum) {
    if (IterationNum == NumIterations)
      return RetVal = PHIVal;  // Got exit value!

    // Compute the value of the PHI node for the next iteration.
    Constant *NextPHI = EvaluateExpression(BEValue, PHIVal);
    if (NextPHI == PHIVal)
      return RetVal = NextPHI;  // Stopped evolving!
    if (NextPHI == 0)
      return 0;        // Couldn't evaluate!
    PHIVal = NextPHI;
  }
}

/// ComputeIterationCountExhaustively - If the trip is known to execute a
/// constant number of times (the condition evolves only from constants),
/// try to evaluate a few iterations of the loop until we get the exit
/// condition gets a value of ExitWhen (true or false).  If we cannot
/// evaluate the trip count of the loop, return UnknownValue.
SCEVHandle ScalarEvolutionsImpl::
ComputeIterationCountExhaustively(const Loop *L, Value *Cond, bool ExitWhen) {
  PHINode *PN = getConstantEvolvingPHI(Cond, L);
  if (PN == 0) return UnknownValue;

  // Since the loop is canonicalized, the PHI node must have two entries.  One
  // entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  Constant *StartCST =
    dyn_cast<Constant>(PN->getIncomingValue(!SecondIsBackedge));
  if (StartCST == 0) return UnknownValue;  // Must be a constant.

  Value *BEValue = PN->getIncomingValue(SecondIsBackedge);
  PHINode *PN2 = getConstantEvolvingPHI(BEValue, L);
  if (PN2 != PN) return UnknownValue;  // Not derived from same PHI.

  // Okay, we find a PHI node that defines the trip count of this loop.  Execute
  // the loop symbolically to determine when the condition gets a value of
  // "ExitWhen".
  unsigned IterationNum = 0;
  unsigned MaxIterations = MaxBruteForceIterations;   // Limit analysis.
  for (Constant *PHIVal = StartCST;
       IterationNum != MaxIterations; ++IterationNum) {
    ConstantInt *CondVal =
      dyn_cast_or_null<ConstantInt>(EvaluateExpression(Cond, PHIVal));

    // Couldn't symbolically evaluate.
    if (!CondVal) return UnknownValue;

    if (CondVal->getValue() == uint64_t(ExitWhen)) {
      ConstantEvolutionLoopExitValue[PN] = PHIVal;
      ++NumBruteForceTripCountsComputed;
      return SE.getConstant(ConstantInt::get(Type::Int32Ty, IterationNum));
    }

    // Compute the value of the PHI node for the next iteration.
    Constant *NextPHI = EvaluateExpression(BEValue, PHIVal);
    if (NextPHI == 0 || NextPHI == PHIVal)
      return UnknownValue;  // Couldn't evaluate or not making progress...
    PHIVal = NextPHI;
  }

  // Too many iterations were needed to evaluate.
  return UnknownValue;
}

/// getSCEVAtScope - Compute the value of the specified expression within the
/// indicated loop (which may be null to indicate in no loop).  If the
/// expression cannot be evaluated, return UnknownValue.
SCEVHandle ScalarEvolutionsImpl::getSCEVAtScope(SCEV *V, const Loop *L) {
  // FIXME: this should be turned into a virtual method on SCEV!

  if (isa<SCEVConstant>(V)) return V;

  // If this instruction is evolved from a constant-evolving PHI, compute the
  // exit value from the loop without using SCEVs.
  if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V)) {
    if (Instruction *I = dyn_cast<Instruction>(SU->getValue())) {
      const Loop *LI = this->LI[I->getParent()];
      if (LI && LI->getParentLoop() == L)  // Looking for loop exit value.
        if (PHINode *PN = dyn_cast<PHINode>(I))
          if (PN->getParent() == LI->getHeader()) {
            // Okay, there is no closed form solution for the PHI node.  Check
            // to see if the loop that contains it has a known iteration count.
            // If so, we may be able to force computation of the exit value.
            SCEVHandle IterationCount = getIterationCount(LI);
            if (SCEVConstant *ICC = dyn_cast<SCEVConstant>(IterationCount)) {
              // Okay, we know how many times the containing loop executes.  If
              // this is a constant evolving PHI node, get the final value at
              // the specified iteration number.
              Constant *RV = getConstantEvolutionLoopExitValue(PN,
                                                    ICC->getValue()->getValue(),
                                                               LI);
              if (RV) return SE.getUnknown(RV);
            }
          }

      // Okay, this is an expression that we cannot symbolically evaluate
      // into a SCEV.  Check to see if it's possible to symbolically evaluate
      // the arguments into constants, and if so, try to constant propagate the
      // result.  This is particularly useful for computing loop exit values.
      if (CanConstantFold(I)) {
        std::vector<Constant*> Operands;
        Operands.reserve(I->getNumOperands());
        for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
          Value *Op = I->getOperand(i);
          if (Constant *C = dyn_cast<Constant>(Op)) {
            Operands.push_back(C);
          } else {
            // If any of the operands is non-constant and if they are
            // non-integer, don't even try to analyze them with scev techniques.
            if (!isa<IntegerType>(Op->getType()))
              return V;
              
            SCEVHandle OpV = getSCEVAtScope(getSCEV(Op), L);
            if (SCEVConstant *SC = dyn_cast<SCEVConstant>(OpV))
              Operands.push_back(ConstantExpr::getIntegerCast(SC->getValue(), 
                                                              Op->getType(), 
                                                              false));
            else if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(OpV)) {
              if (Constant *C = dyn_cast<Constant>(SU->getValue()))
                Operands.push_back(ConstantExpr::getIntegerCast(C, 
                                                                Op->getType(), 
                                                                false));
              else
                return V;
            } else {
              return V;
            }
          }
        }
        
        Constant *C;
        if (const CmpInst *CI = dyn_cast<CmpInst>(I))
          C = ConstantFoldCompareInstOperands(CI->getPredicate(),
                                              &Operands[0], Operands.size());
        else
          C = ConstantFoldInstOperands(I->getOpcode(), I->getType(),
                                       &Operands[0], Operands.size());
        return SE.getUnknown(C);
      }
    }

    // This is some other type of SCEVUnknown, just return it.
    return V;
  }

  if (SCEVCommutativeExpr *Comm = dyn_cast<SCEVCommutativeExpr>(V)) {
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = Comm->getNumOperands(); i != e; ++i) {
      SCEVHandle OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
      if (OpAtScope != Comm->getOperand(i)) {
        if (OpAtScope == UnknownValue) return UnknownValue;
        // Okay, at least one of these operands is loop variant but might be
        // foldable.  Build a new instance of the folded commutative expression.
        std::vector<SCEVHandle> NewOps(Comm->op_begin(), Comm->op_begin()+i);
        NewOps.push_back(OpAtScope);

        for (++i; i != e; ++i) {
          OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
          if (OpAtScope == UnknownValue) return UnknownValue;
          NewOps.push_back(OpAtScope);
        }
        if (isa<SCEVAddExpr>(Comm))
          return SE.getAddExpr(NewOps);
        if (isa<SCEVMulExpr>(Comm))
          return SE.getMulExpr(NewOps);
        if (isa<SCEVSMaxExpr>(Comm))
          return SE.getSMaxExpr(NewOps);
        if (isa<SCEVUMaxExpr>(Comm))
          return SE.getUMaxExpr(NewOps);
        assert(0 && "Unknown commutative SCEV type!");
      }
    }
    // If we got here, all operands are loop invariant.
    return Comm;
  }

  if (SCEVUDivExpr *Div = dyn_cast<SCEVUDivExpr>(V)) {
    SCEVHandle LHS = getSCEVAtScope(Div->getLHS(), L);
    if (LHS == UnknownValue) return LHS;
    SCEVHandle RHS = getSCEVAtScope(Div->getRHS(), L);
    if (RHS == UnknownValue) return RHS;
    if (LHS == Div->getLHS() && RHS == Div->getRHS())
      return Div;   // must be loop invariant
    return SE.getUDivExpr(LHS, RHS);
  }

  // If this is a loop recurrence for a loop that does not contain L, then we
  // are dealing with the final value computed by the loop.
  if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V)) {
    if (!L || !AddRec->getLoop()->contains(L->getHeader())) {
      // To evaluate this recurrence, we need to know how many times the AddRec
      // loop iterates.  Compute this now.
      SCEVHandle IterationCount = getIterationCount(AddRec->getLoop());
      if (IterationCount == UnknownValue) return UnknownValue;

      // Then, evaluate the AddRec.
      return AddRec->evaluateAtIteration(IterationCount, SE);
    }
    return UnknownValue;
  }

  //assert(0 && "Unknown SCEV type!");
  return UnknownValue;
}

/// SolveLinEquationWithOverflow - Finds the minimum unsigned root of the
/// following equation:
///
///     A * X = B (mod N)
///
/// where N = 2^BW and BW is the common bit width of A and B. The signedness of
/// A and B isn't important.
///
/// If the equation does not have a solution, SCEVCouldNotCompute is returned.
static SCEVHandle SolveLinEquationWithOverflow(const APInt &A, const APInt &B,
                                               ScalarEvolution &SE) {
  uint32_t BW = A.getBitWidth();
  assert(BW == B.getBitWidth() && "Bit widths must be the same.");
  assert(A != 0 && "A must be non-zero.");

  // 1. D = gcd(A, N)
  //
  // The gcd of A and N may have only one prime factor: 2. The number of
  // trailing zeros in A is its multiplicity
  uint32_t Mult2 = A.countTrailingZeros();
  // D = 2^Mult2

  // 2. Check if B is divisible by D.
  //
  // B is divisible by D if and only if the multiplicity of prime factor 2 for B
  // is not less than multiplicity of this prime factor for D.
  if (B.countTrailingZeros() < Mult2)
    return new SCEVCouldNotCompute();

  // 3. Compute I: the multiplicative inverse of (A / D) in arithmetic
  // modulo (N / D).
  //
  // (N / D) may need BW+1 bits in its representation.  Hence, we'll use this
  // bit width during computations.
  APInt AD = A.lshr(Mult2).zext(BW + 1);  // AD = A / D
  APInt Mod(BW + 1, 0);
  Mod.set(BW - Mult2);  // Mod = N / D
  APInt I = AD.multiplicativeInverse(Mod);

  // 4. Compute the minimum unsigned root of the equation:
  // I * (B / D) mod (N / D)
  APInt Result = (I * B.lshr(Mult2).zext(BW + 1)).urem(Mod);

  // The result is guaranteed to be less than 2^BW so we may truncate it to BW
  // bits.
  return SE.getConstant(Result.trunc(BW));
}

/// SolveQuadraticEquation - Find the roots of the quadratic equation for the
/// given quadratic chrec {L,+,M,+,N}.  This returns either the two roots (which
/// might be the same) or two SCEVCouldNotCompute objects.
///
static std::pair<SCEVHandle,SCEVHandle>
SolveQuadraticEquation(const SCEVAddRecExpr *AddRec, ScalarEvolution &SE) {
  assert(AddRec->getNumOperands() == 3 && "This is not a quadratic chrec!");
  SCEVConstant *LC = dyn_cast<SCEVConstant>(AddRec->getOperand(0));
  SCEVConstant *MC = dyn_cast<SCEVConstant>(AddRec->getOperand(1));
  SCEVConstant *NC = dyn_cast<SCEVConstant>(AddRec->getOperand(2));

  // We currently can only solve this if the coefficients are constants.
  if (!LC || !MC || !NC) {
    SCEV *CNC = new SCEVCouldNotCompute();
    return std::make_pair(CNC, CNC);
  }

  uint32_t BitWidth = LC->getValue()->getValue().getBitWidth();
  const APInt &L = LC->getValue()->getValue();
  const APInt &M = MC->getValue()->getValue();
  const APInt &N = NC->getValue()->getValue();
  APInt Two(BitWidth, 2);
  APInt Four(BitWidth, 4);

  { 
    using namespace APIntOps;
    const APInt& C = L;
    // Convert from chrec coefficients to polynomial coefficients AX^2+BX+C
    // The B coefficient is M-N/2
    APInt B(M);
    B -= sdiv(N,Two);

    // The A coefficient is N/2
    APInt A(N.sdiv(Two));

    // Compute the B^2-4ac term.
    APInt SqrtTerm(B);
    SqrtTerm *= B;
    SqrtTerm -= Four * (A * C);

    // Compute sqrt(B^2-4ac). This is guaranteed to be the nearest
    // integer value or else APInt::sqrt() will assert.
    APInt SqrtVal(SqrtTerm.sqrt());

    // Compute the two solutions for the quadratic formula. 
    // The divisions must be performed as signed divisions.
    APInt NegB(-B);
    APInt TwoA( A << 1 );
    ConstantInt *Solution1 = ConstantInt::get((NegB + SqrtVal).sdiv(TwoA));
    ConstantInt *Solution2 = ConstantInt::get((NegB - SqrtVal).sdiv(TwoA));

    return std::make_pair(SE.getConstant(Solution1), 
                          SE.getConstant(Solution2));
    } // end APIntOps namespace
}

/// HowFarToZero - Return the number of times a backedge comparing the specified
/// value to zero will execute.  If not computable, return UnknownValue
SCEVHandle ScalarEvolutionsImpl::HowFarToZero(SCEV *V, const Loop *L) {
  // If the value is a constant
  if (SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    // If the value is already zero, the branch will execute zero times.
    if (C->getValue()->isZero()) return C;
    return UnknownValue;  // Otherwise it will loop infinitely.
  }

  SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V);
  if (!AddRec || AddRec->getLoop() != L)
    return UnknownValue;

  if (AddRec->isAffine()) {
    // If this is an affine expression, the execution count of this branch is
    // the minimum unsigned root of the following equation:
    //
    //     Start + Step*N = 0 (mod 2^BW)
    //
    // equivalent to:
    //
    //             Step*N = -Start (mod 2^BW)
    //
    // where BW is the common bit width of Start and Step.

    // Get the initial value for the loop.
    SCEVHandle Start = getSCEVAtScope(AddRec->getStart(), L->getParentLoop());
    if (isa<SCEVCouldNotCompute>(Start)) return UnknownValue;

    SCEVHandle Step = getSCEVAtScope(AddRec->getOperand(1), L->getParentLoop());

    if (SCEVConstant *StepC = dyn_cast<SCEVConstant>(Step)) {
      // For now we handle only constant steps.

      // First, handle unitary steps.
      if (StepC->getValue()->equalsInt(1))      // 1*N = -Start (mod 2^BW), so:
        return SE.getNegativeSCEV(Start);       //   N = -Start (as unsigned)
      if (StepC->getValue()->isAllOnesValue())  // -1*N = -Start (mod 2^BW), so:
        return Start;                           //    N = Start (as unsigned)

      // Then, try to solve the above equation provided that Start is constant.
      if (SCEVConstant *StartC = dyn_cast<SCEVConstant>(Start))
        return SolveLinEquationWithOverflow(StepC->getValue()->getValue(),
                                            -StartC->getValue()->getValue(),SE);
    }
  } else if (AddRec->isQuadratic() && AddRec->getType()->isInteger()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of
    // the quadratic equation to solve it.
    std::pair<SCEVHandle,SCEVHandle> Roots = SolveQuadraticEquation(AddRec, SE);
    SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
#if 0
      cerr << "HFTZ: " << *V << " - sol#1: " << *R1
           << "  sol#2: " << *R2 << "\n";
#endif
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(ICmpInst::ICMP_ULT, 
                                   R1->getValue(), R2->getValue()))) {
        if (CB->getZExtValue() == false)
          std::swap(R1, R2);   // R1 is the minimum root now.

        // We can only use this value if the chrec ends up with an exact zero
        // value at this index.  When solving for "X*X != 5", for example, we
        // should not accept a root of 2.
        SCEVHandle Val = AddRec->evaluateAtIteration(R1, SE);
        if (Val->isZero())
          return R1;  // We found a quadratic root!
      }
    }
  }

  return UnknownValue;
}

/// HowFarToNonZero - Return the number of times a backedge checking the
/// specified value for nonzero will execute.  If not computable, return
/// UnknownValue
SCEVHandle ScalarEvolutionsImpl::HowFarToNonZero(SCEV *V, const Loop *L) {
  // Loops that look like: while (X == 0) are very strange indeed.  We don't
  // handle them yet except for the trivial case.  This could be expanded in the
  // future as needed.

  // If the value is a constant, check to see if it is known to be non-zero
  // already.  If so, the backedge will execute zero times.
  if (SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    if (!C->getValue()->isNullValue())
      return SE.getIntegerSCEV(0, C->getType());
    return UnknownValue;  // Otherwise it will loop infinitely.
  }

  // We could implement others, but I really doubt anyone writes loops like
  // this, and if they did, they would already be constant folded.
  return UnknownValue;
}

/// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
/// (which may not be an immediate predecessor) which has exactly one
/// successor from which BB is reachable, or null if no such block is
/// found.
///
BasicBlock *
ScalarEvolutionsImpl::getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB) {
  // If the block has a unique predecessor, the predecessor must have
  // no other successors from which BB is reachable.
  if (BasicBlock *Pred = BB->getSinglePredecessor())
    return Pred;

  // A loop's header is defined to be a block that dominates the loop.
  // If the loop has a preheader, it must be a block that has exactly
  // one successor that can reach BB. This is slightly more strict
  // than necessary, but works if critical edges are split.
  if (Loop *L = LI.getLoopFor(BB))
    return L->getLoopPreheader();

  return 0;
}

/// executesAtLeastOnce - Test whether entry to the loop is protected by
/// a conditional between LHS and RHS.
bool ScalarEvolutionsImpl::executesAtLeastOnce(const Loop *L, bool isSigned,
                                               SCEV *LHS, SCEV *RHS) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *PreheaderDest = L->getHeader();

  // Starting at the preheader, climb up the predecessor chain, as long as
  // there are predecessors that can be found that have unique successors
  // leading to the original header.
  for (; Preheader;
       PreheaderDest = Preheader,
       Preheader = getPredecessorWithUniqueSuccessorForBB(Preheader)) {

    BranchInst *LoopEntryPredicate =
      dyn_cast<BranchInst>(Preheader->getTerminator());
    if (!LoopEntryPredicate ||
        LoopEntryPredicate->isUnconditional())
      continue;

    ICmpInst *ICI = dyn_cast<ICmpInst>(LoopEntryPredicate->getCondition());
    if (!ICI) continue;

    // Now that we found a conditional branch that dominates the loop, check to
    // see if it is the comparison we are looking for.
    Value *PreCondLHS = ICI->getOperand(0);
    Value *PreCondRHS = ICI->getOperand(1);
    ICmpInst::Predicate Cond;
    if (LoopEntryPredicate->getSuccessor(0) == PreheaderDest)
      Cond = ICI->getPredicate();
    else
      Cond = ICI->getInversePredicate();

    switch (Cond) {
    case ICmpInst::ICMP_UGT:
      if (isSigned) continue;
      std::swap(PreCondLHS, PreCondRHS);
      Cond = ICmpInst::ICMP_ULT;
      break;
    case ICmpInst::ICMP_SGT:
      if (!isSigned) continue;
      std::swap(PreCondLHS, PreCondRHS);
      Cond = ICmpInst::ICMP_SLT;
      break;
    case ICmpInst::ICMP_ULT:
      if (isSigned) continue;
      break;
    case ICmpInst::ICMP_SLT:
      if (!isSigned) continue;
      break;
    default:
      continue;
    }

    if (!PreCondLHS->getType()->isInteger()) continue;

    SCEVHandle PreCondLHSSCEV = getSCEV(PreCondLHS);
    SCEVHandle PreCondRHSSCEV = getSCEV(PreCondRHS);
    if ((LHS == PreCondLHSSCEV && RHS == PreCondRHSSCEV) ||
        (LHS == SE.getNotSCEV(PreCondRHSSCEV) &&
         RHS == SE.getNotSCEV(PreCondLHSSCEV)))
      return true;
  }

  return false;
}

/// HowManyLessThans - Return the number of times a backedge containing the
/// specified less-than comparison will execute.  If not computable, return
/// UnknownValue.
SCEVHandle ScalarEvolutionsImpl::
HowManyLessThans(SCEV *LHS, SCEV *RHS, const Loop *L, bool isSigned) {
  // Only handle:  "ADDREC < LoopInvariant".
  if (!RHS->isLoopInvariant(L)) return UnknownValue;

  SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS);
  if (!AddRec || AddRec->getLoop() != L)
    return UnknownValue;

  if (AddRec->isAffine()) {
    // FORNOW: We only support unit strides.
    SCEVHandle One = SE.getIntegerSCEV(1, RHS->getType());
    if (AddRec->getOperand(1) != One)
      return UnknownValue;

    // We know the LHS is of the form {n,+,1} and the RHS is some loop-invariant
    // m.  So, we count the number of iterations in which {n,+,1} < m is true.
    // Note that we cannot simply return max(m-n,0) because it's not safe to
    // treat m-n as signed nor unsigned due to overflow possibility.

    // First, we get the value of the LHS in the first iteration: n
    SCEVHandle Start = AddRec->getOperand(0);

    if (executesAtLeastOnce(L, isSigned,
                            SE.getMinusSCEV(AddRec->getOperand(0), One), RHS)) {
      // Since we know that the condition is true in order to enter the loop,
      // we know that it will run exactly m-n times.
      return SE.getMinusSCEV(RHS, Start);
    } else {
      // Then, we get the value of the LHS in the first iteration in which the
      // above condition doesn't hold.  This equals to max(m,n).
      SCEVHandle End = isSigned ? SE.getSMaxExpr(RHS, Start)
                                : SE.getUMaxExpr(RHS, Start);

      // Finally, we subtract these two values to get the number of times the
      // backedge is executed: max(m,n)-n.
      return SE.getMinusSCEV(End, Start);
    }
  }

  return UnknownValue;
}

/// getNumIterationsInRange - Return the number of iterations of this loop that
/// produce values in the specified constant range.  Another way of looking at
/// this is that it returns the first iteration number where the value is not in
/// the condition, thus computing the exit count. If the iteration count can't
/// be computed, an instance of SCEVCouldNotCompute is returned.
SCEVHandle SCEVAddRecExpr::getNumIterationsInRange(ConstantRange Range,
                                                   ScalarEvolution &SE) const {
  if (Range.isFullSet())  // Infinite loop.
    return new SCEVCouldNotCompute();

  // If the start is a non-zero constant, shift the range to simplify things.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(getStart()))
    if (!SC->getValue()->isZero()) {
      std::vector<SCEVHandle> Operands(op_begin(), op_end());
      Operands[0] = SE.getIntegerSCEV(0, SC->getType());
      SCEVHandle Shifted = SE.getAddRecExpr(Operands, getLoop());
      if (SCEVAddRecExpr *ShiftedAddRec = dyn_cast<SCEVAddRecExpr>(Shifted))
        return ShiftedAddRec->getNumIterationsInRange(
                           Range.subtract(SC->getValue()->getValue()), SE);
      // This is strange and shouldn't happen.
      return new SCEVCouldNotCompute();
    }

  // The only time we can solve this is when we have all constant indices.
  // Otherwise, we cannot determine the overflow conditions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<SCEVConstant>(getOperand(i)))
      return new SCEVCouldNotCompute();


  // Okay at this point we know that all elements of the chrec are constants and
  // that the start element is zero.

  // First check to see if the range contains zero.  If not, the first
  // iteration exits.
  if (!Range.contains(APInt(getBitWidth(),0))) 
    return SE.getConstant(ConstantInt::get(getType(),0));

  if (isAffine()) {
    // If this is an affine expression then we have this situation:
    //   Solve {0,+,A} in Range  ===  Ax in Range

    // We know that zero is in the range.  If A is positive then we know that
    // the upper value of the range must be the first possible exit value.
    // If A is negative then the lower of the range is the last possible loop
    // value.  Also note that we already checked for a full range.
    APInt One(getBitWidth(),1);
    APInt A     = cast<SCEVConstant>(getOperand(1))->getValue()->getValue();
    APInt End = A.sge(One) ? (Range.getUpper() - One) : Range.getLower();

    // The exit value should be (End+A)/A.
    APInt ExitVal = (End + A).udiv(A);
    ConstantInt *ExitValue = ConstantInt::get(ExitVal);

    // Evaluate at the exit value.  If we really did fall out of the valid
    // range, then we computed our trip count, otherwise wrap around or other
    // things must have happened.
    ConstantInt *Val = EvaluateConstantChrecAtConstant(this, ExitValue, SE);
    if (Range.contains(Val->getValue()))
      return new SCEVCouldNotCompute();  // Something strange happened

    // Ensure that the previous value is in the range.  This is a sanity check.
    assert(Range.contains(
           EvaluateConstantChrecAtConstant(this, 
           ConstantInt::get(ExitVal - One), SE)->getValue()) &&
           "Linear scev computation is off in a bad way!");
    return SE.getConstant(ExitValue);
  } else if (isQuadratic()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of the
    // quadratic equation to solve it.  To do this, we must frame our problem in
    // terms of figuring out when zero is crossed, instead of when
    // Range.getUpper() is crossed.
    std::vector<SCEVHandle> NewOps(op_begin(), op_end());
    NewOps[0] = SE.getNegativeSCEV(SE.getConstant(Range.getUpper()));
    SCEVHandle NewAddRec = SE.getAddRecExpr(NewOps, getLoop());

    // Next, solve the constructed addrec
    std::pair<SCEVHandle,SCEVHandle> Roots =
      SolveQuadraticEquation(cast<SCEVAddRecExpr>(NewAddRec), SE);
    SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(ICmpInst::ICMP_ULT, 
                                   R1->getValue(), R2->getValue()))) {
        if (CB->getZExtValue() == false)
          std::swap(R1, R2);   // R1 is the minimum root now.

        // Make sure the root is not off by one.  The returned iteration should
        // not be in the range, but the previous one should be.  When solving
        // for "X*X < 5", for example, we should not return a root of 2.
        ConstantInt *R1Val = EvaluateConstantChrecAtConstant(this,
                                                             R1->getValue(),
                                                             SE);
        if (Range.contains(R1Val->getValue())) {
          // The next iteration must be out of the range...
          ConstantInt *NextVal = ConstantInt::get(R1->getValue()->getValue()+1);

          R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
          if (!Range.contains(R1Val->getValue()))
            return SE.getConstant(NextVal);
          return new SCEVCouldNotCompute();  // Something strange happened
        }

        // If R1 was not in the range, then it is a good return value.  Make
        // sure that R1-1 WAS in the range though, just in case.
        ConstantInt *NextVal = ConstantInt::get(R1->getValue()->getValue()-1);
        R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
        if (Range.contains(R1Val->getValue()))
          return R1;
        return new SCEVCouldNotCompute();  // Something strange happened
      }
    }
  }

  // Fallback, if this is a general polynomial, figure out the progression
  // through brute force: evaluate until we find an iteration that fails the
  // test.  This is likely to be slow, but getting an accurate trip count is
  // incredibly important, we will be able to simplify the exit test a lot, and
  // we are almost guaranteed to get a trip count in this case.
  ConstantInt *TestVal = ConstantInt::get(getType(), 0);
  ConstantInt *EndVal  = TestVal;  // Stop when we wrap around.
  do {
    ++NumBruteForceEvaluations;
    SCEVHandle Val = evaluateAtIteration(SE.getConstant(TestVal), SE);
    if (!isa<SCEVConstant>(Val))  // This shouldn't happen.
      return new SCEVCouldNotCompute();

    // Check to see if we found the value!
    if (!Range.contains(cast<SCEVConstant>(Val)->getValue()->getValue()))
      return SE.getConstant(TestVal);

    // Increment to test the next index.
    TestVal = ConstantInt::get(TestVal->getValue()+1);
  } while (TestVal != EndVal);

  return new SCEVCouldNotCompute();
}



//===----------------------------------------------------------------------===//
//                   ScalarEvolution Class Implementation
//===----------------------------------------------------------------------===//

bool ScalarEvolution::runOnFunction(Function &F) {
  Impl = new ScalarEvolutionsImpl(*this, F, getAnalysis<LoopInfo>());
  return false;
}

void ScalarEvolution::releaseMemory() {
  delete (ScalarEvolutionsImpl*)Impl;
  Impl = 0;
}

void ScalarEvolution::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<LoopInfo>();
}

SCEVHandle ScalarEvolution::getSCEV(Value *V) const {
  return ((ScalarEvolutionsImpl*)Impl)->getSCEV(V);
}

/// hasSCEV - Return true if the SCEV for this value has already been
/// computed.
bool ScalarEvolution::hasSCEV(Value *V) const {
  return ((ScalarEvolutionsImpl*)Impl)->hasSCEV(V);
}


/// setSCEV - Insert the specified SCEV into the map of current SCEVs for
/// the specified value.
void ScalarEvolution::setSCEV(Value *V, const SCEVHandle &H) {
  ((ScalarEvolutionsImpl*)Impl)->setSCEV(V, H);
}


SCEVHandle ScalarEvolution::getIterationCount(const Loop *L) const {
  return ((ScalarEvolutionsImpl*)Impl)->getIterationCount(L);
}

bool ScalarEvolution::hasLoopInvariantIterationCount(const Loop *L) const {
  return !isa<SCEVCouldNotCompute>(getIterationCount(L));
}

SCEVHandle ScalarEvolution::getSCEVAtScope(Value *V, const Loop *L) const {
  return ((ScalarEvolutionsImpl*)Impl)->getSCEVAtScope(getSCEV(V), L);
}

void ScalarEvolution::deleteValueFromRecords(Value *V) const {
  return ((ScalarEvolutionsImpl*)Impl)->deleteValueFromRecords(V);
}

static void PrintLoopInfo(std::ostream &OS, const ScalarEvolution *SE,
                          const Loop *L) {
  // Print all inner loops first
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    PrintLoopInfo(OS, SE, *I);

  OS << "Loop " << L->getHeader()->getName() << ": ";

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1)
    OS << "<multiple exits> ";

  if (SE->hasLoopInvariantIterationCount(L)) {
    OS << *SE->getIterationCount(L) << " iterations! ";
  } else {
    OS << "Unpredictable iteration count. ";
  }

  OS << "\n";
}

void ScalarEvolution::print(std::ostream &OS, const Module* ) const {
  Function &F = ((ScalarEvolutionsImpl*)Impl)->F;
  LoopInfo &LI = ((ScalarEvolutionsImpl*)Impl)->LI;

  OS << "Classifying expressions for: " << F.getName() << "\n";
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (I->getType()->isInteger()) {
      OS << *I;
      OS << "  -->  ";
      SCEVHandle SV = getSCEV(&*I);
      SV->print(OS);
      OS << "\t\t";

      if (const Loop *L = LI.getLoopFor((*I).getParent())) {
        OS << "Exits: ";
        SCEVHandle ExitValue = getSCEVAtScope(&*I, L->getParentLoop());
        if (isa<SCEVCouldNotCompute>(ExitValue)) {
          OS << "<<Unknown>>";
        } else {
          OS << *ExitValue;
        }
      }


      OS << "\n";
    }

  OS << "Determining loop execution counts for: " << F.getName() << "\n";
  for (LoopInfo::iterator I = LI.begin(), E = LI.end(); I != E; ++I)
    PrintLoopInfo(OS, this, *I);
}
