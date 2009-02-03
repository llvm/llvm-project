//===- LoopVR.cpp - Value Range analysis driven by loop information -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: What does this do?
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loopvr"
#include "llvm/Analysis/LoopVR.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

char LoopVR::ID = 0;
static RegisterPass<LoopVR> X("loopvr", "Loop Value Ranges", true, true);

/// getRange - determine the range for a particular SCEV within a given Loop
ConstantRange LoopVR::getRange(SCEVHandle S, Loop *L, ScalarEvolution &SE) {
  SCEVHandle T = SE.getIterationCount(L);
  if (isa<SCEVCouldNotCompute>(T))
    return ConstantRange(cast<IntegerType>(S->getType())->getBitWidth(), true);

  T = SE.getTruncateOrZeroExtend(T, S->getType());
  return getRange(S, T, SE);
}

/// getRange - determine the range for a particular SCEV with a given trip count
ConstantRange LoopVR::getRange(SCEVHandle S, SCEVHandle T, ScalarEvolution &SE){

  if (SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return ConstantRange(C->getValue()->getValue());

  ConstantRange FullSet(cast<IntegerType>(S->getType())->getBitWidth(), true);

  // {x,+,y,+,...z}. We detect overflow by checking the size of the set after
  // summing the upper and lower.
  if (SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    ConstantRange X = getRange(Add->getOperand(0), T, SE);
    if (X.isFullSet()) return FullSet;
    for (unsigned i = 1, e = Add->getNumOperands(); i != e; ++i) {
      ConstantRange Y = getRange(Add->getOperand(i), T, SE);
      if (Y.isFullSet()) return FullSet;

      APInt Spread_X = X.getSetSize(), Spread_Y = Y.getSetSize();
      APInt NewLower = X.getLower() + Y.getLower();
      APInt NewUpper = X.getUpper() + Y.getUpper() - 1;
      if (NewLower == NewUpper)
        return FullSet;

      X = ConstantRange(NewLower, NewUpper);
      if (X.getSetSize().ult(Spread_X) || X.getSetSize().ult(Spread_Y))
        return FullSet; // we've wrapped, therefore, full set.
    }
    return X;
  }

  // {x,*,y,*,...,z}. In order to detect overflow, we use k*bitwidth where
  // k is the number of terms being multiplied.
  if (SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    ConstantRange X = getRange(Mul->getOperand(0), T, SE);
    if (X.isFullSet()) return FullSet;

    const IntegerType *Ty = IntegerType::get(X.getBitWidth());
    const IntegerType *ExTy = IntegerType::get(X.getBitWidth() *
                                               Mul->getNumOperands());
    ConstantRange XExt = X.zeroExtend(ExTy->getBitWidth());

    for (unsigned i = 1, e = Mul->getNumOperands(); i != e; ++i) {
      ConstantRange Y = getRange(Mul->getOperand(i), T, SE);
      if (Y.isFullSet()) return FullSet;

      ConstantRange YExt = Y.zeroExtend(ExTy->getBitWidth());
      XExt = ConstantRange(XExt.getLower() * YExt.getLower(),
                           ((XExt.getUpper()-1) * (YExt.getUpper()-1)) + 1);
    }
    return XExt.truncate(Ty->getBitWidth());
  }

  // X smax Y smax ... Z is: range(smax(X_smin, Y_smin, ..., Z_smin),
  //                               smax(X_smax, Y_smax, ..., Z_smax))
  // It doesn't matter if one of the SCEVs has FullSet because we're taking
  // a maximum of the minimums across all of them.
  if (SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(S)) {
    ConstantRange X = getRange(SMax->getOperand(0), T, SE);
    if (X.isFullSet()) return FullSet;

    APInt smin = X.getSignedMin(), smax = X.getSignedMax();
    for (unsigned i = 1, e = SMax->getNumOperands(); i != e; ++i) {
      ConstantRange Y = getRange(SMax->getOperand(i), T, SE);
      smin = APIntOps::smax(smin, Y.getSignedMin());
      smax = APIntOps::smax(smax, Y.getSignedMax());
    }
    if (smax + 1 == smin) return FullSet;
    return ConstantRange(smin, smax + 1);
  }

  // X umax Y umax ... Z is: range(umax(X_umin, Y_umin, ..., Z_umin),
  //                               umax(X_umax, Y_umax, ..., Z_umax))
  // It doesn't matter if one of the SCEVs has FullSet because we're taking
  // a maximum of the minimums across all of them.
  if (SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(S)) {
    ConstantRange X = getRange(UMax->getOperand(0), T, SE);
    if (X.isFullSet()) return FullSet;

    APInt umin = X.getUnsignedMin(), umax = X.getUnsignedMax();
    for (unsigned i = 1, e = UMax->getNumOperands(); i != e; ++i) {
      ConstantRange Y = getRange(UMax->getOperand(i), T, SE);
      umin = APIntOps::umax(umin, Y.getUnsignedMin());
      umax = APIntOps::umax(umax, Y.getUnsignedMax());
    }
    if (umax + 1 == umin) return FullSet;
    return ConstantRange(umin, umax + 1);
  }

  // L udiv R. Luckily, there's only ever 2 sides to a udiv.
  if (SCEVUDivExpr *UDiv = dyn_cast<SCEVUDivExpr>(S)) {
    ConstantRange L = getRange(UDiv->getLHS(), T, SE);
    ConstantRange R = getRange(UDiv->getRHS(), T, SE);
    if (L.isFullSet() && R.isFullSet()) return FullSet;

    if (R.getUnsignedMax() == 0) {
      // RHS must be single-element zero. Return an empty set.
      return ConstantRange(R.getBitWidth(), false);
    }

    APInt Lower = L.getUnsignedMin().udiv(R.getUnsignedMax());

    APInt Upper;

    if (R.getUnsignedMin() == 0) {
      // Just because it contains zero, doesn't mean it will also contain one.
      // Use maximalIntersectWith to get the right behaviour.
      ConstantRange NotZero(APInt(L.getBitWidth(), 1),
                            APInt::getNullValue(L.getBitWidth()));
      R = R.maximalIntersectWith(NotZero);
    }
 
    // But, the maximal intersection might still include zero. If it does, then
    // we know it also included one.
    if (R.contains(APInt::getNullValue(L.getBitWidth())))
      Upper = L.getUnsignedMax();
    else
      Upper = L.getUnsignedMax().udiv(R.getUnsignedMin());

    return ConstantRange(Lower, Upper);
  }

  // ConstantRange already implements the cast operators.

  if (SCEVZeroExtendExpr *ZExt = dyn_cast<SCEVZeroExtendExpr>(S)) {
    T = SE.getTruncateOrZeroExtend(T, ZExt->getOperand()->getType());
    ConstantRange X = getRange(ZExt->getOperand(), T, SE);
    return X.zeroExtend(cast<IntegerType>(ZExt->getType())->getBitWidth());
  }

  if (SCEVSignExtendExpr *SExt = dyn_cast<SCEVSignExtendExpr>(S)) {
    T = SE.getTruncateOrZeroExtend(T, SExt->getOperand()->getType());
    ConstantRange X = getRange(SExt->getOperand(), T, SE);
    return X.signExtend(cast<IntegerType>(SExt->getType())->getBitWidth());
  }

  if (SCEVTruncateExpr *Trunc = dyn_cast<SCEVTruncateExpr>(S)) {
    T = SE.getTruncateOrZeroExtend(T, Trunc->getOperand()->getType());
    ConstantRange X = getRange(Trunc->getOperand(), T, SE);
    if (X.isFullSet()) return FullSet;
    return X.truncate(cast<IntegerType>(Trunc->getType())->getBitWidth());
  }

  if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    SCEVConstant *Trip = dyn_cast<SCEVConstant>(T);
    if (!Trip) return FullSet;

    if (AddRec->isAffine()) {
      SCEVHandle StartHandle = AddRec->getStart();
      SCEVHandle StepHandle = AddRec->getOperand(1);

      SCEVConstant *Step = dyn_cast<SCEVConstant>(StepHandle);
      if (!Step) return FullSet;

      uint32_t ExWidth = 2 * Trip->getValue()->getBitWidth();
      APInt TripExt = Trip->getValue()->getValue(); TripExt.zext(ExWidth);
      APInt StepExt = Step->getValue()->getValue(); StepExt.zext(ExWidth);
      if ((TripExt * StepExt).ugt(APInt::getLowBitsSet(ExWidth, ExWidth >> 1)))
        return FullSet;

      SCEVHandle EndHandle = SE.getAddExpr(StartHandle,
                                           SE.getMulExpr(T, StepHandle));
      SCEVConstant *Start = dyn_cast<SCEVConstant>(StartHandle);
      SCEVConstant *End = dyn_cast<SCEVConstant>(EndHandle);
      if (!Start || !End) return FullSet;

      const APInt &StartInt = Start->getValue()->getValue();
      const APInt &EndInt = End->getValue()->getValue();
      const APInt &StepInt = Step->getValue()->getValue();

      if (StepInt.isNegative()) {
        if (EndInt == StartInt + 1) return FullSet;
        return ConstantRange(EndInt, StartInt + 1);
      } else {
        if (StartInt == EndInt + 1) return FullSet;
        return ConstantRange(StartInt, EndInt + 1);
      }
    }
  }

  // TODO: non-affine addrec, udiv, SCEVUnknown (narrowed from elsewhere)?

  return FullSet;
}

bool LoopVR::runOnFunction(Function &F) { Map.clear(); return false; }

void LoopVR::print(std::ostream &os, const Module *) const {
  raw_os_ostream OS(os);
  for (std::map<Value *, ConstantRange *>::const_iterator I = Map.begin(),
       E = Map.end(); I != E; ++I) {
    OS << *I->first << ": " << *I->second << '\n';
  }
}

void LoopVR::releaseMemory() {
  for (std::map<Value *, ConstantRange *>::iterator I = Map.begin(),
       E = Map.end(); I != E; ++I) {
    delete I->second;
  }

  Map.clear();  
}

ConstantRange LoopVR::compute(Value *V) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return ConstantRange(CI->getValue());

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return ConstantRange(cast<IntegerType>(V->getType())->getBitWidth(), false);

  LoopInfo &LI = getAnalysis<LoopInfo>();

  Loop *L = LI.getLoopFor(I->getParent());
  if (!L || L->isLoopInvariant(I))
    return ConstantRange(cast<IntegerType>(V->getType())->getBitWidth(), false);

  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();

  SCEVHandle S = SE.getSCEV(I);
  if (isa<SCEVUnknown>(S) || isa<SCEVCouldNotCompute>(S))
    return ConstantRange(cast<IntegerType>(V->getType())->getBitWidth(), false);

  return ConstantRange(getRange(S, L, SE));
}

ConstantRange LoopVR::get(Value *V) {
  std::map<Value *, ConstantRange *>::iterator I = Map.find(V);
  if (I == Map.end()) {
    ConstantRange *CR = new ConstantRange(compute(V));
    Map[V] = CR;
    return *CR;
  }

  return *I->second;
}

void LoopVR::remove(Value *V) {
  std::map<Value *, ConstantRange *>::iterator I = Map.find(V);
  if (I != Map.end()) {
    delete I->second;
    Map.erase(I);
  }
}

void LoopVR::narrow(Value *V, const ConstantRange &CR) {
  if (CR.isFullSet()) return;

  std::map<Value *, ConstantRange *>::iterator I = Map.find(V);
  if (I == Map.end())
    Map[V] = new ConstantRange(CR);
  else
    Map[V] = new ConstantRange(Map[V]->maximalIntersectWith(CR));
}
