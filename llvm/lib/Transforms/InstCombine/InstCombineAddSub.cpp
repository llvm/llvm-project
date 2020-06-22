//===- InstCombineAddSub.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for add, fadd, sub, and fsub.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/KnownBits.h"
#include <cassert>
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

namespace {

  /// Class representing coefficient of floating-point addend.
  /// This class needs to be highly efficient, which is especially true for
  /// the constructor. As of I write this comment, the cost of the default
  /// constructor is merely 4-byte-store-zero (Assuming compiler is able to
  /// perform write-merging).
  ///
  class FAddendCoef {
  public:
    // The constructor has to initialize a APFloat, which is unnecessary for
    // most addends which have coefficient either 1 or -1. So, the constructor
    // is expensive. In order to avoid the cost of the constructor, we should
    // reuse some instances whenever possible. The pre-created instances
    // FAddCombine::Add[0-5] embodies this idea.
    FAddendCoef() = default;
    ~FAddendCoef();

    // If possible, don't define operator+/operator- etc because these
    // operators inevitably call FAddendCoef's constructor which is not cheap.
    void operator=(const FAddendCoef &A);
    void operator+=(const FAddendCoef &A);
    void operator*=(const FAddendCoef &S);

    void set(short C) {
      assert(!insaneIntVal(C) && "Insane coefficient");
      IsFp = false; IntVal = C;
    }

    void set(const APFloat& C);

    void negate();

    bool isZero() const { return isInt() ? !IntVal : getFpVal().isZero(); }
    Value *getValue(Type *) const;

    bool isOne() const { return isInt() && IntVal == 1; }
    bool isTwo() const { return isInt() && IntVal == 2; }
    bool isMinusOne() const { return isInt() && IntVal == -1; }
    bool isMinusTwo() const { return isInt() && IntVal == -2; }

  private:
    bool insaneIntVal(int V) { return V > 4 || V < -4; }

    APFloat *getFpValPtr()
      { return reinterpret_cast<APFloat *>(&FpValBuf.buffer[0]); }

    const APFloat *getFpValPtr() const
      { return reinterpret_cast<const APFloat *>(&FpValBuf.buffer[0]); }

    const APFloat &getFpVal() const {
      assert(IsFp && BufHasFpVal && "Incorret state");
      return *getFpValPtr();
    }

    APFloat &getFpVal() {
      assert(IsFp && BufHasFpVal && "Incorret state");
      return *getFpValPtr();
    }

    bool isInt() const { return !IsFp; }

    // If the coefficient is represented by an integer, promote it to a
    // floating point.
    void convertToFpType(const fltSemantics &Sem);

    // Construct an APFloat from a signed integer.
    // TODO: We should get rid of this function when APFloat can be constructed
    //       from an *SIGNED* integer.
    APFloat createAPFloatFromInt(const fltSemantics &Sem, int Val);

    bool IsFp = false;

    // True iff FpValBuf contains an instance of APFloat.
    bool BufHasFpVal = false;

    // The integer coefficient of an individual addend is either 1 or -1,
    // and we try to simplify at most 4 addends from neighboring at most
    // two instructions. So the range of <IntVal> falls in [-4, 4]. APInt
    // is overkill of this end.
    short IntVal = 0;

    AlignedCharArrayUnion<APFloat> FpValBuf;
  };

  /// FAddend is used to represent floating-point addend. An addend is
  /// represented as <C, V>, where the V is a symbolic value, and C is a
  /// constant coefficient. A constant addend is represented as <C, 0>.
  class FAddend {
  public:
    FAddend() = default;

    void operator+=(const FAddend &T) {
      assert((Val == T.Val) && "Symbolic-values disagree");
      Coeff += T.Coeff;
    }

    Value *getSymVal() const { return Val; }
    const FAddendCoef &getCoef() const { return Coeff; }

    bool isConstant() const { return Val == nullptr; }
    bool isZero() const { return Coeff.isZero(); }

    void set(short Coefficient, Value *V) {
      Coeff.set(Coefficient);
      Val = V;
    }
    void set(const APFloat &Coefficient, Value *V) {
      Coeff.set(Coefficient);
      Val = V;
    }
    void set(const ConstantFP *Coefficient, Value *V) {
      Coeff.set(Coefficient->getValueAPF());
      Val = V;
    }

    void negate() { Coeff.negate(); }

    /// Drill down the U-D chain one step to find the definition of V, and
    /// try to break the definition into one or two addends.
    static unsigned drillValueDownOneStep(Value* V, FAddend &A0, FAddend &A1);

    /// Similar to FAddend::drillDownOneStep() except that the value being
    /// splitted is the addend itself.
    unsigned drillAddendDownOneStep(FAddend &Addend0, FAddend &Addend1) const;

  private:
    void Scale(const FAddendCoef& ScaleAmt) { Coeff *= ScaleAmt; }

    // This addend has the value of "Coeff * Val".
    Value *Val = nullptr;
    FAddendCoef Coeff;
  };

  /// FAddCombine is the class for optimizing an unsafe fadd/fsub along
  /// with its neighboring at most two instructions.
  ///
  class FAddCombine {
  public:
    FAddCombine(InstCombiner::BuilderTy &B) : Builder(B) {}

    Value *simplify(Instruction *FAdd);

  private:
    using AddendVect = SmallVector<const FAddend *, 4>;

    Value *simplifyFAdd(AddendVect& V, unsigned InstrQuota);

    /// Convert given addend to a Value
    Value *createAddendVal(const FAddend &A, bool& NeedNeg);

    /// Return the number of instructions needed to emit the N-ary addition.
    unsigned calcInstrNumber(const AddendVect& Vect);

    Value *createFSub(Value *Opnd0, Value *Opnd1);
    Value *createFAdd(Value *Opnd0, Value *Opnd1);
    Value *createFMul(Value *Opnd0, Value *Opnd1);
    Value *createFNeg(Value *V);
    Value *createNaryFAdd(const AddendVect& Opnds, unsigned InstrQuota);
    void createInstPostProc(Instruction *NewInst, bool NoNumber = false);

     // Debugging stuff are clustered here.
    #ifndef NDEBUG
      unsigned CreateInstrNum;
      void initCreateInstNum() { CreateInstrNum = 0; }
      void incCreateInstNum() { CreateInstrNum++; }
    #else
      void initCreateInstNum() {}
      void incCreateInstNum() {}
    #endif

    InstCombiner::BuilderTy &Builder;
    Instruction *Instr = nullptr;
  };

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//
// Implementation of
//    {FAddendCoef, FAddend, FAddition, FAddCombine}.
//
//===----------------------------------------------------------------------===//
FAddendCoef::~FAddendCoef() {
  if (BufHasFpVal)
    getFpValPtr()->~APFloat();
}

void FAddendCoef::set(const APFloat& C) {
  APFloat *P = getFpValPtr();

  if (isInt()) {
    // As the buffer is meanless byte stream, we cannot call
    // APFloat::operator=().
    new(P) APFloat(C);
  } else
    *P = C;

  IsFp = BufHasFpVal = true;
}

void FAddendCoef::convertToFpType(const fltSemantics &Sem) {
  if (!isInt())
    return;

  APFloat *P = getFpValPtr();
  if (IntVal > 0)
    new(P) APFloat(Sem, IntVal);
  else {
    new(P) APFloat(Sem, 0 - IntVal);
    P->changeSign();
  }
  IsFp = BufHasFpVal = true;
}

APFloat FAddendCoef::createAPFloatFromInt(const fltSemantics &Sem, int Val) {
  if (Val >= 0)
    return APFloat(Sem, Val);

  APFloat T(Sem, 0 - Val);
  T.changeSign();

  return T;
}

void FAddendCoef::operator=(const FAddendCoef &That) {
  if (That.isInt())
    set(That.IntVal);
  else
    set(That.getFpVal());
}

void FAddendCoef::operator+=(const FAddendCoef &That) {
  RoundingMode RndMode = RoundingMode::NearestTiesToEven;
  if (isInt() == That.isInt()) {
    if (isInt())
      IntVal += That.IntVal;
    else
      getFpVal().add(That.getFpVal(), RndMode);
    return;
  }

  if (isInt()) {
    const APFloat &T = That.getFpVal();
    convertToFpType(T.getSemantics());
    getFpVal().add(T, RndMode);
    return;
  }

  APFloat &T = getFpVal();
  T.add(createAPFloatFromInt(T.getSemantics(), That.IntVal), RndMode);
}

void FAddendCoef::operator*=(const FAddendCoef &That) {
  if (That.isOne())
    return;

  if (That.isMinusOne()) {
    negate();
    return;
  }

  if (isInt() && That.isInt()) {
    int Res = IntVal * (int)That.IntVal;
    assert(!insaneIntVal(Res) && "Insane int value");
    IntVal = Res;
    return;
  }

  const fltSemantics &Semantic =
    isInt() ? That.getFpVal().getSemantics() : getFpVal().getSemantics();

  if (isInt())
    convertToFpType(Semantic);
  APFloat &F0 = getFpVal();

  if (That.isInt())
    F0.multiply(createAPFloatFromInt(Semantic, That.IntVal),
                APFloat::rmNearestTiesToEven);
  else
    F0.multiply(That.getFpVal(), APFloat::rmNearestTiesToEven);
}

void FAddendCoef::negate() {
  if (isInt())
    IntVal = 0 - IntVal;
  else
    getFpVal().changeSign();
}

Value *FAddendCoef::getValue(Type *Ty) const {
  return isInt() ?
    ConstantFP::get(Ty, float(IntVal)) :
    ConstantFP::get(Ty->getContext(), getFpVal());
}

// The definition of <Val>     Addends
// =========================================
//  A + B                     <1, A>, <1,B>
//  A - B                     <1, A>, <1,B>
//  0 - B                     <-1, B>
//  C * A,                    <C, A>
//  A + C                     <1, A> <C, NULL>
//  0 +/- 0                   <0, NULL> (corner case)
//
// Legend: A and B are not constant, C is constant
unsigned FAddend::drillValueDownOneStep
  (Value *Val, FAddend &Addend0, FAddend &Addend1) {
  Instruction *I = nullptr;
  if (!Val || !(I = dyn_cast<Instruction>(Val)))
    return 0;

  unsigned Opcode = I->getOpcode();

  if (Opcode == Instruction::FAdd || Opcode == Instruction::FSub) {
    ConstantFP *C0, *C1;
    Value *Opnd0 = I->getOperand(0);
    Value *Opnd1 = I->getOperand(1);
    if ((C0 = dyn_cast<ConstantFP>(Opnd0)) && C0->isZero())
      Opnd0 = nullptr;

    if ((C1 = dyn_cast<ConstantFP>(Opnd1)) && C1->isZero())
      Opnd1 = nullptr;

    if (Opnd0) {
      if (!C0)
        Addend0.set(1, Opnd0);
      else
        Addend0.set(C0, nullptr);
    }

    if (Opnd1) {
      FAddend &Addend = Opnd0 ? Addend1 : Addend0;
      if (!C1)
        Addend.set(1, Opnd1);
      else
        Addend.set(C1, nullptr);
      if (Opcode == Instruction::FSub)
        Addend.negate();
    }

    if (Opnd0 || Opnd1)
      return Opnd0 && Opnd1 ? 2 : 1;

    // Both operands are zero. Weird!
    Addend0.set(APFloat(C0->getValueAPF().getSemantics()), nullptr);
    return 1;
  }

  if (I->getOpcode() == Instruction::FMul) {
    Value *V0 = I->getOperand(0);
    Value *V1 = I->getOperand(1);
    if (ConstantFP *C = dyn_cast<ConstantFP>(V0)) {
      Addend0.set(C, V1);
      return 1;
    }

    if (ConstantFP *C = dyn_cast<ConstantFP>(V1)) {
      Addend0.set(C, V0);
      return 1;
    }
  }

  return 0;
}

// Try to break *this* addend into two addends. e.g. Suppose this addend is
// <2.3, V>, and V = X + Y, by calling this function, we obtain two addends,
// i.e. <2.3, X> and <2.3, Y>.
unsigned FAddend::drillAddendDownOneStep
  (FAddend &Addend0, FAddend &Addend1) const {
  if (isConstant())
    return 0;

  unsigned BreakNum = FAddend::drillValueDownOneStep(Val, Addend0, Addend1);
  if (!BreakNum || Coeff.isOne())
    return BreakNum;

  Addend0.Scale(Coeff);

  if (BreakNum == 2)
    Addend1.Scale(Coeff);

  return BreakNum;
}

Value *FAddCombine::simplify(Instruction *I) {
  assert(I->hasAllowReassoc() && I->hasNoSignedZeros() &&
         "Expected 'reassoc'+'nsz' instruction");

  // Currently we are not able to handle vector type.
  if (I->getType()->isVectorTy())
    return nullptr;

  assert((I->getOpcode() == Instruction::FAdd ||
          I->getOpcode() == Instruction::FSub) && "Expect add/sub");

  // Save the instruction before calling other member-functions.
  Instr = I;

  FAddend Opnd0, Opnd1, Opnd0_0, Opnd0_1, Opnd1_0, Opnd1_1;

  unsigned OpndNum = FAddend::drillValueDownOneStep(I, Opnd0, Opnd1);

  // Step 1: Expand the 1st addend into Opnd0_0 and Opnd0_1.
  unsigned Opnd0_ExpNum = 0;
  unsigned Opnd1_ExpNum = 0;

  if (!Opnd0.isConstant())
    Opnd0_ExpNum = Opnd0.drillAddendDownOneStep(Opnd0_0, Opnd0_1);

  // Step 2: Expand the 2nd addend into Opnd1_0 and Opnd1_1.
  if (OpndNum == 2 && !Opnd1.isConstant())
    Opnd1_ExpNum = Opnd1.drillAddendDownOneStep(Opnd1_0, Opnd1_1);

  // Step 3: Try to optimize Opnd0_0 + Opnd0_1 + Opnd1_0 + Opnd1_1
  if (Opnd0_ExpNum && Opnd1_ExpNum) {
    AddendVect AllOpnds;
    AllOpnds.push_back(&Opnd0_0);
    AllOpnds.push_back(&Opnd1_0);
    if (Opnd0_ExpNum == 2)
      AllOpnds.push_back(&Opnd0_1);
    if (Opnd1_ExpNum == 2)
      AllOpnds.push_back(&Opnd1_1);

    // Compute instruction quota. We should save at least one instruction.
    unsigned InstQuota = 0;

    Value *V0 = I->getOperand(0);
    Value *V1 = I->getOperand(1);
    InstQuota = ((!isa<Constant>(V0) && V0->hasOneUse()) &&
                 (!isa<Constant>(V1) && V1->hasOneUse())) ? 2 : 1;

    if (Value *R = simplifyFAdd(AllOpnds, InstQuota))
      return R;
  }

  if (OpndNum != 2) {
    // The input instruction is : "I=0.0 +/- V". If the "V" were able to be
    // splitted into two addends, say "V = X - Y", the instruction would have
    // been optimized into "I = Y - X" in the previous steps.
    //
    const FAddendCoef &CE = Opnd0.getCoef();
    return CE.isOne() ? Opnd0.getSymVal() : nullptr;
  }

  // step 4: Try to optimize Opnd0 + Opnd1_0 [+ Opnd1_1]
  if (Opnd1_ExpNum) {
    AddendVect AllOpnds;
    AllOpnds.push_back(&Opnd0);
    AllOpnds.push_back(&Opnd1_0);
    if (Opnd1_ExpNum == 2)
      AllOpnds.push_back(&Opnd1_1);

    if (Value *R = simplifyFAdd(AllOpnds, 1))
      return R;
  }

  // step 5: Try to optimize Opnd1 + Opnd0_0 [+ Opnd0_1]
  if (Opnd0_ExpNum) {
    AddendVect AllOpnds;
    AllOpnds.push_back(&Opnd1);
    AllOpnds.push_back(&Opnd0_0);
    if (Opnd0_ExpNum == 2)
      AllOpnds.push_back(&Opnd0_1);

    if (Value *R = simplifyFAdd(AllOpnds, 1))
      return R;
  }

  return nullptr;
}

Value *FAddCombine::simplifyFAdd(AddendVect& Addends, unsigned InstrQuota) {
  unsigned AddendNum = Addends.size();
  assert(AddendNum <= 4 && "Too many addends");

  // For saving intermediate results;
  unsigned NextTmpIdx = 0;
  FAddend TmpResult[3];

  // Points to the constant addend of the resulting simplified expression.
  // If the resulting expr has constant-addend, this constant-addend is
  // desirable to reside at the top of the resulting expression tree. Placing
  // constant close to supper-expr(s) will potentially reveal some optimization
  // opportunities in super-expr(s).
  const FAddend *ConstAdd = nullptr;

  // Simplified addends are placed <SimpVect>.
  AddendVect SimpVect;

  // The outer loop works on one symbolic-value at a time. Suppose the input
  // addends are : <a1, x>, <b1, y>, <a2, x>, <c1, z>, <b2, y>, ...
  // The symbolic-values will be processed in this order: x, y, z.
  for (unsigned SymIdx = 0; SymIdx < AddendNum; SymIdx++) {

    const FAddend *ThisAddend = Addends[SymIdx];
    if (!ThisAddend) {
      // This addend was processed before.
      continue;
    }

    Value *Val = ThisAddend->getSymVal();
    unsigned StartIdx = SimpVect.size();
    SimpVect.push_back(ThisAddend);

    // The inner loop collects addends sharing same symbolic-value, and these
    // addends will be later on folded into a single addend. Following above
    // example, if the symbolic value "y" is being processed, the inner loop
    // will collect two addends "<b1,y>" and "<b2,Y>". These two addends will
    // be later on folded into "<b1+b2, y>".
    for (unsigned SameSymIdx = SymIdx + 1;
         SameSymIdx < AddendNum; SameSymIdx++) {
      const FAddend *T = Addends[SameSymIdx];
      if (T && T->getSymVal() == Val) {
        // Set null such that next iteration of the outer loop will not process
        // this addend again.
        Addends[SameSymIdx] = nullptr;
        SimpVect.push_back(T);
      }
    }

    // If multiple addends share same symbolic value, fold them together.
    if (StartIdx + 1 != SimpVect.size()) {
      FAddend &R = TmpResult[NextTmpIdx ++];
      R = *SimpVect[StartIdx];
      for (unsigned Idx = StartIdx + 1; Idx < SimpVect.size(); Idx++)
        R += *SimpVect[Idx];

      // Pop all addends being folded and push the resulting folded addend.
      SimpVect.resize(StartIdx);
      if (Val) {
        if (!R.isZero()) {
          SimpVect.push_back(&R);
        }
      } else {
        // Don't push constant addend at this time. It will be the last element
        // of <SimpVect>.
        ConstAdd = &R;
      }
    }
  }

  assert((NextTmpIdx <= array_lengthof(TmpResult) + 1) &&
         "out-of-bound access");

  if (ConstAdd)
    SimpVect.push_back(ConstAdd);

  Value *Result;
  if (!SimpVect.empty())
    Result = createNaryFAdd(SimpVect, InstrQuota);
  else {
    // The addition is folded to 0.0.
    Result = ConstantFP::get(Instr->getType(), 0.0);
  }

  return Result;
}

Value *FAddCombine::createNaryFAdd
  (const AddendVect &Opnds, unsigned InstrQuota) {
  assert(!Opnds.empty() && "Expect at least one addend");

  // Step 1: Check if the # of instructions needed exceeds the quota.

  unsigned InstrNeeded = calcInstrNumber(Opnds);
  if (InstrNeeded > InstrQuota)
    return nullptr;

  initCreateInstNum();

  // step 2: Emit the N-ary addition.
  // Note that at most three instructions are involved in Fadd-InstCombine: the
  // addition in question, and at most two neighboring instructions.
  // The resulting optimized addition should have at least one less instruction
  // than the original addition expression tree. This implies that the resulting
  // N-ary addition has at most two instructions, and we don't need to worry
  // about tree-height when constructing the N-ary addition.

  Value *LastVal = nullptr;
  bool LastValNeedNeg = false;

  // Iterate the addends, creating fadd/fsub using adjacent two addends.
  for (const FAddend *Opnd : Opnds) {
    bool NeedNeg;
    Value *V = createAddendVal(*Opnd, NeedNeg);
    if (!LastVal) {
      LastVal = V;
      LastValNeedNeg = NeedNeg;
      continue;
    }

    if (LastValNeedNeg == NeedNeg) {
      LastVal = createFAdd(LastVal, V);
      continue;
    }

    if (LastValNeedNeg)
      LastVal = createFSub(V, LastVal);
    else
      LastVal = createFSub(LastVal, V);

    LastValNeedNeg = false;
  }

  if (LastValNeedNeg) {
    LastVal = createFNeg(LastVal);
  }

#ifndef NDEBUG
  assert(CreateInstrNum == InstrNeeded &&
         "Inconsistent in instruction numbers");
#endif

  return LastVal;
}

Value *FAddCombine::createFSub(Value *Opnd0, Value *Opnd1) {
  Value *V = Builder.CreateFSub(Opnd0, Opnd1);
  if (Instruction *I = dyn_cast<Instruction>(V))
    createInstPostProc(I);
  return V;
}

Value *FAddCombine::createFNeg(Value *V) {
  Value *NewV = Builder.CreateFNeg(V);
  if (Instruction *I = dyn_cast<Instruction>(NewV))
    createInstPostProc(I, true); // fneg's don't receive instruction numbers.
  return NewV;
}

Value *FAddCombine::createFAdd(Value *Opnd0, Value *Opnd1) {
  Value *V = Builder.CreateFAdd(Opnd0, Opnd1);
  if (Instruction *I = dyn_cast<Instruction>(V))
    createInstPostProc(I);
  return V;
}

Value *FAddCombine::createFMul(Value *Opnd0, Value *Opnd1) {
  Value *V = Builder.CreateFMul(Opnd0, Opnd1);
  if (Instruction *I = dyn_cast<Instruction>(V))
    createInstPostProc(I);
  return V;
}

void FAddCombine::createInstPostProc(Instruction *NewInstr, bool NoNumber) {
  NewInstr->setDebugLoc(Instr->getDebugLoc());

  // Keep track of the number of instruction created.
  if (!NoNumber)
    incCreateInstNum();

  // Propagate fast-math flags
  NewInstr->setFastMathFlags(Instr->getFastMathFlags());
}

// Return the number of instruction needed to emit the N-ary addition.
// NOTE: Keep this function in sync with createAddendVal().
unsigned FAddCombine::calcInstrNumber(const AddendVect &Opnds) {
  unsigned OpndNum = Opnds.size();
  unsigned InstrNeeded = OpndNum - 1;

  // The number of addends in the form of "(-1)*x".
  unsigned NegOpndNum = 0;

  // Adjust the number of instructions needed to emit the N-ary add.
  for (const FAddend *Opnd : Opnds) {
    if (Opnd->isConstant())
      continue;

    // The constant check above is really for a few special constant
    // coefficients.
    if (isa<UndefValue>(Opnd->getSymVal()))
      continue;

    const FAddendCoef &CE = Opnd->getCoef();
    if (CE.isMinusOne() || CE.isMinusTwo())
      NegOpndNum++;

    // Let the addend be "c * x". If "c == +/-1", the value of the addend
    // is immediately available; otherwise, it needs exactly one instruction
    // to evaluate the value.
    if (!CE.isMinusOne() && !CE.isOne())
      InstrNeeded++;
  }
  return InstrNeeded;
}

// Input Addend        Value           NeedNeg(output)
// ================================================================
// Constant C          C               false
// <+/-1, V>           V               coefficient is -1
// <2/-2, V>          "fadd V, V"      coefficient is -2
// <C, V>             "fmul V, C"      false
//
// NOTE: Keep this function in sync with FAddCombine::calcInstrNumber.
Value *FAddCombine::createAddendVal(const FAddend &Opnd, bool &NeedNeg) {
  const FAddendCoef &Coeff = Opnd.getCoef();

  if (Opnd.isConstant()) {
    NeedNeg = false;
    return Coeff.getValue(Instr->getType());
  }

  Value *OpndVal = Opnd.getSymVal();

  if (Coeff.isMinusOne() || Coeff.isOne()) {
    NeedNeg = Coeff.isMinusOne();
    return OpndVal;
  }

  if (Coeff.isTwo() || Coeff.isMinusTwo()) {
    NeedNeg = Coeff.isMinusTwo();
    return createFAdd(OpndVal, OpndVal);
  }

  NeedNeg = false;
  return createFMul(OpndVal, Coeff.getValue(Instr->getType()));
}

// Checks if any operand is negative and we can convert add to sub.
// This function checks for following negative patterns
//   ADD(XOR(OR(Z, NOT(C)), C)), 1) == NEG(AND(Z, C))
//   ADD(XOR(AND(Z, C), C), 1) == NEG(OR(Z, ~C))
//   XOR(AND(Z, C), (C + 1)) == NEG(OR(Z, ~C)) if C is even
static Value *checkForNegativeOperand(BinaryOperator &I,
                                      InstCombiner::BuilderTy &Builder) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  // This function creates 2 instructions to replace ADD, we need at least one
  // of LHS or RHS to have one use to ensure benefit in transform.
  if (!LHS->hasOneUse() && !RHS->hasOneUse())
    return nullptr;

  Value *X = nullptr, *Y = nullptr, *Z = nullptr;
  const APInt *C1 = nullptr, *C2 = nullptr;

  // if ONE is on other side, swap
  if (match(RHS, m_Add(m_Value(X), m_One())))
    std::swap(LHS, RHS);

  if (match(LHS, m_Add(m_Value(X), m_One()))) {
    // if XOR on other side, swap
    if (match(RHS, m_Xor(m_Value(Y), m_APInt(C1))))
      std::swap(X, RHS);

    if (match(X, m_Xor(m_Value(Y), m_APInt(C1)))) {
      // X = XOR(Y, C1), Y = OR(Z, C2), C2 = NOT(C1) ==> X == NOT(AND(Z, C1))
      // ADD(ADD(X, 1), RHS) == ADD(X, ADD(RHS, 1)) == SUB(RHS, AND(Z, C1))
      if (match(Y, m_Or(m_Value(Z), m_APInt(C2))) && (*C2 == ~(*C1))) {
        Value *NewAnd = Builder.CreateAnd(Z, *C1);
        return Builder.CreateSub(RHS, NewAnd, "sub");
      } else if (match(Y, m_And(m_Value(Z), m_APInt(C2))) && (*C1 == *C2)) {
        // X = XOR(Y, C1), Y = AND(Z, C2), C2 == C1 ==> X == NOT(OR(Z, ~C1))
        // ADD(ADD(X, 1), RHS) == ADD(X, ADD(RHS, 1)) == SUB(RHS, OR(Z, ~C1))
        Value *NewOr = Builder.CreateOr(Z, ~(*C1));
        return Builder.CreateSub(RHS, NewOr, "sub");
      }
    }
  }

  // Restore LHS and RHS
  LHS = I.getOperand(0);
  RHS = I.getOperand(1);

  // if XOR is on other side, swap
  if (match(RHS, m_Xor(m_Value(Y), m_APInt(C1))))
    std::swap(LHS, RHS);

  // C2 is ODD
  // LHS = XOR(Y, C1), Y = AND(Z, C2), C1 == (C2 + 1) => LHS == NEG(OR(Z, ~C2))
  // ADD(LHS, RHS) == SUB(RHS, OR(Z, ~C2))
  if (match(LHS, m_Xor(m_Value(Y), m_APInt(C1))))
    if (C1->countTrailingZeros() == 0)
      if (match(Y, m_And(m_Value(Z), m_APInt(C2))) && *C1 == (*C2 + 1)) {
        Value *NewOr = Builder.CreateOr(Z, ~(*C2));
        return Builder.CreateSub(RHS, NewOr, "sub");
      }
  return nullptr;
}

/// Wrapping flags may allow combining constants separated by an extend.
static Instruction *foldNoWrapAdd(BinaryOperator &Add,
                                  InstCombiner::BuilderTy &Builder) {
  Value *Op0 = Add.getOperand(0), *Op1 = Add.getOperand(1);
  Type *Ty = Add.getType();
  Constant *Op1C;
  if (!match(Op1, m_Constant(Op1C)))
    return nullptr;

  // Try this match first because it results in an add in the narrow type.
  // (zext (X +nuw C2)) + C1 --> zext (X + (C2 + trunc(C1)))
  Value *X;
  const APInt *C1, *C2;
  if (match(Op1, m_APInt(C1)) &&
      match(Op0, m_OneUse(m_ZExt(m_NUWAdd(m_Value(X), m_APInt(C2))))) &&
      C1->isNegative() && C1->sge(-C2->sext(C1->getBitWidth()))) {
    Constant *NewC =
        ConstantInt::get(X->getType(), *C2 + C1->trunc(C2->getBitWidth()));
    return new ZExtInst(Builder.CreateNUWAdd(X, NewC), Ty);
  }

  // More general combining of constants in the wide type.
  // (sext (X +nsw NarrowC)) + C --> (sext X) + (sext(NarrowC) + C)
  Constant *NarrowC;
  if (match(Op0, m_OneUse(m_SExt(m_NSWAdd(m_Value(X), m_Constant(NarrowC)))))) {
    Constant *WideC = ConstantExpr::getSExt(NarrowC, Ty);
    Constant *NewC = ConstantExpr::getAdd(WideC, Op1C);
    Value *WideX = Builder.CreateSExt(X, Ty);
    return BinaryOperator::CreateAdd(WideX, NewC);
  }
  // (zext (X +nuw NarrowC)) + C --> (zext X) + (zext(NarrowC) + C)
  if (match(Op0, m_OneUse(m_ZExt(m_NUWAdd(m_Value(X), m_Constant(NarrowC)))))) {
    Constant *WideC = ConstantExpr::getZExt(NarrowC, Ty);
    Constant *NewC = ConstantExpr::getAdd(WideC, Op1C);
    Value *WideX = Builder.CreateZExt(X, Ty);
    return BinaryOperator::CreateAdd(WideX, NewC);
  }

  return nullptr;
}

Instruction *InstCombiner::foldAddWithConstant(BinaryOperator &Add) {
  Value *Op0 = Add.getOperand(0), *Op1 = Add.getOperand(1);
  Constant *Op1C;
  if (!match(Op1, m_Constant(Op1C)))
    return nullptr;

  if (Instruction *NV = foldBinOpIntoSelectOrPhi(Add))
    return NV;

  Value *X;
  Constant *Op00C;

  // add (sub C1, X), C2 --> sub (add C1, C2), X
  if (match(Op0, m_Sub(m_Constant(Op00C), m_Value(X))))
    return BinaryOperator::CreateSub(ConstantExpr::getAdd(Op00C, Op1C), X);

  Value *Y;

  // add (sub X, Y), -1 --> add (not Y), X
  if (match(Op0, m_OneUse(m_Sub(m_Value(X), m_Value(Y)))) &&
      match(Op1, m_AllOnes()))
    return BinaryOperator::CreateAdd(Builder.CreateNot(Y), X);

  // zext(bool) + C -> bool ? C + 1 : C
  if (match(Op0, m_ZExt(m_Value(X))) &&
      X->getType()->getScalarSizeInBits() == 1)
    return SelectInst::Create(X, AddOne(Op1C), Op1);
  // sext(bool) + C -> bool ? C - 1 : C
  if (match(Op0, m_SExt(m_Value(X))) &&
      X->getType()->getScalarSizeInBits() == 1)
    return SelectInst::Create(X, SubOne(Op1C), Op1);

  // ~X + C --> (C-1) - X
  if (match(Op0, m_Not(m_Value(X))))
    return BinaryOperator::CreateSub(SubOne(Op1C), X);

  const APInt *C;
  if (!match(Op1, m_APInt(C)))
    return nullptr;

  // (X | C2) + C --> (X | C2) ^ C2 iff (C2 == -C)
  const APInt *C2;
  if (match(Op0, m_Or(m_Value(), m_APInt(C2))) && *C2 == -*C)
    return BinaryOperator::CreateXor(Op0, ConstantInt::get(Add.getType(), *C2));

  if (C->isSignMask()) {
    // If wrapping is not allowed, then the addition must set the sign bit:
    // X + (signmask) --> X | signmask
    if (Add.hasNoSignedWrap() || Add.hasNoUnsignedWrap())
      return BinaryOperator::CreateOr(Op0, Op1);

    // If wrapping is allowed, then the addition flips the sign bit of LHS:
    // X + (signmask) --> X ^ signmask
    return BinaryOperator::CreateXor(Op0, Op1);
  }

  // Is this add the last step in a convoluted sext?
  // add(zext(xor i16 X, -32768), -32768) --> sext X
  Type *Ty = Add.getType();
  if (match(Op0, m_ZExt(m_Xor(m_Value(X), m_APInt(C2)))) &&
      C2->isMinSignedValue() && C2->sext(Ty->getScalarSizeInBits()) == *C)
    return CastInst::Create(Instruction::SExt, X, Ty);

  if (C->isOneValue() && Op0->hasOneUse()) {
    // add (sext i1 X), 1 --> zext (not X)
    // TODO: The smallest IR representation is (select X, 0, 1), and that would
    // not require the one-use check. But we need to remove a transform in
    // visitSelect and make sure that IR value tracking for select is equal or
    // better than for these ops.
    if (match(Op0, m_SExt(m_Value(X))) &&
        X->getType()->getScalarSizeInBits() == 1)
      return new ZExtInst(Builder.CreateNot(X), Ty);

    // Shifts and add used to flip and mask off the low bit:
    // add (ashr (shl i32 X, 31), 31), 1 --> and (not X), 1
    const APInt *C3;
    if (match(Op0, m_AShr(m_Shl(m_Value(X), m_APInt(C2)), m_APInt(C3))) &&
        C2 == C3 && *C2 == Ty->getScalarSizeInBits() - 1) {
      Value *NotX = Builder.CreateNot(X);
      return BinaryOperator::CreateAnd(NotX, ConstantInt::get(Ty, 1));
    }
  }

  return nullptr;
}

// Matches multiplication expression Op * C where C is a constant. Returns the
// constant value in C and the other operand in Op. Returns true if such a
// match is found.
static bool MatchMul(Value *E, Value *&Op, APInt &C) {
  const APInt *AI;
  if (match(E, m_Mul(m_Value(Op), m_APInt(AI)))) {
    C = *AI;
    return true;
  }
  if (match(E, m_Shl(m_Value(Op), m_APInt(AI)))) {
    C = APInt(AI->getBitWidth(), 1);
    C <<= *AI;
    return true;
  }
  return false;
}

// Matches remainder expression Op % C where C is a constant. Returns the
// constant value in C and the other operand in Op. Returns the signedness of
// the remainder operation in IsSigned. Returns true if such a match is
// found.
static bool MatchRem(Value *E, Value *&Op, APInt &C, bool &IsSigned) {
  const APInt *AI;
  IsSigned = false;
  if (match(E, m_SRem(m_Value(Op), m_APInt(AI)))) {
    IsSigned = true;
    C = *AI;
    return true;
  }
  if (match(E, m_URem(m_Value(Op), m_APInt(AI)))) {
    C = *AI;
    return true;
  }
  if (match(E, m_And(m_Value(Op), m_APInt(AI))) && (*AI + 1).isPowerOf2()) {
    C = *AI + 1;
    return true;
  }
  return false;
}

// Matches division expression Op / C with the given signedness as indicated
// by IsSigned, where C is a constant. Returns the constant value in C and the
// other operand in Op. Returns true if such a match is found.
static bool MatchDiv(Value *E, Value *&Op, APInt &C, bool IsSigned) {
  const APInt *AI;
  if (IsSigned && match(E, m_SDiv(m_Value(Op), m_APInt(AI)))) {
    C = *AI;
    return true;
  }
  if (!IsSigned) {
    if (match(E, m_UDiv(m_Value(Op), m_APInt(AI)))) {
      C = *AI;
      return true;
    }
    if (match(E, m_LShr(m_Value(Op), m_APInt(AI)))) {
      C = APInt(AI->getBitWidth(), 1);
      C <<= *AI;
      return true;
    }
  }
  return false;
}

// Returns whether C0 * C1 with the given signedness overflows.
static bool MulWillOverflow(APInt &C0, APInt &C1, bool IsSigned) {
  bool overflow;
  if (IsSigned)
    (void)C0.smul_ov(C1, overflow);
  else
    (void)C0.umul_ov(C1, overflow);
  return overflow;
}

// Simplifies X % C0 + (( X / C0 ) % C1) * C0 to X % (C0 * C1), where (C0 * C1)
// does not overflow.
Value *InstCombiner::SimplifyAddWithRemainder(BinaryOperator &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  Value *X, *MulOpV;
  APInt C0, MulOpC;
  bool IsSigned;
  // Match I = X % C0 + MulOpV * C0
  if (((MatchRem(LHS, X, C0, IsSigned) && MatchMul(RHS, MulOpV, MulOpC)) ||
       (MatchRem(RHS, X, C0, IsSigned) && MatchMul(LHS, MulOpV, MulOpC))) &&
      C0 == MulOpC) {
    Value *RemOpV;
    APInt C1;
    bool Rem2IsSigned;
    // Match MulOpC = RemOpV % C1
    if (MatchRem(MulOpV, RemOpV, C1, Rem2IsSigned) &&
        IsSigned == Rem2IsSigned) {
      Value *DivOpV;
      APInt DivOpC;
      // Match RemOpV = X / C0
      if (MatchDiv(RemOpV, DivOpV, DivOpC, IsSigned) && X == DivOpV &&
          C0 == DivOpC && !MulWillOverflow(C0, C1, IsSigned)) {
        Value *NewDivisor = ConstantInt::get(X->getType(), C0 * C1);
        return IsSigned ? Builder.CreateSRem(X, NewDivisor, "srem")
                        : Builder.CreateURem(X, NewDivisor, "urem");
      }
    }
  }

  return nullptr;
}

/// Fold
///   (1 << NBits) - 1
/// Into:
///   ~(-(1 << NBits))
/// Because a 'not' is better for bit-tracking analysis and other transforms
/// than an 'add'. The new shl is always nsw, and is nuw if old `and` was.
static Instruction *canonicalizeLowbitMask(BinaryOperator &I,
                                           InstCombiner::BuilderTy &Builder) {
  Value *NBits;
  if (!match(&I, m_Add(m_OneUse(m_Shl(m_One(), m_Value(NBits))), m_AllOnes())))
    return nullptr;

  Constant *MinusOne = Constant::getAllOnesValue(NBits->getType());
  Value *NotMask = Builder.CreateShl(MinusOne, NBits, "notmask");
  // Be wary of constant folding.
  if (auto *BOp = dyn_cast<BinaryOperator>(NotMask)) {
    // Always NSW. But NUW propagates from `add`.
    BOp->setHasNoSignedWrap();
    BOp->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
  }

  return BinaryOperator::CreateNot(NotMask, I.getName());
}

static Instruction *foldToUnsignedSaturatedAdd(BinaryOperator &I) {
  assert(I.getOpcode() == Instruction::Add && "Expecting add instruction");
  Type *Ty = I.getType();
  auto getUAddSat = [&]() {
    return Intrinsic::getDeclaration(I.getModule(), Intrinsic::uadd_sat, Ty);
  };

  // add (umin X, ~Y), Y --> uaddsat X, Y
  Value *X, *Y;
  if (match(&I, m_c_Add(m_c_UMin(m_Value(X), m_Not(m_Value(Y))),
                        m_Deferred(Y))))
    return CallInst::Create(getUAddSat(), { X, Y });

  // add (umin X, ~C), C --> uaddsat X, C
  const APInt *C, *NotC;
  if (match(&I, m_Add(m_UMin(m_Value(X), m_APInt(NotC)), m_APInt(C))) &&
      *C == ~*NotC)
    return CallInst::Create(getUAddSat(), { X, ConstantInt::get(Ty, *C) });

  return nullptr;
}

Instruction *
InstCombiner::canonicalizeCondSignextOfHighBitExtractToSignextHighBitExtract(
    BinaryOperator &I) {
  assert((I.getOpcode() == Instruction::Add ||
          I.getOpcode() == Instruction::Or ||
          I.getOpcode() == Instruction::Sub) &&
         "Expecting add/or/sub instruction");

  // We have a subtraction/addition between a (potentially truncated) *logical*
  // right-shift of X and a "select".
  Value *X, *Select;
  Instruction *LowBitsToSkip, *Extract;
  if (!match(&I, m_c_BinOp(m_TruncOrSelf(m_CombineAnd(
                               m_LShr(m_Value(X), m_Instruction(LowBitsToSkip)),
                               m_Instruction(Extract))),
                           m_Value(Select))))
    return nullptr;

  // `add`/`or` is commutative; but for `sub`, "select" *must* be on RHS.
  if (I.getOpcode() == Instruction::Sub && I.getOperand(1) != Select)
    return nullptr;

  Type *XTy = X->getType();
  bool HadTrunc = I.getType() != XTy;

  // If there was a truncation of extracted value, then we'll need to produce
  // one extra instruction, so we need to ensure one instruction will go away.
  if (HadTrunc && !match(&I, m_c_BinOp(m_OneUse(m_Value()), m_Value())))
    return nullptr;

  // Extraction should extract high NBits bits, with shift amount calculated as:
  //   low bits to skip = shift bitwidth - high bits to extract
  // The shift amount itself may be extended, and we need to look past zero-ext
  // when matching NBits, that will matter for matching later.
  Constant *C;
  Value *NBits;
  if (!match(
          LowBitsToSkip,
          m_ZExtOrSelf(m_Sub(m_Constant(C), m_ZExtOrSelf(m_Value(NBits))))) ||
      !match(C, m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ,
                                   APInt(C->getType()->getScalarSizeInBits(),
                                         X->getType()->getScalarSizeInBits()))))
    return nullptr;

  // Sign-extending value can be zero-extended if we `sub`tract it,
  // or sign-extended otherwise.
  auto SkipExtInMagic = [&I](Value *&V) {
    if (I.getOpcode() == Instruction::Sub)
      match(V, m_ZExtOrSelf(m_Value(V)));
    else
      match(V, m_SExtOrSelf(m_Value(V)));
  };

  // Now, finally validate the sign-extending magic.
  // `select` itself may be appropriately extended, look past that.
  SkipExtInMagic(Select);

  ICmpInst::Predicate Pred;
  const APInt *Thr;
  Value *SignExtendingValue, *Zero;
  bool ShouldSignext;
  // It must be a select between two values we will later establish to be a
  // sign-extending value and a zero constant. The condition guarding the
  // sign-extension must be based on a sign bit of the same X we had in `lshr`.
  if (!match(Select, m_Select(m_ICmp(Pred, m_Specific(X), m_APInt(Thr)),
                              m_Value(SignExtendingValue), m_Value(Zero))) ||
      !isSignBitCheck(Pred, *Thr, ShouldSignext))
    return nullptr;

  // icmp-select pair is commutative.
  if (!ShouldSignext)
    std::swap(SignExtendingValue, Zero);

  // If we should not perform sign-extension then we must add/or/subtract zero.
  if (!match(Zero, m_Zero()))
    return nullptr;
  // Otherwise, it should be some constant, left-shifted by the same NBits we
  // had in `lshr`. Said left-shift can also be appropriately extended.
  // Again, we must look past zero-ext when looking for NBits.
  SkipExtInMagic(SignExtendingValue);
  Constant *SignExtendingValueBaseConstant;
  if (!match(SignExtendingValue,
             m_Shl(m_Constant(SignExtendingValueBaseConstant),
                   m_ZExtOrSelf(m_Specific(NBits)))))
    return nullptr;
  // If we `sub`, then the constant should be one, else it should be all-ones.
  if (I.getOpcode() == Instruction::Sub
          ? !match(SignExtendingValueBaseConstant, m_One())
          : !match(SignExtendingValueBaseConstant, m_AllOnes()))
    return nullptr;

  auto *NewAShr = BinaryOperator::CreateAShr(X, LowBitsToSkip,
                                             Extract->getName() + ".sext");
  NewAShr->copyIRFlags(Extract); // Preserve `exact`-ness.
  if (!HadTrunc)
    return NewAShr;

  Builder.Insert(NewAShr);
  return TruncInst::CreateTruncOrBitCast(NewAShr, I.getType());
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  if (Value *V = SimplifyAddInst(I.getOperand(0), I.getOperand(1),
                                 I.hasNoSignedWrap(), I.hasNoUnsignedWrap(),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  // (A*B)+(A*C) -> A*(B+C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldAddWithConstant(I))
    return X;

  if (Instruction *X = foldNoWrapAdd(I, Builder))
    return X;

  // FIXME: This should be moved into the above helper function to allow these
  // transforms for general constant or constant splat vectors.
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  Type *Ty = I.getType();
  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    Value *XorLHS = nullptr; ConstantInt *XorRHS = nullptr;
    if (match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      unsigned TySizeBits = Ty->getScalarSizeInBits();
      const APInt &RHSVal = CI->getValue();
      unsigned ExtendAmt = 0;
      // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
      // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
      if (XorRHS->getValue() == -RHSVal) {
        if (RHSVal.isPowerOf2())
          ExtendAmt = TySizeBits - RHSVal.logBase2() - 1;
        else if (XorRHS->getValue().isPowerOf2())
          ExtendAmt = TySizeBits - XorRHS->getValue().logBase2() - 1;
      }

      if (ExtendAmt) {
        APInt Mask = APInt::getHighBitsSet(TySizeBits, ExtendAmt);
        if (!MaskedValueIsZero(XorLHS, Mask, 0, &I))
          ExtendAmt = 0;
      }

      if (ExtendAmt) {
        Constant *ShAmt = ConstantInt::get(Ty, ExtendAmt);
        Value *NewShl = Builder.CreateShl(XorLHS, ShAmt, "sext");
        return BinaryOperator::CreateAShr(NewShl, ShAmt);
      }

      // If this is a xor that was canonicalized from a sub, turn it back into
      // a sub and fuse this add with it.
      if (LHS->hasOneUse() && (XorRHS->getValue()+1).isPowerOf2()) {
        KnownBits LHSKnown = computeKnownBits(XorLHS, 0, &I);
        if ((XorRHS->getValue() | LHSKnown.Zero).isAllOnesValue())
          return BinaryOperator::CreateSub(ConstantExpr::getAdd(XorRHS, CI),
                                           XorLHS);
      }
      // (X + signmask) + C could have gotten canonicalized to (X^signmask) + C,
      // transform them into (X + (signmask ^ C))
      if (XorRHS->getValue().isSignMask())
        return BinaryOperator::CreateAdd(XorLHS,
                                         ConstantExpr::getXor(XorRHS, CI));
    }
  }

  if (Ty->isIntOrIntVectorTy(1))
    return BinaryOperator::CreateXor(LHS, RHS);

  // X + X --> X << 1
  if (LHS == RHS) {
    auto *Shl = BinaryOperator::CreateShl(LHS, ConstantInt::get(Ty, 1));
    Shl->setHasNoSignedWrap(I.hasNoSignedWrap());
    Shl->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
    return Shl;
  }

  Value *A, *B;
  if (match(LHS, m_Neg(m_Value(A)))) {
    // -A + -B --> -(A + B)
    if (match(RHS, m_Neg(m_Value(B))))
      return BinaryOperator::CreateNeg(Builder.CreateAdd(A, B));

    // -A + B --> B - A
    return BinaryOperator::CreateSub(RHS, A);
  }

  // A + -B  -->  A - B
  if (match(RHS, m_Neg(m_Value(B))))
    return BinaryOperator::CreateSub(LHS, B);

  if (Value *V = checkForNegativeOperand(I, Builder))
    return replaceInstUsesWith(I, V);

  // (A + 1) + ~B --> A - B
  // ~B + (A + 1) --> A - B
  // (~B + A) + 1 --> A - B
  // (A + ~B) + 1 --> A - B
  if (match(&I, m_c_BinOp(m_Add(m_Value(A), m_One()), m_Not(m_Value(B)))) ||
      match(&I, m_BinOp(m_c_Add(m_Not(m_Value(B)), m_Value(A)), m_One())))
    return BinaryOperator::CreateSub(A, B);

  // (A + RHS) + RHS --> A + (RHS << 1)
  if (match(LHS, m_OneUse(m_c_Add(m_Value(A), m_Specific(RHS)))))
    return BinaryOperator::CreateAdd(A, Builder.CreateShl(RHS, 1, "reass.add"));

  // LHS + (A + LHS) --> A + (LHS << 1)
  if (match(RHS, m_OneUse(m_c_Add(m_Value(A), m_Specific(LHS)))))
    return BinaryOperator::CreateAdd(A, Builder.CreateShl(LHS, 1, "reass.add"));

  // X % C0 + (( X / C0 ) % C1) * C0 => X % (C0 * C1)
  if (Value *V = SimplifyAddWithRemainder(I)) return replaceInstUsesWith(I, V);

  // ((X s/ C1) << C2) + X => X s% -C1 where -C1 is 1 << C2
  const APInt *C1, *C2;
  if (match(LHS, m_Shl(m_SDiv(m_Specific(RHS), m_APInt(C1)), m_APInt(C2)))) {
    APInt one(C2->getBitWidth(), 1);
    APInt minusC1 = -(*C1);
    if (minusC1 == (one << *C2)) {
      Constant *NewRHS = ConstantInt::get(RHS->getType(), minusC1);
      return BinaryOperator::CreateSRem(RHS, NewRHS);
    }
  }

  // A+B --> A|B iff A and B have no bits set in common.
  if (haveNoCommonBitsSet(LHS, RHS, DL, &AC, &I, &DT))
    return BinaryOperator::CreateOr(LHS, RHS);

  // FIXME: We already did a check for ConstantInt RHS above this.
  // FIXME: Is this pattern covered by another fold? No regression tests fail on
  // removal.
  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    // (X & FF00) + xx00  -> (X+xx00) & FF00
    Value *X;
    ConstantInt *C2;
    if (LHS->hasOneUse() &&
        match(LHS, m_And(m_Value(X), m_ConstantInt(C2))) &&
        CRHS->getValue() == (CRHS->getValue() & C2->getValue())) {
      // See if all bits from the first bit set in the Add RHS up are included
      // in the mask.  First, get the rightmost bit.
      const APInt &AddRHSV = CRHS->getValue();

      // Form a mask of all bits from the lowest bit added through the top.
      APInt AddRHSHighBits(~((AddRHSV & -AddRHSV)-1));

      // See if the and mask includes all of these bits.
      APInt AddRHSHighBitsAnd(AddRHSHighBits & C2->getValue());

      if (AddRHSHighBits == AddRHSHighBitsAnd) {
        // Okay, the xform is safe.  Insert the new add pronto.
        Value *NewAdd = Builder.CreateAdd(X, CRHS, LHS->getName());
        return BinaryOperator::CreateAnd(NewAdd, C2);
      }
    }
  }

  // add (select X 0 (sub n A)) A  -->  select X A n
  {
    SelectInst *SI = dyn_cast<SelectInst>(LHS);
    Value *A = RHS;
    if (!SI) {
      SI = dyn_cast<SelectInst>(RHS);
      A = LHS;
    }
    if (SI && SI->hasOneUse()) {
      Value *TV = SI->getTrueValue();
      Value *FV = SI->getFalseValue();
      Value *N;

      // Can we fold the add into the argument of the select?
      // We check both true and false select arguments for a matching subtract.
      if (match(FV, m_Zero()) && match(TV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the true select value.
        return SelectInst::Create(SI->getCondition(), N, A);

      if (match(TV, m_Zero()) && match(FV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the false select value.
        return SelectInst::Create(SI->getCondition(), A, N);
    }
  }

  if (Instruction *Ext = narrowMathIfNoOverflow(I))
    return Ext;

  // (add (xor A, B) (and A, B)) --> (or A, B)
  // (add (and A, B) (xor A, B)) --> (or A, B)
  if (match(&I, m_c_BinOp(m_Xor(m_Value(A), m_Value(B)),
                          m_c_And(m_Deferred(A), m_Deferred(B)))))
    return BinaryOperator::CreateOr(A, B);

  // (add (or A, B) (and A, B)) --> (add A, B)
  // (add (and A, B) (or A, B)) --> (add A, B)
  if (match(&I, m_c_BinOp(m_Or(m_Value(A), m_Value(B)),
                          m_c_And(m_Deferred(A), m_Deferred(B))))) {
    // Replacing operands in-place to preserve nuw/nsw flags.
    replaceOperand(I, 0, A);
    replaceOperand(I, 1, B);
    return &I;
  }

  // TODO(jingyue): Consider willNotOverflowSignedAdd and
  // willNotOverflowUnsignedAdd to reduce the number of invocations of
  // computeKnownBits.
  bool Changed = false;
  if (!I.hasNoSignedWrap() && willNotOverflowSignedAdd(LHS, RHS, I)) {
    Changed = true;
    I.setHasNoSignedWrap(true);
  }
  if (!I.hasNoUnsignedWrap() && willNotOverflowUnsignedAdd(LHS, RHS, I)) {
    Changed = true;
    I.setHasNoUnsignedWrap(true);
  }

  if (Instruction *V = canonicalizeLowbitMask(I, Builder))
    return V;

  if (Instruction *V =
          canonicalizeCondSignextOfHighBitExtractToSignextHighBitExtract(I))
    return V;

  if (Instruction *SatAdd = foldToUnsignedSaturatedAdd(I))
    return SatAdd;

  return Changed ? &I : nullptr;
}

/// Eliminate an op from a linear interpolation (lerp) pattern.
static Instruction *factorizeLerp(BinaryOperator &I,
                                  InstCombiner::BuilderTy &Builder) {
  Value *X, *Y, *Z;
  if (!match(&I, m_c_FAdd(m_OneUse(m_c_FMul(m_Value(Y),
                                            m_OneUse(m_FSub(m_FPOne(),
                                                            m_Value(Z))))),
                          m_OneUse(m_c_FMul(m_Value(X), m_Deferred(Z))))))
    return nullptr;

  // (Y * (1.0 - Z)) + (X * Z) --> Y + Z * (X - Y) [8 commuted variants]
  Value *XY = Builder.CreateFSubFMF(X, Y, &I);
  Value *MulZ = Builder.CreateFMulFMF(Z, XY, &I);
  return BinaryOperator::CreateFAddFMF(Y, MulZ, &I);
}

/// Factor a common operand out of fadd/fsub of fmul/fdiv.
static Instruction *factorizeFAddFSub(BinaryOperator &I,
                                      InstCombiner::BuilderTy &Builder) {
  assert((I.getOpcode() == Instruction::FAdd ||
          I.getOpcode() == Instruction::FSub) && "Expecting fadd/fsub");
  assert(I.hasAllowReassoc() && I.hasNoSignedZeros() &&
         "FP factorization requires FMF");

  if (Instruction *Lerp = factorizeLerp(I, Builder))
    return Lerp;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *X, *Y, *Z;
  bool IsFMul;
  if ((match(Op0, m_OneUse(m_FMul(m_Value(X), m_Value(Z)))) &&
       match(Op1, m_OneUse(m_c_FMul(m_Value(Y), m_Specific(Z))))) ||
      (match(Op0, m_OneUse(m_FMul(m_Value(Z), m_Value(X)))) &&
       match(Op1, m_OneUse(m_c_FMul(m_Value(Y), m_Specific(Z))))))
    IsFMul = true;
  else if (match(Op0, m_OneUse(m_FDiv(m_Value(X), m_Value(Z)))) &&
           match(Op1, m_OneUse(m_FDiv(m_Value(Y), m_Specific(Z)))))
    IsFMul = false;
  else
    return nullptr;

  // (X * Z) + (Y * Z) --> (X + Y) * Z
  // (X * Z) - (Y * Z) --> (X - Y) * Z
  // (X / Z) + (Y / Z) --> (X + Y) / Z
  // (X / Z) - (Y / Z) --> (X - Y) / Z
  bool IsFAdd = I.getOpcode() == Instruction::FAdd;
  Value *XY = IsFAdd ? Builder.CreateFAddFMF(X, Y, &I)
                     : Builder.CreateFSubFMF(X, Y, &I);

  // Bail out if we just created a denormal constant.
  // TODO: This is copied from a previous implementation. Is it necessary?
  const APFloat *C;
  if (match(XY, m_APFloat(C)) && !C->isNormal())
    return nullptr;

  return IsFMul ? BinaryOperator::CreateFMulFMF(XY, Z, &I)
                : BinaryOperator::CreateFDivFMF(XY, Z, &I);
}

Instruction *InstCombiner::visitFAdd(BinaryOperator &I) {
  if (Value *V = SimplifyFAddInst(I.getOperand(0), I.getOperand(1),
                                  I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *FoldedFAdd = foldBinOpIntoSelectOrPhi(I))
    return FoldedFAdd;

  // (-X) + Y --> Y - X
  Value *X, *Y;
  if (match(&I, m_c_FAdd(m_FNeg(m_Value(X)), m_Value(Y))))
    return BinaryOperator::CreateFSubFMF(Y, X, &I);

  // Similar to above, but look through fmul/fdiv for the negated term.
  // (-X * Y) + Z --> Z - (X * Y) [4 commuted variants]
  Value *Z;
  if (match(&I, m_c_FAdd(m_OneUse(m_c_FMul(m_FNeg(m_Value(X)), m_Value(Y))),
                         m_Value(Z)))) {
    Value *XY = Builder.CreateFMulFMF(X, Y, &I);
    return BinaryOperator::CreateFSubFMF(Z, XY, &I);
  }
  // (-X / Y) + Z --> Z - (X / Y) [2 commuted variants]
  // (X / -Y) + Z --> Z - (X / Y) [2 commuted variants]
  if (match(&I, m_c_FAdd(m_OneUse(m_FDiv(m_FNeg(m_Value(X)), m_Value(Y))),
                         m_Value(Z))) ||
      match(&I, m_c_FAdd(m_OneUse(m_FDiv(m_Value(X), m_FNeg(m_Value(Y)))),
                         m_Value(Z)))) {
    Value *XY = Builder.CreateFDivFMF(X, Y, &I);
    return BinaryOperator::CreateFSubFMF(Z, XY, &I);
  }

  // Check for (fadd double (sitofp x), y), see if we can merge this into an
  // integer add followed by a promotion.
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  if (SIToFPInst *LHSConv = dyn_cast<SIToFPInst>(LHS)) {
    Value *LHSIntVal = LHSConv->getOperand(0);
    Type *FPType = LHSConv->getType();

    // TODO: This check is overly conservative. In many cases known bits
    // analysis can tell us that the result of the addition has less significant
    // bits than the integer type can hold.
    auto IsValidPromotion = [](Type *FTy, Type *ITy) {
      Type *FScalarTy = FTy->getScalarType();
      Type *IScalarTy = ITy->getScalarType();

      // Do we have enough bits in the significand to represent the result of
      // the integer addition?
      unsigned MaxRepresentableBits =
          APFloat::semanticsPrecision(FScalarTy->getFltSemantics());
      return IScalarTy->getIntegerBitWidth() <= MaxRepresentableBits;
    };

    // (fadd double (sitofp x), fpcst) --> (sitofp (add int x, intcst))
    // ... if the constant fits in the integer value.  This is useful for things
    // like (double)(x & 1234) + 4.0 -> (double)((X & 1234)+4) which no longer
    // requires a constant pool load, and generally allows the add to be better
    // instcombined.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS))
      if (IsValidPromotion(FPType, LHSIntVal->getType())) {
        Constant *CI =
          ConstantExpr::getFPToSI(CFP, LHSIntVal->getType());
        if (LHSConv->hasOneUse() &&
            ConstantExpr::getSIToFP(CI, I.getType()) == CFP &&
            willNotOverflowSignedAdd(LHSIntVal, CI, I)) {
          // Insert the new integer add.
          Value *NewAdd = Builder.CreateNSWAdd(LHSIntVal, CI, "addconv");
          return new SIToFPInst(NewAdd, I.getType());
        }
      }

    // (fadd double (sitofp x), (sitofp y)) --> (sitofp (add int x, y))
    if (SIToFPInst *RHSConv = dyn_cast<SIToFPInst>(RHS)) {
      Value *RHSIntVal = RHSConv->getOperand(0);
      // It's enough to check LHS types only because we require int types to
      // be the same for this transform.
      if (IsValidPromotion(FPType, LHSIntVal->getType())) {
        // Only do this if x/y have the same type, if at least one of them has a
        // single use (so we don't increase the number of int->fp conversions),
        // and if the integer add will not overflow.
        if (LHSIntVal->getType() == RHSIntVal->getType() &&
            (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
            willNotOverflowSignedAdd(LHSIntVal, RHSIntVal, I)) {
          // Insert the new integer add.
          Value *NewAdd = Builder.CreateNSWAdd(LHSIntVal, RHSIntVal, "addconv");
          return new SIToFPInst(NewAdd, I.getType());
        }
      }
    }
  }

  // Handle specials cases for FAdd with selects feeding the operation
  if (Value *V = SimplifySelectsFeedingBinaryOp(I, LHS, RHS))
    return replaceInstUsesWith(I, V);

  if (I.hasAllowReassoc() && I.hasNoSignedZeros()) {
    if (Instruction *F = factorizeFAddFSub(I, Builder))
      return F;
    if (Value *V = FAddCombine(Builder).simplify(&I))
      return replaceInstUsesWith(I, V);
  }

  return nullptr;
}

/// Optimize pointer differences into the same array into a size.  Consider:
///  &A[10] - &A[0]: we should compile this to "10".  LHS/RHS are the pointer
/// operands to the ptrtoint instructions for the LHS/RHS of the subtract.
Value *InstCombiner::OptimizePointerDifference(Value *LHS, Value *RHS,
                                               Type *Ty, bool IsNUW) {
  // If LHS is a gep based on RHS or RHS is a gep based on LHS, we can optimize
  // this.
  bool Swapped = false;
  GEPOperator *GEP1 = nullptr, *GEP2 = nullptr;

  // For now we require one side to be the base pointer "A" or a constant
  // GEP derived from it.
  if (GEPOperator *LHSGEP = dyn_cast<GEPOperator>(LHS)) {
    // (gep X, ...) - X
    if (LHSGEP->getOperand(0) == RHS) {
      GEP1 = LHSGEP;
      Swapped = false;
    } else if (GEPOperator *RHSGEP = dyn_cast<GEPOperator>(RHS)) {
      // (gep X, ...) - (gep X, ...)
      if (LHSGEP->getOperand(0)->stripPointerCasts() ==
            RHSGEP->getOperand(0)->stripPointerCasts()) {
        GEP2 = RHSGEP;
        GEP1 = LHSGEP;
        Swapped = false;
      }
    }
  }

  if (GEPOperator *RHSGEP = dyn_cast<GEPOperator>(RHS)) {
    // X - (gep X, ...)
    if (RHSGEP->getOperand(0) == LHS) {
      GEP1 = RHSGEP;
      Swapped = true;
    } else if (GEPOperator *LHSGEP = dyn_cast<GEPOperator>(LHS)) {
      // (gep X, ...) - (gep X, ...)
      if (RHSGEP->getOperand(0)->stripPointerCasts() ==
            LHSGEP->getOperand(0)->stripPointerCasts()) {
        GEP2 = LHSGEP;
        GEP1 = RHSGEP;
        Swapped = true;
      }
    }
  }

  if (!GEP1)
    // No GEP found.
    return nullptr;

  if (GEP2) {
    // (gep X, ...) - (gep X, ...)
    //
    // Avoid duplicating the arithmetic if there are more than one non-constant
    // indices between the two GEPs and either GEP has a non-constant index and
    // multiple users. If zero non-constant index, the result is a constant and
    // there is no duplication. If one non-constant index, the result is an add
    // or sub with a constant, which is no larger than the original code, and
    // there's no duplicated arithmetic, even if either GEP has multiple
    // users. If more than one non-constant indices combined, as long as the GEP
    // with at least one non-constant index doesn't have multiple users, there
    // is no duplication.
    unsigned NumNonConstantIndices1 = GEP1->countNonConstantIndices();
    unsigned NumNonConstantIndices2 = GEP2->countNonConstantIndices();
    if (NumNonConstantIndices1 + NumNonConstantIndices2 > 1 &&
        ((NumNonConstantIndices1 > 0 && !GEP1->hasOneUse()) ||
         (NumNonConstantIndices2 > 0 && !GEP2->hasOneUse()))) {
      return nullptr;
    }
  }

  // Emit the offset of the GEP and an intptr_t.
  Value *Result = EmitGEPOffset(GEP1);

  // If this is a single inbounds GEP and the original sub was nuw,
  // then the final multiplication is also nuw. We match an extra add zero
  // here, because that's what EmitGEPOffset() generates.
  Instruction *I;
  if (IsNUW && !GEP2 && !Swapped && GEP1->isInBounds() &&
      match(Result, m_Add(m_Instruction(I), m_Zero())) &&
      I->getOpcode() == Instruction::Mul)
    I->setHasNoUnsignedWrap();

  // If we had a constant expression GEP on the other side offsetting the
  // pointer, subtract it from the offset we have.
  if (GEP2) {
    Value *Offset = EmitGEPOffset(GEP2);
    Result = Builder.CreateSub(Result, Offset);
  }

  // If we have p - gep(p, ...)  then we have to negate the result.
  if (Swapped)
    Result = Builder.CreateNeg(Result, "diff.neg");

  return Builder.CreateIntCast(Result, Ty, true);
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  if (Value *V = SimplifySubInst(I.getOperand(0), I.getOperand(1),
                                 I.hasNoSignedWrap(), I.hasNoUnsignedWrap(),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // If this is a 'B = x-(-A)', change to B = x+A.
  // We deal with this without involving Negator to preserve NSW flag.
  if (Value *V = dyn_castNegVal(Op1)) {
    BinaryOperator *Res = BinaryOperator::CreateAdd(Op0, V);

    if (const auto *BO = dyn_cast<BinaryOperator>(Op1)) {
      assert(BO->getOpcode() == Instruction::Sub &&
             "Expected a subtraction operator!");
      if (BO->hasNoSignedWrap() && I.hasNoSignedWrap())
        Res->setHasNoSignedWrap(true);
    } else {
      if (cast<Constant>(Op1)->isNotMinSignedValue() && I.hasNoSignedWrap())
        Res->setHasNoSignedWrap(true);
    }

    return Res;
  }

  auto TryToNarrowDeduceFlags = [this, &I, &Op0, &Op1]() -> Instruction * {
    if (Instruction *Ext = narrowMathIfNoOverflow(I))
      return Ext;

    bool Changed = false;
    if (!I.hasNoSignedWrap() && willNotOverflowSignedSub(Op0, Op1, I)) {
      Changed = true;
      I.setHasNoSignedWrap(true);
    }
    if (!I.hasNoUnsignedWrap() && willNotOverflowUnsignedSub(Op0, Op1, I)) {
      Changed = true;
      I.setHasNoUnsignedWrap(true);
    }

    return Changed ? &I : nullptr;
  };

  // First, let's try to interpret `sub a, b` as `add a, (sub 0, b)`,
  // and let's try to sink `(sub 0, b)` into `b` itself. But only if this isn't
  // a pure negation used by a select that looks like abs/nabs.
  bool IsNegation = match(Op0, m_ZeroInt());
  if (!IsNegation || none_of(I.users(), [&I, Op1](const User *U) {
        const Instruction *UI = dyn_cast<Instruction>(U);
        if (!UI)
          return false;
        return match(UI,
                     m_Select(m_Value(), m_Specific(Op1), m_Specific(&I))) ||
               match(UI, m_Select(m_Value(), m_Specific(&I), m_Specific(Op1)));
      })) {
    if (Value *NegOp1 = Negator::Negate(IsNegation, Op1, *this))
      return BinaryOperator::CreateAdd(NegOp1, Op0);
  }
  if (IsNegation)
    return TryToNarrowDeduceFlags(); // Should have been handled in Negator!

  // (A*B)-(A*C) -> A*(B-C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  if (I.getType()->isIntOrIntVectorTy(1))
    return BinaryOperator::CreateXor(Op0, Op1);

  // Replace (-1 - A) with (~A).
  if (match(Op0, m_AllOnes()))
    return BinaryOperator::CreateNot(Op1);

  // (~X) - (~Y) --> Y - X
  Value *X, *Y;
  if (match(Op0, m_Not(m_Value(X))) && match(Op1, m_Not(m_Value(Y))))
    return BinaryOperator::CreateSub(Y, X);

  // (X + -1) - Y --> ~Y + X
  if (match(Op0, m_OneUse(m_Add(m_Value(X), m_AllOnes()))))
    return BinaryOperator::CreateAdd(Builder.CreateNot(Op1), X);

  // Reassociate sub/add sequences to create more add instructions and
  // reduce dependency chains:
  // ((X - Y) + Z) - Op1 --> (X + Z) - (Y + Op1)
  Value *Z;
  if (match(Op0, m_OneUse(m_c_Add(m_OneUse(m_Sub(m_Value(X), m_Value(Y))),
                                  m_Value(Z))))) {
    Value *XZ = Builder.CreateAdd(X, Z);
    Value *YW = Builder.CreateAdd(Y, Op1);
    return BinaryOperator::CreateSub(XZ, YW);
  }

  if (Constant *C = dyn_cast<Constant>(Op0)) {
    Value *X;
    if (match(Op1, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
      // C - (zext bool) --> bool ? C - 1 : C
      return SelectInst::Create(X, SubOne(C), C);
    if (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
      // C - (sext bool) --> bool ? C + 1 : C
      return SelectInst::Create(X, AddOne(C), C);

    // C - ~X == X + (1+C)
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::CreateAdd(X, AddOne(C));

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    // Try to fold constant sub into PHI values.
    if (PHINode *PN = dyn_cast<PHINode>(Op1))
      if (Instruction *R = foldOpIntoPhi(I, PN))
        return R;

    Constant *C2;

    // C-(C2-X) --> X+(C-C2)
    if (match(Op1, m_Sub(m_Constant(C2), m_Value(X))) && !isa<ConstantExpr>(C2))
      return BinaryOperator::CreateAdd(X, ConstantExpr::getSub(C, C2));

    // C-(X+C2) --> (C-C2)-X
    if (match(Op1, m_Add(m_Value(X), m_Constant(C2))))
      return BinaryOperator::CreateSub(ConstantExpr::getSub(C, C2), X);
  }

  const APInt *Op0C;
  if (match(Op0, m_APInt(Op0C)) && Op0C->isMask()) {
    // Turn this into a xor if LHS is 2^n-1 and the remaining bits are known
    // zero.
    KnownBits RHSKnown = computeKnownBits(Op1, 0, &I);
    if ((*Op0C | RHSKnown.Zero).isAllOnesValue())
      return BinaryOperator::CreateXor(Op1, Op0);
  }

  {
    Value *Y;
    // X-(X+Y) == -Y    X-(Y+X) == -Y
    if (match(Op1, m_c_Add(m_Specific(Op0), m_Value(Y))))
      return BinaryOperator::CreateNeg(Y);

    // (X-Y)-X == -Y
    if (match(Op0, m_Sub(m_Specific(Op1), m_Value(Y))))
      return BinaryOperator::CreateNeg(Y);
  }

  // (sub (or A, B) (and A, B)) --> (xor A, B)
  {
    Value *A, *B;
    if (match(Op1, m_And(m_Value(A), m_Value(B))) &&
        match(Op0, m_c_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateXor(A, B);
  }

  // (sub (and A, B) (or A, B)) --> neg (xor A, B)
  {
    Value *A, *B;
    if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
        match(Op1, m_c_Or(m_Specific(A), m_Specific(B))) &&
        (Op0->hasOneUse() || Op1->hasOneUse()))
      return BinaryOperator::CreateNeg(Builder.CreateXor(A, B));
  }

  // (sub (or A, B), (xor A, B)) --> (and A, B)
  {
    Value *A, *B;
    if (match(Op1, m_Xor(m_Value(A), m_Value(B))) &&
        match(Op0, m_c_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);
  }

  // (sub (xor A, B) (or A, B)) --> neg (and A, B)
  {
    Value *A, *B;
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))) &&
        match(Op1, m_c_Or(m_Specific(A), m_Specific(B))) &&
        (Op0->hasOneUse() || Op1->hasOneUse()))
      return BinaryOperator::CreateNeg(Builder.CreateAnd(A, B));
  }

  {
    Value *Y;
    // ((X | Y) - X) --> (~X & Y)
    if (match(Op0, m_OneUse(m_c_Or(m_Value(Y), m_Specific(Op1)))))
      return BinaryOperator::CreateAnd(
          Y, Builder.CreateNot(Op1, Op1->getName() + ".not"));
  }

  {
    // (sub (and Op1, (neg X)), Op1) --> neg (and Op1, (add X, -1))
    Value *X;
    if (match(Op0, m_OneUse(m_c_And(m_Specific(Op1),
                                    m_OneUse(m_Neg(m_Value(X))))))) {
      return BinaryOperator::CreateNeg(Builder.CreateAnd(
          Op1, Builder.CreateAdd(X, Constant::getAllOnesValue(I.getType()))));
    }
  }

  {
    // (sub (and Op1, C), Op1) --> neg (and Op1, ~C)
    Constant *C;
    if (match(Op0, m_OneUse(m_And(m_Specific(Op1), m_Constant(C))))) {
      return BinaryOperator::CreateNeg(
          Builder.CreateAnd(Op1, Builder.CreateNot(C)));
    }
  }

  {
    // If we have a subtraction between some value and a select between
    // said value and something else, sink subtraction into select hands, i.e.:
    //   sub (select %Cond, %TrueVal, %FalseVal), %Op1
    //     ->
    //   select %Cond, (sub %TrueVal, %Op1), (sub %FalseVal, %Op1)
    //  or
    //   sub %Op0, (select %Cond, %TrueVal, %FalseVal)
    //     ->
    //   select %Cond, (sub %Op0, %TrueVal), (sub %Op0, %FalseVal)
    // This will result in select between new subtraction and 0.
    auto SinkSubIntoSelect =
        [Ty = I.getType()](Value *Select, Value *OtherHandOfSub,
                           auto SubBuilder) -> Instruction * {
      Value *Cond, *TrueVal, *FalseVal;
      if (!match(Select, m_OneUse(m_Select(m_Value(Cond), m_Value(TrueVal),
                                           m_Value(FalseVal)))))
        return nullptr;
      if (OtherHandOfSub != TrueVal && OtherHandOfSub != FalseVal)
        return nullptr;
      // While it is really tempting to just create two subtractions and let
      // InstCombine fold one of those to 0, it isn't possible to do so
      // because of worklist visitation order. So ugly it is.
      bool OtherHandOfSubIsTrueVal = OtherHandOfSub == TrueVal;
      Value *NewSub = SubBuilder(OtherHandOfSubIsTrueVal ? FalseVal : TrueVal);
      Constant *Zero = Constant::getNullValue(Ty);
      SelectInst *NewSel =
          SelectInst::Create(Cond, OtherHandOfSubIsTrueVal ? Zero : NewSub,
                             OtherHandOfSubIsTrueVal ? NewSub : Zero);
      // Preserve prof metadata if any.
      NewSel->copyMetadata(cast<Instruction>(*Select));
      return NewSel;
    };
    if (Instruction *NewSel = SinkSubIntoSelect(
            /*Select=*/Op0, /*OtherHandOfSub=*/Op1,
            [Builder = &Builder, Op1](Value *OtherHandOfSelect) {
              return Builder->CreateSub(OtherHandOfSelect,
                                        /*OtherHandOfSub=*/Op1);
            }))
      return NewSel;
    if (Instruction *NewSel = SinkSubIntoSelect(
            /*Select=*/Op1, /*OtherHandOfSub=*/Op0,
            [Builder = &Builder, Op0](Value *OtherHandOfSelect) {
              return Builder->CreateSub(/*OtherHandOfSub=*/Op0,
                                        OtherHandOfSelect);
            }))
      return NewSel;
  }

  // (X - (X & Y))   -->   (X & ~Y)
  if (match(Op1, m_c_And(m_Specific(Op0), m_Value(Y))) &&
      (Op1->hasOneUse() || isa<Constant>(Y)))
    return BinaryOperator::CreateAnd(
        Op0, Builder.CreateNot(Y, Y->getName() + ".not"));

  {
    // ~A - Min/Max(~A, O) -> Max/Min(A, ~O) - A
    // ~A - Min/Max(O, ~A) -> Max/Min(A, ~O) - A
    // Min/Max(~A, O) - ~A -> A - Max/Min(A, ~O)
    // Min/Max(O, ~A) - ~A -> A - Max/Min(A, ~O)
    // So long as O here is freely invertible, this will be neutral or a win.
    Value *LHS, *RHS, *A;
    Value *NotA = Op0, *MinMax = Op1;
    SelectPatternFlavor SPF = matchSelectPattern(MinMax, LHS, RHS).Flavor;
    if (!SelectPatternResult::isMinOrMax(SPF)) {
      NotA = Op1;
      MinMax = Op0;
      SPF = matchSelectPattern(MinMax, LHS, RHS).Flavor;
    }
    if (SelectPatternResult::isMinOrMax(SPF) &&
        match(NotA, m_Not(m_Value(A))) && (NotA == LHS || NotA == RHS)) {
      if (NotA == LHS)
        std::swap(LHS, RHS);
      // LHS is now O above and expected to have at least 2 uses (the min/max)
      // NotA is epected to have 2 uses from the min/max and 1 from the sub.
      if (isFreeToInvert(LHS, !LHS->hasNUsesOrMore(3)) &&
          !NotA->hasNUsesOrMore(4)) {
        // Note: We don't generate the inverse max/min, just create the not of
        // it and let other folds do the rest.
        Value *Not = Builder.CreateNot(MinMax);
        if (NotA == Op0)
          return BinaryOperator::CreateSub(Not, A);
        else
          return BinaryOperator::CreateSub(A, Not);
      }
    }
  }

  // Optimize pointer differences into the same array into a size.  Consider:
  //  &A[10] - &A[0]: we should compile this to "10".
  Value *LHSOp, *RHSOp;
  if (match(Op0, m_PtrToInt(m_Value(LHSOp))) &&
      match(Op1, m_PtrToInt(m_Value(RHSOp))))
    if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType(),
                                               I.hasNoUnsignedWrap()))
      return replaceInstUsesWith(I, Res);

  // trunc(p)-trunc(q) -> trunc(p-q)
  if (match(Op0, m_Trunc(m_PtrToInt(m_Value(LHSOp)))) &&
      match(Op1, m_Trunc(m_PtrToInt(m_Value(RHSOp)))))
    if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType(),
                                               /* IsNUW */ false))
      return replaceInstUsesWith(I, Res);

  // Canonicalize a shifty way to code absolute value to the common pattern.
  // There are 2 potential commuted variants.
  // We're relying on the fact that we only do this transform when the shift has
  // exactly 2 uses and the xor has exactly 1 use (otherwise, we might increase
  // instructions).
  Value *A;
  const APInt *ShAmt;
  Type *Ty = I.getType();
  if (match(Op1, m_AShr(m_Value(A), m_APInt(ShAmt))) &&
      Op1->hasNUses(2) && *ShAmt == Ty->getScalarSizeInBits() - 1 &&
      match(Op0, m_OneUse(m_c_Xor(m_Specific(A), m_Specific(Op1))))) {
    // B = ashr i32 A, 31 ; smear the sign bit
    // sub (xor A, B), B  ; flip bits if negative and subtract -1 (add 1)
    // --> (A < 0) ? -A : A
    Value *Cmp = Builder.CreateICmpSLT(A, ConstantInt::getNullValue(Ty));
    // Copy the nuw/nsw flags from the sub to the negate.
    Value *Neg = Builder.CreateNeg(A, "", I.hasNoUnsignedWrap(),
                                   I.hasNoSignedWrap());
    return SelectInst::Create(Cmp, Neg, A);
  }

  if (Instruction *V =
          canonicalizeCondSignextOfHighBitExtractToSignextHighBitExtract(I))
    return V;

  return TryToNarrowDeduceFlags();
}

/// This eliminates floating-point negation in either 'fneg(X)' or
/// 'fsub(-0.0, X)' form by combining into a constant operand.
static Instruction *foldFNegIntoConstant(Instruction &I) {
  Value *X;
  Constant *C;

  // Fold negation into constant operand. This is limited with one-use because
  // fneg is assumed better for analysis and cheaper in codegen than fmul/fdiv.
  // -(X * C) --> X * (-C)
  // FIXME: It's arguable whether these should be m_OneUse or not. The current
  // belief is that the FNeg allows for better reassociation opportunities.
  if (match(&I, m_FNeg(m_OneUse(m_FMul(m_Value(X), m_Constant(C))))))
    return BinaryOperator::CreateFMulFMF(X, ConstantExpr::getFNeg(C), &I);
  // -(X / C) --> X / (-C)
  if (match(&I, m_FNeg(m_OneUse(m_FDiv(m_Value(X), m_Constant(C))))))
    return BinaryOperator::CreateFDivFMF(X, ConstantExpr::getFNeg(C), &I);
  // -(C / X) --> (-C) / X
  if (match(&I, m_FNeg(m_OneUse(m_FDiv(m_Constant(C), m_Value(X))))))
    return BinaryOperator::CreateFDivFMF(ConstantExpr::getFNeg(C), X, &I);

  // With NSZ [ counter-example with -0.0: -(-0.0 + 0.0) != 0.0 + -0.0 ]:
  // -(X + C) --> -X + -C --> -C - X
  if (I.hasNoSignedZeros() &&
      match(&I, m_FNeg(m_OneUse(m_FAdd(m_Value(X), m_Constant(C))))))
    return BinaryOperator::CreateFSubFMF(ConstantExpr::getFNeg(C), X, &I);

  return nullptr;
}

static Instruction *hoistFNegAboveFMulFDiv(Instruction &I,
                                           InstCombiner::BuilderTy &Builder) {
  Value *FNeg;
  if (!match(&I, m_FNeg(m_Value(FNeg))))
    return nullptr;

  Value *X, *Y;
  if (match(FNeg, m_OneUse(m_FMul(m_Value(X), m_Value(Y)))))
    return BinaryOperator::CreateFMulFMF(Builder.CreateFNegFMF(X, &I), Y, &I);

  if (match(FNeg, m_OneUse(m_FDiv(m_Value(X), m_Value(Y)))))
    return BinaryOperator::CreateFDivFMF(Builder.CreateFNegFMF(X, &I), Y, &I);

  return nullptr;
}

Instruction *InstCombiner::visitFNeg(UnaryOperator &I) {
  Value *Op = I.getOperand(0);

  if (Value *V = SimplifyFNegInst(Op, I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldFNegIntoConstant(I))
    return X;

  Value *X, *Y;

  // If we can ignore the sign of zeros: -(X - Y) --> (Y - X)
  if (I.hasNoSignedZeros() &&
      match(Op, m_OneUse(m_FSub(m_Value(X), m_Value(Y)))))
    return BinaryOperator::CreateFSubFMF(Y, X, &I);

  if (Instruction *R = hoistFNegAboveFMulFDiv(I, Builder))
    return R;

  return nullptr;
}

Instruction *InstCombiner::visitFSub(BinaryOperator &I) {
  if (Value *V = SimplifyFSubInst(I.getOperand(0), I.getOperand(1),
                                  I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  // Subtraction from -0.0 is the canonical form of fneg.
  // fsub -0.0, X ==> fneg X
  // fsub nsz 0.0, X ==> fneg nsz X
  //
  // FIXME This matcher does not respect FTZ or DAZ yet:
  // fsub -0.0, Denorm ==> +-0
  // fneg Denorm ==> -Denorm
  Value *Op;
  if (match(&I, m_FNeg(m_Value(Op))))
    return UnaryOperator::CreateFNegFMF(Op, &I);

  if (Instruction *X = foldFNegIntoConstant(I))
    return X;

  if (Instruction *R = hoistFNegAboveFMulFDiv(I, Builder))
    return R;

  Value *X, *Y;
  Constant *C;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  // If Op0 is not -0.0 or we can ignore -0.0: Z - (X - Y) --> Z + (Y - X)
  // Canonicalize to fadd to make analysis easier.
  // This can also help codegen because fadd is commutative.
  // Note that if this fsub was really an fneg, the fadd with -0.0 will get
  // killed later. We still limit that particular transform with 'hasOneUse'
  // because an fneg is assumed better/cheaper than a generic fsub.
  if (I.hasNoSignedZeros() || CannotBeNegativeZero(Op0, SQ.TLI)) {
    if (match(Op1, m_OneUse(m_FSub(m_Value(X), m_Value(Y))))) {
      Value *NewSub = Builder.CreateFSubFMF(Y, X, &I);
      return BinaryOperator::CreateFAddFMF(Op0, NewSub, &I);
    }
  }

  // (-X) - Op1 --> -(X + Op1)
  if (I.hasNoSignedZeros() && !isa<ConstantExpr>(Op0) &&
      match(Op0, m_OneUse(m_FNeg(m_Value(X))))) {
    Value *FAdd = Builder.CreateFAddFMF(X, Op1, &I);
    return UnaryOperator::CreateFNegFMF(FAdd, &I);
  }

  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *NV = FoldOpIntoSelect(I, SI))
        return NV;

  // X - C --> X + (-C)
  // But don't transform constant expressions because there's an inverse fold
  // for X + (-Y) --> X - Y.
  if (match(Op1, m_Constant(C)) && !isa<ConstantExpr>(Op1))
    return BinaryOperator::CreateFAddFMF(Op0, ConstantExpr::getFNeg(C), &I);

  // X - (-Y) --> X + Y
  if (match(Op1, m_FNeg(m_Value(Y))))
    return BinaryOperator::CreateFAddFMF(Op0, Y, &I);

  // Similar to above, but look through a cast of the negated value:
  // X - (fptrunc(-Y)) --> X + fptrunc(Y)
  Type *Ty = I.getType();
  if (match(Op1, m_OneUse(m_FPTrunc(m_FNeg(m_Value(Y))))))
    return BinaryOperator::CreateFAddFMF(Op0, Builder.CreateFPTrunc(Y, Ty), &I);

  // X - (fpext(-Y)) --> X + fpext(Y)
  if (match(Op1, m_OneUse(m_FPExt(m_FNeg(m_Value(Y))))))
    return BinaryOperator::CreateFAddFMF(Op0, Builder.CreateFPExt(Y, Ty), &I);

  // Similar to above, but look through fmul/fdiv of the negated value:
  // Op0 - (-X * Y) --> Op0 + (X * Y)
  // Op0 - (Y * -X) --> Op0 + (X * Y)
  if (match(Op1, m_OneUse(m_c_FMul(m_FNeg(m_Value(X)), m_Value(Y))))) {
    Value *FMul = Builder.CreateFMulFMF(X, Y, &I);
    return BinaryOperator::CreateFAddFMF(Op0, FMul, &I);
  }
  // Op0 - (-X / Y) --> Op0 + (X / Y)
  // Op0 - (X / -Y) --> Op0 + (X / Y)
  if (match(Op1, m_OneUse(m_FDiv(m_FNeg(m_Value(X)), m_Value(Y)))) ||
      match(Op1, m_OneUse(m_FDiv(m_Value(X), m_FNeg(m_Value(Y)))))) {
    Value *FDiv = Builder.CreateFDivFMF(X, Y, &I);
    return BinaryOperator::CreateFAddFMF(Op0, FDiv, &I);
  }

  // Handle special cases for FSub with selects feeding the operation
  if (Value *V = SimplifySelectsFeedingBinaryOp(I, Op0, Op1))
    return replaceInstUsesWith(I, V);

  if (I.hasAllowReassoc() && I.hasNoSignedZeros()) {
    // (Y - X) - Y --> -X
    if (match(Op0, m_FSub(m_Specific(Op1), m_Value(X))))
      return UnaryOperator::CreateFNegFMF(X, &I);

    // Y - (X + Y) --> -X
    // Y - (Y + X) --> -X
    if (match(Op1, m_c_FAdd(m_Specific(Op0), m_Value(X))))
      return UnaryOperator::CreateFNegFMF(X, &I);

    // (X * C) - X --> X * (C - 1.0)
    if (match(Op0, m_FMul(m_Specific(Op1), m_Constant(C)))) {
      Constant *CSubOne = ConstantExpr::getFSub(C, ConstantFP::get(Ty, 1.0));
      return BinaryOperator::CreateFMulFMF(Op1, CSubOne, &I);
    }
    // X - (X * C) --> X * (1.0 - C)
    if (match(Op1, m_FMul(m_Specific(Op0), m_Constant(C)))) {
      Constant *OneSubC = ConstantExpr::getFSub(ConstantFP::get(Ty, 1.0), C);
      return BinaryOperator::CreateFMulFMF(Op0, OneSubC, &I);
    }

    // Reassociate fsub/fadd sequences to create more fadd instructions and
    // reduce dependency chains:
    // ((X - Y) + Z) - Op1 --> (X + Z) - (Y + Op1)
    Value *Z;
    if (match(Op0, m_OneUse(m_c_FAdd(m_OneUse(m_FSub(m_Value(X), m_Value(Y))),
                                     m_Value(Z))))) {
      Value *XZ = Builder.CreateFAddFMF(X, Z, &I);
      Value *YW = Builder.CreateFAddFMF(Y, Op1, &I);
      return BinaryOperator::CreateFSubFMF(XZ, YW, &I);
    }

    auto m_FaddRdx = [](Value *&Sum, Value *&Vec) {
      return m_OneUse(
          m_Intrinsic<Intrinsic::experimental_vector_reduce_v2_fadd>(
              m_Value(Sum), m_Value(Vec)));
    };
    Value *A0, *A1, *V0, *V1;
    if (match(Op0, m_FaddRdx(A0, V0)) && match(Op1, m_FaddRdx(A1, V1)) &&
        V0->getType() == V1->getType()) {
      // Difference of sums is sum of differences:
      // add_rdx(A0, V0) - add_rdx(A1, V1) --> add_rdx(A0, V0 - V1) - A1
      Value *Sub = Builder.CreateFSubFMF(V0, V1, &I);
      Value *Rdx = Builder.CreateIntrinsic(
          Intrinsic::experimental_vector_reduce_v2_fadd,
          {A0->getType(), Sub->getType()}, {A0, Sub}, &I);
      return BinaryOperator::CreateFSubFMF(Rdx, A1, &I);
    }

    if (Instruction *F = factorizeFAddFSub(I, Builder))
      return F;

    // TODO: This performs reassociative folds for FP ops. Some fraction of the
    // functionality has been subsumed by simple pattern matching here and in
    // InstSimplify. We should let a dedicated reassociation pass handle more
    // complex pattern matching and remove this from InstCombine.
    if (Value *V = FAddCombine(Builder).simplify(&I))
      return replaceInstUsesWith(I, V);

    // (X - Y) - Op1 --> X - (Y + Op1)
    if (match(Op0, m_OneUse(m_FSub(m_Value(X), m_Value(Y))))) {
      Value *FAdd = Builder.CreateFAddFMF(Y, Op1, &I);
      return BinaryOperator::CreateFSubFMF(X, FAdd, &I);
    }
  }

  return nullptr;
}
