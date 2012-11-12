//===-- llvm/Support/PatternMatch.h - Match on the LLVM IR ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the LLVM IR.  The power of these routines is
// that it allows you to write concise patterns that are expressive and easy to
// understand.  The other major advantage of this is that it allows you to
// trivially capture/bind elements in the pattern to variables.  For example,
// you can do something like this:
//
//  Value *Exp = ...
//  Value *X, *Y;  ConstantInt *C1, *C2;      // (X & C1) | (Y & C2)
//  if (match(Exp, m_Or(m_And(m_Value(X), m_ConstantInt(C1)),
//                      m_And(m_Value(Y), m_ConstantInt(C2))))) {
//    ... Pattern is matched and variables are bound ...
//  }
//
// This is primarily useful to things like the instruction combiner, but can
// also be useful for static analysis tools or code generators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PATTERNMATCH_H
#define LLVM_SUPPORT_PATTERNMATCH_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"

namespace llvm {
namespace PatternMatch {

template<typename Val, typename Pattern>
bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern&>(P).match(V);
}

  
template<typename SubPattern_t>
struct OneUse_match {
  SubPattern_t SubPattern;
  
  OneUse_match(const SubPattern_t &SP) : SubPattern(SP) {}
  
  template<typename OpTy>
  bool match(OpTy *V) {
    return V->hasOneUse() && SubPattern.match(V);
  }
};

template<typename T>
inline OneUse_match<T> m_OneUse(const T &SubPattern) { return SubPattern; }
  
  
template<typename Class>
struct class_match {
  template<typename ITy>
  bool match(ITy *V) { return isa<Class>(V); }
};

/// m_Value() - Match an arbitrary value and ignore it.
inline class_match<Value> m_Value() { return class_match<Value>(); }
/// m_ConstantInt() - Match an arbitrary ConstantInt and ignore it.
inline class_match<ConstantInt> m_ConstantInt() {
  return class_match<ConstantInt>();
}
/// m_Undef() - Match an arbitrary undef constant.
inline class_match<UndefValue> m_Undef() { return class_match<UndefValue>(); }

inline class_match<Constant> m_Constant() { return class_match<Constant>(); }
  
struct match_zero {
  template<typename ITy>
  bool match(ITy *V) {
    if (const Constant *C = dyn_cast<Constant>(V))
      return C->isNullValue();
    return false;
  }
};
  
/// m_Zero() - Match an arbitrary zero/null constant.  This includes
/// zero_initializer for vectors and ConstantPointerNull for pointers.
inline match_zero m_Zero() { return match_zero(); }
  
  
struct apint_match {
  const APInt *&Res;
  apint_match(const APInt *&R) : Res(R) {}
  template<typename ITy>
  bool match(ITy *V) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      Res = &CI->getValue();
      return true;
    }
    // FIXME: Remove this.
    if (ConstantVector *CV = dyn_cast<ConstantVector>(V))
      if (ConstantInt *CI =
          dyn_cast_or_null<ConstantInt>(CV->getSplatValue())) {
        Res = &CI->getValue();
        return true;
      }
    if (ConstantDataVector *CV = dyn_cast<ConstantDataVector>(V))
      if (ConstantInt *CI =
          dyn_cast_or_null<ConstantInt>(CV->getSplatValue())) {
        Res = &CI->getValue();
        return true;
      }
    return false;
  }
};
  
/// m_APInt - Match a ConstantInt or splatted ConstantVector, binding the
/// specified pointer to the contained APInt.
inline apint_match m_APInt(const APInt *&Res) { return Res; }

  
template<int64_t Val>
struct constantint_match {
  template<typename ITy>
  bool match(ITy *V) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      const APInt &CIV = CI->getValue();
      if (Val >= 0)
        return CIV == static_cast<uint64_t>(Val);
      // If Val is negative, and CI is shorter than it, truncate to the right
      // number of bits.  If it is larger, then we have to sign extend.  Just
      // compare their negated values.
      return -CIV == -Val;
    }
    return false;
  }
};

/// m_ConstantInt<int64_t> - Match a ConstantInt with a specific value.
template<int64_t Val>
inline constantint_match<Val> m_ConstantInt() {
  return constantint_match<Val>();
}

/// cst_pred_ty - This helper class is used to match scalar and vector constants
/// that satisfy a specified predicate.
template<typename Predicate>
struct cst_pred_ty : public Predicate {
  template<typename ITy>
  bool match(ITy *V) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return this->isValue(CI->getValue());
    // FIXME: Remove this.
    if (const ConstantVector *CV = dyn_cast<ConstantVector>(V))
      if (ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue()))
        return this->isValue(CI->getValue());
    if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(V))
      if (ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue()))
        return this->isValue(CI->getValue());
    return false;
  }
};
  
/// api_pred_ty - This helper class is used to match scalar and vector constants
/// that satisfy a specified predicate, and bind them to an APInt.
template<typename Predicate>
struct api_pred_ty : public Predicate {
  const APInt *&Res;
  api_pred_ty(const APInt *&R) : Res(R) {}
  template<typename ITy>
  bool match(ITy *V) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V))
      if (this->isValue(CI->getValue())) {
        Res = &CI->getValue();
        return true;
      }
    
    // FIXME: remove.
    if (const ConstantVector *CV = dyn_cast<ConstantVector>(V))
      if (ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue()))
        if (this->isValue(CI->getValue())) {
          Res = &CI->getValue();
          return true;
        }
    
    if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(V))
      if (ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue()))
        if (this->isValue(CI->getValue())) {
          Res = &CI->getValue();
          return true;
        }

    return false;
  }
};
  
  
struct is_one {
  bool isValue(const APInt &C) { return C == 1; }
};

/// m_One() - Match an integer 1 or a vector with all elements equal to 1.
inline cst_pred_ty<is_one> m_One() { return cst_pred_ty<is_one>(); }
inline api_pred_ty<is_one> m_One(const APInt *&V) { return V; }
    
struct is_all_ones {
  bool isValue(const APInt &C) { return C.isAllOnesValue(); }
};
  
/// m_AllOnes() - Match an integer or vector with all bits set to true.
inline cst_pred_ty<is_all_ones> m_AllOnes() {return cst_pred_ty<is_all_ones>();}
inline api_pred_ty<is_all_ones> m_AllOnes(const APInt *&V) { return V; }

struct is_sign_bit {
  bool isValue(const APInt &C) { return C.isSignBit(); }
};

/// m_SignBit() - Match an integer or vector with only the sign bit(s) set.
inline cst_pred_ty<is_sign_bit> m_SignBit() {return cst_pred_ty<is_sign_bit>();}
inline api_pred_ty<is_sign_bit> m_SignBit(const APInt *&V) { return V; }

struct is_power2 {
  bool isValue(const APInt &C) { return C.isPowerOf2(); }
};

/// m_Power2() - Match an integer or vector power of 2.
inline cst_pred_ty<is_power2> m_Power2() { return cst_pred_ty<is_power2>(); }
inline api_pred_ty<is_power2> m_Power2(const APInt *&V) { return V; }

template<typename Class>
struct bind_ty {
  Class *&VR;
  bind_ty(Class *&V) : VR(V) {}

  template<typename ITy>
  bool match(ITy *V) {
    if (Class *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// m_Value - Match a value, capturing it if we match.
inline bind_ty<Value> m_Value(Value *&V) { return V; }

/// m_ConstantInt - Match a ConstantInt, capturing the value if we match.
inline bind_ty<ConstantInt> m_ConstantInt(ConstantInt *&CI) { return CI; }

/// m_Constant - Match a Constant, capturing the value if we match.
inline bind_ty<Constant> m_Constant(Constant *&C) { return C; }

/// specificval_ty - Match a specified Value*.
struct specificval_ty {
  const Value *Val;
  specificval_ty(const Value *V) : Val(V) {}

  template<typename ITy>
  bool match(ITy *V) {
    return V == Val;
  }
};

/// m_Specific - Match if we have a specific specified value.
inline specificval_ty m_Specific(const Value *V) { return V; }

struct bind_const_intval_ty {
  uint64_t &VR;
  bind_const_intval_ty(uint64_t &V) : VR(V) {}
  
  template<typename ITy>
  bool match(ITy *V) {
    if (ConstantInt *CV = dyn_cast<ConstantInt>(V))
      if (CV->getBitWidth() <= 64) {
        VR = CV->getZExtValue();
        return true;
      }
    return false;
  }
};

/// m_ConstantInt - Match a ConstantInt and bind to its value.  This does not
/// match ConstantInts wider than 64-bits.
inline bind_const_intval_ty m_ConstantInt(uint64_t &V) { return V; }
  
//===----------------------------------------------------------------------===//
// Matchers for specific binary operators.
//

template<typename LHS_t, typename RHS_t, unsigned Opcode>
struct BinaryOp_match {
  LHS_t L;
  RHS_t R;

  BinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      BinaryOperator *I = cast<BinaryOperator>(V);
      return L.match(I->getOperand(0)) && R.match(I->getOperand(1));
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add>
m_Add(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FAdd>
m_FAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FAdd>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Sub>
m_Sub(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Sub>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FSub>
m_FSub(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FSub>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul>
m_Mul(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FMul>
m_FMul(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FMul>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::UDiv>
m_UDiv(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::UDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SDiv>
m_SDiv(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FDiv>
m_FDiv(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::URem>
m_URem(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::URem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SRem>
m_SRem(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SRem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FRem>
m_FRem(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FRem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And>
m_And(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Or>
m_Or(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor>
m_Xor(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shl>
m_Shl(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shl>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::LShr>
m_LShr(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::LShr>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::AShr>
m_AShr(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::AShr>(L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches two different binary ops.
//
template<typename LHS_t, typename RHS_t, unsigned Opc1, unsigned Opc2>
struct BinOp2_match {
  LHS_t L;
  RHS_t R;

  BinOp2_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opc1 ||
        V->getValueID() == Value::InstructionVal + Opc2) {
      BinaryOperator *I = cast<BinaryOperator>(V);
      return L.match(I->getOperand(0)) && R.match(I->getOperand(1));
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return (CE->getOpcode() == Opc1 || CE->getOpcode() == Opc2) &&
             L.match(CE->getOperand(0)) && R.match(CE->getOperand(1));
    return false;
  }
};

/// m_Shr - Matches LShr or AShr.
template<typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::AShr>
m_Shr(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::AShr>(L, R);
}

/// m_LogicalShift - Matches LShr or Shl.
template<typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::Shl>
m_LogicalShift(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::Shl>(L, R);
}

/// m_IDiv - Matches UDiv and SDiv.
template<typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::SDiv, Instruction::UDiv>
m_IDiv(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::SDiv, Instruction::UDiv>(L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches exact binary ops.
//
template<typename SubPattern_t>
struct Exact_match {
  SubPattern_t SubPattern;

  Exact_match(const SubPattern_t &SP) : SubPattern(SP) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (PossiblyExactOperator *PEO = dyn_cast<PossiblyExactOperator>(V))
      return PEO->isExact() && SubPattern.match(V);
    return false;
  }
};

template<typename T>
inline Exact_match<T> m_Exact(const T &SubPattern) { return SubPattern; }

//===----------------------------------------------------------------------===//
// Matchers for CmpInst classes
//

template<typename LHS_t, typename RHS_t, typename Class, typename PredicateTy>
struct CmpClass_match {
  PredicateTy &Predicate;
  LHS_t L;
  RHS_t R;

  CmpClass_match(PredicateTy &Pred, const LHS_t &LHS, const RHS_t &RHS)
    : Predicate(Pred), L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Class *I = dyn_cast<Class>(V))
      if (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) {
        Predicate = I->getPredicate();
        return true;
      }
    return false;
  }
};

template<typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate>
m_ICmp(ICmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS,
                        ICmpInst, ICmpInst::Predicate>(Pred, L, R);
}

template<typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, FCmpInst, FCmpInst::Predicate>
m_FCmp(FCmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS,
                        FCmpInst, FCmpInst::Predicate>(Pred, L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for SelectInst classes
//

template<typename Cond_t, typename LHS_t, typename RHS_t>
struct SelectClass_match {
  Cond_t C;
  LHS_t L;
  RHS_t R;

  SelectClass_match(const Cond_t &Cond, const LHS_t &LHS,
                    const RHS_t &RHS)
    : C(Cond), L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (SelectInst *I = dyn_cast<SelectInst>(V))
      return C.match(I->getOperand(0)) &&
             L.match(I->getOperand(1)) &&
             R.match(I->getOperand(2));
    return false;
  }
};

template<typename Cond, typename LHS, typename RHS>
inline SelectClass_match<Cond, LHS, RHS>
m_Select(const Cond &C, const LHS &L, const RHS &R) {
  return SelectClass_match<Cond, LHS, RHS>(C, L, R);
}

/// m_SelectCst - This matches a select of two constants, e.g.:
///    m_SelectCst<-1, 0>(m_Value(V))
template<int64_t L, int64_t R, typename Cond>
inline SelectClass_match<Cond, constantint_match<L>, constantint_match<R> >
m_SelectCst(const Cond &C) {
  return m_Select(C, m_ConstantInt<L>(), m_ConstantInt<R>());
}


//===----------------------------------------------------------------------===//
// Matchers for CastInst classes
//

template<typename Op_t, unsigned Opcode>
struct CastClass_match {
  Op_t Op;

  CastClass_match(const Op_t &OpMatch) : Op(OpMatch) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Operator *O = dyn_cast<Operator>(V))
      return O->getOpcode() == Opcode && Op.match(O->getOperand(0));
    return false;
  }
};

/// m_BitCast
template<typename OpTy>
inline CastClass_match<OpTy, Instruction::BitCast>
m_BitCast(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::BitCast>(Op);
}
  
/// m_PtrToInt
template<typename OpTy>
inline CastClass_match<OpTy, Instruction::PtrToInt>
m_PtrToInt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::PtrToInt>(Op);
}

/// m_Trunc
template<typename OpTy>
inline CastClass_match<OpTy, Instruction::Trunc>
m_Trunc(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::Trunc>(Op);
}

/// m_SExt
template<typename OpTy>
inline CastClass_match<OpTy, Instruction::SExt>
m_SExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::SExt>(Op);
}

/// m_ZExt
template<typename OpTy>
inline CastClass_match<OpTy, Instruction::ZExt>
m_ZExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::ZExt>(Op);
}
  

//===----------------------------------------------------------------------===//
// Matchers for unary operators
//

template<typename LHS_t>
struct not_match {
  LHS_t L;

  not_match(const LHS_t &LHS) : L(LHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Operator *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::Xor)
        return matchIfNot(O->getOperand(0), O->getOperand(1));
    return false;
  }
private:
  bool matchIfNot(Value *LHS, Value *RHS) {
    return (isa<ConstantInt>(RHS) || isa<ConstantDataVector>(RHS) ||
            // FIXME: Remove CV.
            isa<ConstantVector>(RHS)) &&
           cast<Constant>(RHS)->isAllOnesValue() &&
           L.match(LHS);
  }
};

template<typename LHS>
inline not_match<LHS> m_Not(const LHS &L) { return L; }


template<typename LHS_t>
struct neg_match {
  LHS_t L;

  neg_match(const LHS_t &LHS) : L(LHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Operator *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::Sub)
        return matchIfNeg(O->getOperand(0), O->getOperand(1));
    return false;
  }
private:
  bool matchIfNeg(Value *LHS, Value *RHS) {
    return ((isa<ConstantInt>(LHS) && cast<ConstantInt>(LHS)->isZero()) ||
            isa<ConstantAggregateZero>(LHS)) &&
           L.match(RHS);
  }
};

/// m_Neg - Match an integer negate.
template<typename LHS>
inline neg_match<LHS> m_Neg(const LHS &L) { return L; }


template<typename LHS_t>
struct fneg_match {
  LHS_t L;

  fneg_match(const LHS_t &LHS) : L(LHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Operator *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::FSub)
        return matchIfFNeg(O->getOperand(0), O->getOperand(1));
    return false;
  }
private:
  bool matchIfFNeg(Value *LHS, Value *RHS) {
    if (ConstantFP *C = dyn_cast<ConstantFP>(LHS))
      return C->isNegativeZeroValue() && L.match(RHS);
    return false;
  }
};

/// m_FNeg - Match a floating point negate.
template<typename LHS>
inline fneg_match<LHS> m_FNeg(const LHS &L) { return L; }


//===----------------------------------------------------------------------===//
// Matchers for control flow.
//

template<typename Cond_t>
struct brc_match {
  Cond_t Cond;
  BasicBlock *&T, *&F;
  brc_match(const Cond_t &C, BasicBlock *&t, BasicBlock *&f)
    : Cond(C), T(t), F(f) {
  }

  template<typename OpTy>
  bool match(OpTy *V) {
    if (BranchInst *BI = dyn_cast<BranchInst>(V))
      if (BI->isConditional() && Cond.match(BI->getCondition())) {
        T = BI->getSuccessor(0);
        F = BI->getSuccessor(1);
        return true;
      }
    return false;
  }
};

template<typename Cond_t>
inline brc_match<Cond_t> m_Br(const Cond_t &C, BasicBlock *&T, BasicBlock *&F) {
  return brc_match<Cond_t>(C, T, F);
}


//===----------------------------------------------------------------------===//
// Matchers for max/min idioms, eg: "select (sgt x, y), x, y" -> smax(x,y).
//

template<typename LHS_t, typename RHS_t, typename Pred_t>
struct MaxMin_match {
  LHS_t L;
  RHS_t R;

  MaxMin_match(const LHS_t &LHS, const RHS_t &RHS)
    : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    // Look for "(x pred y) ? x : y" or "(x pred y) ? y : x".
    SelectInst *SI = dyn_cast<SelectInst>(V);
    if (!SI)
      return false;
    ICmpInst *Cmp = dyn_cast<ICmpInst>(SI->getCondition());
    if (!Cmp)
      return false;
    // At this point we have a select conditioned on a comparison.  Check that
    // it is the values returned by the select that are being compared.
    Value *TrueVal = SI->getTrueValue();
    Value *FalseVal = SI->getFalseValue();
    Value *LHS = Cmp->getOperand(0);
    Value *RHS = Cmp->getOperand(1);
    if ((TrueVal != LHS || FalseVal != RHS) &&
        (TrueVal != RHS || FalseVal != LHS))
      return false;
    ICmpInst::Predicate Pred = LHS == TrueVal ?
      Cmp->getPredicate() : Cmp->getSwappedPredicate();
    // Does "(x pred y) ? x : y" represent the desired max/min operation?
    if (!Pred_t::match(Pred))
      return false;
    // It does!  Bind the operands.
    return L.match(LHS) && R.match(RHS);
  }
};

/// smax_pred_ty - Helper class for identifying signed max predicates.
struct smax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SGT || Pred == CmpInst::ICMP_SGE;
  }
};

/// smin_pred_ty - Helper class for identifying signed min predicates.
struct smin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SLT || Pred == CmpInst::ICMP_SLE;
  }
};

/// umax_pred_ty - Helper class for identifying unsigned max predicates.
struct umax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_UGT || Pred == CmpInst::ICMP_UGE;
  }
};

/// umin_pred_ty - Helper class for identifying unsigned min predicates.
struct umin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_ULT || Pred == CmpInst::ICMP_ULE;
  }
};

template<typename LHS, typename RHS>
inline MaxMin_match<LHS, RHS, smax_pred_ty>
m_SMax(const LHS &L, const RHS &R) {
  return MaxMin_match<LHS, RHS, smax_pred_ty>(L, R);
}

template<typename LHS, typename RHS>
inline MaxMin_match<LHS, RHS, smin_pred_ty>
m_SMin(const LHS &L, const RHS &R) {
  return MaxMin_match<LHS, RHS, smin_pred_ty>(L, R);
}

template<typename LHS, typename RHS>
inline MaxMin_match<LHS, RHS, umax_pred_ty>
m_UMax(const LHS &L, const RHS &R) {
  return MaxMin_match<LHS, RHS, umax_pred_ty>(L, R);
}

template<typename LHS, typename RHS>
inline MaxMin_match<LHS, RHS, umin_pred_ty>
m_UMin(const LHS &L, const RHS &R) {
  return MaxMin_match<LHS, RHS, umin_pred_ty>(L, R);
}

} // end namespace PatternMatch
} // end namespace llvm

#endif
