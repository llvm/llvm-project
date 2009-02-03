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

namespace llvm {
namespace PatternMatch {

template<typename Val, typename Pattern>
bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern&>(P).match(V);
}

template<typename Class>
struct leaf_ty {
  template<typename ITy>
  bool match(ITy *V) { return isa<Class>(V); }
};

/// m_Value() - Match an arbitrary value and ignore it.
inline leaf_ty<Value> m_Value() { return leaf_ty<Value>(); }
/// m_ConstantInt() - Match an arbitrary ConstantInt and ignore it.
inline leaf_ty<ConstantInt> m_ConstantInt() { return leaf_ty<ConstantInt>(); }

template<int64_t Val>
struct constantint_ty {
  template<typename ITy>
  bool match(ITy *V) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      const APInt &CIV = CI->getValue();
      if (Val >= 0)
        return CIV == Val;
      // If Val is negative, and CI is shorter than it, truncate to the right
      // number of bits.  If it is larger, then we have to sign extend.  Just
      // compare their negated values.
      return -CIV == -Val;
    }
    return false;
  }
};

/// m_ConstantInt(int64_t) - Match a ConstantInt with a specific value
/// and ignore it.
template<int64_t Val>
inline constantint_ty<Val> m_ConstantInt() {
  return constantint_ty<Val>();
}

struct zero_ty {
  template<typename ITy>
  bool match(ITy *V) {
    if (const Constant *C = dyn_cast<Constant>(V))
      return C->isNullValue();
    return false;
  }
};

/// m_Zero() - Match an arbitrary zero/null constant.
inline zero_ty m_Zero() { return zero_ty(); }


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
  

//===----------------------------------------------------------------------===//
// Matchers for specific binary operators.
//

template<typename LHS_t, typename RHS_t, 
         unsigned Opcode, typename ConcreteTy = BinaryOperator>
struct BinaryOp_match {
  LHS_t L;
  RHS_t R;

  BinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      ConcreteTy *I = cast<ConcreteTy>(V);
      return I->getOpcode() == Opcode && L.match(I->getOperand(0)) &&
             R.match(I->getOperand(1));
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add> m_Add(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Sub> m_Sub(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Sub>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul> m_Mul(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::UDiv> m_UDiv(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::UDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SDiv> m_SDiv(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FDiv> m_FDiv(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FDiv>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::URem> m_URem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::URem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SRem> m_SRem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SRem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FRem> m_FRem(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FRem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And> m_And(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Or> m_Or(const LHS &L,
                                                      const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor> m_Xor(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shl> m_Shl(const LHS &L, 
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shl>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::LShr> m_LShr(const LHS &L, 
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::LShr>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::AShr> m_AShr(const LHS &L, 
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::AShr>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for either AShr or LShr .. for convenience
//
template<typename LHS_t, typename RHS_t, typename ConcreteTy = BinaryOperator>
struct Shr_match {
  LHS_t L;
  RHS_t R;

  Shr_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Instruction::LShr ||
        V->getValueID() == Value::InstructionVal + Instruction::AShr) {
      ConcreteTy *I = cast<ConcreteTy>(V);
      return (I->getOpcode() == Instruction::AShr ||
              I->getOpcode() == Instruction::LShr) &&
             L.match(I->getOperand(0)) &&
             R.match(I->getOperand(1));
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return (CE->getOpcode() == Instruction::LShr ||
              CE->getOpcode() == Instruction::AShr) &&
             L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

template<typename LHS, typename RHS>
inline Shr_match<LHS, RHS> m_Shr(const LHS &L, const RHS &R) {
  return Shr_match<LHS, RHS>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for binary classes
//

template<typename LHS_t, typename RHS_t, typename Class, typename OpcType>
struct BinaryOpClass_match {
  OpcType *Opcode;
  LHS_t L;
  RHS_t R;

  BinaryOpClass_match(OpcType &Op, const LHS_t &LHS,
                      const RHS_t &RHS)
    : Opcode(&Op), L(LHS), R(RHS) {}
  BinaryOpClass_match(const LHS_t &LHS, const RHS_t &RHS)
    : Opcode(0), L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Class *I = dyn_cast<Class>(V))
      if (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) {
        if (Opcode)
          *Opcode = I->getOpcode();
        return true;
      }
#if 0  // Doesn't handle constantexprs yet!
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
#endif
    return false;
  }
};

template<typename LHS, typename RHS>
inline BinaryOpClass_match<LHS, RHS, BinaryOperator, Instruction::BinaryOps>
m_Shift(Instruction::BinaryOps &Op, const LHS &L, const RHS &R) {
  return BinaryOpClass_match<LHS, RHS, 
                             BinaryOperator, Instruction::BinaryOps>(Op, L, R);
}

template<typename LHS, typename RHS>
inline BinaryOpClass_match<LHS, RHS, BinaryOperator, Instruction::BinaryOps>
m_Shift(const LHS &L, const RHS &R) {
  return BinaryOpClass_match<LHS, RHS, 
                             BinaryOperator, Instruction::BinaryOps>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for CmpInst classes
//

template<typename LHS_t, typename RHS_t, typename Class, typename PredicateTy>
struct CmpClass_match {
  PredicateTy &Predicate;
  LHS_t L;
  RHS_t R;

  CmpClass_match(PredicateTy &Pred, const LHS_t &LHS,
                 const RHS_t &RHS)
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
inline SelectClass_match<Cond, RHS, LHS>
m_Select(const Cond &C, const LHS &L, const RHS &R) {
  return SelectClass_match<Cond, LHS, RHS>(C, L, R);
}

/// m_SelectCst - This matches a select of two constants, e.g.:
///    m_SelectCst(m_Value(V), -1, 0)
template<int64_t L, int64_t R, typename Cond>
inline SelectClass_match<Cond, constantint_ty<L>, constantint_ty<R> >
m_SelectCst(const Cond &C) {
  return SelectClass_match<Cond, constantint_ty<L>, 
                           constantint_ty<R> >(C, m_ConstantInt<L>(),
                                           m_ConstantInt<R>());
}


//===----------------------------------------------------------------------===//
// Matchers for CastInst classes
//

template<typename Op_t, typename Class>
struct CastClass_match {
  Op_t Op;
  
  CastClass_match(const Op_t &OpMatch) : Op(OpMatch) {}
  
  template<typename OpTy>
  bool match(OpTy *V) {
    if (Class *I = dyn_cast<Class>(V))
      return Op.match(I->getOperand(0));
    return false;
  }
};

template<typename Class, typename OpTy>
inline CastClass_match<OpTy, Class> m_Cast(const OpTy &Op) {
  return CastClass_match<OpTy, Class>(Op);
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
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Xor)
        return matchIfNot(I->getOperand(0), I->getOperand(1));
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::Xor)
        return matchIfNot(CE->getOperand(0), CE->getOperand(1));
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return L.match(ConstantExpr::getNot(CI));
    return false;
  }
private:
  bool matchIfNot(Value *LHS, Value *RHS) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS))
      return CI->isAllOnesValue() && L.match(LHS);
    if (ConstantInt *CI = dyn_cast<ConstantInt>(LHS))
      return CI->isAllOnesValue() && L.match(RHS);
    if (ConstantVector *CV = dyn_cast<ConstantVector>(RHS))
      return CV->isAllOnesValue() && L.match(LHS);
    if (ConstantVector *CV = dyn_cast<ConstantVector>(LHS))
      return CV->isAllOnesValue() && L.match(RHS);
    return false;
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
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Sub)
        return matchIfNeg(I->getOperand(0), I->getOperand(1));
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::Sub)
        return matchIfNeg(CE->getOperand(0), CE->getOperand(1));
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return L.match(ConstantExpr::getNeg(CI));
    return false;
  }
private:
  bool matchIfNeg(Value *LHS, Value *RHS) {
    return LHS == ConstantExpr::getZeroValueForNegationExpr(LHS->getType()) &&
           L.match(RHS);
  }
};

template<typename LHS>
inline neg_match<LHS> m_Neg(const LHS &L) { return L; }


//===----------------------------------------------------------------------===//
// Matchers for control flow
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
      if (BI->isConditional()) {
        if (Cond.match(BI->getCondition())) {
          T = BI->getSuccessor(0);
          F = BI->getSuccessor(1);
          return true;
        }
      }
    return false;
  }
};

template<typename Cond_t>
inline brc_match<Cond_t> m_Br(const Cond_t &C, BasicBlock *&T, BasicBlock *&F) {
  return brc_match<Cond_t>(C, T, F);
}

} // end namespace PatternMatch
} // end namespace llvm

#endif
