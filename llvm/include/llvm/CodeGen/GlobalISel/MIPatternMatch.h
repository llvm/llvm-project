//==------ llvm/CodeGen/GlobalISel/MIPatternMatch.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Contains matchers for matching SSA Machine Instructions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_GMIR_PATTERNMATCH_H
#define LLVM_GMIR_PATTERNMATCH_H

#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/InstrTypes.h"

namespace llvm {
namespace MIPatternMatch {

template <typename Reg, typename Pattern>
bool mi_match(Reg R, const MachineRegisterInfo &MRI, Pattern &&P) {
  return P.match(MRI, R);
}

// TODO: Extend for N use.
template <typename SubPatternT> struct OneUse_match {
  SubPatternT SubPat;
  OneUse_match(const SubPatternT &SP) : SubPat(SP) {}

  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return MRI.hasOneUse(Reg) && SubPat.match(MRI, Reg);
  }
};

template <typename SubPat>
inline OneUse_match<SubPat> m_OneUse(const SubPat &SP) {
  return SP;
}

struct ConstantMatch {
  int64_t &CR;
  ConstantMatch(int64_t &C) : CR(C) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    if (auto MaybeCst = getConstantVRegSExtVal(Reg, MRI)) {
      CR = *MaybeCst;
      return true;
    }
    return false;
  }
};

inline ConstantMatch m_ICst(int64_t &Cst) { return ConstantMatch(Cst); }

/// Matcher for a specific constant value.
struct SpecificConstantMatch {
  int64_t RequestedVal;
  SpecificConstantMatch(int64_t RequestedVal) : RequestedVal(RequestedVal) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    int64_t MatchedVal;
    return mi_match(Reg, MRI, m_ICst(MatchedVal)) && MatchedVal == RequestedVal;
  }
};

/// Matches a constant equal to \p RequestedValue.
inline SpecificConstantMatch m_SpecificICst(int64_t RequestedValue) {
  return SpecificConstantMatch(RequestedValue);
}

///{
/// Convenience matchers for specific integer values.
inline SpecificConstantMatch m_ZeroInt() { return SpecificConstantMatch(0); }
inline SpecificConstantMatch m_AllOnesInt() {
  return SpecificConstantMatch(-1);
}
///}

// TODO: Rework this for different kinds of MachineOperand.
// Currently assumes the Src for a match is a register.
// We might want to support taking in some MachineOperands and call getReg on
// that.

struct operand_type_match {
  bool match(const MachineRegisterInfo &MRI, Register Reg) { return true; }
  bool match(const MachineRegisterInfo &MRI, MachineOperand *MO) {
    return MO->isReg();
  }
};

inline operand_type_match m_Reg() { return operand_type_match(); }

/// Matching combinators.
template <typename... Preds> struct And {
  template <typename MatchSrc>
  bool match(const MachineRegisterInfo &MRI, MatchSrc &&src) {
    return true;
  }
};

template <typename Pred, typename... Preds>
struct And<Pred, Preds...> : And<Preds...> {
  Pred P;
  And(Pred &&p, Preds &&... preds)
      : And<Preds...>(std::forward<Preds>(preds)...), P(std::forward<Pred>(p)) {
  }
  template <typename MatchSrc>
  bool match(const MachineRegisterInfo &MRI, MatchSrc &&src) {
    return P.match(MRI, src) && And<Preds...>::match(MRI, src);
  }
};

template <typename... Preds> struct Or {
  template <typename MatchSrc>
  bool match(const MachineRegisterInfo &MRI, MatchSrc &&src) {
    return false;
  }
};

template <typename Pred, typename... Preds>
struct Or<Pred, Preds...> : Or<Preds...> {
  Pred P;
  Or(Pred &&p, Preds &&... preds)
      : Or<Preds...>(std::forward<Preds>(preds)...), P(std::forward<Pred>(p)) {}
  template <typename MatchSrc>
  bool match(const MachineRegisterInfo &MRI, MatchSrc &&src) {
    return P.match(MRI, src) || Or<Preds...>::match(MRI, src);
  }
};

template <typename... Preds> And<Preds...> m_all_of(Preds &&... preds) {
  return And<Preds...>(std::forward<Preds>(preds)...);
}

template <typename... Preds> Or<Preds...> m_any_of(Preds &&... preds) {
  return Or<Preds...>(std::forward<Preds>(preds)...);
}

template <typename BindTy> struct bind_helper {
  static bool bind(const MachineRegisterInfo &MRI, BindTy &VR, BindTy &V) {
    VR = V;
    return true;
  }
};

template <> struct bind_helper<MachineInstr *> {
  static bool bind(const MachineRegisterInfo &MRI, MachineInstr *&MI,
                   Register Reg) {
    MI = MRI.getVRegDef(Reg);
    if (MI)
      return true;
    return false;
  }
};

template <> struct bind_helper<LLT> {
  static bool bind(const MachineRegisterInfo &MRI, LLT Ty, Register Reg) {
    Ty = MRI.getType(Reg);
    if (Ty.isValid())
      return true;
    return false;
  }
};

template <> struct bind_helper<const ConstantFP *> {
  static bool bind(const MachineRegisterInfo &MRI, const ConstantFP *&F,
                   Register Reg) {
    F = getConstantFPVRegVal(Reg, MRI);
    if (F)
      return true;
    return false;
  }
};

template <typename Class> struct bind_ty {
  Class &VR;

  bind_ty(Class &V) : VR(V) {}

  template <typename ITy> bool match(const MachineRegisterInfo &MRI, ITy &&V) {
    return bind_helper<Class>::bind(MRI, VR, V);
  }
};

inline bind_ty<Register> m_Reg(Register &R) { return R; }
inline bind_ty<MachineInstr *> m_MInstr(MachineInstr *&MI) { return MI; }
inline bind_ty<LLT> m_Type(LLT Ty) { return Ty; }
inline bind_ty<CmpInst::Predicate> m_Pred(CmpInst::Predicate &P) { return P; }
inline operand_type_match m_Pred() { return operand_type_match(); }

// Helper for matching G_FCONSTANT
inline bind_ty<const ConstantFP *> m_GFCst(const ConstantFP *&C) { return C; }

// General helper for all the binary generic MI such as G_ADD/G_SUB etc
template <typename LHS_P, typename RHS_P, unsigned Opcode,
          bool Commutable = false>
struct BinaryOp_match {
  LHS_P L;
  RHS_P R;

  BinaryOp_match(const LHS_P &LHS, const RHS_P &RHS) : L(LHS), R(RHS) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (mi_match(Op, MRI, m_MInstr(TmpMI))) {
      if (TmpMI->getOpcode() == Opcode && TmpMI->getNumOperands() == 3) {
        return (L.match(MRI, TmpMI->getOperand(1).getReg()) &&
                R.match(MRI, TmpMI->getOperand(2).getReg())) ||
               (Commutable && (R.match(MRI, TmpMI->getOperand(1).getReg()) &&
                               L.match(MRI, TmpMI->getOperand(2).getReg())));
      }
    }
    return false;
  }
};

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_ADD, true>
m_GAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_ADD, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB> m_GSub(const LHS &L,
                                                            const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_MUL, true>
m_GMul(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_MUL, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_FADD, true>
m_GFAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_FADD, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_FMUL, true>
m_GFMul(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_FMUL, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_FSUB, false>
m_GFSub(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_FSUB, false>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_AND, true>
m_GAnd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_AND, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_XOR, true>
m_GXor(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_XOR, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_OR, true> m_GOr(const LHS &L,
                                                                const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_OR, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SHL, false>
m_GShl(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SHL, false>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_LSHR, false>
m_GLShr(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_LSHR, false>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_ASHR, false>
m_GAShr(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_ASHR, false>(L, R);
}

// Helper for unary instructions (G_[ZSA]EXT/G_TRUNC) etc
template <typename SrcTy, unsigned Opcode> struct UnaryOp_match {
  SrcTy L;

  UnaryOp_match(const SrcTy &LHS) : L(LHS) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (mi_match(Op, MRI, m_MInstr(TmpMI))) {
      if (TmpMI->getOpcode() == Opcode && TmpMI->getNumOperands() == 2) {
        return L.match(MRI, TmpMI->getOperand(1).getReg());
      }
    }
    return false;
  }
};

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_ANYEXT>
m_GAnyExt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_ANYEXT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_SEXT> m_GSExt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_SEXT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_ZEXT> m_GZExt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_ZEXT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FPEXT> m_GFPExt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FPEXT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_TRUNC> m_GTrunc(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_TRUNC>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_BITCAST>
m_GBitcast(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_BITCAST>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_PTRTOINT>
m_GPtrToInt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_PTRTOINT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_INTTOPTR>
m_GIntToPtr(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_INTTOPTR>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FPTRUNC>
m_GFPTrunc(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FPTRUNC>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FABS> m_GFabs(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FABS>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FNEG> m_GFNeg(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FNEG>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::COPY> m_Copy(SrcTy &&Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::COPY>(std::forward<SrcTy>(Src));
}

// General helper for generic MI compares, i.e. G_ICMP and G_FCMP
// TODO: Allow checking a specific predicate.
template <typename Pred_P, typename LHS_P, typename RHS_P, unsigned Opcode>
struct CompareOp_match {
  Pred_P P;
  LHS_P L;
  RHS_P R;

  CompareOp_match(const Pred_P &Pred, const LHS_P &LHS, const RHS_P &RHS)
      : P(Pred), L(LHS), R(RHS) {}

  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (!mi_match(Op, MRI, m_MInstr(TmpMI)) || TmpMI->getOpcode() != Opcode)
      return false;

    auto TmpPred =
        static_cast<CmpInst::Predicate>(TmpMI->getOperand(1).getPredicate());
    if (!P.match(MRI, TmpPred))
      return false;

    return L.match(MRI, TmpMI->getOperand(2).getReg()) &&
           R.match(MRI, TmpMI->getOperand(3).getReg());
  }
};

template <typename Pred, typename LHS, typename RHS>
inline CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_ICMP>
m_GICmp(const Pred &P, const LHS &L, const RHS &R) {
  return CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_ICMP>(P, L, R);
}

template <typename Pred, typename LHS, typename RHS>
inline CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_FCMP>
m_GFCmp(const Pred &P, const LHS &L, const RHS &R) {
  return CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_FCMP>(P, L, R);
}

// Helper for checking if a Reg is of specific type.
struct CheckType {
  LLT Ty;
  CheckType(const LLT Ty) : Ty(Ty) {}

  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return MRI.getType(Reg) == Ty;
  }
};

inline CheckType m_SpecificType(LLT Ty) { return Ty; }

template <typename Src0Ty, typename Src1Ty, typename Src2Ty, unsigned Opcode>
struct TernaryOp_match {
  Src0Ty Src0;
  Src1Ty Src1;
  Src2Ty Src2;

  TernaryOp_match(const Src0Ty &Src0, const Src1Ty &Src1, const Src2Ty &Src2)
      : Src0(Src0), Src1(Src1), Src2(Src2) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (mi_match(Op, MRI, m_MInstr(TmpMI))) {
      if (TmpMI->getOpcode() == Opcode && TmpMI->getNumOperands() == 4) {
        return (Src0.match(MRI, TmpMI->getOperand(1).getReg()) &&
                Src1.match(MRI, TmpMI->getOperand(2).getReg()) &&
                Src2.match(MRI, TmpMI->getOperand(3).getReg()));
      }
    }
    return false;
  }
};
template <typename Src0Ty, typename Src1Ty, typename Src2Ty>
inline TernaryOp_match<Src0Ty, Src1Ty, Src2Ty,
                       TargetOpcode::G_INSERT_VECTOR_ELT>
m_GInsertVecElt(const Src0Ty &Src0, const Src1Ty &Src1, const Src2Ty &Src2) {
  return TernaryOp_match<Src0Ty, Src1Ty, Src2Ty,
                         TargetOpcode::G_INSERT_VECTOR_ELT>(Src0, Src1, Src2);
}

/// Matches a register negated by a G_SUB.
/// G_SUB 0, %negated_reg
template <typename SrcTy>
inline BinaryOp_match<SpecificConstantMatch, SrcTy, TargetOpcode::G_SUB>
m_Neg(const SrcTy &&Src) {
  return m_GSub(m_ZeroInt(), Src);
}

/// Matches a register not-ed by a G_XOR.
/// G_XOR %not_reg, -1
template <typename SrcTy>
inline BinaryOp_match<SrcTy, SpecificConstantMatch, TargetOpcode::G_XOR, true>
m_Not(const SrcTy &&Src) {
  return m_GXor(Src, m_AllOnesInt());
}

} // namespace GMIPatternMatch
} // namespace llvm

#endif
