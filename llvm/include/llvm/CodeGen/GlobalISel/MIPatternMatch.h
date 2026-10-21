//==------ llvm/CodeGen/GlobalISel/MIPatternMatch.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Contains matchers for matching SSA Machine Instructions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_MIPATTERNMATCH_H
#define LLVM_CODEGEN_GLOBALISEL_MIPATTERNMATCH_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/InstrTypes.h"
#include <tuple>
#include <utility>

namespace llvm {
namespace MIPatternMatch {

template <typename Reg, typename Pattern>
[[nodiscard]] bool mi_match(Reg R, const MachineRegisterInfo &MRI,
                            Pattern &&P) {
  return P.match(MRI, R);
}

template <typename Pattern>
[[nodiscard]] bool mi_match(MachineInstr &MI, const MachineRegisterInfo &MRI,
                            Pattern &&P) {
  return P.match(MRI, &MI);
}

template <typename Pattern>
[[nodiscard]] bool mi_match(const MachineInstr &MI,
                            const MachineRegisterInfo &MRI, Pattern &&P) {
  return P.match(MRI, &MI);
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

template <typename SubPatternT> struct OneNonDBGUse_match {
  SubPatternT SubPat;
  OneNonDBGUse_match(const SubPatternT &SP) : SubPat(SP) {}

  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return MRI.hasOneNonDBGUse(Reg) && SubPat.match(MRI, Reg);
  }
};

template <typename SubPat>
inline OneNonDBGUse_match<SubPat> m_OneNonDBGUse(const SubPat &SP) {
  return SP;
}

template <typename ConstT>
inline std::optional<ConstT> matchConstant(Register,
                                           const MachineRegisterInfo &);

template <>
inline std::optional<APInt> matchConstant(Register Reg,
                                          const MachineRegisterInfo &MRI) {
  return getIConstantVRegVal(Reg, MRI);
}

template <>
inline std::optional<int64_t> matchConstant(Register Reg,
                                            const MachineRegisterInfo &MRI) {
  return getIConstantVRegSExtVal(Reg, MRI);
}

template <typename ConstT> struct ConstantMatch {
  ConstT &CR;
  ConstantMatch(ConstT &C) : CR(C) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    if (auto MaybeCst = matchConstant<ConstT>(Reg, MRI)) {
      CR = *MaybeCst;
      return true;
    }
    return false;
  }
};

inline ConstantMatch<APInt> m_ICst(APInt &Cst) {
  return ConstantMatch<APInt>(Cst);
}
inline ConstantMatch<int64_t> m_ICst(int64_t &Cst) {
  return ConstantMatch<int64_t>(Cst);
}

template <typename ConstT>
inline std::optional<ConstT> matchConstantSplat(Register,
                                                const MachineRegisterInfo &);

template <>
inline std::optional<APInt> matchConstantSplat(Register Reg,
                                               const MachineRegisterInfo &MRI) {
  return getIConstantSplatVal(Reg, MRI);
}

template <>
inline std::optional<int64_t>
matchConstantSplat(Register Reg, const MachineRegisterInfo &MRI) {
  return getIConstantSplatSExtVal(Reg, MRI);
}

template <typename ConstT> struct ICstOrSplatMatch {
  ConstT &CR;
  ICstOrSplatMatch(ConstT &C) : CR(C) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    if (auto MaybeCst = matchConstant<ConstT>(Reg, MRI)) {
      CR = *MaybeCst;
      return true;
    }

    if (auto MaybeCstSplat = matchConstantSplat<ConstT>(Reg, MRI)) {
      CR = *MaybeCstSplat;
      return true;
    }

    return false;
  };
};

inline ICstOrSplatMatch<APInt> m_ICstOrSplat(APInt &Cst) {
  return ICstOrSplatMatch<APInt>(Cst);
}

inline ICstOrSplatMatch<int64_t> m_ICstOrSplat(int64_t &Cst) {
  return ICstOrSplatMatch<int64_t>(Cst);
}

struct GCstAndRegMatch {
  std::optional<ValueAndVReg> &ValReg;
  GCstAndRegMatch(std::optional<ValueAndVReg> &ValReg) : ValReg(ValReg) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    ValReg = getIConstantVRegValWithLookThrough(Reg, MRI);
    return ValReg ? true : false;
  }
};

inline GCstAndRegMatch m_GCst(std::optional<ValueAndVReg> &ValReg) {
  return GCstAndRegMatch(ValReg);
}

struct GFCstAndRegMatch {
  std::optional<FPValueAndVReg> &FPValReg;
  GFCstAndRegMatch(std::optional<FPValueAndVReg> &FPValReg)
      : FPValReg(FPValReg) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    FPValReg = getFConstantVRegValWithLookThrough(Reg, MRI);
    return FPValReg ? true : false;
  }
};

inline GFCstAndRegMatch m_GFCst(std::optional<FPValueAndVReg> &FPValReg) {
  return GFCstAndRegMatch(FPValReg);
}

struct GFCstOrSplatGFCstMatch {
  std::optional<FPValueAndVReg> &FPValReg;
  GFCstOrSplatGFCstMatch(std::optional<FPValueAndVReg> &FPValReg)
      : FPValReg(FPValReg) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return (FPValReg = getFConstantSplat(Reg, MRI)) ||
           (FPValReg = getFConstantVRegValWithLookThrough(Reg, MRI));
  };
};

inline GFCstOrSplatGFCstMatch
m_GFCstOrSplat(std::optional<FPValueAndVReg> &FPValReg) {
  return GFCstOrSplatGFCstMatch(FPValReg);
}

/// Matches an FP constant whose value satisfies the given predicate.
template <typename Pred> struct GFCstPredMatch {
  Pred P;
  GFCstPredMatch(Pred P) : P(P) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    if (const ConstantFP *FPImm = getConstantFPVRegVal(Reg, MRI))
      return P(FPImm->getValueAPF());
    return false;
  }
};
template <typename Pred> GFCstPredMatch(Pred) -> GFCstPredMatch<Pred>;

/// Matches a floating-point positive zero.
inline auto m_PosZeroFP() {
  return GFCstPredMatch([](const APFloat &V) { return V.isPosZero(); });
}

/// Matcher for a specific constant value.
struct SpecificConstantMatch {
  APInt RequestedVal;
  SpecificConstantMatch(const APInt &RequestedVal)
      : RequestedVal(RequestedVal) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    APInt MatchedVal;
    if (mi_match(Reg, MRI, m_ICst(MatchedVal))) {
      if (MatchedVal.getBitWidth() > RequestedVal.getBitWidth())
        RequestedVal = RequestedVal.sext(MatchedVal.getBitWidth());
      else
        MatchedVal = MatchedVal.sext(RequestedVal.getBitWidth());

      return APInt::isSameValue(MatchedVal, RequestedVal);
    }
    return false;
  }
};

/// Matches a constant equal to \p RequestedValue.
inline SpecificConstantMatch m_SpecificICst(const APInt &RequestedValue) {
  return SpecificConstantMatch(RequestedValue);
}

inline SpecificConstantMatch m_SpecificICst(int64_t RequestedValue) {
  return SpecificConstantMatch(APInt(64, RequestedValue, /* isSigned */ true));
}

struct SpecificImmMatch {
  int64_t RequestedVal;
  SpecificImmMatch(int64_t RequestedVal) : RequestedVal(RequestedVal) {}
  bool match(int64_t Imm) const { return Imm == RequestedVal; }
};

/// Matches an immediate operand equal to \p RequestedValue.
inline SpecificImmMatch m_SpecificImm(int64_t RequestedValue) {
  return SpecificImmMatch(RequestedValue);
}

struct BindImmMatch {
  int64_t &ImmOut;
  BindImmMatch(int64_t &ImmOut) : ImmOut(ImmOut) {}
  bool match(int64_t Imm) const {
    ImmOut = Imm;
    return true;
  }
};

/// Binds an immediate operand's value.
inline BindImmMatch m_Imm(int64_t &Imm) { return BindImmMatch(Imm); }

/// Matches an integer constant with all bits set, regardless of width.
struct AllOnesConstantMatch {
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    APInt MatchedVal;
    return mi_match(Reg, MRI, m_ICst(MatchedVal)) && MatchedVal.isAllOnes();
  }
};

inline AllOnesConstantMatch m_AllOnes() { return {}; }

/// Matcher for a specific constant splat.
struct SpecificConstantSplatMatch {
  APInt RequestedVal;
  SpecificConstantSplatMatch(const APInt &RequestedVal)
      : RequestedVal(RequestedVal) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return isBuildVectorConstantSplat(Reg, MRI, RequestedVal,
                                      /* AllowUndef */ false);
  }
};

/// Matches a constant splat of \p RequestedValue.
inline SpecificConstantSplatMatch
m_SpecificICstSplat(const APInt &RequestedValue) {
  return SpecificConstantSplatMatch(RequestedValue);
}

inline SpecificConstantSplatMatch m_SpecificICstSplat(int64_t RequestedValue) {
  return SpecificConstantSplatMatch(
      APInt(64, RequestedValue, /* isSigned */ true));
}

/// Matcher for a specific constant or constant splat.
struct SpecificConstantOrSplatMatch {
  APInt RequestedVal;
  SpecificConstantOrSplatMatch(const APInt &RequestedVal)
      : RequestedVal(RequestedVal) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    APInt MatchedVal;
    if (mi_match(Reg, MRI, m_ICst(MatchedVal))) {
      if (MatchedVal.getBitWidth() > RequestedVal.getBitWidth())
        RequestedVal = RequestedVal.sext(MatchedVal.getBitWidth());
      else
        MatchedVal = MatchedVal.sext(RequestedVal.getBitWidth());

      if (APInt::isSameValue(MatchedVal, RequestedVal))
        return true;
    }
    return isBuildVectorConstantSplat(Reg, MRI, RequestedVal,
                                      /* AllowUndef */ false);
  }
};

/// Matches a \p RequestedValue constant or a constant splat of \p
/// RequestedValue.
inline SpecificConstantOrSplatMatch
m_SpecificICstOrSplat(const APInt &RequestedValue) {
  return SpecificConstantOrSplatMatch(RequestedValue);
}

inline SpecificConstantOrSplatMatch
m_SpecificICstOrSplat(int64_t RequestedValue) {
  return SpecificConstantOrSplatMatch(
      APInt(64, RequestedValue, /* isSigned */ true));
}

/// Convenience matchers for specific integer values.
inline SpecificConstantMatch m_ZeroInt() {
  return SpecificConstantMatch(APInt::getZero(64));
}
inline SpecificConstantMatch m_AllOnesInt() {
  return SpecificConstantMatch(APInt::getAllOnes(64));
}

/// Matcher for a specific register.
struct SpecificRegisterMatch {
  Register RequestedReg;
  SpecificRegisterMatch(Register RequestedReg) : RequestedReg(RequestedReg) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    return Reg == RequestedReg;
  }
};

/// Matches a register only if it is equal to \p RequestedReg.
inline SpecificRegisterMatch m_SpecificReg(Register RequestedReg) {
  return SpecificRegisterMatch(RequestedReg);
}

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
  static bool bind(const MachineRegisterInfo &MRI, MachineInstr *&MI,
                   MachineInstr *Inst) {
    MI = Inst;
    return MI;
  }
};

template <> struct bind_helper<const MachineInstr *> {
  static bool bind(const MachineRegisterInfo &MRI, const MachineInstr *&MI,
                   Register Reg) {
    MI = MRI.getVRegDef(Reg);
    return MI;
  }
  static bool bind(const MachineRegisterInfo &MRI, const MachineInstr *&MI,
                   const MachineInstr *Inst) {
    MI = Inst;
    return MI;
  }
};

template <> struct bind_helper<LLT> {
  static bool bind(const MachineRegisterInfo &MRI, LLT &Ty, Register Reg) {
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
inline bind_ty<const MachineInstr *> m_MInstr(const MachineInstr *&MI) {
  return MI;
}
inline bind_ty<LLT> m_Type(LLT &Ty) { return Ty; }
inline bind_ty<CmpInst::Predicate> m_Pred(CmpInst::Predicate &P) { return P; }
inline operand_type_match m_Pred() { return operand_type_match(); }
inline bind_ty<FPClassTest> m_FPClassTest(FPClassTest &T) { return T; }

/// Wraps a MIFlags output for use as an optional trailing operand of an
/// instruction matcher (e.g. m_GPtrAdd(L, R, m_MIFlags(Flags))). On a
/// successful match the matched instruction's flags are written to \p Flags.
struct MIFlagsRef {
  uint32_t &Flags;
};

inline MIFlagsRef m_MIFlags(uint32_t &Flags) { return {Flags}; }

/// Optional trailing operand for a load matcher (e.g. m_GLoad(m_Reg(Ptr),
/// m_MMO(MMO))) that binds the matched instruction's MachineMemOperand.
struct MMORef {
  const MachineMemOperand *&MMO;
};

inline MMORef m_MMO(const MachineMemOperand *&MMO) { return {MMO}; }

template <typename BindTy> struct deferred_helper {
  static bool match(const MachineRegisterInfo &MRI, BindTy &VR, BindTy &V) {
    return VR == V;
  }
};

template <> struct deferred_helper<LLT> {
  static bool match(const MachineRegisterInfo &MRI, LLT VT, Register R) {
    return VT == MRI.getType(R);
  }
};

template <typename Class> struct deferred_ty {
  Class &VR;

  deferred_ty(Class &V) : VR(V) {}

  template <typename ITy> bool match(const MachineRegisterInfo &MRI, ITy &&V) {
    return deferred_helper<Class>::match(MRI, VR, V);
  }
};

/// Similar to m_SpecificReg/Type, but the specific value to match originated
/// from an earlier sub-pattern in the same mi_match expression. For example,
/// we cannot match `(add X, X)` with `m_GAdd(m_Reg(X), m_SpecificReg(X))`
/// because `X` is not initialized at the time it's passed to `m_SpecificReg`.
/// Instead, we can use `m_GAdd(m_Reg(x), m_DeferredReg(X))`.
inline deferred_ty<Register> m_DeferredReg(Register &R) { return R; }
inline deferred_ty<LLT> m_DeferredType(LLT &Ty) { return Ty; }

struct ImplicitDefMatch {
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *TmpMI;
    if (mi_match(Reg, MRI, m_MInstr(TmpMI)))
      return TmpMI->getOpcode() == TargetOpcode::G_IMPLICIT_DEF;
    return false;
  }
};

inline ImplicitDefMatch m_GImplicitDef() { return ImplicitDefMatch(); }

/// Binds the defining instruction of \p Reg if it is a \p Class. Prefer the
/// named helpers below so the opcode is spelled out at the call site.
template <typename Class> struct GInstrBind {
  Class *&Inst;

  GInstrBind(Class *&Inst) : Inst(Inst) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *TmpMI;
    if (mi_match(Reg, MRI, m_MInstr(TmpMI))) {
      if (auto *Cst = dyn_cast<Class>(TmpMI)) {
        Inst = Cst;
        return true;
      }
    }
    return false;
  }
};

/// Match a literal G_CONSTANT instruction (no look-through of splats or
/// copies).
inline GInstrBind<GConstant> m_GConstant(GConstant *&Inst) { return Inst; }
inline GInstrBind<const GConstant> m_GConstant(const GConstant *&Inst) {
  return Inst;
}

/// Match a literal G_CONSTANT or G_FCONSTANT, binding its raw bits to \p Bits
/// (the integer value, or the float reinterpreted as an integer).
struct GConstantBitsMatch {
  APInt &Bits;
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *MI = MRI.getVRegDef(Reg);
    if (MI->getOpcode() == TargetOpcode::G_CONSTANT) {
      Bits = MI->getOperand(1).getCImm()->getValue();
      return true;
    }
    if (MI->getOpcode() == TargetOpcode::G_FCONSTANT) {
      Bits = MI->getOperand(1).getFPImm()->getValueAPF().bitcastToAPInt();
      return true;
    }
    return false;
  }
};

inline GConstantBitsMatch m_GConstantOrFConstantBits(APInt &Bits) {
  return {Bits};
}

/// Match a load of type \p Class, binding its pointer operand (like IR's
/// m_Load), and optionally the instruction and/or its MachineMemOperand.
template <typename Class, typename PtrP> struct LoadOp_match {
  PtrP Ptr;
  Class **InstOut = nullptr;
  const MachineMemOperand **MMOOut = nullptr;

  LoadOp_match(const PtrP &Ptr) : Ptr(Ptr) {}
  LoadOp_match(const PtrP &Ptr, MMORef MMO) : Ptr(Ptr), MMOOut(&MMO.MMO) {}
  LoadOp_match(Class *&Inst, const PtrP &Ptr) : Ptr(Ptr), InstOut(&Inst) {}
  LoadOp_match(Class *&Inst, const PtrP &Ptr, MMORef MMO)
      : Ptr(Ptr), InstOut(&Inst), MMOOut(&MMO.MMO) {}

  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *TmpMI;
    if (!mi_match(Reg, MRI, m_MInstr(TmpMI)))
      return false;
    auto *Load = dyn_cast<Class>(TmpMI);
    if (!Load || !Ptr.match(MRI, Load->getPointerReg()))
      return false;
    if (InstOut)
      *InstOut = Load;
    if (MMOOut)
      *MMOOut = &Load->getMMO();
    return true;
  }
};

template <typename PtrP>
inline LoadOp_match<GAnyLoad, PtrP> m_GAnyLoad(const PtrP &Ptr) {
  return LoadOp_match<GAnyLoad, PtrP>(Ptr);
}
template <typename PtrP>
inline LoadOp_match<GAnyLoad, PtrP> m_GAnyLoad(GAnyLoad *&Inst,
                                               const PtrP &Ptr) {
  return LoadOp_match<GAnyLoad, PtrP>(Inst, Ptr);
}
template <typename PtrP>
inline LoadOp_match<GAnyLoad, PtrP> m_GAnyLoad(GAnyLoad *&Inst, const PtrP &Ptr,
                                               MMORef MMO) {
  return LoadOp_match<GAnyLoad, PtrP>(Inst, Ptr, MMO);
}
template <typename PtrP>
inline LoadOp_match<GLoad, PtrP> m_GLoad(const PtrP &Ptr) {
  return LoadOp_match<GLoad, PtrP>(Ptr);
}
template <typename PtrP>
inline LoadOp_match<GLoad, PtrP> m_GLoad(const PtrP &Ptr, MMORef MMO) {
  return LoadOp_match<GLoad, PtrP>(Ptr, MMO);
}

/// Instruction binders for ops with no operand-form matcher (constant-immediate
/// or variadic-source ops).
inline GInstrBind<GUnmerge> m_GUnmerge(GUnmerge *&Inst) { return Inst; }
inline GInstrBind<GVScale> m_GVScale(GVScale *&Inst) { return Inst; }
inline GInstrBind<GBuildVector> m_GBuildVector(GBuildVector *&Inst) {
  return Inst;
}
inline GInstrBind<GConcatVectors> m_GConcatVectors(GConcatVectors *&Inst) {
  return Inst;
}

/// Binds the defining instruction of \p Reg if it is a GIntrinsic (any of the
/// four G_INTRINSIC* opcodes).
inline GInstrBind<GIntrinsic> m_GIntrinsic(GIntrinsic *&Inst) { return Inst; }
inline GInstrBind<const GIntrinsic> m_GIntrinsic(const GIntrinsic *&Inst) {
  return Inst;
}

/// Matches a GIntrinsic with a specific intrinsic ID and optionally, matchers
/// for its leading arguments.
template <Intrinsic::ID IntrID, typename... OpMatchers>
struct GIntrinsic_match {
  std::tuple<OpMatchers...> Operands;

  GIntrinsic_match(const OpMatchers &...Ops) : Operands(Ops...) {}

  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (!mi_match(Op, MRI, m_MInstr(TmpMI)))
      return false;
    auto *GI = dyn_cast<GIntrinsic>(TmpMI);
    if (!GI || !GI->is(IntrID))
      return false;
    return matchOperands(MRI, *GI, std::index_sequence_for<OpMatchers...>{});
  }

private:
  template <size_t... Is>
  bool matchOperands(const MachineRegisterInfo &MRI, GIntrinsic &GI,
                     std::index_sequence<Is...>) {
    // Intrinsic arguments follow the ID operand.
    unsigned FirstArg = GI.getNumExplicitDefs() + 1;
    return (std::get<Is>(Operands).match(
                MRI, GI.getOperand(FirstArg + Is).getReg()) &&
            ...);
  }
};

template <Intrinsic::ID IntrID, typename... OpMatchers>
inline GIntrinsic_match<IntrID, OpMatchers...>
m_GIntrinsic(const OpMatchers &...Ops) {
  return GIntrinsic_match<IntrID, OpMatchers...>(Ops...);
}

/// Matches a G_SHUFFLE_VECTOR, binding its two source operands and its mask.
template <typename Src1Ty, typename Src2Ty> struct ShuffleVectorMatch {
  Src1Ty Src1;
  Src2Ty Src2;
  ArrayRef<int> &Mask;

  ShuffleVectorMatch(const Src1Ty &Src1, const Src2Ty &Src2,
                     ArrayRef<int> &Mask)
      : Src1(Src1), Src2(Src2), Mask(Mask) {}
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *TmpMI;
    if (!mi_match(Reg, MRI, m_MInstr(TmpMI)))
      return false;
    auto *Shuf = dyn_cast<GShuffleVector>(TmpMI);
    if (!Shuf || !Src1.match(MRI, Shuf->getSrc1Reg()) ||
        !Src2.match(MRI, Shuf->getSrc2Reg()))
      return false;
    Mask = Shuf->getMask();
    return true;
  }
};

template <typename Src1Ty, typename Src2Ty>
inline ShuffleVectorMatch<Src1Ty, Src2Ty>
m_GShuffleVector(const Src1Ty &Src1, const Src2Ty &Src2, ArrayRef<int> &Mask) {
  return ShuffleVectorMatch<Src1Ty, Src2Ty>(Src1, Src2, Mask);
}

/// Matches a G_FRAME_INDEX, binding its frame index.
struct GFrameIndexMatch {
  int &FI;
  bool match(const MachineRegisterInfo &MRI, Register Reg) {
    MachineInstr *TmpMI;
    if (mi_match(Reg, MRI, m_MInstr(TmpMI)) &&
        TmpMI->getOpcode() == TargetOpcode::G_FRAME_INDEX) {
      FI = TmpMI->getOperand(1).getIndex();
      return true;
    }
    return false;
  }
};

inline GFrameIndexMatch m_GFrameIndex(int &FI) { return {FI}; }

// Helper for matching G_FCONSTANT
inline bind_ty<const ConstantFP *> m_GFCst(const ConstantFP *&C) { return C; }

// General helper for all the binary generic MI such as G_ADD/G_SUB etc
template <typename LHS_P, typename RHS_P, unsigned Opcode,
          bool Commutable = false, unsigned Flags = MachineInstr::NoFlags>
struct BinaryOp_match {
  LHS_P L;
  RHS_P R;
  // Optional output: when set, receives the matched instruction's flags.
  uint32_t *FlagsOut = nullptr;

  BinaryOp_match(const LHS_P &LHS, const RHS_P &RHS) : L(LHS), R(RHS) {}
  BinaryOp_match(const LHS_P &LHS, const RHS_P &RHS, MIFlagsRef FlagsOut)
      : L(LHS), R(RHS), FlagsOut(&FlagsOut.Flags) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    const MachineInstr *TmpMI;
    if (mi_match(Op, MRI, m_MInstr(TmpMI))) {
      if (TmpMI->getOpcode() == Opcode && TmpMI->getNumOperands() == 3) {
        if ((!L.match(MRI, TmpMI->getOperand(1).getReg()) ||
             !R.match(MRI, TmpMI->getOperand(2).getReg())) &&
            // NOTE: When trying the alternative operand ordering
            // with a commutative operation, it is imperative to always run
            // the LHS sub-pattern  (i.e. `L`) before the RHS sub-pattern
            // (i.e. `R`). Otherwise, m_DeferredReg/Type will not work as
            // expected.
            (!Commutable || !L.match(MRI, TmpMI->getOperand(2).getReg()) ||
             !R.match(MRI, TmpMI->getOperand(1).getReg())))
          return false;
        if ((TmpMI->getFlags() & Flags) != Flags)
          return false;
        if (FlagsOut)
          *FlagsOut = TmpMI->getFlags();
        return true;
      }
    }
    return false;
  }
};

// Helper for (commutative) binary generic MI that checks Opcode.
template <typename LHS_P, typename RHS_P, bool Commutable = false>
struct BinaryOpc_match {
  unsigned Opc;
  LHS_P L;
  RHS_P R;

  BinaryOpc_match(unsigned Opcode, const LHS_P &LHS, const RHS_P &RHS)
      : Opc(Opcode), L(LHS), R(RHS) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (mi_match(Op, MRI, m_MInstr(TmpMI))) {
      if (TmpMI->getOpcode() == Opc && TmpMI->getNumDefs() == 1 &&
          TmpMI->getNumOperands() == 3) {
        return (L.match(MRI, TmpMI->getOperand(1).getReg()) &&
                R.match(MRI, TmpMI->getOperand(2).getReg())) ||
               // NOTE: When trying the alternative operand ordering
               // with a commutative operation, it is imperative to always run
               // the LHS sub-pattern  (i.e. `L`) before the RHS sub-pattern
               // (i.e. `R`). Otherwise, m_DeferredReg/Type will not work as
               // expected.
               (Commutable && (L.match(MRI, TmpMI->getOperand(2).getReg()) &&
                               R.match(MRI, TmpMI->getOperand(1).getReg())));
      }
    }
    return false;
  }
};

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, false> m_BinOp(unsigned Opcode, const LHS &L,
                                                const RHS &R) {
  return BinaryOpc_match<LHS, RHS, false>(Opcode, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true>
m_CommutativeBinOp(unsigned Opcode, const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(Opcode, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_ADD, true>
m_GAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_ADD, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_BUILD_VECTOR, false>
m_GBuildVector(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_BUILD_VECTOR, false>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_BUILD_VECTOR_TRUNC, false>
m_GBuildVectorTrunc(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_BUILD_VECTOR_TRUNC, false>(L,
                                                                             R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_PTR_ADD, false>
m_GPtrAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_PTR_ADD, false>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_PTR_ADD, false>
m_GPtrAdd(const LHS &L, const RHS &R, MIFlagsRef Flags) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_PTR_ADD, false>(L, R, Flags);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB> m_GSub(const LHS &L,
                                                            const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB>
m_GSub(const LHS &L, const RHS &R, MIFlagsRef Flags) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SUB>(L, R, Flags);
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
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_OR, true,
                      MachineInstr::Disjoint>
m_GDisjointOr(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_OR, true,
                        MachineInstr::Disjoint>(L, R);
}

template <typename LHS, typename RHS>
inline auto m_GAddLike(const LHS &L, const RHS &R) {
  return m_any_of(m_GAdd(L, R), m_GDisjointOr(L, R));
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

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SMAX, true>
m_GSMax(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SMAX, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_SMIN, true>
m_GSMin(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_SMIN, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_UMAX, true>
m_GUMax(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_UMAX, true>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, TargetOpcode::G_UMIN, true>
m_GUMin(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, TargetOpcode::G_UMIN, true>(L, R);
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

/// Matches an op that binds a source operand and an immediate operand with
/// sub-matchers (e.g. G_SEXT_INREG, G_ASSERT_ZEXT).
template <typename SrcTy, typename ImmTy, unsigned Opcode>
struct SrcImmOp_match {
  SrcTy L;
  ImmTy Imm;

  SrcImmOp_match(const SrcTy &LHS, const ImmTy &Imm) : L(LHS), Imm(Imm) {}
  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    return mi_match(Op, MRI, m_MInstr(TmpMI)) && TmpMI->getOpcode() == Opcode &&
           L.match(MRI, TmpMI->getOperand(1).getReg()) &&
           Imm.match(TmpMI->getOperand(2).getImm());
  }
};

/// Matches any immediate operand.
struct AnyImmMatch {
  bool match(int64_t) const { return true; }
};

/// Matches a G_SEXT_INREG, binding its source and immediate width.
template <typename SrcTy>
inline SrcImmOp_match<SrcTy, AnyImmMatch, TargetOpcode::G_SEXT_INREG>
m_GSExtInReg(const SrcTy &Src) {
  return {Src, AnyImmMatch()};
}

template <typename SrcTy, typename ImmTy>
inline SrcImmOp_match<SrcTy, ImmTy, TargetOpcode::G_SEXT_INREG>
m_GSExtInReg(const SrcTy &Src, const ImmTy &Imm) {
  return {Src, Imm};
}

/// Matches a G_ASSERT_ZEXT, binding its source and immediate bit width.
template <typename SrcTy>
inline SrcImmOp_match<SrcTy, AnyImmMatch, TargetOpcode::G_ASSERT_ZEXT>
m_GAssertZext(const SrcTy &Src) {
  return {Src, AnyImmMatch()};
}

template <typename SrcTy, typename ImmTy>
inline SrcImmOp_match<SrcTy, ImmTy, TargetOpcode::G_ASSERT_ZEXT>
m_GAssertZext(const SrcTy &Src, const ImmTy &Imm) {
  return {Src, Imm};
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

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FSQRT> m_GFSqrt(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FSQRT>(Src);
}

template <typename SrcTy>
inline UnaryOp_match<SrcTy, TargetOpcode::G_FFLOOR>
m_GFFloor(const SrcTy &Src) {
  return UnaryOp_match<SrcTy, TargetOpcode::G_FFLOOR>(Src);
}

// General helper for generic MI compares, i.e. G_ICMP and G_FCMP
// TODO: Allow checking a specific predicate.
template <typename Pred_P, typename LHS_P, typename RHS_P, unsigned Opcode,
          bool Commutable = false>
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
    Register LHS = TmpMI->getOperand(2).getReg();
    Register RHS = TmpMI->getOperand(3).getReg();
    if (L.match(MRI, LHS) && R.match(MRI, RHS))
      return true;
    // NOTE: When trying the alternative operand ordering
    // with a commutative operation, it is imperative to always run
    // the LHS sub-pattern  (i.e. `L`) before the RHS sub-pattern
    // (i.e. `R`). Otherwise, m_DeferredReg/Type will not work as expected.
    if (Commutable && L.match(MRI, RHS) && R.match(MRI, LHS) &&
        P.match(MRI, CmpInst::getSwappedPredicate(TmpPred)))
      return true;
    return false;
  }
};

template <typename LHS_P, typename Test_P, unsigned Opcode>
struct ClassifyOp_match {
  LHS_P L;
  Test_P T;

  ClassifyOp_match(const LHS_P &LHS, const Test_P &Tst) : L(LHS), T(Tst) {}

  template <typename OpTy>
  bool match(const MachineRegisterInfo &MRI, OpTy &&Op) {
    MachineInstr *TmpMI;
    if (!mi_match(Op, MRI, m_MInstr(TmpMI)) || TmpMI->getOpcode() != Opcode)
      return false;

    Register LHS = TmpMI->getOperand(1).getReg();
    if (!L.match(MRI, LHS))
      return false;

    FPClassTest TmpClass =
        static_cast<FPClassTest>(TmpMI->getOperand(2).getImm());
    if (T.match(MRI, TmpClass))
      return true;

    return false;
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

/// G_ICMP matcher that also matches commuted compares.
/// E.g.
///
/// m_c_GICmp(m_Pred(...), m_GAdd(...), m_GSub(...))
///
/// Could match both of:
///
/// icmp ugt (add x, y) (sub a, b)
/// icmp ult (sub a, b) (add x, y)
template <typename Pred, typename LHS, typename RHS>
inline CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_ICMP, true>
m_c_GICmp(const Pred &P, const LHS &L, const RHS &R) {
  return CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_ICMP, true>(P, L, R);
}

/// G_FCMP matcher that also matches commuted compares.
/// E.g.
///
/// m_c_GFCmp(m_Pred(...), m_FAdd(...), m_GFMul(...))
///
/// Could match both of:
///
/// fcmp ogt (fadd x, y) (fmul a, b)
/// fcmp olt (fmul a, b) (fadd x, y)
template <typename Pred, typename LHS, typename RHS>
inline CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_FCMP, true>
m_c_GFCmp(const Pred &P, const LHS &L, const RHS &R) {
  return CompareOp_match<Pred, LHS, RHS, TargetOpcode::G_FCMP, true>(P, L, R);
}

/// Matches the register and immediate used in a fpclass test
/// G_IS_FPCLASS %val, 96
template <typename LHS, typename Test>
inline ClassifyOp_match<LHS, Test, TargetOpcode::G_IS_FPCLASS>
m_GIsFPClass(const LHS &L, const Test &T) {
  return ClassifyOp_match<LHS, Test, TargetOpcode::G_IS_FPCLASS>(L, T);
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

template <typename Src0Ty, typename Src1Ty, typename Src2Ty>
inline TernaryOp_match<Src0Ty, Src1Ty, Src2Ty, TargetOpcode::G_SELECT>
m_GISelect(const Src0Ty &Src0, const Src1Ty &Src1, const Src2Ty &Src2) {
  return TernaryOp_match<Src0Ty, Src1Ty, Src2Ty, TargetOpcode::G_SELECT>(
      Src0, Src1, Src2);
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

} // namespace MIPatternMatch
} // namespace llvm

#endif
