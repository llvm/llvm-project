//==--------------- llvm/CodeGen/SDPatternMatch.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Contains matchers for matching SelectionDAG nodes and values.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDPATTERNMATCH_H
#define LLVM_CODEGEN_SDPATTERNMATCH_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/KnownBits.h"

#include <type_traits>

namespace llvm {
namespace SDPatternMatch {

template <typename Pattern>
[[nodiscard]] bool sd_match(SDValue N, Pattern &&P) {
  return P.match(N);
}

template <typename Pattern>
[[nodiscard]] bool sd_match(SDNode *N, Pattern &&P) {
  return sd_match(SDValue(N, 0), P);
}

// === Utilities ===
struct Value_match {
  SDValue MatchVal;

  Value_match() = default;

  explicit Value_match(SDValue Match) : MatchVal(Match) {}

  bool match(SDValue N) {
    if (MatchVal)
      return MatchVal == N;
    return N.getNode();
  }
};

/// Match any valid SDValue.
inline Value_match m_Value() { return Value_match(); }

inline Value_match m_Specific(SDValue N) {
  assert(N);
  return Value_match(N);
}

template <unsigned ResNo, typename Pattern> struct Result_match {
  Pattern P;

  explicit Result_match(const Pattern &P) : P(P) {}

  bool match(SDValue N) { return N.getResNo() == ResNo && P.match(N); }
};

/// Match only if the SDValue is a certain result at ResNo.
template <unsigned ResNo, typename Pattern>
inline Result_match<ResNo, Pattern> m_Result(const Pattern &P) {
  return Result_match<ResNo, Pattern>(P);
}

struct DeferredValue_match {
  SDValue &MatchVal;

  explicit DeferredValue_match(SDValue &Match) : MatchVal(Match) {}

  bool match(SDValue N) { return N == MatchVal; }
};

/// Similar to m_Specific, but the specific value to match is determined by
/// another sub-pattern in the same sd_match() expression. For instance,
/// We cannot match `(add V, V)` with `m_Add(m_Value(X), m_Specific(X))` since
/// `X` is not initialized at the time it got copied into `m_Specific`. Instead,
/// we should use `m_Add(m_Value(X), m_Deferred(X))`.
inline DeferredValue_match m_Deferred(SDValue &V) {
  return DeferredValue_match(V);
}

struct Opcode_match {
  unsigned Opcode;

  explicit Opcode_match(unsigned Opc) : Opcode(Opc) {}

  bool match(SDValue N) { return N->getOpcode() == Opcode; }
};

// === Patterns combinators ===
template <typename... Preds> struct And {
  bool match(SDValue N) { return true; }
};

template <typename Pred, typename... Preds>
struct And<Pred, Preds...> : And<Preds...> {
  Pred P;
  And(const Pred &p, const Preds &...preds) : And<Preds...>(preds...), P(p) {}

  bool match(SDValue N) { return P.match(N) && And<Preds...>::match(N); }
};

template <typename... Preds> struct Or {
  bool match(SDValue N) { return false; }
};

template <typename Pred, typename... Preds>
struct Or<Pred, Preds...> : Or<Preds...> {
  Pred P;
  Or(const Pred &p, const Preds &...preds) : Or<Preds...>(preds...), P(p) {}

  bool match(SDValue N) { return P.match(N) || Or<Preds...>::match(N); }
};

template <typename Pred> struct Not {
  Pred P;

  explicit Not(const Pred &P) : P(P) {}

  bool match(SDValue N) { return !P.match(N); }
};
// Explicit deduction guide.
template <typename Pred> Not(const Pred &P) -> Not<Pred>;

/// Match if the inner pattern does NOT match.
template <typename Pred> inline Not<Pred> m_Unless(const Pred &P) {
  return Not{P};
}

template <typename... Preds> And<Preds...> m_AllOf(const Preds &...preds) {
  return And<Preds...>(preds...);
}

template <typename... Preds> Or<Preds...> m_AnyOf(const Preds &...preds) {
  return Or<Preds...>(preds...);
}

template <typename... Preds> auto m_NoneOf(const Preds &...preds) {
  return m_Unless(m_AnyOf(preds...));
}

inline Opcode_match m_SpecificOpc(unsigned Opcode) {
  return Opcode_match(Opcode);
}

inline auto m_Undef() {
  return m_AnyOf(Opcode_match(ISD::UNDEF), Opcode_match(ISD::POISON));
}

inline Opcode_match m_Poison() { return Opcode_match(ISD::POISON); }

template <unsigned NumUses, typename Pattern> struct NUses_match {
  Pattern P;

  explicit NUses_match(const Pattern &P) : P(P) {}

  bool match(SDValue N) {
    // SDNode::hasNUsesOfValue is pretty expensive when the SDNode produces
    // multiple results, hence we check the subsequent pattern here before
    // checking the number of value users.
    return P.match(N) && N->hasNUsesOfValue(NumUses, N.getResNo());
  }
};

template <typename Pattern>
inline NUses_match<1, Pattern> m_OneUse(const Pattern &P) {
  return NUses_match<1, Pattern>(P);
}
template <unsigned N, typename Pattern>
inline NUses_match<N, Pattern> m_NUses(const Pattern &P) {
  return NUses_match<N, Pattern>(P);
}

inline NUses_match<1, Value_match> m_OneUse() {
  return NUses_match<1, Value_match>(m_Value());
}
template <unsigned N> inline NUses_match<N, Value_match> m_NUses() {
  return NUses_match<N, Value_match>(m_Value());
}

template <typename PredPattern> struct Value_bind {
  SDValue &BindVal;
  PredPattern Pred;

  Value_bind(SDValue &N, const PredPattern &P) : BindVal(N), Pred(P) {}

  bool match(SDValue N) {
    if (!Pred.match(N))
      return false;

    BindVal = N;
    return true;
  }
};

inline auto m_Value(SDValue &N) {
  return Value_bind<Value_match>(N, m_Value());
}
/// Conditionally bind an SDValue based on the predicate.
template <typename PredPattern>
inline auto m_Value(SDValue &N, const PredPattern &P) {
  return Value_bind<PredPattern>(N, P);
}

template <typename Pattern, typename PredFuncT> struct TLI_pred_match {
  Pattern P;
  PredFuncT PredFunc;

  TLI_pred_match(const PredFuncT &Pred, const Pattern &P)
      : P(P), PredFunc(Pred) {}

  bool match(SDValue N) { return PredFunc(N) && P.match(N); }
};

// Explicit deduction guide.
template <typename PredFuncT, typename Pattern>
TLI_pred_match(const PredFuncT &Pred, const Pattern &P)
    -> TLI_pred_match<Pattern, PredFuncT>;

/// Match legal SDNodes based on the information provided by TargetLowering.
template <typename Pattern>
inline auto m_LegalOp(const SelectionDAG &DAG, const Pattern &P) {
  return TLI_pred_match{[&DAG](SDValue N) {
                          return DAG.getTargetLoweringInfo().isOperationLegal(
                              N->getOpcode(), N.getValueType());
                        },
                        P};
}

// === Value type ===

template <typename Pattern> struct ValueType_bind {
  EVT &BindVT;
  Pattern P;

  explicit ValueType_bind(EVT &Bind, const Pattern &P) : BindVT(Bind), P(P) {}

  bool match(SDValue N) {
    BindVT = N.getValueType();
    return P.match(N);
  }
};

template <typename Pattern>
ValueType_bind(const Pattern &P) -> ValueType_bind<Pattern>;

/// Retreive the ValueType of the current SDValue.
inline auto m_VT(EVT &VT) { return ValueType_bind(VT, m_Value()); }

template <typename Pattern> inline auto m_VT(EVT &VT, const Pattern &P) {
  return ValueType_bind(VT, P);
}

template <typename Pattern, typename PredFuncT> struct ValueType_match {
  PredFuncT PredFunc;
  Pattern P;

  ValueType_match(const PredFuncT &Pred, const Pattern &P)
      : PredFunc(Pred), P(P) {}

  bool match(SDValue N) { return PredFunc(N.getValueType()) && P.match(N); }
};

// Explicit deduction guide.
template <typename PredFuncT, typename Pattern>
ValueType_match(const PredFuncT &Pred, const Pattern &P)
    -> ValueType_match<Pattern, PredFuncT>;

/// Match a specific ValueType.
template <typename Pattern>
inline auto m_SpecificVT(EVT RefVT, const Pattern &P) {
  return ValueType_match{[=](EVT VT) { return VT == RefVT; }, P};
}
inline auto m_SpecificVT(EVT RefVT) {
  return ValueType_match{[=](EVT VT) { return VT == RefVT; }, m_Value()};
}

inline auto m_Glue() { return m_SpecificVT(MVT::Glue); }
inline auto m_OtherVT() { return m_SpecificVT(MVT::Other); }

/// Match a scalar ValueType.
template <typename Pattern>
inline auto m_SpecificScalarVT(EVT RefVT, const Pattern &P) {
  return ValueType_match{[=](EVT VT) { return VT.getScalarType() == RefVT; },
                         P};
}
inline auto m_SpecificScalarVT(EVT RefVT) {
  return ValueType_match{[=](EVT VT) { return VT.getScalarType() == RefVT; },
                         m_Value()};
}

/// Match a vector ValueType.
template <typename Pattern>
inline auto m_SpecificVectorElementVT(EVT RefVT, const Pattern &P) {
  return ValueType_match{[=](EVT VT) {
                           return VT.isVector() &&
                                  VT.getVectorElementType() == RefVT;
                         },
                         P};
}
inline auto m_SpecificVectorElementVT(EVT RefVT) {
  return ValueType_match{[=](EVT VT) {
                           return VT.isVector() &&
                                  VT.getVectorElementType() == RefVT;
                         },
                         m_Value()};
}

/// Match any integer ValueTypes.
template <typename Pattern> inline auto m_IntegerVT(const Pattern &P) {
  return ValueType_match{[](EVT VT) { return VT.isInteger(); }, P};
}
inline auto m_IntegerVT() {
  return ValueType_match{[](EVT VT) { return VT.isInteger(); }, m_Value()};
}

/// Match any floating point ValueTypes.
template <typename Pattern> inline auto m_FloatingPointVT(const Pattern &P) {
  return ValueType_match{[](EVT VT) { return VT.isFloatingPoint(); }, P};
}
inline auto m_FloatingPointVT() {
  return ValueType_match{[](EVT VT) { return VT.isFloatingPoint(); },
                         m_Value()};
}

/// Match any vector ValueTypes.
template <typename Pattern> inline auto m_VectorVT(const Pattern &P) {
  return ValueType_match{[](EVT VT) { return VT.isVector(); }, P};
}
inline auto m_VectorVT() {
  return ValueType_match{[](EVT VT) { return VT.isVector(); }, m_Value()};
}

/// Match fixed-length vector ValueTypes.
template <typename Pattern> inline auto m_FixedVectorVT(const Pattern &P) {
  return ValueType_match{[](EVT VT) { return VT.isFixedLengthVector(); }, P};
}
inline auto m_FixedVectorVT() {
  return ValueType_match{[](EVT VT) { return VT.isFixedLengthVector(); },
                         m_Value()};
}

/// Match scalable vector ValueTypes.
template <typename Pattern> inline auto m_ScalableVectorVT(const Pattern &P) {
  return ValueType_match{[](EVT VT) { return VT.isScalableVector(); }, P};
}
inline auto m_ScalableVectorVT() {
  return ValueType_match{[](EVT VT) { return VT.isScalableVector(); },
                         m_Value()};
}

/// Match legal ValueTypes based on the information provided by TargetLowering.
template <typename Pattern>
inline auto m_LegalType(const SelectionDAG &DAG, const Pattern &P) {
  return TLI_pred_match{[&DAG](SDValue N) {
                          return DAG.getTargetLoweringInfo().isTypeLegal(
                              N.getValueType());
                        },
                        P};
}

// === Generic node matching ===
template <unsigned OpIdx, typename... OpndPreds> struct Operands_match {
  bool match(SDValue N) {
    // Returns false if there are more operands than predicates;
    return N->getNumOperands() == OpIdx;
  }
};

template <unsigned OpIdx, typename OpndPred, typename... OpndPreds>
struct Operands_match<OpIdx, OpndPred, OpndPreds...>
    : Operands_match<OpIdx + 1, OpndPreds...> {
  OpndPred P;

  Operands_match(const OpndPred &p, const OpndPreds &...preds)
      : Operands_match<OpIdx + 1, OpndPreds...>(preds...), P(p) {}

  bool match(SDValue N) {
    if (OpIdx < N->getNumOperands())
      return P.match(N->getOperand(OpIdx)) &&
             Operands_match<OpIdx + 1, OpndPreds...>::match(N);

    // This is the case where there are more predicates than operands.
    return false;
  }
};

template <typename... OpndPreds>
auto m_Node(unsigned Opcode, const OpndPreds &...preds) {
  return m_AllOf(m_SpecificOpc(Opcode),
                 Operands_match<0, OpndPreds...>(preds...));
}

/// Provide number of operands that are not chain or glue, as well as the first
/// index of such operand.
template <bool ExcludeChain> struct EffectiveOperands {
  unsigned Size = 0;
  unsigned FirstIndex = 0;

  explicit EffectiveOperands(SDValue N) {
    const unsigned TotalNumOps = N->getNumOperands();
    FirstIndex = TotalNumOps;
    for (unsigned I = 0; I < TotalNumOps; ++I) {
      // Count the number of non-chain and non-glue nodes (we ignore chain
      // and glue by default) and retreive the operand index offset.
      EVT VT = N->getOperand(I).getValueType();
      if (VT != MVT::Glue && VT != MVT::Other) {
        ++Size;
        if (FirstIndex == TotalNumOps)
          FirstIndex = I;
      }
    }
  }
};

template <> struct EffectiveOperands<false> {
  unsigned Size = 0;
  unsigned FirstIndex = 0;

  explicit EffectiveOperands(SDValue N) : Size(N->getNumOperands()) {}
};

// === Ternary operations ===
template <typename T0_P, typename T1_P, typename T2_P, bool Commutable = false,
          bool ExcludeChain = false>
struct TernaryOpc_match {
  unsigned Opcode;
  T0_P Op0;
  T1_P Op1;
  T2_P Op2;

  TernaryOpc_match(unsigned Opc, const T0_P &Op0, const T1_P &Op1,
                   const T2_P &Op2)
      : Opcode(Opc), Op0(Op0), Op1(Op1), Op2(Op2) {}

  bool match(SDValue N) {
    if (sd_match(N, m_SpecificOpc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N);
      assert(EO.Size == 3);
      return ((Op0.match(N->getOperand(EO.FirstIndex)) &&
               Op1.match(N->getOperand(EO.FirstIndex + 1))) ||
              (Commutable && Op0.match(N->getOperand(EO.FirstIndex + 1)) &&
               Op1.match(N->getOperand(EO.FirstIndex)))) &&
             Op2.match(N->getOperand(EO.FirstIndex + 2));
    }

    return false;
  }
};

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_SetCC(const T0_P &LHS, const T1_P &RHS, const T2_P &CC) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::SETCC, LHS, RHS, CC);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P, true, false>
m_c_SetCC(const T0_P &LHS, const T1_P &RHS, const T2_P &CC) {
  return TernaryOpc_match<T0_P, T1_P, T2_P, true, false>(ISD::SETCC, LHS, RHS,
                                                         CC);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_Select(const T0_P &Cond, const T1_P &T, const T2_P &F) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::SELECT, Cond, T, F);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_VSelect(const T0_P &Cond, const T1_P &T, const T2_P &F) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::VSELECT, Cond, T, F);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline auto m_SelectLike(const T0_P &Cond, const T1_P &T, const T2_P &F) {
  return m_AnyOf(m_Select(Cond, T, F), m_VSelect(Cond, T, F));
}

template <typename T0_P, typename T1_P, typename T2_P>
inline Result_match<0, TernaryOpc_match<T0_P, T1_P, T2_P>>
m_Load(const T0_P &Ch, const T1_P &Ptr, const T2_P &Offset) {
  return m_Result<0>(
      TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::LOAD, Ch, Ptr, Offset));
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_InsertElt(const T0_P &Vec, const T1_P &Val, const T2_P &Idx) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::INSERT_VECTOR_ELT, Vec, Val,
                                            Idx);
}

template <typename LHS, typename RHS, typename IDX>
inline TernaryOpc_match<LHS, RHS, IDX>
m_InsertSubvector(const LHS &Base, const RHS &Sub, const IDX &Idx) {
  return TernaryOpc_match<LHS, RHS, IDX>(ISD::INSERT_SUBVECTOR, Base, Sub, Idx);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_SpliceRight(const T0_P &V1, const T1_P &V2, const T2_P &Offset) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(ISD::VECTOR_SPLICE_RIGHT, V1, V2,
                                            Offset);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_TernaryOp(unsigned Opc, const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return TernaryOpc_match<T0_P, T1_P, T2_P>(Opc, Op0, Op1, Op2);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P, true>
m_c_TernaryOp(unsigned Opc, const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return TernaryOpc_match<T0_P, T1_P, T2_P, true>(Opc, Op0, Op1, Op2);
}

template <typename LTy, typename RTy, typename TTy, typename FTy, typename CCTy>
inline auto m_SelectCC(const LTy &L, const RTy &R, const TTy &T, const FTy &F,
                       const CCTy &CC) {
  return m_Node(ISD::SELECT_CC, L, R, T, F, CC);
}

template <typename LTy, typename RTy, typename TTy, typename FTy, typename CCTy>
inline auto m_SelectCCLike(const LTy &L, const RTy &R, const TTy &T,
                           const FTy &F, const CCTy &CC) {
  return m_AnyOf(m_Select(m_SetCC(L, R, CC), T, F), m_SelectCC(L, R, T, F, CC));
}

// === Binary operations ===
template <typename LHS_P, typename RHS_P, bool Commutable = false,
          bool ExcludeChain = false>
struct BinaryOpc_match {
  unsigned Opcode;
  LHS_P LHS;
  RHS_P RHS;
  SDNodeFlags Flags;
  BinaryOpc_match(unsigned Opc, const LHS_P &L, const RHS_P &R,
                  SDNodeFlags Flgs = SDNodeFlags())
      : Opcode(Opc), LHS(L), RHS(R), Flags(Flgs) {}

  bool match(SDValue N) {
    if (sd_match(N, m_SpecificOpc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N);
      assert(EO.Size == 2);
      if (!((LHS.match(N->getOperand(EO.FirstIndex)) &&
             RHS.match(N->getOperand(EO.FirstIndex + 1))) ||
            (Commutable && LHS.match(N->getOperand(EO.FirstIndex + 1)) &&
             RHS.match(N->getOperand(EO.FirstIndex)))))
        return false;

      return (Flags & N->getFlags()) == Flags;
    }

    return false;
  }
};

/// Matching while capturing mask
template <typename T0, typename T1, typename T2> struct SDShuffle_match {
  T0 Op1;
  T1 Op2;
  T2 Mask;

  SDShuffle_match(const T0 &Op1, const T1 &Op2, const T2 &Mask)
      : Op1(Op1), Op2(Op2), Mask(Mask) {}

  bool match(SDValue N) {
    if (auto *I = dyn_cast<ShuffleVectorSDNode>(N)) {
      return Op1.match(I->getOperand(0)) && Op2.match(I->getOperand(1)) &&
             Mask.match(I->getMask());
    }
    return false;
  }
};
struct m_Mask {
  ArrayRef<int> &MaskRef;
  m_Mask(ArrayRef<int> &MaskRef) : MaskRef(MaskRef) {}
  bool match(ArrayRef<int> Mask) {
    MaskRef = Mask;
    return true;
  }
};

struct m_SpecificMask {
  ArrayRef<int> MaskRef;
  m_SpecificMask(ArrayRef<int> MaskRef) : MaskRef(MaskRef) {}
  bool match(ArrayRef<int> Mask) { return MaskRef == Mask; }
};

template <typename LHS_P, typename RHS_P, typename Pred_t,
          bool Commutable = false, bool ExcludeChain = false>
struct MaxMin_match {
  using PredType = Pred_t;
  LHS_P LHS;
  RHS_P RHS;

  MaxMin_match(const LHS_P &L, const RHS_P &R) : LHS(L), RHS(R) {}

  bool match(SDValue N) {
    auto MatchMinMax = [&](SDValue L, SDValue R, SDValue TrueValue,
                           SDValue FalseValue, ISD::CondCode CC) {
      if ((TrueValue != L || FalseValue != R) &&
          (TrueValue != R || FalseValue != L))
        return false;

      ISD::CondCode Cond =
          TrueValue == L ? CC : getSetCCInverse(CC, L.getValueType());
      if (!Pred_t::match(Cond))
        return false;

      return (LHS.match(L) && RHS.match(R)) ||
             (Commutable && LHS.match(R) && RHS.match(L));
    };

    if (sd_match(N, m_SpecificOpc(ISD::SELECT)) ||
        sd_match(N, m_SpecificOpc(ISD::VSELECT))) {
      EffectiveOperands<ExcludeChain> EO_SELECT(N);
      assert(EO_SELECT.Size == 3);
      SDValue Cond = N->getOperand(EO_SELECT.FirstIndex);
      SDValue TrueValue = N->getOperand(EO_SELECT.FirstIndex + 1);
      SDValue FalseValue = N->getOperand(EO_SELECT.FirstIndex + 2);

      if (sd_match(Cond, m_SpecificOpc(ISD::SETCC))) {
        EffectiveOperands<ExcludeChain> EO_SETCC(Cond);
        assert(EO_SETCC.Size == 3);
        SDValue L = Cond->getOperand(EO_SETCC.FirstIndex);
        SDValue R = Cond->getOperand(EO_SETCC.FirstIndex + 1);
        auto *CondNode =
            cast<CondCodeSDNode>(Cond->getOperand(EO_SETCC.FirstIndex + 2));
        return MatchMinMax(L, R, TrueValue, FalseValue, CondNode->get());
      }
    }

    if (sd_match(N, m_SpecificOpc(ISD::SELECT_CC))) {
      EffectiveOperands<ExcludeChain> EO_SELECT(N);
      assert(EO_SELECT.Size == 5);
      SDValue L = N->getOperand(EO_SELECT.FirstIndex);
      SDValue R = N->getOperand(EO_SELECT.FirstIndex + 1);
      SDValue TrueValue = N->getOperand(EO_SELECT.FirstIndex + 2);
      SDValue FalseValue = N->getOperand(EO_SELECT.FirstIndex + 3);
      auto *CondNode =
          cast<CondCodeSDNode>(N->getOperand(EO_SELECT.FirstIndex + 4));
      return MatchMinMax(L, R, TrueValue, FalseValue, CondNode->get());
    }

    return false;
  }
};

// Helper class for identifying signed max predicates.
struct smax_pred_ty {
  static bool match(ISD::CondCode Cond) {
    return Cond == ISD::CondCode::SETGT || Cond == ISD::CondCode::SETGE;
  }
};

// Helper class for identifying unsigned max predicates.
struct umax_pred_ty {
  static bool match(ISD::CondCode Cond) {
    return Cond == ISD::CondCode::SETUGT || Cond == ISD::CondCode::SETUGE;
  }
};

// Helper class for identifying signed min predicates.
struct smin_pred_ty {
  static bool match(ISD::CondCode Cond) {
    return Cond == ISD::CondCode::SETLT || Cond == ISD::CondCode::SETLE;
  }
};

// Helper class for identifying unsigned min predicates.
struct umin_pred_ty {
  static bool match(ISD::CondCode Cond) {
    return Cond == ISD::CondCode::SETULT || Cond == ISD::CondCode::SETULE;
  }
};

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_BinOp(unsigned Opc, const LHS &L,
                                         const RHS &R,
                                         SDNodeFlags Flgs = SDNodeFlags()) {
  return BinaryOpc_match<LHS, RHS>(Opc, L, R, Flgs);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true>
m_c_BinOp(unsigned Opc, const LHS &L, const RHS &R,
          SDNodeFlags Flgs = SDNodeFlags()) {
  return BinaryOpc_match<LHS, RHS, true>(Opc, L, R, Flgs);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, false, true>
m_ChainedBinOp(unsigned Opc, const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, false, true>(Opc, L, R);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true, true>
m_c_ChainedBinOp(unsigned Opc, const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true, true>(Opc, L, R);
}

// Common binary operations
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_Add(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::ADD, L, R);
}

template <typename LHS, typename RHS>
inline auto m_NUWAdd(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::ADD, L, R,
                                         SDNodeFlags::NoUnsignedWrap);
}

template <typename LHS, typename RHS>
inline auto m_NSWAdd(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::ADD, L, R,
                                         SDNodeFlags::NoSignedWrap);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Sub(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SUB, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_Mul(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::MUL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_And(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::AND, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_Or(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::OR, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_DisjointOr(const LHS &L,
                                                    const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::OR, L, R, SDNodeFlags::Disjoint);
}

template <typename LHS, typename RHS>
inline auto m_AddLike(const LHS &L, const RHS &R) {
  return m_AnyOf(m_Add(L, R), m_DisjointOr(L, R));
}

template <typename LHS, typename RHS>
inline auto m_NSWAddLike(const LHS &L, const RHS &R) {
  return m_AnyOf(m_NSWAdd(L, R), m_DisjointOr(L, R));
}

template <typename LHS, typename RHS>
inline auto m_NUWAddLike(const LHS &L, const RHS &R) {
  return m_AnyOf(m_NUWAdd(L, R), m_DisjointOr(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_Xor(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::XOR, L, R);
}

template <typename LHS, typename RHS>
inline auto m_BitwiseLogic(const LHS &L, const RHS &R) {
  return m_AnyOf(m_And(L, R), m_Or(L, R), m_Xor(L, R));
}

template <unsigned Opc, typename Pred, typename LHS, typename RHS>
inline auto m_MaxMinLike(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS, true>(Opc, L, R),
                 MaxMin_match<LHS, RHS, Pred, true>(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_SMin(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::SMIN, L, R);
}

template <typename LHS, typename RHS>
inline auto m_SMinLike(const SelectionDAG &DAG, const LHS &L, const RHS &R) {
  return m_AnyOf(m_MaxMinLike<ISD::SMIN, smin_pred_ty>(L, R),
                 m_MaxMinLike<ISD::UMIN, umin_pred_ty>(m_NonNegative(DAG, L),
                                                       m_NonNegative(DAG, R)),
                 m_MaxMinLike<ISD::UMIN, umin_pred_ty>(m_Negative(DAG, L),
                                                       m_Negative(DAG, R)));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_SMax(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::SMAX, L, R);
}

template <typename LHS, typename RHS>
inline auto m_SMaxLike(const SelectionDAG &DAG, const LHS &L, const RHS &R) {
  return m_AnyOf(m_MaxMinLike<ISD::SMAX, smax_pred_ty>(L, R),
                 m_MaxMinLike<ISD::UMAX, umax_pred_ty>(m_NonNegative(DAG, L),
                                                       m_NonNegative(DAG, R)),
                 m_MaxMinLike<ISD::UMAX, umax_pred_ty>(m_Negative(DAG, L),
                                                       m_Negative(DAG, R)));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_UMin(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::UMIN, L, R);
}

template <typename LHS, typename RHS>
inline auto m_UMinLike(const SelectionDAG &DAG, const LHS &L, const RHS &R) {
  return m_AnyOf(m_MaxMinLike<ISD::UMIN, umin_pred_ty>(L, R),
                 m_MaxMinLike<ISD::SMIN, smin_pred_ty>(m_NonNegative(DAG, L),
                                                       m_NonNegative(DAG, R)),
                 m_MaxMinLike<ISD::SMIN, smin_pred_ty>(m_Negative(DAG, L),
                                                       m_Negative(DAG, R)));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_UMax(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::UMAX, L, R);
}

template <typename LHS, typename RHS>
inline auto m_UMaxLike(const SelectionDAG &DAG, const LHS &L, const RHS &R) {
  return m_AnyOf(m_MaxMinLike<ISD::UMAX, umax_pred_ty>(L, R),
                 m_MaxMinLike<ISD::SMAX, smax_pred_ty>(m_NonNegative(DAG, L),
                                                       m_NonNegative(DAG, R)),
                 m_MaxMinLike<ISD::SMAX, smax_pred_ty>(m_Negative(DAG, L),
                                                       m_Negative(DAG, R)));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_UDiv(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::UDIV, L, R);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_SDiv(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SDIV, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_URem(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::UREM, L, R);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_SRem(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SREM, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Shl(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SHL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Sra(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SRA, L, R);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Srl(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::SRL, L, R);
}
template <typename LHS, typename RHS>
inline auto m_ExactSr(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS>(ISD::SRA, L, R, SDNodeFlags::Exact),
                 BinaryOpc_match<LHS, RHS>(ISD::SRL, L, R, SDNodeFlags::Exact));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Rotl(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::ROTL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Rotr(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::ROTR, L, R);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_FShL(const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return m_TernaryOp(ISD::FSHL, Op0, Op1, Op2);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline TernaryOpc_match<T0_P, T1_P, T2_P>
m_FShR(const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return m_TernaryOp(ISD::FSHR, Op0, Op1, Op2);
}

template <typename T0_P, typename T1_P, typename T2_P, bool Left>
struct FunnelShiftLike_match {
  T0_P Op0;
  T1_P Op1;
  T2_P Op2;

  FunnelShiftLike_match(const T0_P &Op0, const T1_P &Op1, const T2_P &Op2)
      : Op0(Op0), Op1(Op1), Op2(Op2) {}

  static bool hasComplementaryConstantShifts(const APInt &ShlV,
                                             const APInt &SrlV,
                                             unsigned BitWidth) {
    unsigned SumWidth = std::max(ShlV.getBitWidth(), SrlV.getBitWidth()) + 1;
    unsigned BitWidthBits = llvm::bit_width(BitWidth);
    if (BitWidthBits > SumWidth)
      return false;

    return ShlV.zext(SumWidth) + SrlV.zext(SumWidth) ==
           APInt(SumWidth, BitWidth);
  }

  bool matchOperands(SDValue X, SDValue Y, SDValue Z) {
    return Op0.match(X) && Op1.match(Y) && Op2.match(Z);
  }

  bool matchShiftOr(SDValue N, unsigned BitWidth);

  bool match(SDValue N) {
    if (sd_match(N, Left ? m_FShL(Op0, Op1, Op2) : m_FShR(Op0, Op1, Op2)))
      return true;

    SDValue X, Z;
    if (sd_match(N, Left ? m_Rotl(m_Value(X), m_Value(Z))
                         : m_Rotr(m_Value(X), m_Value(Z))))
      return matchOperands(X, X, Z);

    return matchShiftOr(N, N.getValueType().getScalarSizeInBits());
  }
};

template <typename T0_P, typename T1_P, typename T2_P>
inline FunnelShiftLike_match<T0_P, T1_P, T2_P, true>
m_FShLLike(const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return FunnelShiftLike_match<T0_P, T1_P, T2_P, true>(Op0, Op1, Op2);
}

template <typename T0_P, typename T1_P, typename T2_P>
inline FunnelShiftLike_match<T0_P, T1_P, T2_P, false>
m_FShRLike(const T0_P &Op0, const T1_P &Op1, const T2_P &Op2) {
  return FunnelShiftLike_match<T0_P, T1_P, T2_P, false>(Op0, Op1, Op2);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_Clmul(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::CLMUL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_FAdd(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::FADD, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_FSub(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::FSUB, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_FMul(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::FMUL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_FDiv(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::FDIV, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_FRem(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::FREM, L, R);
}

template <typename V1_t, typename V2_t>
inline BinaryOpc_match<V1_t, V2_t> m_Shuffle(const V1_t &v1, const V2_t &v2) {
  return BinaryOpc_match<V1_t, V2_t>(ISD::VECTOR_SHUFFLE, v1, v2);
}

template <typename V1_t, typename V2_t, typename Mask_t>
inline SDShuffle_match<V1_t, V2_t, Mask_t>
m_Shuffle(const V1_t &v1, const V2_t &v2, const Mask_t &mask) {
  return SDShuffle_match<V1_t, V2_t, Mask_t>(v1, v2, mask);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_ExtractElt(const LHS &Vec, const RHS &Idx) {
  return BinaryOpc_match<LHS, RHS>(ISD::EXTRACT_VECTOR_ELT, Vec, Idx);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_ExtractSubvector(const LHS &Vec,
                                                    const RHS &Idx) {
  return BinaryOpc_match<LHS, RHS>(ISD::EXTRACT_SUBVECTOR, Vec, Idx);
}

// === Unary operations ===
template <typename Opnd_P, bool ExcludeChain = false> struct UnaryOpc_match {
  unsigned Opcode;
  Opnd_P Opnd;
  SDNodeFlags Flags;
  UnaryOpc_match(unsigned Opc, const Opnd_P &Op,
                 SDNodeFlags Flgs = SDNodeFlags())
      : Opcode(Opc), Opnd(Op), Flags(Flgs) {}

  bool match(SDValue N) {
    if (sd_match(N, m_SpecificOpc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N);
      assert(EO.Size == 1);
      if (!Opnd.match(N->getOperand(EO.FirstIndex)))
        return false;

      return (Flags & N->getFlags()) == Flags;
    }

    return false;
  }
};

template <typename Opnd>
inline UnaryOpc_match<Opnd> m_UnaryOp(unsigned Opc, const Opnd &Op) {
  return UnaryOpc_match<Opnd>(Opc, Op);
}
template <typename Opnd>
inline UnaryOpc_match<Opnd, true> m_ChainedUnaryOp(unsigned Opc,
                                                   const Opnd &Op) {
  return UnaryOpc_match<Opnd, true>(Opc, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_BitCast(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::BITCAST, Op);
}

template <typename Opnd>
inline UnaryOpc_match<Opnd> m_BSwap(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::BSWAP, Op);
}

template <typename Opnd>
inline UnaryOpc_match<Opnd> m_BitReverse(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::BITREVERSE, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_ZExt(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::ZERO_EXTEND, Op);
}

template <typename Opnd>
inline UnaryOpc_match<Opnd> m_NNegZExt(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::ZERO_EXTEND, Op, SDNodeFlags::NonNeg);
}

template <typename Opnd> inline auto m_SExt(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::SIGN_EXTEND, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_AnyExt(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::ANY_EXTEND, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_Trunc(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::TRUNCATE, Op);
}

template <typename Opnd> inline auto m_Abs(const Opnd &Op) {
  return m_AnyOf(UnaryOpc_match<Opnd>(ISD::ABS, Op),
                 UnaryOpc_match<Opnd>(ISD::ABS_MIN_POISON, Op));
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_FAbs(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::FABS, Op);
}

/// Match a zext or identity
/// Allows to peek through optional extensions
template <typename Opnd> inline auto m_ZExtOrSelf(const Opnd &Op) {
  return m_AnyOf(m_ZExt(Op), Op);
}

/// Match a sext or identity
/// Allows to peek through optional extensions
template <typename Opnd> inline auto m_SExtOrSelf(const Opnd &Op) {
  return m_AnyOf(m_SExt(Op), Op);
}

template <typename Opnd> inline auto m_SExtLike(const Opnd &Op) {
  return m_AnyOf(m_SExt(Op), m_NNegZExt(Op));
}

/// Match a aext or identity
/// Allows to peek through optional extensions
template <typename Opnd>
inline Or<UnaryOpc_match<Opnd>, Opnd> m_AExtOrSelf(const Opnd &Op) {
  return Or<UnaryOpc_match<Opnd>, Opnd>(m_AnyExt(Op), Op);
}

/// Match a trunc or identity
/// Allows to peek through optional truncations
template <typename Opnd>
inline Or<UnaryOpc_match<Opnd>, Opnd> m_TruncOrSelf(const Opnd &Op) {
  return Or<UnaryOpc_match<Opnd>, Opnd>(m_Trunc(Op), Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_VScale(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::VSCALE, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_FPToUI(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::FP_TO_UINT, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_FPToSI(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::FP_TO_SINT, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_Ctpop(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::CTPOP, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_Ctlz(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::CTLZ, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_Cttz(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::CTTZ, Op);
}

template <typename Opnd> inline UnaryOpc_match<Opnd> m_FNeg(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::FNEG, Op);
}

template <typename Opnd>
inline UnaryOpc_match<Opnd> m_VectorReverse(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::VECTOR_REVERSE, Op);
}

// === Constants ===
struct ConstantInt_match {
  APInt *BindVal;

  explicit ConstantInt_match(APInt *V) : BindVal(V) {}

  bool match(SDValue N) {
    // The logics here are similar to that in
    // SelectionDAG::isConstantIntBuildVectorOrConstantInt, but the latter also
    // treats GlobalAddressSDNode as a constant, which is difficult to turn into
    // APInt.
    if (auto *C = dyn_cast_or_null<ConstantSDNode>(N.getNode())) {
      if (BindVal)
        *BindVal = C->getAPIntValue();
      return true;
    }

    APInt Discard;
    return ISD::isConstantSplatVector(N.getNode(),
                                      BindVal ? *BindVal : Discard);
  }
};

template <typename T> struct Constant64_match {
  static_assert(sizeof(T) == 8, "T must be 64 bits wide");

  T &BindVal;

  explicit Constant64_match(T &V) : BindVal(V) {}

  bool match(SDValue N) {
    APInt V;
    if (!ConstantInt_match(&V).match(N))
      return false;

    if constexpr (std::is_signed_v<T>) {
      if (std::optional<int64_t> TrySExt = V.trySExtValue()) {
        BindVal = *TrySExt;
        return true;
      }
    }

    if constexpr (std::is_unsigned_v<T>) {
      if (std::optional<uint64_t> TryZExt = V.tryZExtValue()) {
        BindVal = *TryZExt;
        return true;
      }
    }

    return false;
  }
};

/// Match any integer constants or splat of an integer constant.
inline ConstantInt_match m_ConstInt() { return ConstantInt_match(nullptr); }
/// Match any integer constants or splat of an integer constant; return the
/// specific constant or constant splat value.
inline ConstantInt_match m_ConstInt(APInt &V) { return ConstantInt_match(&V); }
/// Match any integer constants or splat of an integer constant that can fit in
/// 64 bits; return the specific constant or constant splat value, zero-extended
/// to 64 bits.
inline Constant64_match<uint64_t> m_ConstInt(uint64_t &V) {
  return Constant64_match<uint64_t>(V);
}
/// Match any integer constants or splat of an integer constant that can fit in
/// 64 bits; return the specific constant or constant splat value, sign-extended
/// to 64 bits.
inline Constant64_match<int64_t> m_ConstInt(int64_t &V) {
  return Constant64_match<int64_t>(V);
}

template <typename T0_P, typename T1_P, typename T2_P, bool Left>
bool FunnelShiftLike_match<T0_P, T1_P, T2_P, Left>::matchShiftOr(
    SDValue N, unsigned BitWidth) {
  SDValue X, Y, ShlAmt, SrlAmt;
  APInt ShlConst, SrlConst;
  if (!sd_match(
          N, m_Or(m_Shl(m_Value(X), m_Value(ShlAmt, m_ConstInt(ShlConst))),
                  m_Srl(m_Value(Y), m_Value(SrlAmt, m_ConstInt(SrlConst))))) ||
      !hasComplementaryConstantShifts(ShlConst, SrlConst, BitWidth))
    return false;

  return matchOperands(X, Y, Left ? ShlAmt : SrlAmt);
}

struct SpecificInt_match {
  APInt IntVal;

  explicit SpecificInt_match(APInt APV) : IntVal(std::move(APV)) {}

  bool match(SDValue N) {
    APInt ConstInt;
    if (sd_match(N, m_ConstInt(ConstInt)))
      return APInt::isSameValue(IntVal, ConstInt);
    return false;
  }
};

/// Match a specific integer constant or constant splat value.
inline SpecificInt_match m_SpecificInt(APInt V) {
  return SpecificInt_match(std::move(V));
}
inline SpecificInt_match m_SpecificInt(uint64_t V) {
  return SpecificInt_match(APInt(64, V));
}

struct SpecificFP_match {
  APFloat Val;

  explicit SpecificFP_match(APFloat V) : Val(V) {}

  bool match(SDValue V) {
    if (const auto *CFP = dyn_cast<ConstantFPSDNode>(V.getNode()))
      return CFP->isExactlyValue(Val);
    if (ConstantFPSDNode *C = isConstOrConstSplatFP(V, /*AllowUndefs=*/true))
      return C->getValueAPF().compare(Val) == APFloat::cmpEqual;
    return false;
  }
};

/// Match a specific float constant.
inline SpecificFP_match m_SpecificFP(APFloat V) { return SpecificFP_match(V); }

inline SpecificFP_match m_SpecificFP(double V) {
  return SpecificFP_match(APFloat(V));
}

struct AnyZeroFP_match {
  bool match(SDValue N) {
    if (ConstantFPSDNode *C = isConstOrConstSplatFP(N))
      return C->isZero();
    return false;
  }
};

/// Match a floating-point +0.0 or -0.0 constant or splat.
inline AnyZeroFP_match m_AnyZeroFP() { return AnyZeroFP_match(); }

struct Negative_match {
  const SelectionDAG &DAG;
  bool match(SDValue N) { return DAG.computeKnownBits(N).isNegative(); }
};

struct NonNegative_match {
  const SelectionDAG &DAG;
  bool match(SDValue N) { return DAG.computeKnownBits(N).isNonNegative(); }
};

struct StrictlyPositive_match {
  const SelectionDAG &DAG;
  bool match(SDValue N) { return DAG.computeKnownBits(N).isStrictlyPositive(); }
};

struct NonPositive_match {
  const SelectionDAG &DAG;
  bool match(SDValue N) { return DAG.computeKnownBits(N).isNonPositive(); }
};

struct NonZero_match {
  const SelectionDAG &DAG;
  bool match(SDValue N) { return DAG.computeKnownBits(N).isNonZero(); }
};

struct Zero_match {
  bool AllowUndefs;

  explicit Zero_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  bool match(SDValue N) const { return isZeroOrZeroSplat(N, AllowUndefs); }
};

struct Ones_match {
  bool AllowUndefs;

  Ones_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  bool match(SDValue N) { return isOnesOrOnesSplat(N, AllowUndefs); }
};

struct AllOnes_match {
  bool AllowUndefs;

  AllOnes_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  bool match(SDValue N) { return isAllOnesOrAllOnesSplat(N, AllowUndefs); }
};

inline Negative_match m_Negative(const SelectionDAG &DAG) { return {DAG}; }
template <typename Pattern>
inline auto m_Negative(const SelectionDAG &DAG, const Pattern &P) {
  return m_AllOf(m_Negative(DAG), P);
}
inline NonNegative_match m_NonNegative(const SelectionDAG &DAG) {
  return {DAG};
}
template <typename Pattern>
inline auto m_NonNegative(const SelectionDAG &DAG, const Pattern &P) {
  return m_AllOf(m_NonNegative(DAG), P);
}
inline StrictlyPositive_match m_StrictlyPositive(const SelectionDAG &DAG) {
  return {DAG};
}
template <typename Pattern>
inline auto m_StrictlyPositive(const SelectionDAG &DAG, const Pattern &P) {
  return m_AllOf(m_StrictlyPositive(DAG), P);
}
inline NonPositive_match m_NonPositive(const SelectionDAG &DAG) {
  return {DAG};
}
template <typename Pattern>
inline auto m_NonPositive(const SelectionDAG &DAG, const Pattern &P) {
  return m_AllOf(m_NonPositive(DAG), P);
}
inline NonZero_match m_NonZero(const SelectionDAG &DAG) { return {DAG}; }
template <typename Pattern>
inline auto m_NonZero(const SelectionDAG &DAG, const Pattern &P) {
  return m_AllOf(m_NonZero(DAG), P);
}
inline Ones_match m_One(bool AllowUndefs = false) {
  return Ones_match(AllowUndefs);
}
inline Zero_match m_Zero(bool AllowUndefs = false) {
  return Zero_match(AllowUndefs);
}
inline AllOnes_match m_AllOnes(bool AllowUndefs = false) {
  return AllOnes_match(AllowUndefs);
}

/// Match true boolean value based on the information provided by
/// TargetLowering.
inline auto m_True(const SelectionDAG &DAG) {
  return TLI_pred_match{
      [&DAG](SDValue N) {
        APInt ConstVal;
        if (sd_match(N, m_ConstInt(ConstVal)))
          switch (DAG.getTargetLoweringInfo().getBooleanContents(
              N.getValueType())) {
          case TargetLowering::ZeroOrOneBooleanContent:
            return ConstVal.isOne();
          case TargetLowering::ZeroOrNegativeOneBooleanContent:
            return ConstVal.isAllOnes();
          case TargetLowering::UndefinedBooleanContent:
            return (ConstVal & 0x01) == 1;
          }

        return false;
      },
      m_Value()};
}
/// Match false boolean value based on the information provided by
/// TargetLowering.
inline auto m_False(const SelectionDAG &DAG) {
  return TLI_pred_match{
      [&DAG](SDValue N) {
        APInt ConstVal;
        if (sd_match(N, m_ConstInt(ConstVal)))
          switch (DAG.getTargetLoweringInfo().getBooleanContents(
              N.getValueType())) {
          case TargetLowering::ZeroOrOneBooleanContent:
          case TargetLowering::ZeroOrNegativeOneBooleanContent:
            return ConstVal.isZero();
          case TargetLowering::UndefinedBooleanContent:
            return (ConstVal & 0x01) == 0;
          }

        return false;
      },
      m_Value()};
}

struct CondCode_match {
  std::optional<ISD::CondCode> CCToMatch;
  ISD::CondCode *BindCC = nullptr;

  explicit CondCode_match(ISD::CondCode CC) : CCToMatch(CC) {}

  explicit CondCode_match(ISD::CondCode *CC) : BindCC(CC) {}

  bool match(SDValue N) {
    if (auto *CC = dyn_cast<CondCodeSDNode>(N.getNode())) {
      if (CCToMatch && *CCToMatch != CC->get())
        return false;

      if (BindCC)
        *BindCC = CC->get();
      return true;
    }

    return false;
  }
};

/// Match any conditional code SDNode.
inline CondCode_match m_CondCode() { return CondCode_match(nullptr); }
/// Match any conditional code SDNode and return its ISD::CondCode value.
inline CondCode_match m_CondCode(ISD::CondCode &CC) {
  return CondCode_match(&CC);
}
/// Match a conditional code SDNode with a specific ISD::CondCode.
inline CondCode_match m_SpecificCondCode(ISD::CondCode CC) {
  return CondCode_match(CC);
}

/// Match a negate as a sub(0, v)
template <typename ValTy>
inline BinaryOpc_match<Zero_match, ValTy, false> m_Neg(const ValTy &V) {
  return m_Sub(m_Zero(), V);
}

/// Match a Not as a xor(v, -1) or xor(-1, v)
template <typename ValTy>
inline BinaryOpc_match<ValTy, AllOnes_match, true> m_Not(const ValTy &V) {
  return m_Xor(V, m_AllOnes());
}

template <unsigned IntrinsicId, typename... OpndPreds>
inline auto m_IntrinsicWOChain(const OpndPreds &...Opnds) {
  return m_Node(ISD::INTRINSIC_WO_CHAIN, m_SpecificInt(IntrinsicId), Opnds...);
}

struct SpecificNeg_match {
  SDValue V;

  explicit SpecificNeg_match(SDValue V) : V(V) {}

  bool match(SDValue N) {
    if (sd_match(N, m_Neg(m_Specific(V))))
      return true;

    return ISD::matchBinaryPredicate(
        V, N, [](ConstantSDNode *LHS, ConstantSDNode *RHS) {
          return LHS->getAPIntValue() == -RHS->getAPIntValue();
        });
  }
};

/// Match a negation of a specific value V, either as sub(0, V) or as
/// constant(s) that are the negation of V's constant(s).
inline SpecificNeg_match m_SpecificNeg(SDValue V) {
  return SpecificNeg_match(V);
}

template <typename... PatternTs> struct ReassociatableOpc_match {
  unsigned Opcode;
  std::tuple<PatternTs...> Patterns;
  constexpr static size_t NumPatterns =
      std::tuple_size_v<std::tuple<PatternTs...>>;

  SDNodeFlags Flags;

  ReassociatableOpc_match(unsigned Opcode, const PatternTs &...Patterns)
      : Opcode(Opcode), Patterns(Patterns...) {}

  ReassociatableOpc_match(unsigned Opcode, SDNodeFlags Flags,
                          const PatternTs &...Patterns)
      : Opcode(Opcode), Patterns(Patterns...), Flags(Flags) {}

  bool match(SDValue N) {
    std::array<SDValue, NumPatterns> Leaves;
    size_t LeavesIdx = 0;
    if (!(collectLeaves(N, Leaves, LeavesIdx) && (LeavesIdx == NumPatterns)))
      return false;

    Bitset<NumPatterns> Used;
    return std::apply(
        [&](auto &...P) -> bool {
          return reassociatableMatchHelper(Leaves, Used, P...);
        },
        Patterns);
  }

  bool collectLeaves(SDValue V, std::array<SDValue, NumPatterns> &Leaves,
                     std::size_t &LeafIdx) {
    if (V->getOpcode() == Opcode && (Flags & V->getFlags()) == Flags) {
      for (size_t I = 0, N = V->getNumOperands(); I < N; I++)
        if ((LeafIdx == NumPatterns) ||
            !collectLeaves(V->getOperand(I), Leaves, LeafIdx))
          return false;
    } else {
      Leaves[LeafIdx] = V;
      LeafIdx++;
    }
    return true;
  }

  // Searchs for a matching leaf for every sub-pattern.
  template <typename PatternHd, typename... PatternTl>
  [[nodiscard]] inline bool
  reassociatableMatchHelper(ArrayRef<SDValue> Leaves, Bitset<NumPatterns> &Used,
                            PatternHd &HeadPattern,
                            PatternTl &...TailPatterns) {
    for (size_t Match = 0, N = Used.size(); Match < N; Match++) {
      if (Used[Match] || !(sd_match(Leaves[Match], HeadPattern)))
        continue;
      Used.set(Match);
      if (reassociatableMatchHelper(Leaves, Used, TailPatterns...))
        return true;
      Used.reset(Match);
    }
    return false;
  }

  [[nodiscard]] inline bool
  reassociatableMatchHelper(ArrayRef<SDValue> Leaves,
                            Bitset<NumPatterns> &Used) {
    return true;
  }
};

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableAdd(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(ISD::ADD, Patterns...);
}

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableOr(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(ISD::OR, Patterns...);
}

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableAnd(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(ISD::AND, Patterns...);
}

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableMul(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(ISD::MUL, Patterns...);
}

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableNSWAdd(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(
      ISD::ADD, SDNodeFlags::NoSignedWrap, Patterns...);
}

template <typename... PatternTs>
inline ReassociatableOpc_match<PatternTs...>
m_ReassociatableNUWAdd(const PatternTs &...Patterns) {
  return ReassociatableOpc_match<PatternTs...>(
      ISD::ADD, SDNodeFlags::NoUnsignedWrap, Patterns...);
}

} // namespace SDPatternMatch
} // namespace llvm
#endif
