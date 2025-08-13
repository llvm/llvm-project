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
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
namespace SDPatternMatch {

/// MatchContext can repurpose existing patterns to behave differently under
/// a certain context. For instance, `m_Opc(ISD::ADD)` matches plain ADD nodes
/// in normal circumstances, but matches VP_ADD nodes under a custom
/// VPMatchContext. This design is meant to facilitate code / pattern reusing.
class BasicMatchContext {
  const SelectionDAG *DAG;
  const TargetLowering *TLI;

public:
  explicit BasicMatchContext(const SelectionDAG *DAG)
      : DAG(DAG), TLI(DAG ? &DAG->getTargetLoweringInfo() : nullptr) {}

  explicit BasicMatchContext(const TargetLowering *TLI)
      : DAG(nullptr), TLI(TLI) {}

  // A valid MatchContext has to implement the following functions.

  const SelectionDAG *getDAG() const { return DAG; }

  const TargetLowering *getTLI() const { return TLI; }

  /// Return true if N effectively has opcode Opcode.
  bool match(SDValue N, unsigned Opcode) const {
    return N->getOpcode() == Opcode;
  }

  unsigned getNumOperands(SDValue N) const { return N->getNumOperands(); }
};

template <typename Pattern, typename MatchContext>
[[nodiscard]] bool sd_context_match(SDValue N, const MatchContext &Ctx,
                                    Pattern &&P) {
  return P.match(Ctx, N);
}

template <typename Pattern, typename MatchContext>
[[nodiscard]] bool sd_context_match(SDNode *N, const MatchContext &Ctx,
                                    Pattern &&P) {
  return sd_context_match(SDValue(N, 0), Ctx, P);
}

template <typename Pattern>
[[nodiscard]] bool sd_match(SDNode *N, const SelectionDAG *DAG, Pattern &&P) {
  return sd_context_match(N, BasicMatchContext(DAG), P);
}

template <typename Pattern>
[[nodiscard]] bool sd_match(SDValue N, const SelectionDAG *DAG, Pattern &&P) {
  return sd_context_match(N, BasicMatchContext(DAG), P);
}

template <typename Pattern>
[[nodiscard]] bool sd_match(SDNode *N, Pattern &&P) {
  return sd_match(N, nullptr, P);
}

template <typename Pattern>
[[nodiscard]] bool sd_match(SDValue N, Pattern &&P) {
  return sd_match(N, nullptr, P);
}

// === Utilities ===
struct Value_match {
  SDValue MatchVal;

  Value_match() = default;

  explicit Value_match(SDValue Match) : MatchVal(Match) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
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

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return N.getResNo() == ResNo && P.match(Ctx, N);
  }
};

/// Match only if the SDValue is a certain result at ResNo.
template <unsigned ResNo, typename Pattern>
inline Result_match<ResNo, Pattern> m_Result(const Pattern &P) {
  return Result_match<ResNo, Pattern>(P);
}

struct DeferredValue_match {
  SDValue &MatchVal;

  explicit DeferredValue_match(SDValue &Match) : MatchVal(Match) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    return N == MatchVal;
  }
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

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return Ctx.match(N, Opcode);
  }
};

inline Opcode_match m_Opc(unsigned Opcode) { return Opcode_match(Opcode); }

inline Opcode_match m_Undef() { return Opcode_match(ISD::UNDEF); }

inline Opcode_match m_Poison() { return Opcode_match(ISD::POISON); }

template <unsigned NumUses, typename Pattern> struct NUses_match {
  Pattern P;

  explicit NUses_match(const Pattern &P) : P(P) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    // SDNode::hasNUsesOfValue is pretty expensive when the SDNode produces
    // multiple results, hence we check the subsequent pattern here before
    // checking the number of value users.
    return P.match(Ctx, N) && N->hasNUsesOfValue(NumUses, N.getResNo());
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

struct Value_bind {
  SDValue &BindVal;

  explicit Value_bind(SDValue &N) : BindVal(N) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    BindVal = N;
    return true;
  }
};

inline Value_bind m_Value(SDValue &N) { return Value_bind(N); }

template <typename Pattern, typename PredFuncT> struct TLI_pred_match {
  Pattern P;
  PredFuncT PredFunc;

  TLI_pred_match(const PredFuncT &Pred, const Pattern &P)
      : P(P), PredFunc(Pred) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    assert(Ctx.getTLI() && "TargetLowering is required for this pattern.");
    return PredFunc(*Ctx.getTLI(), N) && P.match(Ctx, N);
  }
};

// Explicit deduction guide.
template <typename PredFuncT, typename Pattern>
TLI_pred_match(const PredFuncT &Pred, const Pattern &P)
    -> TLI_pred_match<Pattern, PredFuncT>;

/// Match legal SDNodes based on the information provided by TargetLowering.
template <typename Pattern> inline auto m_LegalOp(const Pattern &P) {
  return TLI_pred_match{[](const TargetLowering &TLI, SDValue N) {
                          return TLI.isOperationLegal(N->getOpcode(),
                                                      N.getValueType());
                        },
                        P};
}

/// Switch to a different MatchContext for subsequent patterns.
template <typename NewMatchContext, typename Pattern> struct SwitchContext {
  const NewMatchContext &Ctx;
  Pattern P;

  template <typename OrigMatchContext>
  bool match(const OrigMatchContext &, SDValue N) {
    return P.match(Ctx, N);
  }
};

template <typename MatchContext, typename Pattern>
inline SwitchContext<MatchContext, Pattern> m_Context(const MatchContext &Ctx,
                                                      Pattern &&P) {
  return SwitchContext<MatchContext, Pattern>{Ctx, std::move(P)};
}

// === Value type ===
struct ValueType_bind {
  EVT &BindVT;

  explicit ValueType_bind(EVT &Bind) : BindVT(Bind) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    BindVT = N.getValueType();
    return true;
  }
};

/// Retreive the ValueType of the current SDValue.
inline ValueType_bind m_VT(EVT &VT) { return ValueType_bind(VT); }

template <typename Pattern, typename PredFuncT> struct ValueType_match {
  PredFuncT PredFunc;
  Pattern P;

  ValueType_match(const PredFuncT &Pred, const Pattern &P)
      : PredFunc(Pred), P(P) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return PredFunc(N.getValueType()) && P.match(Ctx, N);
  }
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
template <typename Pattern> inline auto m_LegalType(const Pattern &P) {
  return TLI_pred_match{[](const TargetLowering &TLI, SDValue N) {
                          return TLI.isTypeLegal(N.getValueType());
                        },
                        P};
}

// === Patterns combinators ===
template <typename... Preds> struct And {
  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    return true;
  }
};

template <typename Pred, typename... Preds>
struct And<Pred, Preds...> : And<Preds...> {
  Pred P;
  And(const Pred &p, const Preds &...preds) : And<Preds...>(preds...), P(p) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return P.match(Ctx, N) && And<Preds...>::match(Ctx, N);
  }
};

template <typename... Preds> struct Or {
  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    return false;
  }
};

template <typename Pred, typename... Preds>
struct Or<Pred, Preds...> : Or<Preds...> {
  Pred P;
  Or(const Pred &p, const Preds &...preds) : Or<Preds...>(preds...), P(p) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return P.match(Ctx, N) || Or<Preds...>::match(Ctx, N);
  }
};

template <typename Pred> struct Not {
  Pred P;

  explicit Not(const Pred &P) : P(P) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    return !P.match(Ctx, N);
  }
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

// === Generic node matching ===
template <unsigned OpIdx, typename... OpndPreds> struct Operands_match {
  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    // Returns false if there are more operands than predicates;
    // Ignores the last two operands if both the Context and the Node are VP
    return Ctx.getNumOperands(N) == OpIdx;
  }
};

template <unsigned OpIdx, typename OpndPred, typename... OpndPreds>
struct Operands_match<OpIdx, OpndPred, OpndPreds...>
    : Operands_match<OpIdx + 1, OpndPreds...> {
  OpndPred P;

  Operands_match(const OpndPred &p, const OpndPreds &...preds)
      : Operands_match<OpIdx + 1, OpndPreds...>(preds...), P(p) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    if (OpIdx < N->getNumOperands())
      return P.match(Ctx, N->getOperand(OpIdx)) &&
             Operands_match<OpIdx + 1, OpndPreds...>::match(Ctx, N);

    // This is the case where there are more predicates than operands.
    return false;
  }
};

template <typename... OpndPreds>
auto m_Node(unsigned Opcode, const OpndPreds &...preds) {
  return m_AllOf(m_Opc(Opcode), Operands_match<0, OpndPreds...>(preds...));
}

/// Provide number of operands that are not chain or glue, as well as the first
/// index of such operand.
template <bool ExcludeChain> struct EffectiveOperands {
  unsigned Size = 0;
  unsigned FirstIndex = 0;

  template <typename MatchContext>
  explicit EffectiveOperands(SDValue N, const MatchContext &Ctx) {
    const unsigned TotalNumOps = Ctx.getNumOperands(N);
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

  template <typename MatchContext>
  explicit EffectiveOperands(SDValue N, const MatchContext &Ctx)
      : Size(Ctx.getNumOperands(N)) {}
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

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    if (sd_context_match(N, Ctx, m_Opc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N, Ctx);
      assert(EO.Size == 3);
      return ((Op0.match(Ctx, N->getOperand(EO.FirstIndex)) &&
               Op1.match(Ctx, N->getOperand(EO.FirstIndex + 1))) ||
              (Commutable && Op0.match(Ctx, N->getOperand(EO.FirstIndex + 1)) &&
               Op1.match(Ctx, N->getOperand(EO.FirstIndex)))) &&
             Op2.match(Ctx, N->getOperand(EO.FirstIndex + 2));
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
  std::optional<SDNodeFlags> Flags;
  BinaryOpc_match(unsigned Opc, const LHS_P &L, const RHS_P &R,
                  std::optional<SDNodeFlags> Flgs = std::nullopt)
      : Opcode(Opc), LHS(L), RHS(R), Flags(Flgs) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    if (sd_context_match(N, Ctx, m_Opc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N, Ctx);
      assert(EO.Size == 2);
      if (!((LHS.match(Ctx, N->getOperand(EO.FirstIndex)) &&
             RHS.match(Ctx, N->getOperand(EO.FirstIndex + 1))) ||
            (Commutable && LHS.match(Ctx, N->getOperand(EO.FirstIndex + 1)) &&
             RHS.match(Ctx, N->getOperand(EO.FirstIndex)))))
        return false;

      if (!Flags.has_value())
        return true;

      return (*Flags & N->getFlags()) == *Flags;
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

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    if (auto *I = dyn_cast<ShuffleVectorSDNode>(N)) {
      return Op1.match(Ctx, I->getOperand(0)) &&
             Op2.match(Ctx, I->getOperand(1)) && Mask.match(I->getMask());
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

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    auto MatchMinMax = [&](SDValue L, SDValue R, SDValue TrueValue,
                           SDValue FalseValue, ISD::CondCode CC) {
      if ((TrueValue != L || FalseValue != R) &&
          (TrueValue != R || FalseValue != L))
        return false;

      ISD::CondCode Cond =
          TrueValue == L ? CC : getSetCCInverse(CC, L.getValueType());
      if (!Pred_t::match(Cond))
        return false;

      return (LHS.match(Ctx, L) && RHS.match(Ctx, R)) ||
             (Commutable && LHS.match(Ctx, R) && RHS.match(Ctx, L));
    };

    if (sd_context_match(N, Ctx, m_Opc(ISD::SELECT)) ||
        sd_context_match(N, Ctx, m_Opc(ISD::VSELECT))) {
      EffectiveOperands<ExcludeChain> EO_SELECT(N, Ctx);
      assert(EO_SELECT.Size == 3);
      SDValue Cond = N->getOperand(EO_SELECT.FirstIndex);
      SDValue TrueValue = N->getOperand(EO_SELECT.FirstIndex + 1);
      SDValue FalseValue = N->getOperand(EO_SELECT.FirstIndex + 2);

      if (sd_context_match(Cond, Ctx, m_Opc(ISD::SETCC))) {
        EffectiveOperands<ExcludeChain> EO_SETCC(Cond, Ctx);
        assert(EO_SETCC.Size == 3);
        SDValue L = Cond->getOperand(EO_SETCC.FirstIndex);
        SDValue R = Cond->getOperand(EO_SETCC.FirstIndex + 1);
        auto *CondNode =
            cast<CondCodeSDNode>(Cond->getOperand(EO_SETCC.FirstIndex + 2));
        return MatchMinMax(L, R, TrueValue, FalseValue, CondNode->get());
      }
    }

    if (sd_context_match(N, Ctx, m_Opc(ISD::SELECT_CC))) {
      EffectiveOperands<ExcludeChain> EO_SELECT(N, Ctx);
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
                                         const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(Opc, L, R);
}
template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_c_BinOp(unsigned Opc, const LHS &L,
                                                 const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(Opc, L, R);
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
inline BinaryOpc_match<LHS, RHS, true> m_Xor(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::XOR, L, R);
}

template <typename LHS, typename RHS>
inline auto m_BitwiseLogic(const LHS &L, const RHS &R) {
  return m_AnyOf(m_And(L, R), m_Or(L, R), m_Xor(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_SMin(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::SMIN, L, R);
}

template <typename LHS, typename RHS>
inline auto m_SMinLike(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS, true>(ISD::SMIN, L, R),
                 MaxMin_match<LHS, RHS, smin_pred_ty, true>(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_SMax(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::SMAX, L, R);
}

template <typename LHS, typename RHS>
inline auto m_SMaxLike(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS, true>(ISD::SMAX, L, R),
                 MaxMin_match<LHS, RHS, smax_pred_ty, true>(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_UMin(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::UMIN, L, R);
}

template <typename LHS, typename RHS>
inline auto m_UMinLike(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS, true>(ISD::UMIN, L, R),
                 MaxMin_match<LHS, RHS, umin_pred_ty, true>(L, R));
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS, true> m_UMax(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS, true>(ISD::UMAX, L, R);
}

template <typename LHS, typename RHS>
inline auto m_UMaxLike(const LHS &L, const RHS &R) {
  return m_AnyOf(BinaryOpc_match<LHS, RHS, true>(ISD::UMAX, L, R),
                 MaxMin_match<LHS, RHS, umax_pred_ty, true>(L, R));
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
inline BinaryOpc_match<LHS, RHS> m_Rotl(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::ROTL, L, R);
}

template <typename LHS, typename RHS>
inline BinaryOpc_match<LHS, RHS> m_Rotr(const LHS &L, const RHS &R) {
  return BinaryOpc_match<LHS, RHS>(ISD::ROTR, L, R);
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
  std::optional<SDNodeFlags> Flags;
  UnaryOpc_match(unsigned Opc, const Opnd_P &Op,
                 std::optional<SDNodeFlags> Flgs = std::nullopt)
      : Opcode(Opc), Opnd(Op), Flags(Flgs) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    if (sd_context_match(N, Ctx, m_Opc(Opcode))) {
      EffectiveOperands<ExcludeChain> EO(N, Ctx);
      assert(EO.Size == 1);
      if (!Opnd.match(Ctx, N->getOperand(EO.FirstIndex)))
        return false;
      if (!Flags.has_value())
        return true;

      return (*Flags & N->getFlags()) == *Flags;
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

template <typename Opnd> inline UnaryOpc_match<Opnd> m_Abs(const Opnd &Op) {
  return UnaryOpc_match<Opnd>(ISD::ABS, Op);
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

// === Constants ===
struct ConstantInt_match {
  APInt *BindVal;

  explicit ConstantInt_match(APInt *V) : BindVal(V) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
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
/// Match any integer constants or splat of an integer constant.
inline ConstantInt_match m_ConstInt() { return ConstantInt_match(nullptr); }
/// Match any integer constants or splat of an integer constant; return the
/// specific constant or constant splat value.
inline ConstantInt_match m_ConstInt(APInt &V) { return ConstantInt_match(&V); }

struct SpecificInt_match {
  APInt IntVal;

  explicit SpecificInt_match(APInt APV) : IntVal(std::move(APV)) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    APInt ConstInt;
    if (sd_context_match(N, Ctx, m_ConstInt(ConstInt)))
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

struct Zero_match {
  bool AllowUndefs;

  explicit Zero_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  template <typename MatchContext>
  bool match(const MatchContext &, SDValue N) const {
    return isZeroOrZeroSplat(N, AllowUndefs);
  }
};

struct Ones_match {
  bool AllowUndefs;

  Ones_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    return isOnesOrOnesSplat(N, AllowUndefs);
  }
};

struct AllOnes_match {
  bool AllowUndefs;

  AllOnes_match(bool AllowUndefs) : AllowUndefs(AllowUndefs) {}

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
    return isAllOnesOrAllOnesSplat(N, AllowUndefs);
  }
};

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
inline auto m_True() {
  return TLI_pred_match{
      [](const TargetLowering &TLI, SDValue N) {
        APInt ConstVal;
        if (sd_match(N, m_ConstInt(ConstVal)))
          switch (TLI.getBooleanContents(N.getValueType())) {
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
inline auto m_False() {
  return TLI_pred_match{
      [](const TargetLowering &TLI, SDValue N) {
        APInt ConstVal;
        if (sd_match(N, m_ConstInt(ConstVal)))
          switch (TLI.getBooleanContents(N.getValueType())) {
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

  template <typename MatchContext> bool match(const MatchContext &, SDValue N) {
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

template <typename... PatternTs> struct ReassociatableOpc_match {
  unsigned Opcode;
  std::tuple<PatternTs...> Patterns;

  ReassociatableOpc_match(unsigned Opcode, const PatternTs &...Patterns)
      : Opcode(Opcode), Patterns(Patterns...) {}

  template <typename MatchContext>
  bool match(const MatchContext &Ctx, SDValue N) {
    constexpr size_t NumPatterns = std::tuple_size_v<std::tuple<PatternTs...>>;

    SmallVector<SDValue> Leaves;
    collectLeaves(N, Leaves);
    if (Leaves.size() != NumPatterns)
      return false;

    // Matches[I][J] == true iff sd_context_match(Leaves[I], Ctx,
    // std::get<J>(Patterns)) == true
    std::array<SmallBitVector, NumPatterns> Matches;
    for (size_t I = 0; I != NumPatterns; I++) {
      std::apply(
          [&](auto &...P) {
            (Matches[I].push_back(sd_context_match(Leaves[I], Ctx, P)), ...);
          },
          Patterns);
    }

    SmallBitVector Used(NumPatterns);
    return reassociatableMatchHelper(Matches, Used);
  }

  void collectLeaves(SDValue V, SmallVector<SDValue> &Leaves) {
    if (V->getOpcode() == Opcode) {
      for (size_t I = 0, N = V->getNumOperands(); I < N; I++)
        collectLeaves(V->getOperand(I), Leaves);
    } else {
      Leaves.emplace_back(V);
    }
  }

  [[nodiscard]] inline bool
  reassociatableMatchHelper(const ArrayRef<SmallBitVector> Matches,
                            SmallBitVector &Used, size_t Curr = 0) {
    if (Curr == Matches.size())
      return true;
    for (size_t Match = 0, N = Matches[Curr].size(); Match < N; Match++) {
      if (!Matches[Curr][Match] || Used[Match])
        continue;
      Used[Match] = true;
      if (reassociatableMatchHelper(Matches, Used, Curr + 1))
        return true;
      Used[Match] = false;
    }
    return false;
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

} // namespace SDPatternMatch
} // namespace llvm
#endif
