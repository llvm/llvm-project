//===- ExpandMulByConstant.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Bernstein's algorithm for decomposing multiplication
// by a constant into a sequence of shifts and adds/subs.
// Reference: https://doi.org/10.1002/spe.4380160704
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <unordered_map>

using namespace llvm;

namespace {

struct MulNode {
  enum Op : uint8_t {
    Unset,
    One,
    Neg,
    ShiftAdd,
    ShiftSub,
    ShiftRev,
    FactorAdd,
    FactorSub,
    FactorRev
  };
  Op Opcode = Unset;
  int64_t Parent = 0;
  int Cost = 1;
  bool Valid = false;
};

class MulDecomposer {
public:
  explicit MulDecomposer(function_ref<unsigned(unsigned)> ShlAddCost)
      : ShlAddCost(ShlAddCost) {
    auto &One = Nodes[1];
    One.Opcode = MulNode::One;
    One.Cost = 0;
    One.Valid = true;

    auto &Neg = Nodes[-1];
    Neg.Opcode = MulNode::Neg;
    Neg.Cost = 1;
    Neg.Valid = true;
  }

  int computeCost(int64_t N);
  SDValue build(SelectionDAG &DAG, const SDLoc &DL, EVT VT, SDValue X,
                int64_t N);

private:
  function_ref<unsigned(unsigned)> ShlAddCost;
  // DenseMap would invalidate references on rehash
  std::unordered_map<int64_t, MulNode> Nodes;

  static int64_t makeOdd(int64_t N) {
    if (N == 0)
      return 0;
    assert(N != INT64_MIN && "makeOdd cannot handle INT64_MIN");
    return N >> llvm::countr_zero(static_cast<uint64_t>(N));
  }

  static unsigned estimate(int64_t N) {
    assert(N != 0 && "estimate does not handle zero");
    uint64_t Abs = N < 0 ? static_cast<uint64_t>(-N) : static_cast<uint64_t>(N);
    unsigned C = 2 * llvm::popcount(Abs) - 1;
    if (N < 0)
      C++;
    return C;
  }

  static unsigned shiftBetween(int64_t From, int64_t To) {
    assert(From != 0 && "shiftBetween called with From==0");
    unsigned S = llvm::countr_zero(static_cast<uint64_t>(To)) -
                 llvm::countr_zero(static_cast<uint64_t>(From));
    assert((static_cast<uint64_t>(From) << S) == static_cast<uint64_t>(To) &&
           "From is not a right-shift of To");
    return S;
  }

  void tryOp(MulNode::Op Op, int StepCost, int &Cost, int64_t Target, int Lower,
             int &Upper, MulNode &P) {
    MulNode &Parent = Nodes[Target];
    int K = oddCost(Parent, Target, Lower + StepCost, Upper);
    bool Better = K < Cost;
    if (Better || !P.Valid) {
      if (Better || K < Upper || P.Opcode == MulNode::Unset) {
        P.Parent = Target;
        P.Opcode = Op;
        P.Cost = Parent.Cost + StepCost;
        if (Better || K < Upper)
          P.Valid = true;
      }
      Cost = K;
      Upper = K;
    }
  }

  int oddCost(MulNode &P, int64_t N, int Lower, int Upper) {
    int Cost = P.Cost + Lower;
    if (Cost >= Upper)
      return Cost;
    if (P.Valid)
      return Cost;

    if (N > 0) {
      int64_t I = 4;
      int64_t Half = N >> 1;
      while (I < Half) {
        // FactorSub: N = Q * (I-1), decompose as (Q << log2(I)) - Q.
        if (N % (I - 1) == 0)
          tryOp(MulNode::FactorSub, 2, Cost, N / (I - 1), Lower, Upper, P);
        // FactorAdd: N = Q * (I+1), decompose as (Q << log2(I)) + Q.
        if (N % (I + 1) == 0) {
          unsigned ShAmt = llvm::countr_zero(static_cast<uint64_t>(I));
          tryOp(MulNode::FactorAdd, ShlAddCost(ShAmt), Cost, N / (I + 1), Lower,
                Upper, P);
        }
        I <<= 1;
      }
      // ShiftAdd: N = (make_odd(N-1) << s) + 1.
      {
        int64_t Odd = makeOdd(N - 1);
        unsigned ShAmt = shiftBetween(Odd, N - 1);
        tryOp(MulNode::ShiftAdd, ShlAddCost(ShAmt), Cost, Odd, Lower, Upper, P);
      }
    } else {
      int64_t I = 4;
      int64_t Half = (-N) >> 1;
      while (I < Half) {
        if (N % (1 - I) == 0)
          tryOp(MulNode::FactorRev, 2, Cost, N / (1 - I), Lower, Upper, P);
        if (N % (I + 1) == 0) {
          unsigned ShAmt = llvm::countr_zero(static_cast<uint64_t>(I));
          tryOp(MulNode::FactorAdd, ShlAddCost(ShAmt), Cost, N / (I + 1), Lower,
                Upper, P);
        }
        I <<= 1;
      }
      tryOp(MulNode::ShiftRev, 2, Cost, makeOdd(1 - N), Lower, Upper, P);
    }
    tryOp(MulNode::ShiftSub, 2, Cost, makeOdd(N + 1), Lower, Upper, P);
    return Cost;
  }

  SDValue emitOdd(SelectionDAG &DAG, const SDLoc &DL, EVT VT, SDValue X,
                  int64_t N);
  SDValue emitEven(SelectionDAG &DAG, const SDLoc &DL, EVT VT, SDValue X,
                   int64_t N);

  SDValue shl(SelectionDAG &DAG, const SDLoc &DL, EVT VT, SDValue V,
              unsigned S) {
    if (S == 0)
      return V;
    return DAG.getNode(ISD::SHL, DL, VT, V,
                       DAG.getShiftAmountConstant(S, VT, DL));
  }
};

int MulDecomposer::computeCost(int64_t N) {
  if (N == 0)
    return 0;

  if (N & 1) {
    MulNode &P = Nodes[N];
    return oddCost(P, N, 0, estimate(N));
  }

  // Even: same three-way split as emitEven.
  MulNode &P1 = Nodes[N + 1];
  int C1 = oddCost(P1, N + 1, 0, estimate(N + 1));

  int64_t Odd = makeOdd(N);
  MulNode &P2 = Nodes[Odd];
  int C2 = oddCost(P2, Odd, 0, C1);

  int64_t Delta = N > 0 ? N - 1 : 1 - N;
  MulNode &P3 = Nodes[Delta];
  int C3 = oddCost(P3, Delta, 0, C2);

  int Best = std::min({C1, C2, C3});
  // +1 for the extra add/sub or shift that wraps the odd kernel.
  return Best + 1;
}

SDValue MulDecomposer::build(SelectionDAG &DAG, const SDLoc &DL, EVT VT,
                             SDValue X, int64_t N) {
  if (N == 0)
    return DAG.getConstant(0, DL, VT);

  if (N & 1) {
    MulNode &P = Nodes[N];
    oddCost(P, N, 0, estimate(N));
    return emitOdd(DAG, DL, VT, X, N);
  }
  return emitEven(DAG, DL, VT, X, N);
}

SDValue MulDecomposer::emitOdd(SelectionDAG &DAG, const SDLoc &DL, EVT VT,
                               SDValue X, int64_t N) {
  MulNode &P = Nodes[N];
  switch (P.Opcode) {
  case MulNode::One:
    return X;
  case MulNode::Neg:
    return DAG.getNegative(X, DL, VT);
  case MulNode::ShiftAdd: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, N - 1);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::ADD, DL, VT, Shifted, X);
  }
  case MulNode::ShiftSub: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, N + 1);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::SUB, DL, VT, Shifted, X);
  }
  case MulNode::ShiftRev: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, 1 - N);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::SUB, DL, VT, X, Shifted);
  }
  case MulNode::FactorAdd: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, N - P.Parent);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::ADD, DL, VT, Shifted, V);
  }
  case MulNode::FactorSub: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, N + P.Parent);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::SUB, DL, VT, Shifted, V);
  }
  case MulNode::FactorRev: {
    SDValue V = emitOdd(DAG, DL, VT, X, P.Parent);
    unsigned S = shiftBetween(P.Parent, P.Parent - N);
    SDValue Shifted = shl(DAG, DL, VT, V, S);
    return DAG.getNode(ISD::SUB, DL, VT, V, Shifted);
  }
  default:
    llvm_unreachable("unset MulNode op");
  }
}

SDValue MulDecomposer::emitEven(SelectionDAG &DAG, const SDLoc &DL, EVT VT,
                                SDValue X, int64_t N) {
  // Pick the cheapest:
  //   1) N+1 is odd: build (N+1)*X then subtract X.
  //   2) Factor out trailing zeros: build make_odd(N)*X then shift.
  //   3) N-1 (or 1-N) is odd: build that, then add X.
  MulNode &P1 = Nodes[N + 1];
  int C1 = oddCost(P1, N + 1, 0, estimate(N + 1));

  int64_t Odd = makeOdd(N);
  MulNode &P2 = Nodes[Odd];
  int C2 = oddCost(P2, Odd, 0, C1);

  int64_t Delta = N > 0 ? N - 1 : 1 - N;
  MulNode &P3 = Nodes[Delta];
  int C3 = oddCost(P3, Delta, 0, C2);

  int C23 = std::min(C2, C3);
  if (C1 <= C23) {
    SDValue V = emitOdd(DAG, DL, VT, X, N + 1);
    return DAG.getNode(ISD::SUB, DL, VT, V, X);
  }
  if (C2 <= C3) {
    SDValue V = emitOdd(DAG, DL, VT, X, Odd);
    unsigned S = shiftBetween(Odd, N);
    return shl(DAG, DL, VT, V, S);
  }
  SDValue V = emitOdd(DAG, DL, VT, X, Delta);
  if (N > 0)
    return DAG.getNode(ISD::ADD, DL, VT, V, X);
  return DAG.getNode(ISD::SUB, DL, VT, X, V);
}

} // end anonymous namespace

SDValue TargetLowering::buildMulByConstant(
    SDNode *N, SelectionDAG &DAG, const APInt &MulAmt, unsigned *InstrCount,
    function_ref<unsigned(unsigned)> ShlAddCost) const {
  if (!MulAmt.isSignedIntN(64))
    return SDValue();

  int64_t Val = MulAmt.getSExtValue();
  // INT64_MIN cannot be handled (negation/makeOdd would overflow).
  if (Val == INT64_MIN)
    return SDValue();

  MulDecomposer D(ShlAddCost);
  if (InstrCount)
    *InstrCount = D.computeCost(Val);
  return D.build(DAG, SDLoc(N), N->getValueType(0), N->getOperand(0), Val);
}
