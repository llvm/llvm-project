//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SDNodeInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

static void reportNodeError(const SelectionDAG &DAG, const SDNode *N,
                            const Twine &Msg) {
  std::string S;
  raw_string_ostream SS(S);
  SS << "invalid node: " << Msg << '\n';
  N->printrWithDepth(SS, &DAG, 2);
  report_fatal_error(StringRef(S));
}

static void checkResultType(const SelectionDAG &DAG, const SDNode *N,
                            unsigned ResIdx, EVT ExpectedVT) {
  EVT ActualVT = N->getValueType(ResIdx);
  if (ActualVT != ExpectedVT)
    reportNodeError(
        DAG, N,
        "result #" + Twine(ResIdx) + " has invalid type; expected " +
            ExpectedVT.getEVTString() + ", got " + ActualVT.getEVTString());
}

static void checkOperandType(const SelectionDAG &DAG, const SDNode *N,
                             unsigned OpIdx, EVT ExpectedVT) {
  EVT ActualVT = N->getOperand(OpIdx).getValueType();
  if (ActualVT != ExpectedVT)
    reportNodeError(
        DAG, N,
        "operand #" + Twine(OpIdx) + " has invalid type; expected " +
            ExpectedVT.getEVTString() + ", got " + ActualVT.getEVTString());
}

namespace {

/// Similar to SDValue, but also records whether it is a result or an operand
/// of a node so we can provide more precise diagnostics.
class SDNodeValue {
  const SDNode *N;
  unsigned Idx;
  bool IsRes;

public:
  SDNodeValue(const SDNode *N, unsigned Idx, bool IsRes)
      : N(N), Idx(Idx), IsRes(IsRes) {}

  SDValue getValue() const {
    return IsRes ? SDValue(const_cast<SDNode *>(N), Idx) : N->getOperand(Idx);
  }

  EVT getValueType() const { return getValue().getValueType(); }

  friend raw_ostream &operator<<(raw_ostream &OS, const SDNodeValue &Op) {
    return OS << (Op.IsRes ? "result" : "operand") << " #" << Op.Idx;
  }
};

} // namespace

void SDNodeInfo::verifyNode(const SelectionDAG &DAG, const SDNode *N) const {
  const SDNodeDesc &Desc = getDesc(N->getOpcode());
  bool HasChain = Desc.hasProperty(SDNPHasChain);
  bool HasOutGlue = Desc.hasProperty(SDNPOutGlue);
  bool HasInGlue = Desc.hasProperty(SDNPInGlue);
  bool HasOptInGlue = Desc.hasProperty(SDNPOptInGlue);
  bool IsVariadic = Desc.hasProperty(SDNPVariadic);

  unsigned ActualNumResults = N->getNumValues();
  unsigned ExpectedNumResults = Desc.NumResults + HasChain + HasOutGlue;

  if (ActualNumResults != ExpectedNumResults)
    reportNodeError(DAG, N,
                    "invalid number of results; expected " +
                        Twine(ExpectedNumResults) + ", got " +
                        Twine(ActualNumResults));

  // Chain result comes after all normal results.
  if (HasChain) {
    unsigned ChainResIdx = Desc.NumResults;
    checkResultType(DAG, N, ChainResIdx, MVT::Other);
  }

  // Glue result comes last.
  if (HasOutGlue) {
    unsigned GlueResIdx = Desc.NumResults + HasChain;
    checkResultType(DAG, N, GlueResIdx, MVT::Glue);
  }

  // In the most general case, the operands of a node go in the following order:
  //   chain, fix#0, ..., fix#M-1, var#0, ... var#N-1, glue
  // If the number of operands is < 0, M can be any;
  // If the node has SDNPVariadic property, N can be any.
  bool HasOptionalOperands = Desc.NumOperands < 0 || IsVariadic;

  unsigned ActualNumOperands = N->getNumOperands();
  unsigned ExpectedMinNumOperands =
      (Desc.NumOperands >= 0 ? Desc.NumOperands : 0) + HasChain + HasInGlue;

  // Check the lower bound.
  if (ActualNumOperands < ExpectedMinNumOperands) {
    StringRef How = HasOptionalOperands ? "at least " : "";
    reportNodeError(DAG, N,
                    "invalid number of operands; expected " + How +
                        Twine(ExpectedMinNumOperands) + ", got " +
                        Twine(ActualNumOperands));
  }

  // Check the upper bound. We can only do this if the number of fixed operands
  // is known and there are no variadic operands.
  if (Desc.NumOperands >= 0 && !IsVariadic) {
    // Account for optional input glue.
    unsigned ExpectedMaxNumOperands = ExpectedMinNumOperands + HasOptInGlue;
    if (ActualNumOperands > ExpectedMaxNumOperands) {
      StringRef How = HasOptInGlue ? "at most " : "";
      reportNodeError(DAG, N,
                      "invalid number of operands; expected " + How +
                          Twine(ExpectedMaxNumOperands) + ", got " +
                          Twine(ActualNumOperands));
    }
  }

  // Chain operand comes first.
  if (HasChain)
    checkOperandType(DAG, N, 0, MVT::Other);

  // Glue operand comes last.
  if (HasInGlue)
    checkOperandType(DAG, N, ActualNumOperands - 1, MVT::Glue);
  if (HasOptInGlue && ActualNumOperands >= 1 &&
      N->getOperand(ActualNumOperands - 1).getValueType() == MVT::Glue)
    HasInGlue = true;

  // Check variadic operands. These should be Register or RegisterMask.
  if (IsVariadic && Desc.NumOperands >= 0) {
    unsigned VarOpStart = HasChain + Desc.NumOperands;
    unsigned VarOpEnd = ActualNumOperands - HasInGlue;
    for (unsigned OpIdx = VarOpStart; OpIdx != VarOpEnd; ++OpIdx) {
      unsigned OpOpcode = N->getOperand(OpIdx).getOpcode();
      if (OpOpcode != ISD::Register && OpOpcode != ISD::RegisterMask)
        reportNodeError(DAG, N,
                        "variadic operand #" + Twine(OpIdx) +
                            " must be Register or RegisterMask");
    }
  }

  unsigned VTHwMode =
      DAG.getSubtarget().getHwMode(MCSubtargetInfo::HwMode_ValueType);

  // Returns a constrained or constraining value (result or operand) of a node.
  // ValIdx is the index of a node's value, as defined by SDTypeConstraint;
  // that is, it indexes a node's operands after its results and ignores
  // chain/glue values.
  auto GetConstraintValue = [&](unsigned ValIdx) {
    if (ValIdx < Desc.NumResults)
      return SDNodeValue(N, ValIdx, /*IsRes=*/true);
    return SDNodeValue(N, HasChain + (ValIdx - Desc.NumResults),
                       /*IsRes=*/false);
  };

  auto GetConstraintVT = [&](const SDTypeConstraint &C) {
    if (!C.NumHwModes)
      return static_cast<MVT::SimpleValueType>(C.VT);
    for (auto [Mode, VT] : ArrayRef(&VTByHwModeTable[C.VT], C.NumHwModes))
      if (Mode == VTHwMode)
        return VT;
    llvm_unreachable("No value type for this HW mode");
  };

  SmallString<128> ES;
  raw_svector_ostream SS(ES);

  for (const SDTypeConstraint &C : getConstraints(N->getOpcode())) {
    SDNodeValue Val = GetConstraintValue(C.ConstrainedValIdx);
    EVT VT = Val.getValueType();

    switch (C.Kind) {
    case SDTCisVT: {
      EVT ExpectedVT = GetConstraintVT(C);

      bool IsPtr = ExpectedVT == MVT::iPTR;
      if (IsPtr)
        ExpectedVT =
            DAG.getTargetLoweringInfo().getPointerTy(DAG.getDataLayout());

      if (VT != ExpectedVT) {
        SS << Val << " must have type " << ExpectedVT;
        if (IsPtr)
          SS << " (iPTR)";
        SS << ", but has type " << VT;
        reportNodeError(DAG, N, SS.str());
      }
      break;
    }
    case SDTCisPtrTy:
      break;
    case SDTCisInt:
      break;
    case SDTCisFP:
      break;
    case SDTCisVec:
      break;
    case SDTCisSameAs:
      break;
    case SDTCisVTSmallerThanOp:
      break;
    case SDTCisOpSmallerThanOp:
      break;
    case SDTCisEltOfVec:
      break;
    case SDTCisSubVecOfVec:
      break;
    case SDTCVecEltisVT: {
      EVT ExpectedVT = GetConstraintVT(C);

      if (!VT.isVector()) {
        SS << Val << " must have vector type";
        reportNodeError(DAG, N, SS.str());
      }
      if (VT.getVectorElementType() != ExpectedVT) {
        SS << Val << " must have " << ExpectedVT << " element type, but has "
           << VT.getVectorElementType() << " element type";
        reportNodeError(DAG, N, SS.str());
      }
      break;
    }
    case SDTCisSameNumEltsAs:
      break;
    case SDTCisSameSizeAs:
      break;
    }
  }
}
