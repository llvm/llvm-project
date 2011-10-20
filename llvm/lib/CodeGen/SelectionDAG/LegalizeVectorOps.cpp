//===-- LegalizeVectorOps.cpp - Implement SelectionDAG::LegalizeVectors ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeVectors method.
//
// The vector legalizer looks for vector operations which might need to be
// scalarized and legalizes them. This is a separate step from Legalize because
// scalarizing can introduce illegal types.  For example, suppose we have an
// ISD::SDIV of type v2i64 on x86-32.  The type is legal (for example, addition
// on a v2i64 is legal), but ISD::SDIV isn't legal, so we have to unroll the
// operation, which introduces nodes with the illegal type i64 which must be
// expanded.  Similarly, suppose we have an ISD::SRA of type v16i8 on PowerPC;
// the operation must be unrolled, which introduces nodes with the illegal
// type i8 which must be promoted.
//
// This does not legalize vector manipulations like ISD::BUILD_VECTOR,
// or operations that happen to take a vector which are custom-lowered;
// the legalization for such operations never produces nodes
// with illegal types, so it's okay to put off legalizing them until
// SelectionDAG::Legalize runs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

namespace {
class VectorLegalizer {
  SelectionDAG& DAG;
  const TargetLowering &TLI;
  bool Changed; // Keep track of whether anything changed

  /// LegalizedNodes - For nodes that are of legal width, and that have more
  /// than one use, this map indicates what regularized operand to use.  This
  /// allows us to avoid legalizing the same thing more than once.
  DenseMap<SDValue, SDValue> LegalizedNodes;

  // Adds a node to the translation cache
  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }

  // Legalizes the given node
  SDValue LegalizeOp(SDValue Op);
  // Assuming the node is legal, "legalize" the results
  SDValue TranslateLegalizeResults(SDValue Op, SDValue Result);
  // Implements unrolling a VSETCC.
  SDValue UnrollVSETCC(SDValue Op);
  // Implements expansion for FNEG; falls back to UnrollVectorOp if FSUB
  // isn't legal.
  // Implements expansion for UINT_TO_FLOAT; falls back to UnrollVectorOp if
  // SINT_TO_FLOAT and SHR on vectors isn't legal.
  SDValue ExpandUINT_TO_FLOAT(SDValue Op);
  // Implement vselect in terms of XOR, AND, OR when blend is not supported
  // by the target.
  SDValue ExpandVSELECT(SDValue Op);
  SDValue ExpandFNEG(SDValue Op);
  // Implements vector promotion; this is essentially just bitcasting the
  // operands to a different type and bitcasting the result back to the
  // original type.
  SDValue PromoteVectorOp(SDValue Op);

  public:
  bool Run();
  VectorLegalizer(SelectionDAG& dag) :
      DAG(dag), TLI(dag.getTargetLoweringInfo()), Changed(false) {}
};

bool VectorLegalizer::Run() {
  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = prior(DAG.allnodes_end()); I != llvm::next(E); ++I)
    LegalizeOp(SDValue(I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  LegalizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();

  return Changed;
}

SDValue VectorLegalizer::TranslateLegalizeResults(SDValue Op, SDValue Result) {
  // Generic legalization: just pass the operand through.
  for (unsigned i = 0, e = Op.getNode()->getNumValues(); i != e; ++i)
    AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
  return Result.getValue(Op.getResNo());
}

SDValue VectorLegalizer::LegalizeOp(SDValue Op) {
  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDNode* Node = Op.getNode();

  // Legalize the operands
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
    Ops.push_back(LegalizeOp(Node->getOperand(i)));

  SDValue Result =
    SDValue(DAG.UpdateNodeOperands(Op.getNode(), Ops.data(), Ops.size()), 0);

  bool HasVectorValue = false;
  for (SDNode::value_iterator J = Node->value_begin(), E = Node->value_end();
       J != E;
       ++J)
    HasVectorValue |= J->isVector();
  if (!HasVectorValue)
    return TranslateLegalizeResults(Op, Result);

  EVT QueryType;
  switch (Op.getOpcode()) {
  default:
    return TranslateLegalizeResults(Op, Result);
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
  case ISD::SELECT:
  case ISD::VSELECT:
  case ISD::SELECT_CC:
  case ISD::SETCC:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::TRUNCATE:
  case ISD::SIGN_EXTEND:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FPOWI:
  case ISD::FPOW:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FTRUNC:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FFLOOR:
  case ISD::SIGN_EXTEND_INREG:
    QueryType = Node->getValueType(0);
    break;
  case ISD::FP_ROUND_INREG:
    QueryType = cast<VTSDNode>(Node->getOperand(1))->getVT();
    break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    QueryType = Node->getOperand(0).getValueType();
    break;
  }

  switch (TLI.getOperationAction(Node->getOpcode(), QueryType)) {
  case TargetLowering::Promote:
    // "Promote" the operation by bitcasting
    Result = PromoteVectorOp(Op);
    Changed = true;
    break;
  case TargetLowering::Legal: break;
  case TargetLowering::Custom: {
    SDValue Tmp1 = TLI.LowerOperation(Op, DAG);
    if (Tmp1.getNode()) {
      Result = Tmp1;
      break;
    }
    // FALL THROUGH
  }
  case TargetLowering::Expand:
    if (Node->getOpcode() == ISD::VSELECT)
      Result = ExpandVSELECT(Op);
    else if (Node->getOpcode() == ISD::UINT_TO_FP)
      Result = ExpandUINT_TO_FLOAT(Op);
    else if (Node->getOpcode() == ISD::FNEG)
      Result = ExpandFNEG(Op);
    else if (Node->getOpcode() == ISD::SETCC)
      Result = UnrollVSETCC(Op);
    else
      Result = DAG.UnrollVectorOp(Op.getNode());
    break;
  }

  // Make sure that the generated code is itself legal.
  if (Result != Op) {
    Result = LegalizeOp(Result);
    Changed = true;
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

SDValue VectorLegalizer::PromoteVectorOp(SDValue Op) {
  // Vector "promotion" is basically just bitcasting and doing the operation
  // in a different type.  For example, x86 promotes ISD::AND on v2i32 to
  // v1i64.
  EVT VT = Op.getValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");
  EVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  DebugLoc dl = Op.getDebugLoc();
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      Operands[j] = DAG.getNode(ISD::BITCAST, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  Op = DAG.getNode(Op.getOpcode(), dl, NVT, &Operands[0], Operands.size());

  return DAG.getNode(ISD::BITCAST, dl, VT, Op);
}

SDValue VectorLegalizer::ExpandVSELECT(SDValue Op) {
  // Implement VSELECT in terms of XOR, AND, OR
  // on platforms which do not support blend natively.
  EVT VT =  Op.getOperand(0).getValueType();
  DebugLoc DL = Op.getDebugLoc();

  SDValue Mask = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);

  // If we can't even use the basic vector operations of
  // AND,OR,XOR, we will have to scalarize the op.
  if (!TLI.isOperationLegalOrCustom(ISD::AND, VT) ||
      !TLI.isOperationLegalOrCustom(ISD::XOR, VT) ||
      !TLI.isOperationLegalOrCustom(ISD::OR, VT))
        return DAG.UnrollVectorOp(Op.getNode());

  assert(VT.getSizeInBits() == Op.getOperand(1).getValueType().getSizeInBits()
         && "Invalid mask size");
  // Bitcast the operands to be the same type as the mask.
  // This is needed when we select between FP types because
  // the mask is a vector of integers.
  Op1 = DAG.getNode(ISD::BITCAST, DL, VT, Op1);
  Op2 = DAG.getNode(ISD::BITCAST, DL, VT, Op2);

  SDValue AllOnes = DAG.getConstant(
    APInt::getAllOnesValue(VT.getScalarType().getSizeInBits()), VT);
  SDValue NotMask = DAG.getNode(ISD::XOR, DL, VT, Mask, AllOnes);

  Op1 = DAG.getNode(ISD::AND, DL, VT, Op1, Mask);
  Op2 = DAG.getNode(ISD::AND, DL, VT, Op2, NotMask);
  return DAG.getNode(ISD::OR, DL, VT, Op1, Op2);
}

SDValue VectorLegalizer::ExpandUINT_TO_FLOAT(SDValue Op) {
  EVT VT = Op.getOperand(0).getValueType();
  DebugLoc DL = Op.getDebugLoc();

  // Make sure that the SINT_TO_FP and SRL instructions are available.
  if (!TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, VT) ||
      !TLI.isOperationLegalOrCustom(ISD::SRL, VT))
      return DAG.UnrollVectorOp(Op.getNode());

 EVT SVT = VT.getScalarType();
  assert((SVT.getSizeInBits() == 64 || SVT.getSizeInBits() == 32) &&
      "Elements in vector-UINT_TO_FP must be 32 or 64 bits wide");

  unsigned BW = SVT.getSizeInBits();
  SDValue HalfWord = DAG.getConstant(BW/2, VT);

  // Constants to clear the upper part of the word.
  // Notice that we can also use SHL+SHR, but using a constant is slightly
  // faster on x86.
  uint64_t HWMask = (SVT.getSizeInBits()==64)?0x00000000FFFFFFFF:0x0000FFFF;
  SDValue HalfWordMask = DAG.getConstant(HWMask, VT);

  // Two to the power of half-word-size.
  SDValue TWOHW = DAG.getConstantFP((1<<(BW/2)), Op.getValueType());

  // Clear upper part of LO, lower HI
  SDValue HI = DAG.getNode(ISD::SRL, DL, VT, Op.getOperand(0), HalfWord);
  SDValue LO = DAG.getNode(ISD::AND, DL, VT, Op.getOperand(0), HalfWordMask);

  // Convert hi and lo to floats
  // Convert the hi part back to the upper values
  SDValue fHI = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), HI);
          fHI = DAG.getNode(ISD::FMUL, DL, Op.getValueType(), fHI, TWOHW);
  SDValue fLO = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), LO);

  // Add the two halves
  return DAG.getNode(ISD::FADD, DL, Op.getValueType(), fHI, fLO);
}


SDValue VectorLegalizer::ExpandFNEG(SDValue Op) {
  if (TLI.isOperationLegalOrCustom(ISD::FSUB, Op.getValueType())) {
    SDValue Zero = DAG.getConstantFP(-0.0, Op.getValueType());
    return DAG.getNode(ISD::FSUB, Op.getDebugLoc(), Op.getValueType(),
                       Zero, Op.getOperand(0));
  }
  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::UnrollVSETCC(SDValue Op) {
  EVT VT = Op.getValueType();
  unsigned NumElems = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1), CC = Op.getOperand(2);
  EVT TmpEltVT = LHS.getValueType().getVectorElementType();
  DebugLoc dl = Op.getDebugLoc();
  SmallVector<SDValue, 8> Ops(NumElems);
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue LHSElem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, LHS,
                                  DAG.getIntPtrConstant(i));
    SDValue RHSElem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, RHS,
                                  DAG.getIntPtrConstant(i));
    Ops[i] = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(TmpEltVT),
                         LHSElem, RHSElem, CC);
    Ops[i] = DAG.getNode(ISD::SELECT, dl, EltVT, Ops[i],
                         DAG.getConstant(APInt::getAllOnesValue
                                         (EltVT.getSizeInBits()), EltVT),
                         DAG.getConstant(0, EltVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], NumElems);
}

}

bool SelectionDAG::LegalizeVectors() {
  return VectorLegalizer(*this).Run();
}
