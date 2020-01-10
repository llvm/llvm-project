//===- LegalizeVectorOps.cpp - Implement SelectionDAG::LegalizeVectors ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "legalizevectorops"

namespace {

class VectorLegalizer {
  SelectionDAG& DAG;
  const TargetLowering &TLI;
  bool Changed = false; // Keep track of whether anything changed

  /// For nodes that are of legal width, and that have more than one use, this
  /// map indicates what regularized operand to use.  This allows us to avoid
  /// legalizing the same thing more than once.
  SmallDenseMap<SDValue, SDValue, 64> LegalizedNodes;

  /// Adds a node to the translation cache.
  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }

  /// Legalizes the given node.
  SDValue LegalizeOp(SDValue Op);

  /// Assuming the node is legal, "legalize" the results.
  SDValue TranslateLegalizeResults(SDValue Op, SDNode *Result);

  /// Make sure Results are legal and update the translation cache.
  SDValue RecursivelyLegalizeResults(SDValue Op,
                                     MutableArrayRef<SDValue> Results);

  /// Wrapper to interface LowerOperation with a vector of Results.
  /// Returns false if the target wants to use default expansion. Otherwise
  /// returns true. If return is true and the Results are empty, then the
  /// target wants to keep the input node as is.
  bool LowerOperationWrapper(SDNode *N, SmallVectorImpl<SDValue> &Results);

  /// Implements unrolling a VSETCC.
  SDValue UnrollVSETCC(SDValue Op);

  /// Implement expand-based legalization of vector operations.
  ///
  /// This is just a high-level routine to dispatch to specific code paths for
  /// operations to legalize them.
  void Expand(SDNode *Node, SmallVectorImpl<SDValue> &Results);

  /// Implements expansion for FP_TO_UINT; falls back to UnrollVectorOp if
  /// FP_TO_SINT isn't legal.
  void ExpandFP_TO_UINT(SDValue Op, SmallVectorImpl<SDValue> &Results);

  /// Implements expansion for UINT_TO_FLOAT; falls back to UnrollVectorOp if
  /// SINT_TO_FLOAT and SHR on vectors isn't legal.
  void ExpandUINT_TO_FLOAT(SDValue Op, SmallVectorImpl<SDValue> &Results);

  /// Implement expansion for SIGN_EXTEND_INREG using SRL and SRA.
  SDValue ExpandSEXTINREG(SDValue Op);

  /// Implement expansion for ANY_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place and bitcasts to the proper
  /// type. The contents of the bits in the extended part of each element are
  /// undef.
  SDValue ExpandANY_EXTEND_VECTOR_INREG(SDValue Op);

  /// Implement expansion for SIGN_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place, bitcasts to the proper
  /// type, then shifts left and arithmetic shifts right to introduce a sign
  /// extension.
  SDValue ExpandSIGN_EXTEND_VECTOR_INREG(SDValue Op);

  /// Implement expansion for ZERO_EXTEND_VECTOR_INREG.
  ///
  /// Shuffles the low lanes of the operand into place and blends zeros into
  /// the remaining lanes, finally bitcasting to the proper type.
  SDValue ExpandZERO_EXTEND_VECTOR_INREG(SDValue Op);

  /// Implement expand-based legalization of ABS vector operations.
  /// If following expanding is legal/custom then do it:
  /// (ABS x) --> (XOR (ADD x, (SRA x, sizeof(x)-1)), (SRA x, sizeof(x)-1))
  /// else unroll the operation.
  SDValue ExpandABS(SDValue Op);

  /// Expand bswap of vectors into a shuffle if legal.
  SDValue ExpandBSWAP(SDValue Op);

  /// Implement vselect in terms of XOR, AND, OR when blend is not
  /// supported by the target.
  SDValue ExpandVSELECT(SDValue Op);
  SDValue ExpandSELECT(SDValue Op);
  std::pair<SDValue, SDValue> ExpandLoad(SDNode *N);
  SDValue ExpandStore(SDNode *N);
  SDValue ExpandFNEG(SDValue Op);
  SDValue ExpandFSUB(SDValue Op);
  SDValue ExpandBITREVERSE(SDValue Op);
  SDValue ExpandCTPOP(SDValue Op);
  SDValue ExpandCTLZ(SDValue Op);
  SDValue ExpandCTTZ(SDValue Op);
  SDValue ExpandFunnelShift(SDValue Op);
  SDValue ExpandROT(SDValue Op);
  SDValue ExpandFMINNUM_FMAXNUM(SDValue Op);
  void ExpandUADDSUBO(SDValue Op, SmallVectorImpl<SDValue> &Results);
  void ExpandSADDSUBO(SDValue Op, SmallVectorImpl<SDValue> &Results);
  void ExpandMULO(SDValue Op, SmallVectorImpl<SDValue> &Results);
  SDValue ExpandAddSubSat(SDValue Op);
  SDValue ExpandFixedPointMul(SDValue Op);
  SDValue ExpandFixedPointDiv(SDValue Op);
  SDValue ExpandStrictFPOp(SDValue Op);
  void ExpandStrictFPOp(SDValue Op, SmallVectorImpl<SDValue> &Results);

  void UnrollStrictFPOp(SDValue Op, SmallVectorImpl<SDValue> &Results);

  /// Implements vector promotion.
  ///
  /// This is essentially just bitcasting the operands to a different type and
  /// bitcasting the result back to the original type.
  void Promote(SDNode *Node, SmallVectorImpl<SDValue> &Results);

  /// Implements [SU]INT_TO_FP vector promotion.
  ///
  /// This is a [zs]ext of the input operand to a larger integer type.
  void PromoteINT_TO_FP(SDValue Op, SmallVectorImpl<SDValue> &Results);

  /// Implements FP_TO_[SU]INT vector promotion of the result type.
  ///
  /// It is promoted to a larger integer type.  The result is then
  /// truncated back to the original type.
  void PromoteFP_TO_INT(SDValue Op, SmallVectorImpl<SDValue> &Results);

public:
  VectorLegalizer(SelectionDAG& dag) :
      DAG(dag), TLI(dag.getTargetLoweringInfo()) {}

  /// Begin legalizer the vector operations in the DAG.
  bool Run();
};

} // end anonymous namespace

bool VectorLegalizer::Run() {
  // Before we start legalizing vector nodes, check if there are any vectors.
  bool HasVectors = false;
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = std::prev(DAG.allnodes_end()); I != std::next(E); ++I) {
    // Check if the values of the nodes contain vectors. We don't need to check
    // the operands because we are going to check their values at some point.
    for (SDNode::value_iterator J = I->value_begin(), E = I->value_end();
         J != E; ++J)
      HasVectors |= J->isVector();

    // If we found a vector node we can start the legalization.
    if (HasVectors)
      break;
  }

  // If this basic block has no vectors then no need to legalize vectors.
  if (!HasVectors)
    return false;

  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = std::prev(DAG.allnodes_end()); I != std::next(E); ++I)
    LegalizeOp(SDValue(&*I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  LegalizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();

  return Changed;
}

SDValue VectorLegalizer::TranslateLegalizeResults(SDValue Op, SDNode *Result) {
  assert(Op->getNumValues() == Result->getNumValues() &&
         "Unexpected number of results");
  // Generic legalization: just pass the operand through.
  for (unsigned i = 0, e = Op->getNumValues(); i != e; ++i)
    AddLegalizedOperand(Op.getValue(i), SDValue(Result, i));
  return SDValue(Result, Op.getResNo());
}

SDValue
VectorLegalizer::RecursivelyLegalizeResults(SDValue Op,
                                            MutableArrayRef<SDValue> Results) {
  assert(Results.size() == Op->getNumValues() &&
         "Unexpected number of results");
  // Make sure that the generated code is itself legal.
  for (unsigned i = 0, e = Results.size(); i != e; ++i) {
    Results[i] = LegalizeOp(Results[i]);
    AddLegalizedOperand(Op.getValue(i), Results[i]);
  }

  return Results[Op.getResNo()];
}

SDValue VectorLegalizer::LegalizeOp(SDValue Op) {
  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  // Legalize the operands
  SmallVector<SDValue, 8> Ops;
  for (const SDValue &Oper : Op->op_values())
    Ops.push_back(LegalizeOp(Oper));

  SDNode *Node = DAG.UpdateNodeOperands(Op.getNode(), Ops);

  if (Op.getOpcode() == ISD::LOAD) {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    ISD::LoadExtType ExtType = LD->getExtensionType();
    if (LD->getMemoryVT().isVector() && ExtType != ISD::NON_EXTLOAD) {
      LLVM_DEBUG(dbgs() << "\nLegalizing extending vector load: ";
                 Node->dump(&DAG));
      switch (TLI.getLoadExtAction(LD->getExtensionType(), LD->getValueType(0),
                                   LD->getMemoryVT())) {
      default: llvm_unreachable("This action is not supported yet!");
      case TargetLowering::Legal:
        return TranslateLegalizeResults(Op, Node);
      case TargetLowering::Custom: {
        SmallVector<SDValue, 2> ResultVals;
        if (LowerOperationWrapper(Node, ResultVals)) {
          if (ResultVals.empty())
            return TranslateLegalizeResults(Op, Node);

          Changed = true;
          return RecursivelyLegalizeResults(Op, ResultVals);
        }
        LLVM_FALLTHROUGH;
      }
      case TargetLowering::Expand: {
        Changed = true;
        std::pair<SDValue, SDValue> Tmp = ExpandLoad(Node);
        AddLegalizedOperand(Op.getValue(0), Tmp.first);
        AddLegalizedOperand(Op.getValue(1), Tmp.second);
        return Op.getResNo() ? Tmp.first : Tmp.second;
      }
      }
    }
  } else if (Op.getOpcode() == ISD::STORE) {
    StoreSDNode *ST = cast<StoreSDNode>(Node);
    EVT StVT = ST->getMemoryVT();
    MVT ValVT = ST->getValue().getSimpleValueType();
    if (StVT.isVector() && ST->isTruncatingStore()) {
      LLVM_DEBUG(dbgs() << "\nLegalizing truncating vector store: ";
                 Node->dump(&DAG));
      switch (TLI.getTruncStoreAction(ValVT, StVT)) {
      default: llvm_unreachable("This action is not supported yet!");
      case TargetLowering::Legal:
        return TranslateLegalizeResults(Op, Node);
      case TargetLowering::Custom: {
        SmallVector<SDValue, 1> ResultVals;
        if (LowerOperationWrapper(Node, ResultVals)) {
          if (ResultVals.empty())
            return TranslateLegalizeResults(Op, Node);

          Changed = true;
          return RecursivelyLegalizeResults(Op, ResultVals);
        }
        LLVM_FALLTHROUGH;
      }
      case TargetLowering::Expand: {
        Changed = true;
        SDValue Chain = ExpandStore(Node);
        AddLegalizedOperand(Op, Chain);
        return Chain;
      }
      }
    }
  }

  bool HasVectorValueOrOp = false;
  for (auto J = Node->value_begin(), E = Node->value_end(); J != E; ++J)
    HasVectorValueOrOp |= J->isVector();
  for (const SDValue &Oper : Node->op_values())
    HasVectorValueOrOp |= Oper.getValueType().isVector();

  if (!HasVectorValueOrOp)
    return TranslateLegalizeResults(Op, Node);

  TargetLowering::LegalizeAction Action = TargetLowering::Legal;
  EVT ValVT;
  switch (Op.getOpcode()) {
  default:
    return TranslateLegalizeResults(Op, Node);
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC, DAGN)                   \
  case ISD::STRICT_##DAGN:
#include "llvm/IR/ConstrainedOps.def"
    ValVT = Node->getValueType(0);
    if (Op.getOpcode() == ISD::STRICT_SINT_TO_FP ||
        Op.getOpcode() == ISD::STRICT_UINT_TO_FP)
      ValVT = Node->getOperand(1).getValueType();
    Action = TLI.getOperationAction(Node->getOpcode(), ValVT);
    // If we're asked to expand a strict vector floating-point operation,
    // by default we're going to simply unroll it.  That is usually the
    // best approach, except in the case where the resulting strict (scalar)
    // operations would themselves use the fallback mutation to non-strict.
    // In that specific case, just do the fallback on the vector op.
    if (Action == TargetLowering::Expand && !TLI.isStrictFPEnabled() &&
        TLI.getStrictFPOperationAction(Node->getOpcode(), ValVT) ==
            TargetLowering::Legal) {
      EVT EltVT = ValVT.getVectorElementType();
      if (TLI.getOperationAction(Node->getOpcode(), EltVT)
          == TargetLowering::Expand &&
          TLI.getStrictFPOperationAction(Node->getOpcode(), EltVT)
          == TargetLowering::Legal)
        Action = TargetLowering::Legal;
    }
    break;
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::SDIVREM:
  case ISD::UDIVREM:
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
  case ISD::FSHL:
  case ISD::FSHR:
  case ISD::ROTL:
  case ISD::ROTR:
  case ISD::ABS:
  case ISD::BSWAP:
  case ISD::BITREVERSE:
  case ISD::CTLZ:
  case ISD::CTTZ:
  case ISD::CTLZ_ZERO_UNDEF:
  case ISD::CTTZ_ZERO_UNDEF:
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
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXNUM_IEEE:
  case ISD::FMINIMUM:
  case ISD::FMAXIMUM:
  case ISD::FCOPYSIGN:
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
  case ISD::FROUND:
  case ISD::FFLOOR:
  case ISD::FP_ROUND:
  case ISD::FP_EXTEND:
  case ISD::FMA:
  case ISD::SIGN_EXTEND_INREG:
  case ISD::ANY_EXTEND_VECTOR_INREG:
  case ISD::SIGN_EXTEND_VECTOR_INREG:
  case ISD::ZERO_EXTEND_VECTOR_INREG:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
  case ISD::FCANONICALIZE:
  case ISD::SADDSAT:
  case ISD::UADDSAT:
  case ISD::SSUBSAT:
  case ISD::USUBSAT:
    Action = TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0));
    break;
  case ISD::SMULFIX:
  case ISD::SMULFIXSAT:
  case ISD::UMULFIX:
  case ISD::UMULFIXSAT:
  case ISD::SDIVFIX:
  case ISD::UDIVFIX: {
    unsigned Scale = Node->getConstantOperandVal(2);
    Action = TLI.getFixedPointOperationAction(Node->getOpcode(),
                                              Node->getValueType(0), Scale);
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_MUL:
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_SMIN:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_UMIN:
  case ISD::VECREDUCE_FADD:
  case ISD::VECREDUCE_FMUL:
  case ISD::VECREDUCE_FMAX:
  case ISD::VECREDUCE_FMIN:
    Action = TLI.getOperationAction(Node->getOpcode(),
                                    Node->getOperand(0).getValueType());
    break;
  }

  LLVM_DEBUG(dbgs() << "\nLegalizing vector op: "; Node->dump(&DAG));

  SmallVector<SDValue, 8> ResultVals;
  switch (Action) {
  default: llvm_unreachable("This action is not supported yet!");
  case TargetLowering::Promote:
    LLVM_DEBUG(dbgs() << "Promoting\n");
    Promote(Node, ResultVals);
    assert(!ResultVals.empty() && "No results for promotion?");
    break;
  case TargetLowering::Legal:
    LLVM_DEBUG(dbgs() << "Legal node: nothing to do\n");
    break;
  case TargetLowering::Custom:
    LLVM_DEBUG(dbgs() << "Trying custom legalization\n");
    if (LowerOperationWrapper(Node, ResultVals))
      break;
    LLVM_DEBUG(dbgs() << "Could not custom legalize node\n");
    LLVM_FALLTHROUGH;
  case TargetLowering::Expand:
    LLVM_DEBUG(dbgs() << "Expanding\n");
    Expand(Node, ResultVals);
    break;
  }

  if (ResultVals.empty())
    return TranslateLegalizeResults(Op, Node);

  Changed = true;
  return RecursivelyLegalizeResults(Op, ResultVals);
}

// FIME: This is very similar to the X86 override of
// TargetLowering::LowerOperationWrapper. Can we merge them somehow?
bool VectorLegalizer::LowerOperationWrapper(SDNode *Node,
                                            SmallVectorImpl<SDValue> &Results) {
  SDValue Res = TLI.LowerOperation(SDValue(Node, 0), DAG);

  if (!Res.getNode())
    return false;

  if (Res == SDValue(Node, 0))
    return true;

  // If the original node has one result, take the return value from
  // LowerOperation as is. It might not be result number 0.
  if (Node->getNumValues() == 1) {
    Results.push_back(Res);
    return true;
  }

  // If the original node has multiple results, then the return node should
  // have the same number of results.
  assert((Node->getNumValues() == Res->getNumValues()) &&
         "Lowering returned the wrong number of results!");

  // Places new result values base on N result number.
  for (unsigned I = 0, E = Node->getNumValues(); I != E; ++I)
    Results.push_back(Res.getValue(I));

  return true;
}

void VectorLegalizer::Promote(SDNode *Node, SmallVectorImpl<SDValue> &Results) {
  SDValue Op(Node, 0); // FIXME: Use Node throughout.

  // For a few operations there is a specific concept for promotion based on
  // the operand's type.
  switch (Op.getOpcode()) {
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::STRICT_SINT_TO_FP:
  case ISD::STRICT_UINT_TO_FP:
    // "Promote" the operation by extending the operand.
    PromoteINT_TO_FP(Op, Results);
    return;
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
  case ISD::STRICT_FP_TO_UINT:
  case ISD::STRICT_FP_TO_SINT:
    // Promote the operation by extending the operand.
    PromoteFP_TO_INT(Op, Results);
    return;
  case ISD::FP_ROUND:
  case ISD::FP_EXTEND:
    // These operations are used to do promotion so they can't be promoted
    // themselves.
    llvm_unreachable("Don't know how to promote this operation!");
  }

  // There are currently two cases of vector promotion:
  // 1) Bitcasting a vector of integers to a different type to a vector of the
  //    same overall length. For example, x86 promotes ISD::AND v2i32 to v1i64.
  // 2) Extending a vector of floats to a vector of the same number of larger
  //    floats. For example, AArch64 promotes ISD::FADD on v4f16 to v4f32.
  MVT VT = Op.getSimpleValueType();
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't promote a vector with multiple results!");
  MVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  SDLoc dl(Op);
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      if (Op.getOperand(j)
              .getValueType()
              .getVectorElementType()
              .isFloatingPoint() &&
          NVT.isVector() && NVT.getVectorElementType().isFloatingPoint())
        Operands[j] = DAG.getNode(ISD::FP_EXTEND, dl, NVT, Op.getOperand(j));
      else
        Operands[j] = DAG.getNode(ISD::BITCAST, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  Op = DAG.getNode(Op.getOpcode(), dl, NVT, Operands, Op.getNode()->getFlags());

  SDValue Res;
  if ((VT.isFloatingPoint() && NVT.isFloatingPoint()) ||
      (VT.isVector() && VT.getVectorElementType().isFloatingPoint() &&
       NVT.isVector() && NVT.getVectorElementType().isFloatingPoint()))
    Res = DAG.getNode(ISD::FP_ROUND, dl, VT, Op, DAG.getIntPtrConstant(0, dl));
  else
    Res = DAG.getNode(ISD::BITCAST, dl, VT, Op);

  Results.push_back(Res);
}

void VectorLegalizer::PromoteINT_TO_FP(SDValue Op,
                                       SmallVectorImpl<SDValue> &Results) {
  // INT_TO_FP operations may require the input operand be promoted even
  // when the type is otherwise legal.
  bool IsStrict = Op->isStrictFPOpcode();
  MVT VT = Op.getOperand(IsStrict ? 1 : 0).getSimpleValueType();
  MVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  assert(NVT.getVectorNumElements() == VT.getVectorNumElements() &&
         "Vectors have different number of elements!");

  SDLoc dl(Op);
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());

  unsigned Opc = (Op.getOpcode() == ISD::UINT_TO_FP ||
                  Op.getOpcode() == ISD::STRICT_UINT_TO_FP)
                     ? ISD::ZERO_EXTEND
                     : ISD::SIGN_EXTEND;
  for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
    if (Op.getOperand(j).getValueType().isVector())
      Operands[j] = DAG.getNode(Opc, dl, NVT, Op.getOperand(j));
    else
      Operands[j] = Op.getOperand(j);
  }

  if (IsStrict) {
    SDValue Res = DAG.getNode(Op.getOpcode(), dl,
                              {Op.getValueType(), MVT::Other}, Operands);
    Results.push_back(Res);
    Results.push_back(Res.getValue(1));
    return;
  }

  SDValue Res = DAG.getNode(Op.getOpcode(), dl, Op.getValueType(), Operands);
  Results.push_back(Res);
}

// For FP_TO_INT we promote the result type to a vector type with wider
// elements and then truncate the result.  This is different from the default
// PromoteVector which uses bitcast to promote thus assumning that the
// promoted vector type has the same overall size.
void VectorLegalizer::PromoteFP_TO_INT(SDValue Op,
                                       SmallVectorImpl<SDValue> &Results) {
  MVT VT = Op.getSimpleValueType();
  MVT NVT = TLI.getTypeToPromoteTo(Op.getOpcode(), VT);
  bool IsStrict = Op->isStrictFPOpcode();
  assert(NVT.getVectorNumElements() == VT.getVectorNumElements() &&
         "Vectors have different number of elements!");

  unsigned NewOpc = Op->getOpcode();
  // Change FP_TO_UINT to FP_TO_SINT if possible.
  // TODO: Should we only do this if FP_TO_UINT itself isn't legal?
  if (NewOpc == ISD::FP_TO_UINT &&
      TLI.isOperationLegalOrCustom(ISD::FP_TO_SINT, NVT))
    NewOpc = ISD::FP_TO_SINT;

  if (NewOpc == ISD::STRICT_FP_TO_UINT &&
      TLI.isOperationLegalOrCustom(ISD::STRICT_FP_TO_SINT, NVT))
    NewOpc = ISD::STRICT_FP_TO_SINT;

  SDLoc dl(Op);
  SDValue Promoted, Chain;
  if (IsStrict) {
    Promoted = DAG.getNode(NewOpc, dl, {NVT, MVT::Other},
                           {Op.getOperand(0), Op.getOperand(1)});
    Chain = Promoted.getValue(1);
  } else
    Promoted = DAG.getNode(NewOpc, dl, NVT, Op.getOperand(0));

  // Assert that the converted value fits in the original type.  If it doesn't
  // (eg: because the value being converted is too big), then the result of the
  // original operation was undefined anyway, so the assert is still correct.
  if (Op->getOpcode() == ISD::FP_TO_UINT ||
      Op->getOpcode() == ISD::STRICT_FP_TO_UINT)
    NewOpc = ISD::AssertZext;
  else
    NewOpc = ISD::AssertSext;

  Promoted = DAG.getNode(NewOpc, dl, NVT, Promoted,
                         DAG.getValueType(VT.getScalarType()));
  Promoted = DAG.getNode(ISD::TRUNCATE, dl, VT, Promoted);
  Results.push_back(Promoted);
  if (IsStrict)
    Results.push_back(Chain);
}

std::pair<SDValue, SDValue> VectorLegalizer::ExpandLoad(SDNode *N) {
  LoadSDNode *LD = cast<LoadSDNode>(N);

  EVT SrcVT = LD->getMemoryVT();
  EVT SrcEltVT = SrcVT.getScalarType();
  unsigned NumElem = SrcVT.getVectorNumElements();

  SDValue NewChain;
  SDValue Value;
  if (SrcVT.getVectorNumElements() > 1 && !SrcEltVT.isByteSized()) {
    SDLoc dl(N);

    SmallVector<SDValue, 8> Vals;
    SmallVector<SDValue, 8> LoadChains;

    EVT DstEltVT = LD->getValueType(0).getScalarType();
    SDValue Chain = LD->getChain();
    SDValue BasePTR = LD->getBasePtr();
    ISD::LoadExtType ExtType = LD->getExtensionType();

    // When elements in a vector is not byte-addressable, we cannot directly
    // load each element by advancing pointer, which could only address bytes.
    // Instead, we load all significant words, mask bits off, and concatenate
    // them to form each element. Finally, they are extended to destination
    // scalar type to build the destination vector.
    EVT WideVT = TLI.getPointerTy(DAG.getDataLayout());

    assert(WideVT.isRound() &&
           "Could not handle the sophisticated case when the widest integer is"
           " not power of 2.");
    assert(WideVT.bitsGE(SrcEltVT) &&
           "Type is not legalized?");

    unsigned WideBytes = WideVT.getStoreSize();
    unsigned Offset = 0;
    unsigned RemainingBytes = SrcVT.getStoreSize();
    SmallVector<SDValue, 8> LoadVals;
    while (RemainingBytes > 0) {
      SDValue ScalarLoad;
      unsigned LoadBytes = WideBytes;

      if (RemainingBytes >= LoadBytes) {
        ScalarLoad =
            DAG.getLoad(WideVT, dl, Chain, BasePTR,
                        LD->getPointerInfo().getWithOffset(Offset),
                        MinAlign(LD->getAlignment(), Offset),
                        LD->getMemOperand()->getFlags(), LD->getAAInfo());
      } else {
        EVT LoadVT = WideVT;
        while (RemainingBytes < LoadBytes) {
          LoadBytes >>= 1; // Reduce the load size by half.
          LoadVT = EVT::getIntegerVT(*DAG.getContext(), LoadBytes << 3);
        }
        ScalarLoad =
            DAG.getExtLoad(ISD::EXTLOAD, dl, WideVT, Chain, BasePTR,
                           LD->getPointerInfo().getWithOffset(Offset), LoadVT,
                           MinAlign(LD->getAlignment(), Offset),
                           LD->getMemOperand()->getFlags(), LD->getAAInfo());
      }

      RemainingBytes -= LoadBytes;
      Offset += LoadBytes;

      BasePTR = DAG.getObjectPtrOffset(dl, BasePTR, LoadBytes);

      LoadVals.push_back(ScalarLoad.getValue(0));
      LoadChains.push_back(ScalarLoad.getValue(1));
    }

    unsigned BitOffset = 0;
    unsigned WideIdx = 0;
    unsigned WideBits = WideVT.getSizeInBits();

    // Extract bits, pack and extend/trunc them into destination type.
    unsigned SrcEltBits = SrcEltVT.getSizeInBits();
    SDValue SrcEltBitMask = DAG.getConstant(
        APInt::getLowBitsSet(WideBits, SrcEltBits), dl, WideVT);

    for (unsigned Idx = 0; Idx != NumElem; ++Idx) {
      assert(BitOffset < WideBits && "Unexpected offset!");

      SDValue ShAmt = DAG.getConstant(
          BitOffset, dl, TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
      SDValue Lo = DAG.getNode(ISD::SRL, dl, WideVT, LoadVals[WideIdx], ShAmt);

      BitOffset += SrcEltBits;
      if (BitOffset >= WideBits) {
        WideIdx++;
        BitOffset -= WideBits;
        if (BitOffset > 0) {
          ShAmt = DAG.getConstant(
              SrcEltBits - BitOffset, dl,
              TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
          SDValue Hi =
              DAG.getNode(ISD::SHL, dl, WideVT, LoadVals[WideIdx], ShAmt);
          Lo = DAG.getNode(ISD::OR, dl, WideVT, Lo, Hi);
        }
      }

      Lo = DAG.getNode(ISD::AND, dl, WideVT, Lo, SrcEltBitMask);

      switch (ExtType) {
      default: llvm_unreachable("Unknown extended-load op!");
      case ISD::EXTLOAD:
        Lo = DAG.getAnyExtOrTrunc(Lo, dl, DstEltVT);
        break;
      case ISD::ZEXTLOAD:
        Lo = DAG.getZExtOrTrunc(Lo, dl, DstEltVT);
        break;
      case ISD::SEXTLOAD:
        ShAmt =
            DAG.getConstant(WideBits - SrcEltBits, dl,
                            TLI.getShiftAmountTy(WideVT, DAG.getDataLayout()));
        Lo = DAG.getNode(ISD::SHL, dl, WideVT, Lo, ShAmt);
        Lo = DAG.getNode(ISD::SRA, dl, WideVT, Lo, ShAmt);
        Lo = DAG.getSExtOrTrunc(Lo, dl, DstEltVT);
        break;
      }
      Vals.push_back(Lo);
    }

    NewChain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, LoadChains);
    Value = DAG.getBuildVector(N->getValueType(0), dl, Vals);
  } else {
    std::tie(Value, NewChain) = TLI.scalarizeVectorLoad(LD, DAG);
  }

  return std::make_pair(Value, NewChain);
}

SDValue VectorLegalizer::ExpandStore(SDNode *N) {
  StoreSDNode *ST = cast<StoreSDNode>(N);
  SDValue TF = TLI.scalarizeVectorStore(ST, DAG);
  return TF;
}

void VectorLegalizer::Expand(SDNode *Node, SmallVectorImpl<SDValue> &Results) {
  SDValue Op(Node, 0); // FIXME: Just pass Node to all the expanders.

  switch (Op->getOpcode()) {
  case ISD::SIGN_EXTEND_INREG:
    Results.push_back(ExpandSEXTINREG(Op));
    return;
  case ISD::ANY_EXTEND_VECTOR_INREG:
    Results.push_back(ExpandANY_EXTEND_VECTOR_INREG(Op));
    return;
  case ISD::SIGN_EXTEND_VECTOR_INREG:
    Results.push_back(ExpandSIGN_EXTEND_VECTOR_INREG(Op));
    return;
  case ISD::ZERO_EXTEND_VECTOR_INREG:
    Results.push_back(ExpandZERO_EXTEND_VECTOR_INREG(Op));
    return;
  case ISD::BSWAP:
    Results.push_back(ExpandBSWAP(Op));
    return;
  case ISD::VSELECT:
    Results.push_back(ExpandVSELECT(Op));
    return;
  case ISD::SELECT:
    Results.push_back(ExpandSELECT(Op));
    return;
  case ISD::FP_TO_UINT:
    ExpandFP_TO_UINT(Op, Results);
    return;
  case ISD::UINT_TO_FP:
    ExpandUINT_TO_FLOAT(Op, Results);
    return;
  case ISD::FNEG:
    Results.push_back(ExpandFNEG(Op));
    return;
  case ISD::FSUB:
    if (SDValue Tmp = ExpandFSUB(Op))
      Results.push_back(Tmp);
    return;
  case ISD::SETCC:
    Results.push_back(UnrollVSETCC(Op));
    return;
  case ISD::ABS:
    Results.push_back(ExpandABS(Op));
    return;
  case ISD::BITREVERSE:
    if (SDValue Tmp = ExpandBITREVERSE(Op))
      Results.push_back(Tmp);
    return;
  case ISD::CTPOP:
    Results.push_back(ExpandCTPOP(Op));
    return;
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF:
    Results.push_back(ExpandCTLZ(Op));
    return;
  case ISD::CTTZ:
  case ISD::CTTZ_ZERO_UNDEF:
    Results.push_back(ExpandCTTZ(Op));
    return;
  case ISD::FSHL:
  case ISD::FSHR:
    Results.push_back(ExpandFunnelShift(Op));
    return;
  case ISD::ROTL:
  case ISD::ROTR:
    Results.push_back(ExpandROT(Op));
    return;
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
    Results.push_back(ExpandFMINNUM_FMAXNUM(Op));
    return;
  case ISD::UADDO:
  case ISD::USUBO:
    ExpandUADDSUBO(Op, Results);
    return;
  case ISD::SADDO:
  case ISD::SSUBO:
    ExpandSADDSUBO(Op, Results);
    return;
  case ISD::UMULO:
  case ISD::SMULO:
    ExpandMULO(Op, Results);
    return;
  case ISD::USUBSAT:
  case ISD::SSUBSAT:
  case ISD::UADDSAT:
  case ISD::SADDSAT:
    Results.push_back(ExpandAddSubSat(Op));
    return;
  case ISD::SMULFIX:
  case ISD::UMULFIX:
    Results.push_back(ExpandFixedPointMul(Op));
    return;
  case ISD::SMULFIXSAT:
  case ISD::UMULFIXSAT:
    // FIXME: We do not expand SMULFIXSAT/UMULFIXSAT here yet, not sure exactly
    // why. Maybe it results in worse codegen compared to the unroll for some
    // targets? This should probably be investigated. And if we still prefer to
    // unroll an explanation could be helpful.
    Results.push_back(DAG.UnrollVectorOp(Op.getNode()));
    return;
  case ISD::SDIVFIX:
  case ISD::UDIVFIX:
    Results.push_back(ExpandFixedPointDiv(Op));
    return;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC, DAGN)                   \
  case ISD::STRICT_##DAGN:
#include "llvm/IR/ConstrainedOps.def"
    ExpandStrictFPOp(Op, Results);
    return;
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_MUL:
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_SMIN:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_UMIN:
  case ISD::VECREDUCE_FADD:
  case ISD::VECREDUCE_FMUL:
  case ISD::VECREDUCE_FMAX:
  case ISD::VECREDUCE_FMIN:
    Results.push_back(TLI.expandVecReduce(Op.getNode(), DAG));
    return;
  default:
    Results.push_back(DAG.UnrollVectorOp(Op.getNode()));
    return;
  }
}

SDValue VectorLegalizer::ExpandSELECT(SDValue Op) {
  // Lower a select instruction where the condition is a scalar and the
  // operands are vectors. Lower this select to VSELECT and implement it
  // using XOR AND OR. The selector bit is broadcasted.
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  SDValue Mask = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);

  assert(VT.isVector() && !Mask.getValueType().isVector()
         && Op1.getValueType() == Op2.getValueType() && "Invalid type");

  // If we can't even use the basic vector operations of
  // AND,OR,XOR, we will have to scalarize the op.
  // Notice that the operation may be 'promoted' which means that it is
  // 'bitcasted' to another type which is handled.
  // Also, we need to be able to construct a splat vector using BUILD_VECTOR.
  if (TLI.getOperationAction(ISD::AND, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::XOR, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::OR,  VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::BUILD_VECTOR,  VT) == TargetLowering::Expand)
    return DAG.UnrollVectorOp(Op.getNode());

  // Generate a mask operand.
  EVT MaskTy = VT.changeVectorElementTypeToInteger();

  // What is the size of each element in the vector mask.
  EVT BitTy = MaskTy.getScalarType();

  Mask = DAG.getSelect(DL, BitTy, Mask,
          DAG.getConstant(APInt::getAllOnesValue(BitTy.getSizeInBits()), DL,
                          BitTy),
          DAG.getConstant(0, DL, BitTy));

  // Broadcast the mask so that the entire vector is all-one or all zero.
  Mask = DAG.getSplatBuildVector(MaskTy, DL, Mask);

  // Bitcast the operands to be the same type as the mask.
  // This is needed when we select between FP types because
  // the mask is a vector of integers.
  Op1 = DAG.getNode(ISD::BITCAST, DL, MaskTy, Op1);
  Op2 = DAG.getNode(ISD::BITCAST, DL, MaskTy, Op2);

  SDValue AllOnes = DAG.getConstant(
            APInt::getAllOnesValue(BitTy.getSizeInBits()), DL, MaskTy);
  SDValue NotMask = DAG.getNode(ISD::XOR, DL, MaskTy, Mask, AllOnes);

  Op1 = DAG.getNode(ISD::AND, DL, MaskTy, Op1, Mask);
  Op2 = DAG.getNode(ISD::AND, DL, MaskTy, Op2, NotMask);
  SDValue Val = DAG.getNode(ISD::OR, DL, MaskTy, Op1, Op2);
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Val);
}

SDValue VectorLegalizer::ExpandSEXTINREG(SDValue Op) {
  EVT VT = Op.getValueType();

  // Make sure that the SRA and SHL instructions are available.
  if (TLI.getOperationAction(ISD::SRA, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::SHL, VT) == TargetLowering::Expand)
    return DAG.UnrollVectorOp(Op.getNode());

  SDLoc DL(Op);
  EVT OrigTy = cast<VTSDNode>(Op->getOperand(1))->getVT();

  unsigned BW = VT.getScalarSizeInBits();
  unsigned OrigBW = OrigTy.getScalarSizeInBits();
  SDValue ShiftSz = DAG.getConstant(BW - OrigBW, DL, VT);

  Op = Op.getOperand(0);
  Op =   DAG.getNode(ISD::SHL, DL, VT, Op, ShiftSz);
  return DAG.getNode(ISD::SRA, DL, VT, Op, ShiftSz);
}

// Generically expand a vector anyext in register to a shuffle of the relevant
// lanes into the appropriate locations, with other lanes left undef.
SDValue VectorLegalizer::ExpandANY_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  int NumElements = VT.getVectorNumElements();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  int NumSrcElements = SrcVT.getVectorNumElements();

  // *_EXTEND_VECTOR_INREG SrcVT can be smaller than VT - so insert the vector
  // into a larger vector type.
  if (SrcVT.bitsLE(VT)) {
    assert((VT.getSizeInBits() % SrcVT.getScalarSizeInBits()) == 0 &&
           "ANY_EXTEND_VECTOR_INREG vector size mismatch");
    NumSrcElements = VT.getSizeInBits() / SrcVT.getScalarSizeInBits();
    SrcVT = EVT::getVectorVT(*DAG.getContext(), SrcVT.getScalarType(),
                             NumSrcElements);
    Src = DAG.getNode(
        ISD::INSERT_SUBVECTOR, DL, SrcVT, DAG.getUNDEF(SrcVT), Src,
        DAG.getConstant(0, DL, TLI.getVectorIdxTy(DAG.getDataLayout())));
  }

  // Build a base mask of undef shuffles.
  SmallVector<int, 16> ShuffleMask;
  ShuffleMask.resize(NumSrcElements, -1);

  // Place the extended lanes into the correct locations.
  int ExtLaneScale = NumSrcElements / NumElements;
  int EndianOffset = DAG.getDataLayout().isBigEndian() ? ExtLaneScale - 1 : 0;
  for (int i = 0; i < NumElements; ++i)
    ShuffleMask[i * ExtLaneScale + EndianOffset] = i;

  return DAG.getNode(
      ISD::BITCAST, DL, VT,
      DAG.getVectorShuffle(SrcVT, DL, Src, DAG.getUNDEF(SrcVT), ShuffleMask));
}

SDValue VectorLegalizer::ExpandSIGN_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();

  // First build an any-extend node which can be legalized above when we
  // recurse through it.
  Op = DAG.getNode(ISD::ANY_EXTEND_VECTOR_INREG, DL, VT, Src);

  // Now we need sign extend. Do this by shifting the elements. Even if these
  // aren't legal operations, they have a better chance of being legalized
  // without full scalarization than the sign extension does.
  unsigned EltWidth = VT.getScalarSizeInBits();
  unsigned SrcEltWidth = SrcVT.getScalarSizeInBits();
  SDValue ShiftAmount = DAG.getConstant(EltWidth - SrcEltWidth, DL, VT);
  return DAG.getNode(ISD::SRA, DL, VT,
                     DAG.getNode(ISD::SHL, DL, VT, Op, ShiftAmount),
                     ShiftAmount);
}

// Generically expand a vector zext in register to a shuffle of the relevant
// lanes into the appropriate locations, a blend of zero into the high bits,
// and a bitcast to the wider element type.
SDValue VectorLegalizer::ExpandZERO_EXTEND_VECTOR_INREG(SDValue Op) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  int NumElements = VT.getVectorNumElements();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  int NumSrcElements = SrcVT.getVectorNumElements();

  // *_EXTEND_VECTOR_INREG SrcVT can be smaller than VT - so insert the vector
  // into a larger vector type.
  if (SrcVT.bitsLE(VT)) {
    assert((VT.getSizeInBits() % SrcVT.getScalarSizeInBits()) == 0 &&
           "ZERO_EXTEND_VECTOR_INREG vector size mismatch");
    NumSrcElements = VT.getSizeInBits() / SrcVT.getScalarSizeInBits();
    SrcVT = EVT::getVectorVT(*DAG.getContext(), SrcVT.getScalarType(),
                             NumSrcElements);
    Src = DAG.getNode(
        ISD::INSERT_SUBVECTOR, DL, SrcVT, DAG.getUNDEF(SrcVT), Src,
        DAG.getConstant(0, DL, TLI.getVectorIdxTy(DAG.getDataLayout())));
  }

  // Build up a zero vector to blend into this one.
  SDValue Zero = DAG.getConstant(0, DL, SrcVT);

  // Shuffle the incoming lanes into the correct position, and pull all other
  // lanes from the zero vector.
  SmallVector<int, 16> ShuffleMask;
  ShuffleMask.reserve(NumSrcElements);
  for (int i = 0; i < NumSrcElements; ++i)
    ShuffleMask.push_back(i);

  int ExtLaneScale = NumSrcElements / NumElements;
  int EndianOffset = DAG.getDataLayout().isBigEndian() ? ExtLaneScale - 1 : 0;
  for (int i = 0; i < NumElements; ++i)
    ShuffleMask[i * ExtLaneScale + EndianOffset] = NumSrcElements + i;

  return DAG.getNode(ISD::BITCAST, DL, VT,
                     DAG.getVectorShuffle(SrcVT, DL, Zero, Src, ShuffleMask));
}

static void createBSWAPShuffleMask(EVT VT, SmallVectorImpl<int> &ShuffleMask) {
  int ScalarSizeInBytes = VT.getScalarSizeInBits() / 8;
  for (int I = 0, E = VT.getVectorNumElements(); I != E; ++I)
    for (int J = ScalarSizeInBytes - 1; J >= 0; --J)
      ShuffleMask.push_back((I * ScalarSizeInBytes) + J);
}

SDValue VectorLegalizer::ExpandBSWAP(SDValue Op) {
  EVT VT = Op.getValueType();

  // Generate a byte wise shuffle mask for the BSWAP.
  SmallVector<int, 16> ShuffleMask;
  createBSWAPShuffleMask(VT, ShuffleMask);
  EVT ByteVT = EVT::getVectorVT(*DAG.getContext(), MVT::i8, ShuffleMask.size());

  // Only emit a shuffle if the mask is legal.
  if (!TLI.isShuffleMaskLegal(ShuffleMask, ByteVT))
    return DAG.UnrollVectorOp(Op.getNode());

  SDLoc DL(Op);
  Op = DAG.getNode(ISD::BITCAST, DL, ByteVT, Op.getOperand(0));
  Op = DAG.getVectorShuffle(ByteVT, DL, Op, DAG.getUNDEF(ByteVT), ShuffleMask);
  return DAG.getNode(ISD::BITCAST, DL, VT, Op);
}

SDValue VectorLegalizer::ExpandBITREVERSE(SDValue Op) {
  EVT VT = Op.getValueType();

  // If we have the scalar operation, it's probably cheaper to unroll it.
  if (TLI.isOperationLegalOrCustom(ISD::BITREVERSE, VT.getScalarType()))
    return DAG.UnrollVectorOp(Op.getNode());

  // If the vector element width is a whole number of bytes, test if its legal
  // to BSWAP shuffle the bytes and then perform the BITREVERSE on the byte
  // vector. This greatly reduces the number of bit shifts necessary.
  unsigned ScalarSizeInBits = VT.getScalarSizeInBits();
  if (ScalarSizeInBits > 8 && (ScalarSizeInBits % 8) == 0) {
    SmallVector<int, 16> BSWAPMask;
    createBSWAPShuffleMask(VT, BSWAPMask);

    EVT ByteVT = EVT::getVectorVT(*DAG.getContext(), MVT::i8, BSWAPMask.size());
    if (TLI.isShuffleMaskLegal(BSWAPMask, ByteVT) &&
        (TLI.isOperationLegalOrCustom(ISD::BITREVERSE, ByteVT) ||
         (TLI.isOperationLegalOrCustom(ISD::SHL, ByteVT) &&
          TLI.isOperationLegalOrCustom(ISD::SRL, ByteVT) &&
          TLI.isOperationLegalOrCustomOrPromote(ISD::AND, ByteVT) &&
          TLI.isOperationLegalOrCustomOrPromote(ISD::OR, ByteVT)))) {
      SDLoc DL(Op);
      Op = DAG.getNode(ISD::BITCAST, DL, ByteVT, Op.getOperand(0));
      Op = DAG.getVectorShuffle(ByteVT, DL, Op, DAG.getUNDEF(ByteVT),
                                BSWAPMask);
      Op = DAG.getNode(ISD::BITREVERSE, DL, ByteVT, Op);
      return DAG.getNode(ISD::BITCAST, DL, VT, Op);
    }
  }

  // If we have the appropriate vector bit operations, it is better to use them
  // than unrolling and expanding each component.
  if (!TLI.isOperationLegalOrCustom(ISD::SHL, VT) ||
      !TLI.isOperationLegalOrCustom(ISD::SRL, VT) ||
      !TLI.isOperationLegalOrCustomOrPromote(ISD::AND, VT) ||
      !TLI.isOperationLegalOrCustomOrPromote(ISD::OR, VT))
    return DAG.UnrollVectorOp(Op.getNode());

  // Let LegalizeDAG handle this later.
  return SDValue();
}

SDValue VectorLegalizer::ExpandVSELECT(SDValue Op) {
  // Implement VSELECT in terms of XOR, AND, OR
  // on platforms which do not support blend natively.
  SDLoc DL(Op);

  SDValue Mask = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);

  EVT VT = Mask.getValueType();

  // If we can't even use the basic vector operations of
  // AND,OR,XOR, we will have to scalarize the op.
  // Notice that the operation may be 'promoted' which means that it is
  // 'bitcasted' to another type which is handled.
  // This operation also isn't safe with AND, OR, XOR when the boolean
  // type is 0/1 as we need an all ones vector constant to mask with.
  // FIXME: Sign extend 1 to all ones if thats legal on the target.
  if (TLI.getOperationAction(ISD::AND, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::XOR, VT) == TargetLowering::Expand ||
      TLI.getOperationAction(ISD::OR, VT) == TargetLowering::Expand ||
      TLI.getBooleanContents(Op1.getValueType()) !=
          TargetLowering::ZeroOrNegativeOneBooleanContent)
    return DAG.UnrollVectorOp(Op.getNode());

  // If the mask and the type are different sizes, unroll the vector op. This
  // can occur when getSetCCResultType returns something that is different in
  // size from the operand types. For example, v4i8 = select v4i32, v4i8, v4i8.
  if (VT.getSizeInBits() != Op1.getValueSizeInBits())
    return DAG.UnrollVectorOp(Op.getNode());

  // Bitcast the operands to be the same type as the mask.
  // This is needed when we select between FP types because
  // the mask is a vector of integers.
  Op1 = DAG.getNode(ISD::BITCAST, DL, VT, Op1);
  Op2 = DAG.getNode(ISD::BITCAST, DL, VT, Op2);

  SDValue AllOnes = DAG.getConstant(
    APInt::getAllOnesValue(VT.getScalarSizeInBits()), DL, VT);
  SDValue NotMask = DAG.getNode(ISD::XOR, DL, VT, Mask, AllOnes);

  Op1 = DAG.getNode(ISD::AND, DL, VT, Op1, Mask);
  Op2 = DAG.getNode(ISD::AND, DL, VT, Op2, NotMask);
  SDValue Val = DAG.getNode(ISD::OR, DL, VT, Op1, Op2);
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Val);
}

SDValue VectorLegalizer::ExpandABS(SDValue Op) {
  // Attempt to expand using TargetLowering.
  SDValue Result;
  if (TLI.expandABS(Op.getNode(), Result, DAG))
    return Result;

  // Otherwise go ahead and unroll.
  return DAG.UnrollVectorOp(Op.getNode());
}

void VectorLegalizer::ExpandFP_TO_UINT(SDValue Op,
                                       SmallVectorImpl<SDValue> &Results) {
  // Attempt to expand using TargetLowering.
  SDValue Result, Chain;
  if (TLI.expandFP_TO_UINT(Op.getNode(), Result, Chain, DAG)) {
    Results.push_back(Result);
    if (Op->isStrictFPOpcode())
      Results.push_back(Chain);
    return;
  }

  // Otherwise go ahead and unroll.
  if (Op->isStrictFPOpcode()) {
    UnrollStrictFPOp(Op, Results);
    return;
  }

  Results.push_back(DAG.UnrollVectorOp(Op.getNode()));
}

void VectorLegalizer::ExpandUINT_TO_FLOAT(SDValue Op,
                                          SmallVectorImpl<SDValue> &Results) {
  bool IsStrict = Op.getNode()->isStrictFPOpcode();
  unsigned OpNo = IsStrict ? 1 : 0;
  SDValue Src = Op.getOperand(OpNo);
  EVT VT = Src.getValueType();
  SDLoc DL(Op);

  // Attempt to expand using TargetLowering.
  SDValue Result;
  SDValue Chain;
  if (TLI.expandUINT_TO_FP(Op.getNode(), Result, Chain, DAG)) {
    Results.push_back(Result);
    if (IsStrict)
      Results.push_back(Chain);
    return;
  }

  // Make sure that the SINT_TO_FP and SRL instructions are available.
  if (((!IsStrict && TLI.getOperationAction(ISD::SINT_TO_FP, VT) ==
                         TargetLowering::Expand) ||
       (IsStrict && TLI.getOperationAction(ISD::STRICT_SINT_TO_FP, VT) ==
                        TargetLowering::Expand)) ||
      TLI.getOperationAction(ISD::SRL, VT) == TargetLowering::Expand) {
    if (IsStrict) {
      UnrollStrictFPOp(Op, Results);
      return;
    }

    Results.push_back(DAG.UnrollVectorOp(Op.getNode()));
    return;
  }

  unsigned BW = VT.getScalarSizeInBits();
  assert((BW == 64 || BW == 32) &&
         "Elements in vector-UINT_TO_FP must be 32 or 64 bits wide");

  SDValue HalfWord = DAG.getConstant(BW / 2, DL, VT);

  // Constants to clear the upper part of the word.
  // Notice that we can also use SHL+SHR, but using a constant is slightly
  // faster on x86.
  uint64_t HWMask = (BW == 64) ? 0x00000000FFFFFFFF : 0x0000FFFF;
  SDValue HalfWordMask = DAG.getConstant(HWMask, DL, VT);

  // Two to the power of half-word-size.
  SDValue TWOHW = DAG.getConstantFP(1ULL << (BW / 2), DL, Op.getValueType());

  // Clear upper part of LO, lower HI
  SDValue HI = DAG.getNode(ISD::SRL, DL, VT, Src, HalfWord);
  SDValue LO = DAG.getNode(ISD::AND, DL, VT, Src, HalfWordMask);

  if (IsStrict) {
    // Convert hi and lo to floats
    // Convert the hi part back to the upper values
    // TODO: Can any fast-math-flags be set on these nodes?
    SDValue fHI =
        DAG.getNode(ISD::STRICT_SINT_TO_FP, DL, {Op.getValueType(), MVT::Other},
                    {Op.getOperand(0), HI});
    fHI = DAG.getNode(ISD::STRICT_FMUL, DL, {Op.getValueType(), MVT::Other},
                      {SDValue(fHI.getNode(), 1), fHI, TWOHW});
    SDValue fLO =
        DAG.getNode(ISD::STRICT_SINT_TO_FP, DL, {Op.getValueType(), MVT::Other},
                    {SDValue(fHI.getNode(), 1), LO});

    // Add the two halves
    SDValue Result =
        DAG.getNode(ISD::STRICT_FADD, DL, {Op.getValueType(), MVT::Other},
                    {SDValue(fLO.getNode(), 1), fHI, fLO});

    Results.push_back(Result);
    Results.push_back(Result.getValue(1));
    return;
  }

  // Convert hi and lo to floats
  // Convert the hi part back to the upper values
  // TODO: Can any fast-math-flags be set on these nodes?
  SDValue fHI = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), HI);
          fHI = DAG.getNode(ISD::FMUL, DL, Op.getValueType(), fHI, TWOHW);
  SDValue fLO = DAG.getNode(ISD::SINT_TO_FP, DL, Op.getValueType(), LO);

  // Add the two halves
  Results.push_back(DAG.getNode(ISD::FADD, DL, Op.getValueType(), fHI, fLO));
}

SDValue VectorLegalizer::ExpandFNEG(SDValue Op) {
  if (TLI.isOperationLegalOrCustom(ISD::FSUB, Op.getValueType())) {
    SDLoc DL(Op);
    SDValue Zero = DAG.getConstantFP(-0.0, DL, Op.getValueType());
    // TODO: If FNEG had fast-math-flags, they'd get propagated to this FSUB.
    return DAG.getNode(ISD::FSUB, DL, Op.getValueType(),
                       Zero, Op.getOperand(0));
  }
  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandFSUB(SDValue Op) {
  // For floating-point values, (a-b) is the same as a+(-b). If FNEG is legal,
  // we can defer this to operation legalization where it will be lowered as
  // a+(-b).
  EVT VT = Op.getValueType();
  if (TLI.isOperationLegalOrCustom(ISD::FNEG, VT) &&
      TLI.isOperationLegalOrCustom(ISD::FADD, VT))
    return SDValue(); // Defer to LegalizeDAG

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandCTPOP(SDValue Op) {
  SDValue Result;
  if (TLI.expandCTPOP(Op.getNode(), Result, DAG))
    return Result;

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandCTLZ(SDValue Op) {
  SDValue Result;
  if (TLI.expandCTLZ(Op.getNode(), Result, DAG))
    return Result;

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandCTTZ(SDValue Op) {
  SDValue Result;
  if (TLI.expandCTTZ(Op.getNode(), Result, DAG))
    return Result;

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandFunnelShift(SDValue Op) {
  SDValue Result;
  if (TLI.expandFunnelShift(Op.getNode(), Result, DAG))
    return Result;

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandROT(SDValue Op) {
  SDValue Result;
  if (TLI.expandROT(Op.getNode(), Result, DAG))
    return Result;

  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandFMINNUM_FMAXNUM(SDValue Op) {
  if (SDValue Expanded = TLI.expandFMINNUM_FMAXNUM(Op.getNode(), DAG))
    return Expanded;
  return DAG.UnrollVectorOp(Op.getNode());
}

void VectorLegalizer::ExpandUADDSUBO(SDValue Op,
                                     SmallVectorImpl<SDValue> &Results) {
  SDValue Result, Overflow;
  TLI.expandUADDSUBO(Op.getNode(), Result, Overflow, DAG);
  Results.push_back(Result);
  Results.push_back(Overflow);
}

void VectorLegalizer::ExpandSADDSUBO(SDValue Op,
                                     SmallVectorImpl<SDValue> &Results) {
  SDValue Result, Overflow;
  TLI.expandSADDSUBO(Op.getNode(), Result, Overflow, DAG);
  Results.push_back(Result);
  Results.push_back(Overflow);
}

void VectorLegalizer::ExpandMULO(SDValue Op,
                                 SmallVectorImpl<SDValue> &Results) {
  SDValue Result, Overflow;
  if (!TLI.expandMULO(Op.getNode(), Result, Overflow, DAG))
    std::tie(Result, Overflow) = DAG.UnrollVectorOverflowOp(Op.getNode());

  Results.push_back(Result);
  Results.push_back(Overflow);
}

SDValue VectorLegalizer::ExpandAddSubSat(SDValue Op) {
  if (SDValue Expanded = TLI.expandAddSubSat(Op.getNode(), DAG))
    return Expanded;
  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandFixedPointMul(SDValue Op) {
  if (SDValue Expanded = TLI.expandFixedPointMul(Op.getNode(), DAG))
    return Expanded;
  return DAG.UnrollVectorOp(Op.getNode());
}

SDValue VectorLegalizer::ExpandFixedPointDiv(SDValue Op) {
  SDNode *N = Op.getNode();
  if (SDValue Expanded = TLI.expandFixedPointDiv(N->getOpcode(), SDLoc(N),
          N->getOperand(0), N->getOperand(1), N->getConstantOperandVal(2), DAG))
    return Expanded;
  return DAG.UnrollVectorOp(N);
}

void VectorLegalizer::ExpandStrictFPOp(SDValue Op,
                                       SmallVectorImpl<SDValue> &Results) {
  if (Op.getOpcode() == ISD::STRICT_UINT_TO_FP) {
    ExpandUINT_TO_FLOAT(Op, Results);
    return;
  }
  if (Op.getOpcode() == ISD::STRICT_FP_TO_UINT) {
    ExpandFP_TO_UINT(Op, Results);
    return;
  }

  UnrollStrictFPOp(Op, Results);
}

void VectorLegalizer::UnrollStrictFPOp(SDValue Op,
                                       SmallVectorImpl<SDValue> &Results) {
  EVT VT = Op.getValue(0).getValueType();
  EVT EltVT = VT.getVectorElementType();
  unsigned NumElems = VT.getVectorNumElements();
  unsigned NumOpers = Op.getNumOperands();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  EVT TmpEltVT = EltVT;
  if (Op->getOpcode() == ISD::STRICT_FSETCC ||
      Op->getOpcode() == ISD::STRICT_FSETCCS)
    TmpEltVT = TLI.getSetCCResultType(DAG.getDataLayout(),
                                      *DAG.getContext(), TmpEltVT);

  EVT ValueVTs[] = {TmpEltVT, MVT::Other};
  SDValue Chain = Op.getOperand(0);
  SDLoc dl(Op);

  SmallVector<SDValue, 32> OpValues;
  SmallVector<SDValue, 32> OpChains;
  for (unsigned i = 0; i < NumElems; ++i) {
    SmallVector<SDValue, 4> Opers;
    SDValue Idx = DAG.getConstant(i, dl,
                                  TLI.getVectorIdxTy(DAG.getDataLayout()));

    // The Chain is the first operand.
    Opers.push_back(Chain);

    // Now process the remaining operands.
    for (unsigned j = 1; j < NumOpers; ++j) {
      SDValue Oper = Op.getOperand(j);
      EVT OperVT = Oper.getValueType();

      if (OperVT.isVector())
        Oper = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                           OperVT.getVectorElementType(), Oper, Idx);

      Opers.push_back(Oper);
    }

    SDValue ScalarOp = DAG.getNode(Op->getOpcode(), dl, ValueVTs, Opers);
    SDValue ScalarResult = ScalarOp.getValue(0);
    SDValue ScalarChain = ScalarOp.getValue(1);

    if (Op->getOpcode() == ISD::STRICT_FSETCC ||
        Op->getOpcode() == ISD::STRICT_FSETCCS)
      ScalarResult = DAG.getSelect(dl, EltVT, ScalarResult,
                           DAG.getConstant(APInt::getAllOnesValue
                                           (EltVT.getSizeInBits()), dl, EltVT),
                           DAG.getConstant(0, dl, EltVT));

    OpValues.push_back(ScalarResult);
    OpChains.push_back(ScalarChain);
  }

  SDValue Result = DAG.getBuildVector(VT, dl, OpValues);
  SDValue NewChain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OpChains);

  Results.push_back(Result);
  Results.push_back(NewChain);
}

SDValue VectorLegalizer::UnrollVSETCC(SDValue Op) {
  EVT VT = Op.getValueType();
  unsigned NumElems = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1), CC = Op.getOperand(2);
  EVT TmpEltVT = LHS.getValueType().getVectorElementType();
  SDLoc dl(Op);
  SmallVector<SDValue, 8> Ops(NumElems);
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue LHSElem = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, LHS,
        DAG.getConstant(i, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));
    SDValue RHSElem = DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT, RHS,
        DAG.getConstant(i, dl, TLI.getVectorIdxTy(DAG.getDataLayout())));
    Ops[i] = DAG.getNode(ISD::SETCC, dl,
                         TLI.getSetCCResultType(DAG.getDataLayout(),
                                                *DAG.getContext(), TmpEltVT),
                         LHSElem, RHSElem, CC);
    Ops[i] = DAG.getSelect(dl, EltVT, Ops[i],
                           DAG.getConstant(APInt::getAllOnesValue
                                           (EltVT.getSizeInBits()), dl, EltVT),
                           DAG.getConstant(0, dl, EltVT));
  }
  return DAG.getBuildVector(VT, dl, Ops);
}

bool SelectionDAG::LegalizeVectors() {
  return VectorLegalizer(*this).Run();
}
