//===-- DAGCombiner.cpp - Implement a DAG node combiner -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// This pass is not a substitute for the LLVM IR instcombine pass. This pass is
// primarily intended to handle simplification opportunities that are implicit
// in the LLVM IR and exposed by the various codegen lowering phases.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");
STATISTIC(OpsNarrowed     , "Number of load/op/store narrowed");

namespace {
  static cl::opt<bool>
    CombinerAA("combiner-alias-analysis", cl::Hidden,
               cl::desc("Turn on alias analysis during testing"));

  static cl::opt<bool>
    CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
               cl::desc("Include global information in alias analysis"));

//------------------------------ DAGCombiner ---------------------------------//

  class DAGCombiner {
    SelectionDAG &DAG;
    const TargetLowering &TLI;
    CombineLevel Level;
    CodeGenOpt::Level OptLevel;
    bool LegalOperations;
    bool LegalTypes;

    // Worklist of all of the nodes that need to be simplified.
    std::vector<SDNode*> WorkList;

    // AA - Used for DAG load/store alias analysis.
    AliasAnalysis &AA;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(SDNode *N) {
      for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
           UI != UE; ++UI)
        AddToWorkList(*UI);
    }

    /// visit - call the node-specific routine that knows how to fold each
    /// particular type of node.
    SDValue visit(SDNode *N);

  public:
    /// AddToWorkList - Add to the work list making sure it's instance is at the
    /// the back (next to be processed.)
    void AddToWorkList(SDNode *N) {
      removeFromWorkList(N);
      WorkList.push_back(N);
    }

    /// removeFromWorkList - remove all instances of N from the worklist.
    ///
    void removeFromWorkList(SDNode *N) {
      WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), N),
                     WorkList.end());
    }

    SDValue CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                      bool AddTo = true);

    SDValue CombineTo(SDNode *N, SDValue Res, bool AddTo = true) {
      return CombineTo(N, &Res, 1, AddTo);
    }

    SDValue CombineTo(SDNode *N, SDValue Res0, SDValue Res1,
                      bool AddTo = true) {
      SDValue To[] = { Res0, Res1 };
      return CombineTo(N, To, 2, AddTo);
    }

    void CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO);

  private:

    /// SimplifyDemandedBits - Check the specified integer node value to see if
    /// it can be simplified or if things it uses can be simplified by bit
    /// propagation.  If so, return true.
    bool SimplifyDemandedBits(SDValue Op) {
      unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
      APInt Demanded = APInt::getAllOnesValue(BitWidth);
      return SimplifyDemandedBits(Op, Demanded);
    }

    bool SimplifyDemandedBits(SDValue Op, const APInt &Demanded);

    bool CombineToPreIndexedLoadStore(SDNode *N);
    bool CombineToPostIndexedLoadStore(SDNode *N);


    /// combine - call the node-specific routine that knows how to fold each
    /// particular type of node. If that doesn't do anything, try the
    /// target-specific DAG combines.
    SDValue combine(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDValue.getNode() == 0 - No change was made
    //   SDValue.getNode() == N - N was replaced, is dead and has been handled.
    //   otherwise              - N should be replaced by the returned Operand.
    //
    SDValue visitTokenFactor(SDNode *N);
    SDValue visitMERGE_VALUES(SDNode *N);
    SDValue visitADD(SDNode *N);
    SDValue visitSUB(SDNode *N);
    SDValue visitADDC(SDNode *N);
    SDValue visitADDE(SDNode *N);
    SDValue visitMUL(SDNode *N);
    SDValue visitSDIV(SDNode *N);
    SDValue visitUDIV(SDNode *N);
    SDValue visitSREM(SDNode *N);
    SDValue visitUREM(SDNode *N);
    SDValue visitMULHU(SDNode *N);
    SDValue visitMULHS(SDNode *N);
    SDValue visitSMUL_LOHI(SDNode *N);
    SDValue visitUMUL_LOHI(SDNode *N);
    SDValue visitSDIVREM(SDNode *N);
    SDValue visitUDIVREM(SDNode *N);
    SDValue visitAND(SDNode *N);
    SDValue visitOR(SDNode *N);
    SDValue visitXOR(SDNode *N);
    SDValue SimplifyVBinOp(SDNode *N);
    SDValue visitSHL(SDNode *N);
    SDValue visitSRA(SDNode *N);
    SDValue visitSRL(SDNode *N);
    SDValue visitCTLZ(SDNode *N);
    SDValue visitCTTZ(SDNode *N);
    SDValue visitCTPOP(SDNode *N);
    SDValue visitSELECT(SDNode *N);
    SDValue visitSELECT_CC(SDNode *N);
    SDValue visitSETCC(SDNode *N);
    SDValue visitSIGN_EXTEND(SDNode *N);
    SDValue visitZERO_EXTEND(SDNode *N);
    SDValue visitANY_EXTEND(SDNode *N);
    SDValue visitSIGN_EXTEND_INREG(SDNode *N);
    SDValue visitTRUNCATE(SDNode *N);
    SDValue visitBIT_CONVERT(SDNode *N);
    SDValue visitBUILD_PAIR(SDNode *N);
    SDValue visitFADD(SDNode *N);
    SDValue visitFSUB(SDNode *N);
    SDValue visitFMUL(SDNode *N);
    SDValue visitFDIV(SDNode *N);
    SDValue visitFREM(SDNode *N);
    SDValue visitFCOPYSIGN(SDNode *N);
    SDValue visitSINT_TO_FP(SDNode *N);
    SDValue visitUINT_TO_FP(SDNode *N);
    SDValue visitFP_TO_SINT(SDNode *N);
    SDValue visitFP_TO_UINT(SDNode *N);
    SDValue visitFP_ROUND(SDNode *N);
    SDValue visitFP_ROUND_INREG(SDNode *N);
    SDValue visitFP_EXTEND(SDNode *N);
    SDValue visitFNEG(SDNode *N);
    SDValue visitFABS(SDNode *N);
    SDValue visitBRCOND(SDNode *N);
    SDValue visitBR_CC(SDNode *N);
    SDValue visitLOAD(SDNode *N);
    SDValue visitSTORE(SDNode *N);
    SDValue visitINSERT_VECTOR_ELT(SDNode *N);
    SDValue visitEXTRACT_VECTOR_ELT(SDNode *N);
    SDValue visitBUILD_VECTOR(SDNode *N);
    SDValue visitCONCAT_VECTORS(SDNode *N);
    SDValue visitVECTOR_SHUFFLE(SDNode *N);

    SDValue XformToShuffleWithZero(SDNode *N);
    SDValue ReassociateOps(unsigned Opc, DebugLoc DL, SDValue LHS, SDValue RHS);

    SDValue visitShiftByConstant(SDNode *N, unsigned Amt);

    bool SimplifySelectOps(SDNode *SELECT, SDValue LHS, SDValue RHS);
    SDValue SimplifyBinOpWithSameOpcodeHands(SDNode *N);
    SDValue SimplifySelect(DebugLoc DL, SDValue N0, SDValue N1, SDValue N2);
    SDValue SimplifySelectCC(DebugLoc DL, SDValue N0, SDValue N1, SDValue N2,
                             SDValue N3, ISD::CondCode CC,
                             bool NotExtCompare = false);
    SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1, ISD::CondCode Cond,
                          DebugLoc DL, bool foldBooleans = true);
    SDValue SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                         unsigned HiOp);
    SDValue CombineConsecutiveLoads(SDNode *N, EVT VT);
    SDValue ConstantFoldBIT_CONVERTofBUILD_VECTOR(SDNode *, EVT);
    SDValue BuildSDIV(SDNode *N);
    SDValue BuildUDIV(SDNode *N);
    SDNode *MatchRotate(SDValue LHS, SDValue RHS, DebugLoc DL);
    SDValue ReduceLoadWidth(SDNode *N);
    SDValue ReduceLoadOpStoreWidth(SDNode *N);

    SDValue GetDemandedBits(SDValue V, const APInt &Mask);

    /// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDValue OriginalChain,
                          SmallVector<SDValue, 8> &Aliases);

    /// isAlias - Return true if there is any possibility that the two addresses
    /// overlap.
    bool isAlias(SDValue Ptr1, int64_t Size1,
                 const Value *SrcValue1, int SrcValueOffset1,
                 unsigned SrcValueAlign1,
                 SDValue Ptr2, int64_t Size2,
                 const Value *SrcValue2, int SrcValueOffset2,
                 unsigned SrcValueAlign2) const;

    /// FindAliasInfo - Extracts the relevant alias information from the memory
    /// node.  Returns true if the operand was a load.
    bool FindAliasInfo(SDNode *N,
                       SDValue &Ptr, int64_t &Size,
                       const Value *&SrcValue, int &SrcValueOffset,
                       unsigned &SrcValueAlignment) const;

    /// FindBetterChain - Walk up chain skipping non-aliasing memory nodes,
    /// looking for a better chain (aliasing node.)
    SDValue FindBetterChain(SDNode *N, SDValue Chain);

    /// getShiftAmountTy - Returns a type large enough to hold any valid
    /// shift amount - before type legalization these can be huge.
    EVT getShiftAmountTy() {
      return LegalTypes ?  TLI.getShiftAmountTy() : TLI.getPointerTy();
    }

public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis &A, CodeGenOpt::Level OL)
      : DAG(D),
        TLI(D.getTargetLoweringInfo()),
        Level(Unrestricted),
        OptLevel(OL),
        LegalOperations(false),
        LegalTypes(false),
        AA(A) {}

    /// Run - runs the dag combiner on all nodes in the work list
    void Run(CombineLevel AtLevel);
  };
}


namespace {
/// WorkListRemover - This class is a DAGUpdateListener that removes any deleted
/// nodes from the worklist.
class WorkListRemover : public SelectionDAG::DAGUpdateListener {
  DAGCombiner &DC;
public:
  explicit WorkListRemover(DAGCombiner &dc) : DC(dc) {}

  virtual void NodeDeleted(SDNode *N, SDNode *E) {
    DC.removeFromWorkList(N);
  }

  virtual void NodeUpdated(SDNode *N) {
    // Ignore updates.
  }
};
}

//===----------------------------------------------------------------------===//
//  TargetLowering::DAGCombinerInfo implementation
//===----------------------------------------------------------------------===//

void TargetLowering::DAGCombinerInfo::AddToWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->AddToWorkList(N);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, const std::vector<SDValue> &To, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, &To[0], To.size(), AddTo);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res, AddTo);
}


SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res0, SDValue Res1, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res0, Res1, AddTo);
}

void TargetLowering::DAGCombinerInfo::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  return ((DAGCombiner*)DC)->CommitTargetLoweringOpt(TLO);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// isNegatibleForFree - Return 1 if we can compute the negated form of the
/// specified expression for the same cost as the expression itself, or 2 if we
/// can compute the negated form more cheaply than the expression itself.
static char isNegatibleForFree(SDValue Op, bool LegalOperations,
                               unsigned Depth = 0) {
  // No compile time optimizations on this type.
  if (Op.getValueType() == MVT::ppcf128)
    return 0;

  // fneg is removable even if it has multiple uses.
  if (Op.getOpcode() == ISD::FNEG) return 2;

  // Don't allow anything with multiple uses.
  if (!Op.hasOneUse()) return 0;

  // Don't recurse exponentially.
  if (Depth > 6) return 0;

  switch (Op.getOpcode()) {
  default: return false;
  case ISD::ConstantFP:
    // Don't invert constant FP values after legalize.  The negated constant
    // isn't necessarily legal.
    return LegalOperations ? 0 : 1;
  case ISD::FADD:
    // FIXME: determine better conditions for this xform.
    if (!UnsafeFPMath) return 0;

    // fold (fsub (fadd A, B)) -> (fsub (fneg A), B)
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return V;
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return isNegatibleForFree(Op.getOperand(1), LegalOperations, Depth+1);
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    if (!UnsafeFPMath) return 0;

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return 1;

  case ISD::FMUL:
  case ISD::FDIV:
    if (HonorSignDependentRoundingFPMath()) return 0;

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y) or (fmul X, (fneg Y))
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return V;

    return isNegatibleForFree(Op.getOperand(1), LegalOperations, Depth+1);

  case ISD::FP_EXTEND:
  case ISD::FP_ROUND:
  case ISD::FSIN:
    return isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1);
  }
}

/// GetNegatedExpression - If isNegatibleForFree returns true, this function
/// returns the newly negated expression.
static SDValue GetNegatedExpression(SDValue Op, SelectionDAG &DAG,
                                    bool LegalOperations, unsigned Depth = 0) {
  // fneg is removable even if it has multiple uses.
  if (Op.getOpcode() == ISD::FNEG) return Op.getOperand(0);

  // Don't allow anything with multiple uses.
  assert(Op.hasOneUse() && "Unknown reuse!");

  assert(Depth <= 6 && "GetNegatedExpression doesn't match isNegatibleForFree");
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unknown code");
  case ISD::ConstantFP: {
    APFloat V = cast<ConstantFPSDNode>(Op)->getValueAPF();
    V.changeSign();
    return DAG.getConstantFP(V, Op.getValueType());
  }
  case ISD::FADD:
    // FIXME: determine better conditions for this xform.
    assert(UnsafeFPMath);

    // fold (fneg (fadd A, B)) -> (fsub (fneg A), B)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return DAG.getNode(ISD::FSUB, Op.getDebugLoc(), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return DAG.getNode(ISD::FSUB, Op.getDebugLoc(), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1),
                       Op.getOperand(0));
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    assert(UnsafeFPMath);

    // fold (fneg (fsub 0, B)) -> B
    if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(Op.getOperand(0)))
      if (N0CFP->getValueAPF().isZero())
        return Op.getOperand(1);

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return DAG.getNode(ISD::FSUB, Op.getDebugLoc(), Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(0));

  case ISD::FMUL:
  case ISD::FDIV:
    assert(!HonorSignDependentRoundingFPMath());

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return DAG.getNode(Op.getOpcode(), Op.getDebugLoc(), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));

    // fold (fneg (fmul X, Y)) -> (fmul X, (fneg Y))
    return DAG.getNode(Op.getOpcode(), Op.getDebugLoc(), Op.getValueType(),
                       Op.getOperand(0),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1));

  case ISD::FP_EXTEND:
  case ISD::FSIN:
    return DAG.getNode(Op.getOpcode(), Op.getDebugLoc(), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(0), DAG,
                                            LegalOperations, Depth+1));
  case ISD::FP_ROUND:
      return DAG.getNode(ISD::FP_ROUND, Op.getDebugLoc(), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
  }
}


// isSetCCEquivalent - Return true if this node is a setcc, or is a select_cc
// that selects between the values 1 and 0, making it equivalent to a setcc.
// Also, set the incoming LHS, RHS, and CC references to the appropriate
// nodes based on the type of node we are checking.  This simplifies life a
// bit for the callers.
static bool isSetCCEquivalent(SDValue N, SDValue &LHS, SDValue &RHS,
                              SDValue &CC) {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }
  if (N.getOpcode() == ISD::SELECT_CC &&
      N.getOperand(2).getOpcode() == ISD::Constant &&
      N.getOperand(3).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(2))->getAPIntValue() == 1 &&
      cast<ConstantSDNode>(N.getOperand(3))->isNullValue()) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(4);
    return true;
  }
  return false;
}

// isOneUseSetCC - Return true if this is a SetCC-equivalent operation with only
// one use.  If this is true, it allows the users to invert the operation for
// free when it is profitable to do so.
static bool isOneUseSetCC(SDValue N) {
  SDValue N0, N1, N2;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.getNode()->hasOneUse())
    return true;
  return false;
}

SDValue DAGCombiner::ReassociateOps(unsigned Opc, DebugLoc DL,
                                    SDValue N0, SDValue N1) {
  EVT VT = N0.getValueType();
  if (N0.getOpcode() == Opc && isa<ConstantSDNode>(N0.getOperand(1))) {
    if (isa<ConstantSDNode>(N1)) {
      // reassoc. (op (op x, c1), c2) -> (op x, (op c1, c2))
      SDValue OpNode =
        DAG.FoldConstantArithmetic(Opc, VT,
                                   cast<ConstantSDNode>(N0.getOperand(1)),
                                   cast<ConstantSDNode>(N1));
      return DAG.getNode(Opc, DL, VT, N0.getOperand(0), OpNode);
    } else if (N0.hasOneUse()) {
      // reassoc. (op (op x, c1), y) -> (op (op x, y), c1) iff x+c1 has one use
      SDValue OpNode = DAG.getNode(Opc, N0.getDebugLoc(), VT,
                                   N0.getOperand(0), N1);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, DL, VT, OpNode, N0.getOperand(1));
    }
  }

  if (N1.getOpcode() == Opc && isa<ConstantSDNode>(N1.getOperand(1))) {
    if (isa<ConstantSDNode>(N0)) {
      // reassoc. (op c2, (op x, c1)) -> (op x, (op c1, c2))
      SDValue OpNode =
        DAG.FoldConstantArithmetic(Opc, VT,
                                   cast<ConstantSDNode>(N1.getOperand(1)),
                                   cast<ConstantSDNode>(N0));
      return DAG.getNode(Opc, DL, VT, N1.getOperand(0), OpNode);
    } else if (N1.hasOneUse()) {
      // reassoc. (op y, (op x, c1)) -> (op (op x, y), c1) iff x+c1 has one use
      SDValue OpNode = DAG.getNode(Opc, N0.getDebugLoc(), VT,
                                   N1.getOperand(0), N0);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, DL, VT, OpNode, N1.getOperand(1));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                               bool AddTo) {
  assert(N->getNumValues() == NumTo && "Broken CombineTo call!");
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.1 ";
        N->dump(&DAG);
        dbgs() << "\nWith: ";
        To[0].getNode()->dump(&DAG);
        dbgs() << " and " << NumTo-1 << " other values\n";
        for (unsigned i = 0, e = NumTo; i != e; ++i)
          assert((!To[i].getNode() ||
                  N->getValueType(i) == To[i].getValueType()) &&
                 "Cannot combine value to value of different type!"));
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesWith(N, To, &DeadNodes);

  if (AddTo) {
    // Push the new nodes and any users onto the worklist
    for (unsigned i = 0, e = NumTo; i != e; ++i) {
      if (To[i].getNode()) {
        AddToWorkList(To[i].getNode());
        AddUsersToWorkList(To[i].getNode());
      }
    }
  }

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (N->use_empty()) {
    // Nodes can be reintroduced into the worklist.  Make sure we do not
    // process a node that has been replaced.
    removeFromWorkList(N);

    // Finally, since the node is now dead, remove it from the graph.
    DAG.DeleteNode(N);
  }
  return SDValue(N, 0);
}

void
DAGCombiner::CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &
                                                                          TLO) {
  // Replace all uses.  If any nodes become isomorphic to other nodes and
  // are deleted, make sure to remove them from our worklist.
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(TLO.Old, TLO.New, &DeadNodes);

  // Push the new node and any (possibly new) users onto the worklist.
  AddToWorkList(TLO.New.getNode());
  AddUsersToWorkList(TLO.New.getNode());

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (TLO.Old.getNode()->use_empty()) {
    removeFromWorkList(TLO.Old.getNode());

    // If the operands of this node are only used by the node, they will now
    // be dead.  Make sure to visit them first to delete dead nodes early.
    for (unsigned i = 0, e = TLO.Old.getNode()->getNumOperands(); i != e; ++i)
      if (TLO.Old.getNode()->getOperand(i).getNode()->hasOneUse())
        AddToWorkList(TLO.Old.getNode()->getOperand(i).getNode());

    DAG.DeleteNode(TLO.Old.getNode());
  }
}

/// SimplifyDemandedBits - Check the specified integer node value to see if
/// it can be simplified or if things it uses can be simplified by bit
/// propagation.  If so, return true.
bool DAGCombiner::SimplifyDemandedBits(SDValue Op, const APInt &Demanded) {
  TargetLowering::TargetLoweringOpt TLO(DAG);
  APInt KnownZero, KnownOne;
  if (!TLI.SimplifyDemandedBits(Op, Demanded, KnownZero, KnownOne, TLO))
    return false;

  // Revisit the node.
  AddToWorkList(Op.getNode());

  // Replace the old value with the new one.
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.2 "; 
        TLO.Old.getNode()->dump(&DAG);
        dbgs() << "\nWith: ";
        TLO.New.getNode()->dump(&DAG);
        dbgs() << '\n');

  CommitTargetLoweringOpt(TLO);
  return true;
}

//===----------------------------------------------------------------------===//
//  Main DAG Combiner implementation
//===----------------------------------------------------------------------===//

void DAGCombiner::Run(CombineLevel AtLevel) {
  // set the instance variables, so that the various visit routines may use it.
  Level = AtLevel;
  LegalOperations = Level >= NoIllegalOperations;
  LegalTypes = Level >= NoIllegalTypes;

  // Add all the dag nodes to the worklist.
  WorkList.reserve(DAG.allnodes_size());
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I)
    WorkList.push_back(I);

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // The root of the dag may dangle to deleted nodes until the dag combiner is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDValue());

  // while the worklist isn't empty, inspect the node on the end of it and
  // try and combine it.
  while (!WorkList.empty()) {
    SDNode *N = WorkList.back();
    WorkList.pop_back();

    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead or may have a
    // reduced number of uses, allowing other xforms.
    if (N->use_empty() && N != &Dummy) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        AddToWorkList(N->getOperand(i).getNode());

      DAG.DeleteNode(N);
      continue;
    }

    SDValue RV = combine(N);

    if (RV.getNode() == 0)
      continue;

    ++NodesCombined;

    // If we get back the same node we passed in, rather than a new node or
    // zero, we know that the node must have defined multiple values and
    // CombineTo was used.  Since CombineTo takes care of the worklist
    // mechanics for us, we have no work to do in this case.
    if (RV.getNode() == N)
      continue;

    assert(N->getOpcode() != ISD::DELETED_NODE &&
           RV.getNode()->getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned new node!");

    DEBUG(dbgs() << "\nReplacing.3 "; 
          N->dump(&DAG);
          dbgs() << "\nWith: ";
          RV.getNode()->dump(&DAG);
          dbgs() << '\n');
    WorkListRemover DeadNodes(*this);
    if (N->getNumValues() == RV.getNode()->getNumValues())
      DAG.ReplaceAllUsesWith(N, RV.getNode(), &DeadNodes);
    else {
      assert(N->getValueType(0) == RV.getValueType() &&
             N->getNumValues() == 1 && "Type mismatch");
      SDValue OpV = RV;
      DAG.ReplaceAllUsesWith(N, &OpV, &DeadNodes);
    }

    // Push the new node and any users onto the worklist
    AddToWorkList(RV.getNode());
    AddUsersToWorkList(RV.getNode());

    // Add any uses of the old node to the worklist in case this node is the
    // last one that uses them.  They may become dead after this node is
    // deleted.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      AddToWorkList(N->getOperand(i).getNode());

    // Finally, if the node is now dead, remove it from the graph.  The node
    // may not be dead if the replacement process recursively simplified to
    // something else needing this node.
    if (N->use_empty()) {
      // Nodes can be reintroduced into the worklist.  Make sure we do not
      // process a node that has been replaced.
      removeFromWorkList(N);

      // Finally, since the node is now dead, remove it from the graph.
      DAG.DeleteNode(N);
    }
  }

  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());
}

SDValue DAGCombiner::visit(SDNode *N) {
  switch(N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::MERGE_VALUES:       return visitMERGE_VALUES(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::ADDC:               return visitADDC(N);
  case ISD::ADDE:               return visitADDE(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:               return visitSREM(N);
  case ISD::UREM:               return visitUREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::SMUL_LOHI:          return visitSMUL_LOHI(N);
  case ISD::UMUL_LOHI:          return visitUMUL_LOHI(N);
  case ISD::SDIVREM:            return visitSDIVREM(N);
  case ISD::UDIVREM:            return visitUDIVREM(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SELECT:             return visitSELECT(N);
  case ISD::SELECT_CC:          return visitSELECT_CC(N);
  case ISD::SETCC:              return visitSETCC(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::ANY_EXTEND:         return visitANY_EXTEND(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::BIT_CONVERT:        return visitBIT_CONVERT(N);
  case ISD::BUILD_PAIR:         return visitBUILD_PAIR(N);
  case ISD::FADD:               return visitFADD(N);
  case ISD::FSUB:               return visitFSUB(N);
  case ISD::FMUL:               return visitFMUL(N);
  case ISD::FDIV:               return visitFDIV(N);
  case ISD::FREM:               return visitFREM(N);
  case ISD::FCOPYSIGN:          return visitFCOPYSIGN(N);
  case ISD::SINT_TO_FP:         return visitSINT_TO_FP(N);
  case ISD::UINT_TO_FP:         return visitUINT_TO_FP(N);
  case ISD::FP_TO_SINT:         return visitFP_TO_SINT(N);
  case ISD::FP_TO_UINT:         return visitFP_TO_UINT(N);
  case ISD::FP_ROUND:           return visitFP_ROUND(N);
  case ISD::FP_ROUND_INREG:     return visitFP_ROUND_INREG(N);
  case ISD::FP_EXTEND:          return visitFP_EXTEND(N);
  case ISD::FNEG:               return visitFNEG(N);
  case ISD::FABS:               return visitFABS(N);
  case ISD::BRCOND:             return visitBRCOND(N);
  case ISD::BR_CC:              return visitBR_CC(N);
  case ISD::LOAD:               return visitLOAD(N);
  case ISD::STORE:              return visitSTORE(N);
  case ISD::INSERT_VECTOR_ELT:  return visitINSERT_VECTOR_ELT(N);
  case ISD::EXTRACT_VECTOR_ELT: return visitEXTRACT_VECTOR_ELT(N);
  case ISD::BUILD_VECTOR:       return visitBUILD_VECTOR(N);
  case ISD::CONCAT_VECTORS:     return visitCONCAT_VECTORS(N);
  case ISD::VECTOR_SHUFFLE:     return visitVECTOR_SHUFFLE(N);
  }
  return SDValue();
}

SDValue DAGCombiner::combine(SDNode *N) {
  SDValue RV = visit(N);

  // If nothing happened, try a target-specific DAG combine.
  if (RV.getNode() == 0) {
    assert(N->getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned NULL!");

    if (N->getOpcode() >= ISD::BUILTIN_OP_END ||
        TLI.hasTargetDAGCombine((ISD::NodeType)N->getOpcode())) {

      // Expose the DAG combiner to the target combiner impls.
      TargetLowering::DAGCombinerInfo
        DagCombineInfo(DAG, !LegalTypes, !LegalOperations, false, this);

      RV = TLI.PerformDAGCombine(N, DagCombineInfo);
    }
  }

  // If N is a commutative binary node, try commuting it to enable more
  // sdisel CSE.
  if (RV.getNode() == 0 &&
      SelectionDAG::isCommutativeBinOp(N->getOpcode()) &&
      N->getNumValues() == 1) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);

    // Constant operands are canonicalized to RHS.
    if (isa<ConstantSDNode>(N0) || !isa<ConstantSDNode>(N1)) {
      SDValue Ops[] = { N1, N0 };
      SDNode *CSENode = DAG.getNodeIfExists(N->getOpcode(), N->getVTList(),
                                            Ops, 2);
      if (CSENode)
        return SDValue(CSENode, 0);
    }
  }

  return RV;
}

/// getInputChainForNode - Given a node, return its input chain if it has one,
/// otherwise return a null sd operand.
static SDValue getInputChainForNode(SDNode *N) {
  if (unsigned NumOps = N->getNumOperands()) {
    if (N->getOperand(0).getValueType() == MVT::Other)
      return N->getOperand(0);
    else if (N->getOperand(NumOps-1).getValueType() == MVT::Other)
      return N->getOperand(NumOps-1);
    for (unsigned i = 1; i < NumOps-1; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        return N->getOperand(i);
  }
  return SDValue();
}

SDValue DAGCombiner::visitTokenFactor(SDNode *N) {
  // If N has two operands, where one has an input chain equal to the other,
  // the 'other' chain is redundant.
  if (N->getNumOperands() == 2) {
    if (getInputChainForNode(N->getOperand(0).getNode()) == N->getOperand(1))
      return N->getOperand(0);
    if (getInputChainForNode(N->getOperand(1).getNode()) == N->getOperand(0))
      return N->getOperand(1);
  }

  SmallVector<SDNode *, 8> TFs;     // List of token factors to visit.
  SmallVector<SDValue, 8> Ops;    // Ops for replacing token factor.
  SmallPtrSet<SDNode*, 16> SeenOps;
  bool Changed = false;             // If we should replace this token factor.

  // Start out with this token factor.
  TFs.push_back(N);

  // Iterate through token factors.  The TFs grows when new token factors are
  // encountered.
  for (unsigned i = 0; i < TFs.size(); ++i) {
    SDNode *TF = TFs[i];

    // Check each of the operands.
    for (unsigned i = 0, ie = TF->getNumOperands(); i != ie; ++i) {
      SDValue Op = TF->getOperand(i);

      switch (Op.getOpcode()) {
      case ISD::EntryToken:
        // Entry tokens don't need to be added to the list. They are
        // rededundant.
        Changed = true;
        break;

      case ISD::TokenFactor:
        if (Op.hasOneUse() &&
            std::find(TFs.begin(), TFs.end(), Op.getNode()) == TFs.end()) {
          // Queue up for processing.
          TFs.push_back(Op.getNode());
          // Clean up in case the token factor is removed.
          AddToWorkList(Op.getNode());
          Changed = true;
          break;
        }
        // Fall thru

      default:
        // Only add if it isn't already in the list.
        if (SeenOps.insert(Op.getNode()))
          Ops.push_back(Op);
        else
          Changed = true;
        break;
      }
    }
  }
  
  SDValue Result;

  // If we've change things around then replace token factor.
  if (Changed) {
    if (Ops.empty()) {
      // The entry token is the only possible outcome.
      Result = DAG.getEntryNode();
    } else {
      // New and improved token factor.
      Result = DAG.getNode(ISD::TokenFactor, N->getDebugLoc(),
                           MVT::Other, &Ops[0], Ops.size());
    }

    // Don't add users to work list.
    return CombineTo(N, Result, false);
  }

  return Result;
}

/// MERGE_VALUES can always be eliminated.
SDValue DAGCombiner::visitMERGE_VALUES(SDNode *N) {
  WorkListRemover DeadNodes(*this);
  // Replacing results may cause a different MERGE_VALUES to suddenly
  // be CSE'd with N, and carry its uses with it. Iterate until no
  // uses remain, to ensure that the node can be safely deleted.
  do {
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, i), N->getOperand(i),
                                    &DeadNodes);
  } while (!N->use_empty());
  removeFromWorkList(N);
  DAG.DeleteNode(N);
  return SDValue(N, 0);   // Return N so it doesn't get rechecked!
}

static
SDValue combineShlAddConstant(DebugLoc DL, SDValue N0, SDValue N1,
                              SelectionDAG &DAG) {
  EVT VT = N0.getValueType();
  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);
  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N01);

  if (N01C && N00.getOpcode() == ISD::ADD && N00.getNode()->hasOneUse() &&
      isa<ConstantSDNode>(N00.getOperand(1))) {
    // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
    N0 = DAG.getNode(ISD::ADD, N0.getDebugLoc(), VT,
                     DAG.getNode(ISD::SHL, N00.getDebugLoc(), VT,
                                 N00.getOperand(0), N01),
                     DAG.getNode(ISD::SHL, N01.getDebugLoc(), VT,
                                 N00.getOperand(1), N01));
    return DAG.getNode(ISD::ADD, DL, VT, N0, N1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (add x, undef) -> undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;
  // fold (add c1, c2) -> c1+c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::ADD, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N1, N0);
  // fold (add x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (add Sym, c) -> Sym+c
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA) && N1C &&
        GA->getOpcode() == ISD::GlobalAddress)
      return DAG.getGlobalAddress(GA->getGlobal(), VT,
                                  GA->getOffset() +
                                    (uint64_t)N1C->getSExtValue());
  // fold ((c1-A)+c2) -> (c1+c2)-A
  if (N1C && N0.getOpcode() == ISD::SUB)
    if (ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getOperand(0)))
      return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                         DAG.getConstant(N1C->getAPIntValue()+
                                         N0C->getAPIntValue(), VT),
                         N0.getOperand(1));
  // reassociate add
  SDValue RADD = ReassociateOps(ISD::ADD, N->getDebugLoc(), N0, N1);
  if (RADD.getNode() != 0)
    return RADD;
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);
  // fold ((B-A)+A) -> B
  if (N0.getOpcode() == ISD::SUB && N1 == N0.getOperand(1))
    return N0.getOperand(0);
  // fold (A+(B-(A+C))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(0))
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(1));
  // fold (A+(B-(C+A))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(1))
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(0));
  // fold (A+((B-A)+or-C)) to (B+or-C)
  if ((N1.getOpcode() == ISD::SUB || N1.getOpcode() == ISD::ADD) &&
      N1.getOperand(0).getOpcode() == ISD::SUB &&
      N0 == N1.getOperand(0).getOperand(1))
    return DAG.getNode(N1.getOpcode(), N->getDebugLoc(), VT,
                       N1.getOperand(0).getOperand(0), N1.getOperand(1));

  // fold (A-B)+(C-D) to (A+C)-(B+D) when A or C is constant
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB) {
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    SDValue N10 = N1.getOperand(0);
    SDValue N11 = N1.getOperand(1);

    if (isa<ConstantSDNode>(N00) || isa<ConstantSDNode>(N10))
      return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                         DAG.getNode(ISD::ADD, N0.getDebugLoc(), VT, N00, N10),
                         DAG.getNode(ISD::ADD, N1.getDebugLoc(), VT, N01, N11));
  }

  if (!VT.isVector() && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (a+b) -> (a|b) iff a and b share no bits.
  if (VT.isInteger() && !VT.isVector()) {
    APInt LHSZero, LHSOne;
    APInt RHSZero, RHSOne;
    APInt Mask = APInt::getAllOnesValue(VT.getScalarType().getSizeInBits());
    DAG.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);

    if (LHSZero.getBoolValue()) {
      DAG.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);

      // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
      // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
      if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
          (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
        return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, N0, N1);
    }
  }

  // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
  if (N0.getOpcode() == ISD::SHL && N0.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(N->getDebugLoc(), N0, N1, DAG);
    if (Result.getNode()) return Result;
  }
  if (N1.getOpcode() == ISD::SHL && N1.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(N->getDebugLoc(), N1, N0, DAG);
    if (Result.getNode()) return Result;
  }

  // fold (add x, shl(0 - y, n)) -> sub(x, shl(y, n))
  if (N1.getOpcode() == ISD::SHL &&
      N1.getOperand(0).getOpcode() == ISD::SUB)
    if (ConstantSDNode *C =
          dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(0)))
      if (C->getAPIntValue() == 0)
        return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N0,
                           DAG.getNode(ISD::SHL, N->getDebugLoc(), VT,
                                       N1.getOperand(0).getOperand(1),
                                       N1.getOperand(1)));
  if (N0.getOpcode() == ISD::SHL &&
      N0.getOperand(0).getOpcode() == ISD::SUB)
    if (ConstantSDNode *C =
          dyn_cast<ConstantSDNode>(N0.getOperand(0).getOperand(0)))
      if (C->getAPIntValue() == 0)
        return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N1,
                           DAG.getNode(ISD::SHL, N->getDebugLoc(), VT,
                                       N0.getOperand(0).getOperand(1),
                                       N0.getOperand(1)));

  return SDValue();
}

SDValue DAGCombiner::visitADDC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // If the flag result is dead, turn this into an ADD.
  if (N->hasNUsesOfValue(0, 1))
    return CombineTo(N, DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE,
                                 N->getDebugLoc(), MVT::Flag));

  // canonicalize constant to RHS.
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDC, N->getDebugLoc(), N->getVTList(), N1, N0);

  // fold (addc x, 0) -> x + no carry out
  if (N1C && N1C->isNullValue())
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE,
                                        N->getDebugLoc(), MVT::Flag));

  // fold (addc a, b) -> (or a, b), CARRY_FALSE iff a and b share no bits.
  APInt LHSZero, LHSOne;
  APInt RHSZero, RHSOne;
  APInt Mask = APInt::getAllOnesValue(VT.getScalarType().getSizeInBits());
  DAG.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);

  if (LHSZero.getBoolValue()) {
    DAG.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);

    // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
    // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
    if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
        (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
      return CombineTo(N, DAG.getNode(ISD::OR, N->getDebugLoc(), VT, N0, N1),
                       DAG.getNode(ISD::CARRY_FALSE,
                                   N->getDebugLoc(), MVT::Flag));
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);

  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDE, N->getDebugLoc(), N->getVTList(),
                       N1, N0, CarryIn);

  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::ADDC, N->getDebugLoc(), N->getVTList(), N1, N0);

  return SDValue();
}

SDValue DAGCombiner::visitSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (sub x, x) -> 0
  if (N0 == N1)
    return DAG.getConstant(0, N->getValueType(0));
  // fold (sub c1, c2) -> c1-c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SUB, VT, N0C, N1C);
  // fold (sub x, c) -> (add x, -c)
  if (N1C)
    return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N0,
                       DAG.getConstant(-N1C->getAPIntValue(), VT));
  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1)
  if (N0C && N0C->isAllOnesValue())
    return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT, N1, N0);
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);
  // fold ((A+(B+or-C))-B) -> A+or-C
  if (N0.getOpcode() == ISD::ADD &&
      (N0.getOperand(1).getOpcode() == ISD::SUB ||
       N0.getOperand(1).getOpcode() == ISD::ADD) &&
      N0.getOperand(1).getOperand(0) == N1)
    return DAG.getNode(N0.getOperand(1).getOpcode(), N->getDebugLoc(), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(1));
  // fold ((A+(C+B))-B) -> A+C
  if (N0.getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(0));
  // fold ((A-(B-C))-C) -> A-B
  if (N0.getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(0));

  // If either operand of a sub is undef, the result is undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  // If the relocation model supports it, consider symbol offsets.
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA)) {
      // fold (sub Sym, c) -> Sym-c
      if (N1C && GA->getOpcode() == ISD::GlobalAddress)
        return DAG.getGlobalAddress(GA->getGlobal(), VT,
                                    GA->getOffset() -
                                      (uint64_t)N1C->getSExtValue());
      // fold (sub Sym+c1, Sym+c2) -> c1-c2
      if (GlobalAddressSDNode *GB = dyn_cast<GlobalAddressSDNode>(N1))
        if (GA->getGlobal() == GB->getGlobal())
          return DAG.getConstant((uint64_t)GA->getOffset() - GB->getOffset(),
                                 VT);
    }

  return SDValue();
}

SDValue DAGCombiner::visitMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (mul x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // fold (mul c1, c2) -> c1*c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::MUL, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::MUL, N->getDebugLoc(), VT, N1, N0);
  // fold (mul x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mul x, -1) -> 0-x
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                       DAG.getConstant(0, VT), N0);
  // fold (mul x, (1 << c)) -> x << c
  if (N1C && N1C->getAPIntValue().isPowerOf2())
    return DAG.getNode(ISD::SHL, N->getDebugLoc(), VT, N0,
                       DAG.getConstant(N1C->getAPIntValue().logBase2(),
                                       getShiftAmountTy()));
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1C && (-N1C->getAPIntValue()).isPowerOf2()) {
    unsigned Log2Val = (-N1C->getAPIntValue()).logBase2();
    // FIXME: If the input is something that is easily negated (e.g. a
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                       DAG.getConstant(0, VT),
                       DAG.getNode(ISD::SHL, N->getDebugLoc(), VT, N0,
                            DAG.getConstant(Log2Val, getShiftAmountTy())));
  }
  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N1C && N0.getOpcode() == ISD::SHL &&
      isa<ConstantSDNode>(N0.getOperand(1))) {
    SDValue C3 = DAG.getNode(ISD::SHL, N->getDebugLoc(), VT,
                             N1, N0.getOperand(1));
    AddToWorkList(C3.getNode());
    return DAG.getNode(ISD::MUL, N->getDebugLoc(), VT,
                       N0.getOperand(0), C3);
  }

  // Change (mul (shl X, C), Y) -> (shl (mul X, Y), C) when the shift has one
  // use.
  {
    SDValue Sh(0,0), Y(0,0);
    // Check for both (mul (shl X, C), Y)  and  (mul Y, (shl X, C)).
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.getNode()->hasOneUse()) {
      Sh = N0; Y = N1;
    } else if (N1.getOpcode() == ISD::SHL &&
               isa<ConstantSDNode>(N1.getOperand(1)) &&
               N1.getNode()->hasOneUse()) {
      Sh = N1; Y = N0;
    }

    if (Sh.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, N->getDebugLoc(), VT,
                                Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, N->getDebugLoc(), VT,
                         Mul, Sh.getOperand(1));
    }
  }

  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (N1C && N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse() &&
      isa<ConstantSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT,
                       DAG.getNode(ISD::MUL, N0.getDebugLoc(), VT,
                                   N0.getOperand(0), N1),
                       DAG.getNode(ISD::MUL, N1.getDebugLoc(), VT,
                                   N0.getOperand(1), N1));

  // reassociate mul
  SDValue RMUL = ReassociateOps(ISD::MUL, N->getDebugLoc(), N0, N1);
  if (RMUL.getNode() != 0)
    return RMUL;

  return SDValue();
}

SDValue DAGCombiner::visitSDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (sdiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::SDIV, VT, N0C, N1C);
  // fold (sdiv X, 1) -> X
  if (N1C && N1C->getSExtValue() == 1LL)
    return N0;
  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                       DAG.getConstant(0, VT), N0);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UDIV, N->getDebugLoc(), N1.getValueType(),
                         N0, N1);
  }
  // fold (sdiv X, pow2) -> simple ops after legalize
  if (N1C && !N1C->isNullValue() && !TLI.isIntDivCheap() &&
      (isPowerOf2_64(N1C->getSExtValue()) ||
       isPowerOf2_64(-N1C->getSExtValue()))) {
    // If dividing by powers of two is cheap, then don't perform the following
    // fold.
    if (TLI.isPow2DivCheap())
      return SDValue();

    int64_t pow2 = N1C->getSExtValue();
    int64_t abs2 = pow2 > 0 ? pow2 : -pow2;
    unsigned lg2 = Log2_64(abs2);

    // Splat the sign bit into the register
    SDValue SGN = DAG.getNode(ISD::SRA, N->getDebugLoc(), VT, N0,
                              DAG.getConstant(VT.getSizeInBits()-1,
                                              getShiftAmountTy()));
    AddToWorkList(SGN.getNode());

    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDValue SRL = DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, SGN,
                              DAG.getConstant(VT.getSizeInBits() - lg2,
                                              getShiftAmountTy()));
    SDValue ADD = DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N0, SRL);
    AddToWorkList(SRL.getNode());
    AddToWorkList(ADD.getNode());    // Divide by pow2
    SDValue SRA = DAG.getNode(ISD::SRA, N->getDebugLoc(), VT, ADD,
                              DAG.getConstant(lg2, getShiftAmountTy()));

    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (pow2 > 0)
      return SRA;

    AddToWorkList(SRA.getNode());
    return DAG.getNode(ISD::SUB, N->getDebugLoc(), VT,
                       DAG.getConstant(0, VT), SRA);
  }

  // if integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.
  if (N1C && (N1C->getSExtValue() < -1 || N1C->getSExtValue() > 1) &&
      !TLI.isIntDivCheap()) {
    SDValue Op = BuildSDIV(N);
    if (Op.getNode()) return Op;
  }

  // undef / X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // X / undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitUDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (udiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::UDIV, VT, N0C, N1C);
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N1C && N1C->getAPIntValue().isPowerOf2())
    return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0,
                       DAG.getConstant(N1C->getAPIntValue().logBase2(),
                                       getShiftAmountTy()));
  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        EVT ADDVT = N1.getOperand(1).getValueType();
        SDValue Add = DAG.getNode(ISD::ADD, N->getDebugLoc(), ADDVT,
                                  N1.getOperand(1),
                                  DAG.getConstant(SHC->getAPIntValue()
                                                                  .logBase2(),
                                                  ADDVT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0, Add);
      }
    }
  }
  // fold (udiv x, c) -> alternate
  if (N1C && !N1C->isNullValue() && !TLI.isIntDivCheap()) {
    SDValue Op = BuildUDIV(N);
    if (Op.getNode()) return Op;
  }

  // undef / X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // X / undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitSREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (srem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::SREM, VT, N0C, N1C);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UREM, N->getDebugLoc(), VT, N0, N1);
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::SDIV, N->getDebugLoc(), VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, N->getDebugLoc(), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N0, Mul);
      AddToWorkList(Mul.getNode());
      return Sub;
    }
  }

  // undef % X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // X % undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitUREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (urem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::UREM, VT, N0C, N1C);
  // fold (urem x, pow2) -> (and x, pow2-1)
  if (N1C && !N1C->isNullValue() && N1C->getAPIntValue().isPowerOf2())
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N0,
                       DAG.getConstant(N1C->getAPIntValue()-1,VT));
  // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        SDValue Add =
          DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N1,
                 DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()),
                                 VT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N0, Add);
      }
    }
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::UDIV, N->getDebugLoc(), VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, N->getDebugLoc(), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, N->getDebugLoc(), VT, N0, Mul);
      AddToWorkList(Mul.getNode());
      return Sub;
    }
  }

  // undef % X -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // X % undef -> undef
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;

  return SDValue();
}

SDValue DAGCombiner::visitMULHS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::SRA, N->getDebugLoc(), N0.getValueType(), N0,
                       DAG.getConstant(N0.getValueType().getSizeInBits() - 1,
                                       getShiftAmountTy()));
  // fold (mulhs x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  return SDValue();
}

SDValue DAGCombiner::visitMULHU(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (mulhu x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhu x, 1) -> 0
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getConstant(0, N0.getValueType());
  // fold (mulhu x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  return SDValue();
}

/// SimplifyNodeWithTwoResults - Perform optimizations common to nodes that
/// compute two values. LoOp and HiOp give the opcodes for the two computations
/// that are being performed. Return true if a simplification was made.
///
SDValue DAGCombiner::SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                                unsigned HiOp) {
  // If the high half is not needed, just compute the low half.
  bool HiExists = N->hasAnyUseOfValue(1);
  if (!HiExists &&
      (!LegalOperations ||
       TLI.isOperationLegal(LoOp, N->getValueType(0)))) {
    SDValue Res = DAG.getNode(LoOp, N->getDebugLoc(), N->getValueType(0),
                              N->op_begin(), N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If the low half is not needed, just compute the high half.
  bool LoExists = N->hasAnyUseOfValue(0);
  if (!LoExists &&
      (!LegalOperations ||
       TLI.isOperationLegal(HiOp, N->getValueType(1)))) {
    SDValue Res = DAG.getNode(HiOp, N->getDebugLoc(), N->getValueType(1),
                              N->op_begin(), N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If both halves are used, return as it is.
  if (LoExists && HiExists)
    return SDValue();

  // If the two computed results can be simplified separately, separate them.
  if (LoExists) {
    SDValue Lo = DAG.getNode(LoOp, N->getDebugLoc(), N->getValueType(0),
                             N->op_begin(), N->getNumOperands());
    AddToWorkList(Lo.getNode());
    SDValue LoOpt = combine(Lo.getNode());
    if (LoOpt.getNode() && LoOpt.getNode() != Lo.getNode() &&
        (!LegalOperations ||
         TLI.isOperationLegal(LoOpt.getOpcode(), LoOpt.getValueType())))
      return CombineTo(N, LoOpt, LoOpt);
  }

  if (HiExists) {
    SDValue Hi = DAG.getNode(HiOp, N->getDebugLoc(), N->getValueType(1),
                             N->op_begin(), N->getNumOperands());
    AddToWorkList(Hi.getNode());
    SDValue HiOpt = combine(Hi.getNode());
    if (HiOpt.getNode() && HiOpt != Hi &&
        (!LegalOperations ||
         TLI.isOperationLegal(HiOpt.getOpcode(), HiOpt.getValueType())))
      return CombineTo(N, HiOpt, HiOpt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSMUL_LOHI(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHS);
  if (Res.getNode()) return Res;

  return SDValue();
}

SDValue DAGCombiner::visitUMUL_LOHI(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHU);
  if (Res.getNode()) return Res;

  return SDValue();
}

SDValue DAGCombiner::visitSDIVREM(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::SDIV, ISD::SREM);
  if (Res.getNode()) return Res;

  return SDValue();
}

SDValue DAGCombiner::visitUDIVREM(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::UDIV, ISD::UREM);
  if (Res.getNode()) return Res;

  return SDValue();
}

/// SimplifyBinOpWithSameOpcodeHands - If this is a binary operator with
/// two operands of the same opcode, try to simplify it.
SDValue DAGCombiner::SimplifyBinOpWithSameOpcodeHands(SDNode *N) {
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  assert(N0.getOpcode() == N1.getOpcode() && "Bad input!");

  // Bail early if none of these transforms apply.
  if (N0.getNode()->getNumOperands() == 0) return SDValue();

  // For each of OP in AND/OR/XOR:
  // fold (OP (zext x), (zext y)) -> (zext (OP x, y))
  // fold (OP (sext x), (sext y)) -> (sext (OP x, y))
  // fold (OP (aext x), (aext y)) -> (aext (OP x, y))
  // fold (OP (trunc x), (trunc y)) -> (trunc (OP x, y))
  //
  // do not sink logical op inside of a vector extend, since it may combine
  // into a vsetcc.
  EVT Op0VT = N0.getOperand(0).getValueType();
  if ((N0.getOpcode() == ISD::ZERO_EXTEND ||
       N0.getOpcode() == ISD::ANY_EXTEND  ||
       N0.getOpcode() == ISD::SIGN_EXTEND ||
       (N0.getOpcode() == ISD::TRUNCATE && TLI.isTypeLegal(Op0VT))) &&
      !VT.isVector() &&
      Op0VT == N1.getOperand(0).getValueType() &&
      (!LegalOperations || TLI.isOperationLegal(N->getOpcode(), Op0VT))) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), N0.getDebugLoc(),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), N->getDebugLoc(), VT, ORNode);
  }

  // For each of OP in SHL/SRL/SRA/AND...
  //   fold (and (OP x, z), (OP y, z)) -> (OP (and x, y), z)
  //   fold (or  (OP x, z), (OP y, z)) -> (OP (or  x, y), z)
  //   fold (xor (OP x, z), (OP y, z)) -> (OP (xor x, y), z)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL ||
       N0.getOpcode() == ISD::SRA || N0.getOpcode() == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), N0.getDebugLoc(),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), N->getDebugLoc(), VT,
                       ORNode, N0.getOperand(1));
  }

  return SDValue();
}

SDValue DAGCombiner::visitAND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N1.getValueType();
  unsigned BitWidth = VT.getScalarType().getSizeInBits();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (and x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // fold (and c1, c2) -> c1&c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::AND, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N1, N0);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(BitWidth)))
    return DAG.getConstant(0, VT);
  // reassociate and
  SDValue RAND = ReassociateOps(ISD::AND, N->getDebugLoc(), N0, N1);
  if (RAND.getNode() != 0)
    return RAND;
  // fold (and (or x, C), D) -> D if (C & D) == D
  if (N1C && N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getAPIntValue() & N1C->getAPIntValue()) == N1C->getAPIntValue())
        return N1;
  // fold (and (any_ext V), c) -> (zero_ext V) if 'and' only clears top bits.
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N0Op0 = N0.getOperand(0);
    APInt Mask = ~N1C->getAPIntValue();
    Mask.trunc(N0Op0.getValueSizeInBits());
    if (DAG.MaskedValueIsZero(N0Op0, Mask)) {
      SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(),
                                 N0.getValueType(), N0Op0);

      // Replace uses of the AND with uses of the Zero extend node.
      CombineTo(N, Zext);

      // We actually want to replace all uses of the any_extend with the
      // zero_extend, to avoid duplicating things.  This will later cause this
      // AND to be folded.
      CombineTo(N0.getNode(), Zext);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (and (setcc x), (setcc y)) -> (setcc (and x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();

    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        LL.getValueType().isInteger()) {
      // fold (and (seteq X, 0), (seteq Y, 0)) -> (seteq (or X, Y), 0)
      if (cast<ConstantSDNode>(LR)->isNullValue() && Op1 == ISD::SETEQ) {
        SDValue ORNode = DAG.getNode(ISD::OR, N0.getDebugLoc(),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(N->getDebugLoc(), VT, ORNode, LR, Op1);
      }
      // fold (and (seteq X, -1), (seteq Y, -1)) -> (seteq (and X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETEQ) {
        SDValue ANDNode = DAG.getNode(ISD::AND, N0.getDebugLoc(),
                                      LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(N->getDebugLoc(), VT, ANDNode, LR, Op1);
      }
      // fold (and (setgt X,  -1), (setgt Y,  -1)) -> (setgt (or X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETGT) {
        SDValue ORNode = DAG.getNode(ISD::OR, N0.getDebugLoc(),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(N->getDebugLoc(), VT, ORNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = LL.getValueType().isInteger();
      ISD::CondCode Result = ISD::getSetCCAndOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID &&
          (!LegalOperations || TLI.isCondCodeLegal(Result, LL.getValueType())))
        return DAG.getSetCC(N->getDebugLoc(), N0.getValueType(),
                            LL, LR, Result);
    }
  }

  // Simplify: (and (op x...), (op y...))  -> (op (and x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  // fold (and (sra)) -> (and (srl)) when possible.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (zext_inreg (extload x)) -> (zextload x)
  if (ISD::isEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode())) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueType().getScalarType().getSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                           BitWidth - MemVT.getScalarType().getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, MemVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, N0.getDebugLoc(), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), MemVT,
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (ISD::isSEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueType().getScalarType().getSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                           BitWidth - MemVT.getScalarType().getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, MemVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, N0.getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), MemVT,
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (and (load x), 255) -> (zextload x, i8)
  // fold (and (extload x, i16), 255) -> (zextload x, i8)
  // fold (and (any_ext (extload x, i16)), 255) -> (zextload x, i8)
  if (N1C && (N0.getOpcode() == ISD::LOAD ||
              (N0.getOpcode() == ISD::ANY_EXTEND &&
               N0.getOperand(0).getOpcode() == ISD::LOAD))) {
    bool HasAnyExt = N0.getOpcode() == ISD::ANY_EXTEND;
    LoadSDNode *LN0 = HasAnyExt
      ? cast<LoadSDNode>(N0.getOperand(0))
      : cast<LoadSDNode>(N0);
    if (LN0->getExtensionType() != ISD::SEXTLOAD &&
        LN0->isUnindexed() && N0.hasOneUse() && LN0->hasOneUse()) {
      uint32_t ActiveBits = N1C->getAPIntValue().getActiveBits();
      if (ActiveBits > 0 && APIntOps::isMask(ActiveBits, N1C->getAPIntValue())){
        EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
        EVT LoadedVT = LN0->getMemoryVT();

        if (ExtVT == LoadedVT &&
            (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, ExtVT))) {
          EVT LoadResultTy = HasAnyExt ? LN0->getValueType(0) : VT;
          
          SDValue NewLoad = 
            DAG.getExtLoad(ISD::ZEXTLOAD, LN0->getDebugLoc(), LoadResultTy,
                           LN0->getChain(), LN0->getBasePtr(),
                           LN0->getSrcValue(), LN0->getSrcValueOffset(),
                           ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                           LN0->getAlignment());
          AddToWorkList(N);
          CombineTo(LN0, NewLoad, NewLoad.getValue(1));
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
        
        // Do not change the width of a volatile load.
        // Do not generate loads of non-round integer types since these can
        // be expensive (and would be wrong if the type is not byte sized).
        if (!LN0->isVolatile() && LoadedVT.bitsGT(ExtVT) && ExtVT.isRound() &&
            (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, ExtVT))) {
          EVT PtrType = LN0->getOperand(1).getValueType();

          unsigned Alignment = LN0->getAlignment();
          SDValue NewPtr = LN0->getBasePtr();

          // For big endian targets, we need to add an offset to the pointer
          // to load the correct bytes.  For little endian systems, we merely
          // need to read fewer bytes from the same pointer.
          if (TLI.isBigEndian()) {
            unsigned LVTStoreBytes = LoadedVT.getStoreSize();
            unsigned EVTStoreBytes = ExtVT.getStoreSize();
            unsigned PtrOff = LVTStoreBytes - EVTStoreBytes;
            NewPtr = DAG.getNode(ISD::ADD, LN0->getDebugLoc(), PtrType,
                                 NewPtr, DAG.getConstant(PtrOff, PtrType));
            Alignment = MinAlign(Alignment, PtrOff);
          }

          AddToWorkList(NewPtr.getNode());
          
          EVT LoadResultTy = HasAnyExt ? LN0->getValueType(0) : VT;
          SDValue Load =
            DAG.getExtLoad(ISD::ZEXTLOAD, LN0->getDebugLoc(), LoadResultTy,
                           LN0->getChain(), NewPtr,
                           LN0->getSrcValue(), LN0->getSrcValueOffset(),
                           ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                           Alignment);
          AddToWorkList(N);
          CombineTo(LN0, Load, Load.getValue(1));
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N1.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (or x, undef) -> -1
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF) {
    EVT EltVT = VT.isVector() ? VT.getVectorElementType() : VT;
    return DAG.getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), VT);
  }
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::OR, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, N1, N0);
  // fold (or x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (or x, -1) -> -1
  if (N1C && N1C->isAllOnesValue())
    return N1;
  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && DAG.MaskedValueIsZero(N0, ~N1C->getAPIntValue()))
    return N1;
  // reassociate or
  SDValue ROR = ReassociateOps(ISD::OR, N->getDebugLoc(), N0, N1);
  if (ROR.getNode() != 0)
    return ROR;
  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  // iff (c1 & c2) == 0.
  if (N1C && N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    if ((C1->getAPIntValue() & N1C->getAPIntValue()) != 0)
      return DAG.getNode(ISD::AND, N->getDebugLoc(), VT,
                         DAG.getNode(ISD::OR, N0.getDebugLoc(), VT,
                                     N0.getOperand(0), N1),
                         DAG.FoldConstantArithmetic(ISD::OR, VT, N1C, C1));
  }
  // fold (or (setcc x), (setcc y)) -> (setcc (or x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();

    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        LL.getValueType().isInteger()) {
      // fold (or (setne X, 0), (setne Y, 0)) -> (setne (or X, Y), 0)
      // fold (or (setlt X, 0), (setlt Y, 0)) -> (setne (or X, Y), 0)
      if (cast<ConstantSDNode>(LR)->isNullValue() &&
          (Op1 == ISD::SETNE || Op1 == ISD::SETLT)) {
        SDValue ORNode = DAG.getNode(ISD::OR, LR.getDebugLoc(),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(N->getDebugLoc(), VT, ORNode, LR, Op1);
      }
      // fold (or (setne X, -1), (setne Y, -1)) -> (setne (and X, Y), -1)
      // fold (or (setgt X, -1), (setgt Y  -1)) -> (setgt (and X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() &&
          (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDValue ANDNode = DAG.getNode(ISD::AND, LR.getDebugLoc(),
                                      LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(N->getDebugLoc(), VT, ANDNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = LL.getValueType().isInteger();
      ISD::CondCode Result = ISD::getSetCCOrOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID &&
          (!LegalOperations || TLI.isCondCodeLegal(Result, LL.getValueType())))
        return DAG.getSetCC(N->getDebugLoc(), N0.getValueType(),
                            LL, LR, Result);
    }
  }

  // Simplify: (or (op x...), (op y...))  -> (op (or x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // (or (and X, C1), (and Y, C2))  -> (and (or X, Y), C3) if possible.
  if (N0.getOpcode() == ISD::AND &&
      N1.getOpcode() == ISD::AND &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      N1.getOperand(1).getOpcode() == ISD::Constant &&
      // Don't increase # computations.
      (N0.getNode()->hasOneUse() || N1.getNode()->hasOneUse())) {
    // We can only do this xform if we know that bits from X that are set in C2
    // but not in C1 are already zero.  Likewise for Y.
    const APInt &LHSMask =
      cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    const APInt &RHSMask =
      cast<ConstantSDNode>(N1.getOperand(1))->getAPIntValue();

    if (DAG.MaskedValueIsZero(N0.getOperand(0), RHSMask&~LHSMask) &&
        DAG.MaskedValueIsZero(N1.getOperand(0), LHSMask&~RHSMask)) {
      SDValue X = DAG.getNode(ISD::OR, N0.getDebugLoc(), VT,
                              N0.getOperand(0), N1.getOperand(0));
      return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, X,
                         DAG.getConstant(LHSMask | RHSMask, VT));
    }
  }

  // See if this is some rotate idiom.
  if (SDNode *Rot = MatchRotate(N0, N1, N->getDebugLoc()))
    return SDValue(Rot, 0);

  return SDValue();
}

/// MatchRotateHalf - Match "(X shl/srl V1) & V2" where V2 may not be present.
static bool MatchRotateHalf(SDValue Op, SDValue &Shift, SDValue &Mask) {
  if (Op.getOpcode() == ISD::AND) {
    if (isa<ConstantSDNode>(Op.getOperand(1))) {
      Mask = Op.getOperand(1);
      Op = Op.getOperand(0);
    } else {
      return false;
    }
  }

  if (Op.getOpcode() == ISD::SRL || Op.getOpcode() == ISD::SHL) {
    Shift = Op;
    return true;
  }

  return false;
}

// MatchRotate - Handle an 'or' of two operands.  If this is one of the many
// idioms for rotate, and if the target supports rotation instructions, generate
// a rot[lr].
SDNode *DAGCombiner::MatchRotate(SDValue LHS, SDValue RHS, DebugLoc DL) {
  // Must be a legal type.  Expanded 'n promoted things won't work with rotates.
  EVT VT = LHS.getValueType();
  if (!TLI.isTypeLegal(VT)) return 0;

  // The target must have at least one rotate flavor.
  bool HasROTL = TLI.isOperationLegalOrCustom(ISD::ROTL, VT);
  bool HasROTR = TLI.isOperationLegalOrCustom(ISD::ROTR, VT);
  if (!HasROTL && !HasROTR) return 0;

  // Match "(X shl/srl V1) & V2" where V2 may not be present.
  SDValue LHSShift;   // The shift.
  SDValue LHSMask;    // AND value if any.
  if (!MatchRotateHalf(LHS, LHSShift, LHSMask))
    return 0; // Not part of a rotate.

  SDValue RHSShift;   // The shift.
  SDValue RHSMask;    // AND value if any.
  if (!MatchRotateHalf(RHS, RHSShift, RHSMask))
    return 0; // Not part of a rotate.

  if (LHSShift.getOperand(0) != RHSShift.getOperand(0))
    return 0;   // Not shifting the same value.

  if (LHSShift.getOpcode() == RHSShift.getOpcode())
    return 0;   // Shifts must disagree.

  // Canonicalize shl to left side in a shl/srl pair.
  if (RHSShift.getOpcode() == ISD::SHL) {
    std::swap(LHS, RHS);
    std::swap(LHSShift, RHSShift);
    std::swap(LHSMask , RHSMask );
  }

  unsigned OpSizeInBits = VT.getSizeInBits();
  SDValue LHSShiftArg = LHSShift.getOperand(0);
  SDValue LHSShiftAmt = LHSShift.getOperand(1);
  SDValue RHSShiftAmt = RHSShift.getOperand(1);

  // fold (or (shl x, C1), (srl x, C2)) -> (rotl x, C1)
  // fold (or (shl x, C1), (srl x, C2)) -> (rotr x, C2)
  if (LHSShiftAmt.getOpcode() == ISD::Constant &&
      RHSShiftAmt.getOpcode() == ISD::Constant) {
    uint64_t LShVal = cast<ConstantSDNode>(LHSShiftAmt)->getZExtValue();
    uint64_t RShVal = cast<ConstantSDNode>(RHSShiftAmt)->getZExtValue();
    if ((LShVal + RShVal) != OpSizeInBits)
      return 0;

    SDValue Rot;
    if (HasROTL)
      Rot = DAG.getNode(ISD::ROTL, DL, VT, LHSShiftArg, LHSShiftAmt);
    else
      Rot = DAG.getNode(ISD::ROTR, DL, VT, LHSShiftArg, RHSShiftAmt);

    // If there is an AND of either shifted operand, apply it to the result.
    if (LHSMask.getNode() || RHSMask.getNode()) {
      APInt Mask = APInt::getAllOnesValue(OpSizeInBits);

      if (LHSMask.getNode()) {
        APInt RHSBits = APInt::getLowBitsSet(OpSizeInBits, LShVal);
        Mask &= cast<ConstantSDNode>(LHSMask)->getAPIntValue() | RHSBits;
      }
      if (RHSMask.getNode()) {
        APInt LHSBits = APInt::getHighBitsSet(OpSizeInBits, RShVal);
        Mask &= cast<ConstantSDNode>(RHSMask)->getAPIntValue() | LHSBits;
      }

      Rot = DAG.getNode(ISD::AND, DL, VT, Rot, DAG.getConstant(Mask, VT));
    }

    return Rot.getNode();
  }

  // If there is a mask here, and we have a variable shift, we can't be sure
  // that we're masking out the right stuff.
  if (LHSMask.getNode() || RHSMask.getNode())
    return 0;

  // fold (or (shl x, y), (srl x, (sub 32, y))) -> (rotl x, y)
  // fold (or (shl x, y), (srl x, (sub 32, y))) -> (rotr x, (sub 32, y))
  if (RHSShiftAmt.getOpcode() == ISD::SUB &&
      LHSShiftAmt == RHSShiftAmt.getOperand(1)) {
    if (ConstantSDNode *SUBC =
          dyn_cast<ConstantSDNode>(RHSShiftAmt.getOperand(0))) {
      if (SUBC->getAPIntValue() == OpSizeInBits) {
        if (HasROTL)
          return DAG.getNode(ISD::ROTL, DL, VT,
                             LHSShiftArg, LHSShiftAmt).getNode();
        else
          return DAG.getNode(ISD::ROTR, DL, VT,
                             LHSShiftArg, RHSShiftAmt).getNode();
      }
    }
  }

  // fold (or (shl x, (sub 32, y)), (srl x, r)) -> (rotr x, y)
  // fold (or (shl x, (sub 32, y)), (srl x, r)) -> (rotl x, (sub 32, y))
  if (LHSShiftAmt.getOpcode() == ISD::SUB &&
      RHSShiftAmt == LHSShiftAmt.getOperand(1)) {
    if (ConstantSDNode *SUBC =
          dyn_cast<ConstantSDNode>(LHSShiftAmt.getOperand(0))) {
      if (SUBC->getAPIntValue() == OpSizeInBits) {
        if (HasROTR)
          return DAG.getNode(ISD::ROTR, DL, VT,
                             LHSShiftArg, RHSShiftAmt).getNode();
        else
          return DAG.getNode(ISD::ROTL, DL, VT,
                             LHSShiftArg, LHSShiftAmt).getNode();
      }
    }
  }

  // Look for sign/zext/any-extended or truncate cases:
  if ((LHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND
       || LHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND
       || LHSShiftAmt.getOpcode() == ISD::ANY_EXTEND
       || LHSShiftAmt.getOpcode() == ISD::TRUNCATE) &&
      (RHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND
       || RHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND
       || RHSShiftAmt.getOpcode() == ISD::ANY_EXTEND
       || RHSShiftAmt.getOpcode() == ISD::TRUNCATE)) {
    SDValue LExtOp0 = LHSShiftAmt.getOperand(0);
    SDValue RExtOp0 = RHSShiftAmt.getOperand(0);
    if (RExtOp0.getOpcode() == ISD::SUB &&
        RExtOp0.getOperand(1) == LExtOp0) {
      // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
      //   (rotl x, y)
      // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
      //   (rotr x, (sub 32, y))
      if (ConstantSDNode *SUBC =
            dyn_cast<ConstantSDNode>(RExtOp0.getOperand(0))) {
        if (SUBC->getAPIntValue() == OpSizeInBits) {
          return DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, DL, VT,
                             LHSShiftArg,
                             HasROTL ? LHSShiftAmt : RHSShiftAmt).getNode();
        }
      }
    } else if (LExtOp0.getOpcode() == ISD::SUB &&
               RExtOp0 == LExtOp0.getOperand(1)) {
      // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext y))) ->
      //   (rotr x, y)
      // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext y))) ->
      //   (rotl x, (sub 32, y))
      if (ConstantSDNode *SUBC =
            dyn_cast<ConstantSDNode>(LExtOp0.getOperand(0))) {
        if (SUBC->getAPIntValue() == OpSizeInBits) {
          return DAG.getNode(HasROTR ? ISD::ROTR : ISD::ROTL, DL, VT,
                             LHSShiftArg,
                             HasROTR ? RHSShiftAmt : LHSShiftAmt).getNode();
        }
      }
    }
  }

  return 0;
}

SDValue DAGCombiner::visitXOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue LHS, RHS, CC;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (xor undef, undef) -> 0. This is a common idiom (misuse).
  if (N0.getOpcode() == ISD::UNDEF && N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // fold (xor x, undef) -> undef
  if (N0.getOpcode() == ISD::UNDEF)
    return N0;
  if (N1.getOpcode() == ISD::UNDEF)
    return N1;
  // fold (xor c1, c2) -> c1^c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::XOR, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT, N1, N0);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // reassociate xor
  SDValue RXOR = ReassociateOps(ISD::XOR, N->getDebugLoc(), N0, N1);
  if (RXOR.getNode() != 0)
    return RXOR;

  // fold !(x cc y) -> (x !cc y)
  if (N1C && N1C->getAPIntValue() == 1 && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = LHS.getValueType().isInteger();
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);

    if (!LegalOperations || TLI.isCondCodeLegal(NotCC, LHS.getValueType())) {
      switch (N0.getOpcode()) {
      default:
        llvm_unreachable("Unhandled SetCC Equivalent!");
      case ISD::SETCC:
        return DAG.getSetCC(N->getDebugLoc(), VT, LHS, RHS, NotCC);
      case ISD::SELECT_CC:
        return DAG.getSelectCC(N->getDebugLoc(), LHS, RHS, N0.getOperand(2),
                               N0.getOperand(3), NotCC);
      }
    }
  }

  // fold (not (zext (setcc x, y))) -> (zext (not (setcc x, y)))
  if (N1C && N1C->getAPIntValue() == 1 && N0.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getNode()->hasOneUse() &&
      isSetCCEquivalent(N0.getOperand(0), LHS, RHS, CC)){
    SDValue V = N0.getOperand(0);
    V = DAG.getNode(ISD::XOR, N0.getDebugLoc(), V.getValueType(), V,
                    DAG.getConstant(1, V.getValueType()));
    AddToWorkList(V.getNode());
    return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT, V);
  }

  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are setcc
  if (N1C && N1C->getAPIntValue() == 1 && VT == MVT::i1 &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, LHS.getDebugLoc(), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, RHS.getDebugLoc(), VT, RHS, N1); // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, N->getDebugLoc(), VT, LHS, RHS);
    }
  }
  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are constants
  if (N1C && N1C->isAllOnesValue() &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, LHS.getDebugLoc(), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, RHS.getDebugLoc(), VT, RHS, N1); // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, N->getDebugLoc(), VT, LHS, RHS);
    }
  }
  // fold (xor (xor x, c1), c2) -> (xor x, (xor c1, c2))
  if (N1C && N0.getOpcode() == ISD::XOR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N00C->getAPIntValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N01C->getAPIntValue(), VT));
  }
  // fold (xor x, x) -> 0
  if (N0 == N1) {
    if (!VT.isVector()) {
      return DAG.getConstant(0, VT);
    } else if (!LegalOperations || TLI.isOperationLegal(ISD::BUILD_VECTOR, VT)){
      // Produce a vector of zeros.
      SDValue El = DAG.getConstant(0, VT.getVectorElementType());
      std::vector<SDValue> Ops(VT.getVectorNumElements(), El);
      return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(), VT,
                         &Ops[0], Ops.size());
    }
  }

  // Simplify: xor (op x...), (op y...)  -> (op (xor x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }

  // Simplify the expression using non-local knowledge.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// visitShiftByConstant - Handle transforms common to the three shifts, when
/// the shift amount is a constant.
SDValue DAGCombiner::visitShiftByConstant(SDNode *N, unsigned Amt) {
  SDNode *LHS = N->getOperand(0).getNode();
  if (!LHS->hasOneUse()) return SDValue();

  // We want to pull some binops through shifts, so that we have (and (shift))
  // instead of (shift (and)), likewise for add, or, xor, etc.  This sort of
  // thing happens with address calculations, so it's important to canonicalize
  // it.
  bool HighBitSet = false;  // Can we transform this if the high bit is set?

  switch (LHS->getOpcode()) {
  default: return SDValue();
  case ISD::OR:
  case ISD::XOR:
    HighBitSet = false; // We can only transform sra if the high bit is clear.
    break;
  case ISD::AND:
    HighBitSet = true;  // We can only transform sra if the high bit is set.
    break;
  case ISD::ADD:
    if (N->getOpcode() != ISD::SHL)
      return SDValue(); // only shl(add) not sr[al](add).
    HighBitSet = false; // We can only transform sra if the high bit is clear.
    break;
  }

  // We require the RHS of the binop to be a constant as well.
  ConstantSDNode *BinOpCst = dyn_cast<ConstantSDNode>(LHS->getOperand(1));
  if (!BinOpCst) return SDValue();

  // FIXME: disable this unless the input to the binop is a shift by a constant.
  // If it is not a shift, it pessimizes some common cases like:
  //
  //    void foo(int *X, int i) { X[i & 1235] = 1; }
  //    int bar(int *X, int i) { return X[i & 255]; }
  SDNode *BinOpLHSVal = LHS->getOperand(0).getNode();
  if ((BinOpLHSVal->getOpcode() != ISD::SHL &&
       BinOpLHSVal->getOpcode() != ISD::SRA &&
       BinOpLHSVal->getOpcode() != ISD::SRL) ||
      !isa<ConstantSDNode>(BinOpLHSVal->getOperand(1)))
    return SDValue();

  EVT VT = N->getValueType(0);

  // If this is a signed shift right, and the high bit is modified by the
  // logical operation, do not perform the transformation. The highBitSet
  // boolean indicates the value of the high bit of the constant which would
  // cause it to be modified for this operation.
  if (N->getOpcode() == ISD::SRA) {
    bool BinOpRHSSignSet = BinOpCst->getAPIntValue().isNegative();
    if (BinOpRHSSignSet != HighBitSet)
      return SDValue();
  }

  // Fold the constants, shifting the binop RHS by the shift amount.
  SDValue NewRHS = DAG.getNode(N->getOpcode(), LHS->getOperand(1).getDebugLoc(),
                               N->getValueType(0),
                               LHS->getOperand(1), N->getOperand(1));

  // Create the new shift.
  SDValue NewShift = DAG.getNode(N->getOpcode(), LHS->getOperand(0).getDebugLoc(),
                                 VT, LHS->getOperand(0), N->getOperand(1));

  // Create the new binop.
  return DAG.getNode(LHS->getOpcode(), N->getDebugLoc(), VT, NewShift, NewRHS);
}

SDValue DAGCombiner::visitSHL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold (shl c1, c2) -> c1<<c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SHL, VT, N0C, N1C);
  // fold (shl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getUNDEF(VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (shl x, c) is known to be zero, return 0
  if (DAG.MaskedValueIsZero(SDValue(N, 0),
                            APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, VT);
  // fold (shl x, (trunc (and y, c))) -> (shl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      EVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      APInt TruncC = N101C->getAPIntValue();
      TruncC.trunc(TruncVT.getSizeInBits());
      return DAG.getNode(ISD::SHL, N->getDebugLoc(), VT, N0,
                         DAG.getNode(ISD::AND, N->getDebugLoc(), TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 N->getDebugLoc(),
                                                 TruncVT, N100),
                                     DAG.getConstant(TruncC, TruncVT)));
    }
  }

  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (shl (shl x, c1), c2) -> 0 or (shl x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SHL &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, N->getDebugLoc(), VT, N0.getOperand(0),
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, (shl -1, c1)), (sub c2, c1)) or
  //                               (srl (and x, (shl -1, c1)), (sub c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRL &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    if (c1 < VT.getSizeInBits()) {
      uint64_t c2 = N1C->getZExtValue();
      SDValue HiBitsMask =
        DAG.getConstant(APInt::getHighBitsSet(VT.getSizeInBits(),
                                              VT.getSizeInBits() - c1),
                        VT);
      SDValue Mask = DAG.getNode(ISD::AND, N0.getDebugLoc(), VT,
                                 N0.getOperand(0),
                                 HiBitsMask);
      if (c2 > c1)
        return DAG.getNode(ISD::SHL, N->getDebugLoc(), VT, Mask,
                           DAG.getConstant(c2-c1, N1.getValueType()));
      else
        return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, Mask,
                           DAG.getConstant(c1-c2, N1.getValueType()));
    }
  }
  // fold (shl (sra x, c1), c1) -> (and x, (shl -1, c1))
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1)) {
    SDValue HiBitsMask =
      DAG.getConstant(APInt::getHighBitsSet(VT.getSizeInBits(),
                                            VT.getSizeInBits() -
                                              N1C->getZExtValue()),
                      VT);
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N0.getOperand(0),
                       HiBitsMask);
  }

  return N1C ? visitShiftByConstant(N, N1C->getZExtValue()) : SDValue();
}

SDValue DAGCombiner::visitSRA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold (sra c1, c2) -> (sra c1, c2)
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SRA, VT, N0C, N1C);
  // fold (sra 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (sra -1, x) -> -1
  if (N0C && N0C->isAllOnesValue())
    return N0;
  // fold (sra x, (setge c, size(x))) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getUNDEF(VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (sra (shl x, c1), c1) -> sext_inreg for some c1 and target supports
  // sext_inreg.
  if (N1C && N0.getOpcode() == ISD::SHL && N1 == N0.getOperand(1)) {
    unsigned LowBits = OpSizeInBits - (unsigned)N1C->getZExtValue();
    EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), LowBits);
    if (VT.isVector())
      ExtVT = EVT::getVectorVT(*DAG.getContext(),
                               ExtVT, VT.getVectorNumElements());
    if ((!LegalOperations ||
         TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, ExtVT)))
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, N->getDebugLoc(), VT,
                         N0.getOperand(0), DAG.getValueType(ExtVT));
  }

  // fold (sra (sra x, c1), c2) -> (sra x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRA) {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      unsigned Sum = N1C->getZExtValue() + C1->getZExtValue();
      if (Sum >= OpSizeInBits) Sum = OpSizeInBits-1;
      return DAG.getNode(ISD::SRA, N->getDebugLoc(), VT, N0.getOperand(0),
                         DAG.getConstant(Sum, N1C->getValueType(0)));
    }
  }

  // fold (sra (shl X, m), (sub result_size, n))
  // -> (sign_extend (trunc (shl X, (sub (sub result_size, n), m)))) for
  // result_size - n != m.
  // If truncate is free for the target sext(shl) is likely to result in better
  // code.
  if (N0.getOpcode() == ISD::SHL) {
    // Get the two constanst of the shifts, CN0 = m, CN = n.
    const ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N01C && N1C) {
      // Determine what the truncate's result bitsize and type would be.
      EVT TruncVT =
        EVT::getIntegerVT(*DAG.getContext(), OpSizeInBits - N1C->getZExtValue());
      // Determine the residual right-shift amount.
      signed ShiftAmt = N1C->getZExtValue() - N01C->getZExtValue();

      // If the shift is not a no-op (in which case this should be just a sign
      // extend already), the truncated to type is legal, sign_extend is legal
      // on that type, and the truncate to that type is both legal and free,
      // perform the transform.
      if ((ShiftAmt > 0) &&
          TLI.isOperationLegalOrCustom(ISD::SIGN_EXTEND, TruncVT) &&
          TLI.isOperationLegalOrCustom(ISD::TRUNCATE, VT) &&
          TLI.isTruncateFree(VT, TruncVT)) {

          SDValue Amt = DAG.getConstant(ShiftAmt, getShiftAmountTy());
          SDValue Shift = DAG.getNode(ISD::SRL, N0.getDebugLoc(), VT,
                                      N0.getOperand(0), Amt);
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(), TruncVT,
                                      Shift);
          return DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(),
                             N->getValueType(0), Trunc);
      }
    }
  }

  // fold (sra x, (trunc (and y, c))) -> (sra x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      EVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      APInt TruncC = N101C->getAPIntValue();
      TruncC.trunc(TruncVT.getScalarType().getSizeInBits());
      return DAG.getNode(ISD::SRA, N->getDebugLoc(), VT, N0,
                         DAG.getNode(ISD::AND, N->getDebugLoc(),
                                     TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 N->getDebugLoc(),
                                                 TruncVT, N100),
                                     DAG.getConstant(TruncC, TruncVT)));
    }
  }

  // Simplify, based on bits shifted out of the LHS.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);


  // If the sign bit is known to be zero, switch this to a SRL.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0, N1);

  return N1C ? visitShiftByConstant(N, N1C->getZExtValue()) : SDValue();
}

SDValue DAGCombiner::visitSRL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold (srl c1, c2) -> c1 >>u c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SRL, VT, N0C, N1C);
  // fold (srl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getUNDEF(VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, VT);

  // fold (srl (srl x, c1), c2) -> 0 or (srl x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRL &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0.getOperand(0),
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }

  // fold (srl (anyextend x), c) -> (anyextend (srl x, c))
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    EVT SmallVT = N0.getOperand(0).getValueType();
    if (N1C->getZExtValue() >= SmallVT.getSizeInBits())
      return DAG.getUNDEF(VT);

    SDValue SmallShift = DAG.getNode(ISD::SRL, N0.getDebugLoc(), SmallVT,
                                     N0.getOperand(0), N1);
    AddToWorkList(SmallShift.getNode());
    return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, SmallShift);
  }

  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getZExtValue() + 1 == VT.getSizeInBits()) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0.getOperand(0), N1);
  }

  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ &&
      N1C->getAPIntValue() == Log2_32(VT.getSizeInBits())) {
    APInt KnownZero, KnownOne;
    APInt Mask = APInt::getAllOnesValue(VT.getScalarType().getSizeInBits());
    DAG.ComputeMaskedBits(N0.getOperand(0), Mask, KnownZero, KnownOne);

    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (KnownOne.getBoolValue()) return DAG.getConstant(0, VT);

    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    APInt UnknownBits = ~KnownZero & Mask;
    if (UnknownBits == 0) return DAG.getConstant(1, VT);

    // Otherwise, check to see if there is exactly one bit input to the ctlz.
    if ((UnknownBits & (UnknownBits - 1)) == 0) {
      // Okay, we know that only that the single bit specified by UnknownBits
      // could be set on input to the CTLZ node. If this bit is set, the SRL
      // will return 0, if it is clear, it returns 1. Change the CTLZ/SRL pair
      // to an SRL/XOR pair, which is likely to simplify more.
      unsigned ShAmt = UnknownBits.countTrailingZeros();
      SDValue Op = N0.getOperand(0);

      if (ShAmt) {
        Op = DAG.getNode(ISD::SRL, N0.getDebugLoc(), VT, Op,
                         DAG.getConstant(ShAmt, getShiftAmountTy()));
        AddToWorkList(Op.getNode());
      }

      return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT,
                         Op, DAG.getConstant(1, VT));
    }
  }

  // fold (srl x, (trunc (and y, c))) -> (srl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      EVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      APInt TruncC = N101C->getAPIntValue();
      TruncC.trunc(TruncVT.getSizeInBits());
      return DAG.getNode(ISD::SRL, N->getDebugLoc(), VT, N0,
                         DAG.getNode(ISD::AND, N->getDebugLoc(),
                                     TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 N->getDebugLoc(),
                                                 TruncVT, N100),
                                     DAG.getConstant(TruncC, TruncVT)));
    }
  }

  // fold operands of srl based on knowledge that the low bits are not
  // demanded.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  if (N1C) {
    SDValue NewSRL = visitShiftByConstant(N, N1C->getZExtValue());
    if (NewSRL.getNode())
      return NewSRL;
  }

  // Here is a common situation. We want to optimize:
  //
  //   %a = ...
  //   %b = and i32 %a, 2
  //   %c = srl i32 %b, 1
  //   brcond i32 %c ...
  //
  // into
  // 
  //   %a = ...
  //   %b = and %a, 2
  //   %c = setcc eq %b, 0
  //   brcond %c ...
  //
  // However when after the source operand of SRL is optimized into AND, the SRL
  // itself may not be optimized further. Look for it and add the BRCOND into
  // the worklist.
  if (N->hasOneUse()) {
    SDNode *Use = *N->use_begin();
    if (Use->getOpcode() == ISD::BRCOND)
      AddToWorkList(Use);
    else if (Use->getOpcode() == ISD::TRUNCATE && Use->hasOneUse()) {
      // Also look pass the truncate.
      Use = *Use->use_begin();
      if (Use->getOpcode() == ISD::BRCOND)
        AddToWorkList(Use);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitCTLZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTLZ, N->getDebugLoc(), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTTZ, N->getDebugLoc(), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTPOP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctpop c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTPOP, N->getDebugLoc(), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  EVT VT = N->getValueType(0);
  EVT VT0 = N0.getValueType();

  // fold (select C, X, X) -> X
  if (N1 == N2)
    return N1;
  // fold (select true, X, Y) -> X
  if (N0C && !N0C->isNullValue())
    return N1;
  // fold (select false, X, Y) -> Y
  if (N0C && N0C->isNullValue())
    return N2;
  // fold (select C, 1, X) -> (or C, X)
  if (VT == MVT::i1 && N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, N0, N2);
  // fold (select C, 0, 1) -> (xor C, 1)
  if (VT.isInteger() &&
      (VT0 == MVT::i1 ||
       (VT0.isInteger() &&
        TLI.getBooleanContents() == TargetLowering::ZeroOrOneBooleanContent)) &&
      N1C && N2C && N1C->isNullValue() && N2C->getAPIntValue() == 1) {
    SDValue XORNode;
    if (VT == VT0)
      return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT0,
                         N0, DAG.getConstant(1, VT0));
    XORNode = DAG.getNode(ISD::XOR, N0.getDebugLoc(), VT0,
                          N0, DAG.getConstant(1, VT0));
    AddToWorkList(XORNode.getNode());
    if (VT.bitsGT(VT0))
      return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT, XORNode);
    return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, XORNode);
  }
  // fold (select C, 0, X) -> (and (not C), X)
  if (VT == VT0 && VT == MVT::i1 && N1C && N1C->isNullValue()) {
    SDValue NOTNode = DAG.getNOT(N0.getDebugLoc(), N0, VT);
    AddToWorkList(NOTNode.getNode());
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, NOTNode, N2);
  }
  // fold (select C, X, 1) -> (or (not C), X)
  if (VT == VT0 && VT == MVT::i1 && N2C && N2C->getAPIntValue() == 1) {
    SDValue NOTNode = DAG.getNOT(N0.getDebugLoc(), N0, VT);
    AddToWorkList(NOTNode.getNode());
    return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, NOTNode, N1);
  }
  // fold (select C, X, 0) -> (and C, X)
  if (VT == MVT::i1 && N2C && N2C->isNullValue())
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N0, N1);
  // fold (select X, X, Y) -> (or X, Y)
  // fold (select X, 1, Y) -> (or X, Y)
  if (VT == MVT::i1 && (N0 == N1 || (N1C && N1C->getAPIntValue() == 1)))
    return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, N0, N2);
  // fold (select X, Y, X) -> (and X, Y)
  // fold (select X, Y, 0) -> (and X, Y)
  if (VT == MVT::i1 && (N0 == N2 || (N2C && N2C->getAPIntValue() == 0)))
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT, N0, N1);

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0);  // Don't revisit N.

  // fold selects based on a setcc into other things, such as min/max/abs
  if (N0.getOpcode() == ISD::SETCC) {
    // FIXME:
    // Check against MVT::Other for SELECT_CC, which is a workaround for targets
    // having to say they don't support SELECT_CC on every type the DAG knows
    // about, since there is no way to mark an opcode illegal at all value types
    if (TLI.isOperationLegalOrCustom(ISD::SELECT_CC, MVT::Other) &&
        TLI.isOperationLegalOrCustom(ISD::SELECT_CC, VT))
      return DAG.getNode(ISD::SELECT_CC, N->getDebugLoc(), VT,
                         N0.getOperand(0), N0.getOperand(1),
                         N1, N2, N0.getOperand(2));
    return SimplifySelect(N->getDebugLoc(), N0, N1, N2);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSELECT_CC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDValue N3 = N->getOperand(3);
  SDValue N4 = N->getOperand(4);
  ISD::CondCode CC = cast<CondCodeSDNode>(N4)->get();

  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N2 == N3)
    return N2;

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(TLI.getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, N->getDebugLoc(), false);
  if (SCC.getNode()) AddToWorkList(SCC.getNode());

  if (ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.getNode())) {
    if (!SCCC->isNullValue())
      return N2;    // cond always true -> true val
    else
      return N3;    // cond always false -> false val
  }

  // Fold to a simpler select_cc
  if (SCC.getNode() && SCC.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::SELECT_CC, N->getDebugLoc(), N2.getValueType(),
                       SCC.getOperand(0), SCC.getOperand(1), N2, N3,
                       SCC.getOperand(2));

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDValue(N, 0);  // Don't revisit N.

  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(N->getDebugLoc(), N0, N1, N2, N3, CC);
}

SDValue DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get(),
                       N->getDebugLoc());
}

// ExtendUsesToFormExtLoad - Trying to extend uses of a load to enable this:
// "fold ({s|z|a}ext (load x)) -> ({s|z|a}ext (truncate ({s|z|a}extload x)))"
// transformation. Returns true if extension are possible and the above
// mentioned transformation is profitable.
static bool ExtendUsesToFormExtLoad(SDNode *N, SDValue N0,
                                    unsigned ExtOpc,
                                    SmallVector<SDNode*, 4> &ExtendNodes,
                                    const TargetLowering &TLI) {
  bool HasCopyToRegUses = false;
  bool isTruncFree = TLI.isTruncateFree(N->getValueType(0), N0.getValueType());
  for (SDNode::use_iterator UI = N0.getNode()->use_begin(),
                            UE = N0.getNode()->use_end();
       UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User == N)
      continue;
    if (UI.getUse().getResNo() != N0.getResNo())
      continue;
    // FIXME: Only extend SETCC N, N and SETCC N, c for now.
    if (ExtOpc != ISD::ANY_EXTEND && User->getOpcode() == ISD::SETCC) {
      ISD::CondCode CC = cast<CondCodeSDNode>(User->getOperand(2))->get();
      if (ExtOpc == ISD::ZERO_EXTEND && ISD::isSignedIntSetCC(CC))
        // Sign bits will be lost after a zext.
        return false;
      bool Add = false;
      for (unsigned i = 0; i != 2; ++i) {
        SDValue UseOp = User->getOperand(i);
        if (UseOp == N0)
          continue;
        if (!isa<ConstantSDNode>(UseOp))
          return false;
        Add = true;
      }
      if (Add)
        ExtendNodes.push_back(User);
      continue;
    }
    // If truncates aren't free and there are users we can't
    // extend, it isn't worthwhile.
    if (!isTruncFree)
      return false;
    // Remember if this value is live-out.
    if (User->getOpcode() == ISD::CopyToReg)
      HasCopyToRegUses = true;
  }

  if (HasCopyToRegUses) {
    bool BothLiveOut = false;
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDUse &Use = UI.getUse();
      if (Use.getResNo() == 0 && Use.getUser()->getOpcode() == ISD::CopyToReg) {
        BothLiveOut = true;
        break;
      }
    }
    if (BothLiveOut)
      // Both unextended and extended values are live out. There had better be
      // good a reason for the transformation.
      return ExtendNodes.size();
  }
  return true;
}

SDValue DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(), VT, N0);

  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(), VT,
                       N0.getOperand(0));

  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (sext (truncate (load x))) -> (sext (smaller load x))
    // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }

    // See if the value being truncated is already sign extended.  If so, just
    // eliminate the trunc/sext pair.
    SDValue Op = N0.getOperand(0);
    unsigned OpBits   = Op.getValueType().getScalarType().getSizeInBits();
    unsigned MidBits  = N0.getValueType().getScalarType().getSizeInBits();
    unsigned DestBits = VT.getScalarType().getSizeInBits();
    unsigned NumSignBits = DAG.ComputeNumSignBits(Op);

    if (OpBits == DestBits) {
      // Op is i32, Mid is i8, and Dest is i32.  If Op has more than 24 sign
      // bits, it is already ready.
      if (NumSignBits > DestBits-MidBits)
        return Op;
    } else if (OpBits < DestBits) {
      // Op is i32, Mid is i8, and Dest is i64.  If Op has more than 24 sign
      // bits, just sext from i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(), VT, Op);
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, Op);
    }

    // fold (sext (truncate x)) -> (sextinreg x).
    if (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                                 N0.getValueType())) {
      if (OpBits < DestBits)
        Op = DAG.getNode(ISD::ANY_EXTEND, N0.getDebugLoc(), VT, Op);
      else if (OpBits > DestBits)
        Op = DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(), VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, N->getDebugLoc(), VT, Op,
                         DAG.getValueType(N0.getValueType()));
    }
  }

  // fold (sext (load x)) -> (sext (truncate (sextload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::SIGN_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, N->getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));

      // Extend SetCC uses if necessary.
      for (unsigned i = 0, e = SetCCs.size(); i != e; ++i) {
        SDNode *SetCC = SetCCs[i];
        SmallVector<SDValue, 4> Ops;

        for (unsigned j = 0; j != 2; ++j) {
          SDValue SOp = SetCC->getOperand(j);
          if (SOp == Trunc)
            Ops.push_back(ExtLoad);
          else
            Ops.push_back(DAG.getNode(ISD::SIGN_EXTEND,
                                      N->getDebugLoc(), VT, SOp));
        }

        Ops.push_back(SetCC->getOperand(2));
        CombineTo(SetCC, DAG.getNode(ISD::SETCC, N->getDebugLoc(),
                                     SetCC->getValueType(0),
                                     &Ops[0], Ops.size()));
      }

      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (sext (sextload x)) -> (sext (truncate (sextload x)))
  // fold (sext ( extload x)) -> (sext (truncate (sextload x)))
  if ((ISD::isSEXTLoad(N0.getNode()) || ISD::isEXTLoad(N0.getNode())) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::SEXTLOAD, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, N->getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), MemVT,
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(),
                            N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    // sext(setcc) -> sext_in_reg(vsetcc) for vectors.
    if (VT.isVector() &&
        // We know that the # elements of the results is the same as the
        // # elements of the compare (and the # elements of the compare result
        // for that matter).  Check to see that they are the same size.  If so,
        // we know that the element size of the sext'd result matches the
        // element size of the compare operands.
        VT.getSizeInBits() == N0.getOperand(0).getValueType().getSizeInBits() &&
      
        // Only do this before legalize for now.
        !LegalOperations) {
      return DAG.getVSetCC(N->getDebugLoc(), VT, N0.getOperand(0),
                           N0.getOperand(1),
                           cast<CondCodeSDNode>(N0.getOperand(2))->get());
    }
    
    // sext(setcc x, y, cc) -> (select_cc x, y, -1, 0, cc)
    SDValue NegOne =
      DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), VT);
    SDValue SCC =
      SimplifySelectCC(N->getDebugLoc(), N0.getOperand(0), N0.getOperand(1),
                       NegOne, DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC, TLI.getSetCCResultType(VT)))
      return DAG.getNode(ISD::SELECT, N->getDebugLoc(), VT,
                         DAG.getSetCC(N->getDebugLoc(),
                                      TLI.getSetCCResultType(VT),
                                      N0.getOperand(0), N0.getOperand(1),
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get()),
                         NegOne, DAG.getConstant(0, VT));
  }
  
  

  // fold (sext x) -> (zext x) if the sign bit is known zero.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ZERO_EXTEND, VT)) &&
      DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (zext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT, N0);
  // fold (zext (zext x)) -> (zext x)
  // fold (zext (aext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT,
                       N0.getOperand(0));

  // fold (zext (truncate (load x))) -> (zext (smaller load x))
  // fold (zext (truncate (srl (load x), c))) -> (zext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return DAG.getNode(ISD::ZERO_EXTEND, N->getDebugLoc(), VT, NarrowLoad);
    }
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (!LegalOperations || TLI.isOperationLegal(ISD::AND, VT)) &&
      (!TLI.isTruncateFree(N0.getOperand(0).getValueType(),
                           N0.getValueType()) ||
       !TLI.isZExtFree(N0.getValueType(), VT))) {
    SDValue Op = N0.getOperand(0);
    if (Op.getValueType().bitsLT(VT)) {
      Op = DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, Op);
    } else if (Op.getValueType().bitsGT(VT)) {
      Op = DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, Op);
    }
    return DAG.getZeroExtendInReg(Op, N->getDebugLoc(),
                                  N0.getValueType().getScalarType());
  }

  // Fold (zext (and (trunc x), cst)) -> (and x, cst),
  // if either of the casts is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      (!TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                           N0.getValueType()) ||
       !TLI.isZExtFree(N0.getValueType(), VT))) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, X.getDebugLoc(), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, X.getDebugLoc(), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT,
                       X, DAG.getConstant(Mask, VT));
  }

  // fold (zext (load x)) -> (zext (truncate (zextload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::ZEXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ZERO_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, N->getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));

      // Extend SetCC uses if necessary.
      for (unsigned i = 0, e = SetCCs.size(); i != e; ++i) {
        SDNode *SetCC = SetCCs[i];
        SmallVector<SDValue, 4> Ops;

        for (unsigned j = 0; j != 2; ++j) {
          SDValue SOp = SetCC->getOperand(j);
          if (SOp == Trunc)
            Ops.push_back(ExtLoad);
          else
            Ops.push_back(DAG.getNode(ISD::ZERO_EXTEND,
                                      N->getDebugLoc(), VT, SOp));
        }

        Ops.push_back(SetCC->getOperand(2));
        CombineTo(SetCC, DAG.getNode(ISD::SETCC, N->getDebugLoc(),
                                     SetCC->getValueType(0),
                                     &Ops[0], Ops.size()));
      }

      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (zext (zextload x)) -> (zext (truncate (zextload x)))
  // fold (zext ( extload x)) -> (zext (truncate (zextload x)))
  if ((ISD::isZEXTLoad(N0.getNode()) || ISD::isEXTLoad(N0.getNode())) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::ZEXTLOAD, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, N->getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), MemVT,
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(), N0.getValueType(),
                            ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // zext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue SCC =
      SimplifySelectCC(N->getDebugLoc(), N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
  }

  // (zext (shl (zext x), cst)) -> (shl (zext x), cst)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL) &&
      isa<ConstantSDNode>(N0.getOperand(1)) &&
      N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
      N0.hasOneUse()) {
    if (N0.getOpcode() == ISD::SHL) {
      // If the original shl may be shifting out bits, do not perform this
      // transformation.
      unsigned ShAmt = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
      unsigned KnownZeroBits = N0.getOperand(0).getValueType().getSizeInBits() -
        N0.getOperand(0).getOperand(0).getValueType().getSizeInBits();
      if (ShAmt > KnownZeroBits)
        return SDValue();
    }
    DebugLoc dl = N->getDebugLoc();
    return DAG.getNode(N0.getOpcode(), dl, VT,
                       DAG.getNode(ISD::ZERO_EXTEND, dl, VT, N0.getOperand(0)),
                       DAG.getNode(ISD::ZERO_EXTEND, dl,
                                   N0.getOperand(1).getValueType(),
                                   N0.getOperand(1)));
  }

  return SDValue();
}

SDValue DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (aext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, N0);
  // fold (aext (aext x)) -> (aext x)
  // fold (aext (zext x)) -> (zext x)
  // fold (aext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::ANY_EXTEND  ||
      N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(N0.getOpcode(), N->getDebugLoc(), VT, N0.getOperand(0));

  // fold (aext (truncate (load x))) -> (aext (smaller load x))
  // fold (aext (truncate (srl (load x), c))) -> (aext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, NarrowLoad);
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue TruncOp = N0.getOperand(0);
    if (TruncOp.getValueType() == VT)
      return TruncOp; // x iff x size == zext size.
    if (TruncOp.getValueType().bitsGT(VT))
      return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, TruncOp);
    return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, TruncOp);
  }

  // Fold (aext (and (trunc x), cst)) -> (and x, cst)
  // if the trunc is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      !TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                          N0.getValueType())) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT,
                       X, DAG.getConstant(Mask, VT));
  }

  // fold (aext (load x)) -> (aext (truncate (extload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ANY_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, N->getDebugLoc(), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), LN0->isNonTemporal(),
                                       LN0->getAlignment());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));

      // Extend SetCC uses if necessary.
      for (unsigned i = 0, e = SetCCs.size(); i != e; ++i) {
        SDNode *SetCC = SetCCs[i];
        SmallVector<SDValue, 4> Ops;

        for (unsigned j = 0; j != 2; ++j) {
          SDValue SOp = SetCC->getOperand(j);
          if (SOp == Trunc)
            Ops.push_back(ExtLoad);
          else
            Ops.push_back(DAG.getNode(ISD::ANY_EXTEND,
                                      N->getDebugLoc(), VT, SOp));
        }

        Ops.push_back(SetCC->getOperand(2));
        CombineTo(SetCC, DAG.getNode(ISD::SETCC, N->getDebugLoc(),
                                     SetCC->getValueType(0),
                                     &Ops[0], Ops.size()));
      }

      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (zextload x)) -> (aext (truncate (zextload x)))
  // fold (aext (sextload x)) -> (aext (truncate (sextload x)))
  // fold (aext ( extload x)) -> (aext (truncate (extload  x)))
  if (N0.getOpcode() == ISD::LOAD &&
      !ISD::isNON_EXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    SDValue ExtLoad = DAG.getExtLoad(LN0->getExtensionType(), N->getDebugLoc(),
                                     VT, LN0->getChain(), LN0->getBasePtr(),
                                     LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), MemVT,
                                     LN0->isVolatile(), LN0->isNonTemporal(),
                                     LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::TRUNCATE, N0.getDebugLoc(),
                          N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue SCC =
      SimplifySelectCC(N->getDebugLoc(), N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode())
      return SCC;
  }

  return SDValue();
}

/// GetDemandedBits - See if the specified operand can be simplified with the
/// knowledge that only the bits specified by Mask are used.  If so, return the
/// simpler operand, otherwise return a null SDValue.
SDValue DAGCombiner::GetDemandedBits(SDValue V, const APInt &Mask) {
  switch (V.getOpcode()) {
  default: break;
  case ISD::OR:
  case ISD::XOR:
    // If the LHS or RHS don't contribute bits to the or, drop them.
    if (DAG.MaskedValueIsZero(V.getOperand(0), Mask))
      return V.getOperand(1);
    if (DAG.MaskedValueIsZero(V.getOperand(1), Mask))
      return V.getOperand(0);
    break;
  case ISD::SRL:
    // Only look at single-use SRLs.
    if (!V.getNode()->hasOneUse())
      break;
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(V.getOperand(1))) {
      // See if we can recursively simplify the LHS.
      unsigned Amt = RHSC->getZExtValue();

      // Watch out for shift count overflow though.
      if (Amt >= Mask.getBitWidth()) break;
      APInt NewMask = Mask << Amt;
      SDValue SimplifyLHS = GetDemandedBits(V.getOperand(0), NewMask);
      if (SimplifyLHS.getNode())
        return DAG.getNode(ISD::SRL, V.getDebugLoc(), V.getValueType(),
                           SimplifyLHS, V.getOperand(1));
    }
  }
  return SDValue();
}

/// ReduceLoadWidth - If the result of a wider load is shifted to right of N
/// bits and then truncated to a narrower type and where N is a multiple
/// of number of bits of the narrower type, transform it to a narrower load
/// from address + N / num of bits of new type. If the result is to be
/// extended, also fold the extension to form a extending load.
SDValue DAGCombiner::ReduceLoadWidth(SDNode *N) {
  unsigned Opc = N->getOpcode();
  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT ExtVT = VT;

  // This transformation isn't valid for vector loads.
  if (VT.isVector())
    return SDValue();

  // Special case: SIGN_EXTEND_INREG is basically truncating to ExtVT then
  // extended to VT.
  if (Opc == ISD::SIGN_EXTEND_INREG) {
    ExtType = ISD::SEXTLOAD;
    ExtVT = cast<VTSDNode>(N->getOperand(1))->getVT();
    if (LegalOperations && !TLI.isLoadExtLegal(ISD::SEXTLOAD, ExtVT))
      return SDValue();
  }

  unsigned EVTBits = ExtVT.getSizeInBits();
  unsigned ShAmt = 0;
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse() && ExtVT.isRound()) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShAmt = N01->getZExtValue();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        // Is the load width a multiple of size of VT?
        if ((N0.getValueType().getSizeInBits() & (EVTBits-1)) != 0)
          return SDValue();
      }
    }
  }

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (isa<LoadSDNode>(N0) && N0.hasOneUse() && ExtVT.isRound() &&
      cast<LoadSDNode>(N0)->getMemoryVT().getSizeInBits() > EVTBits &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT PtrType = N0.getOperand(1).getValueType();

    // For big endian targets, we need to adjust the offset to the pointer to
    // load the correct bytes.
    if (TLI.isBigEndian()) {
      unsigned LVTStoreBits = LN0->getMemoryVT().getStoreSizeInBits();
      unsigned EVTStoreBits = ExtVT.getStoreSizeInBits();
      ShAmt = LVTStoreBits - EVTStoreBits - ShAmt;
    }

    uint64_t PtrOff =  ShAmt / 8;
    unsigned NewAlign = MinAlign(LN0->getAlignment(), PtrOff);
    SDValue NewPtr = DAG.getNode(ISD::ADD, LN0->getDebugLoc(),
                                 PtrType, LN0->getBasePtr(),
                                 DAG.getConstant(PtrOff, PtrType));
    AddToWorkList(NewPtr.getNode());

    SDValue Load = (ExtType == ISD::NON_EXTLOAD)
      ? DAG.getLoad(VT, N0.getDebugLoc(), LN0->getChain(), NewPtr,
                    LN0->getSrcValue(), LN0->getSrcValueOffset() + PtrOff,
                    LN0->isVolatile(), LN0->isNonTemporal(), NewAlign)
      : DAG.getExtLoad(ExtType, N0.getDebugLoc(), VT, LN0->getChain(), NewPtr,
                       LN0->getSrcValue(), LN0->getSrcValueOffset() + PtrOff,
                       ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                       NewAlign);

    // Replace the old load's chain with the new load's chain.
    WorkListRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1),
                                  &DeadNodes);

    // Return the new loaded value.
    return Load;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT EVT = cast<VTSDNode>(N1)->getVT();
  unsigned VTBits = VT.getScalarType().getSizeInBits();
  unsigned EVTBits = EVT.getScalarType().getSizeInBits();

  // fold (sext_in_reg c1) -> c1
  if (isa<ConstantSDNode>(N0) || N0.getOpcode() == ISD::UNDEF)
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, N->getDebugLoc(), VT, N0, N1);

  // If the input is already sign extended, just drop the extension.
  if (DAG.ComputeNumSignBits(N0) >= VTBits-EVTBits+1)
    return N0;

  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT.bitsLT(cast<VTSDNode>(N0.getOperand(1))->getVT())) {
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, N->getDebugLoc(), VT,
                       N0.getOperand(0), N1);
  }

  // fold (sext_in_reg (sext x)) -> (sext x)
  // fold (sext_in_reg (aext x)) -> (sext x)
  // if x is small enough.
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getValueType().getScalarType().getSizeInBits() < EVTBits)
      return DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(), VT, N00, N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (DAG.MaskedValueIsZero(N0, APInt::getBitsSet(VTBits, EVTBits-1, EVTBits)))
    return DAG.getZeroExtendInReg(N0, N->getDebugLoc(), EVT);

  // fold operands of sext_in_reg based on knowledge that the top bits are not
  // demanded.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (sext_in_reg (load x)) -> (smaller sextload x)
  // fold (sext_in_reg (srl (load x), c)) -> (smaller sextload (x+c/evtbits))
  SDValue NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.getNode())
    return NarrowLoad;

  // fold (sext_in_reg (srl X, 24), i8) -> (sra X, 24)
  // fold (sext_in_reg (srl X, 23), i8) -> (sra X, 23) iff possible.
  // We already fold "(sext_in_reg (srl X, 25), i8) -> srl X, 25" above.
  if (N0.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if (ShAmt->getZExtValue()+EVTBits <= VTBits) {
        // We can turn this into an SRA iff the input to the SRL is already sign
        // extended enough.
        unsigned InSignBits = DAG.ComputeNumSignBits(N0.getOperand(0));
        if (VTBits-(ShAmt->getZExtValue()+EVTBits) < InSignBits)
          return DAG.getNode(ISD::SRA, N->getDebugLoc(), VT,
                             N0.getOperand(0), N0.getOperand(1));
      }
  }

  // fold (sext_inreg (extload x)) -> (sextload x)
  if (ISD::isEXTLoad(N0.getNode()) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, N->getDebugLoc(), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), EVT,
                                     LN0->isVolatile(), LN0->isNonTemporal(),
                                     LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (ISD::isZEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse() &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, N->getDebugLoc(), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), EVT,
                                     LN0->isVolatile(), LN0->isNonTemporal(),
                                     LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  return SDValue();
}

SDValue DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, N0);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::SIGN_EXTEND||
      N0.getOpcode() == ISD::ANY_EXTEND) {
    if (N0.getOperand(0).getValueType().bitsLT(VT))
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), N->getDebugLoc(), VT,
                         N0.getOperand(0));
    else if (N0.getOperand(0).getValueType().bitsGT(VT))
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, N0.getOperand(0));
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate.
      return N0.getOperand(0);
  }

  // See if we can simplify the input to this truncate through knowledge that
  // only the low bits are being used.  For example "trunc (or (shl x, 8), y)"
  // -> trunc y
  SDValue Shorter =
    GetDemandedBits(N0, APInt::getLowBitsSet(N0.getValueSizeInBits(),
                                             VT.getSizeInBits()));
  if (Shorter.getNode())
    return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, Shorter);

  // fold (truncate (load x)) -> (smaller load x)
  // fold (truncate (srl (load x), c)) -> (smaller load (x+c/evtbits))
  return ReduceLoadWidth(N);
}

static SDNode *getBuildPairElt(SDNode *N, unsigned i) {
  SDValue Elt = N->getOperand(i);
  if (Elt.getOpcode() != ISD::MERGE_VALUES)
    return Elt.getNode();
  return Elt.getOperand(Elt.getResNo()).getNode();
}

/// CombineConsecutiveLoads - build_pair (load, load) -> load
/// if load locations are consecutive.
SDValue DAGCombiner::CombineConsecutiveLoads(SDNode *N, EVT VT) {
  assert(N->getOpcode() == ISD::BUILD_PAIR);

  LoadSDNode *LD1 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 0));
  LoadSDNode *LD2 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 1));
  if (!LD1 || !LD2 || !ISD::isNON_EXTLoad(LD1) || !LD1->hasOneUse())
    return SDValue();
  EVT LD1VT = LD1->getValueType(0);

  if (ISD::isNON_EXTLoad(LD2) &&
      LD2->hasOneUse() &&
      // If both are volatile this would reduce the number of volatile loads.
      // If one is volatile it might be ok, but play conservative and bail out.
      !LD1->isVolatile() &&
      !LD2->isVolatile() &&
      DAG.isConsecutiveLoad(LD2, LD1, LD1VT.getSizeInBits()/8, 1)) {
    unsigned Align = LD1->getAlignment();
    unsigned NewAlign = TLI.getTargetData()->
      getABITypeAlignment(VT.getTypeForEVT(*DAG.getContext()));

    if (NewAlign <= Align &&
        (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)))
      return DAG.getLoad(VT, N->getDebugLoc(), LD1->getChain(),
                         LD1->getBasePtr(), LD1->getSrcValue(),
                         LD1->getSrcValueOffset(), false, false, Align);
  }

  return SDValue();
}

SDValue DAGCombiner::visitBIT_CONVERT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // If the input is a BUILD_VECTOR with all constant elements, fold this now.
  // Only do this before legalize, since afterward the target may be depending
  // on the bitconvert.
  // First check to see if this is all constant.
  if (!LegalTypes &&
      N0.getOpcode() == ISD::BUILD_VECTOR && N0.getNode()->hasOneUse() &&
      VT.isVector()) {
    bool isSimple = true;
    for (unsigned i = 0, e = N0.getNumOperands(); i != e; ++i)
      if (N0.getOperand(i).getOpcode() != ISD::UNDEF &&
          N0.getOperand(i).getOpcode() != ISD::Constant &&
          N0.getOperand(i).getOpcode() != ISD::ConstantFP) {
        isSimple = false;
        break;
      }

    EVT DestEltVT = N->getValueType(0).getVectorElementType();
    assert(!DestEltVT.isVector() &&
           "Element type of vector ValueType must not be vector!");
    if (isSimple)
      return ConstantFoldBIT_CONVERTofBUILD_VECTOR(N0.getNode(), DestEltVT);
  }

  // If the input is a constant, let getNode fold it.
  if (isa<ConstantSDNode>(N0) || isa<ConstantFPSDNode>(N0)) {
    SDValue Res = DAG.getNode(ISD::BIT_CONVERT, N->getDebugLoc(), VT, N0);
    if (Res.getNode() != N) {
      if (!LegalOperations ||
          TLI.isOperationLegal(Res.getNode()->getOpcode(), VT))
        return Res;

      // Folding it resulted in an illegal node, and it's too late to
      // do that. Clean up the old node and forego the transformation.
      // Ideally this won't happen very often, because instcombine
      // and the earlier dagcombine runs (where illegal nodes are
      // permitted) should have folded most of them already.
      DAG.DeleteNode(Res.getNode());
    }
  }

  // (conv (conv x, t1), t2) -> (conv x, t2)
  if (N0.getOpcode() == ISD::BIT_CONVERT)
    return DAG.getNode(ISD::BIT_CONVERT, N->getDebugLoc(), VT,
                       N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a higher alignment than the original!
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile() &&
      (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    unsigned Align = TLI.getTargetData()->
      getABITypeAlignment(VT.getTypeForEVT(*DAG.getContext()));
    unsigned OrigAlign = LN0->getAlignment();

    if (Align <= OrigAlign) {
      SDValue Load = DAG.getLoad(VT, N->getDebugLoc(), LN0->getChain(),
                                 LN0->getBasePtr(),
                                 LN0->getSrcValue(), LN0->getSrcValueOffset(),
                                 LN0->isVolatile(), LN0->isNonTemporal(),
                                 OrigAlign);
      AddToWorkList(N);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::BIT_CONVERT, N0.getDebugLoc(),
                            N0.getValueType(), Load),
                Load.getValue(1));
      return Load;
    }
  }

  // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
  // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
  // This often reduces constant pool loads.
  if ((N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FABS) &&
      N0.getNode()->hasOneUse() && VT.isInteger() && !VT.isVector()) {
    SDValue NewConv = DAG.getNode(ISD::BIT_CONVERT, N0.getDebugLoc(), VT,
                                  N0.getOperand(0));
    AddToWorkList(NewConv.getNode());

    APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
    if (N0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, N->getDebugLoc(), VT,
                         NewConv, DAG.getConstant(SignBit, VT));
    assert(N0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, N->getDebugLoc(), VT,
                       NewConv, DAG.getConstant(~SignBit, VT));
  }

  // fold (bitconvert (fcopysign cst, x)) ->
  //         (or (and (bitconvert x), sign), (and cst, (not sign)))
  // Note that we don't handle (copysign x, cst) because this can always be
  // folded to an fneg or fabs.
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse() &&
      isa<ConstantFPSDNode>(N0.getOperand(0)) &&
      VT.isInteger() && !VT.isVector()) {
    unsigned OrigXWidth = N0.getOperand(1).getValueType().getSizeInBits();
    EVT IntXVT = EVT::getIntegerVT(*DAG.getContext(), OrigXWidth);
    if (TLI.isTypeLegal(IntXVT) || !LegalTypes) {
      SDValue X = DAG.getNode(ISD::BIT_CONVERT, N0.getDebugLoc(),
                              IntXVT, N0.getOperand(1));
      AddToWorkList(X.getNode());

      // If X has a different width than the result/lhs, sext it or truncate it.
      unsigned VTWidth = VT.getSizeInBits();
      if (OrigXWidth < VTWidth) {
        X = DAG.getNode(ISD::SIGN_EXTEND, N->getDebugLoc(), VT, X);
        AddToWorkList(X.getNode());
      } else if (OrigXWidth > VTWidth) {
        // To get the sign bit in the right place, we have to shift it right
        // before truncating.
        X = DAG.getNode(ISD::SRL, X.getDebugLoc(),
                        X.getValueType(), X,
                        DAG.getConstant(OrigXWidth-VTWidth, X.getValueType()));
        AddToWorkList(X.getNode());
        X = DAG.getNode(ISD::TRUNCATE, X.getDebugLoc(), VT, X);
        AddToWorkList(X.getNode());
      }

      APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
      X = DAG.getNode(ISD::AND, X.getDebugLoc(), VT,
                      X, DAG.getConstant(SignBit, VT));
      AddToWorkList(X.getNode());

      SDValue Cst = DAG.getNode(ISD::BIT_CONVERT, N0.getDebugLoc(),
                                VT, N0.getOperand(0));
      Cst = DAG.getNode(ISD::AND, Cst.getDebugLoc(), VT,
                        Cst, DAG.getConstant(~SignBit, VT));
      AddToWorkList(Cst.getNode());

      return DAG.getNode(ISD::OR, N->getDebugLoc(), VT, X, Cst);
    }
  }

  // bitconvert(build_pair(ld, ld)) -> ld iff load locations are consecutive.
  if (N0.getOpcode() == ISD::BUILD_PAIR) {
    SDValue CombineLD = CombineConsecutiveLoads(N0.getNode(), VT);
    if (CombineLD.getNode())
      return CombineLD;
  }

  return SDValue();
}

SDValue DAGCombiner::visitBUILD_PAIR(SDNode *N) {
  EVT VT = N->getValueType(0);
  return CombineConsecutiveLoads(N, VT);
}

/// ConstantFoldBIT_CONVERTofBUILD_VECTOR - We know that BV is a build_vector
/// node with Constant, ConstantFP or Undef operands.  DstEltVT indicates the
/// destination element value type.
SDValue DAGCombiner::
ConstantFoldBIT_CONVERTofBUILD_VECTOR(SDNode *BV, EVT DstEltVT) {
  EVT SrcEltVT = BV->getValueType(0).getVectorElementType();

  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDValue(BV, 0);

  unsigned SrcBitSize = SrcEltVT.getSizeInBits();
  unsigned DstBitSize = DstEltVT.getSizeInBits();

  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
      SDValue Op = BV->getOperand(i);
      // If the vector element type is not legal, the BUILD_VECTOR operands
      // are promoted and implicitly truncated.  Make that explicit here.
      if (Op.getValueType() != SrcEltVT)
        Op = DAG.getNode(ISD::TRUNCATE, BV->getDebugLoc(), SrcEltVT, Op);
      Ops.push_back(DAG.getNode(ISD::BIT_CONVERT, BV->getDebugLoc(),
                                DstEltVT, Op));
      AddToWorkList(Ops.back().getNode());
    }
    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                              BV->getValueType(0).getVectorNumElements());
    return DAG.getNode(ISD::BUILD_VECTOR, BV->getDebugLoc(), VT,
                       &Ops[0], Ops.size());
  }

  // Otherwise, we're growing or shrinking the elements.  To avoid having to
  // handle annoying details of growing/shrinking FP values, we convert them to
  // int first.
  if (SrcEltVT.isFloatingPoint()) {
    // Convert the input float vector to a int vector where the elements are the
    // same sizes.
    assert((SrcEltVT == MVT::f32 || SrcEltVT == MVT::f64) && "Unknown FP VT!");
    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), SrcEltVT.getSizeInBits());
    BV = ConstantFoldBIT_CONVERTofBUILD_VECTOR(BV, IntVT).getNode();
    SrcEltVT = IntVT;
  }

  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (DstEltVT.isFloatingPoint()) {
    assert((DstEltVT == MVT::f32 || DstEltVT == MVT::f64) && "Unknown FP VT!");
    EVT TmpVT = EVT::getIntegerVT(*DAG.getContext(), DstEltVT.getSizeInBits());
    SDNode *Tmp = ConstantFoldBIT_CONVERTofBUILD_VECTOR(BV, TmpVT).getNode();

    // Next, convert to FP elements of the same size.
    return ConstantFoldBIT_CONVERTofBUILD_VECTOR(Tmp, DstEltVT);
  }

  // Okay, we know the src/dst types are both integers of differing types.
  // Handling growing first.
  assert(SrcEltVT.isInteger() && DstEltVT.isInteger());
  if (SrcBitSize < DstBitSize) {
    unsigned NumInputsPerOutput = DstBitSize/SrcBitSize;

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e;
         i += NumInputsPerOutput) {
      bool isLE = TLI.isLittleEndian();
      APInt NewBits = APInt(DstBitSize, 0);
      bool EltIsUndef = true;
      for (unsigned j = 0; j != NumInputsPerOutput; ++j) {
        // Shift the previously computed bits over.
        NewBits <<= SrcBitSize;
        SDValue Op = BV->getOperand(i+ (isLE ? (NumInputsPerOutput-j-1) : j));
        if (Op.getOpcode() == ISD::UNDEF) continue;
        EltIsUndef = false;

        NewBits |= (APInt(cast<ConstantSDNode>(Op)->getAPIntValue()).
                    zextOrTrunc(SrcBitSize).zext(DstBitSize));
      }

      if (EltIsUndef)
        Ops.push_back(DAG.getUNDEF(DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DstEltVT));
    }

    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT, Ops.size());
    return DAG.getNode(ISD::BUILD_VECTOR, BV->getDebugLoc(), VT,
                       &Ops[0], Ops.size());
  }

  // Finally, this must be the case where we are shrinking elements: each input
  // turns into multiple outputs.
  bool isS2V = ISD::isScalarToVector(BV);
  unsigned NumOutputsPerInput = SrcBitSize/DstBitSize;
  EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                            NumOutputsPerInput*BV->getNumOperands());
  SmallVector<SDValue, 8> Ops;

  for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
    if (BV->getOperand(i).getOpcode() == ISD::UNDEF) {
      for (unsigned j = 0; j != NumOutputsPerInput; ++j)
        Ops.push_back(DAG.getUNDEF(DstEltVT));
      continue;
    }

    APInt OpVal = APInt(cast<ConstantSDNode>(BV->getOperand(i))->
                        getAPIntValue()).zextOrTrunc(SrcBitSize);

    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      APInt ThisVal = APInt(OpVal).trunc(DstBitSize);
      Ops.push_back(DAG.getConstant(ThisVal, DstEltVT));
      if (isS2V && i == 0 && j == 0 && APInt(ThisVal).zext(SrcBitSize) == OpVal)
        // Simply turn this into a SCALAR_TO_VECTOR of the new type.
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, BV->getDebugLoc(), VT,
                           Ops[0]);
      OpVal = OpVal.lshr(DstBitSize);
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (TLI.isBigEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }

  return DAG.getNode(ISD::BUILD_VECTOR, BV->getDebugLoc(), VT,
                     &Ops[0], Ops.size());
}

SDValue DAGCombiner::visitFADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fadd c1, c2) -> (fadd c1, c2)
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FADD, N->getDebugLoc(), VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, N->getDebugLoc(), VT, N1, N0);
  // fold (fadd A, 0) -> A
  if (UnsafeFPMath && N1CFP && N1CFP->getValueAPF().isZero())
    return N0;
  // fold (fadd A, (fneg B)) -> (fsub A, B)
  if (isNegatibleForFree(N1, LegalOperations) == 2)
    return DAG.getNode(ISD::FSUB, N->getDebugLoc(), VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));
  // fold (fadd (fneg A), B) -> (fsub B, A)
  if (isNegatibleForFree(N0, LegalOperations) == 2)
    return DAG.getNode(ISD::FSUB, N->getDebugLoc(), VT, N1,
                       GetNegatedExpression(N0, DAG, LegalOperations));

  // If allowed, fold (fadd (fadd x, c1), c2) -> (fadd x, (fadd c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FADD &&
      N0.getNode()->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FADD, N->getDebugLoc(), VT, N0.getOperand(0),
                       DAG.getNode(ISD::FADD, N->getDebugLoc(), VT,
                                   N0.getOperand(1), N1));

  return SDValue();
}

SDValue DAGCombiner::visitFSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FSUB, N->getDebugLoc(), VT, N0, N1);
  // fold (fsub A, 0) -> A
  if (UnsafeFPMath && N1CFP && N1CFP->getValueAPF().isZero())
    return N0;
  // fold (fsub 0, B) -> -B
  if (UnsafeFPMath && N0CFP && N0CFP->getValueAPF().isZero()) {
    if (isNegatibleForFree(N1, LegalOperations))
      return GetNegatedExpression(N1, DAG, LegalOperations);
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, N->getDebugLoc(), VT, N1);
  }
  // fold (fsub A, (fneg B)) -> (fadd A, B)
  if (isNegatibleForFree(N1, LegalOperations))
    return DAG.getNode(ISD::FADD, N->getDebugLoc(), VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));

  return SDValue();
}

SDValue DAGCombiner::visitFMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FMUL, N->getDebugLoc(), VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMUL, N->getDebugLoc(), VT, N1, N0);
  // fold (fmul A, 0) -> 0
  if (UnsafeFPMath && N1CFP && N1CFP->getValueAPF().isZero())
    return N1;
  // fold (fmul A, 0) -> 0, vector edition.
  if (UnsafeFPMath && ISD::isBuildVectorAllZeros(N1.getNode()))
    return N1;
  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, N->getDebugLoc(), VT, N0, N0);
  // fold (fmul X, -1.0) -> (fneg X)
  if (N1CFP && N1CFP->isExactlyValue(-1.0))
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, N->getDebugLoc(), VT, N0);

  // fold (fmul (fneg X), (fneg Y)) -> (fmul X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FMUL, N->getDebugLoc(), VT,
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }

  // If allowed, fold (fmul (fmul x, c1), c2) -> (fmul x, (fmul c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FMUL &&
      N0.getNode()->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FMUL, N->getDebugLoc(), VT, N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, N->getDebugLoc(), VT,
                                   N0.getOperand(1), N1));

  return SDValue();
}

SDValue DAGCombiner::visitFDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FDIV, N->getDebugLoc(), VT, N0, N1);


  // (fdiv (fneg X), (fneg Y)) -> (fdiv X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FDIV, N->getDebugLoc(), VT,
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  // fold (frem c1, c2) -> fmod(c1,c2)
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FREM, N->getDebugLoc(), VT, N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  if (N0CFP && N1CFP && VT != MVT::ppcf128)  // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, N->getDebugLoc(), VT, N0, N1);

  if (N1CFP) {
    const APFloat& V = N1CFP->getValueAPF();
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    if (!V.isNegative()) {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FABS, VT))
        return DAG.getNode(ISD::FABS, N->getDebugLoc(), VT, N0);
    } else {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, N->getDebugLoc(), VT,
                           DAG.getNode(ISD::FABS, N0.getDebugLoc(), VT, N0));
    }
  }

  // copysign(fabs(x), y) -> copysign(x, y)
  // copysign(fneg(x), y) -> copysign(x, y)
  // copysign(copysign(x,z), y) -> copysign(x, y)
  if (N0.getOpcode() == ISD::FABS || N0.getOpcode() == ISD::FNEG ||
      N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, N->getDebugLoc(), VT,
                       N0.getOperand(0), N1);

  // copysign(x, abs(y)) -> abs(x)
  if (N1.getOpcode() == ISD::FABS)
    return DAG.getNode(ISD::FABS, N->getDebugLoc(), VT, N0);

  // copysign(x, copysign(y,z)) -> copysign(x, z)
  if (N1.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, N->getDebugLoc(), VT,
                       N0, N1.getOperand(1));

  // copysign(x, fp_extend(y)) -> copysign(x, y)
  // copysign(x, fp_round(y)) -> copysign(x, y)
  if (N1.getOpcode() == ISD::FP_EXTEND || N1.getOpcode() == ISD::FP_ROUND)
    return DAG.getNode(ISD::FCOPYSIGN, N->getDebugLoc(), VT,
                       N0, N1.getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (sint_to_fp c1) -> c1fp
  if (N0C && OpVT != MVT::ppcf128)
    return DAG.getNode(ISD::SINT_TO_FP, N->getDebugLoc(), VT, N0);

  // If the input is a legal type, and SINT_TO_FP is not legal on this target,
  // but UINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to UINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UINT_TO_FP, N->getDebugLoc(), VT, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (uint_to_fp c1) -> c1fp
  if (N0C && OpVT != MVT::ppcf128)
    return DAG.getNode(ISD::UINT_TO_FP, N->getDebugLoc(), VT, N0);

  // If the input is a legal type, and UINT_TO_FP is not legal on this target,
  // but SINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to SINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::SINT_TO_FP, N->getDebugLoc(), VT, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_SINT, N->getDebugLoc(), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FP_TO_UINT, N->getDebugLoc(), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_round c1fp) -> c1fp
  if (N0CFP && N0.getValueType() != MVT::ppcf128)
    return DAG.getNode(ISD::FP_ROUND, N->getDebugLoc(), VT, N0, N1);

  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);

  // fold (fp_round (fp_round x)) -> (fp_round x)
  if (N0.getOpcode() == ISD::FP_ROUND) {
    // This is a value preserving truncation if both round's are.
    bool IsTrunc = N->getConstantOperandVal(1) == 1 &&
                   N0.getNode()->getConstantOperandVal(1) == 1;
    return DAG.getNode(ISD::FP_ROUND, N->getDebugLoc(), VT, N0.getOperand(0),
                       DAG.getIntPtrConstant(IsTrunc));
  }

  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse()) {
    SDValue Tmp = DAG.getNode(ISD::FP_ROUND, N0.getDebugLoc(), VT,
                              N0.getOperand(0), N1);
    AddToWorkList(Tmp.getNode());
    return DAG.getNode(ISD::FCOPYSIGN, N->getDebugLoc(), VT,
                       Tmp, N0.getOperand(1));
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);

  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP && (TLI.isTypeLegal(EVT) || !LegalTypes)) {
    SDValue Round = DAG.getConstantFP(*N0CFP->getConstantFPValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, N->getDebugLoc(), VT, Round);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // If this is fp_round(fpextend), don't fold it, allow ourselves to be folded.
  if (N->hasOneUse() &&
      N->use_begin()->getOpcode() == ISD::FP_ROUND)
    return SDValue();

  // fold (fp_extend c1fp) -> c1fp
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FP_EXTEND, N->getDebugLoc(), VT, N0);

  // Turn fp_extend(fp_round(X, 1)) -> x since the fp_round doesn't affect the
  // value of X.
  if (N0.getOpcode() == ISD::FP_ROUND
      && N0.getNode()->getConstantOperandVal(1) == 1) {
    SDValue In = N0.getOperand(0);
    if (In.getValueType() == VT) return In;
    if (VT.bitsLT(In.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, N->getDebugLoc(), VT,
                         In, N0.getOperand(1));
    return DAG.getNode(ISD::FP_EXTEND, N->getDebugLoc(), VT, In);
  }

  // fold (fpext (load x)) -> (fpext (fptrunc (extload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) && N0.hasOneUse() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, N->getDebugLoc(), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(),
                                     N0.getValueType(),
                                     LN0->isVolatile(), LN0->isNonTemporal(),
                                     LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::FP_ROUND, N0.getDebugLoc(),
                          N0.getValueType(), ExtLoad, DAG.getIntPtrConstant(1)),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  return SDValue();
}

SDValue DAGCombiner::visitFNEG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (isNegatibleForFree(N0, LegalOperations))
    return GetNegatedExpression(N0, DAG, LegalOperations);

  // Transform fneg(bitconvert(x)) -> bitconvert(x^sign) to avoid loading
  // constant pool values.
  if (N0.getOpcode() == ISD::BIT_CONVERT && 
      !VT.isVector() &&
      N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::XOR, N0.getDebugLoc(), IntVT, Int,
              DAG.getConstant(APInt::getSignBit(IntVT.getSizeInBits()), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BIT_CONVERT, N->getDebugLoc(),
                         VT, Int);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fabs c1) -> fabs(c1)
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FABS, N->getDebugLoc(), VT, N0);
  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, N->getDebugLoc(), VT, N0.getOperand(0));

  // Transform fabs(bitconvert(x)) -> bitconvert(x&~sign) to avoid loading
  // constant pool values.
  if (N0.getOpcode() == ISD::BIT_CONVERT && N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger() &&
      !N0.getOperand(0).getValueType().isVector()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::AND, N0.getDebugLoc(), IntVT, Int,
             DAG.getConstant(~APInt::getSignBit(IntVT.getSizeInBits()), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BIT_CONVERT, N->getDebugLoc(),
                         N->getValueType(0), Int);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitBRCOND(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // fold a brcond with a setcc condition into a BR_CC node if BR_CC is legal
  // on the target.
  if (N1.getOpcode() == ISD::SETCC &&
      TLI.isOperationLegalOrCustom(ISD::BR_CC, MVT::Other)) {
    return DAG.getNode(ISD::BR_CC, N->getDebugLoc(), MVT::Other,
                       Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }

  SDNode *Trunc = 0;
  if (N1.getOpcode() == ISD::TRUNCATE && N1.hasOneUse()) {
    // Look past truncate.
    Trunc = N1.getNode();
    N1 = N1.getOperand(0);
  }

  if (N1.hasOneUse() && N1.getOpcode() == ISD::SRL) {
    // Match this pattern so that we can generate simpler code:
    //
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = srl i32 %b, 1
    //   brcond i32 %c ...
    //
    // into
    // 
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = setcc eq %b, 0
    //   brcond %c ...
    //
    // This applies only when the AND constant value has one bit set and the
    // SRL constant is equal to the log2 of the AND constant. The back-end is
    // smart enough to convert the result into a TEST/JMP sequence.
    SDValue Op0 = N1.getOperand(0);
    SDValue Op1 = N1.getOperand(1);

    if (Op0.getOpcode() == ISD::AND &&
        Op1.getOpcode() == ISD::Constant) {
      SDValue AndOp1 = Op0.getOperand(1);

      if (AndOp1.getOpcode() == ISD::Constant) {
        const APInt &AndConst = cast<ConstantSDNode>(AndOp1)->getAPIntValue();

        if (AndConst.isPowerOf2() &&
            cast<ConstantSDNode>(Op1)->getAPIntValue()==AndConst.logBase2()) {
          SDValue SetCC =
            DAG.getSetCC(N->getDebugLoc(),
                         TLI.getSetCCResultType(Op0.getValueType()),
                         Op0, DAG.getConstant(0, Op0.getValueType()),
                         ISD::SETNE);

          SDValue NewBRCond = DAG.getNode(ISD::BRCOND, N->getDebugLoc(),
                                          MVT::Other, Chain, SetCC, N2);
          // Don't add the new BRCond into the worklist or else SimplifySelectCC
          // will convert it back to (X & C1) >> C2.
          CombineTo(N, NewBRCond, false);
          // Truncate is dead.
          if (Trunc) {
            removeFromWorkList(Trunc);
            DAG.DeleteNode(Trunc);
          }
          // Replace the uses of SRL with SETCC
          WorkListRemover DeadNodes(*this);
          DAG.ReplaceAllUsesOfValueWith(N1, SetCC, &DeadNodes);
          removeFromWorkList(N1.getNode());
          DAG.DeleteNode(N1.getNode());
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }
  }
  
  // Transform br(xor(x, y)) -> br(x != y)
  // Transform br(xor(xor(x,y), 1)) -> br (x == y)
  if (N1.hasOneUse() && N1.getOpcode() == ISD::XOR) {
    SDNode *TheXor = N1.getNode();
    SDValue Op0 = TheXor->getOperand(0);
    SDValue Op1 = TheXor->getOperand(1);
    if (Op0.getOpcode() == Op1.getOpcode()) {
      // Avoid missing important xor optimizations.
      SDValue Tmp = visitXOR(TheXor);
      if (Tmp.getNode()) {
        DEBUG(dbgs() << "\nReplacing.8 ";
              TheXor->dump(&DAG);
              dbgs() << "\nWith: ";
              Tmp.getNode()->dump(&DAG);
              dbgs() << '\n');
        WorkListRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(N1, Tmp, &DeadNodes);
        removeFromWorkList(TheXor);
        DAG.DeleteNode(TheXor);
        return DAG.getNode(ISD::BRCOND, N->getDebugLoc(),
                           MVT::Other, Chain, Tmp, N2);
      }
    }

    if (Op0.getOpcode() != ISD::SETCC && Op1.getOpcode() != ISD::SETCC) {
      bool Equal = false;
      if (ConstantSDNode *RHSCI = dyn_cast<ConstantSDNode>(Op0))
        if (RHSCI->getAPIntValue() == 1 && Op0.hasOneUse() &&
            Op0.getOpcode() == ISD::XOR) {
          TheXor = Op0.getNode();
          Equal = true;
        }

      SDValue NodeToReplace = Trunc ? SDValue(Trunc, 0) : N1;
      
      EVT SetCCVT = NodeToReplace.getValueType();
      if (LegalTypes)
        SetCCVT = TLI.getSetCCResultType(SetCCVT);
      SDValue SetCC = DAG.getSetCC(TheXor->getDebugLoc(),
                                   SetCCVT,
                                   Op0, Op1,
                                   Equal ? ISD::SETEQ : ISD::SETNE);
      // Replace the uses of XOR with SETCC
      WorkListRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(NodeToReplace, SetCC, &DeadNodes);
      removeFromWorkList(NodeToReplace.getNode());
      DAG.DeleteNode(NodeToReplace.getNode());
      return DAG.getNode(ISD::BRCOND, N->getDebugLoc(),
                         MVT::Other, Chain, SetCC, N2);
    }
  }

  return SDValue();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDValue DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDValue CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // Use SimplifySetCC to simplify SETCC's.
  SDValue Simp = SimplifySetCC(TLI.getSetCCResultType(CondLHS.getValueType()),
                               CondLHS, CondRHS, CC->get(), N->getDebugLoc(),
                               false);
  if (Simp.getNode()) AddToWorkList(Simp.getNode());

  // fold to a simpler setcc
  if (Simp.getNode() && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, N->getDebugLoc(), MVT::Other,
                       N->getOperand(0), Simp.getOperand(2),
                       Simp.getOperand(0), Simp.getOperand(1),
                       N->getOperand(4));

  return SDValue();
}

/// CombineToPreIndexedLoadStore - Try turning a load / store into a
/// pre-indexed load / store when the base pointer is an add or subtract
/// and it has other uses besides the load / store. After the
/// transformation, the new indexed load / store has effectively folded
/// the add / subtract in and all of its other uses are redirected to the
/// new load / store.
bool DAGCombiner::CombineToPreIndexedLoadStore(SDNode *N) {
  if (!LegalOperations)
    return false;

  bool isLoad = true;
  SDValue Ptr;
  EVT VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    VT = LD->getMemoryVT();
    if (!TLI.isIndexedLoadLegal(ISD::PRE_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::PRE_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    VT = ST->getMemoryVT();
    if (!TLI.isIndexedStoreLegal(ISD::PRE_INC, VT) &&
        !TLI.isIndexedStoreLegal(ISD::PRE_DEC, VT))
      return false;
    Ptr = ST->getBasePtr();
    isLoad = false;
  } else {
    return false;
  }

  // If the pointer is not an add/sub, or if it doesn't have multiple uses, bail
  // out.  There is no reason to make this a preinc/predec.
  if ((Ptr.getOpcode() != ISD::ADD && Ptr.getOpcode() != ISD::SUB) ||
      Ptr.getNode()->hasOneUse())
    return false;

  // Ask the target to do addressing mode selection.
  SDValue BasePtr;
  SDValue Offset;
  ISD::MemIndexedMode AM = ISD::UNINDEXED;
  if (!TLI.getPreIndexedAddressParts(N, BasePtr, Offset, AM, DAG))
    return false;
  // Don't create a indexed load / store with zero offset.
  if (isa<ConstantSDNode>(Offset) &&
      cast<ConstantSDNode>(Offset)->isNullValue())
    return false;

  // Try turning it into a pre-indexed load / store except when:
  // 1) The new base ptr is a frame index.
  // 2) If N is a store and the new base ptr is either the same as or is a
  //    predecessor of the value being stored.
  // 3) Another use of old base ptr is a predecessor of N. If ptr is folded
  //    that would create a cycle.
  // 4) All uses are load / store ops that use it as old base ptr.

  // Check #1.  Preinc'ing a frame index would require copying the stack pointer
  // (plus the implicit offset) to a register to preinc anyway.
  if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
    return false;

  // Check #2.
  if (!isLoad) {
    SDValue Val = cast<StoreSDNode>(N)->getValue();
    if (Val == BasePtr || BasePtr.getNode()->isPredecessorOf(Val.getNode()))
      return false;
  }

  // Now check for #3 and #4.
  bool RealUse = false;
  for (SDNode::use_iterator I = Ptr.getNode()->use_begin(),
         E = Ptr.getNode()->use_end(); I != E; ++I) {
    SDNode *Use = *I;
    if (Use == N)
      continue;
    if (Use->isPredecessorOf(N))
      return false;

    if (!((Use->getOpcode() == ISD::LOAD &&
           cast<LoadSDNode>(Use)->getBasePtr() == Ptr) ||
          (Use->getOpcode() == ISD::STORE &&
           cast<StoreSDNode>(Use)->getBasePtr() == Ptr)))
      RealUse = true;
  }

  if (!RealUse)
    return false;

  SDValue Result;
  if (isLoad)
    Result = DAG.getIndexedLoad(SDValue(N,0), N->getDebugLoc(),
                                BasePtr, Offset, AM);
  else
    Result = DAG.getIndexedStore(SDValue(N,0), N->getDebugLoc(),
                                 BasePtr, Offset, AM);
  ++PreIndexedNodes;
  ++NodesCombined;
  DEBUG(dbgs() << "\nReplacing.4 ";
        N->dump(&DAG);
        dbgs() << "\nWith: ";
        Result.getNode()->dump(&DAG);
        dbgs() << '\n');
  WorkListRemover DeadNodes(*this);
  if (isLoad) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0),
                                  &DeadNodes);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2),
                                  &DeadNodes);
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1),
                                  &DeadNodes);
  }

  // Finally, since the node is now dead, remove it from the graph.
  DAG.DeleteNode(N);

  // Replace the uses of Ptr with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(Ptr, Result.getValue(isLoad ? 1 : 0),
                                &DeadNodes);
  removeFromWorkList(Ptr.getNode());
  DAG.DeleteNode(Ptr.getNode());

  return true;
}

/// CombineToPostIndexedLoadStore - Try to combine a load / store with a
/// add / sub of the base pointer node into a post-indexed load / store.
/// The transformation folded the add / subtract into the new indexed
/// load / store effectively and all of its uses are redirected to the
/// new load / store.
bool DAGCombiner::CombineToPostIndexedLoadStore(SDNode *N) {
  if (!LegalOperations)
    return false;

  bool isLoad = true;
  SDValue Ptr;
  EVT VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    VT = LD->getMemoryVT();
    if (!TLI.isIndexedLoadLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    VT = ST->getMemoryVT();
    if (!TLI.isIndexedStoreLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedStoreLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = ST->getBasePtr();
    isLoad = false;
  } else {
    return false;
  }

  if (Ptr.getNode()->hasOneUse())
    return false;

  for (SDNode::use_iterator I = Ptr.getNode()->use_begin(),
         E = Ptr.getNode()->use_end(); I != E; ++I) {
    SDNode *Op = *I;
    if (Op == N ||
        (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB))
      continue;

    SDValue BasePtr;
    SDValue Offset;
    ISD::MemIndexedMode AM = ISD::UNINDEXED;
    if (TLI.getPostIndexedAddressParts(N, Op, BasePtr, Offset, AM, DAG)) {
      if (Ptr == Offset && Op->getOpcode() == ISD::ADD)
        std::swap(BasePtr, Offset);
      if (Ptr != BasePtr)
        continue;
      // Don't create a indexed load / store with zero offset.
      if (isa<ConstantSDNode>(Offset) &&
          cast<ConstantSDNode>(Offset)->isNullValue())
        continue;

      // Try turning it into a post-indexed load / store except when
      // 1) All uses are load / store ops that use it as base ptr.
      // 2) Op must be independent of N, i.e. Op is neither a predecessor
      //    nor a successor of N. Otherwise, if Op is folded that would
      //    create a cycle.

      if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
        continue;

      // Check for #1.
      bool TryNext = false;
      for (SDNode::use_iterator II = BasePtr.getNode()->use_begin(),
             EE = BasePtr.getNode()->use_end(); II != EE; ++II) {
        SDNode *Use = *II;
        if (Use == Ptr.getNode())
          continue;

        // If all the uses are load / store addresses, then don't do the
        // transformation.
        if (Use->getOpcode() == ISD::ADD || Use->getOpcode() == ISD::SUB){
          bool RealUse = false;
          for (SDNode::use_iterator III = Use->use_begin(),
                 EEE = Use->use_end(); III != EEE; ++III) {
            SDNode *UseUse = *III;
            if (!((UseUse->getOpcode() == ISD::LOAD &&
                   cast<LoadSDNode>(UseUse)->getBasePtr().getNode() == Use) ||
                  (UseUse->getOpcode() == ISD::STORE &&
                   cast<StoreSDNode>(UseUse)->getBasePtr().getNode() == Use)))
              RealUse = true;
          }

          if (!RealUse) {
            TryNext = true;
            break;
          }
        }
      }

      if (TryNext)
        continue;

      // Check for #2
      if (!Op->isPredecessorOf(N) && !N->isPredecessorOf(Op)) {
        SDValue Result = isLoad
          ? DAG.getIndexedLoad(SDValue(N,0), N->getDebugLoc(),
                               BasePtr, Offset, AM)
          : DAG.getIndexedStore(SDValue(N,0), N->getDebugLoc(),
                                BasePtr, Offset, AM);
        ++PostIndexedNodes;
        ++NodesCombined;
        DEBUG(dbgs() << "\nReplacing.5 ";
              N->dump(&DAG);
              dbgs() << "\nWith: ";
              Result.getNode()->dump(&DAG);
              dbgs() << '\n');
        WorkListRemover DeadNodes(*this);
        if (isLoad) {
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0),
                                        &DeadNodes);
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2),
                                        &DeadNodes);
        } else {
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1),
                                        &DeadNodes);
        }

        // Finally, since the node is now dead, remove it from the graph.
        DAG.DeleteNode(N);

        // Replace the uses of Use with uses of the updated base value.
        DAG.ReplaceAllUsesOfValueWith(SDValue(Op, 0),
                                      Result.getValue(isLoad ? 1 : 0),
                                      &DeadNodes);
        removeFromWorkList(Op);
        DAG.DeleteNode(Op);
        return true;
      }
    }
  }

  return false;
}

SDValue DAGCombiner::visitLOAD(SDNode *N) {
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr   = LD->getBasePtr();

  // Try to infer better alignment information than the load already has.
  if (OptLevel != CodeGenOpt::None && LD->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > LD->getAlignment())
        return DAG.getExtLoad(LD->getExtensionType(), N->getDebugLoc(),
                              LD->getValueType(0),
                              Chain, Ptr, LD->getSrcValue(),
                              LD->getSrcValueOffset(), LD->getMemoryVT(),
                              LD->isVolatile(), LD->isNonTemporal(), Align);
    }
  }

  // If load is not volatile and there are no uses of the loaded value (and
  // the updated indexed value in case of indexed loads), change uses of the
  // chain value into uses of the chain input (i.e. delete the dead load).
  if (!LD->isVolatile()) {
    if (N->getValueType(1) == MVT::Other) {
      // Unindexed loads.
      if (N->hasNUsesOfValue(0, 0)) {
        // It's not safe to use the two value CombineTo variant here. e.g.
        // v1, chain2 = load chain1, loc
        // v2, chain3 = load chain2, loc
        // v3         = add v2, c
        // Now we replace use of chain2 with chain1.  This makes the second load
        // isomorphic to the one we are deleting, and thus makes this load live.
        DEBUG(dbgs() << "\nReplacing.6 ";
              N->dump(&DAG);
              dbgs() << "\nWith chain: ";
              Chain.getNode()->dump(&DAG);
              dbgs() << "\n");
        WorkListRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain, &DeadNodes);

        if (N->use_empty()) {
          removeFromWorkList(N);
          DAG.DeleteNode(N);
        }

        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    } else {
      // Indexed loads.
      assert(N->getValueType(2) == MVT::Other && "Malformed indexed loads?");
      if (N->hasNUsesOfValue(0, 0) && N->hasNUsesOfValue(0, 1)) {
        SDValue Undef = DAG.getUNDEF(N->getValueType(0));
        DEBUG(dbgs() << "\nReplacing.7 ";
              N->dump(&DAG);
              dbgs() << "\nWith: ";
              Undef.getNode()->dump(&DAG);
              dbgs() << " and 2 other values\n");
        WorkListRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Undef, &DeadNodes);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1),
                                      DAG.getUNDEF(N->getValueType(1)),
                                      &DeadNodes);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 2), Chain, &DeadNodes);
        removeFromWorkList(N);
        DAG.DeleteNode(N);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  // If this load is directly stored, replace the load value with the stored
  // value.
  // TODO: Handle store large -> read small portion.
  // TODO: Handle TRUNCSTORE/LOADEXT
  if (LD->getExtensionType() == ISD::NON_EXTLOAD &&
      !LD->isVolatile()) {
    if (ISD::isNON_TRUNCStore(Chain.getNode())) {
      StoreSDNode *PrevST = cast<StoreSDNode>(Chain);
      if (PrevST->getBasePtr() == Ptr &&
          PrevST->getValue().getValueType() == N->getValueType(0))
      return CombineTo(N, Chain.getOperand(1), Chain);
    }
  }

  if (CombinerAA) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplLoad;

      // Replace the chain to void dependency.
      if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
        ReplLoad = DAG.getLoad(N->getValueType(0), LD->getDebugLoc(),
                               BetterChain, Ptr,
                               LD->getSrcValue(), LD->getSrcValueOffset(),
                               LD->isVolatile(), LD->isNonTemporal(),
                               LD->getAlignment());
      } else {
        ReplLoad = DAG.getExtLoad(LD->getExtensionType(), LD->getDebugLoc(),
                                  LD->getValueType(0),
                                  BetterChain, Ptr, LD->getSrcValue(),
                                  LD->getSrcValueOffset(),
                                  LD->getMemoryVT(),
                                  LD->isVolatile(),
                                  LD->isNonTemporal(),
                                  LD->getAlignment());
      }

      // Create token factor to keep old chain connected.
      SDValue Token = DAG.getNode(ISD::TokenFactor, N->getDebugLoc(),
                                  MVT::Other, Chain, ReplLoad.getValue(1));
      
      // Make sure the new and old chains are cleaned up.
      AddToWorkList(Token.getNode());
      
      // Replace uses with load result and token factor. Don't add users
      // to work list.
      return CombineTo(N, ReplLoad.getValue(0), Token, false);
    }
  }

  // Try transforming N to an indexed load.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  return SDValue();
}


/// ReduceLoadOpStoreWidth - Look for sequence of load / op / store where op is
/// one of 'or', 'xor', and 'and' of immediates. If 'op' is only touching some
/// of the loaded bits, try narrowing the load and store if it would end up
/// being a win for performance or code size.
SDValue DAGCombiner::ReduceLoadOpStoreWidth(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  if (ST->isVolatile())
    return SDValue();

  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();
  EVT VT = Value.getValueType();

  if (ST->isTruncatingStore() || VT.isVector() || !Value.hasOneUse())
    return SDValue();

  unsigned Opc = Value.getOpcode();
  if ((Opc != ISD::OR && Opc != ISD::XOR && Opc != ISD::AND) ||
      Value.getOperand(1).getOpcode() != ISD::Constant)
    return SDValue();

  SDValue N0 = Value.getOperand(0);
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LD = cast<LoadSDNode>(N0);
    if (LD->getBasePtr() != Ptr)
      return SDValue();

    // Find the type to narrow it the load / op / store to.
    SDValue N1 = Value.getOperand(1);
    unsigned BitWidth = N1.getValueSizeInBits();
    APInt Imm = cast<ConstantSDNode>(N1)->getAPIntValue();
    if (Opc == ISD::AND)
      Imm ^= APInt::getAllOnesValue(BitWidth);
    if (Imm == 0 || Imm.isAllOnesValue())
      return SDValue();
    unsigned ShAmt = Imm.countTrailingZeros();
    unsigned MSB = BitWidth - Imm.countLeadingZeros() - 1;
    unsigned NewBW = NextPowerOf2(MSB - ShAmt);
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    while (NewBW < BitWidth &&
           !(TLI.isOperationLegalOrCustom(Opc, NewVT) &&
             TLI.isNarrowingProfitable(VT, NewVT))) {
      NewBW = NextPowerOf2(NewBW);
      NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    }
    if (NewBW >= BitWidth)
      return SDValue();

    // If the lsb changed does not start at the type bitwidth boundary,
    // start at the previous one.
    if (ShAmt % NewBW)
      ShAmt = (((ShAmt + NewBW - 1) / NewBW) * NewBW) - NewBW;
    APInt Mask = APInt::getBitsSet(BitWidth, ShAmt, ShAmt + NewBW);
    if ((Imm & Mask) == Imm) {
      APInt NewImm = (Imm & Mask).lshr(ShAmt).trunc(NewBW);
      if (Opc == ISD::AND)
        NewImm ^= APInt::getAllOnesValue(NewBW);
      uint64_t PtrOff = ShAmt / 8;
      // For big endian targets, we need to adjust the offset to the pointer to
      // load the correct bytes.
      if (TLI.isBigEndian())
        PtrOff = (BitWidth + 7 - NewBW) / 8 - PtrOff;

      unsigned NewAlign = MinAlign(LD->getAlignment(), PtrOff);
      if (NewAlign <
          TLI.getTargetData()->getABITypeAlignment(NewVT.getTypeForEVT(*DAG.getContext())))
        return SDValue();

      SDValue NewPtr = DAG.getNode(ISD::ADD, LD->getDebugLoc(),
                                   Ptr.getValueType(), Ptr,
                                   DAG.getConstant(PtrOff, Ptr.getValueType()));
      SDValue NewLD = DAG.getLoad(NewVT, N0.getDebugLoc(),
                                  LD->getChain(), NewPtr,
                                  LD->getSrcValue(), LD->getSrcValueOffset(),
                                  LD->isVolatile(), LD->isNonTemporal(),
                                  NewAlign);
      SDValue NewVal = DAG.getNode(Opc, Value.getDebugLoc(), NewVT, NewLD,
                                   DAG.getConstant(NewImm, NewVT));
      SDValue NewST = DAG.getStore(Chain, N->getDebugLoc(),
                                   NewVal, NewPtr,
                                   ST->getSrcValue(), ST->getSrcValueOffset(),
                                   false, false, NewAlign);

      AddToWorkList(NewPtr.getNode());
      AddToWorkList(NewLD.getNode());
      AddToWorkList(NewVal.getNode());
      WorkListRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLD.getValue(1),
                                    &DeadNodes);
      ++OpsNarrowed;
      return NewST;
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();

  // Try to infer better alignment information than the store already has.
  if (OptLevel != CodeGenOpt::None && ST->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > ST->getAlignment())
        return DAG.getTruncStore(Chain, N->getDebugLoc(), Value,
                                 Ptr, ST->getSrcValue(),
                                 ST->getSrcValueOffset(), ST->getMemoryVT(),
                                 ST->isVolatile(), ST->isNonTemporal(), Align);
    }
  }

  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BIT_CONVERT && !ST->isTruncatingStore() &&
      ST->isUnindexed()) {
    unsigned OrigAlign = ST->getAlignment();
    EVT SVT = Value.getOperand(0).getValueType();
    unsigned Align = TLI.getTargetData()->
      getABITypeAlignment(SVT.getTypeForEVT(*DAG.getContext()));
    if (Align <= OrigAlign &&
        ((!LegalOperations && !ST->isVolatile()) ||
         TLI.isOperationLegalOrCustom(ISD::STORE, SVT)))
      return DAG.getStore(Chain, N->getDebugLoc(), Value.getOperand(0),
                          Ptr, ST->getSrcValue(),
                          ST->getSrcValueOffset(), ST->isVolatile(),
                          ST->isNonTemporal(), OrigAlign);
  }

  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Value)) {
    // NOTE: If the original store is volatile, this transform must not increase
    // the number of stores.  For example, on x86-32 an f64 can be stored in one
    // processor operation but an i64 (which is not legal) requires two.  So the
    // transform should not be done in this case.
    if (Value.getOpcode() != ISD::TargetConstantFP) {
      SDValue Tmp;
      switch (CFP->getValueType(0).getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unknown FP type");
      case MVT::f80:    // We don't do this for these yet.
      case MVT::f128:
      case MVT::ppcf128:
        break;
      case MVT::f32:
        if (((TLI.isTypeLegal(MVT::i32) || !LegalTypes) && !LegalOperations &&
             !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          Tmp = DAG.getConstant((uint32_t)CFP->getValueAPF().
                              bitcastToAPInt().getZExtValue(), MVT::i32);
          return DAG.getStore(Chain, N->getDebugLoc(), Tmp,
                              Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset(), ST->isVolatile(),
                              ST->isNonTemporal(), ST->getAlignment());
        }
        break;
      case MVT::f64:
        if (((TLI.isTypeLegal(MVT::i64) || !LegalTypes) && !LegalOperations &&
             !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i64)) {
          Tmp = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                getZExtValue(), MVT::i64);
          return DAG.getStore(Chain, N->getDebugLoc(), Tmp,
                              Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset(), ST->isVolatile(),
                              ST->isNonTemporal(), ST->getAlignment());
        } else if (!ST->isVolatile() &&
                   TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          // Many FP stores are not made apparent until after legalize, e.g. for
          // argument passing.  Since this is so common, custom legalize the
          // 64-bit integer store into two 32-bit stores.
          uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
          SDValue Lo = DAG.getConstant(Val & 0xFFFFFFFF, MVT::i32);
          SDValue Hi = DAG.getConstant(Val >> 32, MVT::i32);
          if (TLI.isBigEndian()) std::swap(Lo, Hi);

          int SVOffset = ST->getSrcValueOffset();
          unsigned Alignment = ST->getAlignment();
          bool isVolatile = ST->isVolatile();
          bool isNonTemporal = ST->isNonTemporal();

          SDValue St0 = DAG.getStore(Chain, ST->getDebugLoc(), Lo,
                                     Ptr, ST->getSrcValue(),
                                     ST->getSrcValueOffset(),
                                     isVolatile, isNonTemporal,
                                     ST->getAlignment());
          Ptr = DAG.getNode(ISD::ADD, N->getDebugLoc(), Ptr.getValueType(), Ptr,
                            DAG.getConstant(4, Ptr.getValueType()));
          SVOffset += 4;
          Alignment = MinAlign(Alignment, 4U);
          SDValue St1 = DAG.getStore(Chain, ST->getDebugLoc(), Hi,
                                     Ptr, ST->getSrcValue(),
                                     SVOffset, isVolatile, isNonTemporal,
                                     Alignment);
          return DAG.getNode(ISD::TokenFactor, N->getDebugLoc(), MVT::Other,
                             St0, St1);
        }

        break;
      }
    }
  }

  if (CombinerAA) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplStore;

      // Replace the chain to avoid dependency.
      if (ST->isTruncatingStore()) {
        ReplStore = DAG.getTruncStore(BetterChain, N->getDebugLoc(), Value, Ptr,
                                      ST->getSrcValue(),ST->getSrcValueOffset(),
                                      ST->getMemoryVT(), ST->isVolatile(),
                                      ST->isNonTemporal(), ST->getAlignment());
      } else {
        ReplStore = DAG.getStore(BetterChain, N->getDebugLoc(), Value, Ptr,
                                 ST->getSrcValue(), ST->getSrcValueOffset(),
                                 ST->isVolatile(), ST->isNonTemporal(),
                                 ST->getAlignment());
      }

      // Create token to keep both nodes around.
      SDValue Token = DAG.getNode(ISD::TokenFactor, N->getDebugLoc(),
                                  MVT::Other, Chain, ReplStore);

      // Make sure the new and old chains are cleaned up.
      AddToWorkList(Token.getNode());

      // Don't add users to work list.
      return CombineTo(N, Token, false);
    }
  }

  // Try transforming N to an indexed store.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  // FIXME: is there such a thing as a truncating indexed store?
  if (ST->isTruncatingStore() && ST->isUnindexed() &&
      Value.getValueType().isInteger()) {
    // See if we can simplify the input to this truncstore with knowledge that
    // only the low bits are being used.  For example:
    // "truncstore (or (shl x, 8), y), i8"  -> "truncstore y, i8"
    SDValue Shorter =
      GetDemandedBits(Value,
                      APInt::getLowBitsSet(Value.getValueSizeInBits(),
                                           ST->getMemoryVT().getSizeInBits()));
    AddToWorkList(Value.getNode());
    if (Shorter.getNode())
      return DAG.getTruncStore(Chain, N->getDebugLoc(), Shorter,
                               Ptr, ST->getSrcValue(),
                               ST->getSrcValueOffset(), ST->getMemoryVT(),
                               ST->isVolatile(), ST->isNonTemporal(),
                               ST->getAlignment());

    // Otherwise, see if we can simplify the operation with
    // SimplifyDemandedBits, which only works if the value has a single use.
    if (SimplifyDemandedBits(Value,
                             APInt::getLowBitsSet(
                               Value.getValueType().getScalarType().getSizeInBits(),
                               ST->getMemoryVT().getSizeInBits())))
      return SDValue(N, 0);
  }

  // If this is a load followed by a store to the same location, then the store
  // is dead/noop.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Value)) {
    if (Ld->getBasePtr() == Ptr && ST->getMemoryVT() == Ld->getMemoryVT() &&
        ST->isUnindexed() && !ST->isVolatile() &&
        // There can't be any side effects between the load and store, such as
        // a call or store.
        Chain.reachesChainWithoutSideEffects(SDValue(Ld, 1))) {
      // The store is dead, remove it.
      return Chain;
    }
  }

  // If this is an FP_ROUND or TRUNC followed by a store, fold this into a
  // truncating store.  We can do this even if this is already a truncstore.
  if ((Value.getOpcode() == ISD::FP_ROUND || Value.getOpcode() == ISD::TRUNCATE)
      && Value.getNode()->hasOneUse() && ST->isUnindexed() &&
      TLI.isTruncStoreLegal(Value.getOperand(0).getValueType(),
                            ST->getMemoryVT())) {
    return DAG.getTruncStore(Chain, N->getDebugLoc(), Value.getOperand(0),
                             Ptr, ST->getSrcValue(),
                             ST->getSrcValueOffset(), ST->getMemoryVT(),
                             ST->isVolatile(), ST->isNonTemporal(),
                             ST->getAlignment());
  }

  return ReduceLoadOpStoreWidth(N);
}

SDValue DAGCombiner::visitINSERT_VECTOR_ELT(SDNode *N) {
  SDValue InVec = N->getOperand(0);
  SDValue InVal = N->getOperand(1);
  SDValue EltNo = N->getOperand(2);

  // If the invec is a BUILD_VECTOR and if EltNo is a constant, build a new
  // vector with the inserted element.
  if (InVec.getOpcode() == ISD::BUILD_VECTOR && isa<ConstantSDNode>(EltNo)) {
    unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    SmallVector<SDValue, 8> Ops(InVec.getNode()->op_begin(),
                                InVec.getNode()->op_end());
    if (Elt < Ops.size())
      Ops[Elt] = InVal;
    return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(),
                       InVec.getValueType(), &Ops[0], Ops.size());
  }
  // If the invec is an UNDEF and if EltNo is a constant, create a new 
  // BUILD_VECTOR with undef elements and the inserted element.
  if (!LegalOperations && InVec.getOpcode() == ISD::UNDEF && 
      isa<ConstantSDNode>(EltNo)) {
    EVT VT = InVec.getValueType();
    EVT EltVT = VT.getVectorElementType();
    unsigned NElts = VT.getVectorNumElements();
    SmallVector<SDValue, 8> Ops(NElts, DAG.getUNDEF(EltVT));

    unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    if (Elt < Ops.size())
      Ops[Elt] = InVal;
    return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(),
                       InVec.getValueType(), &Ops[0], Ops.size());
  }
  return SDValue();
}

SDValue DAGCombiner::visitEXTRACT_VECTOR_ELT(SDNode *N) {
  // (vextract (scalar_to_vector val, 0) -> val
  SDValue InVec = N->getOperand(0);

 if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR) {
   // Check if the result type doesn't match the inserted element type. A
   // SCALAR_TO_VECTOR may truncate the inserted element and the
   // EXTRACT_VECTOR_ELT may widen the extracted vector.
   EVT EltVT = InVec.getValueType().getVectorElementType();
   SDValue InOp = InVec.getOperand(0);
   EVT NVT = N->getValueType(0);
   if (InOp.getValueType() != NVT) {
     assert(InOp.getValueType().isInteger() && NVT.isInteger());
     return DAG.getSExtOrTrunc(InOp, InVec.getDebugLoc(), NVT);
   }
   return InOp;
 }

  // Perform only after legalization to ensure build_vector / vector_shuffle
  // optimizations have already been done.
  if (!LegalOperations) return SDValue();

  // (vextract (v4f32 load $addr), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 s2v (f32 load $addr)), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 shuffle (load $addr), <1,u,u,u>), 0) -> (f32 load $addr)
  SDValue EltNo = N->getOperand(1);

  if (isa<ConstantSDNode>(EltNo)) {
    unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    bool NewLoad = false;
    bool BCNumEltsChanged = false;
    EVT VT = InVec.getValueType();
    EVT ExtVT = VT.getVectorElementType();
    EVT LVT = ExtVT;

    if (InVec.getOpcode() == ISD::BIT_CONVERT) {
      EVT BCVT = InVec.getOperand(0).getValueType();
      if (!BCVT.isVector() || ExtVT.bitsGT(BCVT.getVectorElementType()))
        return SDValue();
      if (VT.getVectorNumElements() != BCVT.getVectorNumElements())
        BCNumEltsChanged = true;
      InVec = InVec.getOperand(0);
      ExtVT = BCVT.getVectorElementType();
      NewLoad = true;
    }

    LoadSDNode *LN0 = NULL;
    const ShuffleVectorSDNode *SVN = NULL;
    if (ISD::isNormalLoad(InVec.getNode())) {
      LN0 = cast<LoadSDNode>(InVec);
    } else if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR &&
               InVec.getOperand(0).getValueType() == ExtVT &&
               ISD::isNormalLoad(InVec.getOperand(0).getNode())) {
      LN0 = cast<LoadSDNode>(InVec.getOperand(0));
    } else if ((SVN = dyn_cast<ShuffleVectorSDNode>(InVec))) {
      // (vextract (vector_shuffle (load $addr), v2, <1, u, u, u>), 1)
      // =>
      // (load $addr+1*size)

      // If the bit convert changed the number of elements, it is unsafe
      // to examine the mask.
      if (BCNumEltsChanged)
        return SDValue();

      // Select the input vector, guarding against out of range extract vector.
      unsigned NumElems = VT.getVectorNumElements();
      int Idx = (Elt > NumElems) ? -1 : SVN->getMaskElt(Elt);
      InVec = (Idx < (int)NumElems) ? InVec.getOperand(0) : InVec.getOperand(1);

      if (InVec.getOpcode() == ISD::BIT_CONVERT)
        InVec = InVec.getOperand(0);
      if (ISD::isNormalLoad(InVec.getNode())) {
        LN0 = cast<LoadSDNode>(InVec);
        Elt = (Idx < (int)NumElems) ? Idx : Idx - NumElems;
      }
    }

    if (!LN0 || !LN0->hasOneUse() || LN0->isVolatile())
      return SDValue();

    unsigned Align = LN0->getAlignment();
    if (NewLoad) {
      // Check the resultant load doesn't need a higher alignment than the
      // original load.
      unsigned NewAlign =
        TLI.getTargetData()->getABITypeAlignment(LVT.getTypeForEVT(*DAG.getContext()));

      if (NewAlign > Align || !TLI.isOperationLegalOrCustom(ISD::LOAD, LVT))
        return SDValue();

      Align = NewAlign;
    }

    SDValue NewPtr = LN0->getBasePtr();
    if (Elt) {
      unsigned PtrOff = LVT.getSizeInBits() * Elt / 8;
      EVT PtrType = NewPtr.getValueType();
      if (TLI.isBigEndian())
        PtrOff = VT.getSizeInBits() / 8 - PtrOff;
      NewPtr = DAG.getNode(ISD::ADD, N->getDebugLoc(), PtrType, NewPtr,
                           DAG.getConstant(PtrOff, PtrType));
    }

    return DAG.getLoad(LVT, N->getDebugLoc(), LN0->getChain(), NewPtr,
                       LN0->getSrcValue(), LN0->getSrcValueOffset(),
                       LN0->isVolatile(), LN0->isNonTemporal(), Align);
  }

  return SDValue();
}

SDValue DAGCombiner::visitBUILD_VECTOR(SDNode *N) {
  unsigned NumInScalars = N->getNumOperands();
  EVT VT = N->getValueType(0);

  // Check to see if this is a BUILD_VECTOR of a bunch of EXTRACT_VECTOR_ELT
  // operations.  If so, and if the EXTRACT_VECTOR_ELT vector inputs come from
  // at most two distinct vectors, turn this into a shuffle node.
  SDValue VecIn1, VecIn2;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    // Ignore undef inputs.
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;

    // If this input is something other than a EXTRACT_VECTOR_ELT with a
    // constant index, bail out.
    if (N->getOperand(i).getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(N->getOperand(i).getOperand(1))) {
      VecIn1 = VecIn2 = SDValue(0, 0);
      break;
    }

    // If the input vector type disagrees with the result of the build_vector,
    // we can't make a shuffle.
    SDValue ExtractedFromVec = N->getOperand(i).getOperand(0);
    if (ExtractedFromVec.getValueType() != VT) {
      VecIn1 = VecIn2 = SDValue(0, 0);
      break;
    }

    // Otherwise, remember this.  We allow up to two distinct input vectors.
    if (ExtractedFromVec == VecIn1 || ExtractedFromVec == VecIn2)
      continue;

    if (VecIn1.getNode() == 0) {
      VecIn1 = ExtractedFromVec;
    } else if (VecIn2.getNode() == 0) {
      VecIn2 = ExtractedFromVec;
    } else {
      // Too many inputs.
      VecIn1 = VecIn2 = SDValue(0, 0);
      break;
    }
  }

  // If everything is good, we can make a shuffle operation.
  if (VecIn1.getNode()) {
    SmallVector<int, 8> Mask;
    for (unsigned i = 0; i != NumInScalars; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::UNDEF) {
        Mask.push_back(-1);
        continue;
      }

      // If extracting from the first vector, just use the index directly.
      SDValue Extract = N->getOperand(i);
      SDValue ExtVal = Extract.getOperand(1);
      if (Extract.getOperand(0) == VecIn1) {
        unsigned ExtIndex = cast<ConstantSDNode>(ExtVal)->getZExtValue();
        if (ExtIndex > VT.getVectorNumElements())
          return SDValue();
        
        Mask.push_back(ExtIndex);
        continue;
      }

      // Otherwise, use InIdx + VecSize
      unsigned Idx = cast<ConstantSDNode>(ExtVal)->getZExtValue();
      Mask.push_back(Idx+NumInScalars);
    }

    // Add count and size info.
    if (!TLI.isTypeLegal(VT) && LegalTypes)
      return SDValue();

    // Return the new VECTOR_SHUFFLE node.
    SDValue Ops[2];
    Ops[0] = VecIn1;
    Ops[1] = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(VT);
    return DAG.getVectorShuffle(VT, N->getDebugLoc(), Ops[0], Ops[1], &Mask[0]);
  }

  return SDValue();
}

SDValue DAGCombiner::visitCONCAT_VECTORS(SDNode *N) {
  // TODO: Check to see if this is a CONCAT_VECTORS of a bunch of
  // EXTRACT_SUBVECTOR operations.  If so, and if the EXTRACT_SUBVECTOR vector
  // inputs come from at most two distinct vectors, turn this into a shuffle
  // node.

  // If we only have one input vector, we don't need to do any concatenation.
  if (N->getNumOperands() == 1)
    return N->getOperand(0);

  return SDValue();
}

SDValue DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  return SDValue();
  
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);

  assert(N0.getValueType().getVectorNumElements() == NumElts &&
        "Vector shuffle must be normalized in DAG");

  // FIXME: implement canonicalizations from DAG.getVectorShuffle()

  // If it is a splat, check if the argument vector is a build_vector with
  // all scalar elements the same.
  if (cast<ShuffleVectorSDNode>(N)->isSplat()) {
    SDNode *V = N0.getNode();
    

    // If this is a bit convert that changes the element type of the vector but
    // not the number of vector elements, look through it.  Be careful not to
    // look though conversions that change things like v4f32 to v2f64.
    if (V->getOpcode() == ISD::BIT_CONVERT) {
      SDValue ConvInput = V->getOperand(0);
      if (ConvInput.getValueType().isVector() &&
          ConvInput.getValueType().getVectorNumElements() == NumElts)
        V = ConvInput.getNode();
    }

    if (V->getOpcode() == ISD::BUILD_VECTOR) {
      unsigned NumElems = V->getNumOperands();
      unsigned BaseIdx = cast<ShuffleVectorSDNode>(N)->getSplatIndex();
      if (NumElems > BaseIdx) {
        SDValue Base;
        bool AllSame = true;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i).getOpcode() != ISD::UNDEF) {
            Base = V->getOperand(i);
            break;
          }
        }
        // Splat of <u, u, u, u>, return <u, u, u, u>
        if (!Base.getNode())
          return N0;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i) != Base) {
            AllSame = false;
            break;
          }
        }
        // Splat of <x, x, x, x>, return <x, x, x, x>
        if (AllSame)
          return N0;
      }
    }
  }
  return SDValue();
}

/// XformToShuffleWithZero - Returns a vector_shuffle if it able to transform
/// an AND to a vector_shuffle with the destination vector and a zero vector.
/// e.g. AND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDValue DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (N->getOpcode() == ISD::AND) {
    if (RHS.getOpcode() == ISD::BIT_CONVERT)
      RHS = RHS.getOperand(0);
    if (RHS.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<int, 8> Indices;
      unsigned NumElts = RHS.getNumOperands();
      for (unsigned i = 0; i != NumElts; ++i) {
        SDValue Elt = RHS.getOperand(i);
        if (!isa<ConstantSDNode>(Elt))
          return SDValue();
        else if (cast<ConstantSDNode>(Elt)->isAllOnesValue())
          Indices.push_back(i);
        else if (cast<ConstantSDNode>(Elt)->isNullValue())
          Indices.push_back(NumElts);
        else
          return SDValue();
      }

      // Let's see if the target supports this vector_shuffle.
      EVT RVT = RHS.getValueType();
      if (!TLI.isVectorClearMaskLegal(Indices, RVT))
        return SDValue();

      // Return the new VECTOR_SHUFFLE node.
      EVT EltVT = RVT.getVectorElementType();
      SmallVector<SDValue,8> ZeroOps(RVT.getVectorNumElements(),
                                     DAG.getConstant(0, EltVT));
      SDValue Zero = DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(),
                                 RVT, &ZeroOps[0], ZeroOps.size());
      LHS = DAG.getNode(ISD::BIT_CONVERT, dl, RVT, LHS);
      SDValue Shuf = DAG.getVectorShuffle(RVT, dl, LHS, Zero, &Indices[0]);
      return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Shuf);
    }
  }

  return SDValue();
}

/// SimplifyVBinOp - Visit a binary vector operation, like ADD.
SDValue DAGCombiner::SimplifyVBinOp(SDNode *N) {
  // After legalize, the target may be depending on adds and other
  // binary ops to provide legal ways to construct constants or other
  // things. Simplifying them may result in a loss of legality.
  if (LegalOperations) return SDValue();

  EVT VT = N->getValueType(0);
  assert(VT.isVector() && "SimplifyVBinOp only works on vectors!");

  EVT EltType = VT.getVectorElementType();
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Shuffle = XformToShuffleWithZero(N);
  if (Shuffle.getNode()) return Shuffle;

  // If the LHS and RHS are BUILD_VECTOR nodes, see if we can constant fold
  // this operation.
  if (LHS.getOpcode() == ISD::BUILD_VECTOR &&
      RHS.getOpcode() == ISD::BUILD_VECTOR) {
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = LHS.getNumOperands(); i != e; ++i) {
      SDValue LHSOp = LHS.getOperand(i);
      SDValue RHSOp = RHS.getOperand(i);
      // If these two elements can't be folded, bail out.
      if ((LHSOp.getOpcode() != ISD::UNDEF &&
           LHSOp.getOpcode() != ISD::Constant &&
           LHSOp.getOpcode() != ISD::ConstantFP) ||
          (RHSOp.getOpcode() != ISD::UNDEF &&
           RHSOp.getOpcode() != ISD::Constant &&
           RHSOp.getOpcode() != ISD::ConstantFP))
        break;

      // Can't fold divide by zero.
      if (N->getOpcode() == ISD::SDIV || N->getOpcode() == ISD::UDIV ||
          N->getOpcode() == ISD::FDIV) {
        if ((RHSOp.getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(RHSOp.getNode())->isNullValue()) ||
            (RHSOp.getOpcode() == ISD::ConstantFP &&
             cast<ConstantFPSDNode>(RHSOp.getNode())->getValueAPF().isZero()))
          break;
      }

      Ops.push_back(DAG.getNode(N->getOpcode(), LHS.getDebugLoc(),
                                EltType, LHSOp, RHSOp));
      AddToWorkList(Ops.back().getNode());
      assert((Ops.back().getOpcode() == ISD::UNDEF ||
              Ops.back().getOpcode() == ISD::Constant ||
              Ops.back().getOpcode() == ISD::ConstantFP) &&
             "Scalar binop didn't fold!");
    }

    if (Ops.size() == LHS.getNumOperands()) {
      EVT VT = LHS.getValueType();
      return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(), VT,
                         &Ops[0], Ops.size());
    }
  }

  return SDValue();
}

SDValue DAGCombiner::SimplifySelect(DebugLoc DL, SDValue N0,
                                    SDValue N1, SDValue N2){
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");

  SDValue SCC = SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());

  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.getNode()) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      SDValue SETCC = DAG.getNode(ISD::SETCC, N0.getDebugLoc(),
                                  N0.getValueType(),
                                  SCC.getOperand(0), SCC.getOperand(1),
                                  SCC.getOperand(4));
      AddToWorkList(SETCC.getNode());
      return DAG.getNode(ISD::SELECT, SCC.getDebugLoc(), SCC.getValueType(),
                         SCC.getOperand(2), SCC.getOperand(3), SETCC);
    }

    return SCC;
  }
  return SDValue();
}

/// SimplifySelectOps - Given a SELECT or a SELECT_CC node, where LHS and RHS
/// are the two values being selected between, see if we can simplify the
/// select.  Callers of this should assume that TheSelect is deleted if this
/// returns true.  As such, they should return the appropriate thing (e.g. the
/// node) back to the top-level of the DAG combiner loop to avoid it being
/// looked at.
bool DAGCombiner::SimplifySelectOps(SDNode *TheSelect, SDValue LHS,
                                    SDValue RHS) {

  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() == RHS.getOpcode() && LHS.hasOneUse() && RHS.hasOneUse()){
    // If this is a load and the token chain is identical, replace the select
    // of two loads with a load through a select of the address to load from.
    // This triggers in things like "select bool X, 10.0, 123.0" after the FP
    // constants have been dropped into the constant pool.
    if (LHS.getOpcode() == ISD::LOAD &&
        // Do not let this transformation reduce the number of volatile loads.
        !cast<LoadSDNode>(LHS)->isVolatile() &&
        !cast<LoadSDNode>(RHS)->isVolatile() &&
        // Token chains must be identical.
        LHS.getOperand(0) == RHS.getOperand(0)) {
      LoadSDNode *LLD = cast<LoadSDNode>(LHS);
      LoadSDNode *RLD = cast<LoadSDNode>(RHS);

      // If this is an EXTLOAD, the VT's must match.
      if (LLD->getMemoryVT() == RLD->getMemoryVT()) {
        // FIXME: this discards src value information.  This is
        // over-conservative. It would be beneficial to be able to remember
        // both potential memory locations.  Since we are discarding
        // src value info, don't do the transformation if the memory
        // locations are not in the default address space.
        unsigned LLDAddrSpace = 0, RLDAddrSpace = 0;
        if (const Value *LLDVal = LLD->getMemOperand()->getValue()) {
          if (const PointerType *PT = dyn_cast<PointerType>(LLDVal->getType()))
            LLDAddrSpace = PT->getAddressSpace();
        }
        if (const Value *RLDVal = RLD->getMemOperand()->getValue()) {
          if (const PointerType *PT = dyn_cast<PointerType>(RLDVal->getType()))
            RLDAddrSpace = PT->getAddressSpace();
        }
        SDValue Addr;
        if (LLDAddrSpace == 0 && RLDAddrSpace == 0) {
          if (TheSelect->getOpcode() == ISD::SELECT) {
            // Check that the condition doesn't reach either load.  If so, folding
            // this will induce a cycle into the DAG.
            if ((!LLD->hasAnyUseOfValue(1) ||
                 !LLD->isPredecessorOf(TheSelect->getOperand(0).getNode())) &&
                (!RLD->hasAnyUseOfValue(1) ||
                 !RLD->isPredecessorOf(TheSelect->getOperand(0).getNode()))) {
              Addr = DAG.getNode(ISD::SELECT, TheSelect->getDebugLoc(),
                                 LLD->getBasePtr().getValueType(),
                                 TheSelect->getOperand(0), LLD->getBasePtr(),
                                 RLD->getBasePtr());
            }
          } else {
            // Check that the condition doesn't reach either load.  If so, folding
            // this will induce a cycle into the DAG.
            if ((!LLD->hasAnyUseOfValue(1) ||
                 (!LLD->isPredecessorOf(TheSelect->getOperand(0).getNode()) &&
                  !LLD->isPredecessorOf(TheSelect->getOperand(1).getNode()))) &&
                (!RLD->hasAnyUseOfValue(1) ||
                 (!RLD->isPredecessorOf(TheSelect->getOperand(0).getNode()) &&
                  !RLD->isPredecessorOf(TheSelect->getOperand(1).getNode())))) {
              Addr = DAG.getNode(ISD::SELECT_CC, TheSelect->getDebugLoc(),
                                 LLD->getBasePtr().getValueType(),
                                 TheSelect->getOperand(0),
                                 TheSelect->getOperand(1),
                                 LLD->getBasePtr(), RLD->getBasePtr(),
                                 TheSelect->getOperand(4));
            }
          }
        }

        if (Addr.getNode()) {
          SDValue Load;
          if (LLD->getExtensionType() == ISD::NON_EXTLOAD) {
            Load = DAG.getLoad(TheSelect->getValueType(0),
                               TheSelect->getDebugLoc(),
                               LLD->getChain(),
                               Addr, 0, 0,
                               LLD->isVolatile(),
                               LLD->isNonTemporal(),
                               LLD->getAlignment());
          } else {
            Load = DAG.getExtLoad(LLD->getExtensionType(),
                                  TheSelect->getDebugLoc(),
                                  TheSelect->getValueType(0),
                                  LLD->getChain(), Addr, 0, 0,
                                  LLD->getMemoryVT(),
                                  LLD->isVolatile(),
                                  LLD->isNonTemporal(),
                                  LLD->getAlignment());
          }

          // Users of the select now use the result of the load.
          CombineTo(TheSelect, Load);

          // Users of the old loads now use the new load's chain.  We know the
          // old-load value is dead now.
          CombineTo(LHS.getNode(), Load.getValue(0), Load.getValue(1));
          CombineTo(RHS.getNode(), Load.getValue(0), Load.getValue(1));
          return true;
        }
      }
    }
  }

  return false;
}

/// SimplifySelectCC - Simplify an expression of the form (N0 cond N1) ? N2 : N3
/// where 'cond' is the comparison specified by CC.
SDValue DAGCombiner::SimplifySelectCC(DebugLoc DL, SDValue N0, SDValue N1,
                                      SDValue N2, SDValue N3,
                                      ISD::CondCode CC, bool NotExtCompare) {
  // (x ? y : y) -> y.
  if (N2 == N3) return N2;
  
  EVT VT = N2.getValueType();
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.getNode());

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(TLI.getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, DL, false);
  if (SCC.getNode()) AddToWorkList(SCC.getNode());
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.getNode());

  // fold select_cc true, x, y -> x
  if (SCCC && !SCCC->isNullValue())
    return N2;
  // fold select_cc false, x, y -> y
  if (SCCC && SCCC->isNullValue())
    return N3;

  // Check to see if we can simplify the select into an fabs node
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1)) {
    // Allow either -0.0 or 0.0
    if (CFP->getValueAPF().isZero()) {
      // select (setg[te] X, +/-0.0), X, fneg(X) -> fabs
      if ((CC == ISD::SETGE || CC == ISD::SETGT) &&
          N0 == N2 && N3.getOpcode() == ISD::FNEG &&
          N2 == N3.getOperand(0))
        return DAG.getNode(ISD::FABS, DL, VT, N0);

      // select (setl[te] X, +/-0.0), fneg(X), X -> fabs
      if ((CC == ISD::SETLT || CC == ISD::SETLE) &&
          N0 == N3 && N2.getOpcode() == ISD::FNEG &&
          N2.getOperand(0) == N3)
        return DAG.getNode(ISD::FABS, DL, VT, N3);
    }
  }
  
  // Turn "(a cond b) ? 1.0f : 2.0f" into "load (tmp + ((a cond b) ? 0 : 4)"
  // where "tmp" is a constant pool entry containing an array with 1.0 and 2.0
  // in it.  This is a win when the constant is not otherwise available because
  // it replaces two constant pool loads with one.  We only do this if the FP
  // type is known to be legal, because if it isn't, then we are before legalize
  // types an we want the other legalization to happen first (e.g. to avoid
  // messing with soft float) and if the ConstantFP is not legal, because if
  // it is legal, we may not need to store the FP constant in a constant pool.
  if (ConstantFPSDNode *TV = dyn_cast<ConstantFPSDNode>(N2))
    if (ConstantFPSDNode *FV = dyn_cast<ConstantFPSDNode>(N3)) {
      if (TLI.isTypeLegal(N2.getValueType()) &&
          (TLI.getOperationAction(ISD::ConstantFP, N2.getValueType()) !=
           TargetLowering::Legal) &&
          // If both constants have multiple uses, then we won't need to do an
          // extra load, they are likely around in registers for other users.
          (TV->hasOneUse() || FV->hasOneUse())) {
        Constant *Elts[] = {
          const_cast<ConstantFP*>(FV->getConstantFPValue()),
          const_cast<ConstantFP*>(TV->getConstantFPValue())
        };
        const Type *FPTy = Elts[0]->getType();
        const TargetData &TD = *TLI.getTargetData();
        
        // Create a ConstantArray of the two constants.
        Constant *CA = ConstantArray::get(ArrayType::get(FPTy, 2), Elts, 2);
        SDValue CPIdx = DAG.getConstantPool(CA, TLI.getPointerTy(),
                                            TD.getPrefTypeAlignment(FPTy));
        unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();

        // Get the offsets to the 0 and 1 element of the array so that we can
        // select between them.
        SDValue Zero = DAG.getIntPtrConstant(0);
        unsigned EltSize = (unsigned)TD.getTypeAllocSize(Elts[0]->getType());
        SDValue One = DAG.getIntPtrConstant(EltSize);
        
        SDValue Cond = DAG.getSetCC(DL,
                                    TLI.getSetCCResultType(N0.getValueType()),
                                    N0, N1, CC);
        SDValue CstOffset = DAG.getNode(ISD::SELECT, DL, Zero.getValueType(),
                                        Cond, One, Zero);
        CPIdx = DAG.getNode(ISD::ADD, DL, TLI.getPointerTy(), CPIdx,
                            CstOffset);
        return DAG.getLoad(TV->getValueType(0), DL, DAG.getEntryNode(), CPIdx,
                           PseudoSourceValue::getConstantPool(), 0, false,
                           false, Alignment);

      }
    }  

  // Check to see if we can perform the "gzip trick", transforming
  // (select_cc setlt X, 0, A, 0) -> (and (sra X, (sub size(X), 1), A)
  if (N1C && N3C && N3C->isNullValue() && CC == ISD::SETLT &&
      N0.getValueType().isInteger() &&
      N2.getValueType().isInteger() &&
      (N1C->isNullValue() ||                         // (a < 0) ? b : 0
       (N1C->getAPIntValue() == 1 && N0 == N2))) {   // (a < 1) ? a : 0
    EVT XType = N0.getValueType();
    EVT AType = N2.getValueType();
    if (XType.bitsGE(AType)) {
      // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
      // single-bit constant.
      if (N2C && ((N2C->getAPIntValue() & (N2C->getAPIntValue()-1)) == 0)) {
        unsigned ShCtV = N2C->getAPIntValue().logBase2();
        ShCtV = XType.getSizeInBits()-ShCtV-1;
        SDValue ShCt = DAG.getConstant(ShCtV, getShiftAmountTy());
        SDValue Shift = DAG.getNode(ISD::SRL, N0.getDebugLoc(),
                                    XType, N0, ShCt);
        AddToWorkList(Shift.getNode());

        if (XType.bitsGT(AType)) {
          Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
          AddToWorkList(Shift.getNode());
        }

        return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
      }

      SDValue Shift = DAG.getNode(ISD::SRA, N0.getDebugLoc(),
                                  XType, N0,
                                  DAG.getConstant(XType.getSizeInBits()-1,
                                                  getShiftAmountTy()));
      AddToWorkList(Shift.getNode());

      if (XType.bitsGT(AType)) {
        Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
        AddToWorkList(Shift.getNode());
      }

      return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
    }
  }

  // fold select C, 16, 0 -> shl C, 4
  if (N2C && N3C && N3C->isNullValue() && N2C->getAPIntValue().isPowerOf2() &&
      TLI.getBooleanContents() == TargetLowering::ZeroOrOneBooleanContent) {

    // If the caller doesn't want us to simplify this into a zext of a compare,
    // don't do it.
    if (NotExtCompare && N2C->getAPIntValue() == 1)
      return SDValue();

    // Get a SetCC of the condition
    // FIXME: Should probably make sure that setcc is legal if we ever have a
    // target where it isn't.
    SDValue Temp, SCC;
    // cast from setcc result type to select result type
    if (LegalTypes) {
      SCC  = DAG.getSetCC(DL, TLI.getSetCCResultType(N0.getValueType()),
                          N0, N1, CC);
      if (N2.getValueType().bitsLT(SCC.getValueType()))
        Temp = DAG.getZeroExtendInReg(SCC, N2.getDebugLoc(), N2.getValueType());
      else
        Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getDebugLoc(),
                           N2.getValueType(), SCC);
    } else {
      SCC  = DAG.getSetCC(N0.getDebugLoc(), MVT::i1, N0, N1, CC);
      Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getDebugLoc(),
                         N2.getValueType(), SCC);
    }

    AddToWorkList(SCC.getNode());
    AddToWorkList(Temp.getNode());

    if (N2C->getAPIntValue() == 1)
      return Temp;

    // shl setcc result by log2 n2c
    return DAG.getNode(ISD::SHL, DL, N2.getValueType(), Temp,
                       DAG.getConstant(N2C->getAPIntValue().logBase2(),
                                       getShiftAmountTy()));
  }

  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && N3C && N3C->isNullValue() && N2C && (N2C->getAPIntValue() == 1ULL)) {
    EVT XType = N0.getValueType();
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC, TLI.getSetCCResultType(XType))) {
      SDValue Res = DAG.getSetCC(DL, TLI.getSetCCResultType(XType), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Res);
      return Res;
    }

    // fold (seteq X, 0) -> (srl (ctlz X, log2(size(X))))
    if (N1C && N1C->isNullValue() && CC == ISD::SETEQ &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::CTLZ, XType))) {
      SDValue Ctlz = DAG.getNode(ISD::CTLZ, N0.getDebugLoc(), XType, N0);
      return DAG.getNode(ISD::SRL, DL, XType, Ctlz,
                         DAG.getConstant(Log2_32(XType.getSizeInBits()),
                                         getShiftAmountTy()));
    }
    // fold (setgt X, 0) -> (srl (and (-X, ~X), size(X)-1))
    if (N1C && N1C->isNullValue() && CC == ISD::SETGT) {
      SDValue NegN0 = DAG.getNode(ISD::SUB, N0.getDebugLoc(),
                                  XType, DAG.getConstant(0, XType), N0);
      SDValue NotN0 = DAG.getNOT(N0.getDebugLoc(), N0, XType);
      return DAG.getNode(ISD::SRL, DL, XType,
                         DAG.getNode(ISD::AND, DL, XType, NegN0, NotN0),
                         DAG.getConstant(XType.getSizeInBits()-1,
                                         getShiftAmountTy()));
    }
    // fold (setgt X, -1) -> (xor (srl (X, size(X)-1), 1))
    if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT) {
      SDValue Sign = DAG.getNode(ISD::SRL, N0.getDebugLoc(), XType, N0,
                                 DAG.getConstant(XType.getSizeInBits()-1,
                                                 getShiftAmountTy()));
      return DAG.getNode(ISD::XOR, DL, XType, Sign, DAG.getConstant(1, XType));
    }
  }

  // Check to see if this is an integer abs. select_cc setl[te] X, 0, -X, X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE) &&
      N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1) &&
      N2.getOperand(0) == N1 && N0.getValueType().isInteger()) {
    EVT XType = N0.getValueType();
    SDValue Shift = DAG.getNode(ISD::SRA, N0.getDebugLoc(), XType, N0,
                                DAG.getConstant(XType.getSizeInBits()-1,
                                                getShiftAmountTy()));
    SDValue Add = DAG.getNode(ISD::ADD, N0.getDebugLoc(), XType,
                              N0, Shift);
    AddToWorkList(Shift.getNode());
    AddToWorkList(Add.getNode());
    return DAG.getNode(ISD::XOR, DL, XType, Add, Shift);
  }
  // Check to see if this is an integer abs. select_cc setgt X, -1, X, -X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT &&
      N0 == N2 && N3.getOpcode() == ISD::SUB && N0 == N3.getOperand(1)) {
    if (ConstantSDNode *SubC = dyn_cast<ConstantSDNode>(N3.getOperand(0))) {
      EVT XType = N0.getValueType();
      if (SubC->isNullValue() && XType.isInteger()) {
        SDValue Shift = DAG.getNode(ISD::SRA, N0.getDebugLoc(), XType,
                                    N0,
                                    DAG.getConstant(XType.getSizeInBits()-1,
                                                    getShiftAmountTy()));
        SDValue Add = DAG.getNode(ISD::ADD, N0.getDebugLoc(),
                                  XType, N0, Shift);
        AddToWorkList(Shift.getNode());
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::XOR, DL, XType, Add, Shift);
      }
    }
  }

  return SDValue();
}

/// SimplifySetCC - This is a stub for TargetLowering::SimplifySetCC.
SDValue DAGCombiner::SimplifySetCC(EVT VT, SDValue N0,
                                   SDValue N1, ISD::CondCode Cond,
                                   DebugLoc DL, bool foldBooleans) {
  TargetLowering::DAGCombinerInfo
    DagCombineInfo(DAG, !LegalTypes, !LegalOperations, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo, DL);
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDValue DAGCombiner::BuildSDIV(SDNode *N) {
  std::vector<SDNode*> Built;
  SDValue S = TLI.BuildSDIV(N, DAG, &Built);

  for (std::vector<SDNode*>::iterator ii = Built.begin(), ee = Built.end();
       ii != ee; ++ii)
    AddToWorkList(*ii);
  return S;
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDValue DAGCombiner::BuildUDIV(SDNode *N) {
  std::vector<SDNode*> Built;
  SDValue S = TLI.BuildUDIV(N, DAG, &Built);

  for (std::vector<SDNode*>::iterator ii = Built.begin(), ee = Built.end();
       ii != ee; ++ii)
    AddToWorkList(*ii);
  return S;
}

/// FindBaseOffset - Return true if base is a frame index, which is known not
// to alias with anything but itself.  Provides base object and offset as results.
static bool FindBaseOffset(SDValue Ptr, SDValue &Base, int64_t &Offset,
                           GlobalValue *&GV, void *&CV) {
  // Assume it is a primitive operation.
  Base = Ptr; Offset = 0; GV = 0; CV = 0;

  // If it's an adding a simple constant then integrate the offset.
  if (Base.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Base.getOperand(1))) {
      Base = Base.getOperand(0);
      Offset += C->getZExtValue();
    }
  }
  
  // Return the underlying GlobalValue, and update the Offset.  Return false
  // for GlobalAddressSDNode since the same GlobalAddress may be represented
  // by multiple nodes with different offsets.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Base)) {
    GV = G->getGlobal();
    Offset += G->getOffset();
    return false;
  }

  // Return the underlying Constant value, and update the Offset.  Return false
  // for ConstantSDNodes since the same constant pool entry may be represented
  // by multiple nodes with different offsets.
  if (ConstantPoolSDNode *C = dyn_cast<ConstantPoolSDNode>(Base)) {
    CV = C->isMachineConstantPoolEntry() ? (void *)C->getMachineCPVal()
                                         : (void *)C->getConstVal();
    Offset += C->getOffset();
    return false;
  }
  // If it's any of the following then it can't alias with anything but itself.
  return isa<FrameIndexSDNode>(Base);
}

/// isAlias - Return true if there is any possibility that the two addresses
/// overlap.
bool DAGCombiner::isAlias(SDValue Ptr1, int64_t Size1,
                          const Value *SrcValue1, int SrcValueOffset1,
                          unsigned SrcValueAlign1,
                          SDValue Ptr2, int64_t Size2,
                          const Value *SrcValue2, int SrcValueOffset2,
                          unsigned SrcValueAlign2) const {
  // If they are the same then they must be aliases.
  if (Ptr1 == Ptr2) return true;

  // Gather base node and offset information.
  SDValue Base1, Base2;
  int64_t Offset1, Offset2;
  GlobalValue *GV1, *GV2;
  void *CV1, *CV2;
  bool isFrameIndex1 = FindBaseOffset(Ptr1, Base1, Offset1, GV1, CV1);
  bool isFrameIndex2 = FindBaseOffset(Ptr2, Base2, Offset2, GV2, CV2);

  // If they have a same base address then check to see if they overlap.
  if (Base1 == Base2 || (GV1 && (GV1 == GV2)) || (CV1 && (CV1 == CV2)))
    return !((Offset1 + Size1) <= Offset2 || (Offset2 + Size2) <= Offset1);

  // If we know what the bases are, and they aren't identical, then we know they
  // cannot alias.
  if ((isFrameIndex1 || CV1 || GV1) && (isFrameIndex2 || CV2 || GV2))
    return false;

  // If we know required SrcValue1 and SrcValue2 have relatively large alignment
  // compared to the size and offset of the access, we may be able to prove they
  // do not alias.  This check is conservative for now to catch cases created by
  // splitting vector types.
  if ((SrcValueAlign1 == SrcValueAlign2) &&
      (SrcValueOffset1 != SrcValueOffset2) &&
      (Size1 == Size2) && (SrcValueAlign1 > Size1)) {
    int64_t OffAlign1 = SrcValueOffset1 % SrcValueAlign1;
    int64_t OffAlign2 = SrcValueOffset2 % SrcValueAlign1;
    
    // There is no overlap between these relatively aligned accesses of similar
    // size, return no alias.
    if ((OffAlign1 + Size1) <= OffAlign2 || (OffAlign2 + Size2) <= OffAlign1)
      return false;
  }
  
  if (CombinerGlobalAA) {
    // Use alias analysis information.
    int64_t MinOffset = std::min(SrcValueOffset1, SrcValueOffset2);
    int64_t Overlap1 = Size1 + SrcValueOffset1 - MinOffset;
    int64_t Overlap2 = Size2 + SrcValueOffset2 - MinOffset;
    AliasAnalysis::AliasResult AAResult =
                             AA.alias(SrcValue1, Overlap1, SrcValue2, Overlap2);
    if (AAResult == AliasAnalysis::NoAlias)
      return false;
  }

  // Otherwise we have to assume they alias.
  return true;
}

/// FindAliasInfo - Extracts the relevant alias information from the memory
/// node.  Returns true if the operand was a load.
bool DAGCombiner::FindAliasInfo(SDNode *N,
                        SDValue &Ptr, int64_t &Size,
                        const Value *&SrcValue, 
                        int &SrcValueOffset,
                        unsigned &SrcValueAlign) const {
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    Size = LD->getMemoryVT().getSizeInBits() >> 3;
    SrcValue = LD->getSrcValue();
    SrcValueOffset = LD->getSrcValueOffset();
    SrcValueAlign = LD->getOriginalAlignment();
    return true;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    Size = ST->getMemoryVT().getSizeInBits() >> 3;
    SrcValue = ST->getSrcValue();
    SrcValueOffset = ST->getSrcValueOffset();
    SrcValueAlign = ST->getOriginalAlignment();
  } else {
    llvm_unreachable("FindAliasInfo expected a memory operand");
  }

  return false;
}

/// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDValue OriginalChain,
                                   SmallVector<SDValue, 8> &Aliases) {
  SmallVector<SDValue, 8> Chains;     // List of chains to visit.
  SmallPtrSet<SDNode *, 16> Visited;  // Visited node set.

  // Get alias information for node.
  SDValue Ptr;
  int64_t Size;
  const Value *SrcValue;
  int SrcValueOffset;
  unsigned SrcValueAlign;
  bool IsLoad = FindAliasInfo(N, Ptr, Size, SrcValue, SrcValueOffset, 
                              SrcValueAlign);

  // Starting off.
  Chains.push_back(OriginalChain);
  unsigned Depth = 0;
  
  // Look at each chain and determine if it is an alias.  If so, add it to the
  // aliases list.  If not, then continue up the chain looking for the next
  // candidate.
  while (!Chains.empty()) {
    SDValue Chain = Chains.back();
    Chains.pop_back();
    
    // For TokenFactor nodes, look at each operand and only continue up the 
    // chain until we find two aliases.  If we've seen two aliases, assume we'll 
    // find more and revert to original chain since the xform is unlikely to be
    // profitable.
    // 
    // FIXME: The depth check could be made to return the last non-aliasing 
    // chain we found before we hit a tokenfactor rather than the original
    // chain.
    if (Depth > 6 || Aliases.size() == 2) {
      Aliases.clear();
      Aliases.push_back(OriginalChain);
      break;
    }

    // Don't bother if we've been before.
    if (!Visited.insert(Chain.getNode()))
      continue;

    switch (Chain.getOpcode()) {
    case ISD::EntryToken:
      // Entry token is ideal chain operand, but handled in FindBetterChain.
      break;

    case ISD::LOAD:
    case ISD::STORE: {
      // Get alias information for Chain.
      SDValue OpPtr;
      int64_t OpSize;
      const Value *OpSrcValue;
      int OpSrcValueOffset;
      unsigned OpSrcValueAlign;
      bool IsOpLoad = FindAliasInfo(Chain.getNode(), OpPtr, OpSize,
                                    OpSrcValue, OpSrcValueOffset,
                                    OpSrcValueAlign);

      // If chain is alias then stop here.
      if (!(IsLoad && IsOpLoad) &&
          isAlias(Ptr, Size, SrcValue, SrcValueOffset, SrcValueAlign,
                  OpPtr, OpSize, OpSrcValue, OpSrcValueOffset,
                  OpSrcValueAlign)) {
        Aliases.push_back(Chain);
      } else {
        // Look further up the chain.
        Chains.push_back(Chain.getOperand(0));
        ++Depth;
      }
      break;
    }

    case ISD::TokenFactor:
      // We have to check each of the operands of the token factor for "small"
      // token factors, so we queue them up.  Adding the operands to the queue
      // (stack) in reverse order maintains the original order and increases the
      // likelihood that getNode will find a matching token factor (CSE.)
      if (Chain.getNumOperands() > 16) {
        Aliases.push_back(Chain);
        break;
      }
      for (unsigned n = Chain.getNumOperands(); n;)
        Chains.push_back(Chain.getOperand(--n));
      ++Depth;
      break;

    default:
      // For all other instructions we will just have to take what we can get.
      Aliases.push_back(Chain);
      break;
    }
  }
}

/// FindBetterChain - Walk up chain skipping non-aliasing memory nodes, looking
/// for a better chain (aliasing node.)
SDValue DAGCombiner::FindBetterChain(SDNode *N, SDValue OldChain) {
  SmallVector<SDValue, 8> Aliases;  // Ops for replacing token factor.

  // Accumulate all the aliases to this node.
  GatherAllAliases(N, OldChain, Aliases);

  if (Aliases.size() == 0) {
    // If no operands then chain to entry token.
    return DAG.getEntryNode();
  } else if (Aliases.size() == 1) {
    // If a single operand then chain to it.  We don't need to revisit it.
    return Aliases[0];
  }
  
  // Construct a custom tailored token factor.
  return DAG.getNode(ISD::TokenFactor, N->getDebugLoc(), MVT::Other, 
                     &Aliases[0], Aliases.size());
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(CombineLevel Level, AliasAnalysis &AA,
                           CodeGenOpt::Level OptLevel) {
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this, AA, OptLevel).Run(Level);
}
