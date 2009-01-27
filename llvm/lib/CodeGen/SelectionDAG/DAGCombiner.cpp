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
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");

namespace {
  static cl::opt<bool>
    CombinerAA("combiner-alias-analysis", cl::Hidden,
               cl::desc("Turn on alias analysis during testing"));

  static cl::opt<bool>
    CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
               cl::desc("Include global information in alias analysis"));

//------------------------------ DAGCombiner ---------------------------------//

  class VISIBILITY_HIDDEN DAGCombiner {
    SelectionDAG &DAG;
    const TargetLowering &TLI;
    CombineLevel Level;
    bool LegalOperations;
    bool LegalTypes;
    bool Fast;

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
    
  private:    
    
    /// SimplifyDemandedBits - Check the specified integer node value to see if
    /// it can be simplified or if things it uses can be simplified by bit
    /// propagation.  If so, return true.
    bool SimplifyDemandedBits(SDValue Op) {
      APInt Demanded = APInt::getAllOnesValue(Op.getValueSizeInBits());
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
    SDValue ReassociateOps(unsigned Opc, SDValue LHS, SDValue RHS);
    
    SDValue visitShiftByConstant(SDNode *N, unsigned Amt);

    bool SimplifySelectOps(SDNode *SELECT, SDValue LHS, SDValue RHS);
    SDValue SimplifyBinOpWithSameOpcodeHands(SDNode *N);
    SDValue SimplifySelect(SDValue N0, SDValue N1, SDValue N2);
    SDValue SimplifySelectCC(SDValue N0, SDValue N1, SDValue N2, 
                               SDValue N3, ISD::CondCode CC, 
                               bool NotExtCompare = false);
    SDValue SimplifySetCC(MVT VT, SDValue N0, SDValue N1, ISD::CondCode Cond,
                          bool foldBooleans = true);
    SDValue SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp, 
                                         unsigned HiOp);
    SDValue CombineConsecutiveLoads(SDNode *N, MVT VT);
    SDValue ConstantFoldBIT_CONVERTofBUILD_VECTOR(SDNode *, MVT);
    SDValue BuildSDIV(SDNode *N);
    SDValue BuildUDIV(SDNode *N);
    SDNode *MatchRotate(SDValue LHS, SDValue RHS);
    SDValue ReduceLoadWidth(SDNode *N);
    
    SDValue GetDemandedBits(SDValue V, const APInt &Mask);
    
    /// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDValue OriginalChain,
                          SmallVector<SDValue, 8> &Aliases);

    /// isAlias - Return true if there is any possibility that the two addresses
    /// overlap.
    bool isAlias(SDValue Ptr1, int64_t Size1,
                 const Value *SrcValue1, int SrcValueOffset1,
                 SDValue Ptr2, int64_t Size2,
                 const Value *SrcValue2, int SrcValueOffset2);
                 
    /// FindAliasInfo - Extracts the relevant alias information from the memory
    /// node.  Returns true if the operand was a load.
    bool FindAliasInfo(SDNode *N,
                       SDValue &Ptr, int64_t &Size,
                       const Value *&SrcValue, int &SrcValueOffset);
                       
    /// FindBetterChain - Walk up chain skipping non-aliasing memory nodes,
    /// looking for a better chain (aliasing node.)
    SDValue FindBetterChain(SDNode *N, SDValue Chain);
    
public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis &A, bool fast)
      : DAG(D),
        TLI(D.getTargetLoweringInfo()),
        Level(Unrestricted),
        LegalOperations(false),
        LegalTypes(false),
        Fast(fast),
        AA(A) {}
    
    /// Run - runs the dag combiner on all nodes in the work list
    void Run(CombineLevel AtLevel);
  };
}


namespace {
/// WorkListRemover - This class is a DAGUpdateListener that removes any deleted
/// nodes from the worklist.
class VISIBILITY_HIDDEN WorkListRemover : 
  public SelectionDAG::DAGUpdateListener {
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
CombineTo(SDNode *N, const std::vector<SDValue> &To) {
  return ((DAGCombiner*)DC)->CombineTo(N, &To[0], To.size());
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res);
}


SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res0, SDValue Res1) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res0, Res1);
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
    
    // -(A+B) -> -A - B
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return V;
    // -(A+B) -> -B - A
    return isNegatibleForFree(Op.getOperand(1), LegalOperations, Depth+1);
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros. 
    if (!UnsafeFPMath) return 0;
    
    // -(A-B) -> B-A
    return 1;
    
  case ISD::FMUL:
  case ISD::FDIV:
    if (HonorSignDependentRoundingFPMath()) return 0;
    
    // -(X*Y) -> (-X * Y) or (X*-Y)
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
  default: assert(0 && "Unknown code");
  case ISD::ConstantFP: {
    APFloat V = cast<ConstantFPSDNode>(Op)->getValueAPF();
    V.changeSign();
    return DAG.getConstantFP(V, Op.getValueType());
  }
  case ISD::FADD:
    // FIXME: determine better conditions for this xform.
    assert(UnsafeFPMath);
    
    // -(A+B) -> -A - B
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return DAG.getNode(ISD::FSUB, Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG, 
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
    // -(A+B) -> -B - A
    return DAG.getNode(ISD::FSUB, Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(1), DAG, 
                                            LegalOperations, Depth+1),
                       Op.getOperand(0));
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros. 
    assert(UnsafeFPMath);

    // -(0-B) -> B
    if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(Op.getOperand(0)))
      if (N0CFP->getValueAPF().isZero())
        return Op.getOperand(1);
    
    // -(A-B) -> B-A
    return DAG.getNode(ISD::FSUB, Op.getValueType(), Op.getOperand(1),
                       Op.getOperand(0));
    
  case ISD::FMUL:
  case ISD::FDIV:
    assert(!HonorSignDependentRoundingFPMath());
    
    // -(X*Y) -> -X * Y
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations, Depth+1))
      return DAG.getNode(Op.getOpcode(), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG, 
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
      
    // -(X*Y) -> X * -Y
    return DAG.getNode(Op.getOpcode(), Op.getValueType(),
                       Op.getOperand(0),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1));
    
  case ISD::FP_EXTEND:
  case ISD::FSIN:
    return DAG.getNode(Op.getOpcode(), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(0), DAG, 
                                            LegalOperations, Depth+1));
  case ISD::FP_ROUND:
      return DAG.getNode(ISD::FP_ROUND, Op.getValueType(),
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

SDValue DAGCombiner::ReassociateOps(unsigned Opc, SDValue N0, SDValue N1){
  MVT VT = N0.getValueType();
  // reassoc. (op (op x, c1), y) -> (op (op x, y), c1) iff x+c1 has one use
  // reassoc. (op (op x, c1), c2) -> (op x, (op c1, c2))
  if (N0.getOpcode() == Opc && isa<ConstantSDNode>(N0.getOperand(1))) {
    if (isa<ConstantSDNode>(N1)) {
      SDValue OpNode = DAG.getNode(Opc, VT, N0.getOperand(1), N1);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, VT, OpNode, N0.getOperand(0));
    } else if (N0.hasOneUse()) {
      SDValue OpNode = DAG.getNode(Opc, VT, N0.getOperand(0), N1);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, VT, OpNode, N0.getOperand(1));
    }
  }
  // reassoc. (op y, (op x, c1)) -> (op (op x, y), c1) iff x+c1 has one use
  // reassoc. (op c2, (op x, c1)) -> (op x, (op c1, c2))
  if (N1.getOpcode() == Opc && isa<ConstantSDNode>(N1.getOperand(1))) {
    if (isa<ConstantSDNode>(N0)) {
      SDValue OpNode = DAG.getNode(Opc, VT, N1.getOperand(1), N0);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, VT, OpNode, N1.getOperand(0));
    } else if (N1.hasOneUse()) {
      SDValue OpNode = DAG.getNode(Opc, VT, N1.getOperand(0), N0);
      AddToWorkList(OpNode.getNode());
      return DAG.getNode(Opc, VT, OpNode, N1.getOperand(1));
    }
  }
  return SDValue();
}

SDValue DAGCombiner::CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                               bool AddTo) {
  assert(N->getNumValues() == NumTo && "Broken CombineTo call!");
  ++NodesCombined;
  DOUT << "\nReplacing.1 "; DEBUG(N->dump(&DAG));
  DOUT << "\nWith: "; DEBUG(To[0].getNode()->dump(&DAG));
  DOUT << " and " << NumTo-1 << " other values\n";
  DEBUG(for (unsigned i = 0, e = NumTo; i != e; ++i)
          assert(N->getValueType(i) == To[i].getValueType() &&
                 "Cannot combine value to value of different type!"));
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesWith(N, To, &DeadNodes);
  
  if (AddTo) {
    // Push the new nodes and any users onto the worklist
    for (unsigned i = 0, e = NumTo; i != e; ++i) {
      AddToWorkList(To[i].getNode());
      AddUsersToWorkList(To[i].getNode());
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
  DOUT << "\nReplacing.2 "; DEBUG(TLO.Old.getNode()->dump(&DAG));
  DOUT << "\nWith: "; DEBUG(TLO.New.getNode()->dump(&DAG));
  DOUT << '\n';
  
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

    DOUT << "\nReplacing.3 "; DEBUG(N->dump(&DAG));
    DOUT << "\nWith: "; DEBUG(RV.getNode()->dump(&DAG));
    DOUT << '\n';
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
        DagCombineInfo(DAG, Level == Unrestricted, false, this);

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
  return SDValue(0, 0);
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
        if ((CombinerAA || Op.hasOneUse()) &&
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
      Result = DAG.getNode(ISD::TokenFactor, MVT::Other, &Ops[0], Ops.size());
    }
    
    // Don't add users to work list.
    return CombineTo(N, Result, false);
  }
  
  return Result;
}

/// MERGE_VALUES can always be eliminated.
SDValue DAGCombiner::visitMERGE_VALUES(SDNode *N) {
  WorkListRemover DeadNodes(*this);
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, i), N->getOperand(i),
                                  &DeadNodes);
  removeFromWorkList(N);
  DAG.DeleteNode(N);
  return SDValue(N, 0);   // Return N so it doesn't get rechecked!
}


static
SDValue combineShlAddConstant(SDValue N0, SDValue N1, SelectionDAG &DAG) {
  MVT VT = N0.getValueType();
  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);
  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N01);
  if (N01C && N00.getOpcode() == ISD::ADD && N00.getNode()->hasOneUse() &&
      isa<ConstantSDNode>(N00.getOperand(1))) {
    N0 = DAG.getNode(ISD::ADD, VT,
                     DAG.getNode(ISD::SHL, VT, N00.getOperand(0), N01),
                     DAG.getNode(ISD::SHL, VT, N00.getOperand(1), N01));
    return DAG.getNode(ISD::ADD, VT, N0, N1);
  }
  return SDValue();
}

static
SDValue combineSelectAndUse(SDNode *N, SDValue Slct, SDValue OtherOp,
                            SelectionDAG &DAG, const TargetLowering &TLI,
                            bool LegalOperations) {
  MVT VT = N->getValueType(0);
  unsigned Opc = N->getOpcode();
  bool isSlctCC = Slct.getOpcode() == ISD::SELECT_CC;
  SDValue LHS = isSlctCC ? Slct.getOperand(2) : Slct.getOperand(1);
  SDValue RHS = isSlctCC ? Slct.getOperand(3) : Slct.getOperand(2);
  ISD::CondCode CC = ISD::SETCC_INVALID;

  if (isSlctCC) {
    CC = cast<CondCodeSDNode>(Slct.getOperand(4))->get();
  } else {
    SDValue CCOp = Slct.getOperand(0);
    if (CCOp.getOpcode() == ISD::SETCC)
      CC = cast<CondCodeSDNode>(CCOp.getOperand(2))->get();
  }

  bool DoXform = false;
  bool InvCC = false;
  assert ((Opc == ISD::ADD || (Opc == ISD::SUB && Slct == N->getOperand(1))) &&
          "Bad input!");

  if (LHS.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(LHS)->isNullValue()) {
    DoXform = true;
  } else if (CC != ISD::SETCC_INVALID &&
             RHS.getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(RHS)->isNullValue()) {
    std::swap(LHS, RHS);
    SDValue Op0 = Slct.getOperand(0);
    MVT OpVT = isSlctCC ? Op0.getValueType() :
                          Op0.getOperand(0).getValueType();
    bool isInt = OpVT.isInteger();
    CC = ISD::getSetCCInverse(CC, isInt);

    if (LegalOperations && !TLI.isCondCodeLegal(CC, OpVT))
      return SDValue();         // Inverse operator isn't legal.

    DoXform = true;
    InvCC = true;
  }

  if (DoXform) {
    SDValue Result = DAG.getNode(Opc, VT, OtherOp, RHS);
    if (isSlctCC)
      return DAG.getSelectCC(OtherOp, Result,
                             Slct.getOperand(0), Slct.getOperand(1), CC);
    SDValue CCOp = Slct.getOperand(0);
    if (InvCC)
      CCOp = DAG.getSetCC(CCOp.getValueType(), CCOp.getOperand(0),
                          CCOp.getOperand(1), CC);
    return DAG.getNode(ISD::SELECT, VT, CCOp, OtherOp, Result);
  }
  return SDValue();
}

SDValue DAGCombiner::visitADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N0.getValueType();

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
    return DAG.getNode(ISD::ADD, VT, N1, N0);
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
      return DAG.getNode(ISD::SUB, VT,
                         DAG.getConstant(N1C->getAPIntValue()+
                                         N0C->getAPIntValue(), VT),
                         N0.getOperand(1));
  // reassociate add
  SDValue RADD = ReassociateOps(ISD::ADD, N0, N1);
  if (RADD.getNode() != 0)
    return RADD;
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);
  // fold ((B-A)+A) -> B
  if (N0.getOpcode() == ISD::SUB && N1 == N0.getOperand(1))
    return N0.getOperand(0);
  // fold (A+(B-(A+C))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(0)) {
    return DAG.getNode(ISD::SUB, VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(1));
  }
  // fold (A+(B-(C+A))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(1)) {
    return DAG.getNode(ISD::SUB, VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(0));
  }
  // fold (A+((B-A)+or-C)) to (B+or-C)
  if ((N1.getOpcode() == ISD::SUB || N1.getOpcode() == ISD::ADD) &&
      N1.getOperand(0).getOpcode() == ISD::SUB &&
      N0 == N1.getOperand(0).getOperand(1)) {
    return DAG.getNode(N1.getOpcode(), VT, N1.getOperand(0).getOperand(0),
                       N1.getOperand(1));
  }

  // fold (A-B)+(C-D) to (A+C)-(B+D) when A or C is constant
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB) {
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    SDValue N10 = N1.getOperand(0);
    SDValue N11 = N1.getOperand(1);
    if (isa<ConstantSDNode>(N00) || isa<ConstantSDNode>(N10)) {
      return DAG.getNode(ISD::SUB, VT,
                 DAG.getNode(ISD::ADD, VT, N00, N10),
                 DAG.getNode(ISD::ADD, VT, N01, N11));
    }
  }

  if (!VT.isVector() && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);
  
  // fold (a+b) -> (a|b) iff a and b share no bits.
  if (VT.isInteger() && !VT.isVector()) {
    APInt LHSZero, LHSOne;
    APInt RHSZero, RHSOne;
    APInt Mask = APInt::getAllOnesValue(VT.getSizeInBits());
    DAG.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);
    if (LHSZero.getBoolValue()) {
      DAG.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);
      
      // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
      // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
      if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
          (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
        return DAG.getNode(ISD::OR, VT, N0, N1);
    }
  }

  // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
  if (N0.getOpcode() == ISD::SHL && N0.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(N0, N1, DAG);
    if (Result.getNode()) return Result;
  }
  if (N1.getOpcode() == ISD::SHL && N1.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(N1, N0, DAG);
    if (Result.getNode()) return Result;
  }

  // fold (add (select cc, 0, c), x) -> (select cc, x, (add, x, c))
  if (N0.getOpcode() == ISD::SELECT && N0.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N0, N1, DAG, TLI, LegalOperations);
    if (Result.getNode()) return Result;
  }
  if (N1.getOpcode() == ISD::SELECT && N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DAG, TLI, LegalOperations);
    if (Result.getNode()) return Result;
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N0.getValueType();
  
  // If the flag result is dead, turn this into an ADD.
  if (N->hasNUsesOfValue(0, 1))
    return CombineTo(N, DAG.getNode(ISD::ADD, VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  
  // canonicalize constant to RHS.
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDC, N->getVTList(), N1, N0);
  
  // fold (addc x, 0) -> x + no carry out
  if (N1C && N1C->isNullValue())
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  
  // fold (addc a, b) -> (or a, b), CARRY_FALSE iff a and b share no bits.
  APInt LHSZero, LHSOne;
  APInt RHSZero, RHSOne;
  APInt Mask = APInt::getAllOnesValue(VT.getSizeInBits());
  DAG.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);
  if (LHSZero.getBoolValue()) {
    DAG.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);
    
    // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
    // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
    if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
        (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
      return CombineTo(N, DAG.getNode(ISD::OR, VT, N0, N1),
                       DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitADDE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  //MVT VT = N0.getValueType();
  
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDE, N->getVTList(), N1, N0, CarryIn);
  
  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::ADDC, N->getVTList(), N1, N0);
  
  return SDValue();
}



SDValue DAGCombiner::visitSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  MVT VT = N0.getValueType();
  
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
    return DAG.getNode(ISD::ADD, VT, N0,
                       DAG.getConstant(-N1C->getAPIntValue(), VT));
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
    return DAG.getNode(N0.getOperand(1).getOpcode(), VT, N0.getOperand(0), 
                                     N0.getOperand(1).getOperand(1));
  // fold ((A+(C+B))-B) -> A+C
  if (N0.getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::ADD, VT, N0.getOperand(0), 
                                     N0.getOperand(1).getOperand(0));
  // fold ((A-(B-C))-C) -> A-B
  if (N0.getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::SUB, VT, N0.getOperand(0), 
                                     N0.getOperand(1).getOperand(0));
  // fold (sub x, (select cc, 0, c)) -> (select cc, x, (sub, x, c))
  if (N1.getOpcode() == ISD::SELECT && N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DAG, TLI, LegalOperations);
    if (Result.getNode()) return Result;
  }
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
  MVT VT = N0.getValueType();
  
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
    return DAG.getNode(ISD::MUL, VT, N1, N0);
  // fold (mul x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mul x, -1) -> 0-x
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), N0);
  // fold (mul x, (1 << c)) -> x << c
  if (N1C && N1C->getAPIntValue().isPowerOf2())
    return DAG.getNode(ISD::SHL, VT, N0,
                       DAG.getConstant(N1C->getAPIntValue().logBase2(),
                                       TLI.getShiftAmountTy()));
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1C && isPowerOf2_64(-N1C->getSExtValue())) {
    // FIXME: If the input is something that is easily negated (e.g. a 
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT),
                       DAG.getNode(ISD::SHL, VT, N0,
                            DAG.getConstant(Log2_64(-N1C->getSExtValue()),
                                            TLI.getShiftAmountTy())));
  }

  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      isa<ConstantSDNode>(N0.getOperand(1))) {
    SDValue C3 = DAG.getNode(ISD::SHL, VT, N1, N0.getOperand(1));
    AddToWorkList(C3.getNode());
    return DAG.getNode(ISD::MUL, VT, N0.getOperand(0), C3);
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
      SDValue Mul = DAG.getNode(ISD::MUL, VT, Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, VT, Mul, Sh.getOperand(1));
    }
  }
  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (N1C && N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse() && 
      isa<ConstantSDNode>(N0.getOperand(1))) {
    return DAG.getNode(ISD::ADD, VT, 
                       DAG.getNode(ISD::MUL, VT, N0.getOperand(0), N1),
                       DAG.getNode(ISD::MUL, VT, N0.getOperand(1), N1));
  }
  
  // reassociate mul
  SDValue RMUL = ReassociateOps(ISD::MUL, N0, N1);
  if (RMUL.getNode() != 0)
    return RMUL;

  return SDValue();
}

SDValue DAGCombiner::visitSDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  MVT VT = N->getValueType(0);

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
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), N0);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UDIV, N1.getValueType(), N0, N1);
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
    SDValue SGN = DAG.getNode(ISD::SRA, VT, N0,
                                DAG.getConstant(VT.getSizeInBits()-1,
                                                TLI.getShiftAmountTy()));
    AddToWorkList(SGN.getNode());
    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDValue SRL = DAG.getNode(ISD::SRL, VT, SGN,
                                DAG.getConstant(VT.getSizeInBits()-lg2,
                                                TLI.getShiftAmountTy()));
    SDValue ADD = DAG.getNode(ISD::ADD, VT, N0, SRL);
    AddToWorkList(SRL.getNode());
    AddToWorkList(ADD.getNode());    // Divide by pow2
    SDValue SRA = DAG.getNode(ISD::SRA, VT, ADD,
                                DAG.getConstant(lg2, TLI.getShiftAmountTy()));
    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (pow2 > 0)
      return SRA;
    AddToWorkList(SRA.getNode());
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), SRA);
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
  MVT VT = N->getValueType(0);
  
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
    return DAG.getNode(ISD::SRL, VT, N0, 
                       DAG.getConstant(N1C->getAPIntValue().logBase2(),
                                       TLI.getShiftAmountTy()));
  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        MVT ADDVT = N1.getOperand(1).getValueType();
        SDValue Add = DAG.getNode(ISD::ADD, ADDVT, N1.getOperand(1),
                                    DAG.getConstant(SHC->getAPIntValue()
                                                                    .logBase2(),
                                                    ADDVT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::SRL, VT, N0, Add);
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
  MVT VT = N->getValueType(0);
  
  // fold (srem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::SREM, VT, N0C, N1C);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UREM, VT, N0, N1);
  }
  
  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::SDIV, VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, VT, OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, VT, N0, Mul);
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
  MVT VT = N->getValueType(0);
  
  // fold (urem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.FoldConstantArithmetic(ISD::UREM, VT, N0C, N1C);
  // fold (urem x, pow2) -> (and x, pow2-1)
  if (N1C && !N1C->isNullValue() && N1C->getAPIntValue().isPowerOf2())
    return DAG.getNode(ISD::AND, VT, N0,
                       DAG.getConstant(N1C->getAPIntValue()-1,VT));
  // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        SDValue Add =
          DAG.getNode(ISD::ADD, VT, N1,
                 DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()),
                                 VT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::AND, VT, N0, Add);
      }
    }
  }
  
  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::UDIV, VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, VT, OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, VT, N0, Mul);
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
  MVT VT = N->getValueType(0);
  
  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0, 
                       DAG.getConstant(N0.getValueType().getSizeInBits()-1,
                                       TLI.getShiftAmountTy()));
  // fold (mulhs x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  return SDValue();
}

SDValue DAGCombiner::visitMULHU(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N->getValueType(0);
  
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
    SDValue Res = DAG.getNode(LoOp, N->getValueType(0), N->op_begin(),
                              N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If the low half is not needed, just compute the high half.
  bool LoExists = N->hasAnyUseOfValue(0);
  if (!LoExists &&
      (!LegalOperations ||
       TLI.isOperationLegal(HiOp, N->getValueType(1)))) {
    SDValue Res = DAG.getNode(HiOp, N->getValueType(1), N->op_begin(),
                              N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If both halves are used, return as it is.
  if (LoExists && HiExists)
    return SDValue();

  // If the two computed results can be simplified separately, separate them.
  if (LoExists) {
    SDValue Lo = DAG.getNode(LoOp, N->getValueType(0),
                               N->op_begin(), N->getNumOperands());
    AddToWorkList(Lo.getNode());
    SDValue LoOpt = combine(Lo.getNode());
    if (LoOpt.getNode() && LoOpt.getNode() != Lo.getNode() &&
        (!LegalOperations ||
         TLI.isOperationLegal(LoOpt.getOpcode(), LoOpt.getValueType())))
      return CombineTo(N, LoOpt, LoOpt);
  }

  if (HiExists) {
    SDValue Hi = DAG.getNode(HiOp, N->getValueType(1),
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
  MVT VT = N0.getValueType();
  assert(N0.getOpcode() == N1.getOpcode() && "Bad input!");
  
  // For each of OP in AND/OR/XOR:
  // fold (OP (zext x), (zext y)) -> (zext (OP x, y))
  // fold (OP (sext x), (sext y)) -> (sext (OP x, y))
  // fold (OP (aext x), (aext y)) -> (aext (OP x, y))
  // fold (OP (trunc x), (trunc y)) -> (trunc (OP x, y))
  if ((N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND||
       N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::TRUNCATE) &&
      N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), 
                                   N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), VT, ORNode);
  }
  
  // For each of OP in SHL/SRL/SRA/AND...
  //   fold (and (OP x, z), (OP y, z)) -> (OP (and x, y), z)
  //   fold (or  (OP x, z), (OP y, z)) -> (OP (or  x, y), z)
  //   fold (xor (OP x, z), (OP y, z)) -> (OP (xor x, y), z)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL ||
       N0.getOpcode() == ISD::SRA || N0.getOpcode() == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDValue ORNode = DAG.getNode(N->getOpcode(),
                                   N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), VT, ORNode, N0.getOperand(1));
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitAND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N1.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  
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
    return DAG.getNode(ISD::AND, VT, N1, N0);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(BitWidth)))
    return DAG.getConstant(0, VT);
  // reassociate and
  SDValue RAND = ReassociateOps(ISD::AND, N0, N1);
  if (RAND.getNode() != 0)
    return RAND;
  // fold (and (or x, 0xFFFF), 0xFF) -> 0xFF
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
      SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, N0.getValueType(),
                                   N0Op0);
      
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
      // fold (X == 0) & (Y == 0) -> (X|Y == 0)
      if (cast<ConstantSDNode>(LR)->isNullValue() && Op1 == ISD::SETEQ) {
        SDValue ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X == -1) & (Y == -1) -> (X&Y == -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETEQ) {
        SDValue ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
      }
      // fold (X >  -1) & (Y >  -1) -> (X|Y > -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETGT) {
        SDValue ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(VT, ORNode, LR, Op1);
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
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }

  // Simplify: and (op x...), (op y...)  -> (op (and x, y))
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
    MVT EVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                                     BitWidth - EVT.getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, EVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (ISD::isSEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT EVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned BitWidth = N1.getValueSizeInBits();
    if (DAG.MaskedValueIsZero(N1, APInt::getHighBitsSet(BitWidth,
                                     BitWidth - EVT.getSizeInBits())) &&
        ((!LegalOperations && !LN0->isVolatile()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, EVT))) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  
  // fold (and (load x), 255) -> (zextload x, i8)
  // fold (and (extload x, i16), 255) -> (zextload x, i8)
  if (N1C && N0.getOpcode() == ISD::LOAD) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    if (LN0->getExtensionType() != ISD::SEXTLOAD &&
        LN0->isUnindexed() && N0.hasOneUse() &&
        // Do not change the width of a volatile load.
        !LN0->isVolatile()) {
      MVT EVT = MVT::Other;
      uint32_t ActiveBits = N1C->getAPIntValue().getActiveBits();
      if (ActiveBits > 0 && APIntOps::isMask(ActiveBits, N1C->getAPIntValue()))
        EVT = MVT::getIntegerVT(ActiveBits);

      MVT LoadedVT = LN0->getMemoryVT();
      // Do not generate loads of non-round integer types since these can
      // be expensive (and would be wrong if the type is not byte sized).
      if (EVT != MVT::Other && LoadedVT.bitsGT(EVT) && EVT.isRound() &&
          (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, EVT))) {
        MVT PtrType = N0.getOperand(1).getValueType();
        // For big endian targets, we need to add an offset to the pointer to
        // load the correct bytes.  For little endian systems, we merely need to
        // read fewer bytes from the same pointer.
        unsigned LVTStoreBytes = LoadedVT.getStoreSizeInBits()/8;
        unsigned EVTStoreBytes = EVT.getStoreSizeInBits()/8;
        unsigned PtrOff = LVTStoreBytes - EVTStoreBytes;
        unsigned Alignment = LN0->getAlignment();
        SDValue NewPtr = LN0->getBasePtr();
        if (TLI.isBigEndian()) {
          NewPtr = DAG.getNode(ISD::ADD, PtrType, NewPtr,
                               DAG.getConstant(PtrOff, PtrType));
          Alignment = MinAlign(Alignment, PtrOff);
        }
        AddToWorkList(NewPtr.getNode());
        SDValue Load =
          DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(), NewPtr,
                         LN0->getSrcValue(), LN0->getSrcValueOffset(), EVT,
                         LN0->isVolatile(), Alignment);
        AddToWorkList(N);
        CombineTo(N0.getNode(), Load, Load.getValue(1));
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
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
  MVT VT = N1.getValueType();
  
  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }
  
  // fold (or x, undef) -> -1
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(~0ULL, VT);
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::OR, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::OR, VT, N1, N0);
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
  SDValue ROR = ReassociateOps(ISD::OR, N0, N1);
  if (ROR.getNode() != 0)
    return ROR;
  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  if (N1C && N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    return DAG.getNode(ISD::AND, VT, DAG.getNode(ISD::OR, VT, N0.getOperand(0),
                                                 N1),
                       DAG.getConstant(N1C->getAPIntValue() |
                                       C1->getAPIntValue(), VT));
  }
  // fold (or (setcc x), (setcc y)) -> (setcc (or x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();
    
    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        LL.getValueType().isInteger()) {
      // fold (X != 0) | (Y != 0) -> (X|Y != 0)
      // fold (X <  0) | (Y <  0) -> (X|Y < 0)
      if (cast<ConstantSDNode>(LR)->isNullValue() && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETLT)) {
        SDValue ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X != -1) | (Y != -1) -> (X&Y != -1)
      // fold (X >  -1) | (Y >  -1) -> (X&Y >  -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDValue ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
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
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }
  
  // Simplify: or (op x...), (op y...)  -> (op (or x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDValue Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.getNode()) return Tmp;
  }
  
  // (X & C1) | (Y & C2)  -> (X|Y) & C3  if possible.
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
      SDValue X =DAG.getNode(ISD::OR, VT, N0.getOperand(0), N1.getOperand(0));
      return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(LHSMask|RHSMask, VT));
    }
  }
  
  
  // See if this is some rotate idiom.
  if (SDNode *Rot = MatchRotate(N0, N1))
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
SDNode *DAGCombiner::MatchRotate(SDValue LHS, SDValue RHS) {
  // Must be a legal type.  Expanded 'n promoted things won't work with rotates.
  MVT VT = LHS.getValueType();
  if (!TLI.isTypeLegal(VT)) return 0;

  // The target must have at least one rotate flavor.
  bool HasROTL = TLI.isOperationLegal(ISD::ROTL, VT);
  bool HasROTR = TLI.isOperationLegal(ISD::ROTR, VT);
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
      Rot = DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt);
    else
      Rot = DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt);
    
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
        
      Rot = DAG.getNode(ISD::AND, VT, Rot, DAG.getConstant(Mask, VT));
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
          return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).getNode();
        else
          return DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt).getNode();
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
          return DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt).getNode();
        else
          return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).getNode();
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
          return DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, VT, LHSShiftArg,
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
          return DAG.getNode(HasROTR ? ISD::ROTR : ISD::ROTL, VT, LHSShiftArg,
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
  MVT VT = N0.getValueType();
  
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
    return DAG.getNode(ISD::XOR, VT, N1, N0);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // reassociate xor
  SDValue RXOR = ReassociateOps(ISD::XOR, N0, N1);
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
        assert(0 && "Unhandled SetCC Equivalent!");
        abort();
      case ISD::SETCC:
        return DAG.getSetCC(VT, LHS, RHS, NotCC);
      case ISD::SELECT_CC:
        return DAG.getSelectCC(LHS, RHS, N0.getOperand(2),
                               N0.getOperand(3), NotCC);
      }
    }
  }

  // fold (not (zext (setcc x, y))) -> (zext (not (setcc x, y)))
  if (N1C && N1C->getAPIntValue() == 1 && N0.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getNode()->hasOneUse() &&
      isSetCCEquivalent(N0.getOperand(0), LHS, RHS, CC)){
    SDValue V = N0.getOperand(0);
    V = DAG.getNode(ISD::XOR, V.getValueType(), V, 
                    DAG.getConstant(1, V.getValueType()));
    AddToWorkList(V.getNode());
    return DAG.getNode(ISD::ZERO_EXTEND, VT, V);
  }
  
  // fold !(x or y) -> (!x and !y) iff x or y are setcc
  if (N1C && N1C->getAPIntValue() == 1 && VT == MVT::i1 &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold !(x or y) -> (!x and !y) iff x or y are constants
  if (N1C && N1C->isAllOnesValue() && 
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold (xor (xor x, c1), c2) -> (xor x, c1^c2)
  if (N1C && N0.getOpcode() == ISD::XOR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getAPIntValue()^
                                         N00C->getAPIntValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getAPIntValue()^
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
      return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
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
  
  
  // FIXME: disable this for unless the input to the binop is a shift by a
  // constant.  If it is not a shift, it pessimizes some common cases like:
  //
  //void foo(int *X, int i) { X[i & 1235] = 1; }
  //int bar(int *X, int i) { return X[i & 255]; }
  SDNode *BinOpLHSVal = LHS->getOperand(0).getNode();
  if ((BinOpLHSVal->getOpcode() != ISD::SHL && 
       BinOpLHSVal->getOpcode() != ISD::SRA &&
       BinOpLHSVal->getOpcode() != ISD::SRL) ||
      !isa<ConstantSDNode>(BinOpLHSVal->getOperand(1)))
    return SDValue();
  
  MVT VT = N->getValueType(0);
  
  // If this is a signed shift right, and the high bit is modified
  // by the logical operation, do not perform the transformation.
  // The highBitSet boolean indicates the value of the high bit of
  // the constant which would cause it to be modified for this
  // operation.
  if (N->getOpcode() == ISD::SRA) {
    bool BinOpRHSSignSet = BinOpCst->getAPIntValue().isNegative();
    if (BinOpRHSSignSet != HighBitSet)
      return SDValue();
  }
  
  // Fold the constants, shifting the binop RHS by the shift amount.
  SDValue NewRHS = DAG.getNode(N->getOpcode(), N->getValueType(0),
                                 LHS->getOperand(1), N->getOperand(1));

  // Create the new shift.
  SDValue NewShift = DAG.getNode(N->getOpcode(), VT, LHS->getOperand(0),
                                   N->getOperand(1));

  // Create the new binop.
  return DAG.getNode(LHS->getOpcode(), VT, NewShift, NewRHS);
}


SDValue DAGCombiner::visitSHL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getSizeInBits();
  
  // fold (shl c1, c2) -> c1<<c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SHL, VT, N0C, N1C);
  // fold (shl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (shl x, c) is known to be zero, return 0
  if (DAG.MaskedValueIsZero(SDValue(N, 0),
                            APInt::getAllOnesValue(VT.getSizeInBits())))
    return DAG.getConstant(0, VT);
  // fold (shl x, (trunc (and y, c))) -> (shl x, (and (trunc y), c))
  // iff (trunc c) == c
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      MVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      return DAG.getNode(ISD::SHL, VT, N0,
                         DAG.getNode(ISD::AND, TruncVT,
                                     DAG.getNode(ISD::TRUNCATE, TruncVT, N100),
                                     DAG.getConstant(N101C->getZExtValue(),
                                                     TruncVT)));
    }
  }

  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);
  // fold (shl (shl x, c1), c2) -> 0 or (shl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, -1 << c1), c2-c1) or
  //                               (srl (and x, -1 << c1), c1-c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    SDValue Mask = DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                                 DAG.getConstant(~0ULL << c1, VT));
    if (c2 > c1)
      return DAG.getNode(ISD::SHL, VT, Mask, 
                         DAG.getConstant(c2-c1, N1.getValueType()));
    else
      return DAG.getNode(ISD::SRL, VT, Mask, 
                         DAG.getConstant(c1-c2, N1.getValueType()));
  }
  // fold (shl (sra x, c1), c1) -> (and x, -1 << c1)
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1))
    return DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                       DAG.getConstant(~0ULL << N1C->getZExtValue(), VT));
  
  return N1C ? visitShiftByConstant(N, N1C->getZExtValue()) : SDValue();
}

SDValue DAGCombiner::visitSRA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N0.getValueType();
  
  // fold (sra c1, c2) -> c1>>c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SRA, VT, N0C, N1C);
  // fold (sra 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (sra -1, x) -> -1
  if (N0C && N0C->isAllOnesValue())
    return N0;
  // fold (sra x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= VT.getSizeInBits())
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (sra (shl x, c1), c1) -> sext_inreg for some c1 and target supports
  // sext_inreg.
  if (N1C && N0.getOpcode() == ISD::SHL && N1 == N0.getOperand(1)) {
    unsigned LowBits = VT.getSizeInBits() - (unsigned)N1C->getZExtValue();
    MVT EVT = MVT::getIntegerVT(LowBits);
    if ((!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, EVT)))
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0),
                         DAG.getValueType(EVT));
  }

  // fold (sra (sra x, c1), c2) -> (sra x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRA) {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      unsigned Sum = N1C->getZExtValue() + C1->getZExtValue();
      if (Sum >= VT.getSizeInBits()) Sum = VT.getSizeInBits()-1;
      return DAG.getNode(ISD::SRA, VT, N0.getOperand(0),
                         DAG.getConstant(Sum, N1C->getValueType(0)));
    }
  }

  // fold sra (shl X, m), result_size - n
  // -> (sign_extend (trunc (shl X, result_size - n - m))) for
  // result_size - n != m. 
  // If truncate is free for the target sext(shl) is likely to result in better 
  // code.
  if (N0.getOpcode() == ISD::SHL) {
    // Get the two constanst of the shifts, CN0 = m, CN = n.
    const ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N01C && N1C) {
      // Determine what the truncate's result bitsize and type would be.
      unsigned VTValSize = VT.getSizeInBits();
      MVT TruncVT =
        MVT::getIntegerVT(VTValSize - N1C->getZExtValue());
      // Determine the residual right-shift amount.
      unsigned ShiftAmt = N1C->getZExtValue() - N01C->getZExtValue();

      // If the shift is not a no-op (in which case this should be just a sign 
      // extend already), the truncated to type is legal, sign_extend is legal 
      // on that type, and the the truncate to that type is both legal and free,
      // perform the transform.
      if (ShiftAmt &&
          TLI.isOperationLegal(ISD::SIGN_EXTEND, TruncVT) &&
          TLI.isOperationLegal(ISD::TRUNCATE, VT) &&
          TLI.isTruncateFree(VT, TruncVT)) {

          SDValue Amt = DAG.getConstant(ShiftAmt, TLI.getShiftAmountTy());
          SDValue Shift = DAG.getNode(ISD::SRL, VT, N0.getOperand(0), Amt);
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, TruncVT, Shift);
          return DAG.getNode(ISD::SIGN_EXTEND, N->getValueType(0), Trunc);
      }
    }
  }
  
  // fold (sra x, (trunc (and y, c))) -> (sra x, (and (trunc y), c))
  // iff (trunc c) == c
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      MVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      return DAG.getNode(ISD::SRA, VT, N0,
                         DAG.getNode(ISD::AND, TruncVT,
                                     DAG.getNode(ISD::TRUNCATE, TruncVT, N100),
                                     DAG.getConstant(N101C->getZExtValue(),
                                                     TruncVT)));
    }
  }

  // Simplify, based on bits shifted out of the LHS. 
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);
  
  
  // If the sign bit is known to be zero, switch this to a SRL.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SRL, VT, N0, N1);

  return N1C ? visitShiftByConstant(N, N1C->getZExtValue()) : SDValue();
}

SDValue DAGCombiner::visitSRL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getSizeInBits();
  
  // fold (srl c1, c2) -> c1 >>u c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SRL, VT, N0C, N1C);
  // fold (srl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getZExtValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, VT);
  
  // fold (srl (srl x, c1), c2) -> 0 or (srl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  
  // fold (srl (anyextend x), c) -> (anyextend (srl x, c))
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    MVT SmallVT = N0.getOperand(0).getValueType();
    if (N1C->getZExtValue() >= SmallVT.getSizeInBits())
      return DAG.getNode(ISD::UNDEF, VT);

    SDValue SmallShift = DAG.getNode(ISD::SRL, SmallVT, N0.getOperand(0), N1);
    AddToWorkList(SmallShift.getNode());
    return DAG.getNode(ISD::ANY_EXTEND, VT, SmallShift);
  }
  
  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getZExtValue()+1 == VT.getSizeInBits()) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), N1);
  }
  
  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ && 
      N1C->getAPIntValue() == Log2_32(VT.getSizeInBits())) {
    APInt KnownZero, KnownOne;
    APInt Mask = APInt::getAllOnesValue(VT.getSizeInBits());
    DAG.ComputeMaskedBits(N0.getOperand(0), Mask, KnownZero, KnownOne);
    
    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (KnownOne.getBoolValue()) return DAG.getConstant(0, VT);
    
    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    APInt UnknownBits = ~KnownZero & Mask;
    if (UnknownBits == 0) return DAG.getConstant(1, VT);
    
    // Otherwise, check to see if there is exactly one bit input to the ctlz.
    if ((UnknownBits & (UnknownBits-1)) == 0) {
      // Okay, we know that only that the single bit specified by UnknownBits
      // could be set on input to the CTLZ node.  If this bit is set, the SRL
      // will return 0, if it is clear, it returns 1.  Change the CTLZ/SRL pair
      // to an SRL,XOR pair, which is likely to simplify more.
      unsigned ShAmt = UnknownBits.countTrailingZeros();
      SDValue Op = N0.getOperand(0);
      if (ShAmt) {
        Op = DAG.getNode(ISD::SRL, VT, Op,
                         DAG.getConstant(ShAmt, TLI.getShiftAmountTy()));
        AddToWorkList(Op.getNode());
      }
      return DAG.getNode(ISD::XOR, VT, Op, DAG.getConstant(1, VT));
    }
  }

  // fold (srl x, (trunc (and y, c))) -> (srl x, (and (trunc y), c))
  // iff (trunc c) == c
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND &&
      N1.hasOneUse() && N1.getOperand(0).hasOneUse()) {
    SDValue N101 = N1.getOperand(0).getOperand(1);
    if (ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N101)) {
      MVT TruncVT = N1.getValueType();
      SDValue N100 = N1.getOperand(0).getOperand(0);
      return DAG.getNode(ISD::SRL, VT, N0,
                         DAG.getNode(ISD::AND, TruncVT,
                                     DAG.getNode(ISD::TRUNCATE, TruncVT, N100),
                                     DAG.getConstant(N101C->getZExtValue(),
                                                     TruncVT)));
    }
  }
  
  // fold operands of srl based on knowledge that the low bits are not
  // demanded.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);
  
  return N1C ? visitShiftByConstant(N, N1C->getZExtValue()) : SDValue();
}

SDValue DAGCombiner::visitCTLZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);

  // fold (ctlz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTLZ, VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);
  
  // fold (cttz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTTZ, VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTPOP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);
  
  // fold (ctpop c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTPOP, VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  MVT VT = N->getValueType(0);
  MVT VT0 = N0.getValueType();

  // fold select C, X, X -> X
  if (N1 == N2)
    return N1;
  // fold select true, X, Y -> X
  if (N0C && !N0C->isNullValue())
    return N1;
  // fold select false, X, Y -> Y
  if (N0C && N0C->isNullValue())
    return N2;
  // fold select C, 1, X -> C | X
  if (VT == MVT::i1 && N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold select C, 0, 1 -> ~C
  if (VT.isInteger() && VT0.isInteger() &&
      N1C && N2C && N1C->isNullValue() && N2C->getAPIntValue() == 1) {
    SDValue XORNode = DAG.getNode(ISD::XOR, VT0, N0, DAG.getConstant(1, VT0));
    if (VT == VT0)
      return XORNode;
    AddToWorkList(XORNode.getNode());
    if (VT.bitsGT(VT0))
      return DAG.getNode(ISD::ZERO_EXTEND, VT, XORNode);
    return DAG.getNode(ISD::TRUNCATE, VT, XORNode);
  }
  // fold select C, 0, X -> ~C & X
  if (VT == VT0 && VT == MVT::i1 && N1C && N1C->isNullValue()) {
    SDValue XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    AddToWorkList(XORNode.getNode());
    return DAG.getNode(ISD::AND, VT, XORNode, N2);
  }
  // fold select C, X, 1 -> ~C | X
  if (VT == VT0 && VT == MVT::i1 && N2C && N2C->getAPIntValue() == 1) {
    SDValue XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    AddToWorkList(XORNode.getNode());
    return DAG.getNode(ISD::OR, VT, XORNode, N1);
  }
  // fold select C, X, 0 -> C & X
  // FIXME: this should check for C type == X type, not i1?
  if (VT == MVT::i1 && N2C && N2C->isNullValue())
    return DAG.getNode(ISD::AND, VT, N0, N1);
  // fold  X ? X : Y --> X ? 1 : Y --> X | Y
  if (VT == MVT::i1 && N0 == N1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold X ? Y : X --> X ? Y : 0 --> X & Y
  if (VT == MVT::i1 && N0 == N2)
    return DAG.getNode(ISD::AND, VT, N0, N1);
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0);  // Don't revisit N.

  // fold selects based on a setcc into other things, such as min/max/abs
  if (N0.getOpcode() == ISD::SETCC) {
    // FIXME:
    // Check against MVT::Other for SELECT_CC, which is a workaround for targets
    // having to say they don't support SELECT_CC on every type the DAG knows
    // about, since there is no way to mark an opcode illegal at all value types
    if (TLI.isOperationLegal(ISD::SELECT_CC, MVT::Other))
      return DAG.getNode(ISD::SELECT_CC, VT, N0.getOperand(0), N0.getOperand(1),
                         N1, N2, N0.getOperand(2));
    else
      return SimplifySelect(N0, N1, N2);
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
                              N0, N1, CC, false);
  if (SCC.getNode()) AddToWorkList(SCC.getNode());

  if (ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.getNode())) {
    if (!SCCC->isNullValue())
      return N2;    // cond always true -> true val
    else
      return N3;    // cond always false -> false val
  }
  
  // Fold to a simpler select_cc
  if (SCC.getNode() && SCC.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::SELECT_CC, N2.getValueType(), 
                       SCC.getOperand(0), SCC.getOperand(1), N2, N3, 
                       SCC.getOperand(2));
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDValue(N, 0);  // Don't revisit N.
  
  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(N0, N1, N2, N3, CC);
}

SDValue DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get());
}

// ExtendUsesToFormExtLoad - Trying to extend uses of a load to enable this:
// "fold ({s|z}ext (load x)) -> ({s|z}ext (truncate ({s|z}extload x)))"
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
    // FIXME: Only extend SETCC N, N and SETCC N, c for now.
    if (User->getOpcode() == ISD::SETCC) {
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
    } else {
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
        SDValue UseOp = User->getOperand(i);
        if (UseOp == N0) {
          // If truncate from extended type to original load type is free
          // on this target, then it's ok to extend a CopyToReg.
          if (isTruncFree && User->getOpcode() == ISD::CopyToReg)
            HasCopyToRegUses = true;
          else
            return false;
        }
      }
    }
  }

  if (HasCopyToRegUses) {
    bool BothLiveOut = false;
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDNode *User = *UI;
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
        SDValue UseOp = User->getOperand(i);
        if (UseOp.getNode() == N && UseOp.getResNo() == 0) {
          BothLiveOut = true;
          break;
        }
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
  MVT VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0);
  
  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0.getOperand(0));
  
  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (sext (truncate (load x))) -> (sext (smaller load x))
    // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return DAG.getNode(ISD::SIGN_EXTEND, VT, NarrowLoad);
    }

    // See if the value being truncated is already sign extended.  If so, just
    // eliminate the trunc/sext pair.
    SDValue Op = N0.getOperand(0);
    unsigned OpBits   = Op.getValueType().getSizeInBits();
    unsigned MidBits  = N0.getValueType().getSizeInBits();
    unsigned DestBits = VT.getSizeInBits();
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
        return DAG.getNode(ISD::SIGN_EXTEND, VT, Op);
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::TRUNCATE, VT, Op);
    }
    
    // fold (sext (truncate x)) -> (sextinreg x).
    if (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                                 N0.getValueType())) {
      if (Op.getValueType().bitsLT(VT))
        Op = DAG.getNode(ISD::ANY_EXTEND, VT, Op);
      else if (Op.getValueType().bitsGT(VT))
        Op = DAG.getNode(ISD::TRUNCATE, VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, Op,
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), LN0->getAlignment());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad);
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
            Ops.push_back(DAG.getNode(ISD::SIGN_EXTEND, VT, SOp));
          }
        Ops.push_back(SetCC->getOperand(2));
        CombineTo(SetCC, DAG.getNode(ISD::SETCC, SetCC->getValueType(0),
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
    MVT EVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), LN0->getAlignment());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  
  // sext(setcc x,y,cc) -> select_cc x, y, -1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(~0ULL, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
  }
  
  // fold (sext x) -> (zext x) if the sign bit is known zero.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ZERO_EXTEND, VT)) &&
      DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0);
  
  return SDValue();
}

SDValue DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);

  // fold (zext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0);
  // fold (zext (zext x)) -> (zext x)
  // fold (zext (aext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0.getOperand(0));

  // fold (zext (truncate (load x))) -> (zext (smaller load x))
  // fold (zext (truncate (srl (load x), c))) -> (zext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return DAG.getNode(ISD::ZERO_EXTEND, VT, NarrowLoad);
    }
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (!LegalOperations || TLI.isOperationLegal(ISD::AND, VT))) {
    SDValue Op = N0.getOperand(0);
    if (Op.getValueType().bitsLT(VT)) {
      Op = DAG.getNode(ISD::ANY_EXTEND, VT, Op);
    } else if (Op.getValueType().bitsGT(VT)) {
      Op = DAG.getNode(ISD::TRUNCATE, VT, Op);
    }
    return DAG.getZeroExtendInReg(Op, N0.getValueType());
  }
  
  // fold (zext (and (trunc x), cst)) -> (and x, cst).
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(Mask, VT));
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), LN0->getAlignment());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad);
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
            Ops.push_back(DAG.getNode(ISD::ZERO_EXTEND, VT, SOp));
          }
        Ops.push_back(SetCC->getOperand(2));
        CombineTo(SetCC, DAG.getNode(ISD::SETCC, SetCC->getValueType(0),
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
    MVT EVT = LN0->getMemoryVT();
    if ((!LegalOperations && !LN0->isVolatile()) ||
        TLI.isLoadExtLegal(ISD::ZEXTLOAD, EVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), LN0->getAlignment());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  
  // zext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);
  
  // fold (aext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ANY_EXTEND, VT, N0);
  // fold (aext (aext x)) -> (aext x)
  // fold (aext (zext x)) -> (zext x)
  // fold (aext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::ANY_EXTEND  ||
      N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0));
  
  // fold (aext (truncate (load x))) -> (aext (smaller load x))
  // fold (aext (truncate (srl (load x), c))) -> (aext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      if (NarrowLoad.getNode() != N0.getNode())
        CombineTo(N0.getNode(), NarrowLoad);
      return DAG.getNode(ISD::ANY_EXTEND, VT, NarrowLoad);
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue TruncOp = N0.getOperand(0);
    if (TruncOp.getValueType() == VT)
      return TruncOp; // x iff x size == zext size.
    if (TruncOp.getValueType().bitsGT(VT))
      return DAG.getNode(ISD::TRUNCATE, VT, TruncOp);
    return DAG.getNode(ISD::ANY_EXTEND, VT, TruncOp);
  }
  
  // fold (aext (and (trunc x), cst)) -> (and x, cst).
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    SDValue X = N0.getOperand(0).getOperand(0);
    if (X.getValueType().bitsLT(VT)) {
      X = DAG.getNode(ISD::ANY_EXTEND, VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(Mask, VT));
  }
  
  // fold (aext (load x)) -> (aext (truncate (extload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) && N0.hasOneUse() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, VT, LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(),
                                     N0.getValueType(),
                                     LN0->isVolatile(), LN0->getAlignment());
    CombineTo(N, ExtLoad);
    // Redirect any chain users to the new load.
    DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1),
                                  SDValue(ExtLoad.getNode(), 1));
    // If any node needs the original loaded value, recompute it.
    if (!LN0->use_empty())
      CombineTo(LN0, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  // fold (aext (zextload x)) -> (aext (truncate (zextload x)))
  // fold (aext (sextload x)) -> (aext (truncate (sextload x)))
  // fold (aext ( extload x)) -> (aext (truncate (extload  x)))
  if (N0.getOpcode() == ISD::LOAD &&
      !ISD::isNON_EXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT EVT = LN0->getMemoryVT();
    SDValue ExtLoad = DAG.getExtLoad(LN0->getExtensionType(), VT,
                                     LN0->getChain(), LN0->getBasePtr(),
                                     LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), EVT,
                                     LN0->isVolatile(), LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
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
      if (SimplifyLHS.getNode()) {
        return DAG.getNode(ISD::SRL, V.getValueType(), 
                           SimplifyLHS, V.getOperand(1));
      }
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
  MVT VT = N->getValueType(0);
  MVT EVT = VT;

  // This transformation isn't valid for vector loads.
  if (VT.isVector())
    return SDValue();

  // Special case: SIGN_EXTEND_INREG is basically truncating to EVT then
  // extended to VT.
  if (Opc == ISD::SIGN_EXTEND_INREG) {
    ExtType = ISD::SEXTLOAD;
    EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
    if (LegalOperations && !TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT))
      return SDValue();
  }

  unsigned EVTBits = EVT.getSizeInBits();
  unsigned ShAmt = 0;
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShAmt = N01->getZExtValue();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        if (N0.getValueType().getSizeInBits() <= EVTBits)
          return SDValue();
      }
    }
  }

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (isa<LoadSDNode>(N0) && N0.hasOneUse() && EVT.isRound() &&
      cast<LoadSDNode>(N0)->getMemoryVT().getSizeInBits() > EVTBits &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT PtrType = N0.getOperand(1).getValueType();
    // For big endian targets, we need to adjust the offset to the pointer to
    // load the correct bytes.
    if (TLI.isBigEndian()) {
      unsigned LVTStoreBits = LN0->getMemoryVT().getStoreSizeInBits();
      unsigned EVTStoreBits = EVT.getStoreSizeInBits();
      ShAmt = LVTStoreBits - EVTStoreBits - ShAmt;
    }
    uint64_t PtrOff =  ShAmt / 8;
    unsigned NewAlign = MinAlign(LN0->getAlignment(), PtrOff);
    SDValue NewPtr = DAG.getNode(ISD::ADD, PtrType, LN0->getBasePtr(),
                                 DAG.getConstant(PtrOff, PtrType));
    AddToWorkList(NewPtr.getNode());
    SDValue Load = (ExtType == ISD::NON_EXTLOAD)
      ? DAG.getLoad(VT, LN0->getChain(), NewPtr,
                    LN0->getSrcValue(), LN0->getSrcValueOffset() + PtrOff,
                    LN0->isVolatile(), NewAlign)
      : DAG.getExtLoad(ExtType, VT, LN0->getChain(), NewPtr,
                       LN0->getSrcValue(), LN0->getSrcValueOffset() + PtrOff,
                       EVT, LN0->isVolatile(), NewAlign);
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
  MVT VT = N->getValueType(0);
  MVT EVT = cast<VTSDNode>(N1)->getVT();
  unsigned VTBits = VT.getSizeInBits();
  unsigned EVTBits = EVT.getSizeInBits();
  
  // fold (sext_in_reg c1) -> c1
  if (isa<ConstantSDNode>(N0) || N0.getOpcode() == ISD::UNDEF)
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0, N1);
  
  // If the input is already sign extended, just drop the extension.
  if (DAG.ComputeNumSignBits(N0) >= VT.getSizeInBits()-EVTBits+1)
    return N0;
  
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT.bitsLT(cast<VTSDNode>(N0.getOperand(1))->getVT())) {
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0), N1);
  }

  // fold (sext_in_reg (sext x)) -> (sext x)
  // fold (sext_in_reg (aext x)) -> (sext x)
  // if x is small enough.
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getValueType().getSizeInBits() < EVTBits)
      return DAG.getNode(ISD::SIGN_EXTEND, VT, N00, N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (DAG.MaskedValueIsZero(N0, APInt::getBitsSet(VTBits, EVTBits-1, EVTBits)))
    return DAG.getZeroExtendInReg(N0, EVT);
  
  // fold operands of sext_in_reg based on knowledge that the top bits are not
  // demanded.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);
  
  // fold (sext_in_reg (load x)) -> (smaller sextload x)
  // fold (sext_in_reg (srl (load x), c)) -> (smaller sextload (x+c/evtbits))
  SDValue NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.getNode())
    return NarrowLoad;

  // fold (sext_in_reg (srl X, 24), i8) -> sra X, 24
  // fold (sext_in_reg (srl X, 23), i8) -> sra X, 23 iff possible.
  // We already fold "(sext_in_reg (srl X, 25), i8) -> srl X, 25" above.
  if (N0.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if (ShAmt->getZExtValue()+EVTBits <= VT.getSizeInBits()) {
        // We can turn this into an SRA iff the input to the SRL is already sign
        // extended enough.
        unsigned InSignBits = DAG.ComputeNumSignBits(N0.getOperand(0));
        if (VT.getSizeInBits()-(ShAmt->getZExtValue()+EVTBits) < InSignBits)
          return DAG.getNode(ISD::SRA, VT, N0.getOperand(0), N0.getOperand(1));
      }
  }

  // fold (sext_inreg (extload x)) -> (sextload x)
  if (ISD::isEXTLoad(N0.getNode()) && 
      ISD::isUNINDEXEDLoad(N0.getNode()) &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), EVT,
                                     LN0->isVolatile(), LN0->getAlignment());
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
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(), EVT,
                                     LN0->isVolatile(), LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  return SDValue();
}

SDValue DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::TRUNCATE, VT, N0);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::SIGN_EXTEND||
      N0.getOpcode() == ISD::ANY_EXTEND) {
    if (N0.getOperand(0).getValueType().bitsLT(VT))
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0));
    else if (N0.getOperand(0).getValueType().bitsGT(VT))
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate
      return N0.getOperand(0);
  }

  // See if we can simplify the input to this truncate through knowledge that
  // only the low bits are being used.  For example "trunc (or (shl x, 8), y)"
  // -> trunc y
  SDValue Shorter =
    GetDemandedBits(N0, APInt::getLowBitsSet(N0.getValueSizeInBits(),
                                             VT.getSizeInBits()));
  if (Shorter.getNode())
    return DAG.getNode(ISD::TRUNCATE, VT, Shorter);

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
SDValue DAGCombiner::CombineConsecutiveLoads(SDNode *N, MVT VT) {
  assert(N->getOpcode() == ISD::BUILD_PAIR);

  SDNode *LD1 = getBuildPairElt(N, 0);
  if (!ISD::isNON_EXTLoad(LD1) || !LD1->hasOneUse())
    return SDValue();
  MVT LD1VT = LD1->getValueType(0);
  SDNode *LD2 = getBuildPairElt(N, 1);
  const MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  if (ISD::isNON_EXTLoad(LD2) &&
      LD2->hasOneUse() &&
      // If both are volatile this would reduce the number of volatile loads.
      // If one is volatile it might be ok, but play conservative and bail out.
      !cast<LoadSDNode>(LD1)->isVolatile() &&
      !cast<LoadSDNode>(LD2)->isVolatile() &&
      TLI.isConsecutiveLoad(LD2, LD1, LD1VT.getSizeInBits()/8, 1, MFI)) {
    LoadSDNode *LD = cast<LoadSDNode>(LD1);
    unsigned Align = LD->getAlignment();
    unsigned NewAlign = TLI.getTargetData()->
      getABITypeAlignment(VT.getTypeForMVT());
    if (NewAlign <= Align &&
        (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)))
      return DAG.getLoad(VT, LD->getChain(), LD->getBasePtr(),
                         LD->getSrcValue(), LD->getSrcValueOffset(),
                         false, Align);
  }
  return SDValue();
}

SDValue DAGCombiner::visitBIT_CONVERT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);

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
        
    MVT DestEltVT = N->getValueType(0).getVectorElementType();
    assert(!DestEltVT.isVector() &&
           "Element type of vector ValueType must not be vector!");
    if (isSimple) {
      return ConstantFoldBIT_CONVERTofBUILD_VECTOR(N0.getNode(), DestEltVT);
    }
  }
  
  // If the input is a constant, let getNode fold it.
  if (isa<ConstantSDNode>(N0) || isa<ConstantFPSDNode>(N0)) {
    SDValue Res = DAG.getNode(ISD::BIT_CONVERT, VT, N0);
    if (Res.getNode() != N) return Res;
  }
  
  if (N0.getOpcode() == ISD::BIT_CONVERT)  // conv(conv(x,t1),t2) -> conv(x,t2)
    return DAG.getNode(ISD::BIT_CONVERT, VT, N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a higher alignment than the original!
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile() &&
      (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    unsigned Align = TLI.getTargetData()->
      getABITypeAlignment(VT.getTypeForMVT());
    unsigned OrigAlign = LN0->getAlignment();
    if (Align <= OrigAlign) {
      SDValue Load = DAG.getLoad(VT, LN0->getChain(), LN0->getBasePtr(),
                                 LN0->getSrcValue(), LN0->getSrcValueOffset(),
                                 LN0->isVolatile(), OrigAlign);
      AddToWorkList(N);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::BIT_CONVERT, N0.getValueType(), Load),
                Load.getValue(1));
      return Load;
    }
  }

  // Fold bitconvert(fneg(x)) -> xor(bitconvert(x), signbit)
  // Fold bitconvert(fabs(x)) -> and(bitconvert(x), ~signbit)
  // This often reduces constant pool loads.
  if ((N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FABS) &&
      N0.getNode()->hasOneUse() && VT.isInteger() && !VT.isVector()) {
    SDValue NewConv = DAG.getNode(ISD::BIT_CONVERT, VT, N0.getOperand(0));
    AddToWorkList(NewConv.getNode());
    
    APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
    if (N0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, VT, NewConv, DAG.getConstant(SignBit, VT));
    assert(N0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, VT, NewConv, DAG.getConstant(~SignBit, VT));
  }
  
  // Fold bitconvert(fcopysign(cst, x)) -> bitconvert(x)&sign | cst&~sign'
  // Note that we don't handle copysign(x,cst) because this can always be folded
  // to an fneg or fabs.
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse() &&
      isa<ConstantFPSDNode>(N0.getOperand(0)) &&
      VT.isInteger() && !VT.isVector()) {
    unsigned OrigXWidth = N0.getOperand(1).getValueType().getSizeInBits();
    MVT IntXVT = MVT::getIntegerVT(OrigXWidth);
    if (TLI.isTypeLegal(IntXVT) || !LegalTypes) {
      SDValue X = DAG.getNode(ISD::BIT_CONVERT, IntXVT, N0.getOperand(1));
      AddToWorkList(X.getNode());

      // If X has a different width than the result/lhs, sext it or truncate it.
      unsigned VTWidth = VT.getSizeInBits();
      if (OrigXWidth < VTWidth) {
        X = DAG.getNode(ISD::SIGN_EXTEND, VT, X);
        AddToWorkList(X.getNode());
      } else if (OrigXWidth > VTWidth) {
        // To get the sign bit in the right place, we have to shift it right
        // before truncating.
        X = DAG.getNode(ISD::SRL, X.getValueType(), X,
                        DAG.getConstant(OrigXWidth-VTWidth, X.getValueType()));
        AddToWorkList(X.getNode());
        X = DAG.getNode(ISD::TRUNCATE, VT, X);
        AddToWorkList(X.getNode());
      }
    
      APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
      X = DAG.getNode(ISD::AND, VT, X, DAG.getConstant(SignBit, VT));
      AddToWorkList(X.getNode());

      SDValue Cst = DAG.getNode(ISD::BIT_CONVERT, VT, N0.getOperand(0));
      Cst = DAG.getNode(ISD::AND, VT, Cst, DAG.getConstant(~SignBit, VT));
      AddToWorkList(Cst.getNode());

      return DAG.getNode(ISD::OR, VT, X, Cst);
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
  MVT VT = N->getValueType(0);
  return CombineConsecutiveLoads(N, VT);
}

/// ConstantFoldBIT_CONVERTofBUILD_VECTOR - We know that BV is a build_vector
/// node with Constant, ConstantFP or Undef operands.  DstEltVT indicates the 
/// destination element value type.
SDValue DAGCombiner::
ConstantFoldBIT_CONVERTofBUILD_VECTOR(SDNode *BV, MVT DstEltVT) {
  MVT SrcEltVT = BV->getOperand(0).getValueType();
  
  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDValue(BV, 0);
  
  unsigned SrcBitSize = SrcEltVT.getSizeInBits();
  unsigned DstBitSize = DstEltVT.getSizeInBits();
  
  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
      Ops.push_back(DAG.getNode(ISD::BIT_CONVERT, DstEltVT, BV->getOperand(i)));
      AddToWorkList(Ops.back().getNode());
    }
    MVT VT = MVT::getVectorVT(DstEltVT,
                              BV->getValueType(0).getVectorNumElements());
    return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
  
  // Otherwise, we're growing or shrinking the elements.  To avoid having to
  // handle annoying details of growing/shrinking FP values, we convert them to
  // int first.
  if (SrcEltVT.isFloatingPoint()) {
    // Convert the input float vector to a int vector where the elements are the
    // same sizes.
    assert((SrcEltVT == MVT::f32 || SrcEltVT == MVT::f64) && "Unknown FP VT!");
    MVT IntVT = MVT::getIntegerVT(SrcEltVT.getSizeInBits());
    BV = ConstantFoldBIT_CONVERTofBUILD_VECTOR(BV, IntVT).getNode();
    SrcEltVT = IntVT;
  }
  
  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (DstEltVT.isFloatingPoint()) {
    assert((DstEltVT == MVT::f32 || DstEltVT == MVT::f64) && "Unknown FP VT!");
    MVT TmpVT = MVT::getIntegerVT(DstEltVT.getSizeInBits());
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
        
        NewBits |=
          APInt(cast<ConstantSDNode>(Op)->getAPIntValue()).zext(DstBitSize);
      }
      
      if (EltIsUndef)
        Ops.push_back(DAG.getNode(ISD::UNDEF, DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DstEltVT));
    }

    MVT VT = MVT::getVectorVT(DstEltVT, Ops.size());
    return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
  
  // Finally, this must be the case where we are shrinking elements: each input
  // turns into multiple outputs.
  bool isS2V = ISD::isScalarToVector(BV);
  unsigned NumOutputsPerInput = SrcBitSize/DstBitSize;
  MVT VT = MVT::getVectorVT(DstEltVT, NumOutputsPerInput*BV->getNumOperands());
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
    if (BV->getOperand(i).getOpcode() == ISD::UNDEF) {
      for (unsigned j = 0; j != NumOutputsPerInput; ++j)
        Ops.push_back(DAG.getNode(ISD::UNDEF, DstEltVT));
      continue;
    }
    APInt OpVal = cast<ConstantSDNode>(BV->getOperand(i))->getAPIntValue();
    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      APInt ThisVal = APInt(OpVal).trunc(DstBitSize);
      Ops.push_back(DAG.getConstant(ThisVal, DstEltVT));
      if (isS2V && i == 0 && j == 0 && APInt(ThisVal).zext(SrcBitSize) == OpVal)
        // Simply turn this into a SCALAR_TO_VECTOR of the new type.
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Ops[0]);
      OpVal = OpVal.lshr(DstBitSize);
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (TLI.isBigEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }
  return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
}



SDValue DAGCombiner::visitFADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT VT = N->getValueType(0);
  
  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }
  
  // fold (fadd c1, c2) -> c1+c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FADD, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, VT, N1, N0);
  // fold (A + 0) -> A
  if (UnsafeFPMath && N1CFP && N1CFP->getValueAPF().isZero())
    return N0;
  // fold (A + (-B)) -> A-B
  if (isNegatibleForFree(N1, LegalOperations) == 2)
    return DAG.getNode(ISD::FSUB, VT, N0, 
                       GetNegatedExpression(N1, DAG, LegalOperations));
  // fold ((-A) + B) -> B-A
  if (isNegatibleForFree(N0, LegalOperations) == 2)
    return DAG.getNode(ISD::FSUB, VT, N1, 
                       GetNegatedExpression(N0, DAG, LegalOperations));
  
  // If allowed, fold (fadd (fadd x, c1), c2) -> (fadd x, (fadd c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FADD &&
      N0.getNode()->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FADD, VT, N0.getOperand(0),
                       DAG.getNode(ISD::FADD, VT, N0.getOperand(1), N1));
  
  return SDValue();
}

SDValue DAGCombiner::visitFSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT VT = N->getValueType(0);
  
  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }
  
  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FSUB, VT, N0, N1);
  // fold (0-B) -> -B
  if (UnsafeFPMath && N0CFP && N0CFP->getValueAPF().isZero()) {
    if (isNegatibleForFree(N1, LegalOperations))
      return GetNegatedExpression(N1, DAG, LegalOperations);
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, VT, N1);
  }
  // fold (A-(-B)) -> A+B
  if (isNegatibleForFree(N1, LegalOperations))
    return DAG.getNode(ISD::FADD, VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));
  
  return SDValue();
}

SDValue DAGCombiner::visitFMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }
  
  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FMUL, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMUL, VT, N1, N0);
  // fold (A * 0) -> 0
  if (UnsafeFPMath && N1CFP && N1CFP->getValueAPF().isZero())
    return N1;
  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, VT, N0, N0);
  // fold (fmul X, -1.0) -> (fneg X)
  if (N1CFP && N1CFP->isExactlyValue(-1.0))
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, VT, N0);
  
  // -X * -Y -> X*Y
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FMUL, VT, 
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }
  
  // If allowed, fold (fmul (fmul x, c1), c2) -> (fmul x, (fmul c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FMUL &&
      N0.getNode()->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FMUL, VT, N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, VT, N0.getOperand(1), N1));
  
  return SDValue();
}

SDValue DAGCombiner::visitFDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT VT = N->getValueType(0);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }
  
  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FDIV, VT, N0, N1);
  
  
  // -X / -Y -> X*Y
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FDIV, VT, 
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
  MVT VT = N->getValueType(0);

  // fold (frem c1, c2) -> fmod(c1,c2)
  if (N0CFP && N1CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FREM, VT, N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT VT = N->getValueType(0);

  if (N0CFP && N1CFP && VT != MVT::ppcf128)  // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, VT, N0, N1);
  
  if (N1CFP) {
    const APFloat& V = N1CFP->getValueAPF();
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    if (!V.isNegative()) {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FABS, VT))
        return DAG.getNode(ISD::FABS, VT, N0);
    } else {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, VT, DAG.getNode(ISD::FABS, VT, N0));
    }
  }
  
  // copysign(fabs(x), y) -> copysign(x, y)
  // copysign(fneg(x), y) -> copysign(x, y)
  // copysign(copysign(x,z), y) -> copysign(x, y)
  if (N0.getOpcode() == ISD::FABS || N0.getOpcode() == ISD::FNEG ||
      N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, VT, N0.getOperand(0), N1);

  // copysign(x, abs(y)) -> abs(x)
  if (N1.getOpcode() == ISD::FABS)
    return DAG.getNode(ISD::FABS, VT, N0);
  
  // copysign(x, copysign(y,z)) -> copysign(x, z)
  if (N1.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, VT, N0, N1.getOperand(1));
  
  // copysign(x, fp_extend(y)) -> copysign(x, y)
  // copysign(x, fp_round(y)) -> copysign(x, y)
  if (N1.getOpcode() == ISD::FP_EXTEND || N1.getOpcode() == ISD::FP_ROUND)
    return DAG.getNode(ISD::FCOPYSIGN, VT, N0, N1.getOperand(0));
  
  return SDValue();
}



SDValue DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT VT = N->getValueType(0);
  MVT OpVT = N0.getValueType();

  // fold (sint_to_fp c1) -> c1fp
  if (N0C && OpVT != MVT::ppcf128)
    return DAG.getNode(ISD::SINT_TO_FP, VT, N0);
  
  // If the input is a legal type, and SINT_TO_FP is not legal on this target,
  // but UINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegal(ISD::SINT_TO_FP, OpVT) &&
      TLI.isOperationLegal(ISD::UINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to UINT_TO_FP. 
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UINT_TO_FP, VT, N0);
  }
  
  
  return SDValue();
}

SDValue DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT VT = N->getValueType(0);
  MVT OpVT = N0.getValueType();

  // fold (uint_to_fp c1) -> c1fp
  if (N0C && OpVT != MVT::ppcf128)
    return DAG.getNode(ISD::UINT_TO_FP, VT, N0);
  
  // If the input is a legal type, and UINT_TO_FP is not legal on this target,
  // but SINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegal(ISD::UINT_TO_FP, OpVT) &&
      TLI.isOperationLegal(ISD::SINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to SINT_TO_FP. 
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::SINT_TO_FP, VT, N0);
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT VT = N->getValueType(0);
  
  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_SINT, VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT VT = N->getValueType(0);
  
  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FP_TO_UINT, VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT VT = N->getValueType(0);
  
  // fold (fp_round c1fp) -> c1fp
  if (N0CFP && N0.getValueType() != MVT::ppcf128)
    return DAG.getNode(ISD::FP_ROUND, VT, N0, N1);
  
  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);
  
  // fold (fp_round (fp_round x)) -> (fp_round x)
  if (N0.getOpcode() == ISD::FP_ROUND) {
    // This is a value preserving truncation if both round's are.
    bool IsTrunc = N->getConstantOperandVal(1) == 1 &&
                   N0.getNode()->getConstantOperandVal(1) == 1;
    return DAG.getNode(ISD::FP_ROUND, VT, N0.getOperand(0),
                       DAG.getIntPtrConstant(IsTrunc));
  }
  
  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse()) {
    SDValue Tmp = DAG.getNode(ISD::FP_ROUND, VT, N0.getOperand(0), N1);
    AddToWorkList(Tmp.getNode());
    return DAG.getNode(ISD::FCOPYSIGN, VT, Tmp, N0.getOperand(1));
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  MVT VT = N->getValueType(0);
  MVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  
  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP && (TLI.isTypeLegal(EVT) || !LegalTypes)) {
    SDValue Round = DAG.getConstantFP(*N0CFP->getConstantFPValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, VT, Round);
  }
  return SDValue();
}

SDValue DAGCombiner::visitFP_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT VT = N->getValueType(0);
  
  // If this is fp_round(fpextend), don't fold it, allow ourselves to be folded.
  if (N->hasOneUse() && 
      N->use_begin().getUse().getSDValue().getOpcode() == ISD::FP_ROUND)
    return SDValue();

  // fold (fp_extend c1fp) -> c1fp
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FP_EXTEND, VT, N0);

  // Turn fp_extend(fp_round(X, 1)) -> x since the fp_round doesn't affect the
  // value of X.
  if (N0.getOpcode() == ISD::FP_ROUND
      && N0.getNode()->getConstantOperandVal(1) == 1) {
    SDValue In = N0.getOperand(0);
    if (In.getValueType() == VT) return In;
    if (VT.bitsLT(In.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, VT, In, N0.getOperand(1));
    return DAG.getNode(ISD::FP_EXTEND, VT, In);
  }
      
  // fold (fpext (load x)) -> (fpext (fptrunc (extload x)))
  if (ISD::isNON_EXTLoad(N0.getNode()) && N0.hasOneUse() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, VT, LN0->getChain(),
                                     LN0->getBasePtr(), LN0->getSrcValue(),
                                     LN0->getSrcValueOffset(),
                                     N0.getValueType(),
                                     LN0->isVolatile(), LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), DAG.getNode(ISD::FP_ROUND, N0.getValueType(),
                                        ExtLoad, DAG.getIntPtrConstant(1)),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  return SDValue();
}

SDValue DAGCombiner::visitFNEG(SDNode *N) {
  SDValue N0 = N->getOperand(0);

  if (isNegatibleForFree(N0, LegalOperations))
    return GetNegatedExpression(N0, DAG, LegalOperations);

  // Transform fneg(bitconvert(x)) -> bitconvert(x^sign) to avoid loading
  // constant pool values.
  if (N0.getOpcode() == ISD::BIT_CONVERT && N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger() &&
      !N0.getOperand(0).getValueType().isVector()) {
    SDValue Int = N0.getOperand(0);
    MVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::XOR, IntVT, Int, 
                        DAG.getConstant(IntVT.getIntegerVTSignBit(), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0), Int);
    }
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitFABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT VT = N->getValueType(0);
  
  // fold (fabs c1) -> fabs(c1)
  if (N0CFP && VT != MVT::ppcf128)
    return DAG.getNode(ISD::FABS, VT, N0);
  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, VT, N0.getOperand(0));
  
  // Transform fabs(bitconvert(x)) -> bitconvert(x&~sign) to avoid loading
  // constant pool values.
  if (N0.getOpcode() == ISD::BIT_CONVERT && N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger() &&
      !N0.getOperand(0).getValueType().isVector()) {
    SDValue Int = N0.getOperand(0);
    MVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::AND, IntVT, Int, 
                        DAG.getConstant(~IntVT.getIntegerVTSignBit(), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0), Int);
    }
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitBRCOND(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // never taken branch, fold to chain
  if (N1C && N1C->isNullValue())
    return Chain;
  // unconditional branch
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N2);
  // fold a brcond with a setcc condition into a BR_CC node if BR_CC is legal
  // on the target.
  if (N1.getOpcode() == ISD::SETCC && 
      TLI.isOperationLegal(ISD::BR_CC, MVT::Other)) {
    return DAG.getNode(ISD::BR_CC, MVT::Other, Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }
  return SDValue();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDValue DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDValue CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);
  
  // Use SimplifySetCC to simplify SETCC's.
  SDValue Simp = SimplifySetCC(TLI.getSetCCResultType(CondLHS.getValueType()),
                               CondLHS, CondRHS, CC->get(), false);
  if (Simp.getNode()) AddToWorkList(Simp.getNode());

  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(Simp.getNode());

  // fold br_cc true, dest -> br dest (unconditional branch)
  if (SCCC && !SCCC->isNullValue())
    return DAG.getNode(ISD::BR, MVT::Other, N->getOperand(0),
                       N->getOperand(4));
  // fold br_cc false, dest -> unconditional fall through
  if (SCCC && SCCC->isNullValue())
    return N->getOperand(0);

  // fold to a simpler setcc
  if (Simp.getNode() && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, MVT::Other, N->getOperand(0), 
                       Simp.getOperand(2), Simp.getOperand(0),
                       Simp.getOperand(1), N->getOperand(4));
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
  MVT VT;
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
  } else
    return false;

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
  if (isa<FrameIndexSDNode>(BasePtr))
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
    Result = DAG.getIndexedLoad(SDValue(N,0), BasePtr, Offset, AM);
  else
    Result = DAG.getIndexedStore(SDValue(N,0), BasePtr, Offset, AM);
  ++PreIndexedNodes;
  ++NodesCombined;
  DOUT << "\nReplacing.4 "; DEBUG(N->dump(&DAG));
  DOUT << "\nWith: "; DEBUG(Result.getNode()->dump(&DAG));
  DOUT << '\n';
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
  MVT VT;
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
  } else
    return false;

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
      if (Ptr == Offset)
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
          ? DAG.getIndexedLoad(SDValue(N,0), BasePtr, Offset, AM)
          : DAG.getIndexedStore(SDValue(N,0), BasePtr, Offset, AM);
        ++PostIndexedNodes;
        ++NodesCombined;
        DOUT << "\nReplacing.5 "; DEBUG(N->dump(&DAG));
        DOUT << "\nWith: "; DEBUG(Result.getNode()->dump(&DAG));
        DOUT << '\n';
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

/// InferAlignment - If we can infer some alignment information from this
/// pointer, return it.
static unsigned InferAlignment(SDValue Ptr, SelectionDAG &DAG) {
  // If this is a direct reference to a stack slot, use information about the
  // stack slot's alignment.
  int FrameIdx = 1 << 31;
  int64_t FrameOffset = 0;
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr)) {
    FrameIdx = FI->getIndex();
  } else if (Ptr.getOpcode() == ISD::ADD && 
             isa<ConstantSDNode>(Ptr.getOperand(1)) &&
             isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
    FrameIdx = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
    FrameOffset = Ptr.getConstantOperandVal(1);
  }
             
  if (FrameIdx != (1 << 31)) {
    // FIXME: Handle FI+CST.
    const MachineFrameInfo &MFI = *DAG.getMachineFunction().getFrameInfo();
    if (MFI.isFixedObjectIndex(FrameIdx)) {
      int64_t ObjectOffset = MFI.getObjectOffset(FrameIdx) + FrameOffset;

      // The alignment of the frame index can be determined from its offset from
      // the incoming frame position.  If the frame object is at offset 32 and
      // the stack is guaranteed to be 16-byte aligned, then we know that the
      // object is 16-byte aligned.
      unsigned StackAlign = DAG.getTarget().getFrameInfo()->getStackAlignment();
      unsigned Align = MinAlign(ObjectOffset, StackAlign);
      
      // Finally, the frame object itself may have a known alignment.  Factor
      // the alignment + offset into a new alignment.  For example, if we know
      // the  FI is 8 byte aligned, but the pointer is 4 off, we really have a
      // 4-byte alignment of the resultant pointer.  Likewise align 4 + 4-byte
      // offset = 4-byte alignment, align 4 + 1-byte offset = align 1, etc.
      unsigned FIInfoAlign = MinAlign(MFI.getObjectAlignment(FrameIdx), 
                                      FrameOffset);
      return std::max(Align, FIInfoAlign);
    }
  }
  
  return 0;
}

SDValue DAGCombiner::visitLOAD(SDNode *N) {
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr   = LD->getBasePtr();
  
  // Try to infer better alignment information than the load already has.
  if (!Fast && LD->isUnindexed()) {
    if (unsigned Align = InferAlignment(Ptr, DAG)) {
      if (Align > LD->getAlignment())
        return DAG.getExtLoad(LD->getExtensionType(), LD->getValueType(0),
                              Chain, Ptr, LD->getSrcValue(),
                              LD->getSrcValueOffset(), LD->getMemoryVT(),
                              LD->isVolatile(), Align);
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
        DOUT << "\nReplacing.6 "; DEBUG(N->dump(&DAG));
        DOUT << "\nWith chain: "; DEBUG(Chain.getNode()->dump(&DAG));
        DOUT << "\n";
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
        SDValue Undef = DAG.getNode(ISD::UNDEF, N->getValueType(0));
        DOUT << "\nReplacing.6 "; DEBUG(N->dump(&DAG));
        DOUT << "\nWith: "; DEBUG(Undef.getNode()->dump(&DAG));
        DOUT << " and 2 other values\n";
        WorkListRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Undef, &DeadNodes);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1),
                                    DAG.getNode(ISD::UNDEF, N->getValueType(1)),
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
        ReplLoad = DAG.getLoad(N->getValueType(0), BetterChain, Ptr,
                               LD->getSrcValue(), LD->getSrcValueOffset(),
                               LD->isVolatile(), LD->getAlignment());
      } else {
        ReplLoad = DAG.getExtLoad(LD->getExtensionType(),
                                  LD->getValueType(0),
                                  BetterChain, Ptr, LD->getSrcValue(),
                                  LD->getSrcValueOffset(),
                                  LD->getMemoryVT(),
                                  LD->isVolatile(), 
                                  LD->getAlignment());
      }

      // Create token factor to keep old chain connected.
      SDValue Token = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                    Chain, ReplLoad.getValue(1));
      
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


SDValue DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();
  
  // Try to infer better alignment information than the store already has.
  if (!Fast && ST->isUnindexed()) {
    if (unsigned Align = InferAlignment(Ptr, DAG)) {
      if (Align > ST->getAlignment())
        return DAG.getTruncStore(Chain, Value, Ptr, ST->getSrcValue(),
                                 ST->getSrcValueOffset(), ST->getMemoryVT(),
                                 ST->isVolatile(), Align);
    }
  }

  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BIT_CONVERT && !ST->isTruncatingStore() &&
      ST->isUnindexed()) {
    unsigned Align = ST->getAlignment();
    MVT SVT = Value.getOperand(0).getValueType();
    unsigned OrigAlign = TLI.getTargetData()->
      getABITypeAlignment(SVT.getTypeForMVT());
    if (Align <= OrigAlign &&
        ((!LegalOperations && !ST->isVolatile()) ||
         TLI.isOperationLegal(ISD::STORE, SVT)))
      return DAG.getStore(Chain, Value.getOperand(0), Ptr, ST->getSrcValue(),
                          ST->getSrcValueOffset(), ST->isVolatile(), OrigAlign);
  }

  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Value)) {
    // NOTE: If the original store is volatile, this transform must not increase
    // the number of stores.  For example, on x86-32 an f64 can be stored in one
    // processor operation but an i64 (which is not legal) requires two.  So the
    // transform should not be done in this case.
    if (Value.getOpcode() != ISD::TargetConstantFP) {
      SDValue Tmp;
      switch (CFP->getValueType(0).getSimpleVT()) {
      default: assert(0 && "Unknown FP type");
      case MVT::f80:    // We don't do this for these yet.
      case MVT::f128:
      case MVT::ppcf128:
        break;
      case MVT::f32:
        if (((TLI.isTypeLegal(MVT::i32) || !LegalTypes) && !LegalOperations &&
             !ST->isVolatile()) || TLI.isOperationLegal(ISD::STORE, MVT::i32)) {
          Tmp = DAG.getConstant((uint32_t)CFP->getValueAPF().
                              bitcastToAPInt().getZExtValue(), MVT::i32);
          return DAG.getStore(Chain, Tmp, Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset(), ST->isVolatile(),
                              ST->getAlignment());
        }
        break;
      case MVT::f64:
        if (((TLI.isTypeLegal(MVT::i64) || !LegalTypes) && !LegalOperations &&
             !ST->isVolatile()) || TLI.isOperationLegal(ISD::STORE, MVT::i64)) {
          Tmp = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                  getZExtValue(), MVT::i64);
          return DAG.getStore(Chain, Tmp, Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset(), ST->isVolatile(),
                              ST->getAlignment());
        } else if (!ST->isVolatile() &&
                   TLI.isOperationLegal(ISD::STORE, MVT::i32)) {
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

          SDValue St0 = DAG.getStore(Chain, Lo, Ptr, ST->getSrcValue(),
                                       ST->getSrcValueOffset(),
                                       isVolatile, ST->getAlignment());
          Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                            DAG.getConstant(4, Ptr.getValueType()));
          SVOffset += 4;
          Alignment = MinAlign(Alignment, 4U);
          SDValue St1 = DAG.getStore(Chain, Hi, Ptr, ST->getSrcValue(),
                                       SVOffset, isVolatile, Alignment);
          return DAG.getNode(ISD::TokenFactor, MVT::Other, St0, St1);
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
      // Replace the chain to avoid dependency.
      SDValue ReplStore;
      if (ST->isTruncatingStore()) {
        ReplStore = DAG.getTruncStore(BetterChain, Value, Ptr,
                                      ST->getSrcValue(),ST->getSrcValueOffset(),
                                      ST->getMemoryVT(),
                                      ST->isVolatile(), ST->getAlignment());
      } else {
        ReplStore = DAG.getStore(BetterChain, Value, Ptr,
                                 ST->getSrcValue(), ST->getSrcValueOffset(),
                                 ST->isVolatile(), ST->getAlignment());
      }
      
      // Create token to keep both nodes around.
      SDValue Token =
        DAG.getNode(ISD::TokenFactor, MVT::Other, Chain, ReplStore);
        
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
      return DAG.getTruncStore(Chain, Shorter, Ptr, ST->getSrcValue(),
                               ST->getSrcValueOffset(), ST->getMemoryVT(),
                               ST->isVolatile(), ST->getAlignment());
    
    // Otherwise, see if we can simplify the operation with
    // SimplifyDemandedBits, which only works if the value has a single use.
    if (SimplifyDemandedBits(Value,
                             APInt::getLowBitsSet(
                               Value.getValueSizeInBits(),
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
    return DAG.getTruncStore(Chain, Value.getOperand(0), Ptr, ST->getSrcValue(),
                             ST->getSrcValueOffset(), ST->getMemoryVT(),
                             ST->isVolatile(), ST->getAlignment());
  }

  return SDValue();
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
    return DAG.getNode(ISD::BUILD_VECTOR, InVec.getValueType(),
                       &Ops[0], Ops.size());
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitEXTRACT_VECTOR_ELT(SDNode *N) {
  // (vextract (scalar_to_vector val, 0) -> val
  SDValue InVec = N->getOperand(0);

 if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR)
   return InVec.getOperand(0);

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
    MVT VT = InVec.getValueType();
    MVT EVT = VT.getVectorElementType();
    MVT LVT = EVT;
    if (InVec.getOpcode() == ISD::BIT_CONVERT) {
      MVT BCVT = InVec.getOperand(0).getValueType();
      if (!BCVT.isVector() || EVT.bitsGT(BCVT.getVectorElementType()))
        return SDValue();
      if (VT.getVectorNumElements() != BCVT.getVectorNumElements())
        BCNumEltsChanged = true;
      InVec = InVec.getOperand(0);
      EVT = BCVT.getVectorElementType();
      NewLoad = true;
    }

    LoadSDNode *LN0 = NULL;
    if (ISD::isNormalLoad(InVec.getNode()))
      LN0 = cast<LoadSDNode>(InVec);
    else if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR &&
             InVec.getOperand(0).getValueType() == EVT &&
             ISD::isNormalLoad(InVec.getOperand(0).getNode())) {
      LN0 = cast<LoadSDNode>(InVec.getOperand(0));
    } else if (InVec.getOpcode() == ISD::VECTOR_SHUFFLE) {
      // (vextract (vector_shuffle (load $addr), v2, <1, u, u, u>), 1)
      // =>
      // (load $addr+1*size)
      
      // If the bit convert changed the number of elements, it is unsafe
      // to examine the mask.
      if (BCNumEltsChanged)
        return SDValue();
      unsigned Idx = cast<ConstantSDNode>(InVec.getOperand(2).
                                          getOperand(Elt))->getZExtValue();
      unsigned NumElems = InVec.getOperand(2).getNumOperands();
      InVec = (Idx < NumElems) ? InVec.getOperand(0) : InVec.getOperand(1);
      if (InVec.getOpcode() == ISD::BIT_CONVERT)
        InVec = InVec.getOperand(0);
      if (ISD::isNormalLoad(InVec.getNode())) {
        LN0 = cast<LoadSDNode>(InVec);
        Elt = (Idx < NumElems) ? Idx : Idx - NumElems;
      }
    }
    if (!LN0 || !LN0->hasOneUse() || LN0->isVolatile())
      return SDValue();

    unsigned Align = LN0->getAlignment();
    if (NewLoad) {
      // Check the resultant load doesn't need a higher alignment than the
      // original load.
      unsigned NewAlign = TLI.getTargetData()->
        getABITypeAlignment(LVT.getTypeForMVT());
      if (NewAlign > Align || !TLI.isOperationLegal(ISD::LOAD, LVT))
        return SDValue();
      Align = NewAlign;
    }

    SDValue NewPtr = LN0->getBasePtr();
    if (Elt) {
      unsigned PtrOff = LVT.getSizeInBits() * Elt / 8;
      MVT PtrType = NewPtr.getValueType();
      if (TLI.isBigEndian())
        PtrOff = VT.getSizeInBits() / 8 - PtrOff;
      NewPtr = DAG.getNode(ISD::ADD, PtrType, NewPtr,
                           DAG.getConstant(PtrOff, PtrType));
    }
    return DAG.getLoad(LVT, LN0->getChain(), NewPtr,
                       LN0->getSrcValue(), LN0->getSrcValueOffset(),
                       LN0->isVolatile(), Align);
  }
  return SDValue();
}
  

SDValue DAGCombiner::visitBUILD_VECTOR(SDNode *N) {
  unsigned NumInScalars = N->getNumOperands();
  MVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  MVT EltType = VT.getVectorElementType();

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
    SmallVector<SDValue, 8> BuildVecIndices;
    for (unsigned i = 0; i != NumInScalars; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::UNDEF) {
        BuildVecIndices.push_back(DAG.getNode(ISD::UNDEF, TLI.getPointerTy()));
        continue;
      }
      
      SDValue Extract = N->getOperand(i);
      
      // If extracting from the first vector, just use the index directly.
      if (Extract.getOperand(0) == VecIn1) {
        BuildVecIndices.push_back(Extract.getOperand(1));
        continue;
      }

      // Otherwise, use InIdx + VecSize
      unsigned Idx =
        cast<ConstantSDNode>(Extract.getOperand(1))->getZExtValue();
      BuildVecIndices.push_back(DAG.getIntPtrConstant(Idx+NumInScalars));
    }
    
    // Add count and size info.
    MVT BuildVecVT = MVT::getVectorVT(TLI.getPointerTy(), NumElts);
    if (!TLI.isTypeLegal(BuildVecVT) && LegalTypes)
      return SDValue();

    // Return the new VECTOR_SHUFFLE node.
    SDValue Ops[5];
    Ops[0] = VecIn1;
    if (VecIn2.getNode()) {
      Ops[1] = VecIn2;
    } else {
      // Use an undef build_vector as input for the second operand.
      std::vector<SDValue> UnOps(NumInScalars,
                                   DAG.getNode(ISD::UNDEF, 
                                               EltType));
      Ops[1] = DAG.getNode(ISD::BUILD_VECTOR, VT,
                           &UnOps[0], UnOps.size());
      AddToWorkList(Ops[1].getNode());
    }
    Ops[2] = DAG.getNode(ISD::BUILD_VECTOR, BuildVecVT,
                         &BuildVecIndices[0], BuildVecIndices.size());
    return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, Ops, 3);
  }
  
  return SDValue();
}

SDValue DAGCombiner::visitCONCAT_VECTORS(SDNode *N) {
  // TODO: Check to see if this is a CONCAT_VECTORS of a bunch of
  // EXTRACT_SUBVECTOR operations.  If so, and if the EXTRACT_SUBVECTOR vector
  // inputs come from at most two distinct vectors, turn this into a shuffle
  // node.

  // If we only have one input vector, we don't need to do any concatenation.
  if (N->getNumOperands() == 1) {
    return N->getOperand(0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  SDValue ShufMask = N->getOperand(2);
  unsigned NumElts = ShufMask.getNumOperands();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  assert(N0.getValueType().getVectorNumElements() == NumElts &&
        "Vector shuffle must be normalized in DAG");

  // If the shuffle mask is an identity operation on the LHS, return the LHS.
  bool isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getZExtValue() != i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity) return N->getOperand(0);

  // If the shuffle mask is an identity operation on the RHS, return the RHS.
  isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getZExtValue() !=
          i+NumElts) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity) return N->getOperand(1);

  // Check if the shuffle is a unary shuffle, i.e. one of the vectors is not
  // needed at all.
  bool isUnary = true;
  bool isSplat = true;
  int VecNum = -1;
  unsigned BaseIdx = 0;
  for (unsigned i = 0; i != NumElts; ++i)
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF) {
      unsigned Idx=cast<ConstantSDNode>(ShufMask.getOperand(i))->getZExtValue();
      int V = (Idx < NumElts) ? 0 : 1;
      if (VecNum == -1) {
        VecNum = V;
        BaseIdx = Idx;
      } else {
        if (BaseIdx != Idx)
          isSplat = false;
        if (VecNum != V) {
          isUnary = false;
          break;
        }
      }
    }

  // Normalize unary shuffle so the RHS is undef.
  if (isUnary && VecNum == 1)
    std::swap(N0, N1);

  // If it is a splat, check if the argument vector is a build_vector with
  // all scalar elements the same.
  if (isSplat) {
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

  // If it is a unary or the LHS and the RHS are the same node, turn the RHS
  // into an undef.
  if (isUnary || N0 == N1) {
    // Check the SHUFFLE mask, mapping any inputs from the 2nd operand into the
    // first operand.
    SmallVector<SDValue, 8> MappedOps;
    for (unsigned i = 0; i != NumElts; ++i) {
      if (ShufMask.getOperand(i).getOpcode() == ISD::UNDEF ||
          cast<ConstantSDNode>(ShufMask.getOperand(i))->getZExtValue() <
            NumElts) {
        MappedOps.push_back(ShufMask.getOperand(i));
      } else {
        unsigned NewIdx = 
          cast<ConstantSDNode>(ShufMask.getOperand(i))->getZExtValue() -
          NumElts;
        MappedOps.push_back(DAG.getConstant(NewIdx,
                                        ShufMask.getOperand(i).getValueType()));
      }
    }
    ShufMask = DAG.getNode(ISD::BUILD_VECTOR, ShufMask.getValueType(),
                           &MappedOps[0], MappedOps.size());
    AddToWorkList(ShufMask.getNode());
    return DAG.getNode(ISD::VECTOR_SHUFFLE, N->getValueType(0),
                       N0,
                       DAG.getNode(ISD::UNDEF, N->getValueType(0)),
                       ShufMask);
  }
 
  return SDValue();
}

/// XformToShuffleWithZero - Returns a vector_shuffle if it able to transform
/// an AND to a vector_shuffle with the destination vector and a zero vector.
/// e.g. AND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDValue DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (N->getOpcode() == ISD::AND) {
    if (RHS.getOpcode() == ISD::BIT_CONVERT)
      RHS = RHS.getOperand(0);
    if (RHS.getOpcode() == ISD::BUILD_VECTOR) {
      std::vector<SDValue> IdxOps;
      unsigned NumOps = RHS.getNumOperands();
      unsigned NumElts = NumOps;
      for (unsigned i = 0; i != NumElts; ++i) {
        SDValue Elt = RHS.getOperand(i);
        if (!isa<ConstantSDNode>(Elt))
          return SDValue();
        else if (cast<ConstantSDNode>(Elt)->isAllOnesValue())
          IdxOps.push_back(DAG.getIntPtrConstant(i));
        else if (cast<ConstantSDNode>(Elt)->isNullValue())
          IdxOps.push_back(DAG.getIntPtrConstant(NumElts));
        else
          return SDValue();
      }

      // Let's see if the target supports this vector_shuffle.
      if (!TLI.isVectorClearMaskLegal(IdxOps, TLI.getPointerTy(), DAG))
        return SDValue();

      // Return the new VECTOR_SHUFFLE node.
      MVT EVT = RHS.getValueType().getVectorElementType();
      MVT VT = MVT::getVectorVT(EVT, NumElts);
      MVT MaskVT = MVT::getVectorVT(TLI.getPointerTy(), NumElts);
      std::vector<SDValue> Ops;
      LHS = DAG.getNode(ISD::BIT_CONVERT, VT, LHS);
      Ops.push_back(LHS);
      AddToWorkList(LHS.getNode());
      std::vector<SDValue> ZeroOps(NumElts, DAG.getConstant(0, EVT));
      Ops.push_back(DAG.getNode(ISD::BUILD_VECTOR, VT,
                                &ZeroOps[0], ZeroOps.size()));
      Ops.push_back(DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                &IdxOps[0], IdxOps.size()));
      SDValue Result = DAG.getNode(ISD::VECTOR_SHUFFLE, VT,
                                     &Ops[0], Ops.size());
      if (VT != N->getValueType(0))
        Result = DAG.getNode(ISD::BIT_CONVERT, N->getValueType(0), Result);
      return Result;
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

  MVT VT = N->getValueType(0);
  assert(VT.isVector() && "SimplifyVBinOp only works on vectors!");

  MVT EltType = VT.getVectorElementType();
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
      Ops.push_back(DAG.getNode(N->getOpcode(), EltType, LHSOp, RHSOp));
      AddToWorkList(Ops.back().getNode());
      assert((Ops.back().getOpcode() == ISD::UNDEF ||
              Ops.back().getOpcode() == ISD::Constant ||
              Ops.back().getOpcode() == ISD::ConstantFP) &&
             "Scalar binop didn't fold!");
    }
    
    if (Ops.size() == LHS.getNumOperands()) {
      MVT VT = LHS.getValueType();
      return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
    }
  }
  
  return SDValue();
}

SDValue DAGCombiner::SimplifySelect(SDValue N0, SDValue N1, SDValue N2){
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");
  
  SDValue SCC = SimplifySelectCC(N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());
  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.getNode()) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      SDValue SETCC = DAG.getNode(ISD::SETCC, N0.getValueType(),
                                    SCC.getOperand(0), SCC.getOperand(1), 
                                    SCC.getOperand(4));
      AddToWorkList(SETCC.getNode());
      return DAG.getNode(ISD::SELECT, SCC.getValueType(), SCC.getOperand(2),
                         SCC.getOperand(3), SETCC);
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
///
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
        // FIXME: this conflates two src values, discarding one.  This is not
        // the right thing to do, but nothing uses srcvalues now.  When they do,
        // turn SrcValue into a list of locations.
        SDValue Addr;
        if (TheSelect->getOpcode() == ISD::SELECT) {
          // Check that the condition doesn't reach either load.  If so, folding
          // this will induce a cycle into the DAG.
          if (!LLD->isPredecessorOf(TheSelect->getOperand(0).getNode()) &&
              !RLD->isPredecessorOf(TheSelect->getOperand(0).getNode())) {
            Addr = DAG.getNode(ISD::SELECT, LLD->getBasePtr().getValueType(),
                               TheSelect->getOperand(0), LLD->getBasePtr(),
                               RLD->getBasePtr());
          }
        } else {
          // Check that the condition doesn't reach either load.  If so, folding
          // this will induce a cycle into the DAG.
          if (!LLD->isPredecessorOf(TheSelect->getOperand(0).getNode()) &&
              !RLD->isPredecessorOf(TheSelect->getOperand(0).getNode()) &&
              !LLD->isPredecessorOf(TheSelect->getOperand(1).getNode()) &&
              !RLD->isPredecessorOf(TheSelect->getOperand(1).getNode())) {
            Addr = DAG.getNode(ISD::SELECT_CC, LLD->getBasePtr().getValueType(),
                             TheSelect->getOperand(0),
                             TheSelect->getOperand(1), 
                             LLD->getBasePtr(), RLD->getBasePtr(),
                             TheSelect->getOperand(4));
          }
        }
        
        if (Addr.getNode()) {
          SDValue Load;
          if (LLD->getExtensionType() == ISD::NON_EXTLOAD)
            Load = DAG.getLoad(TheSelect->getValueType(0), LLD->getChain(),
                               Addr,LLD->getSrcValue(), 
                               LLD->getSrcValueOffset(),
                               LLD->isVolatile(), 
                               LLD->getAlignment());
          else {
            Load = DAG.getExtLoad(LLD->getExtensionType(),
                                  TheSelect->getValueType(0),
                                  LLD->getChain(), Addr, LLD->getSrcValue(),
                                  LLD->getSrcValueOffset(),
                                  LLD->getMemoryVT(),
                                  LLD->isVolatile(), 
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

SDValue DAGCombiner::SimplifySelectCC(SDValue N0, SDValue N1, 
                                      SDValue N2, SDValue N3,
                                      ISD::CondCode CC, bool NotExtCompare) {
  
  MVT VT = N2.getValueType();
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.getNode());

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(TLI.getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, false);
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
        return DAG.getNode(ISD::FABS, VT, N0);
      
      // select (setl[te] X, +/-0.0), fneg(X), X -> fabs
      if ((CC == ISD::SETLT || CC == ISD::SETLE) &&
          N0 == N3 && N2.getOpcode() == ISD::FNEG &&
          N2.getOperand(0) == N3)
        return DAG.getNode(ISD::FABS, VT, N3);
    }
  }
  
  // Check to see if we can perform the "gzip trick", transforming
  // select_cc setlt X, 0, A, 0 -> and (sra X, size(X)-1), A
  if (N1C && N3C && N3C->isNullValue() && CC == ISD::SETLT &&
      N0.getValueType().isInteger() &&
      N2.getValueType().isInteger() &&
      (N1C->isNullValue() ||                         // (a < 0) ? b : 0
       (N1C->getAPIntValue() == 1 && N0 == N2))) {   // (a < 1) ? a : 0
    MVT XType = N0.getValueType();
    MVT AType = N2.getValueType();
    if (XType.bitsGE(AType)) {
      // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
      // single-bit constant.
      if (N2C && ((N2C->getAPIntValue() & (N2C->getAPIntValue()-1)) == 0)) {
        unsigned ShCtV = N2C->getAPIntValue().logBase2();
        ShCtV = XType.getSizeInBits()-ShCtV-1;
        SDValue ShCt = DAG.getConstant(ShCtV, TLI.getShiftAmountTy());
        SDValue Shift = DAG.getNode(ISD::SRL, XType, N0, ShCt);
        AddToWorkList(Shift.getNode());
        if (XType.bitsGT(AType)) {
          Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
          AddToWorkList(Shift.getNode());
        }
        return DAG.getNode(ISD::AND, AType, Shift, N2);
      }
      SDValue Shift = DAG.getNode(ISD::SRA, XType, N0,
                                    DAG.getConstant(XType.getSizeInBits()-1,
                                                    TLI.getShiftAmountTy()));
      AddToWorkList(Shift.getNode());
      if (XType.bitsGT(AType)) {
        Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
        AddToWorkList(Shift.getNode());
      }
      return DAG.getNode(ISD::AND, AType, Shift, N2);
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
      SCC  = DAG.getSetCC(TLI.getSetCCResultType(N0.getValueType()),
                          N0, N1, CC);
      if (N2.getValueType().bitsLT(SCC.getValueType()))
        Temp = DAG.getZeroExtendInReg(SCC, N2.getValueType());
      else
        Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getValueType(), SCC);
    } else {
      SCC  = DAG.getSetCC(MVT::i1, N0, N1, CC);
      Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getValueType(), SCC);
    }
    AddToWorkList(SCC.getNode());
    AddToWorkList(Temp.getNode());
    
    if (N2C->getAPIntValue() == 1)
      return Temp;
    // shl setcc result by log2 n2c
    return DAG.getNode(ISD::SHL, N2.getValueType(), Temp,
                       DAG.getConstant(N2C->getAPIntValue().logBase2(),
                                       TLI.getShiftAmountTy()));
  }
    
  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && N3C && N3C->isNullValue() && N2C && (N2C->getAPIntValue() == 1ULL)) {
    MVT XType = N0.getValueType();
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC, TLI.getSetCCResultType(XType))) {
      SDValue Res = DAG.getSetCC(TLI.getSetCCResultType(XType), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, VT, Res);
      return Res;
    }
    
    // seteq X, 0 -> srl (ctlz X, log2(size(X)))
    if (N1C && N1C->isNullValue() && CC == ISD::SETEQ && 
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::CTLZ, XType))) {
      SDValue Ctlz = DAG.getNode(ISD::CTLZ, XType, N0);
      return DAG.getNode(ISD::SRL, XType, Ctlz, 
                         DAG.getConstant(Log2_32(XType.getSizeInBits()),
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, 0 -> srl (and (-X, ~X), size(X)-1)
    if (N1C && N1C->isNullValue() && CC == ISD::SETGT) { 
      SDValue NegN0 = DAG.getNode(ISD::SUB, XType, DAG.getConstant(0, XType),
                                    N0);
      SDValue NotN0 = DAG.getNode(ISD::XOR, XType, N0, 
                                    DAG.getConstant(~0ULL, XType));
      return DAG.getNode(ISD::SRL, XType, 
                         DAG.getNode(ISD::AND, XType, NegN0, NotN0),
                         DAG.getConstant(XType.getSizeInBits()-1,
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, -1 -> xor (srl (X, size(X)-1), 1)
    if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT) {
      SDValue Sign = DAG.getNode(ISD::SRL, XType, N0,
                                   DAG.getConstant(XType.getSizeInBits()-1,
                                                   TLI.getShiftAmountTy()));
      return DAG.getNode(ISD::XOR, XType, Sign, DAG.getConstant(1, XType));
    }
  }
  
  // Check to see if this is an integer abs. select_cc setl[te] X, 0, -X, X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE) &&
      N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1) &&
      N2.getOperand(0) == N1 && N0.getValueType().isInteger()) {
    MVT XType = N0.getValueType();
    SDValue Shift = DAG.getNode(ISD::SRA, XType, N0,
                                  DAG.getConstant(XType.getSizeInBits()-1,
                                                  TLI.getShiftAmountTy()));
    SDValue Add = DAG.getNode(ISD::ADD, XType, N0, Shift);
    AddToWorkList(Shift.getNode());
    AddToWorkList(Add.getNode());
    return DAG.getNode(ISD::XOR, XType, Add, Shift);
  }
  // Check to see if this is an integer abs. select_cc setgt X, -1, X, -X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT &&
      N0 == N2 && N3.getOpcode() == ISD::SUB && N0 == N3.getOperand(1)) {
    if (ConstantSDNode *SubC = dyn_cast<ConstantSDNode>(N3.getOperand(0))) {
      MVT XType = N0.getValueType();
      if (SubC->isNullValue() && XType.isInteger()) {
        SDValue Shift = DAG.getNode(ISD::SRA, XType, N0,
                                      DAG.getConstant(XType.getSizeInBits()-1,
                                                      TLI.getShiftAmountTy()));
        SDValue Add = DAG.getNode(ISD::ADD, XType, N0, Shift);
        AddToWorkList(Shift.getNode());
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::XOR, XType, Add, Shift);
      }
    }
  }
  
  return SDValue();
}

/// SimplifySetCC - This is a stub for TargetLowering::SimplifySetCC.
SDValue DAGCombiner::SimplifySetCC(MVT VT, SDValue N0,
                                   SDValue N1, ISD::CondCode Cond,
                                   bool foldBooleans) {
  TargetLowering::DAGCombinerInfo 
    DagCombineInfo(DAG, Level == Unrestricted, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo);
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

/// FindBaseOffset - Return true if base is known not to alias with anything
/// but itself.  Provides base object and offset as results.
static bool FindBaseOffset(SDValue Ptr, SDValue &Base, int64_t &Offset) {
  // Assume it is a primitive operation.
  Base = Ptr; Offset = 0;
  
  // If it's an adding a simple constant then integrate the offset.
  if (Base.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Base.getOperand(1))) {
      Base = Base.getOperand(0);
      Offset += C->getZExtValue();
    }
  }
  
  // If it's any of the following then it can't alias with anything but itself.
  return isa<FrameIndexSDNode>(Base) ||
         isa<ConstantPoolSDNode>(Base) ||
         isa<GlobalAddressSDNode>(Base);
}

/// isAlias - Return true if there is any possibility that the two addresses
/// overlap.
bool DAGCombiner::isAlias(SDValue Ptr1, int64_t Size1,
                          const Value *SrcValue1, int SrcValueOffset1,
                          SDValue Ptr2, int64_t Size2,
                          const Value *SrcValue2, int SrcValueOffset2)
{
  // If they are the same then they must be aliases.
  if (Ptr1 == Ptr2) return true;
  
  // Gather base node and offset information.
  SDValue Base1, Base2;
  int64_t Offset1, Offset2;
  bool KnownBase1 = FindBaseOffset(Ptr1, Base1, Offset1);
  bool KnownBase2 = FindBaseOffset(Ptr2, Base2, Offset2);
  
  // If they have a same base address then...
  if (Base1 == Base2) {
    // Check to see if the addresses overlap.
    return!((Offset1 + Size1) <= Offset2 || (Offset2 + Size2) <= Offset1);
  }
  
  // If we know both bases then they can't alias.
  if (KnownBase1 && KnownBase2) return false;

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
                        const Value *&SrcValue, int &SrcValueOffset) {
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    Size = LD->getMemoryVT().getSizeInBits() >> 3;
    SrcValue = LD->getSrcValue();
    SrcValueOffset = LD->getSrcValueOffset();
    return true;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    Size = ST->getMemoryVT().getSizeInBits() >> 3;
    SrcValue = ST->getSrcValue();
    SrcValueOffset = ST->getSrcValueOffset();
  } else {
    assert(0 && "FindAliasInfo expected a memory operand");
  }
  
  return false;
}

/// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDValue OriginalChain,
                                   SmallVector<SDValue, 8> &Aliases) {
  SmallVector<SDValue, 8> Chains;     // List of chains to visit.
  std::set<SDNode *> Visited;           // Visited node set.
  
  // Get alias information for node.
  SDValue Ptr;
  int64_t Size;
  const Value *SrcValue;
  int SrcValueOffset;
  bool IsLoad = FindAliasInfo(N, Ptr, Size, SrcValue, SrcValueOffset);

  // Starting off.
  Chains.push_back(OriginalChain);
  
  // Look at each chain and determine if it is an alias.  If so, add it to the
  // aliases list.  If not, then continue up the chain looking for the next
  // candidate.  
  while (!Chains.empty()) {
    SDValue Chain = Chains.back();
    Chains.pop_back();
    
     // Don't bother if we've been before.
    if (Visited.find(Chain.getNode()) != Visited.end()) continue;
    Visited.insert(Chain.getNode());
  
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
      bool IsOpLoad = FindAliasInfo(Chain.getNode(), OpPtr, OpSize,
                                    OpSrcValue, OpSrcValueOffset);
      
      // If chain is alias then stop here.
      if (!(IsLoad && IsOpLoad) &&
          isAlias(Ptr, Size, SrcValue, SrcValueOffset,
                  OpPtr, OpSize, OpSrcValue, OpSrcValueOffset)) {
        Aliases.push_back(Chain);
      } else {
        // Look further up the chain.
        Chains.push_back(Chain.getOperand(0));      
        // Clean up old chain.
        AddToWorkList(Chain.getNode());
      }
      break;
    }
    
    case ISD::TokenFactor:
      // We have to check each of the operands of the token factor, so we queue
      // then up.  Adding the  operands to the queue (stack) in reverse order
      // maintains the original order and increases the likelihood that getNode
      // will find a matching token factor (CSE.)
      for (unsigned n = Chain.getNumOperands(); n;)
        Chains.push_back(Chain.getOperand(--n));
      // Eliminate the token factor if we can.
      AddToWorkList(Chain.getNode());
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
  SDValue NewChain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                   &Aliases[0], Aliases.size());

  // Make sure the old chain gets cleaned up.
  if (NewChain != OldChain) AddToWorkList(OldChain.getNode());
  
  return NewChain;
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(CombineLevel Level, AliasAnalysis &AA, bool Fast) {
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this, AA, Fast).Run(Level);
}
