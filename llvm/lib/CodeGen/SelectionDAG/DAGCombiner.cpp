//===-- DAGCombiner.cpp - Implement a DAG node combiner -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// FIXME: Missing folds
// sdiv, udiv, srem, urem (X, const) where X is an integer can be expanded into
//  a sequence of multiplies, shifts, and adds.  This should be controlled by
//  some kind of hint from the target that int div is expensive.
// various folds of mulh[s,u] by constants such as -1, powers of 2, etc.
//
// FIXME: select C, pow2, pow2 -> something smart
// FIXME: trunc(select X, Y, Z) -> select X, trunc(Y), trunc(Z)
// FIXME: Dead stores -> nuke
// FIXME: shr X, (and Y,31) -> shr X, Y   (TRICKY!)
// FIXME: mul (x, const) -> shifts + adds
// FIXME: undef values
// FIXME: divide by zero is currently left unfolded.  do we want to turn this
//        into an undef?
// FIXME: select ne (select cc, 1, 0), 0, true, false -> select cc, true, false
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");

namespace {
#ifndef NDEBUG
  static cl::opt<bool>
    ViewDAGCombine1("view-dag-combine1-dags", cl::Hidden,
                    cl::desc("Pop up a window to show dags before the first "
                             "dag combine pass"));
  static cl::opt<bool>
    ViewDAGCombine2("view-dag-combine2-dags", cl::Hidden,
                    cl::desc("Pop up a window to show dags before the second "
                             "dag combine pass"));
#else
  static const bool ViewDAGCombine1 = false;
  static const bool ViewDAGCombine2 = false;
#endif
  
  static cl::opt<bool>
    CombinerAA("combiner-alias-analysis", cl::Hidden,
               cl::desc("Turn on alias analysis during testing"));

  static cl::opt<bool>
    CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
               cl::desc("Include global information in alias analysis"));

//------------------------------ DAGCombiner ---------------------------------//

  class VISIBILITY_HIDDEN DAGCombiner {
    SelectionDAG &DAG;
    TargetLowering &TLI;
    bool AfterLegalize;

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

    /// removeFromWorkList - remove all instances of N from the worklist.
    ///
    void removeFromWorkList(SDNode *N) {
      WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), N),
                     WorkList.end());
    }
    
  public:
    /// AddToWorkList - Add to the work list making sure it's instance is at the
    /// the back (next to be processed.)
    void AddToWorkList(SDNode *N) {
      removeFromWorkList(N);
      WorkList.push_back(N);
    }

    SDOperand CombineTo(SDNode *N, const SDOperand *To, unsigned NumTo,
                        bool AddTo = true) {
      assert(N->getNumValues() == NumTo && "Broken CombineTo call!");
      ++NodesCombined;
      DOUT << "\nReplacing.1 "; DEBUG(N->dump());
      DOUT << "\nWith: "; DEBUG(To[0].Val->dump(&DAG));
      DOUT << " and " << NumTo-1 << " other values\n";
      std::vector<SDNode*> NowDead;
      DAG.ReplaceAllUsesWith(N, To, &NowDead);
      
      if (AddTo) {
        // Push the new nodes and any users onto the worklist
        for (unsigned i = 0, e = NumTo; i != e; ++i) {
          AddToWorkList(To[i].Val);
          AddUsersToWorkList(To[i].Val);
        }
      }
      
      // Nodes can be reintroduced into the worklist.  Make sure we do not
      // process a node that has been replaced.
      removeFromWorkList(N);
      for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
        removeFromWorkList(NowDead[i]);
      
      // Finally, since the node is now dead, remove it from the graph.
      DAG.DeleteNode(N);
      return SDOperand(N, 0);
    }
    
    SDOperand CombineTo(SDNode *N, SDOperand Res, bool AddTo = true) {
      return CombineTo(N, &Res, 1, AddTo);
    }
    
    SDOperand CombineTo(SDNode *N, SDOperand Res0, SDOperand Res1,
                        bool AddTo = true) {
      SDOperand To[] = { Res0, Res1 };
      return CombineTo(N, To, 2, AddTo);
    }
  private:    
    
    /// SimplifyDemandedBits - Check the specified integer node value to see if
    /// it can be simplified or if things it uses can be simplified by bit
    /// propagation.  If so, return true.
    bool SimplifyDemandedBits(SDOperand Op) {
      TargetLowering::TargetLoweringOpt TLO(DAG);
      uint64_t KnownZero, KnownOne;
      uint64_t Demanded = MVT::getIntVTBitMask(Op.getValueType());
      if (!TLI.SimplifyDemandedBits(Op, Demanded, KnownZero, KnownOne, TLO))
        return false;

      // Revisit the node.
      AddToWorkList(Op.Val);
      
      // Replace the old value with the new one.
      ++NodesCombined;
      DOUT << "\nReplacing.2 "; DEBUG(TLO.Old.Val->dump());
      DOUT << "\nWith: "; DEBUG(TLO.New.Val->dump(&DAG));
      DOUT << '\n';

      std::vector<SDNode*> NowDead;
      DAG.ReplaceAllUsesOfValueWith(TLO.Old, TLO.New, NowDead);
      
      // Push the new node and any (possibly new) users onto the worklist.
      AddToWorkList(TLO.New.Val);
      AddUsersToWorkList(TLO.New.Val);
      
      // Nodes can end up on the worklist more than once.  Make sure we do
      // not process a node that has been replaced.
      for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
        removeFromWorkList(NowDead[i]);
      
      // Finally, if the node is now dead, remove it from the graph.  The node
      // may not be dead if the replacement process recursively simplified to
      // something else needing this node.
      if (TLO.Old.Val->use_empty()) {
        removeFromWorkList(TLO.Old.Val);
        
        // If the operands of this node are only used by the node, they will now
        // be dead.  Make sure to visit them first to delete dead nodes early.
        for (unsigned i = 0, e = TLO.Old.Val->getNumOperands(); i != e; ++i)
          if (TLO.Old.Val->getOperand(i).Val->hasOneUse())
            AddToWorkList(TLO.Old.Val->getOperand(i).Val);
        
        DAG.DeleteNode(TLO.Old.Val);
      }
      return true;
    }

    bool CombineToPreIndexedLoadStore(SDNode *N);
    bool CombineToPostIndexedLoadStore(SDNode *N);
    
    
    /// visit - call the node-specific routine that knows how to fold each
    /// particular type of node.
    SDOperand visit(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDOperand.Val == 0   - No change was made
    //   SDOperand.Val == N   - N was replaced, is dead, and is already handled.
    //   otherwise            - N should be replaced by the returned Operand.
    //
    SDOperand visitTokenFactor(SDNode *N);
    SDOperand visitADD(SDNode *N);
    SDOperand visitSUB(SDNode *N);
    SDOperand visitADDC(SDNode *N);
    SDOperand visitADDE(SDNode *N);
    SDOperand visitMUL(SDNode *N);
    SDOperand visitSDIV(SDNode *N);
    SDOperand visitUDIV(SDNode *N);
    SDOperand visitSREM(SDNode *N);
    SDOperand visitUREM(SDNode *N);
    SDOperand visitMULHU(SDNode *N);
    SDOperand visitMULHS(SDNode *N);
    SDOperand visitAND(SDNode *N);
    SDOperand visitOR(SDNode *N);
    SDOperand visitXOR(SDNode *N);
    SDOperand visitVBinOp(SDNode *N, ISD::NodeType IntOp, ISD::NodeType FPOp);
    SDOperand visitSHL(SDNode *N);
    SDOperand visitSRA(SDNode *N);
    SDOperand visitSRL(SDNode *N);
    SDOperand visitCTLZ(SDNode *N);
    SDOperand visitCTTZ(SDNode *N);
    SDOperand visitCTPOP(SDNode *N);
    SDOperand visitSELECT(SDNode *N);
    SDOperand visitSELECT_CC(SDNode *N);
    SDOperand visitSETCC(SDNode *N);
    SDOperand visitSIGN_EXTEND(SDNode *N);
    SDOperand visitZERO_EXTEND(SDNode *N);
    SDOperand visitANY_EXTEND(SDNode *N);
    SDOperand visitSIGN_EXTEND_INREG(SDNode *N);
    SDOperand visitTRUNCATE(SDNode *N);
    SDOperand visitBIT_CONVERT(SDNode *N);
    SDOperand visitVBIT_CONVERT(SDNode *N);
    SDOperand visitFADD(SDNode *N);
    SDOperand visitFSUB(SDNode *N);
    SDOperand visitFMUL(SDNode *N);
    SDOperand visitFDIV(SDNode *N);
    SDOperand visitFREM(SDNode *N);
    SDOperand visitFCOPYSIGN(SDNode *N);
    SDOperand visitSINT_TO_FP(SDNode *N);
    SDOperand visitUINT_TO_FP(SDNode *N);
    SDOperand visitFP_TO_SINT(SDNode *N);
    SDOperand visitFP_TO_UINT(SDNode *N);
    SDOperand visitFP_ROUND(SDNode *N);
    SDOperand visitFP_ROUND_INREG(SDNode *N);
    SDOperand visitFP_EXTEND(SDNode *N);
    SDOperand visitFNEG(SDNode *N);
    SDOperand visitFABS(SDNode *N);
    SDOperand visitBRCOND(SDNode *N);
    SDOperand visitBR_CC(SDNode *N);
    SDOperand visitLOAD(SDNode *N);
    SDOperand visitSTORE(SDNode *N);
    SDOperand visitINSERT_VECTOR_ELT(SDNode *N);
    SDOperand visitVINSERT_VECTOR_ELT(SDNode *N);
    SDOperand visitVBUILD_VECTOR(SDNode *N);
    SDOperand visitVECTOR_SHUFFLE(SDNode *N);
    SDOperand visitVVECTOR_SHUFFLE(SDNode *N);

    SDOperand XformToShuffleWithZero(SDNode *N);
    SDOperand ReassociateOps(unsigned Opc, SDOperand LHS, SDOperand RHS);
    
    bool SimplifySelectOps(SDNode *SELECT, SDOperand LHS, SDOperand RHS);
    SDOperand SimplifyBinOpWithSameOpcodeHands(SDNode *N);
    SDOperand SimplifySelect(SDOperand N0, SDOperand N1, SDOperand N2);
    SDOperand SimplifySelectCC(SDOperand N0, SDOperand N1, SDOperand N2, 
                               SDOperand N3, ISD::CondCode CC, 
                               bool NotExtCompare = false);
    SDOperand SimplifySetCC(MVT::ValueType VT, SDOperand N0, SDOperand N1,
                            ISD::CondCode Cond, bool foldBooleans = true);
    SDOperand ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(SDNode *, MVT::ValueType);
    SDOperand BuildSDIV(SDNode *N);
    SDOperand BuildUDIV(SDNode *N);
    SDNode *MatchRotate(SDOperand LHS, SDOperand RHS);
    SDOperand ReduceLoadWidth(SDNode *N);
    
    /// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDOperand OriginalChain,
                          SmallVector<SDOperand, 8> &Aliases);

    /// isAlias - Return true if there is any possibility that the two addresses
    /// overlap.
    bool isAlias(SDOperand Ptr1, int64_t Size1,
                 const Value *SrcValue1, int SrcValueOffset1,
                 SDOperand Ptr2, int64_t Size2,
                 const Value *SrcValue2, int SrcValueOffset2);
                 
    /// FindAliasInfo - Extracts the relevant alias information from the memory
    /// node.  Returns true if the operand was a load.
    bool FindAliasInfo(SDNode *N,
                       SDOperand &Ptr, int64_t &Size,
                       const Value *&SrcValue, int &SrcValueOffset);
                       
    /// FindBetterChain - Walk up chain skipping non-aliasing memory nodes,
    /// looking for a better chain (aliasing node.)
    SDOperand FindBetterChain(SDNode *N, SDOperand Chain);
    
public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis &A)
      : DAG(D),
        TLI(D.getTargetLoweringInfo()),
        AfterLegalize(false),
        AA(A) {}
    
    /// Run - runs the dag combiner on all nodes in the work list
    void Run(bool RunningAfterLegalize); 
  };
}

//===----------------------------------------------------------------------===//
//  TargetLowering::DAGCombinerInfo implementation
//===----------------------------------------------------------------------===//

void TargetLowering::DAGCombinerInfo::AddToWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->AddToWorkList(N);
}

SDOperand TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, const std::vector<SDOperand> &To) {
  return ((DAGCombiner*)DC)->CombineTo(N, &To[0], To.size());
}

SDOperand TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDOperand Res) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res);
}


SDOperand TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDOperand Res0, SDOperand Res1) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res0, Res1);
}




//===----------------------------------------------------------------------===//


// isSetCCEquivalent - Return true if this node is a setcc, or is a select_cc
// that selects between the values 1 and 0, making it equivalent to a setcc.
// Also, set the incoming LHS, RHS, and CC references to the appropriate 
// nodes based on the type of node we are checking.  This simplifies life a
// bit for the callers.
static bool isSetCCEquivalent(SDOperand N, SDOperand &LHS, SDOperand &RHS,
                              SDOperand &CC) {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }
  if (N.getOpcode() == ISD::SELECT_CC && 
      N.getOperand(2).getOpcode() == ISD::Constant &&
      N.getOperand(3).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(2))->getValue() == 1 &&
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
static bool isOneUseSetCC(SDOperand N) {
  SDOperand N0, N1, N2;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.Val->hasOneUse())
    return true;
  return false;
}

SDOperand DAGCombiner::ReassociateOps(unsigned Opc, SDOperand N0, SDOperand N1){
  MVT::ValueType VT = N0.getValueType();
  // reassoc. (op (op x, c1), y) -> (op (op x, y), c1) iff x+c1 has one use
  // reassoc. (op (op x, c1), c2) -> (op x, (op c1, c2))
  if (N0.getOpcode() == Opc && isa<ConstantSDNode>(N0.getOperand(1))) {
    if (isa<ConstantSDNode>(N1)) {
      SDOperand OpNode = DAG.getNode(Opc, VT, N0.getOperand(1), N1);
      AddToWorkList(OpNode.Val);
      return DAG.getNode(Opc, VT, OpNode, N0.getOperand(0));
    } else if (N0.hasOneUse()) {
      SDOperand OpNode = DAG.getNode(Opc, VT, N0.getOperand(0), N1);
      AddToWorkList(OpNode.Val);
      return DAG.getNode(Opc, VT, OpNode, N0.getOperand(1));
    }
  }
  // reassoc. (op y, (op x, c1)) -> (op (op x, y), c1) iff x+c1 has one use
  // reassoc. (op c2, (op x, c1)) -> (op x, (op c1, c2))
  if (N1.getOpcode() == Opc && isa<ConstantSDNode>(N1.getOperand(1))) {
    if (isa<ConstantSDNode>(N0)) {
      SDOperand OpNode = DAG.getNode(Opc, VT, N1.getOperand(1), N0);
      AddToWorkList(OpNode.Val);
      return DAG.getNode(Opc, VT, OpNode, N1.getOperand(0));
    } else if (N1.hasOneUse()) {
      SDOperand OpNode = DAG.getNode(Opc, VT, N1.getOperand(0), N0);
      AddToWorkList(OpNode.Val);
      return DAG.getNode(Opc, VT, OpNode, N1.getOperand(1));
    }
  }
  return SDOperand();
}

void DAGCombiner::Run(bool RunningAfterLegalize) {
  // set the instance variable, so that the various visit routines may use it.
  AfterLegalize = RunningAfterLegalize;

  // Add all the dag nodes to the worklist.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I)
    WorkList.push_back(I);
  
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());
  
  // The root of the dag may dangle to deleted nodes until the dag combiner is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDOperand());
  
  /// DagCombineInfo - Expose the DAG combiner to the target combiner impls.
  TargetLowering::DAGCombinerInfo 
    DagCombineInfo(DAG, !RunningAfterLegalize, false, this);

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
        AddToWorkList(N->getOperand(i).Val);
      
      DAG.DeleteNode(N);
      continue;
    }
    
    SDOperand RV = visit(N);
    
    // If nothing happened, try a target-specific DAG combine.
    if (RV.Val == 0) {
      assert(N->getOpcode() != ISD::DELETED_NODE &&
             "Node was deleted but visit returned NULL!");
      if (N->getOpcode() >= ISD::BUILTIN_OP_END ||
          TLI.hasTargetDAGCombine((ISD::NodeType)N->getOpcode()))
        RV = TLI.PerformDAGCombine(N, DagCombineInfo);
    }
    
    if (RV.Val) {
      ++NodesCombined;
      // If we get back the same node we passed in, rather than a new node or
      // zero, we know that the node must have defined multiple values and
      // CombineTo was used.  Since CombineTo takes care of the worklist 
      // mechanics for us, we have no work to do in this case.
      if (RV.Val != N) {
        assert(N->getOpcode() != ISD::DELETED_NODE &&
               RV.Val->getOpcode() != ISD::DELETED_NODE &&
               "Node was deleted but visit returned new node!");

        DOUT << "\nReplacing.3 "; DEBUG(N->dump());
        DOUT << "\nWith: "; DEBUG(RV.Val->dump(&DAG));
        DOUT << '\n';
        std::vector<SDNode*> NowDead;
        if (N->getNumValues() == RV.Val->getNumValues())
          DAG.ReplaceAllUsesWith(N, RV.Val, &NowDead);
        else {
          assert(N->getValueType(0) == RV.getValueType() && "Type mismatch");
          SDOperand OpV = RV;
          DAG.ReplaceAllUsesWith(N, &OpV, &NowDead);
        }
          
        // Push the new node and any users onto the worklist
        AddToWorkList(RV.Val);
        AddUsersToWorkList(RV.Val);
          
        // Nodes can be reintroduced into the worklist.  Make sure we do not
        // process a node that has been replaced.
        removeFromWorkList(N);
        for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
          removeFromWorkList(NowDead[i]);
        
        // Finally, since the node is now dead, remove it from the graph.
        DAG.DeleteNode(N);
      }
    }
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());
}

SDOperand DAGCombiner::visit(SDNode *N) {
  switch(N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
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
  case ISD::VBIT_CONVERT:       return visitVBIT_CONVERT(N);
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
  case ISD::VINSERT_VECTOR_ELT: return visitVINSERT_VECTOR_ELT(N);
  case ISD::VBUILD_VECTOR:      return visitVBUILD_VECTOR(N);
  case ISD::VECTOR_SHUFFLE:     return visitVECTOR_SHUFFLE(N);
  case ISD::VVECTOR_SHUFFLE:    return visitVVECTOR_SHUFFLE(N);
  case ISD::VADD:               return visitVBinOp(N, ISD::ADD , ISD::FADD);
  case ISD::VSUB:               return visitVBinOp(N, ISD::SUB , ISD::FSUB);
  case ISD::VMUL:               return visitVBinOp(N, ISD::MUL , ISD::FMUL);
  case ISD::VSDIV:              return visitVBinOp(N, ISD::SDIV, ISD::FDIV);
  case ISD::VUDIV:              return visitVBinOp(N, ISD::UDIV, ISD::UDIV);
  case ISD::VAND:               return visitVBinOp(N, ISD::AND , ISD::AND);
  case ISD::VOR:                return visitVBinOp(N, ISD::OR  , ISD::OR);
  case ISD::VXOR:               return visitVBinOp(N, ISD::XOR , ISD::XOR);
  }
  return SDOperand();
}

/// getInputChainForNode - Given a node, return its input chain if it has one,
/// otherwise return a null sd operand.
static SDOperand getInputChainForNode(SDNode *N) {
  if (unsigned NumOps = N->getNumOperands()) {
    if (N->getOperand(0).getValueType() == MVT::Other)
      return N->getOperand(0);
    else if (N->getOperand(NumOps-1).getValueType() == MVT::Other)
      return N->getOperand(NumOps-1);
    for (unsigned i = 1; i < NumOps-1; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        return N->getOperand(i);
  }
  return SDOperand(0, 0);
}

SDOperand DAGCombiner::visitTokenFactor(SDNode *N) {
  // If N has two operands, where one has an input chain equal to the other,
  // the 'other' chain is redundant.
  if (N->getNumOperands() == 2) {
    if (getInputChainForNode(N->getOperand(0).Val) == N->getOperand(1))
      return N->getOperand(0);
    if (getInputChainForNode(N->getOperand(1).Val) == N->getOperand(0))
      return N->getOperand(1);
  }
  
  
  SmallVector<SDNode *, 8> TFs;   // List of token factors to visit.
  SmallVector<SDOperand, 8> Ops;  // Ops for replacing token factor.
  bool Changed = false;           // If we should replace this token factor.
  
  // Start out with this token factor.
  TFs.push_back(N);
  
  // Iterate through token factors.  The TFs grows when new token factors are
  // encountered.
  for (unsigned i = 0; i < TFs.size(); ++i) {
    SDNode *TF = TFs[i];
    
    // Check each of the operands.
    for (unsigned i = 0, ie = TF->getNumOperands(); i != ie; ++i) {
      SDOperand Op = TF->getOperand(i);
      
      switch (Op.getOpcode()) {
      case ISD::EntryToken:
        // Entry tokens don't need to be added to the list. They are
        // rededundant.
        Changed = true;
        break;
        
      case ISD::TokenFactor:
        if ((CombinerAA || Op.hasOneUse()) &&
            std::find(TFs.begin(), TFs.end(), Op.Val) == TFs.end()) {
          // Queue up for processing.
          TFs.push_back(Op.Val);
          // Clean up in case the token factor is removed.
          AddToWorkList(Op.Val);
          Changed = true;
          break;
        }
        // Fall thru
        
      default:
        // Only add if not there prior.
        if (std::find(Ops.begin(), Ops.end(), Op) == Ops.end())
          Ops.push_back(Op);
        break;
      }
    }
  }

  SDOperand Result;

  // If we've change things around then replace token factor.
  if (Changed) {
    if (Ops.size() == 0) {
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

static
SDOperand combineShlAddConstant(SDOperand N0, SDOperand N1, SelectionDAG &DAG) {
  MVT::ValueType VT = N0.getValueType();
  SDOperand N00 = N0.getOperand(0);
  SDOperand N01 = N0.getOperand(1);
  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N01);
  if (N01C && N00.getOpcode() == ISD::ADD && N00.Val->hasOneUse() &&
      isa<ConstantSDNode>(N00.getOperand(1))) {
    N0 = DAG.getNode(ISD::ADD, VT,
                     DAG.getNode(ISD::SHL, VT, N00.getOperand(0), N01),
                     DAG.getNode(ISD::SHL, VT, N00.getOperand(1), N01));
    return DAG.getNode(ISD::ADD, VT, N0, N1);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitADD(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (add c1, c2) -> c1+c2
  if (N0C && N1C)
    return DAG.getNode(ISD::ADD, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADD, VT, N1, N0);
  // fold (add x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold ((c1-A)+c2) -> (c1+c2)-A
  if (N1C && N0.getOpcode() == ISD::SUB)
    if (ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getOperand(0)))
      return DAG.getNode(ISD::SUB, VT,
                         DAG.getConstant(N1C->getValue()+N0C->getValue(), VT),
                         N0.getOperand(1));
  // reassociate add
  SDOperand RADD = ReassociateOps(ISD::ADD, N0, N1);
  if (RADD.Val != 0)
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

  if (!MVT::isVector(VT) && SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  
  // fold (a+b) -> (a|b) iff a and b share no bits.
  if (MVT::isInteger(VT) && !MVT::isVector(VT)) {
    uint64_t LHSZero, LHSOne;
    uint64_t RHSZero, RHSOne;
    uint64_t Mask = MVT::getIntVTBitMask(VT);
    TLI.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);
    if (LHSZero) {
      TLI.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);
      
      // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
      // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
      if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
          (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
        return DAG.getNode(ISD::OR, VT, N0, N1);
    }
  }

  // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
  if (N0.getOpcode() == ISD::SHL && N0.Val->hasOneUse()) {
    SDOperand Result = combineShlAddConstant(N0, N1, DAG);
    if (Result.Val) return Result;
  }
  if (N1.getOpcode() == ISD::SHL && N1.Val->hasOneUse()) {
    SDOperand Result = combineShlAddConstant(N1, N0, DAG);
    if (Result.Val) return Result;
  }

  return SDOperand();
}

SDOperand DAGCombiner::visitADDC(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // If the flag result is dead, turn this into an ADD.
  if (N->hasNUsesOfValue(0, 1))
    return CombineTo(N, DAG.getNode(ISD::ADD, VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  
  // canonicalize constant to RHS.
  if (N0C && !N1C) {
    SDOperand Ops[] = { N1, N0 };
    return DAG.getNode(ISD::ADDC, N->getVTList(), Ops, 2);
  }
  
  // fold (addc x, 0) -> x + no carry out
  if (N1C && N1C->isNullValue())
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  
  // fold (addc a, b) -> (or a, b), CARRY_FALSE iff a and b share no bits.
  uint64_t LHSZero, LHSOne;
  uint64_t RHSZero, RHSOne;
  uint64_t Mask = MVT::getIntVTBitMask(VT);
  TLI.ComputeMaskedBits(N0, Mask, LHSZero, LHSOne);
  if (LHSZero) {
    TLI.ComputeMaskedBits(N1, Mask, RHSZero, RHSOne);
    
    // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
    // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
    if ((RHSZero & (~LHSZero & Mask)) == (~LHSZero & Mask) ||
        (LHSZero & (~RHSZero & Mask)) == (~RHSZero & Mask))
      return CombineTo(N, DAG.getNode(ISD::OR, VT, N0, N1),
                       DAG.getNode(ISD::CARRY_FALSE, MVT::Flag));
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitADDE(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand CarryIn = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  //MVT::ValueType VT = N0.getValueType();
  
  // canonicalize constant to RHS
  if (N0C && !N1C) {
    SDOperand Ops[] = { N1, N0, CarryIn };
    return DAG.getNode(ISD::ADDE, N->getVTList(), Ops, 3);
  }
  
  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE) {
    SDOperand Ops[] = { N1, N0 };
    return DAG.getNode(ISD::ADDC, N->getVTList(), Ops, 2);
  }
  
  return SDOperand();
}



SDOperand DAGCombiner::visitSUB(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (sub x, x) -> 0
  if (N0 == N1)
    return DAG.getConstant(0, N->getValueType(0));
  // fold (sub c1, c2) -> c1-c2
  if (N0C && N1C)
    return DAG.getNode(ISD::SUB, VT, N0, N1);
  // fold (sub x, c) -> (add x, -c)
  if (N1C)
    return DAG.getNode(ISD::ADD, VT, N0, DAG.getConstant(-N1C->getValue(), VT));
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitMUL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (mul c1, c2) -> c1*c2
  if (N0C && N1C)
    return DAG.getNode(ISD::MUL, VT, N0, N1);
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
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SHL, VT, N0,
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1C && isPowerOf2_64(-N1C->getSignExtended())) {
    // FIXME: If the input is something that is easily negated (e.g. a 
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT),
                       DAG.getNode(ISD::SHL, VT, N0,
                            DAG.getConstant(Log2_64(-N1C->getSignExtended()),
                                            TLI.getShiftAmountTy())));
  }

  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      isa<ConstantSDNode>(N0.getOperand(1))) {
    SDOperand C3 = DAG.getNode(ISD::SHL, VT, N1, N0.getOperand(1));
    AddToWorkList(C3.Val);
    return DAG.getNode(ISD::MUL, VT, N0.getOperand(0), C3);
  }
  
  // Change (mul (shl X, C), Y) -> (shl (mul X, Y), C) when the shift has one
  // use.
  {
    SDOperand Sh(0,0), Y(0,0);
    // Check for both (mul (shl X, C), Y)  and  (mul Y, (shl X, C)).
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.Val->hasOneUse()) {
      Sh = N0; Y = N1;
    } else if (N1.getOpcode() == ISD::SHL && 
               isa<ConstantSDNode>(N1.getOperand(1)) && N1.Val->hasOneUse()) {
      Sh = N1; Y = N0;
    }
    if (Sh.Val) {
      SDOperand Mul = DAG.getNode(ISD::MUL, VT, Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, VT, Mul, Sh.getOperand(1));
    }
  }
  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (N1C && N0.getOpcode() == ISD::ADD && N0.Val->hasOneUse() && 
      isa<ConstantSDNode>(N0.getOperand(1))) {
    return DAG.getNode(ISD::ADD, VT, 
                       DAG.getNode(ISD::MUL, VT, N0.getOperand(0), N1),
                       DAG.getNode(ISD::MUL, VT, N0.getOperand(1), N1));
  }
  
  // reassociate mul
  SDOperand RMUL = ReassociateOps(ISD::MUL, N0, N1);
  if (RMUL.Val != 0)
    return RMUL;
  return SDOperand();
}

SDOperand DAGCombiner::visitSDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N->getValueType(0);

  // fold (sdiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getNode(ISD::SDIV, VT, N0, N1);
  // fold (sdiv X, 1) -> X
  if (N1C && N1C->getSignExtended() == 1LL)
    return N0;
  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), N0);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  uint64_t SignBit = 1ULL << (MVT::getSizeInBits(VT)-1);
  if (TLI.MaskedValueIsZero(N1, SignBit) &&
      TLI.MaskedValueIsZero(N0, SignBit))
    return DAG.getNode(ISD::UDIV, N1.getValueType(), N0, N1);
  // fold (sdiv X, pow2) -> simple ops after legalize
  if (N1C && N1C->getValue() && !TLI.isIntDivCheap() &&
      (isPowerOf2_64(N1C->getSignExtended()) || 
       isPowerOf2_64(-N1C->getSignExtended()))) {
    // If dividing by powers of two is cheap, then don't perform the following
    // fold.
    if (TLI.isPow2DivCheap())
      return SDOperand();
    int64_t pow2 = N1C->getSignExtended();
    int64_t abs2 = pow2 > 0 ? pow2 : -pow2;
    unsigned lg2 = Log2_64(abs2);
    // Splat the sign bit into the register
    SDOperand SGN = DAG.getNode(ISD::SRA, VT, N0,
                                DAG.getConstant(MVT::getSizeInBits(VT)-1,
                                                TLI.getShiftAmountTy()));
    AddToWorkList(SGN.Val);
    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDOperand SRL = DAG.getNode(ISD::SRL, VT, SGN,
                                DAG.getConstant(MVT::getSizeInBits(VT)-lg2,
                                                TLI.getShiftAmountTy()));
    SDOperand ADD = DAG.getNode(ISD::ADD, VT, N0, SRL);
    AddToWorkList(SRL.Val);
    AddToWorkList(ADD.Val);    // Divide by pow2
    SDOperand SRA = DAG.getNode(ISD::SRA, VT, ADD,
                                DAG.getConstant(lg2, TLI.getShiftAmountTy()));
    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (pow2 > 0)
      return SRA;
    AddToWorkList(SRA.Val);
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), SRA);
  }
  // if integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.
  if (N1C && (N1C->getSignExtended() < -1 || N1C->getSignExtended() > 1) && 
      !TLI.isIntDivCheap()) {
    SDOperand Op = BuildSDIV(N);
    if (Op.Val) return Op;
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitUDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (udiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getNode(ISD::UDIV, VT, N0, N1);
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SRL, VT, N0, 
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (isPowerOf2_64(SHC->getValue())) {
        MVT::ValueType ADDVT = N1.getOperand(1).getValueType();
        SDOperand Add = DAG.getNode(ISD::ADD, ADDVT, N1.getOperand(1),
                                    DAG.getConstant(Log2_64(SHC->getValue()),
                                                    ADDVT));
        AddToWorkList(Add.Val);
        return DAG.getNode(ISD::SRL, VT, N0, Add);
      }
    }
  }
  // fold (udiv x, c) -> alternate
  if (N1C && N1C->getValue() && !TLI.isIntDivCheap()) {
    SDOperand Op = BuildUDIV(N);
    if (Op.Val) return Op;
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (srem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getNode(ISD::SREM, VT, N0, N1);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
  uint64_t SignBit = 1ULL << (MVT::getSizeInBits(VT)-1);
  if (TLI.MaskedValueIsZero(N1, SignBit) &&
      TLI.MaskedValueIsZero(N0, SignBit))
    return DAG.getNode(ISD::UREM, VT, N0, N1);
  
  // Unconditionally lower X%C -> X-X/C*C.  This allows the X/C logic to hack on
  // the remainder operation.
  if (N1C && !N1C->isNullValue()) {
    SDOperand Div = DAG.getNode(ISD::SDIV, VT, N0, N1);
    SDOperand Mul = DAG.getNode(ISD::MUL, VT, Div, N1);
    SDOperand Sub = DAG.getNode(ISD::SUB, VT, N0, Mul);
    AddToWorkList(Div.Val);
    AddToWorkList(Mul.Val);
    return Sub;
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitUREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (urem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getNode(ISD::UREM, VT, N0, N1);
  // fold (urem x, pow2) -> (and x, pow2-1)
  if (N1C && !N1C->isNullValue() && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::AND, VT, N0, DAG.getConstant(N1C->getValue()-1,VT));
  // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (isPowerOf2_64(SHC->getValue())) {
        SDOperand Add = DAG.getNode(ISD::ADD, VT, N1,DAG.getConstant(~0ULL,VT));
        AddToWorkList(Add.Val);
        return DAG.getNode(ISD::AND, VT, N0, Add);
      }
    }
  }
  
  // Unconditionally lower X%C -> X-X/C*C.  This allows the X/C logic to hack on
  // the remainder operation.
  if (N1C && !N1C->isNullValue()) {
    SDOperand Div = DAG.getNode(ISD::UDIV, VT, N0, N1);
    SDOperand Mul = DAG.getNode(ISD::MUL, VT, Div, N1);
    SDOperand Sub = DAG.getNode(ISD::SUB, VT, N0, Mul);
    AddToWorkList(Div.Val);
    AddToWorkList(Mul.Val);
    return Sub;
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHS(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0, 
                       DAG.getConstant(MVT::getSizeInBits(N0.getValueType())-1,
                                       TLI.getShiftAmountTy()));
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHU(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhu x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhu x, 1) -> 0
  if (N1C && N1C->getValue() == 1)
    return DAG.getConstant(0, N0.getValueType());
  return SDOperand();
}

/// SimplifyBinOpWithSameOpcodeHands - If this is a binary operator with
/// two operands of the same opcode, try to simplify it.
SDOperand DAGCombiner::SimplifyBinOpWithSameOpcodeHands(SDNode *N) {
  SDOperand N0 = N->getOperand(0), N1 = N->getOperand(1);
  MVT::ValueType VT = N0.getValueType();
  assert(N0.getOpcode() == N1.getOpcode() && "Bad input!");
  
  // For each of OP in AND/OR/XOR:
  // fold (OP (zext x), (zext y)) -> (zext (OP x, y))
  // fold (OP (sext x), (sext y)) -> (sext (OP x, y))
  // fold (OP (aext x), (aext y)) -> (aext (OP x, y))
  // fold (OP (trunc x), (trunc y)) -> (trunc (OP x, y))
  if ((N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND||
       N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::TRUNCATE) &&
      N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
    SDOperand ORNode = DAG.getNode(N->getOpcode(), 
                                   N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.Val);
    return DAG.getNode(N0.getOpcode(), VT, ORNode);
  }
  
  // For each of OP in SHL/SRL/SRA/AND...
  //   fold (and (OP x, z), (OP y, z)) -> (OP (and x, y), z)
  //   fold (or  (OP x, z), (OP y, z)) -> (OP (or  x, y), z)
  //   fold (xor (OP x, z), (OP y, z)) -> (OP (xor x, y), z)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL ||
       N0.getOpcode() == ISD::SRA || N0.getOpcode() == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDOperand ORNode = DAG.getNode(N->getOpcode(),
                                   N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.Val);
    return DAG.getNode(N0.getOpcode(), VT, ORNode, N0.getOperand(1));
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitAND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  
  // fold (and c1, c2) -> c1&c2
  if (N0C && N1C)
    return DAG.getNode(ISD::AND, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::AND, VT, N1, N0);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && TLI.MaskedValueIsZero(SDOperand(N, 0), MVT::getIntVTBitMask(VT)))
    return DAG.getConstant(0, VT);
  // reassociate and
  SDOperand RAND = ReassociateOps(ISD::AND, N0, N1);
  if (RAND.Val != 0)
    return RAND;
  // fold (and (or x, 0xFFFF), 0xFF) -> 0xFF
  if (N1C && N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getValue() & N1C->getValue()) == N1C->getValue())
        return N1;
  // fold (and (any_ext V), c) -> (zero_ext V) if 'and' only clears top bits.
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    unsigned InMask = MVT::getIntVTBitMask(N0.getOperand(0).getValueType());
    if (TLI.MaskedValueIsZero(N0.getOperand(0),
                              ~N1C->getValue() & InMask)) {
      SDOperand Zext = DAG.getNode(ISD::ZERO_EXTEND, N0.getValueType(),
                                   N0.getOperand(0));
      
      // Replace uses of the AND with uses of the Zero extend node.
      CombineTo(N, Zext);
      
      // We actually want to replace all uses of the any_extend with the
      // zero_extend, to avoid duplicating things.  This will later cause this
      // AND to be folded.
      CombineTo(N0.Val, Zext);
      return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (and (setcc x), (setcc y)) -> (setcc (and x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();
    
    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        MVT::isInteger(LL.getValueType())) {
      // fold (X == 0) & (Y == 0) -> (X|Y == 0)
      if (cast<ConstantSDNode>(LR)->getValue() == 0 && Op1 == ISD::SETEQ) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X == -1) & (Y == -1) -> (X&Y == -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETEQ) {
        SDOperand ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.Val);
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
      }
      // fold (X >  -1) & (Y >  -1) -> (X|Y > -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETGT) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = MVT::isInteger(LL.getValueType());
      ISD::CondCode Result = ISD::getSetCCAndOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID)
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }

  // Simplify: and (op x...), (op y...)  -> (op (and x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDOperand Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.Val) return Tmp;
  }
  
  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  // fold (and (sra)) -> (and (srl)) when possible.
  if (!MVT::isVector(VT) &&
      SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  // fold (zext_inreg (extload x)) -> (zextload x)
  if (ISD::isEXTLoad(N0.Val) && ISD::isUNINDEXEDLoad(N0.Val)) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType EVT = LN0->getLoadedVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    if (TLI.MaskedValueIsZero(N1, ~0ULL << MVT::getSizeInBits(EVT)) &&
        (!AfterLegalize || TLI.isLoadXLegal(ISD::ZEXTLOAD, EVT))) {
      SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                         LN0->getBasePtr(), LN0->getSrcValue(),
                                         LN0->getSrcValueOffset(), EVT,
                                         LN0->isVolatile(), 
                                         LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
      return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (ISD::isSEXTLoad(N0.Val) && ISD::isUNINDEXEDLoad(N0.Val) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType EVT = LN0->getLoadedVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    if (TLI.MaskedValueIsZero(N1, ~0ULL << MVT::getSizeInBits(EVT)) &&
        (!AfterLegalize || TLI.isLoadXLegal(ISD::ZEXTLOAD, EVT))) {
      SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                         LN0->getBasePtr(), LN0->getSrcValue(),
                                         LN0->getSrcValueOffset(), EVT,
                                         LN0->isVolatile(), 
                                         LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
      return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  
  // fold (and (load x), 255) -> (zextload x, i8)
  // fold (and (extload x, i16), 255) -> (zextload x, i8)
  if (N1C && N0.getOpcode() == ISD::LOAD) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    if (LN0->getExtensionType() != ISD::SEXTLOAD &&
        LN0->getAddressingMode() == ISD::UNINDEXED &&
        N0.hasOneUse()) {
      MVT::ValueType EVT, LoadedVT;
      if (N1C->getValue() == 255)
        EVT = MVT::i8;
      else if (N1C->getValue() == 65535)
        EVT = MVT::i16;
      else if (N1C->getValue() == ~0U)
        EVT = MVT::i32;
      else
        EVT = MVT::Other;
    
      LoadedVT = LN0->getLoadedVT();
      if (EVT != MVT::Other && LoadedVT > EVT &&
          (!AfterLegalize || TLI.isLoadXLegal(ISD::ZEXTLOAD, EVT))) {
        MVT::ValueType PtrType = N0.getOperand(1).getValueType();
        // For big endian targets, we need to add an offset to the pointer to
        // load the correct bytes.  For little endian systems, we merely need to
        // read fewer bytes from the same pointer.
        unsigned PtrOff =
          (MVT::getSizeInBits(LoadedVT) - MVT::getSizeInBits(EVT)) / 8;
        SDOperand NewPtr = LN0->getBasePtr();
        if (!TLI.isLittleEndian())
          NewPtr = DAG.getNode(ISD::ADD, PtrType, NewPtr,
                               DAG.getConstant(PtrOff, PtrType));
        AddToWorkList(NewPtr.Val);
        SDOperand Load =
          DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(), NewPtr,
                         LN0->getSrcValue(), LN0->getSrcValueOffset(), EVT,
                         LN0->isVolatile(), LN0->getAlignment());
        AddToWorkList(N);
        CombineTo(N0.Val, Load, Load.getValue(1));
        return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.getNode(ISD::OR, VT, N0, N1);
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
  if (N1C && 
      TLI.MaskedValueIsZero(N0,~N1C->getValue() & (~0ULL>>(64-OpSizeInBits))))
    return N1;
  // reassociate or
  SDOperand ROR = ReassociateOps(ISD::OR, N0, N1);
  if (ROR.Val != 0)
    return ROR;
  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  if (N1C && N0.getOpcode() == ISD::AND && N0.Val->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    return DAG.getNode(ISD::AND, VT, DAG.getNode(ISD::OR, VT, N0.getOperand(0),
                                                 N1),
                       DAG.getConstant(N1C->getValue() | C1->getValue(), VT));
  }
  // fold (or (setcc x), (setcc y)) -> (setcc (or x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();
    
    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        MVT::isInteger(LL.getValueType())) {
      // fold (X != 0) | (Y != 0) -> (X|Y != 0)
      // fold (X <  0) | (Y <  0) -> (X|Y < 0)
      if (cast<ConstantSDNode>(LR)->getValue() == 0 && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETLT)) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X != -1) | (Y != -1) -> (X&Y != -1)
      // fold (X >  -1) | (Y >  -1) -> (X&Y >  -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDOperand ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.Val);
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = MVT::isInteger(LL.getValueType());
      ISD::CondCode Result = ISD::getSetCCOrOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID)
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }
  
  // Simplify: or (op x...), (op y...)  -> (op (or x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDOperand Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.Val) return Tmp;
  }
  
  // (X & C1) | (Y & C2)  -> (X|Y) & C3  if possible.
  if (N0.getOpcode() == ISD::AND &&
      N1.getOpcode() == ISD::AND &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      N1.getOperand(1).getOpcode() == ISD::Constant &&
      // Don't increase # computations.
      (N0.Val->hasOneUse() || N1.Val->hasOneUse())) {
    // We can only do this xform if we know that bits from X that are set in C2
    // but not in C1 are already zero.  Likewise for Y.
    uint64_t LHSMask = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t RHSMask = cast<ConstantSDNode>(N1.getOperand(1))->getValue();
    
    if (TLI.MaskedValueIsZero(N0.getOperand(0), RHSMask&~LHSMask) &&
        TLI.MaskedValueIsZero(N1.getOperand(0), LHSMask&~RHSMask)) {
      SDOperand X =DAG.getNode(ISD::OR, VT, N0.getOperand(0), N1.getOperand(0));
      return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(LHSMask|RHSMask, VT));
    }
  }
  
  
  // See if this is some rotate idiom.
  if (SDNode *Rot = MatchRotate(N0, N1))
    return SDOperand(Rot, 0);

  return SDOperand();
}


/// MatchRotateHalf - Match "(X shl/srl V1) & V2" where V2 may not be present.
static bool MatchRotateHalf(SDOperand Op, SDOperand &Shift, SDOperand &Mask) {
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
SDNode *DAGCombiner::MatchRotate(SDOperand LHS, SDOperand RHS) {
  // Must be a legal type.  Expanded an promoted things won't work with rotates.
  MVT::ValueType VT = LHS.getValueType();
  if (!TLI.isTypeLegal(VT)) return 0;

  // The target must have at least one rotate flavor.
  bool HasROTL = TLI.isOperationLegal(ISD::ROTL, VT);
  bool HasROTR = TLI.isOperationLegal(ISD::ROTR, VT);
  if (!HasROTL && !HasROTR) return 0;
  
  // Match "(X shl/srl V1) & V2" where V2 may not be present.
  SDOperand LHSShift;   // The shift.
  SDOperand LHSMask;    // AND value if any.
  if (!MatchRotateHalf(LHS, LHSShift, LHSMask))
    return 0; // Not part of a rotate.

  SDOperand RHSShift;   // The shift.
  SDOperand RHSMask;    // AND value if any.
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

  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  SDOperand LHSShiftArg = LHSShift.getOperand(0);
  SDOperand LHSShiftAmt = LHSShift.getOperand(1);
  SDOperand RHSShiftAmt = RHSShift.getOperand(1);

  // fold (or (shl x, C1), (srl x, C2)) -> (rotl x, C1)
  // fold (or (shl x, C1), (srl x, C2)) -> (rotr x, C2)
  if (LHSShiftAmt.getOpcode() == ISD::Constant &&
      RHSShiftAmt.getOpcode() == ISD::Constant) {
    uint64_t LShVal = cast<ConstantSDNode>(LHSShiftAmt)->getValue();
    uint64_t RShVal = cast<ConstantSDNode>(RHSShiftAmt)->getValue();
    if ((LShVal + RShVal) != OpSizeInBits)
      return 0;

    SDOperand Rot;
    if (HasROTL)
      Rot = DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt);
    else
      Rot = DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt);
    
    // If there is an AND of either shifted operand, apply it to the result.
    if (LHSMask.Val || RHSMask.Val) {
      uint64_t Mask = MVT::getIntVTBitMask(VT);
      
      if (LHSMask.Val) {
        uint64_t RHSBits = (1ULL << LShVal)-1;
        Mask &= cast<ConstantSDNode>(LHSMask)->getValue() | RHSBits;
      }
      if (RHSMask.Val) {
        uint64_t LHSBits = ~((1ULL << (OpSizeInBits-RShVal))-1);
        Mask &= cast<ConstantSDNode>(RHSMask)->getValue() | LHSBits;
      }
        
      Rot = DAG.getNode(ISD::AND, VT, Rot, DAG.getConstant(Mask, VT));
    }
    
    return Rot.Val;
  }
  
  // If there is a mask here, and we have a variable shift, we can't be sure
  // that we're masking out the right stuff.
  if (LHSMask.Val || RHSMask.Val)
    return 0;
  
  // fold (or (shl x, y), (srl x, (sub 32, y))) -> (rotl x, y)
  // fold (or (shl x, y), (srl x, (sub 32, y))) -> (rotr x, (sub 32, y))
  if (RHSShiftAmt.getOpcode() == ISD::SUB &&
      LHSShiftAmt == RHSShiftAmt.getOperand(1)) {
    if (ConstantSDNode *SUBC = 
          dyn_cast<ConstantSDNode>(RHSShiftAmt.getOperand(0))) {
      if (SUBC->getValue() == OpSizeInBits)
        if (HasROTL)
          return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).Val;
        else
          return DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt).Val;
    }
  }
  
  // fold (or (shl x, (sub 32, y)), (srl x, r)) -> (rotr x, y)
  // fold (or (shl x, (sub 32, y)), (srl x, r)) -> (rotl x, (sub 32, y))
  if (LHSShiftAmt.getOpcode() == ISD::SUB &&
      RHSShiftAmt == LHSShiftAmt.getOperand(1)) {
    if (ConstantSDNode *SUBC = 
          dyn_cast<ConstantSDNode>(LHSShiftAmt.getOperand(0))) {
      if (SUBC->getValue() == OpSizeInBits)
        if (HasROTL)
          return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).Val;
        else
          return DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt).Val;
    }
  }

  // Look for sign/zext/any-extended cases:
  if ((LHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND
       || LHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND
       || LHSShiftAmt.getOpcode() == ISD::ANY_EXTEND) &&
      (RHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND
       || RHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND
       || RHSShiftAmt.getOpcode() == ISD::ANY_EXTEND)) {
    SDOperand LExtOp0 = LHSShiftAmt.getOperand(0);
    SDOperand RExtOp0 = RHSShiftAmt.getOperand(0);
    if (RExtOp0.getOpcode() == ISD::SUB &&
        RExtOp0.getOperand(1) == LExtOp0) {
      // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
      //   (rotr x, y)
      // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
      //   (rotl x, (sub 32, y))
      if (ConstantSDNode *SUBC = cast<ConstantSDNode>(RExtOp0.getOperand(0))) {
        if (SUBC->getValue() == OpSizeInBits) {
          if (HasROTL)
            return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).Val;
          else
            return DAG.getNode(ISD::ROTR, VT, LHSShiftArg, RHSShiftAmt).Val;
        }
      }
    } else if (LExtOp0.getOpcode() == ISD::SUB &&
               RExtOp0 == LExtOp0.getOperand(1)) {
      // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext r))) -> 
      //   (rotl x, y)
      // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext r))) ->
      //   (rotr x, (sub 32, y))
      if (ConstantSDNode *SUBC = cast<ConstantSDNode>(LExtOp0.getOperand(0))) {
        if (SUBC->getValue() == OpSizeInBits) {
          if (HasROTL)
            return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, RHSShiftAmt).Val;
          else
            return DAG.getNode(ISD::ROTL, VT, LHSShiftArg, LHSShiftAmt).Val;
        }
      }
    }
  }
  
  return 0;
}


SDOperand DAGCombiner::visitXOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LHS, RHS, CC;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (xor c1, c2) -> c1^c2
  if (N0C && N1C)
    return DAG.getNode(ISD::XOR, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::XOR, VT, N1, N0);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // reassociate xor
  SDOperand RXOR = ReassociateOps(ISD::XOR, N0, N1);
  if (RXOR.Val != 0)
    return RXOR;
  // fold !(x cc y) -> (x !cc y)
  if (N1C && N1C->getValue() == 1 && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = MVT::isInteger(LHS.getValueType());
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);
    if (N0.getOpcode() == ISD::SETCC)
      return DAG.getSetCC(VT, LHS, RHS, NotCC);
    if (N0.getOpcode() == ISD::SELECT_CC)
      return DAG.getSelectCC(LHS, RHS, N0.getOperand(2),N0.getOperand(3),NotCC);
    assert(0 && "Unhandled SetCC Equivalent!");
    abort();
  }
  // fold !(x or y) -> (!x and !y) iff x or y are setcc
  if (N1C && N1C->getValue() == 1 && VT == MVT::i1 &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      AddToWorkList(LHS.Val); AddToWorkList(RHS.Val);
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold !(x or y) -> (!x and !y) iff x or y are constants
  if (N1C && N1C->isAllOnesValue() && 
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      AddToWorkList(LHS.Val); AddToWorkList(RHS.Val);
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold (xor (xor x, c1), c2) -> (xor x, c1^c2)
  if (N1C && N0.getOpcode() == ISD::XOR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()^N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()^N01C->getValue(), VT));
  }
  // fold (xor x, x) -> 0
  if (N0 == N1) {
    if (!MVT::isVector(VT)) {
      return DAG.getConstant(0, VT);
    } else if (!AfterLegalize || TLI.isOperationLegal(ISD::BUILD_VECTOR, VT)) {
      // Produce a vector of zeros.
      SDOperand El = DAG.getConstant(0, MVT::getVectorBaseType(VT));
      std::vector<SDOperand> Ops(MVT::getVectorNumElements(VT), El);
      return DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
    }
  }
  
  // Simplify: xor (op x...), (op y...)  -> (op (xor x, y))
  if (N0.getOpcode() == N1.getOpcode()) {
    SDOperand Tmp = SimplifyBinOpWithSameOpcodeHands(N);
    if (Tmp.Val) return Tmp;
  }
  
  // Simplify the expression using non-local knowledge.
  if (!MVT::isVector(VT) &&
      SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  
  return SDOperand();
}

SDOperand DAGCombiner::visitSHL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (shl c1, c2) -> c1<<c2
  if (N0C && N1C)
    return DAG.getNode(ISD::SHL, VT, N0, N1);
  // fold (shl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (shl x, c) is known to be zero, return 0
  if (TLI.MaskedValueIsZero(SDOperand(N, 0), MVT::getIntVTBitMask(VT)))
    return DAG.getConstant(0, VT);
  if (N1C && SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  // fold (shl (shl x, c1), c2) -> 0 or (shl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, -1 << c1), c2-c1) or
  //                               (srl (and x, -1 << c1), c1-c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    SDOperand Mask = DAG.getNode(ISD::AND, VT, N0.getOperand(0),
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
                       DAG.getConstant(~0ULL << N1C->getValue(), VT));
  return SDOperand();
}

SDOperand DAGCombiner::visitSRA(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (sra c1, c2) -> c1>>c2
  if (N0C && N1C)
    return DAG.getNode(ISD::SRA, VT, N0, N1);
  // fold (sra 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (sra -1, x) -> -1
  if (N0C && N0C->isAllOnesValue())
    return N0;
  // fold (sra x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= MVT::getSizeInBits(VT))
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (sra (shl x, c1), c1) -> sext_inreg for some c1 and target supports
  // sext_inreg.
  if (N1C && N0.getOpcode() == ISD::SHL && N1 == N0.getOperand(1)) {
    unsigned LowBits = MVT::getSizeInBits(VT) - (unsigned)N1C->getValue();
    MVT::ValueType EVT;
    switch (LowBits) {
    default: EVT = MVT::Other; break;
    case  1: EVT = MVT::i1;    break;
    case  8: EVT = MVT::i8;    break;
    case 16: EVT = MVT::i16;   break;
    case 32: EVT = MVT::i32;   break;
    }
    if (EVT > MVT::Other && TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, EVT))
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0),
                         DAG.getValueType(EVT));
  }
  
  // fold (sra (sra x, c1), c2) -> (sra x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRA) {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      unsigned Sum = N1C->getValue() + C1->getValue();
      if (Sum >= MVT::getSizeInBits(VT)) Sum = MVT::getSizeInBits(VT)-1;
      return DAG.getNode(ISD::SRA, VT, N0.getOperand(0),
                         DAG.getConstant(Sum, N1C->getValueType(0)));
    }
  }
  
  // Simplify, based on bits shifted out of the LHS. 
  if (N1C && SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  
  
  // If the sign bit is known to be zero, switch this to a SRL.
  if (TLI.MaskedValueIsZero(N0, MVT::getIntVTSignBit(VT)))
    return DAG.getNode(ISD::SRL, VT, N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitSRL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (srl c1, c2) -> c1 >>u c2
  if (N0C && N1C)
    return DAG.getNode(ISD::SRL, VT, N0, N1);
  // fold (srl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && TLI.MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits)))
    return DAG.getConstant(0, VT);
  
  // fold (srl (srl x, c1), c2) -> 0 or (srl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  
  // fold (srl (anyextend x), c) -> (anyextend (srl x, c))
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    MVT::ValueType SmallVT = N0.getOperand(0).getValueType();
    if (N1C->getValue() >= MVT::getSizeInBits(SmallVT))
      return DAG.getNode(ISD::UNDEF, VT);

    SDOperand SmallShift = DAG.getNode(ISD::SRL, SmallVT, N0.getOperand(0), N1);
    AddToWorkList(SmallShift.Val);
    return DAG.getNode(ISD::ANY_EXTEND, VT, SmallShift);
  }
  
  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getValue()+1 == MVT::getSizeInBits(VT)) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), N1);
  }
  
  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ && 
      N1C->getValue() == Log2_32(MVT::getSizeInBits(VT))) {
    uint64_t KnownZero, KnownOne, Mask = MVT::getIntVTBitMask(VT);
    TLI.ComputeMaskedBits(N0.getOperand(0), Mask, KnownZero, KnownOne);
    
    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (KnownOne) return DAG.getConstant(0, VT);
    
    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    uint64_t UnknownBits = ~KnownZero & Mask;
    if (UnknownBits == 0) return DAG.getConstant(1, VT);
    
    // Otherwise, check to see if there is exactly one bit input to the ctlz.
    if ((UnknownBits & (UnknownBits-1)) == 0) {
      // Okay, we know that only that the single bit specified by UnknownBits
      // could be set on input to the CTLZ node.  If this bit is set, the SRL
      // will return 0, if it is clear, it returns 1.  Change the CTLZ/SRL pair
      // to an SRL,XOR pair, which is likely to simplify more.
      unsigned ShAmt = CountTrailingZeros_64(UnknownBits);
      SDOperand Op = N0.getOperand(0);
      if (ShAmt) {
        Op = DAG.getNode(ISD::SRL, VT, Op,
                         DAG.getConstant(ShAmt, TLI.getShiftAmountTy()));
        AddToWorkList(Op.Val);
      }
      return DAG.getNode(ISD::XOR, VT, Op, DAG.getConstant(1, VT));
    }
  }
  
  // fold operands of srl based on knowledge that the low bits are not
  // demanded.
  if (N1C && SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  
  return SDOperand();
}

SDOperand DAGCombiner::visitCTLZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (ctlz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTLZ, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitCTTZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (cttz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTTZ, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitCTPOP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (ctpop c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTPOP, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitSELECT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  MVT::ValueType VT = N->getValueType(0);

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
  if (MVT::i1 == VT && N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold select C, 0, X -> ~C & X
  // FIXME: this should check for C type == X type, not i1?
  if (MVT::i1 == VT && N1C && N1C->isNullValue()) {
    SDOperand XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    AddToWorkList(XORNode.Val);
    return DAG.getNode(ISD::AND, VT, XORNode, N2);
  }
  // fold select C, X, 1 -> ~C | X
  if (MVT::i1 == VT && N2C && N2C->getValue() == 1) {
    SDOperand XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    AddToWorkList(XORNode.Val);
    return DAG.getNode(ISD::OR, VT, XORNode, N1);
  }
  // fold select C, X, 0 -> C & X
  // FIXME: this should check for C type == X type, not i1?
  if (MVT::i1 == VT && N2C && N2C->isNullValue())
    return DAG.getNode(ISD::AND, VT, N0, N1);
  // fold  X ? X : Y --> X ? 1 : Y --> X | Y
  if (MVT::i1 == VT && N0 == N1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold X ? Y : X --> X ? Y : 0 --> X & Y
  if (MVT::i1 == VT && N0 == N2)
    return DAG.getNode(ISD::AND, VT, N0, N1);
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDOperand(N, 0);  // Don't revisit N.
  
  // fold selects based on a setcc into other things, such as min/max/abs
  if (N0.getOpcode() == ISD::SETCC)
    // FIXME:
    // Check against MVT::Other for SELECT_CC, which is a workaround for targets
    // having to say they don't support SELECT_CC on every type the DAG knows
    // about, since there is no way to mark an opcode illegal at all value types
    if (TLI.isOperationLegal(ISD::SELECT_CC, MVT::Other))
      return DAG.getNode(ISD::SELECT_CC, VT, N0.getOperand(0), N0.getOperand(1),
                         N1, N2, N0.getOperand(2));
    else
      return SimplifySelect(N0, N1, N2);
  return SDOperand();
}

SDOperand DAGCombiner::visitSELECT_CC(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  SDOperand N3 = N->getOperand(3);
  SDOperand N4 = N->getOperand(4);
  ISD::CondCode CC = cast<CondCodeSDNode>(N4)->get();
  
  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N2 == N3)
    return N2;
  
  // Determine if the condition we're dealing with is constant
  SDOperand SCC = SimplifySetCC(TLI.getSetCCResultTy(), N0, N1, CC, false);
  if (SCC.Val) AddToWorkList(SCC.Val);

  if (ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.Val)) {
    if (SCCC->getValue())
      return N2;    // cond always true -> true val
    else
      return N3;    // cond always false -> false val
  }
  
  // Fold to a simpler select_cc
  if (SCC.Val && SCC.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::SELECT_CC, N2.getValueType(), 
                       SCC.getOperand(0), SCC.getOperand(1), N2, N3, 
                       SCC.getOperand(2));
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDOperand(N, 0);  // Don't revisit N.
  
  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(N0, N1, N2, N3, CC);
}

SDOperand DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get());
}

SDOperand DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0);
  
  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0.getOperand(0));
  
  // fold (sext (truncate (load x))) -> (sext (smaller load x))
  // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDOperand NarrowLoad = ReduceLoadWidth(N0.Val);
    if (NarrowLoad.Val) {
      if (NarrowLoad.Val != N0.Val)
        CombineTo(N0.Val, NarrowLoad);
      return DAG.getNode(ISD::SIGN_EXTEND, VT, NarrowLoad);
    }
  }

  // See if the value being truncated is already sign extended.  If so, just
  // eliminate the trunc/sext pair.
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDOperand Op = N0.getOperand(0);
    unsigned OpBits   = MVT::getSizeInBits(Op.getValueType());
    unsigned MidBits  = MVT::getSizeInBits(N0.getValueType());
    unsigned DestBits = MVT::getSizeInBits(VT);
    unsigned NumSignBits = TLI.ComputeNumSignBits(Op);
    
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
    if (!AfterLegalize || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                               N0.getValueType())) {
      if (Op.getValueType() < VT)
        Op = DAG.getNode(ISD::ANY_EXTEND, VT, Op);
      else if (Op.getValueType() > VT)
        Op = DAG.getNode(ISD::TRUNCATE, VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, Op,
                         DAG.getValueType(N0.getValueType()));
    }
  }
  
  // fold (sext (load x)) -> (sext (truncate (sextload x)))
  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      (!AfterLegalize||TLI.isLoadXLegal(ISD::SEXTLOAD, N0.getValueType()))){
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(), 
                                       LN0->isVolatile());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }

  // fold (sext (sextload x)) -> (sext (truncate (sextload x)))
  // fold (sext ( extload x)) -> (sext (truncate (sextload x)))
  if ((ISD::isSEXTLoad(N0.Val) || ISD::isEXTLoad(N0.Val)) &&
      ISD::isUNINDEXEDLoad(N0.Val) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType EVT = LN0->getLoadedVT();
    if (!AfterLegalize || TLI.isLoadXLegal(ISD::SEXTLOAD, EVT)) {
      SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                         LN0->getBasePtr(), LN0->getSrcValue(),
                                         LN0->getSrcValueOffset(), EVT,
                                         LN0->isVolatile(), 
                                         LN0->getAlignment());
      CombineTo(N, ExtLoad);
      CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
    }
  }
  
  // sext(setcc x,y,cc) -> select_cc x, y, -1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDOperand SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(~0ULL, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.Val) return SCC;
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

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
    SDOperand NarrowLoad = ReduceLoadWidth(N0.Val);
    if (NarrowLoad.Val) {
      if (NarrowLoad.Val != N0.Val)
        CombineTo(N0.Val, NarrowLoad);
      return DAG.getNode(ISD::ZERO_EXTEND, VT, NarrowLoad);
    }
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (!AfterLegalize || TLI.isOperationLegal(ISD::AND, VT))) {
    SDOperand Op = N0.getOperand(0);
    if (Op.getValueType() < VT) {
      Op = DAG.getNode(ISD::ANY_EXTEND, VT, Op);
    } else if (Op.getValueType() > VT) {
      Op = DAG.getNode(ISD::TRUNCATE, VT, Op);
    }
    return DAG.getZeroExtendInReg(Op, N0.getValueType());
  }
  
  // fold (zext (and (trunc x), cst)) -> (and x, cst).
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    SDOperand X = N0.getOperand(0).getOperand(0);
    if (X.getValueType() < VT) {
      X = DAG.getNode(ISD::ANY_EXTEND, VT, X);
    } else if (X.getValueType() > VT) {
      X = DAG.getNode(ISD::TRUNCATE, VT, X);
    }
    uint64_t Mask = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(Mask, VT));
  }
  
  // fold (zext (load x)) -> (zext (truncate (zextload x)))
  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      (!AfterLegalize||TLI.isLoadXLegal(ISD::ZEXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }

  // fold (zext (zextload x)) -> (zext (truncate (zextload x)))
  // fold (zext ( extload x)) -> (zext (truncate (zextload x)))
  if ((ISD::isZEXTLoad(N0.Val) || ISD::isEXTLoad(N0.Val)) &&
      ISD::isUNINDEXEDLoad(N0.Val) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType EVT = LN0->getLoadedVT();
    SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  // zext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDOperand SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.Val) return SCC;
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  
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
    SDOperand NarrowLoad = ReduceLoadWidth(N0.Val);
    if (NarrowLoad.Val) {
      if (NarrowLoad.Val != N0.Val)
        CombineTo(N0.Val, NarrowLoad);
      return DAG.getNode(ISD::ANY_EXTEND, VT, NarrowLoad);
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDOperand TruncOp = N0.getOperand(0);
    if (TruncOp.getValueType() == VT)
      return TruncOp; // x iff x size == zext size.
    if (TruncOp.getValueType() > VT)
      return DAG.getNode(ISD::TRUNCATE, VT, TruncOp);
    return DAG.getNode(ISD::ANY_EXTEND, VT, TruncOp);
  }
  
  // fold (aext (and (trunc x), cst)) -> (and x, cst).
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    SDOperand X = N0.getOperand(0).getOperand(0);
    if (X.getValueType() < VT) {
      X = DAG.getNode(ISD::ANY_EXTEND, VT, X);
    } else if (X.getValueType() > VT) {
      X = DAG.getNode(ISD::TRUNCATE, VT, X);
    }
    uint64_t Mask = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    return DAG.getNode(ISD::AND, VT, X, DAG.getConstant(Mask, VT));
  }
  
  // fold (aext (load x)) -> (aext (truncate (extload x)))
  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      (!AfterLegalize||TLI.isLoadXLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  // fold (aext (zextload x)) -> (aext (truncate (zextload x)))
  // fold (aext (sextload x)) -> (aext (truncate (sextload x)))
  // fold (aext ( extload x)) -> (aext (truncate (extload  x)))
  if (N0.getOpcode() == ISD::LOAD &&
      !ISD::isNON_EXTLoad(N0.Val) && ISD::isUNINDEXEDLoad(N0.Val) &&
      N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType EVT = LN0->getLoadedVT();
    SDOperand ExtLoad = DAG.getExtLoad(LN0->getExtensionType(), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
  if (N0.getOpcode() == ISD::SETCC) {
    SDOperand SCC = 
      SimplifySelectCC(N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.Val)
      return SCC;
  }
  
  return SDOperand();
}

/// ReduceLoadWidth - If the result of a wider load is shifted to right of N
/// bits and then truncated to a narrower type and where N is a multiple
/// of number of bits of the narrower type, transform it to a narrower load
/// from address + N / num of bits of new type. If the result is to be
/// extended, also fold the extension to form a extending load.
SDOperand DAGCombiner::ReduceLoadWidth(SDNode *N) {
  unsigned Opc = N->getOpcode();
  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = N->getValueType(0);

  // Special case: SIGN_EXTEND_INREG is basically truncating to EVT then
  // extended to VT.
  if (Opc == ISD::SIGN_EXTEND_INREG) {
    ExtType = ISD::SEXTLOAD;
    EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
    if (AfterLegalize && !TLI.isLoadXLegal(ISD::SEXTLOAD, EVT))
      return SDOperand();
  }

  unsigned EVTBits = MVT::getSizeInBits(EVT);
  unsigned ShAmt = 0;
  bool CombineSRL =  false;
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShAmt = N01->getValue();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        if (MVT::getSizeInBits(N0.getValueType()) <= EVTBits)
          return SDOperand();
        CombineSRL = true;
      }
    }
  }

  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      // Do not allow folding to i1 here.  i1 is implicitly stored in memory in
      // zero extended form: by shrinking the load, we lose track of the fact
      // that it is already zero extended.
      // FIXME: This should be reevaluated.
      VT != MVT::i1) {
    assert(MVT::getSizeInBits(N0.getValueType()) > EVTBits &&
           "Cannot truncate to larger type!");
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    MVT::ValueType PtrType = N0.getOperand(1).getValueType();
    // For big endian targets, we need to adjust the offset to the pointer to
    // load the correct bytes.
    if (!TLI.isLittleEndian())
      ShAmt = MVT::getSizeInBits(N0.getValueType()) - ShAmt - EVTBits;
    uint64_t PtrOff =  ShAmt / 8;
    SDOperand NewPtr = DAG.getNode(ISD::ADD, PtrType, LN0->getBasePtr(),
                                   DAG.getConstant(PtrOff, PtrType));
    AddToWorkList(NewPtr.Val);
    SDOperand Load = (ExtType == ISD::NON_EXTLOAD)
      ? DAG.getLoad(VT, LN0->getChain(), NewPtr,
                    LN0->getSrcValue(), LN0->getSrcValueOffset(),
                    LN0->isVolatile(), LN0->getAlignment())
      : DAG.getExtLoad(ExtType, VT, LN0->getChain(), NewPtr,
                       LN0->getSrcValue(), LN0->getSrcValueOffset(), EVT,
                       LN0->isVolatile(), LN0->getAlignment());
    AddToWorkList(N);
    if (CombineSRL) {
      std::vector<SDNode*> NowDead;
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1), NowDead);
      CombineTo(N->getOperand(0).Val, Load);
    } else
      CombineTo(N0.Val, Load, Load.getValue(1));
    if (ShAmt) {
      if (Opc == ISD::SIGN_EXTEND_INREG)
        return DAG.getNode(Opc, VT, Load, N->getOperand(1));
      else
        return DAG.getNode(Opc, VT, Load);
    }
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }

  return SDOperand();
}


SDOperand DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N1)->getVT();
  unsigned EVTBits = MVT::getSizeInBits(EVT);
  
  // fold (sext_in_reg c1) -> c1
  if (isa<ConstantSDNode>(N0) || N0.getOpcode() == ISD::UNDEF)
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0, N1);
  
  // If the input is already sign extended, just drop the extension.
  if (TLI.ComputeNumSignBits(N0) >= MVT::getSizeInBits(VT)-EVTBits+1)
    return N0;
  
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT < cast<VTSDNode>(N0.getOperand(1))->getVT()) {
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0), N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (TLI.MaskedValueIsZero(N0, 1ULL << (EVTBits-1)))
    return DAG.getZeroExtendInReg(N0, EVT);
  
  // fold operands of sext_in_reg based on knowledge that the top bits are not
  // demanded.
  if (SimplifyDemandedBits(SDOperand(N, 0)))
    return SDOperand(N, 0);
  
  // fold (sext_in_reg (load x)) -> (smaller sextload x)
  // fold (sext_in_reg (srl (load x), c)) -> (smaller sextload (x+c/evtbits))
  SDOperand NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.Val)
    return NarrowLoad;

  // fold (sext_in_reg (srl X, 24), i8) -> sra X, 24
  // fold (sext_in_reg (srl X, 23), i8) -> sra X, 23 iff possible.
  // We already fold "(sext_in_reg (srl X, 25), i8) -> srl X, 25" above.
  if (N0.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if (ShAmt->getValue()+EVTBits <= MVT::getSizeInBits(VT)) {
        // We can turn this into an SRA iff the input to the SRL is already sign
        // extended enough.
        unsigned InSignBits = TLI.ComputeNumSignBits(N0.getOperand(0));
        if (MVT::getSizeInBits(VT)-(ShAmt->getValue()+EVTBits) < InSignBits)
          return DAG.getNode(ISD::SRA, VT, N0.getOperand(0), N0.getOperand(1));
      }
  }

  // fold (sext_inreg (extload x)) -> (sextload x)
  if (ISD::isEXTLoad(N0.Val) && 
      ISD::isUNINDEXEDLoad(N0.Val) &&
      EVT == cast<LoadSDNode>(N0)->getLoadedVT() &&
      (!AfterLegalize || TLI.isLoadXLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (ISD::isZEXTLoad(N0.Val) && ISD::isUNINDEXEDLoad(N0.Val) &&
      N0.hasOneUse() &&
      EVT == cast<LoadSDNode>(N0)->getLoadedVT() &&
      (!AfterLegalize || TLI.isLoadXLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(), EVT,
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

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
    if (N0.getOperand(0).getValueType() < VT)
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0));
    else if (N0.getOperand(0).getValueType() > VT)
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate
      return N0.getOperand(0);
  }

  // fold (truncate (load x)) -> (smaller load x)
  // fold (truncate (srl (load x), c)) -> (smaller load (x+c/evtbits))
  return ReduceLoadWidth(N);
}

SDOperand DAGCombiner::visitBIT_CONVERT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

  // If the input is a constant, let getNode() fold it.
  if (isa<ConstantSDNode>(N0) || isa<ConstantFPSDNode>(N0)) {
    SDOperand Res = DAG.getNode(ISD::BIT_CONVERT, VT, N0);
    if (Res.Val != N) return Res;
  }
  
  if (N0.getOpcode() == ISD::BIT_CONVERT)  // conv(conv(x,t1),t2) -> conv(x,t2)
    return DAG.getNode(ISD::BIT_CONVERT, VT, N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a  higher alignment than the original!
  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      TLI.isOperationLegal(ISD::LOAD, VT)) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    unsigned Align = TLI.getTargetMachine().getTargetData()->
      getPrefTypeAlignment(getTypeForValueType(VT));
    unsigned OrigAlign = LN0->getAlignment();
    if (Align <= OrigAlign) {
      SDOperand Load = DAG.getLoad(VT, LN0->getChain(), LN0->getBasePtr(),
                                   LN0->getSrcValue(), LN0->getSrcValueOffset(),
                                   LN0->isVolatile(), LN0->getAlignment());
      AddToWorkList(N);
      CombineTo(N0.Val, DAG.getNode(ISD::BIT_CONVERT, N0.getValueType(), Load),
                Load.getValue(1));
      return Load;
    }
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitVBIT_CONVERT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);

  // If the input is a VBUILD_VECTOR with all constant elements, fold this now.
  // First check to see if this is all constant.
  if (N0.getOpcode() == ISD::VBUILD_VECTOR && N0.Val->hasOneUse() &&
      VT == MVT::Vector) {
    bool isSimple = true;
    for (unsigned i = 0, e = N0.getNumOperands()-2; i != e; ++i)
      if (N0.getOperand(i).getOpcode() != ISD::UNDEF &&
          N0.getOperand(i).getOpcode() != ISD::Constant &&
          N0.getOperand(i).getOpcode() != ISD::ConstantFP) {
        isSimple = false; 
        break;
      }
        
    MVT::ValueType DestEltVT = cast<VTSDNode>(N->getOperand(2))->getVT();
    if (isSimple && !MVT::isVector(DestEltVT)) {
      return ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(N0.Val, DestEltVT);
    }
  }
  
  return SDOperand();
}

/// ConstantFoldVBIT_CONVERTofVBUILD_VECTOR - We know that BV is a vbuild_vector
/// node with Constant, ConstantFP or Undef operands.  DstEltVT indicates the 
/// destination element value type.
SDOperand DAGCombiner::
ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(SDNode *BV, MVT::ValueType DstEltVT) {
  MVT::ValueType SrcEltVT = BV->getOperand(0).getValueType();
  
  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDOperand(BV, 0);
  
  unsigned SrcBitSize = MVT::getSizeInBits(SrcEltVT);
  unsigned DstBitSize = MVT::getSizeInBits(DstEltVT);
  
  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    SmallVector<SDOperand, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands()-2; i != e; ++i) {
      Ops.push_back(DAG.getNode(ISD::BIT_CONVERT, DstEltVT, BV->getOperand(i)));
      AddToWorkList(Ops.back().Val);
    }
    Ops.push_back(*(BV->op_end()-2)); // Add num elements.
    Ops.push_back(DAG.getValueType(DstEltVT));
    return DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0], Ops.size());
  }
  
  // Otherwise, we're growing or shrinking the elements.  To avoid having to
  // handle annoying details of growing/shrinking FP values, we convert them to
  // int first.
  if (MVT::isFloatingPoint(SrcEltVT)) {
    // Convert the input float vector to a int vector where the elements are the
    // same sizes.
    assert((SrcEltVT == MVT::f32 || SrcEltVT == MVT::f64) && "Unknown FP VT!");
    MVT::ValueType IntVT = SrcEltVT == MVT::f32 ? MVT::i32 : MVT::i64;
    BV = ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(BV, IntVT).Val;
    SrcEltVT = IntVT;
  }
  
  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (MVT::isFloatingPoint(DstEltVT)) {
    assert((DstEltVT == MVT::f32 || DstEltVT == MVT::f64) && "Unknown FP VT!");
    MVT::ValueType TmpVT = DstEltVT == MVT::f32 ? MVT::i32 : MVT::i64;
    SDNode *Tmp = ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(BV, TmpVT).Val;
    
    // Next, convert to FP elements of the same size.
    return ConstantFoldVBIT_CONVERTofVBUILD_VECTOR(Tmp, DstEltVT);
  }
  
  // Okay, we know the src/dst types are both integers of differing types.
  // Handling growing first.
  assert(MVT::isInteger(SrcEltVT) && MVT::isInteger(DstEltVT));
  if (SrcBitSize < DstBitSize) {
    unsigned NumInputsPerOutput = DstBitSize/SrcBitSize;
    
    SmallVector<SDOperand, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands()-2; i != e;
         i += NumInputsPerOutput) {
      bool isLE = TLI.isLittleEndian();
      uint64_t NewBits = 0;
      bool EltIsUndef = true;
      for (unsigned j = 0; j != NumInputsPerOutput; ++j) {
        // Shift the previously computed bits over.
        NewBits <<= SrcBitSize;
        SDOperand Op = BV->getOperand(i+ (isLE ? (NumInputsPerOutput-j-1) : j));
        if (Op.getOpcode() == ISD::UNDEF) continue;
        EltIsUndef = false;
        
        NewBits |= cast<ConstantSDNode>(Op)->getValue();
      }
      
      if (EltIsUndef)
        Ops.push_back(DAG.getNode(ISD::UNDEF, DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DstEltVT));
    }

    Ops.push_back(DAG.getConstant(Ops.size(), MVT::i32)); // Add num elements.
    Ops.push_back(DAG.getValueType(DstEltVT));            // Add element size.
    return DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0], Ops.size());
  }
  
  // Finally, this must be the case where we are shrinking elements: each input
  // turns into multiple outputs.
  unsigned NumOutputsPerInput = SrcBitSize/DstBitSize;
  SmallVector<SDOperand, 8> Ops;
  for (unsigned i = 0, e = BV->getNumOperands()-2; i != e; ++i) {
    if (BV->getOperand(i).getOpcode() == ISD::UNDEF) {
      for (unsigned j = 0; j != NumOutputsPerInput; ++j)
        Ops.push_back(DAG.getNode(ISD::UNDEF, DstEltVT));
      continue;
    }
    uint64_t OpVal = cast<ConstantSDNode>(BV->getOperand(i))->getValue();

    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      unsigned ThisVal = OpVal & ((1ULL << DstBitSize)-1);
      OpVal >>= DstBitSize;
      Ops.push_back(DAG.getConstant(ThisVal, DstEltVT));
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (!TLI.isLittleEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }
  Ops.push_back(DAG.getConstant(Ops.size(), MVT::i32)); // Add num elements.
  Ops.push_back(DAG.getValueType(DstEltVT));            // Add element size.
  return DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0], Ops.size());
}



SDOperand DAGCombiner::visitFADD(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fadd c1, c2) -> c1+c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FADD, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, VT, N1, N0);
  // fold (A + (-B)) -> A-B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FSUB, VT, N0, N1.getOperand(0));
  // fold ((-A) + B) -> B-A
  if (N0.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FSUB, VT, N1, N0.getOperand(0));
  
  // If allowed, fold (fadd (fadd x, c1), c2) -> (fadd x, (fadd c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FADD &&
      N0.Val->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FADD, VT, N0.getOperand(0),
                       DAG.getNode(ISD::FADD, VT, N0.getOperand(1), N1));
  
  return SDOperand();
}

SDOperand DAGCombiner::visitFSUB(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FSUB, VT, N0, N1);
  // fold (A-(-B)) -> A+B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FADD, VT, N0, N1.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFMUL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FMUL, VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMUL, VT, N1, N0);
  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, VT, N0, N0);
  
  // If allowed, fold (fmul (fmul x, c1), c2) -> (fmul x, (fmul c1, c2))
  if (UnsafeFPMath && N1CFP && N0.getOpcode() == ISD::FMUL &&
      N0.Val->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FMUL, VT, N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, VT, N0.getOperand(1), N1));
  
  return SDOperand();
}

SDOperand DAGCombiner::visitFDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);

  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FDIV, VT, N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitFREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);

  // fold (frem c1, c2) -> fmod(c1,c2)
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FREM, VT, N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);

  if (N0CFP && N1CFP)  // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, VT, N0, N1);
  
  if (N1CFP) {
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    union {
      double d;
      int64_t i;
    } u;
    u.d = N1CFP->getValue();
    if (u.i >= 0)
      return DAG.getNode(ISD::FABS, VT, N0);
    else
      return DAG.getNode(ISD::FNEG, VT, DAG.getNode(ISD::FABS, VT, N0));
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
  
  return SDOperand();
}



SDOperand DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (sint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getNode(ISD::SINT_TO_FP, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (uint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getNode(ISD::UINT_TO_FP, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_SINT, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_UINT, VT, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getNode(ISD::FP_ROUND, VT, N0);
  
  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);
  
  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.Val->hasOneUse()) {
    SDOperand Tmp = DAG.getNode(ISD::FP_ROUND, VT, N0.getOperand(0));
    AddToWorkList(Tmp.Val);
    return DAG.getNode(ISD::FCOPYSIGN, VT, Tmp, N0.getOperand(1));
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  
  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP) {
    SDOperand Round = DAG.getConstantFP(N0CFP->getValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, VT, Round);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fp_extend c1fp) -> c1fp
  if (N0CFP)
    return DAG.getNode(ISD::FP_EXTEND, VT, N0);
  
  // fold (fpext (load x)) -> (fpext (fpround (extload x)))
  if (ISD::isNON_EXTLoad(N0.Val) && N0.hasOneUse() &&
      (!AfterLegalize||TLI.isLoadXLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDOperand ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, VT, LN0->getChain(),
                                       LN0->getBasePtr(), LN0->getSrcValue(),
                                       LN0->getSrcValueOffset(),
                                       N0.getValueType(),
                                       LN0->isVolatile(), 
                                       LN0->getAlignment());
    CombineTo(N, ExtLoad);
    CombineTo(N0.Val, DAG.getNode(ISD::FP_ROUND, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand(N, 0);   // Return N so it doesn't get rechecked!
  }
  
  
  return SDOperand();
}

SDOperand DAGCombiner::visitFNEG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (fneg c1) -> -c1
  if (N0CFP)
    return DAG.getNode(ISD::FNEG, VT, N0);
  // fold (fneg (sub x, y)) -> (sub y, x)
  if (N0.getOpcode() == ISD::SUB)
    return DAG.getNode(ISD::SUB, VT, N0.getOperand(1), N0.getOperand(0));
  // fold (fneg (fneg x)) -> x
  if (N0.getOpcode() == ISD::FNEG)
    return N0.getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFABS(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fabs c1) -> fabs(c1)
  if (N0CFP)
    return DAG.getNode(ISD::FABS, VT, N0);
  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, VT, N0.getOperand(0));
  
  return SDOperand();
}

SDOperand DAGCombiner::visitBRCOND(SDNode *N) {
  SDOperand Chain = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // never taken branch, fold to chain
  if (N1C && N1C->isNullValue())
    return Chain;
  // unconditional branch
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N2);
  // fold a brcond with a setcc condition into a BR_CC node if BR_CC is legal
  // on the target.
  if (N1.getOpcode() == ISD::SETCC && 
      TLI.isOperationLegal(ISD::BR_CC, MVT::Other)) {
    return DAG.getNode(ISD::BR_CC, MVT::Other, Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }
  return SDOperand();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDOperand DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDOperand CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);
  
  // Use SimplifySetCC  to simplify SETCC's.
  SDOperand Simp = SimplifySetCC(MVT::i1, CondLHS, CondRHS, CC->get(), false);
  if (Simp.Val) AddToWorkList(Simp.Val);

  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(Simp.Val);

  // fold br_cc true, dest -> br dest (unconditional branch)
  if (SCCC && SCCC->getValue())
    return DAG.getNode(ISD::BR, MVT::Other, N->getOperand(0),
                       N->getOperand(4));
  // fold br_cc false, dest -> unconditional fall through
  if (SCCC && SCCC->isNullValue())
    return N->getOperand(0);

  // fold to a simpler setcc
  if (Simp.Val && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, MVT::Other, N->getOperand(0), 
                       Simp.getOperand(2), Simp.getOperand(0),
                       Simp.getOperand(1), N->getOperand(4));
  return SDOperand();
}


/// CombineToPreIndexedLoadStore - Try turning a load / store and a
/// pre-indexed load / store when the base pointer is a add or subtract
/// and it has other uses besides the load / store. After the
/// transformation, the new indexed load / store has effectively folded
/// the add / subtract in and all of its other uses are redirected to the
/// new load / store.
bool DAGCombiner::CombineToPreIndexedLoadStore(SDNode *N) {
  if (!AfterLegalize)
    return false;

  bool isLoad = true;
  SDOperand Ptr;
  MVT::ValueType VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->getAddressingMode() != ISD::UNINDEXED)
      return false;
    VT = LD->getLoadedVT();
    if (!TLI.isIndexedLoadLegal(ISD::PRE_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::PRE_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->getAddressingMode() != ISD::UNINDEXED)
      return false;
    VT = ST->getStoredVT();
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
      Ptr.Val->hasOneUse())
    return false;

  // Ask the target to do addressing mode selection.
  SDOperand BasePtr;
  SDOperand Offset;
  ISD::MemIndexedMode AM = ISD::UNINDEXED;
  if (!TLI.getPreIndexedAddressParts(N, BasePtr, Offset, AM, DAG))
    return false;
  // Don't create a indexed load / store with zero offset.
  if (isa<ConstantSDNode>(Offset) &&
      cast<ConstantSDNode>(Offset)->getValue() == 0)
    return false;
  
  // Try turning it into a pre-indexed load / store except when:
  // 1) The base is a frame index.
  // 2) If N is a store and the ptr is either the same as or is a
  //    predecessor of the value being stored.
  // 3) Another use of base ptr is a predecessor of N. If ptr is folded
  //    that would create a cycle.
  // 4) All uses are load / store ops that use it as base ptr.

  // Check #1.  Preinc'ing a frame index would require copying the stack pointer
  // (plus the implicit offset) to a register to preinc anyway.
  if (isa<FrameIndexSDNode>(BasePtr))
    return false;
  
  // Check #2.
  if (!isLoad) {
    SDOperand Val = cast<StoreSDNode>(N)->getValue();
    if (Val == Ptr || Ptr.Val->isPredecessor(Val.Val))
      return false;
  }

  // Now check for #2 and #3.
  bool RealUse = false;
  for (SDNode::use_iterator I = Ptr.Val->use_begin(),
         E = Ptr.Val->use_end(); I != E; ++I) {
    SDNode *Use = *I;
    if (Use == N)
      continue;
    if (Use->isPredecessor(N))
      return false;

    if (!((Use->getOpcode() == ISD::LOAD &&
           cast<LoadSDNode>(Use)->getBasePtr() == Ptr) ||
          (Use->getOpcode() == ISD::STORE) &&
          cast<StoreSDNode>(Use)->getBasePtr() == Ptr))
      RealUse = true;
  }
  if (!RealUse)
    return false;

  SDOperand Result;
  if (isLoad)
    Result = DAG.getIndexedLoad(SDOperand(N,0), BasePtr, Offset, AM);
  else
    Result = DAG.getIndexedStore(SDOperand(N,0), BasePtr, Offset, AM);
  ++PreIndexedNodes;
  ++NodesCombined;
  DOUT << "\nReplacing.4 "; DEBUG(N->dump());
  DOUT << "\nWith: "; DEBUG(Result.Val->dump(&DAG));
  DOUT << '\n';
  std::vector<SDNode*> NowDead;
  if (isLoad) {
    DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 0), Result.getValue(0),
                                  NowDead);
    DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 1), Result.getValue(2),
                                  NowDead);
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 0), Result.getValue(1),
                                  NowDead);
  }

  // Nodes can end up on the worklist more than once.  Make sure we do
  // not process a node that has been replaced.
  for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
    removeFromWorkList(NowDead[i]);
  // Finally, since the node is now dead, remove it from the graph.
  DAG.DeleteNode(N);

  // Replace the uses of Ptr with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(Ptr, Result.getValue(isLoad ? 1 : 0),
                                NowDead);
  removeFromWorkList(Ptr.Val);
  for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
    removeFromWorkList(NowDead[i]);
  DAG.DeleteNode(Ptr.Val);

  return true;
}

/// CombineToPostIndexedLoadStore - Try combine a load / store with a
/// add / sub of the base pointer node into a post-indexed load / store.
/// The transformation folded the add / subtract into the new indexed
/// load / store effectively and all of its uses are redirected to the
/// new load / store.
bool DAGCombiner::CombineToPostIndexedLoadStore(SDNode *N) {
  if (!AfterLegalize)
    return false;

  bool isLoad = true;
  SDOperand Ptr;
  MVT::ValueType VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(N)) {
    if (LD->getAddressingMode() != ISD::UNINDEXED)
      return false;
    VT = LD->getLoadedVT();
    if (!TLI.isIndexedLoadLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedLoadLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(N)) {
    if (ST->getAddressingMode() != ISD::UNINDEXED)
      return false;
    VT = ST->getStoredVT();
    if (!TLI.isIndexedStoreLegal(ISD::POST_INC, VT) &&
        !TLI.isIndexedStoreLegal(ISD::POST_DEC, VT))
      return false;
    Ptr = ST->getBasePtr();
    isLoad = false;
  } else
    return false;

  if (Ptr.Val->hasOneUse())
    return false;
  
  for (SDNode::use_iterator I = Ptr.Val->use_begin(),
         E = Ptr.Val->use_end(); I != E; ++I) {
    SDNode *Op = *I;
    if (Op == N ||
        (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB))
      continue;

    SDOperand BasePtr;
    SDOperand Offset;
    ISD::MemIndexedMode AM = ISD::UNINDEXED;
    if (TLI.getPostIndexedAddressParts(N, Op, BasePtr, Offset, AM, DAG)) {
      if (Ptr == Offset)
        std::swap(BasePtr, Offset);
      if (Ptr != BasePtr)
        continue;
      // Don't create a indexed load / store with zero offset.
      if (isa<ConstantSDNode>(Offset) &&
          cast<ConstantSDNode>(Offset)->getValue() == 0)
        continue;

      // Try turning it into a post-indexed load / store except when
      // 1) All uses are load / store ops that use it as base ptr.
      // 2) Op must be independent of N, i.e. Op is neither a predecessor
      //    nor a successor of N. Otherwise, if Op is folded that would
      //    create a cycle.

      // Check for #1.
      bool TryNext = false;
      for (SDNode::use_iterator II = BasePtr.Val->use_begin(),
             EE = BasePtr.Val->use_end(); II != EE; ++II) {
        SDNode *Use = *II;
        if (Use == Ptr.Val)
          continue;

        // If all the uses are load / store addresses, then don't do the
        // transformation.
        if (Use->getOpcode() == ISD::ADD || Use->getOpcode() == ISD::SUB){
          bool RealUse = false;
          for (SDNode::use_iterator III = Use->use_begin(),
                 EEE = Use->use_end(); III != EEE; ++III) {
            SDNode *UseUse = *III;
            if (!((UseUse->getOpcode() == ISD::LOAD &&
                   cast<LoadSDNode>(UseUse)->getBasePtr().Val == Use) ||
                  (UseUse->getOpcode() == ISD::STORE) &&
                  cast<StoreSDNode>(UseUse)->getBasePtr().Val == Use))
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
      if (!Op->isPredecessor(N) && !N->isPredecessor(Op)) {
        SDOperand Result = isLoad
          ? DAG.getIndexedLoad(SDOperand(N,0), BasePtr, Offset, AM)
          : DAG.getIndexedStore(SDOperand(N,0), BasePtr, Offset, AM);
        ++PostIndexedNodes;
        ++NodesCombined;
        DOUT << "\nReplacing.5 "; DEBUG(N->dump());
        DOUT << "\nWith: "; DEBUG(Result.Val->dump(&DAG));
        DOUT << '\n';
        std::vector<SDNode*> NowDead;
        if (isLoad) {
          DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 0), Result.getValue(0),
                                        NowDead);
          DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 1), Result.getValue(2),
                                        NowDead);
        } else {
          DAG.ReplaceAllUsesOfValueWith(SDOperand(N, 0), Result.getValue(1),
                                        NowDead);
        }

        // Nodes can end up on the worklist more than once.  Make sure we do
        // not process a node that has been replaced.
        for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
          removeFromWorkList(NowDead[i]);
        // Finally, since the node is now dead, remove it from the graph.
        DAG.DeleteNode(N);

        // Replace the uses of Use with uses of the updated base value.
        DAG.ReplaceAllUsesOfValueWith(SDOperand(Op, 0),
                                      Result.getValue(isLoad ? 1 : 0),
                                      NowDead);
        removeFromWorkList(Op);
        for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
          removeFromWorkList(NowDead[i]);
        DAG.DeleteNode(Op);

        return true;
      }
    }
  }
  return false;
}


SDOperand DAGCombiner::visitLOAD(SDNode *N) {
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDOperand Chain = LD->getChain();
  SDOperand Ptr   = LD->getBasePtr();

  // If load is not volatile and there are no uses of the loaded value (and
  // the updated indexed value in case of indexed loads), change uses of the
  // chain value into uses of the chain input (i.e. delete the dead load).
  if (!LD->isVolatile()) {
    if (N->getValueType(1) == MVT::Other) {
      // Unindexed loads.
      if (N->hasNUsesOfValue(0, 0))
        return CombineTo(N, DAG.getNode(ISD::UNDEF, N->getValueType(0)), Chain);
    } else {
      // Indexed loads.
      assert(N->getValueType(2) == MVT::Other && "Malformed indexed loads?");
      if (N->hasNUsesOfValue(0, 0) && N->hasNUsesOfValue(0, 1)) {
        SDOperand Undef0 = DAG.getNode(ISD::UNDEF, N->getValueType(0));
        SDOperand Undef1 = DAG.getNode(ISD::UNDEF, N->getValueType(1));
        SDOperand To[] = { Undef0, Undef1, Chain };
        return CombineTo(N, To, 3);
      }
    }
  }
  
  // If this load is directly stored, replace the load value with the stored
  // value.
  // TODO: Handle store large -> read small portion.
  // TODO: Handle TRUNCSTORE/LOADEXT
  if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
    if (ISD::isNON_TRUNCStore(Chain.Val)) {
      StoreSDNode *PrevST = cast<StoreSDNode>(Chain);
      if (PrevST->getBasePtr() == Ptr &&
          PrevST->getValue().getValueType() == N->getValueType(0))
      return CombineTo(N, Chain.getOperand(1), Chain);
    }
  }
    
  if (CombinerAA) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDOperand BetterChain = FindBetterChain(N, Chain);
    
    // If there is a better chain.
    if (Chain != BetterChain) {
      SDOperand ReplLoad;

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
                                  LD->getLoadedVT(),
                                  LD->isVolatile(), 
                                  LD->getAlignment());
      }

      // Create token factor to keep old chain connected.
      SDOperand Token = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                    Chain, ReplLoad.getValue(1));
      
      // Replace uses with load result and token factor. Don't add users
      // to work list.
      return CombineTo(N, ReplLoad.getValue(0), Token, false);
    }
  }

  // Try transforming N to an indexed load.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDOperand(N, 0);

  return SDOperand();
}

SDOperand DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDOperand Chain = ST->getChain();
  SDOperand Value = ST->getValue();
  SDOperand Ptr   = ST->getBasePtr();
  
  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BIT_CONVERT && !ST->isTruncatingStore()) {
    unsigned Align = ST->getAlignment();
    MVT::ValueType SVT = Value.getOperand(0).getValueType();
    unsigned OrigAlign = TLI.getTargetMachine().getTargetData()->
      getPrefTypeAlignment(getTypeForValueType(SVT));
    if (Align <= OrigAlign && TLI.isOperationLegal(ISD::STORE, SVT))
      return DAG.getStore(Chain, Value.getOperand(0), Ptr, ST->getSrcValue(),
                          ST->getSrcValueOffset());
  }
  
  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Value)) {
    if (Value.getOpcode() != ISD::TargetConstantFP) {
      SDOperand Tmp;
      switch (CFP->getValueType(0)) {
      default: assert(0 && "Unknown FP type");
      case MVT::f32:
        if (!AfterLegalize || TLI.isTypeLegal(MVT::i32)) {
          Tmp = DAG.getConstant(FloatToBits(CFP->getValue()), MVT::i32);
          return DAG.getStore(Chain, Tmp, Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset());
        }
        break;
      case MVT::f64:
        if (!AfterLegalize || TLI.isTypeLegal(MVT::i64)) {
          Tmp = DAG.getConstant(DoubleToBits(CFP->getValue()), MVT::i64);
          return DAG.getStore(Chain, Tmp, Ptr, ST->getSrcValue(),
                              ST->getSrcValueOffset());
        } else if (TLI.isTypeLegal(MVT::i32)) {
          // Many FP stores are not make apparent until after legalize, e.g. for
          // argument passing.  Since this is so common, custom legalize the
          // 64-bit integer store into two 32-bit stores.
          uint64_t Val = DoubleToBits(CFP->getValue());
          SDOperand Lo = DAG.getConstant(Val & 0xFFFFFFFF, MVT::i32);
          SDOperand Hi = DAG.getConstant(Val >> 32, MVT::i32);
          if (!TLI.isLittleEndian()) std::swap(Lo, Hi);

          SDOperand St0 = DAG.getStore(Chain, Lo, Ptr, ST->getSrcValue(),
                                       ST->getSrcValueOffset());
          Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                            DAG.getConstant(4, Ptr.getValueType()));
          SDOperand St1 = DAG.getStore(Chain, Hi, Ptr, ST->getSrcValue(),
                                       ST->getSrcValueOffset()+4);
          return DAG.getNode(ISD::TokenFactor, MVT::Other, St0, St1);
        }
        break;
      }
    }
  }

  if (CombinerAA) { 
    // Walk up chain skipping non-aliasing memory nodes.
    SDOperand BetterChain = FindBetterChain(N, Chain);
    
    // If there is a better chain.
    if (Chain != BetterChain) {
      // Replace the chain to avoid dependency.
      SDOperand ReplStore;
      if (ST->isTruncatingStore()) {
        ReplStore = DAG.getTruncStore(BetterChain, Value, Ptr,
          ST->getSrcValue(),ST->getSrcValueOffset(), ST->getStoredVT());
      } else {
        ReplStore = DAG.getStore(BetterChain, Value, Ptr,
          ST->getSrcValue(), ST->getSrcValueOffset());
      }
      
      // Create token to keep both nodes around.
      SDOperand Token =
        DAG.getNode(ISD::TokenFactor, MVT::Other, Chain, ReplStore);
        
      // Don't add users to work list.
      return CombineTo(N, Token, false);
    }
  }
  
  // Try transforming N to an indexed store.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDOperand(N, 0);

  return SDOperand();
}

SDOperand DAGCombiner::visitINSERT_VECTOR_ELT(SDNode *N) {
  SDOperand InVec = N->getOperand(0);
  SDOperand InVal = N->getOperand(1);
  SDOperand EltNo = N->getOperand(2);
  
  // If the invec is a BUILD_VECTOR and if EltNo is a constant, build a new
  // vector with the inserted element.
  if (InVec.getOpcode() == ISD::BUILD_VECTOR && isa<ConstantSDNode>(EltNo)) {
    unsigned Elt = cast<ConstantSDNode>(EltNo)->getValue();
    SmallVector<SDOperand, 8> Ops(InVec.Val->op_begin(), InVec.Val->op_end());
    if (Elt < Ops.size())
      Ops[Elt] = InVal;
    return DAG.getNode(ISD::BUILD_VECTOR, InVec.getValueType(),
                       &Ops[0], Ops.size());
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitVINSERT_VECTOR_ELT(SDNode *N) {
  SDOperand InVec = N->getOperand(0);
  SDOperand InVal = N->getOperand(1);
  SDOperand EltNo = N->getOperand(2);
  SDOperand NumElts = N->getOperand(3);
  SDOperand EltType = N->getOperand(4);
  
  // If the invec is a VBUILD_VECTOR and if EltNo is a constant, build a new
  // vector with the inserted element.
  if (InVec.getOpcode() == ISD::VBUILD_VECTOR && isa<ConstantSDNode>(EltNo)) {
    unsigned Elt = cast<ConstantSDNode>(EltNo)->getValue();
    SmallVector<SDOperand, 8> Ops(InVec.Val->op_begin(), InVec.Val->op_end());
    if (Elt < Ops.size()-2)
      Ops[Elt] = InVal;
    return DAG.getNode(ISD::VBUILD_VECTOR, InVec.getValueType(),
                       &Ops[0], Ops.size());
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitVBUILD_VECTOR(SDNode *N) {
  unsigned NumInScalars = N->getNumOperands()-2;
  SDOperand NumElts = N->getOperand(NumInScalars);
  SDOperand EltType = N->getOperand(NumInScalars+1);

  // Check to see if this is a VBUILD_VECTOR of a bunch of VEXTRACT_VECTOR_ELT
  // operations.  If so, and if the EXTRACT_ELT vector inputs come from at most
  // two distinct vectors, turn this into a shuffle node.
  SDOperand VecIn1, VecIn2;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    // Ignore undef inputs.
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    
    // If this input is something other than a VEXTRACT_VECTOR_ELT with a
    // constant index, bail out.
    if (N->getOperand(i).getOpcode() != ISD::VEXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(N->getOperand(i).getOperand(1))) {
      VecIn1 = VecIn2 = SDOperand(0, 0);
      break;
    }
    
    // If the input vector type disagrees with the result of the vbuild_vector,
    // we can't make a shuffle.
    SDOperand ExtractedFromVec = N->getOperand(i).getOperand(0);
    if (*(ExtractedFromVec.Val->op_end()-2) != NumElts ||
        *(ExtractedFromVec.Val->op_end()-1) != EltType) {
      VecIn1 = VecIn2 = SDOperand(0, 0);
      break;
    }
    
    // Otherwise, remember this.  We allow up to two distinct input vectors.
    if (ExtractedFromVec == VecIn1 || ExtractedFromVec == VecIn2)
      continue;
    
    if (VecIn1.Val == 0) {
      VecIn1 = ExtractedFromVec;
    } else if (VecIn2.Val == 0) {
      VecIn2 = ExtractedFromVec;
    } else {
      // Too many inputs.
      VecIn1 = VecIn2 = SDOperand(0, 0);
      break;
    }
  }
  
  // If everything is good, we can make a shuffle operation.
  if (VecIn1.Val) {
    SmallVector<SDOperand, 8> BuildVecIndices;
    for (unsigned i = 0; i != NumInScalars; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::UNDEF) {
        BuildVecIndices.push_back(DAG.getNode(ISD::UNDEF, TLI.getPointerTy()));
        continue;
      }
      
      SDOperand Extract = N->getOperand(i);
      
      // If extracting from the first vector, just use the index directly.
      if (Extract.getOperand(0) == VecIn1) {
        BuildVecIndices.push_back(Extract.getOperand(1));
        continue;
      }

      // Otherwise, use InIdx + VecSize
      unsigned Idx = cast<ConstantSDNode>(Extract.getOperand(1))->getValue();
      BuildVecIndices.push_back(DAG.getConstant(Idx+NumInScalars,
                                                TLI.getPointerTy()));
    }
    
    // Add count and size info.
    BuildVecIndices.push_back(NumElts);
    BuildVecIndices.push_back(DAG.getValueType(TLI.getPointerTy()));
    
    // Return the new VVECTOR_SHUFFLE node.
    SDOperand Ops[5];
    Ops[0] = VecIn1;
    if (VecIn2.Val) {
      Ops[1] = VecIn2;
    } else {
       // Use an undef vbuild_vector as input for the second operand.
      std::vector<SDOperand> UnOps(NumInScalars,
                                   DAG.getNode(ISD::UNDEF, 
                                           cast<VTSDNode>(EltType)->getVT()));
      UnOps.push_back(NumElts);
      UnOps.push_back(EltType);
      Ops[1] = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                           &UnOps[0], UnOps.size());
      AddToWorkList(Ops[1].Val);
    }
    Ops[2] = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                         &BuildVecIndices[0], BuildVecIndices.size());
    Ops[3] = NumElts;
    Ops[4] = EltType;
    return DAG.getNode(ISD::VVECTOR_SHUFFLE, MVT::Vector, Ops, 5);
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  SDOperand ShufMask = N->getOperand(2);
  unsigned NumElts = ShufMask.getNumOperands();

  // If the shuffle mask is an identity operation on the LHS, return the LHS.
  bool isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() != i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity) return N->getOperand(0);

  // If the shuffle mask is an identity operation on the RHS, return the RHS.
  isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() != i+NumElts) {
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
      unsigned Idx = cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue();
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

  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  // Normalize unary shuffle so the RHS is undef.
  if (isUnary && VecNum == 1)
    std::swap(N0, N1);

  // If it is a splat, check if the argument vector is a build_vector with
  // all scalar elements the same.
  if (isSplat) {
    SDNode *V = N0.Val;
    if (V->getOpcode() == ISD::BIT_CONVERT)
      V = V->getOperand(0).Val;
    if (V->getOpcode() == ISD::BUILD_VECTOR) {
      unsigned NumElems = V->getNumOperands()-2;
      if (NumElems > BaseIdx) {
        SDOperand Base;
        bool AllSame = true;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i).getOpcode() != ISD::UNDEF) {
            Base = V->getOperand(i);
            break;
          }
        }
        // Splat of <u, u, u, u>, return <u, u, u, u>
        if (!Base.Val)
          return N0;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i).getOpcode() != ISD::UNDEF &&
              V->getOperand(i) != Base) {
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
    if (N0.getOpcode() == ISD::UNDEF)
      return DAG.getNode(ISD::UNDEF, N->getValueType(0));
    // Check the SHUFFLE mask, mapping any inputs from the 2nd operand into the
    // first operand.
    SmallVector<SDOperand, 8> MappedOps;
    for (unsigned i = 0, e = ShufMask.getNumOperands(); i != e; ++i) {
      if (ShufMask.getOperand(i).getOpcode() == ISD::UNDEF ||
          cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() < NumElts) {
        MappedOps.push_back(ShufMask.getOperand(i));
      } else {
        unsigned NewIdx = 
           cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() - NumElts;
        MappedOps.push_back(DAG.getConstant(NewIdx, MVT::i32));
      }
    }
    ShufMask = DAG.getNode(ISD::BUILD_VECTOR, ShufMask.getValueType(),
                           &MappedOps[0], MappedOps.size());
    AddToWorkList(ShufMask.Val);
    return DAG.getNode(ISD::VECTOR_SHUFFLE, N->getValueType(0),
                       N0, 
                       DAG.getNode(ISD::UNDEF, N->getValueType(0)),
                       ShufMask);
  }
 
  return SDOperand();
}

SDOperand DAGCombiner::visitVVECTOR_SHUFFLE(SDNode *N) {
  SDOperand ShufMask = N->getOperand(2);
  unsigned NumElts = ShufMask.getNumOperands()-2;
  
  // If the shuffle mask is an identity operation on the LHS, return the LHS.
  bool isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() != i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity) return N->getOperand(0);
  
  // If the shuffle mask is an identity operation on the RHS, return the RHS.
  isIdentity = true;
  for (unsigned i = 0; i != NumElts; ++i) {
    if (ShufMask.getOperand(i).getOpcode() != ISD::UNDEF &&
        cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() != i+NumElts) {
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
      unsigned Idx = cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue();
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

  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  // Normalize unary shuffle so the RHS is undef.
  if (isUnary && VecNum == 1)
    std::swap(N0, N1);

  // If it is a splat, check if the argument vector is a build_vector with
  // all scalar elements the same.
  if (isSplat) {
    SDNode *V = N0.Val;

    // If this is a vbit convert that changes the element type of the vector but
    // not the number of vector elements, look through it.  Be careful not to
    // look though conversions that change things like v4f32 to v2f64.
    if (V->getOpcode() == ISD::VBIT_CONVERT) {
      SDOperand ConvInput = V->getOperand(0);
      if (ConvInput.getValueType() == MVT::Vector &&
          NumElts ==
          ConvInput.getConstantOperandVal(ConvInput.getNumOperands()-2))
        V = ConvInput.Val;
    }

    if (V->getOpcode() == ISD::VBUILD_VECTOR) {
      unsigned NumElems = V->getNumOperands()-2;
      if (NumElems > BaseIdx) {
        SDOperand Base;
        bool AllSame = true;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i).getOpcode() != ISD::UNDEF) {
            Base = V->getOperand(i);
            break;
          }
        }
        // Splat of <u, u, u, u>, return <u, u, u, u>
        if (!Base.Val)
          return N0;
        for (unsigned i = 0; i != NumElems; ++i) {
          if (V->getOperand(i).getOpcode() != ISD::UNDEF &&
              V->getOperand(i) != Base) {
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
    SmallVector<SDOperand, 8> MappedOps;
    for (unsigned i = 0; i != NumElts; ++i) {
      if (ShufMask.getOperand(i).getOpcode() == ISD::UNDEF ||
          cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() < NumElts) {
        MappedOps.push_back(ShufMask.getOperand(i));
      } else {
        unsigned NewIdx = 
          cast<ConstantSDNode>(ShufMask.getOperand(i))->getValue() - NumElts;
        MappedOps.push_back(DAG.getConstant(NewIdx, MVT::i32));
      }
    }
    // Add the type/#elts values.
    MappedOps.push_back(ShufMask.getOperand(NumElts));
    MappedOps.push_back(ShufMask.getOperand(NumElts+1));

    ShufMask = DAG.getNode(ISD::VBUILD_VECTOR, ShufMask.getValueType(),
                           &MappedOps[0], MappedOps.size());
    AddToWorkList(ShufMask.Val);
    
    // Build the undef vector.
    SDOperand UDVal = DAG.getNode(ISD::UNDEF, MappedOps[0].getValueType());
    for (unsigned i = 0; i != NumElts; ++i)
      MappedOps[i] = UDVal;
    MappedOps[NumElts  ] = *(N0.Val->op_end()-2);
    MappedOps[NumElts+1] = *(N0.Val->op_end()-1);
    UDVal = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                        &MappedOps[0], MappedOps.size());
    
    return DAG.getNode(ISD::VVECTOR_SHUFFLE, MVT::Vector, 
                       N0, UDVal, ShufMask,
                       MappedOps[NumElts], MappedOps[NumElts+1]);
  }
  
  return SDOperand();
}

/// XformToShuffleWithZero - Returns a vector_shuffle if it able to transform
/// a VAND to a vector_shuffle with the destination vector and a zero vector.
/// e.g. VAND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDOperand DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  SDOperand LHS = N->getOperand(0);
  SDOperand RHS = N->getOperand(1);
  if (N->getOpcode() == ISD::VAND) {
    SDOperand DstVecSize = *(LHS.Val->op_end()-2);
    SDOperand DstVecEVT  = *(LHS.Val->op_end()-1);
    if (RHS.getOpcode() == ISD::VBIT_CONVERT)
      RHS = RHS.getOperand(0);
    if (RHS.getOpcode() == ISD::VBUILD_VECTOR) {
      std::vector<SDOperand> IdxOps;
      unsigned NumOps = RHS.getNumOperands();
      unsigned NumElts = NumOps-2;
      MVT::ValueType EVT = cast<VTSDNode>(RHS.getOperand(NumOps-1))->getVT();
      for (unsigned i = 0; i != NumElts; ++i) {
        SDOperand Elt = RHS.getOperand(i);
        if (!isa<ConstantSDNode>(Elt))
          return SDOperand();
        else if (cast<ConstantSDNode>(Elt)->isAllOnesValue())
          IdxOps.push_back(DAG.getConstant(i, EVT));
        else if (cast<ConstantSDNode>(Elt)->isNullValue())
          IdxOps.push_back(DAG.getConstant(NumElts, EVT));
        else
          return SDOperand();
      }

      // Let's see if the target supports this vector_shuffle.
      if (!TLI.isVectorClearMaskLegal(IdxOps, EVT, DAG))
        return SDOperand();

      // Return the new VVECTOR_SHUFFLE node.
      SDOperand NumEltsNode = DAG.getConstant(NumElts, MVT::i32);
      SDOperand EVTNode = DAG.getValueType(EVT);
      std::vector<SDOperand> Ops;
      LHS = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, LHS, NumEltsNode,
                        EVTNode);
      Ops.push_back(LHS);
      AddToWorkList(LHS.Val);
      std::vector<SDOperand> ZeroOps(NumElts, DAG.getConstant(0, EVT));
      ZeroOps.push_back(NumEltsNode);
      ZeroOps.push_back(EVTNode);
      Ops.push_back(DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                                &ZeroOps[0], ZeroOps.size()));
      IdxOps.push_back(NumEltsNode);
      IdxOps.push_back(EVTNode);
      Ops.push_back(DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                                &IdxOps[0], IdxOps.size()));
      Ops.push_back(NumEltsNode);
      Ops.push_back(EVTNode);
      SDOperand Result = DAG.getNode(ISD::VVECTOR_SHUFFLE, MVT::Vector,
                                     &Ops[0], Ops.size());
      if (NumEltsNode != DstVecSize || EVTNode != DstVecEVT) {
        Result = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, Result,
                             DstVecSize, DstVecEVT);
      }
      return Result;
    }
  }
  return SDOperand();
}

/// visitVBinOp - Visit a binary vector operation, like VADD.  IntOp indicates
/// the scalar operation of the vop if it is operating on an integer vector
/// (e.g. ADD) and FPOp indicates the FP version (e.g. FADD).
SDOperand DAGCombiner::visitVBinOp(SDNode *N, ISD::NodeType IntOp, 
                                   ISD::NodeType FPOp) {
  MVT::ValueType EltType = cast<VTSDNode>(*(N->op_end()-1))->getVT();
  ISD::NodeType ScalarOp = MVT::isInteger(EltType) ? IntOp : FPOp;
  SDOperand LHS = N->getOperand(0);
  SDOperand RHS = N->getOperand(1);
  SDOperand Shuffle = XformToShuffleWithZero(N);
  if (Shuffle.Val) return Shuffle;

  // If the LHS and RHS are VBUILD_VECTOR nodes, see if we can constant fold
  // this operation.
  if (LHS.getOpcode() == ISD::VBUILD_VECTOR && 
      RHS.getOpcode() == ISD::VBUILD_VECTOR) {
    SmallVector<SDOperand, 8> Ops;
    for (unsigned i = 0, e = LHS.getNumOperands()-2; i != e; ++i) {
      SDOperand LHSOp = LHS.getOperand(i);
      SDOperand RHSOp = RHS.getOperand(i);
      // If these two elements can't be folded, bail out.
      if ((LHSOp.getOpcode() != ISD::UNDEF &&
           LHSOp.getOpcode() != ISD::Constant &&
           LHSOp.getOpcode() != ISD::ConstantFP) ||
          (RHSOp.getOpcode() != ISD::UNDEF &&
           RHSOp.getOpcode() != ISD::Constant &&
           RHSOp.getOpcode() != ISD::ConstantFP))
        break;
      // Can't fold divide by zero.
      if (N->getOpcode() == ISD::VSDIV || N->getOpcode() == ISD::VUDIV) {
        if ((RHSOp.getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(RHSOp.Val)->isNullValue()) ||
            (RHSOp.getOpcode() == ISD::ConstantFP &&
             !cast<ConstantFPSDNode>(RHSOp.Val)->getValue()))
          break;
      }
      Ops.push_back(DAG.getNode(ScalarOp, EltType, LHSOp, RHSOp));
      AddToWorkList(Ops.back().Val);
      assert((Ops.back().getOpcode() == ISD::UNDEF ||
              Ops.back().getOpcode() == ISD::Constant ||
              Ops.back().getOpcode() == ISD::ConstantFP) &&
             "Scalar binop didn't fold!");
    }
    
    if (Ops.size() == LHS.getNumOperands()-2) {
      Ops.push_back(*(LHS.Val->op_end()-2));
      Ops.push_back(*(LHS.Val->op_end()-1));
      return DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0], Ops.size());
    }
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::SimplifySelect(SDOperand N0, SDOperand N1, SDOperand N2){
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");
  
  SDOperand SCC = SimplifySelectCC(N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());
  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.Val) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      SDOperand SETCC = DAG.getNode(ISD::SETCC, N0.getValueType(),
                                    SCC.getOperand(0), SCC.getOperand(1), 
                                    SCC.getOperand(4));
      AddToWorkList(SETCC.Val);
      return DAG.getNode(ISD::SELECT, SCC.getValueType(), SCC.getOperand(2),
                         SCC.getOperand(3), SETCC);
    }
    return SCC;
  }
  return SDOperand();
}

/// SimplifySelectOps - Given a SELECT or a SELECT_CC node, where LHS and RHS
/// are the two values being selected between, see if we can simplify the
/// select.  Callers of this should assume that TheSelect is deleted if this
/// returns true.  As such, they should return the appropriate thing (e.g. the
/// node) back to the top-level of the DAG combiner loop to avoid it being
/// looked at.
///
bool DAGCombiner::SimplifySelectOps(SDNode *TheSelect, SDOperand LHS, 
                                    SDOperand RHS) {
  
  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() == RHS.getOpcode() && LHS.hasOneUse() && RHS.hasOneUse()){
    // If this is a load and the token chain is identical, replace the select
    // of two loads with a load through a select of the address to load from.
    // This triggers in things like "select bool X, 10.0, 123.0" after the FP
    // constants have been dropped into the constant pool.
    if (LHS.getOpcode() == ISD::LOAD &&
        // Token chains must be identical.
        LHS.getOperand(0) == RHS.getOperand(0)) {
      LoadSDNode *LLD = cast<LoadSDNode>(LHS);
      LoadSDNode *RLD = cast<LoadSDNode>(RHS);

      // If this is an EXTLOAD, the VT's must match.
      if (LLD->getLoadedVT() == RLD->getLoadedVT()) {
        // FIXME: this conflates two src values, discarding one.  This is not
        // the right thing to do, but nothing uses srcvalues now.  When they do,
        // turn SrcValue into a list of locations.
        SDOperand Addr;
        if (TheSelect->getOpcode() == ISD::SELECT) {
          // Check that the condition doesn't reach either load.  If so, folding
          // this will induce a cycle into the DAG.
          if (!LLD->isPredecessor(TheSelect->getOperand(0).Val) &&
              !RLD->isPredecessor(TheSelect->getOperand(0).Val)) {
            Addr = DAG.getNode(ISD::SELECT, LLD->getBasePtr().getValueType(),
                               TheSelect->getOperand(0), LLD->getBasePtr(),
                               RLD->getBasePtr());
          }
        } else {
          // Check that the condition doesn't reach either load.  If so, folding
          // this will induce a cycle into the DAG.
          if (!LLD->isPredecessor(TheSelect->getOperand(0).Val) &&
              !RLD->isPredecessor(TheSelect->getOperand(0).Val) &&
              !LLD->isPredecessor(TheSelect->getOperand(1).Val) &&
              !RLD->isPredecessor(TheSelect->getOperand(1).Val)) {
            Addr = DAG.getNode(ISD::SELECT_CC, LLD->getBasePtr().getValueType(),
                             TheSelect->getOperand(0),
                             TheSelect->getOperand(1), 
                             LLD->getBasePtr(), RLD->getBasePtr(),
                             TheSelect->getOperand(4));
          }
        }
        
        if (Addr.Val) {
          SDOperand Load;
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
                                  LLD->getLoadedVT(),
                                  LLD->isVolatile(), 
                                  LLD->getAlignment());
          }
          // Users of the select now use the result of the load.
          CombineTo(TheSelect, Load);
        
          // Users of the old loads now use the new load's chain.  We know the
          // old-load value is dead now.
          CombineTo(LHS.Val, Load.getValue(0), Load.getValue(1));
          CombineTo(RHS.Val, Load.getValue(0), Load.getValue(1));
          return true;
        }
      }
    }
  }
  
  return false;
}

SDOperand DAGCombiner::SimplifySelectCC(SDOperand N0, SDOperand N1, 
                                        SDOperand N2, SDOperand N3,
                                        ISD::CondCode CC, bool NotExtCompare) {
  
  MVT::ValueType VT = N2.getValueType();
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.Val);

  // Determine if the condition we're dealing with is constant
  SDOperand SCC = SimplifySetCC(TLI.getSetCCResultTy(), N0, N1, CC, false);
  if (SCC.Val) AddToWorkList(SCC.Val);
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.Val);

  // fold select_cc true, x, y -> x
  if (SCCC && SCCC->getValue())
    return N2;
  // fold select_cc false, x, y -> y
  if (SCCC && SCCC->getValue() == 0)
    return N3;
  
  // Check to see if we can simplify the select into an fabs node
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1)) {
    // Allow either -0.0 or 0.0
    if (CFP->getValue() == 0.0) {
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
      MVT::isInteger(N0.getValueType()) && 
      MVT::isInteger(N2.getValueType()) && 
      (N1C->isNullValue() ||                    // (a < 0) ? b : 0
       (N1C->getValue() == 1 && N0 == N2))) {   // (a < 1) ? a : 0
    MVT::ValueType XType = N0.getValueType();
    MVT::ValueType AType = N2.getValueType();
    if (XType >= AType) {
      // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
      // single-bit constant.
      if (N2C && ((N2C->getValue() & (N2C->getValue()-1)) == 0)) {
        unsigned ShCtV = Log2_64(N2C->getValue());
        ShCtV = MVT::getSizeInBits(XType)-ShCtV-1;
        SDOperand ShCt = DAG.getConstant(ShCtV, TLI.getShiftAmountTy());
        SDOperand Shift = DAG.getNode(ISD::SRL, XType, N0, ShCt);
        AddToWorkList(Shift.Val);
        if (XType > AType) {
          Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
          AddToWorkList(Shift.Val);
        }
        return DAG.getNode(ISD::AND, AType, Shift, N2);
      }
      SDOperand Shift = DAG.getNode(ISD::SRA, XType, N0,
                                    DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                    TLI.getShiftAmountTy()));
      AddToWorkList(Shift.Val);
      if (XType > AType) {
        Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
        AddToWorkList(Shift.Val);
      }
      return DAG.getNode(ISD::AND, AType, Shift, N2);
    }
  }
  
  // fold select C, 16, 0 -> shl C, 4
  if (N2C && N3C && N3C->isNullValue() && isPowerOf2_64(N2C->getValue()) &&
      TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult) {
    
    // If the caller doesn't want us to simplify this into a zext of a compare,
    // don't do it.
    if (NotExtCompare && N2C->getValue() == 1)
      return SDOperand();
    
    // Get a SetCC of the condition
    // FIXME: Should probably make sure that setcc is legal if we ever have a
    // target where it isn't.
    SDOperand Temp, SCC;
    // cast from setcc result type to select result type
    if (AfterLegalize) {
      SCC  = DAG.getSetCC(TLI.getSetCCResultTy(), N0, N1, CC);
      if (N2.getValueType() < SCC.getValueType())
        Temp = DAG.getZeroExtendInReg(SCC, N2.getValueType());
      else
        Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getValueType(), SCC);
    } else {
      SCC  = DAG.getSetCC(MVT::i1, N0, N1, CC);
      Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getValueType(), SCC);
    }
    AddToWorkList(SCC.Val);
    AddToWorkList(Temp.Val);
    
    if (N2C->getValue() == 1)
      return Temp;
    // shl setcc result by log2 n2c
    return DAG.getNode(ISD::SHL, N2.getValueType(), Temp,
                       DAG.getConstant(Log2_64(N2C->getValue()),
                                       TLI.getShiftAmountTy()));
  }
    
  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && N3C && N3C->isNullValue() && N2C && (N2C->getValue() == 1ULL)) {
    MVT::ValueType XType = N0.getValueType();
    if (TLI.isOperationLegal(ISD::SETCC, TLI.getSetCCResultTy())) {
      SDOperand Res = DAG.getSetCC(TLI.getSetCCResultTy(), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, VT, Res);
      return Res;
    }
    
    // seteq X, 0 -> srl (ctlz X, log2(size(X)))
    if (N1C && N1C->isNullValue() && CC == ISD::SETEQ && 
        TLI.isOperationLegal(ISD::CTLZ, XType)) {
      SDOperand Ctlz = DAG.getNode(ISD::CTLZ, XType, N0);
      return DAG.getNode(ISD::SRL, XType, Ctlz, 
                         DAG.getConstant(Log2_32(MVT::getSizeInBits(XType)),
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, 0 -> srl (and (-X, ~X), size(X)-1)
    if (N1C && N1C->isNullValue() && CC == ISD::SETGT) { 
      SDOperand NegN0 = DAG.getNode(ISD::SUB, XType, DAG.getConstant(0, XType),
                                    N0);
      SDOperand NotN0 = DAG.getNode(ISD::XOR, XType, N0, 
                                    DAG.getConstant(~0ULL, XType));
      return DAG.getNode(ISD::SRL, XType, 
                         DAG.getNode(ISD::AND, XType, NegN0, NotN0),
                         DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, -1 -> xor (srl (X, size(X)-1), 1)
    if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT) {
      SDOperand Sign = DAG.getNode(ISD::SRL, XType, N0,
                                   DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                   TLI.getShiftAmountTy()));
      return DAG.getNode(ISD::XOR, XType, Sign, DAG.getConstant(1, XType));
    }
  }
  
  // Check to see if this is an integer abs. select_cc setl[te] X, 0, -X, X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE) &&
      N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1) &&
      N2.getOperand(0) == N1 && MVT::isInteger(N0.getValueType())) {
    MVT::ValueType XType = N0.getValueType();
    SDOperand Shift = DAG.getNode(ISD::SRA, XType, N0,
                                  DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                  TLI.getShiftAmountTy()));
    SDOperand Add = DAG.getNode(ISD::ADD, XType, N0, Shift);
    AddToWorkList(Shift.Val);
    AddToWorkList(Add.Val);
    return DAG.getNode(ISD::XOR, XType, Add, Shift);
  }
  // Check to see if this is an integer abs. select_cc setgt X, -1, X, -X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT &&
      N0 == N2 && N3.getOpcode() == ISD::SUB && N0 == N3.getOperand(1)) {
    if (ConstantSDNode *SubC = dyn_cast<ConstantSDNode>(N3.getOperand(0))) {
      MVT::ValueType XType = N0.getValueType();
      if (SubC->isNullValue() && MVT::isInteger(XType)) {
        SDOperand Shift = DAG.getNode(ISD::SRA, XType, N0,
                                    DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                      TLI.getShiftAmountTy()));
        SDOperand Add = DAG.getNode(ISD::ADD, XType, N0, Shift);
        AddToWorkList(Shift.Val);
        AddToWorkList(Add.Val);
        return DAG.getNode(ISD::XOR, XType, Add, Shift);
      }
    }
  }
  
  return SDOperand();
}

/// SimplifySetCC - This is a stub for TargetLowering::SimplifySetCC.
SDOperand DAGCombiner::SimplifySetCC(MVT::ValueType VT, SDOperand N0,
                                     SDOperand N1, ISD::CondCode Cond,
                                     bool foldBooleans) {
  TargetLowering::DAGCombinerInfo 
    DagCombineInfo(DAG, !AfterLegalize, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo);
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand DAGCombiner::BuildSDIV(SDNode *N) {
  std::vector<SDNode*> Built;
  SDOperand S = TLI.BuildSDIV(N, DAG, &Built);

  for (std::vector<SDNode*>::iterator ii = Built.begin(), ee = Built.end();
       ii != ee; ++ii)
    AddToWorkList(*ii);
  return S;
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand DAGCombiner::BuildUDIV(SDNode *N) {
  std::vector<SDNode*> Built;
  SDOperand S = TLI.BuildUDIV(N, DAG, &Built);

  for (std::vector<SDNode*>::iterator ii = Built.begin(), ee = Built.end();
       ii != ee; ++ii)
    AddToWorkList(*ii);
  return S;
}

/// FindBaseOffset - Return true if base is known not to alias with anything
/// but itself.  Provides base object and offset as results.
static bool FindBaseOffset(SDOperand Ptr, SDOperand &Base, int64_t &Offset) {
  // Assume it is a primitive operation.
  Base = Ptr; Offset = 0;
  
  // If it's an adding a simple constant then integrate the offset.
  if (Base.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Base.getOperand(1))) {
      Base = Base.getOperand(0);
      Offset += C->getValue();
    }
  }
  
  // If it's any of the following then it can't alias with anything but itself.
  return isa<FrameIndexSDNode>(Base) ||
         isa<ConstantPoolSDNode>(Base) ||
         isa<GlobalAddressSDNode>(Base);
}

/// isAlias - Return true if there is any possibility that the two addresses
/// overlap.
bool DAGCombiner::isAlias(SDOperand Ptr1, int64_t Size1,
                          const Value *SrcValue1, int SrcValueOffset1,
                          SDOperand Ptr2, int64_t Size2,
                          const Value *SrcValue2, int SrcValueOffset2)
{
  // If they are the same then they must be aliases.
  if (Ptr1 == Ptr2) return true;
  
  // Gather base node and offset information.
  SDOperand Base1, Base2;
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
    int Overlap1 = Size1 + SrcValueOffset1 + Offset1;
    int Overlap2 = Size2 + SrcValueOffset2 + Offset2;
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
                        SDOperand &Ptr, int64_t &Size,
                        const Value *&SrcValue, int &SrcValueOffset) {
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    Size = MVT::getSizeInBits(LD->getLoadedVT()) >> 3;
    SrcValue = LD->getSrcValue();
    SrcValueOffset = LD->getSrcValueOffset();
    return true;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    Size = MVT::getSizeInBits(ST->getStoredVT()) >> 3;
    SrcValue = ST->getSrcValue();
    SrcValueOffset = ST->getSrcValueOffset();
  } else {
    assert(0 && "FindAliasInfo expected a memory operand");
  }
  
  return false;
}

/// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDOperand OriginalChain,
                                   SmallVector<SDOperand, 8> &Aliases) {
  SmallVector<SDOperand, 8> Chains;     // List of chains to visit.
  std::set<SDNode *> Visited;           // Visited node set.
  
  // Get alias information for node.
  SDOperand Ptr;
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
    SDOperand Chain = Chains.back();
    Chains.pop_back();
    
     // Don't bother if we've been before.
    if (Visited.find(Chain.Val) != Visited.end()) continue;
    Visited.insert(Chain.Val);
  
    switch (Chain.getOpcode()) {
    case ISD::EntryToken:
      // Entry token is ideal chain operand, but handled in FindBetterChain.
      break;
      
    case ISD::LOAD:
    case ISD::STORE: {
      // Get alias information for Chain.
      SDOperand OpPtr;
      int64_t OpSize;
      const Value *OpSrcValue;
      int OpSrcValueOffset;
      bool IsOpLoad = FindAliasInfo(Chain.Val, OpPtr, OpSize,
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
        AddToWorkList(Chain.Val);
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
      AddToWorkList(Chain.Val);
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
SDOperand DAGCombiner::FindBetterChain(SDNode *N, SDOperand OldChain) {
  SmallVector<SDOperand, 8> Aliases;  // Ops for replacing token factor.
  
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
  SDOperand NewChain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                   &Aliases[0], Aliases.size());

  // Make sure the old chain gets cleaned up.
  if (NewChain != OldChain) AddToWorkList(OldChain.Val);
  
  return NewChain;
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(bool RunningAfterLegalize, AliasAnalysis &AA) {
  if (!RunningAfterLegalize && ViewDAGCombine1)
    viewGraph();
  if (RunningAfterLegalize && ViewDAGCombine2)
    viewGraph();
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this, AA).Run(RunningAfterLegalize);
}
