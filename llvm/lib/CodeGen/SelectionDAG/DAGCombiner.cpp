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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");
STATISTIC(OpsNarrowed     , "Number of load/op/store narrowed");
STATISTIC(LdStFP2Int      , "Number of fp load/store pairs transformed to int");
STATISTIC(SlicedLoads, "Number of load sliced");

namespace {
  static cl::opt<bool>
    CombinerAA("combiner-alias-analysis", cl::Hidden,
               cl::desc("Turn on alias analysis during testing"));

  static cl::opt<bool>
    CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
               cl::desc("Include global information in alias analysis"));

  /// Hidden option to stress test load slicing, i.e., when this option
  /// is enabled, load slicing bypasses most of its profitability guards.
  static cl::opt<bool>
  StressLoadSlicing("combiner-stress-load-slicing", cl::Hidden,
                    cl::desc("Bypass the profitability model of load "
                             "slicing"),
                    cl::init(false));

//------------------------------ DAGCombiner ---------------------------------//

  class DAGCombiner {
    SelectionDAG &DAG;
    const TargetLowering &TLI;
    CombineLevel Level;
    CodeGenOpt::Level OptLevel;
    bool LegalOperations;
    bool LegalTypes;
    bool ForCodeSize;

    // Worklist of all of the nodes that need to be simplified.
    //
    // This has the semantics that when adding to the worklist,
    // the item added must be next to be processed. It should
    // also only appear once. The naive approach to this takes
    // linear time.
    //
    // To reduce the insert/remove time to logarithmic, we use
    // a set and a vector to maintain our worklist.
    //
    // The set contains the items on the worklist, but does not
    // maintain the order they should be visited.
    //
    // The vector maintains the order nodes should be visited, but may
    // contain duplicate or removed nodes. When choosing a node to
    // visit, we pop off the order stack until we find an item that is
    // also in the contents set. All operations are O(log N).
    SmallPtrSet<SDNode*, 64> WorkListContents;
    SmallVector<SDNode*, 64> WorkListOrder;

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
    /// AddToWorkList - Add to the work list making sure its instance is at the
    /// back (next to be processed.)
    void AddToWorkList(SDNode *N) {
      WorkListContents.insert(N);
      WorkListOrder.push_back(N);
    }

    /// removeFromWorkList - remove all instances of N from the worklist.
    ///
    void removeFromWorkList(SDNode *N) {
      WorkListContents.erase(N);
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
    bool SliceUpLoad(SDNode *N);

    void ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad);
    SDValue PromoteOperand(SDValue Op, EVT PVT, bool &Replace);
    SDValue SExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue ZExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue PromoteIntBinOp(SDValue Op);
    SDValue PromoteIntShiftOp(SDValue Op);
    SDValue PromoteExtend(SDValue Op);
    bool PromoteLoad(SDValue Op);

    void ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                         SDValue Trunc, SDValue ExtLoad, SDLoc DL,
                         ISD::NodeType ExtType);

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
    SDValue visitSUBC(SDNode *N);
    SDValue visitADDE(SDNode *N);
    SDValue visitSUBE(SDNode *N);
    SDValue visitMUL(SDNode *N);
    SDValue visitSDIV(SDNode *N);
    SDValue visitUDIV(SDNode *N);
    SDValue visitSREM(SDNode *N);
    SDValue visitUREM(SDNode *N);
    SDValue visitMULHU(SDNode *N);
    SDValue visitMULHS(SDNode *N);
    SDValue visitSMUL_LOHI(SDNode *N);
    SDValue visitUMUL_LOHI(SDNode *N);
    SDValue visitSMULO(SDNode *N);
    SDValue visitUMULO(SDNode *N);
    SDValue visitSDIVREM(SDNode *N);
    SDValue visitUDIVREM(SDNode *N);
    SDValue visitAND(SDNode *N);
    SDValue visitOR(SDNode *N);
    SDValue visitXOR(SDNode *N);
    SDValue SimplifyVBinOp(SDNode *N);
    SDValue SimplifyVUnaryOp(SDNode *N);
    SDValue visitSHL(SDNode *N);
    SDValue visitSRA(SDNode *N);
    SDValue visitSRL(SDNode *N);
    SDValue visitCTLZ(SDNode *N);
    SDValue visitCTLZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTTZ(SDNode *N);
    SDValue visitCTTZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTPOP(SDNode *N);
    SDValue visitSELECT(SDNode *N);
    SDValue visitVSELECT(SDNode *N);
    SDValue visitSELECT_CC(SDNode *N);
    SDValue visitSETCC(SDNode *N);
    SDValue visitSIGN_EXTEND(SDNode *N);
    SDValue visitZERO_EXTEND(SDNode *N);
    SDValue visitANY_EXTEND(SDNode *N);
    SDValue visitSIGN_EXTEND_INREG(SDNode *N);
    SDValue visitTRUNCATE(SDNode *N);
    SDValue visitBITCAST(SDNode *N);
    SDValue visitBUILD_PAIR(SDNode *N);
    SDValue visitFADD(SDNode *N);
    SDValue visitFSUB(SDNode *N);
    SDValue visitFMUL(SDNode *N);
    SDValue visitFMA(SDNode *N);
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
    SDValue visitFCEIL(SDNode *N);
    SDValue visitFTRUNC(SDNode *N);
    SDValue visitFFLOOR(SDNode *N);
    SDValue visitBRCOND(SDNode *N);
    SDValue visitBR_CC(SDNode *N);
    SDValue visitLOAD(SDNode *N);
    SDValue visitSTORE(SDNode *N);
    SDValue visitINSERT_VECTOR_ELT(SDNode *N);
    SDValue visitEXTRACT_VECTOR_ELT(SDNode *N);
    SDValue visitBUILD_VECTOR(SDNode *N);
    SDValue visitCONCAT_VECTORS(SDNode *N);
    SDValue visitEXTRACT_SUBVECTOR(SDNode *N);
    SDValue visitVECTOR_SHUFFLE(SDNode *N);

    SDValue XformToShuffleWithZero(SDNode *N);
    SDValue ReassociateOps(unsigned Opc, SDLoc DL, SDValue LHS, SDValue RHS);

    SDValue visitShiftByConstant(SDNode *N, unsigned Amt);

    bool SimplifySelectOps(SDNode *SELECT, SDValue LHS, SDValue RHS);
    SDValue SimplifyBinOpWithSameOpcodeHands(SDNode *N);
    SDValue SimplifySelect(SDLoc DL, SDValue N0, SDValue N1, SDValue N2);
    SDValue SimplifySelectCC(SDLoc DL, SDValue N0, SDValue N1, SDValue N2,
                             SDValue N3, ISD::CondCode CC,
                             bool NotExtCompare = false);
    SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1, ISD::CondCode Cond,
                          SDLoc DL, bool foldBooleans = true);
    SDValue SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                         unsigned HiOp);
    SDValue CombineConsecutiveLoads(SDNode *N, EVT VT);
    SDValue ConstantFoldBITCASTofBUILD_VECTOR(SDNode *, EVT);
    SDValue BuildSDIV(SDNode *N);
    SDValue BuildUDIV(SDNode *N);
    SDValue MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                               bool DemandHighBits = true);
    SDValue MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1);
    SDNode *MatchRotate(SDValue LHS, SDValue RHS, SDLoc DL);
    SDValue ReduceLoadWidth(SDNode *N);
    SDValue ReduceLoadOpStoreWidth(SDNode *N);
    SDValue TransformFPLoadStorePair(SDNode *N);
    SDValue reduceBuildVecExtToExtBuildVec(SDNode *N);
    SDValue reduceBuildVecConvertToConvertBuildVec(SDNode *N);

    SDValue GetDemandedBits(SDValue V, const APInt &Mask);

    /// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDValue OriginalChain,
                          SmallVectorImpl<SDValue> &Aliases);

    /// isAlias - Return true if there is any possibility that the two addresses
    /// overlap.
    bool isAlias(SDValue Ptr1, int64_t Size1, bool IsVolatile1,
                 const Value *SrcValue1, int SrcValueOffset1,
                 unsigned SrcValueAlign1,
                 const MDNode *TBAAInfo1,
                 SDValue Ptr2, int64_t Size2, bool IsVolatile2,
                 const Value *SrcValue2, int SrcValueOffset2,
                 unsigned SrcValueAlign2,
                 const MDNode *TBAAInfo2) const;

    /// isAlias - Return true if there is any possibility that the two addresses
    /// overlap.
    bool isAlias(LSBaseSDNode *Op0, LSBaseSDNode *Op1);

    /// FindAliasInfo - Extracts the relevant alias information from the memory
    /// node.  Returns true if the operand was a load.
    bool FindAliasInfo(SDNode *N,
                       SDValue &Ptr, int64_t &Size, bool &IsVolatile,
                       const Value *&SrcValue, int &SrcValueOffset,
                       unsigned &SrcValueAlignment,
                       const MDNode *&TBAAInfo) const;

    /// FindBetterChain - Walk up chain skipping non-aliasing memory nodes,
    /// looking for a better chain (aliasing node.)
    SDValue FindBetterChain(SDNode *N, SDValue Chain);

    /// Merge consecutive store operations into a wide store.
    /// This optimization uses wide integers or vectors when possible.
    /// \return True if some memory operations were changed.
    bool MergeConsecutiveStores(StoreSDNode *N);

  public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis &A, CodeGenOpt::Level OL)
        : DAG(D), TLI(D.getTargetLoweringInfo()), Level(BeforeLegalizeTypes),
          OptLevel(OL), LegalOperations(false), LegalTypes(false), AA(A) {
      AttributeSet FnAttrs =
          DAG.getMachineFunction().getFunction()->getAttributes();
      ForCodeSize =
          FnAttrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::OptimizeForSize) ||
          FnAttrs.hasAttribute(AttributeSet::FunctionIndex, Attribute::MinSize);
    }

    /// Run - runs the dag combiner on all nodes in the work list
    void Run(CombineLevel AtLevel);

    SelectionDAG &getDAG() const { return DAG; }

    /// getShiftAmountTy - Returns a type large enough to hold any valid
    /// shift amount - before type legalization these can be huge.
    EVT getShiftAmountTy(EVT LHSTy) {
      assert(LHSTy.isInteger() && "Shift amount is not an integer type!");
      if (LHSTy.isVector())
        return LHSTy;
      return LegalTypes ? TLI.getScalarShiftAmountTy(LHSTy)
                        : TLI.getPointerTy();
    }

    /// isTypeLegal - This method returns true if we are running before type
    /// legalization or if the specified VT is legal.
    bool isTypeLegal(const EVT &VT) {
      if (!LegalTypes) return true;
      return TLI.isTypeLegal(VT);
    }

    /// getSetCCResultType - Convenience wrapper around
    /// TargetLowering::getSetCCResultType
    EVT getSetCCResultType(EVT VT) const {
      return TLI.getSetCCResultType(*DAG.getContext(), VT);
    }
  };
}


namespace {
/// WorkListRemover - This class is a DAGUpdateListener that removes any deleted
/// nodes from the worklist.
class WorkListRemover : public SelectionDAG::DAGUpdateListener {
  DAGCombiner &DC;
public:
  explicit WorkListRemover(DAGCombiner &dc)
    : SelectionDAG::DAGUpdateListener(dc.getDAG()), DC(dc) {}

  virtual void NodeDeleted(SDNode *N, SDNode *E) {
    DC.removeFromWorkList(N);
  }
};
}

//===----------------------------------------------------------------------===//
//  TargetLowering::DAGCombinerInfo implementation
//===----------------------------------------------------------------------===//

void TargetLowering::DAGCombinerInfo::AddToWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->AddToWorkList(N);
}

void TargetLowering::DAGCombinerInfo::RemoveFromWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->removeFromWorkList(N);
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
                               const TargetLowering &TLI,
                               const TargetOptions *Options,
                               unsigned Depth = 0) {
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
    if (!Options->UnsafeFPMath) return 0;

    // After operation legalization, it might not be legal to create new FSUBs.
    if (LegalOperations &&
        !TLI.isOperationLegalOrCustom(ISD::FSUB,  Op.getValueType()))
      return 0;

    // fold (fneg (fadd A, B)) -> (fsub (fneg A), B)
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI,
                                    Options, Depth + 1))
      return V;
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return isNegatibleForFree(Op.getOperand(1), LegalOperations, TLI, Options,
                              Depth + 1);
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    if (!Options->UnsafeFPMath) return 0;

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return 1;

  case ISD::FMUL:
  case ISD::FDIV:
    if (Options->HonorSignDependentRoundingFPMath()) return 0;

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y) or (fmul X, (fneg Y))
    if (char V = isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI,
                                    Options, Depth + 1))
      return V;

    return isNegatibleForFree(Op.getOperand(1), LegalOperations, TLI, Options,
                              Depth + 1);

  case ISD::FP_EXTEND:
  case ISD::FP_ROUND:
  case ISD::FSIN:
    return isNegatibleForFree(Op.getOperand(0), LegalOperations, TLI, Options,
                              Depth + 1);
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
    assert(DAG.getTarget().Options.UnsafeFPMath);

    // fold (fneg (fadd A, B)) -> (fsub (fneg A), B)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations,
                           DAG.getTargetLoweringInfo(),
                           &DAG.getTarget().Options, Depth+1))
      return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));
    // fold (fneg (fadd A, B)) -> (fsub (fneg B), A)
    return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1),
                       Op.getOperand(0));
  case ISD::FSUB:
    // We can't turn -(A-B) into B-A when we honor signed zeros.
    assert(DAG.getTarget().Options.UnsafeFPMath);

    // fold (fneg (fsub 0, B)) -> B
    if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(Op.getOperand(0)))
      if (N0CFP->getValueAPF().isZero())
        return Op.getOperand(1);

    // fold (fneg (fsub A, B)) -> (fsub B, A)
    return DAG.getNode(ISD::FSUB, SDLoc(Op), Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(0));

  case ISD::FMUL:
  case ISD::FDIV:
    assert(!DAG.getTarget().Options.HonorSignDependentRoundingFPMath());

    // fold (fneg (fmul X, Y)) -> (fmul (fneg X), Y)
    if (isNegatibleForFree(Op.getOperand(0), LegalOperations,
                           DAG.getTargetLoweringInfo(),
                           &DAG.getTarget().Options, Depth+1))
      return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                         GetNegatedExpression(Op.getOperand(0), DAG,
                                              LegalOperations, Depth+1),
                         Op.getOperand(1));

    // fold (fneg (fmul X, Y)) -> (fmul X, (fneg Y))
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                       Op.getOperand(0),
                       GetNegatedExpression(Op.getOperand(1), DAG,
                                            LegalOperations, Depth+1));

  case ISD::FP_EXTEND:
  case ISD::FSIN:
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), Op.getValueType(),
                       GetNegatedExpression(Op.getOperand(0), DAG,
                                            LegalOperations, Depth+1));
  case ISD::FP_ROUND:
      return DAG.getNode(ISD::FP_ROUND, SDLoc(Op), Op.getValueType(),
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

SDValue DAGCombiner::ReassociateOps(unsigned Opc, SDLoc DL,
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
    }
    if (N0.hasOneUse()) {
      // reassoc. (op (op x, c1), y) -> (op (op x, y), c1) iff x+c1 has one use
      SDValue OpNode = DAG.getNode(Opc, SDLoc(N0), VT,
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
    }
    if (N1.hasOneUse()) {
      // reassoc. (op y, (op x, c1)) -> (op (op x, y), c1) iff x+c1 has one use
      SDValue OpNode = DAG.getNode(Opc, SDLoc(N0), VT,
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
  DAG.ReplaceAllUsesWith(N, To);
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

void DAGCombiner::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  // Replace all uses.  If any nodes become isomorphic to other nodes and
  // are deleted, make sure to remove them from our worklist.
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(TLO.Old, TLO.New);

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
  TargetLowering::TargetLoweringOpt TLO(DAG, LegalTypes, LegalOperations);
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

void DAGCombiner::ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad) {
  SDLoc dl(Load);
  EVT VT = Load->getValueType(0);
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, dl, VT, SDValue(ExtLoad, 0));

  DEBUG(dbgs() << "\nReplacing.9 ";
        Load->dump(&DAG);
        dbgs() << "\nWith: ";
        Trunc.getNode()->dump(&DAG);
        dbgs() << '\n');
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 0), Trunc);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 1), SDValue(ExtLoad, 1));
  removeFromWorkList(Load);
  DAG.DeleteNode(Load);
  AddToWorkList(Trunc.getNode());
}

SDValue DAGCombiner::PromoteOperand(SDValue Op, EVT PVT, bool &Replace) {
  Replace = false;
  SDLoc dl(Op);
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(Op)) {
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD)
      ? (TLI.isLoadExtLegal(ISD::ZEXTLOAD, MemVT) ? ISD::ZEXTLOAD
                                                  : ISD::EXTLOAD)
      : LD->getExtensionType();
    Replace = true;
    return DAG.getExtLoad(ExtType, dl, PVT,
                          LD->getChain(), LD->getBasePtr(),
                          MemVT, LD->getMemOperand());
  }

  unsigned Opc = Op.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::AssertSext:
    return DAG.getNode(ISD::AssertSext, dl, PVT,
                       SExtPromoteOperand(Op.getOperand(0), PVT),
                       Op.getOperand(1));
  case ISD::AssertZext:
    return DAG.getNode(ISD::AssertZext, dl, PVT,
                       ZExtPromoteOperand(Op.getOperand(0), PVT),
                       Op.getOperand(1));
  case ISD::Constant: {
    unsigned ExtOpc =
      Op.getValueType().isByteSized() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    return DAG.getNode(ExtOpc, dl, PVT, Op);
  }
  }

  if (!TLI.isOperationLegal(ISD::ANY_EXTEND, PVT))
    return SDValue();
  return DAG.getNode(ISD::ANY_EXTEND, dl, PVT, Op);
}

SDValue DAGCombiner::SExtPromoteOperand(SDValue Op, EVT PVT) {
  if (!TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, PVT))
    return SDValue();
  EVT OldVT = Op.getValueType();
  SDLoc dl(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (NewOp.getNode() == 0)
    return SDValue();
  AddToWorkList(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NewOp.getValueType(), NewOp,
                     DAG.getValueType(OldVT));
}

SDValue DAGCombiner::ZExtPromoteOperand(SDValue Op, EVT PVT) {
  EVT OldVT = Op.getValueType();
  SDLoc dl(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (NewOp.getNode() == 0)
    return SDValue();
  AddToWorkList(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getZeroExtendInReg(NewOp, dl, OldVT);
}

/// PromoteIntBinOp - Promote the specified integer binary operation if the
/// target indicates it is beneficial. e.g. On x86, it's usually better to
/// promote i16 operations to i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntBinOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    bool Replace0 = false;
    SDValue N0 = Op.getOperand(0);
    SDValue NN0 = PromoteOperand(N0, PVT, Replace0);
    if (NN0.getNode() == 0)
      return SDValue();

    bool Replace1 = false;
    SDValue N1 = Op.getOperand(1);
    SDValue NN1;
    if (N0 == N1)
      NN1 = NN0;
    else {
      NN1 = PromoteOperand(N1, PVT, Replace1);
      if (NN1.getNode() == 0)
        return SDValue();
    }

    AddToWorkList(NN0.getNode());
    if (NN1.getNode())
      AddToWorkList(NN1.getNode());

    if (Replace0)
      ReplaceLoadWithPromotedLoad(N0.getNode(), NN0.getNode());
    if (Replace1)
      ReplaceLoadWithPromotedLoad(N1.getNode(), NN1.getNode());

    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    SDLoc dl(Op);
    return DAG.getNode(ISD::TRUNCATE, dl, VT,
                       DAG.getNode(Opc, dl, PVT, NN0, NN1));
  }
  return SDValue();
}

/// PromoteIntShiftOp - Promote the specified integer shift operation if the
/// target indicates it is beneficial. e.g. On x86, it's usually better to
/// promote i16 operations to i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntShiftOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    bool Replace = false;
    SDValue N0 = Op.getOperand(0);
    if (Opc == ISD::SRA)
      N0 = SExtPromoteOperand(Op.getOperand(0), PVT);
    else if (Opc == ISD::SRL)
      N0 = ZExtPromoteOperand(Op.getOperand(0), PVT);
    else
      N0 = PromoteOperand(N0, PVT, Replace);
    if (N0.getNode() == 0)
      return SDValue();

    AddToWorkList(N0.getNode());
    if (Replace)
      ReplaceLoadWithPromotedLoad(Op.getOperand(0).getNode(), N0.getNode());

    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    SDLoc dl(Op);
    return DAG.getNode(ISD::TRUNCATE, dl, VT,
                       DAG.getNode(Opc, dl, PVT, N0, Op.getOperand(1)));
  }
  return SDValue();
}

SDValue DAGCombiner::PromoteExtend(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");
    // fold (aext (aext x)) -> (aext x)
    // fold (aext (zext x)) -> (zext x)
    // fold (aext (sext x)) -> (sext x)
    DEBUG(dbgs() << "\nPromoting ";
          Op.getNode()->dump(&DAG));
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), VT, Op.getOperand(0));
  }
  return SDValue();
}

bool DAGCombiner::PromoteLoad(SDValue Op) {
  if (!LegalOperations)
    return false;

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return false;

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return false;

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    SDLoc dl(Op);
    SDNode *N = Op.getNode();
    LoadSDNode *LD = cast<LoadSDNode>(N);
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD)
      ? (TLI.isLoadExtLegal(ISD::ZEXTLOAD, MemVT) ? ISD::ZEXTLOAD
                                                  : ISD::EXTLOAD)
      : LD->getExtensionType();
    SDValue NewLD = DAG.getExtLoad(ExtType, dl, PVT,
                                   LD->getChain(), LD->getBasePtr(),
                                   MemVT, LD->getMemOperand());
    SDValue Result = DAG.getNode(ISD::TRUNCATE, dl, VT, NewLD);

    DEBUG(dbgs() << "\nPromoting ";
          N->dump(&DAG);
          dbgs() << "\nTo: ";
          Result.getNode()->dump(&DAG);
          dbgs() << '\n');
    WorkListRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), NewLD.getValue(1));
    removeFromWorkList(N);
    DAG.DeleteNode(N);
    AddToWorkList(Result.getNode());
    return true;
  }
  return false;
}


//===----------------------------------------------------------------------===//
//  Main DAG Combiner implementation
//===----------------------------------------------------------------------===//

void DAGCombiner::Run(CombineLevel AtLevel) {
  // set the instance variables, so that the various visit routines may use it.
  Level = AtLevel;
  LegalOperations = Level >= AfterLegalizeVectorOps;
  LegalTypes = Level >= AfterLegalizeTypes;

  // Add all the dag nodes to the worklist.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I)
    AddToWorkList(I);

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // The root of the dag may dangle to deleted nodes until the dag combiner is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDValue());

  // while the worklist isn't empty, find a node and
  // try and combine it.
  while (!WorkListContents.empty()) {
    SDNode *N;
    // The WorkListOrder holds the SDNodes in order, but it may contain
    // duplicates.
    // In order to avoid a linear scan, we use a set (O(log N)) to hold what the
    // worklist *should* contain, and check the node we want to visit is should
    // actually be visited.
    do {
      N = WorkListOrder.pop_back_val();
    } while (!WorkListContents.erase(N));

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

    // Transfer debug value.
    DAG.TransferDbgValues(SDValue(N, 0), RV);
    WorkListRemover DeadNodes(*this);
    if (N->getNumValues() == RV.getNode()->getNumValues())
      DAG.ReplaceAllUsesWith(N, RV.getNode());
    else {
      assert(N->getValueType(0) == RV.getValueType() &&
             N->getNumValues() == 1 && "Type mismatch");
      SDValue OpV = RV;
      DAG.ReplaceAllUsesWith(N, &OpV);
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
  DAG.RemoveDeadNodes();
}

SDValue DAGCombiner::visit(SDNode *N) {
  switch (N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::MERGE_VALUES:       return visitMERGE_VALUES(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::ADDC:               return visitADDC(N);
  case ISD::SUBC:               return visitSUBC(N);
  case ISD::ADDE:               return visitADDE(N);
  case ISD::SUBE:               return visitSUBE(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:               return visitSREM(N);
  case ISD::UREM:               return visitUREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::SMUL_LOHI:          return visitSMUL_LOHI(N);
  case ISD::UMUL_LOHI:          return visitUMUL_LOHI(N);
  case ISD::SMULO:              return visitSMULO(N);
  case ISD::UMULO:              return visitUMULO(N);
  case ISD::SDIVREM:            return visitSDIVREM(N);
  case ISD::UDIVREM:            return visitUDIVREM(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTLZ_ZERO_UNDEF:    return visitCTLZ_ZERO_UNDEF(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTTZ_ZERO_UNDEF:    return visitCTTZ_ZERO_UNDEF(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SELECT:             return visitSELECT(N);
  case ISD::VSELECT:            return visitVSELECT(N);
  case ISD::SELECT_CC:          return visitSELECT_CC(N);
  case ISD::SETCC:              return visitSETCC(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::ANY_EXTEND:         return visitANY_EXTEND(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::BITCAST:            return visitBITCAST(N);
  case ISD::BUILD_PAIR:         return visitBUILD_PAIR(N);
  case ISD::FADD:               return visitFADD(N);
  case ISD::FSUB:               return visitFSUB(N);
  case ISD::FMUL:               return visitFMUL(N);
  case ISD::FMA:                return visitFMA(N);
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
  case ISD::FFLOOR:             return visitFFLOOR(N);
  case ISD::FCEIL:              return visitFCEIL(N);
  case ISD::FTRUNC:             return visitFTRUNC(N);
  case ISD::BRCOND:             return visitBRCOND(N);
  case ISD::BR_CC:              return visitBR_CC(N);
  case ISD::LOAD:               return visitLOAD(N);
  case ISD::STORE:              return visitSTORE(N);
  case ISD::INSERT_VECTOR_ELT:  return visitINSERT_VECTOR_ELT(N);
  case ISD::EXTRACT_VECTOR_ELT: return visitEXTRACT_VECTOR_ELT(N);
  case ISD::BUILD_VECTOR:       return visitBUILD_VECTOR(N);
  case ISD::CONCAT_VECTORS:     return visitCONCAT_VECTORS(N);
  case ISD::EXTRACT_SUBVECTOR:  return visitEXTRACT_SUBVECTOR(N);
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
        DagCombineInfo(DAG, Level, false, this);

      RV = TLI.PerformDAGCombine(N, DagCombineInfo);
    }
  }

  // If nothing happened still, try promoting the operation.
  if (RV.getNode() == 0) {
    switch (N->getOpcode()) {
    default: break;
    case ISD::ADD:
    case ISD::SUB:
    case ISD::MUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      RV = PromoteIntBinOp(SDValue(N, 0));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
      RV = PromoteIntShiftOp(SDValue(N, 0));
      break;
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::ANY_EXTEND:
      RV = PromoteExtend(SDValue(N, 0));
      break;
    case ISD::LOAD:
      if (PromoteLoad(SDValue(N, 0)))
        RV = SDValue(N, 0);
      break;
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
    if (N->getOperand(NumOps-1).getValueType() == MVT::Other)
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
      Result = DAG.getNode(ISD::TokenFactor, SDLoc(N),
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
  // First add the users of this node to the work list so that they
  // can be tried again once they have new operands.
  AddUsersToWorkList(N);
  do {
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, i), N->getOperand(i));
  } while (!N->use_empty());
  removeFromWorkList(N);
  DAG.DeleteNode(N);
  return SDValue(N, 0);   // Return N so it doesn't get rechecked!
}

static
SDValue combineShlAddConstant(SDLoc DL, SDValue N0, SDValue N1,
                              SelectionDAG &DAG) {
  EVT VT = N0.getValueType();
  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);
  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N01);

  if (N01C && N00.getOpcode() == ISD::ADD && N00.getNode()->hasOneUse() &&
      isa<ConstantSDNode>(N00.getOperand(1))) {
    // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
    N0 = DAG.getNode(ISD::ADD, SDLoc(N0), VT,
                     DAG.getNode(ISD::SHL, SDLoc(N00), VT,
                                 N00.getOperand(0), N01),
                     DAG.getNode(ISD::SHL, SDLoc(N01), VT,
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

    // fold (add x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
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
    return DAG.getNode(ISD::ADD, SDLoc(N), VT, N1, N0);
  // fold (add x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (add Sym, c) -> Sym+c
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA) && N1C &&
        GA->getOpcode() == ISD::GlobalAddress)
      return DAG.getGlobalAddress(GA->getGlobal(), SDLoc(N1C), VT,
                                  GA->getOffset() +
                                    (uint64_t)N1C->getSExtValue());
  // fold ((c1-A)+c2) -> (c1+c2)-A
  if (N1C && N0.getOpcode() == ISD::SUB)
    if (ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getOperand(0)))
      return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                         DAG.getConstant(N1C->getAPIntValue()+
                                         N0C->getAPIntValue(), VT),
                         N0.getOperand(1));
  // reassociate add
  SDValue RADD = ReassociateOps(ISD::ADD, SDLoc(N), N0, N1);
  if (RADD.getNode() != 0)
    return RADD;
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);
  // fold ((B-A)+A) -> B
  if (N0.getOpcode() == ISD::SUB && N1 == N0.getOperand(1))
    return N0.getOperand(0);
  // fold (A+(B-(A+C))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(0))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(1));
  // fold (A+(B-(C+A))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(1))
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(0));
  // fold (A+((B-A)+or-C)) to (B+or-C)
  if ((N1.getOpcode() == ISD::SUB || N1.getOpcode() == ISD::ADD) &&
      N1.getOperand(0).getOpcode() == ISD::SUB &&
      N0 == N1.getOperand(0).getOperand(1))
    return DAG.getNode(N1.getOpcode(), SDLoc(N), VT,
                       N1.getOperand(0).getOperand(0), N1.getOperand(1));

  // fold (A-B)+(C-D) to (A+C)-(B+D) when A or C is constant
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB) {
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    SDValue N10 = N1.getOperand(0);
    SDValue N11 = N1.getOperand(1);

    if (isa<ConstantSDNode>(N00) || isa<ConstantSDNode>(N10))
      return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                         DAG.getNode(ISD::ADD, SDLoc(N0), VT, N00, N10),
                         DAG.getNode(ISD::ADD, SDLoc(N1), VT, N01, N11));
  }

  if (!VT.isVector() && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (a+b) -> (a|b) iff a and b share no bits.
  if (VT.isInteger() && !VT.isVector()) {
    APInt LHSZero, LHSOne;
    APInt RHSZero, RHSOne;
    DAG.ComputeMaskedBits(N0, LHSZero, LHSOne);

    if (LHSZero.getBoolValue()) {
      DAG.ComputeMaskedBits(N1, RHSZero, RHSOne);

      // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
      // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
      if ((RHSZero & ~LHSZero) == ~LHSZero || (LHSZero & ~RHSZero) == ~RHSZero)
        return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N1);
    }
  }

  // fold (add (shl (add x, c1), c2), ) -> (add (add (shl x, c2), c1<<c2), )
  if (N0.getOpcode() == ISD::SHL && N0.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(SDLoc(N), N0, N1, DAG);
    if (Result.getNode()) return Result;
  }
  if (N1.getOpcode() == ISD::SHL && N1.getNode()->hasOneUse()) {
    SDValue Result = combineShlAddConstant(SDLoc(N), N1, N0, DAG);
    if (Result.getNode()) return Result;
  }

  // fold (add x, shl(0 - y, n)) -> sub(x, shl(y, n))
  if (N1.getOpcode() == ISD::SHL &&
      N1.getOperand(0).getOpcode() == ISD::SUB)
    if (ConstantSDNode *C =
          dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(0)))
      if (C->getAPIntValue() == 0)
        return DAG.getNode(ISD::SUB, SDLoc(N), VT, N0,
                           DAG.getNode(ISD::SHL, SDLoc(N), VT,
                                       N1.getOperand(0).getOperand(1),
                                       N1.getOperand(1)));
  if (N0.getOpcode() == ISD::SHL &&
      N0.getOperand(0).getOpcode() == ISD::SUB)
    if (ConstantSDNode *C =
          dyn_cast<ConstantSDNode>(N0.getOperand(0).getOperand(0)))
      if (C->getAPIntValue() == 0)
        return DAG.getNode(ISD::SUB, SDLoc(N), VT, N1,
                           DAG.getNode(ISD::SHL, SDLoc(N), VT,
                                       N0.getOperand(0).getOperand(1),
                                       N0.getOperand(1)));

  if (N1.getOpcode() == ISD::AND) {
    SDValue AndOp0 = N1.getOperand(0);
    ConstantSDNode *AndOp1 = dyn_cast<ConstantSDNode>(N1->getOperand(1));
    unsigned NumSignBits = DAG.ComputeNumSignBits(AndOp0);
    unsigned DestBits = VT.getScalarType().getSizeInBits();

    // (add z, (and (sbbl x, x), 1)) -> (sub z, (sbbl x, x))
    // and similar xforms where the inner op is either ~0 or 0.
    if (NumSignBits == DestBits && AndOp1 && AndOp1->isOne()) {
      SDLoc DL(N);
      return DAG.getNode(ISD::SUB, DL, VT, N->getOperand(0), AndOp0);
    }
  }

  // add (sext i1), X -> sub X, (zext i1)
  if (N0.getOpcode() == ISD::SIGN_EXTEND &&
      N0.getOperand(0).getValueType() == MVT::i1 &&
      !TLI.isOperationLegal(ISD::SIGN_EXTEND, MVT::i1)) {
    SDLoc DL(N);
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0));
    return DAG.getNode(ISD::SUB, DL, VT, N1, ZExt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // If the flag result is dead, turn this into an ADD.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::ADD, SDLoc(N), VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE,
                                 SDLoc(N), MVT::Glue));

  // canonicalize constant to RHS.
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDC, SDLoc(N), N->getVTList(), N1, N0);

  // fold (addc x, 0) -> x + no carry out
  if (N1C && N1C->isNullValue())
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE,
                                        SDLoc(N), MVT::Glue));

  // fold (addc a, b) -> (or a, b), CARRY_FALSE iff a and b share no bits.
  APInt LHSZero, LHSOne;
  APInt RHSZero, RHSOne;
  DAG.ComputeMaskedBits(N0, LHSZero, LHSOne);

  if (LHSZero.getBoolValue()) {
    DAG.ComputeMaskedBits(N1, RHSZero, RHSOne);

    // If all possibly-set bits on the LHS are clear on the RHS, return an OR.
    // If all possibly-set bits on the RHS are clear on the LHS, return an OR.
    if ((RHSZero & ~LHSZero) == ~LHSZero || (LHSZero & ~RHSZero) == ~RHSZero)
      return CombineTo(N, DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N1),
                       DAG.getNode(ISD::CARRY_FALSE,
                                   SDLoc(N), MVT::Glue));
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
    return DAG.getNode(ISD::ADDE, SDLoc(N), N->getVTList(),
                       N1, N0, CarryIn);

  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::ADDC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

// Since it may not be valid to emit a fold to zero for vector initializers
// check if we can before folding.
static SDValue tryFoldToZero(SDLoc DL, const TargetLowering &TLI, EVT VT,
                             SelectionDAG &DAG,
                             bool LegalOperations, bool LegalTypes) {
  if (!VT.isVector())
    return DAG.getConstant(0, VT);
  if (!LegalOperations || TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return DAG.getConstant(0, VT);
  return SDValue();
}

SDValue DAGCombiner::visitSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.getNode());
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N1C1 = N1.getOpcode() != ISD::ADD ? 0 :
    dyn_cast<ConstantSDNode>(N1.getOperand(1).getNode());
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;

    // fold (sub x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (sub x, x) -> 0
  // FIXME: Refactor this and xor and other similar operations together.
  if (N0 == N1)
    return tryFoldToZero(SDLoc(N), TLI, VT, DAG, LegalOperations, LegalTypes);
  // fold (sub c1, c2) -> c1-c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::SUB, VT, N0C, N1C);
  // fold (sub x, c) -> (add x, -c)
  if (N1C)
    return DAG.getNode(ISD::ADD, SDLoc(N), VT, N0,
                       DAG.getConstant(-N1C->getAPIntValue(), VT));
  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1)
  if (N0C && N0C->isAllOnesValue())
    return DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0);
  // fold A-(A-B) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(0))
    return N1.getOperand(1);
  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);
  // fold C2-(A+C1) -> (C2-C1)-A
  if (N1.getOpcode() == ISD::ADD && N0C && N1C1) {
    SDValue NewC = DAG.getConstant(N0C->getAPIntValue() - N1C1->getAPIntValue(),
                                   VT);
    return DAG.getNode(ISD::SUB, SDLoc(N), VT, NewC,
                       N1.getOperand(0));
  }
  // fold ((A+(B+or-C))-B) -> A+or-C
  if (N0.getOpcode() == ISD::ADD &&
      (N0.getOperand(1).getOpcode() == ISD::SUB ||
       N0.getOperand(1).getOpcode() == ISD::ADD) &&
      N0.getOperand(1).getOperand(0) == N1)
    return DAG.getNode(N0.getOperand(1).getOpcode(), SDLoc(N), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(1));
  // fold ((A+(C+B))-B) -> A+C
  if (N0.getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::ADD, SDLoc(N), VT,
                       N0.getOperand(0), N0.getOperand(1).getOperand(0));
  // fold ((A-(B-C))-C) -> A-B
  if (N0.getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
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
        return DAG.getGlobalAddress(GA->getGlobal(), SDLoc(N1C), VT,
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

SDValue DAGCombiner::visitSUBC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // If the flag result is dead, turn this into an SUB.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                 MVT::Glue));

  // fold (subc x, x) -> 0 + no borrow
  if (N0 == N1)
    return CombineTo(N, DAG.getConstant(0, VT),
                     DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                 MVT::Glue));

  // fold (subc x, 0) -> x + no borrow
  if (N1C && N1C->isNullValue())
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                        MVT::Glue));

  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1) + no borrow
  if (N0C && N0C->isAllOnesValue())
    return CombineTo(N, DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE, SDLoc(N),
                                 MVT::Glue));

  return SDValue();
}

SDValue DAGCombiner::visitSUBE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // fold (sube x, y, false) -> (subc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::SUBC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

/// isConstantSplatVector - Returns true if N is a BUILD_VECTOR node whose
/// elements are all the same constant or undefined.
static bool isConstantSplatVector(SDNode *N, APInt& SplatValue) {
  BuildVectorSDNode *C = dyn_cast<BuildVectorSDNode>(N);
  if (!C)
    return false;

  APInt SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  EVT EltVT = N->getValueType(0).getVectorElementType();
  return (C->isConstantSplat(SplatValue, SplatUndef, SplatBitSize,
                             HasAnyUndefs) &&
          EltVT.getSizeInBits() >= SplatBitSize);
}

SDValue DAGCombiner::visitMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold (mul x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  bool N0IsConst = false;
  bool N1IsConst = false;
  APInt ConstValue0, ConstValue1;
  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;

    N0IsConst = isConstantSplatVector(N0.getNode(), ConstValue0);
    N1IsConst = isConstantSplatVector(N1.getNode(), ConstValue1);
  } else {
    N0IsConst = dyn_cast<ConstantSDNode>(N0) != 0;
    ConstValue0 = N0IsConst ? (dyn_cast<ConstantSDNode>(N0))->getAPIntValue()
                            : APInt();
    N1IsConst = dyn_cast<ConstantSDNode>(N1) != 0;
    ConstValue1 = N1IsConst ? (dyn_cast<ConstantSDNode>(N1))->getAPIntValue()
                            : APInt();
  }

  // fold (mul c1, c2) -> c1*c2
  if (N0IsConst && N1IsConst)
    return DAG.FoldConstantArithmetic(ISD::MUL, VT, N0.getNode(), N1.getNode());

  // canonicalize constant to RHS
  if (N0IsConst && !N1IsConst)
    return DAG.getNode(ISD::MUL, SDLoc(N), VT, N1, N0);
  // fold (mul x, 0) -> 0
  if (N1IsConst && ConstValue1 == 0)
    return N1;
  // We require a splat of the entire scalar bit width for non-contiguous
  // bit patterns.
  bool IsFullSplat =
    ConstValue1.getBitWidth() == VT.getScalarType().getSizeInBits();
  // fold (mul x, 1) -> x
  if (N1IsConst && ConstValue1 == 1 && IsFullSplat)
    return N0;
  // fold (mul x, -1) -> 0-x
  if (N1IsConst && ConstValue1.isAllOnesValue())
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                       DAG.getConstant(0, VT), N0);
  // fold (mul x, (1 << c)) -> x << c
  if (N1IsConst && ConstValue1.isPowerOf2() && IsFullSplat)
    return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0,
                       DAG.getConstant(ConstValue1.logBase2(),
                                       getShiftAmountTy(N0.getValueType())));
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1IsConst && (-ConstValue1).isPowerOf2() && IsFullSplat) {
    unsigned Log2Val = (-ConstValue1).logBase2();
    // FIXME: If the input is something that is easily negated (e.g. a
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                       DAG.getConstant(0, VT),
                       DAG.getNode(ISD::SHL, SDLoc(N), VT, N0,
                            DAG.getConstant(Log2Val,
                                      getShiftAmountTy(N0.getValueType()))));
  }

  APInt Val;
  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N1IsConst && N0.getOpcode() == ISD::SHL &&
      (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                     isa<ConstantSDNode>(N0.getOperand(1)))) {
    SDValue C3 = DAG.getNode(ISD::SHL, SDLoc(N), VT,
                             N1, N0.getOperand(1));
    AddToWorkList(C3.getNode());
    return DAG.getNode(ISD::MUL, SDLoc(N), VT,
                       N0.getOperand(0), C3);
  }

  // Change (mul (shl X, C), Y) -> (shl (mul X, Y), C) when the shift has one
  // use.
  {
    SDValue Sh(0,0), Y(0,0);
    // Check for both (mul (shl X, C), Y)  and  (mul Y, (shl X, C)).
    if (N0.getOpcode() == ISD::SHL &&
        (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                       isa<ConstantSDNode>(N0.getOperand(1))) &&
        N0.getNode()->hasOneUse()) {
      Sh = N0; Y = N1;
    } else if (N1.getOpcode() == ISD::SHL &&
               isa<ConstantSDNode>(N1.getOperand(1)) &&
               N1.getNode()->hasOneUse()) {
      Sh = N1; Y = N0;
    }

    if (Sh.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, SDLoc(N), VT,
                         Mul, Sh.getOperand(1));
    }
  }

  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (N1IsConst && N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse() &&
      (isConstantSplatVector(N0.getOperand(1).getNode(), Val) ||
                     isa<ConstantSDNode>(N0.getOperand(1))))
    return DAG.getNode(ISD::ADD, SDLoc(N), VT,
                       DAG.getNode(ISD::MUL, SDLoc(N0), VT,
                                   N0.getOperand(0), N1),
                       DAG.getNode(ISD::MUL, SDLoc(N1), VT,
                                   N0.getOperand(1), N1));

  // reassociate mul
  SDValue RMUL = ReassociateOps(ISD::MUL, SDLoc(N), N0, N1);
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
  if (N1C && N1C->getAPIntValue() == 1LL)
    return N0;
  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                       DAG.getConstant(0, VT), N0);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  if (!VT.isVector()) {
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UDIV, SDLoc(N), N1.getValueType(),
                         N0, N1);
  }
  // fold (sdiv X, pow2) -> simple ops after legalize
  if (N1C && !N1C->isNullValue() &&
      (N1C->getAPIntValue().isPowerOf2() ||
       (-N1C->getAPIntValue()).isPowerOf2())) {
    // If dividing by powers of two is cheap, then don't perform the following
    // fold.
    if (TLI.isPow2DivCheap())
      return SDValue();

    unsigned lg2 = N1C->getAPIntValue().countTrailingZeros();

    // Splat the sign bit into the register
    SDValue SGN = DAG.getNode(ISD::SRA, SDLoc(N), VT, N0,
                              DAG.getConstant(VT.getSizeInBits()-1,
                                       getShiftAmountTy(N0.getValueType())));
    AddToWorkList(SGN.getNode());

    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDValue SRL = DAG.getNode(ISD::SRL, SDLoc(N), VT, SGN,
                              DAG.getConstant(VT.getSizeInBits() - lg2,
                                       getShiftAmountTy(SGN.getValueType())));
    SDValue ADD = DAG.getNode(ISD::ADD, SDLoc(N), VT, N0, SRL);
    AddToWorkList(SRL.getNode());
    AddToWorkList(ADD.getNode());    // Divide by pow2
    SDValue SRA = DAG.getNode(ISD::SRA, SDLoc(N), VT, ADD,
                  DAG.getConstant(lg2, getShiftAmountTy(ADD.getValueType())));

    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (N1C->getAPIntValue().isNonNegative())
      return SRA;

    AddToWorkList(SRA.getNode());
    return DAG.getNode(ISD::SUB, SDLoc(N), VT,
                       DAG.getConstant(0, VT), SRA);
  }

  // if integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.
  if (N1C && !N1C->isNullValue() && !TLI.isIntDivCheap()) {
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
    return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0,
                       DAG.getConstant(N1C->getAPIntValue().logBase2(),
                                       getShiftAmountTy(N0.getValueType())));
  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        EVT ADDVT = N1.getOperand(1).getValueType();
        SDValue Add = DAG.getNode(ISD::ADD, SDLoc(N), ADDVT,
                                  N1.getOperand(1),
                                  DAG.getConstant(SHC->getAPIntValue()
                                                                  .logBase2(),
                                                  ADDVT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, Add);
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
      return DAG.getNode(ISD::UREM, SDLoc(N), VT, N0, N1);
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::SDIV, SDLoc(N), VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, Mul);
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
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0,
                       DAG.getConstant(N1C->getAPIntValue()-1,VT));
  // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
  if (N1.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *SHC = dyn_cast<ConstantSDNode>(N1.getOperand(0))) {
      if (SHC->getAPIntValue().isPowerOf2()) {
        SDValue Add =
          DAG.getNode(ISD::ADD, SDLoc(N), VT, N1,
                 DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()),
                                 VT));
        AddToWorkList(Add.getNode());
        return DAG.getNode(ISD::AND, SDLoc(N), VT, N0, Add);
      }
    }
  }

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  if (N1C && !N1C->isNullValue()) {
    SDValue Div = DAG.getNode(ISD::UDIV, SDLoc(N), VT, N0, N1);
    AddToWorkList(Div.getNode());
    SDValue OptimizedDiv = combine(Div.getNode());
    if (OptimizedDiv.getNode() && OptimizedDiv.getNode() != Div.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT,
                                OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, SDLoc(N), VT, N0, Mul);
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
  SDLoc DL(N);

  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getNode(ISD::SRA, SDLoc(N), N0.getValueType(), N0,
                       DAG.getConstant(N0.getValueType().getSizeInBits() - 1,
                                       getShiftAmountTy(N0.getValueType())));
  // fold (mulhs x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  // If the type twice as wide is legal, transform the mulhs to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitMULHU(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // fold (mulhu x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhu x, 1) -> 0
  if (N1C && N1C->getAPIntValue() == 1)
    return DAG.getConstant(0, N0.getValueType());
  // fold (mulhu x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);

  // If the type twice as wide is legal, transform the mulhu to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

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
    SDValue Res = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0),
                              N->op_begin(), N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If the low half is not needed, just compute the high half.
  bool LoExists = N->hasAnyUseOfValue(0);
  if (!LoExists &&
      (!LegalOperations ||
       TLI.isOperationLegal(HiOp, N->getValueType(1)))) {
    SDValue Res = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1),
                              N->op_begin(), N->getNumOperands());
    return CombineTo(N, Res, Res);
  }

  // If both halves are used, return as it is.
  if (LoExists && HiExists)
    return SDValue();

  // If the two computed results can be simplified separately, separate them.
  if (LoExists) {
    SDValue Lo = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0),
                             N->op_begin(), N->getNumOperands());
    AddToWorkList(Lo.getNode());
    SDValue LoOpt = combine(Lo.getNode());
    if (LoOpt.getNode() && LoOpt.getNode() != Lo.getNode() &&
        (!LegalOperations ||
         TLI.isOperationLegal(LoOpt.getOpcode(), LoOpt.getValueType())))
      return CombineTo(N, LoOpt, LoOpt);
  }

  if (HiExists) {
    SDValue Hi = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1),
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

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // If the type twice as wide is legal, transform the mulhu to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitUMUL_LOHI(SDNode *N) {
  SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHU);
  if (Res.getNode()) return Res;

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // If the type twice as wide is legal, transform the mulhu to a wider multiply
  // plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSMULO(SDNode *N) {
  // (smulo x, 2) -> (saddo x, x)
  if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    if (C2->getAPIntValue() == 2)
      return DAG.getNode(ISD::SADDO, SDLoc(N), N->getVTList(),
                         N->getOperand(0), N->getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitUMULO(SDNode *N) {
  // (umulo x, 2) -> (uaddo x, x)
  if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    if (C2->getAPIntValue() == 2)
      return DAG.getNode(ISD::UADDO, SDLoc(N), N->getVTList(),
                         N->getOperand(0), N->getOperand(0));

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
  // fold (OP (trunc x), (trunc y)) -> (trunc (OP x, y)) (if trunc isn't free)
  //
  // do not sink logical op inside of a vector extend, since it may combine
  // into a vsetcc.
  EVT Op0VT = N0.getOperand(0).getValueType();
  if ((N0.getOpcode() == ISD::ZERO_EXTEND ||
       N0.getOpcode() == ISD::SIGN_EXTEND ||
       // Avoid infinite looping with PromoteIntBinOp.
       (N0.getOpcode() == ISD::ANY_EXTEND &&
        (!LegalTypes || TLI.isTypeDesirableForOp(N->getOpcode(), Op0VT))) ||
       (N0.getOpcode() == ISD::TRUNCATE &&
        (!TLI.isZExtFree(VT, Op0VT) ||
         !TLI.isTruncateFree(Op0VT, VT)) &&
        TLI.isTypeLegal(Op0VT))) &&
      !VT.isVector() &&
      Op0VT == N1.getOperand(0).getValueType() &&
      (!LegalOperations || TLI.isOperationLegal(N->getOpcode(), Op0VT))) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), SDLoc(N0),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, ORNode);
  }

  // For each of OP in SHL/SRL/SRA/AND...
  //   fold (and (OP x, z), (OP y, z)) -> (OP (and x, y), z)
  //   fold (or  (OP x, z), (OP y, z)) -> (OP (or  x, y), z)
  //   fold (xor (OP x, z), (OP y, z)) -> (OP (xor x, y), z)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL ||
       N0.getOpcode() == ISD::SRA || N0.getOpcode() == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDValue ORNode = DAG.getNode(N->getOpcode(), SDLoc(N0),
                                 N0.getOperand(0).getValueType(),
                                 N0.getOperand(0), N1.getOperand(0));
    AddToWorkList(ORNode.getNode());
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                       ORNode, N0.getOperand(1));
  }

  // Simplify xor/and/or (bitcast(A), bitcast(B)) -> bitcast(op (A,B))
  // Only perform this optimization after type legalization and before
  // LegalizeVectorOprs. LegalizeVectorOprs promotes vector operations by
  // adding bitcasts. For example (xor v4i32) is promoted to (v2i64), and
  // we don't want to undo this promotion.
  // We also handle SCALAR_TO_VECTOR because xor/or/and operations are cheaper
  // on scalars.
  if ((N0.getOpcode() == ISD::BITCAST ||
       N0.getOpcode() == ISD::SCALAR_TO_VECTOR) &&
      Level == AfterLegalizeTypes) {
    SDValue In0 = N0.getOperand(0);
    SDValue In1 = N1.getOperand(0);
    EVT In0Ty = In0.getValueType();
    EVT In1Ty = In1.getValueType();
    SDLoc DL(N);
    // If both incoming values are integers, and the original types are the
    // same.
    if (In0Ty.isInteger() && In1Ty.isInteger() && In0Ty == In1Ty) {
      SDValue Op = DAG.getNode(N->getOpcode(), DL, In0Ty, In0, In1);
      SDValue BC = DAG.getNode(N0.getOpcode(), DL, VT, Op);
      AddToWorkList(Op.getNode());
      return BC;
    }
  }

  // Xor/and/or are indifferent to the swizzle operation (shuffle of one value).
  // Simplify xor/and/or (shuff(A), shuff(B)) -> shuff(op (A,B))
  // If both shuffles use the same mask, and both shuffle within a single
  // vector, then it is worthwhile to move the swizzle after the operation.
  // The type-legalizer generates this pattern when loading illegal
  // vector types from memory. In many cases this allows additional shuffle
  // optimizations.
  if (N0.getOpcode() == ISD::VECTOR_SHUFFLE && Level < AfterLegalizeDAG &&
      N0.getOperand(1).getOpcode() == ISD::UNDEF &&
      N1.getOperand(1).getOpcode() == ISD::UNDEF) {
    ShuffleVectorSDNode *SVN0 = cast<ShuffleVectorSDNode>(N0);
    ShuffleVectorSDNode *SVN1 = cast<ShuffleVectorSDNode>(N1);

    assert(N0.getOperand(0).getValueType() == N1.getOperand(1).getValueType() &&
           "Inputs to shuffles are not the same type");

    unsigned NumElts = VT.getVectorNumElements();

    // Check that both shuffles use the same mask. The masks are known to be of
    // the same length because the result vector type is the same.
    bool SameMask = true;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx0 = SVN0->getMaskElt(i);
      int Idx1 = SVN1->getMaskElt(i);
      if (Idx0 != Idx1) {
        SameMask = false;
        break;
      }
    }

    if (SameMask) {
      SDValue Op = DAG.getNode(N->getOpcode(), SDLoc(N), VT,
                               N0.getOperand(0), N1.getOperand(0));
      AddToWorkList(Op.getNode());
      return DAG.getVectorShuffle(VT, SDLoc(N), Op,
                                  DAG.getUNDEF(VT), &SVN0->getMask()[0]);
    }
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

    // fold (and x, 0) -> 0, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N0;
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N1;

    // fold (and x, -1) -> x, vector edition
    if (ISD::isBuildVectorAllOnes(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllOnes(N1.getNode()))
      return N0;
  }

  // fold (and x, undef) -> 0
  if (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
  // fold (and c1, c2) -> c1&c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::AND, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N1, N0);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(BitWidth)))
    return DAG.getConstant(0, VT);
  // reassociate and
  SDValue RAND = ReassociateOps(ISD::AND, SDLoc(N), N0, N1);
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
    Mask = Mask.trunc(N0Op0.getValueSizeInBits());
    if (DAG.MaskedValueIsZero(N0Op0, Mask)) {
      SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N),
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
  // similarly fold (and (X (load ([non_ext|any_ext|zero_ext] V))), c) ->
  // (X (load ([non_ext|zero_ext] V))) if 'and' only clears top bits which must
  // already be zero by virtue of the width of the base type of the load.
  //
  // the 'X' node here can either be nothing or an extract_vector_elt to catch
  // more cases.
  if ((N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
       N0.getOperand(0).getOpcode() == ISD::LOAD) ||
      N0.getOpcode() == ISD::LOAD) {
    LoadSDNode *Load = cast<LoadSDNode>( (N0.getOpcode() == ISD::LOAD) ?
                                         N0 : N0.getOperand(0) );

    // Get the constant (if applicable) the zero'th operand is being ANDed with.
    // This can be a pure constant or a vector splat, in which case we treat the
    // vector as a scalar and use the splat value.
    APInt Constant = APInt::getNullValue(1);
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      Constant = C->getAPIntValue();
    } else if (BuildVectorSDNode *Vector = dyn_cast<BuildVectorSDNode>(N1)) {
      APInt SplatValue, SplatUndef;
      unsigned SplatBitSize;
      bool HasAnyUndefs;
      bool IsSplat = Vector->isConstantSplat(SplatValue, SplatUndef,
                                             SplatBitSize, HasAnyUndefs);
      if (IsSplat) {
        // Undef bits can contribute to a possible optimisation if set, so
        // set them.
        SplatValue |= SplatUndef;

        // The splat value may be something like "0x00FFFFFF", which means 0 for
        // the first vector value and FF for the rest, repeating. We need a mask
        // that will apply equally to all members of the vector, so AND all the
        // lanes of the constant together.
        EVT VT = Vector->getValueType(0);
        unsigned BitWidth = VT.getVectorElementType().getSizeInBits();

        // If the splat value has been compressed to a bitlength lower
        // than the size of the vector lane, we need to re-expand it to
        // the lane size.
        if (BitWidth > SplatBitSize)
          for (SplatValue = SplatValue.zextOrTrunc(BitWidth);
               SplatBitSize < BitWidth;
               SplatBitSize = SplatBitSize * 2)
            SplatValue |= SplatValue.shl(SplatBitSize);

        Constant = APInt::getAllOnesValue(BitWidth);
        for (unsigned i = 0, n = SplatBitSize/BitWidth; i < n; ++i)
          Constant &= SplatValue.lshr(i*BitWidth).zextOrTrunc(BitWidth);
      }
    }

    // If we want to change an EXTLOAD to a ZEXTLOAD, ensure a ZEXTLOAD is
    // actually legal and isn't going to get expanded, else this is a false
    // optimisation.
    bool CanZextLoadProfitably = TLI.isLoadExtLegal(ISD::ZEXTLOAD,
                                                    Load->getMemoryVT());

    // Resize the constant to the same size as the original memory access before
    // extension. If it is still the AllOnesValue then this AND is completely
    // unneeded.
    Constant =
      Constant.zextOrTrunc(Load->getMemoryVT().getScalarType().getSizeInBits());

    bool B;
    switch (Load->getExtensionType()) {
    default: B = false; break;
    case ISD::EXTLOAD: B = CanZextLoadProfitably; break;
    case ISD::ZEXTLOAD:
    case ISD::NON_EXTLOAD: B = true; break;
    }

    if (B && Constant.isAllOnesValue()) {
      // If the load type was an EXTLOAD, convert to ZEXTLOAD in order to
      // preserve semantics once we get rid of the AND.
      SDValue NewLoad(Load, 0);
      if (Load->getExtensionType() == ISD::EXTLOAD) {
        NewLoad = DAG.getLoad(Load->getAddressingMode(), ISD::ZEXTLOAD,
                              Load->getValueType(0), SDLoc(Load),
                              Load->getChain(), Load->getBasePtr(),
                              Load->getOffset(), Load->getMemoryVT(),
                              Load->getMemOperand());
        // Replace uses of the EXTLOAD with the new ZEXTLOAD.
        if (Load->getNumValues() == 3) {
          // PRE/POST_INC loads have 3 values.
          SDValue To[] = { NewLoad.getValue(0), NewLoad.getValue(1),
                           NewLoad.getValue(2) };
          CombineTo(Load, To, 3, true);
        } else {
          CombineTo(Load, NewLoad.getValue(0), NewLoad.getValue(1));
        }
      }

      // Fold the AND away, taking care not to fold to the old load node if we
      // replaced it.
      CombineTo(N, (N0.getNode() == Load) ? NewLoad : N0);

      return SDValue(N, 0); // Return N so it doesn't get rechecked!
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
        SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(N0),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(SDLoc(N), VT, ORNode, LR, Op1);
      }
      // fold (and (seteq X, -1), (seteq Y, -1)) -> (seteq (and X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETEQ) {
        SDValue ANDNode = DAG.getNode(ISD::AND, SDLoc(N0),
                                      LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(SDLoc(N), VT, ANDNode, LR, Op1);
      }
      // fold (and (setgt X,  -1), (setgt Y,  -1)) -> (setgt (or X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETGT) {
        SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(N0),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(SDLoc(N), VT, ORNode, LR, Op1);
      }
    }
    // Simplify (and (setne X, 0), (setne X, -1)) -> (setuge (add X, 1), 2)
    if (LL == RL && isa<ConstantSDNode>(LR) && isa<ConstantSDNode>(RR) &&
        Op0 == Op1 && LL.getValueType().isInteger() &&
      Op0 == ISD::SETNE && ((cast<ConstantSDNode>(LR)->isNullValue() &&
                                 cast<ConstantSDNode>(RR)->isAllOnesValue()) ||
                                (cast<ConstantSDNode>(LR)->isAllOnesValue() &&
                                 cast<ConstantSDNode>(RR)->isNullValue()))) {
      SDValue ADDNode = DAG.getNode(ISD::ADD, SDLoc(N0), LL.getValueType(),
                                    LL, DAG.getConstant(1, LL.getValueType()));
      AddToWorkList(ADDNode.getNode());
      return DAG.getSetCC(SDLoc(N), VT, ADDNode,
                          DAG.getConstant(2, LL.getValueType()), ISD::SETUGE);
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
          (!LegalOperations ||
           (TLI.isCondCodeLegal(Result, LL.getSimpleValueType()) &&
            TLI.isOperationLegal(ISD::SETCC,
                            getSetCCResultType(N0.getSimpleValueType())))))
        return DAG.getSetCC(SDLoc(N), N0.getValueType(),
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N0), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N0), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
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
        LN0->isUnindexed() && N0.hasOneUse() && SDValue(LN0, 0).hasOneUse()) {
      uint32_t ActiveBits = N1C->getAPIntValue().getActiveBits();
      if (ActiveBits > 0 && APIntOps::isMask(ActiveBits, N1C->getAPIntValue())){
        EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
        EVT LoadedVT = LN0->getMemoryVT();

        if (ExtVT == LoadedVT &&
            (!LegalOperations || TLI.isLoadExtLegal(ISD::ZEXTLOAD, ExtVT))) {
          EVT LoadResultTy = HasAnyExt ? LN0->getValueType(0) : VT;

          SDValue NewLoad =
            DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), LoadResultTy,
                           LN0->getChain(), LN0->getBasePtr(), ExtVT,
                           LN0->getMemOperand());
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
            NewPtr = DAG.getNode(ISD::ADD, SDLoc(LN0), PtrType,
                                 NewPtr, DAG.getConstant(PtrOff, PtrType));
            Alignment = MinAlign(Alignment, PtrOff);
          }

          AddToWorkList(NewPtr.getNode());

          EVT LoadResultTy = HasAnyExt ? LN0->getValueType(0) : VT;
          SDValue Load =
            DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), LoadResultTy,
                           LN0->getChain(), NewPtr,
                           LN0->getPointerInfo(),
                           ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                           Alignment, LN0->getTBAAInfo());
          AddToWorkList(N);
          CombineTo(LN0, Load, Load.getValue(1));
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }
  }

  if (N0.getOpcode() == ISD::ADD && N1.getOpcode() == ISD::SRL &&
      VT.getSizeInBits() <= 64) {
    if (ConstantSDNode *ADDI = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      APInt ADDC = ADDI->getAPIntValue();
      if (!TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
        // Look for (and (add x, c1), (lshr y, c2)). If C1 wasn't a legal
        // immediate for an add, but it is legal if its top c2 bits are set,
        // transform the ADD so the immediate doesn't need to be materialized
        // in a register.
        if (ConstantSDNode *SRLI = dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
          APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                             SRLI->getZExtValue());
          if (DAG.MaskedValueIsZero(N0.getOperand(1), Mask)) {
            ADDC |= Mask;
            if (TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
              SDValue NewAdd =
                DAG.getNode(ISD::ADD, SDLoc(N0), VT,
                            N0.getOperand(0), DAG.getConstant(ADDC, VT));
              CombineTo(N0.getNode(), NewAdd);
              return SDValue(N, 0); // Return N so it doesn't get rechecked!
            }
          }
        }
      }
    }
  }

  // fold (and (or (srl N, 8), (shl N, 8)), 0xffff) -> (srl (bswap N), const)
  if (N1C && N1C->getAPIntValue() == 0xffff && N0.getOpcode() == ISD::OR) {
    SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                       N0.getOperand(1), false);
    if (BSwap.getNode())
      return BSwap;
  }

  return SDValue();
}

/// MatchBSwapHWord - Match (a >> 8) | (a << 8) as (bswap a) >> 16
///
SDValue DAGCombiner::MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                                        bool DemandHighBits) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i64 && VT != MVT::i32 && VT != MVT::i16)
    return SDValue();
  if (!TLI.isOperationLegal(ISD::BSWAP, VT))
    return SDValue();

  // Recognize (and (shl a, 8), 0xff), (and (srl a, 8), 0xff00)
  bool LookPassAnd0 = false;
  bool LookPassAnd1 = false;
  if (N0.getOpcode() == ISD::AND && N0.getOperand(0).getOpcode() == ISD::SRL)
      std::swap(N0, N1);
  if (N1.getOpcode() == ISD::AND && N1.getOperand(0).getOpcode() == ISD::SHL)
      std::swap(N0, N1);
  if (N0.getOpcode() == ISD::AND) {
    if (!N0.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01C || N01C->getZExtValue() != 0xFF00)
      return SDValue();
    N0 = N0.getOperand(0);
    LookPassAnd0 = true;
  }

  if (N1.getOpcode() == ISD::AND) {
    if (!N1.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
    if (!N11C || N11C->getZExtValue() != 0xFF)
      return SDValue();
    N1 = N1.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N0.getOpcode() == ISD::SRL && N1.getOpcode() == ISD::SHL)
    std::swap(N0, N1);
  if (N0.getOpcode() != ISD::SHL || N1.getOpcode() != ISD::SRL)
    return SDValue();
  if (!N0.getNode()->hasOneUse() ||
      !N1.getNode()->hasOneUse())
    return SDValue();

  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
  if (!N01C || !N11C)
    return SDValue();
  if (N01C->getZExtValue() != 8 || N11C->getZExtValue() != 8)
    return SDValue();

  // Look for (shl (and a, 0xff), 8), (srl (and a, 0xff00), 8)
  SDValue N00 = N0->getOperand(0);
  if (!LookPassAnd0 && N00.getOpcode() == ISD::AND) {
    if (!N00.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N001C = dyn_cast<ConstantSDNode>(N00.getOperand(1));
    if (!N001C || N001C->getZExtValue() != 0xFF)
      return SDValue();
    N00 = N00.getOperand(0);
    LookPassAnd0 = true;
  }

  SDValue N10 = N1->getOperand(0);
  if (!LookPassAnd1 && N10.getOpcode() == ISD::AND) {
    if (!N10.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N10.getOperand(1));
    if (!N101C || N101C->getZExtValue() != 0xFF00)
      return SDValue();
    N10 = N10.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N00 != N10)
    return SDValue();

  // Make sure everything beyond the low halfword gets set to zero since the SRL
  // 16 will clear the top bits.
  unsigned OpSizeInBits = VT.getSizeInBits();
  if (DemandHighBits && OpSizeInBits > 16) {
    // If the left-shift isn't masked out then the only way this is a bswap is
    // if all bits beyond the low 8 are 0. In that case the entire pattern
    // reduces to a left shift anyway: leave it for other parts of the combiner.
    if (!LookPassAnd0)
      return SDValue();

    // However, if the right shift isn't masked out then it might be because
    // it's not needed. See if we can spot that too.
    if (!LookPassAnd1 &&
        !DAG.MaskedValueIsZero(
            N10, APInt::getHighBitsSet(OpSizeInBits, OpSizeInBits - 16)))
      return SDValue();
  }

  SDValue Res = DAG.getNode(ISD::BSWAP, SDLoc(N), VT, N00);
  if (OpSizeInBits > 16)
    Res = DAG.getNode(ISD::SRL, SDLoc(N), VT, Res,
                      DAG.getConstant(OpSizeInBits-16, getShiftAmountTy(VT)));
  return Res;
}

/// isBSwapHWordElement - Return true if the specified node is an element
/// that makes up a 32-bit packed halfword byteswap. i.e.
/// ((x&0xff)<<8)|((x&0xff00)>>8)|((x&0x00ff0000)<<8)|((x&0xff000000)>>8)
static bool isBSwapHWordElement(SDValue N, SmallVectorImpl<SDNode *> &Parts) {
  if (!N.getNode()->hasOneUse())
    return false;

  unsigned Opc = N.getOpcode();
  if (Opc != ISD::AND && Opc != ISD::SHL && Opc != ISD::SRL)
    return false;

  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (!N1C)
    return false;

  unsigned Num;
  switch (N1C->getZExtValue()) {
  default:
    return false;
  case 0xFF:       Num = 0; break;
  case 0xFF00:     Num = 1; break;
  case 0xFF0000:   Num = 2; break;
  case 0xFF000000: Num = 3; break;
  }

  // Look for (x & 0xff) << 8 as well as ((x << 8) & 0xff00).
  SDValue N0 = N.getOperand(0);
  if (Opc == ISD::AND) {
    if (Num == 0 || Num == 2) {
      // (x >> 8) & 0xff
      // (x >> 8) & 0xff0000
      if (N0.getOpcode() != ISD::SRL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    } else {
      // (x << 8) & 0xff00
      // (x << 8) & 0xff000000
      if (N0.getOpcode() != ISD::SHL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    }
  } else if (Opc == ISD::SHL) {
    // (x & 0xff) << 8
    // (x & 0xff0000) << 8
    if (Num != 0 && Num != 2)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  } else { // Opc == ISD::SRL
    // (x & 0xff00) >> 8
    // (x & 0xff000000) >> 8
    if (Num != 1 && Num != 3)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  }

  if (Parts[Num])
    return false;

  Parts[Num] = N0.getOperand(0).getNode();
  return true;
}

/// MatchBSwapHWord - Match a 32-bit packed halfword bswap. That is
/// ((x&0xff)<<8)|((x&0xff00)>>8)|((x&0x00ff0000)<<8)|((x&0xff000000)>>8)
/// => (rotl (bswap x), 16)
SDValue DAGCombiner::MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();
  if (!TLI.isOperationLegal(ISD::BSWAP, VT))
    return SDValue();

  SmallVector<SDNode*,4> Parts(4, (SDNode*)0);
  // Look for either
  // (or (or (and), (and)), (or (and), (and)))
  // (or (or (or (and), (and)), (and)), (and))
  if (N0.getOpcode() != ISD::OR)
    return SDValue();
  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);

  if (N1.getOpcode() == ISD::OR &&
      N00.getNumOperands() == 2 && N01.getNumOperands() == 2) {
    // (or (or (and), (and)), (or (and), (and)))
    SDValue N000 = N00.getOperand(0);
    if (!isBSwapHWordElement(N000, Parts))
      return SDValue();

    SDValue N001 = N00.getOperand(1);
    if (!isBSwapHWordElement(N001, Parts))
      return SDValue();
    SDValue N010 = N01.getOperand(0);
    if (!isBSwapHWordElement(N010, Parts))
      return SDValue();
    SDValue N011 = N01.getOperand(1);
    if (!isBSwapHWordElement(N011, Parts))
      return SDValue();
  } else {
    // (or (or (or (and), (and)), (and)), (and))
    if (!isBSwapHWordElement(N1, Parts))
      return SDValue();
    if (!isBSwapHWordElement(N01, Parts))
      return SDValue();
    if (N00.getOpcode() != ISD::OR)
      return SDValue();
    SDValue N000 = N00.getOperand(0);
    if (!isBSwapHWordElement(N000, Parts))
      return SDValue();
    SDValue N001 = N00.getOperand(1);
    if (!isBSwapHWordElement(N001, Parts))
      return SDValue();
  }

  // Make sure the parts are all coming from the same node.
  if (Parts[0] != Parts[1] || Parts[0] != Parts[2] || Parts[0] != Parts[3])
    return SDValue();

  SDValue BSwap = DAG.getNode(ISD::BSWAP, SDLoc(N), VT,
                              SDValue(Parts[0],0));

  // Result of the bswap should be rotated by 16. If it's not legal, then
  // do  (x << 16) | (x >> 16).
  SDValue ShAmt = DAG.getConstant(16, getShiftAmountTy(VT));
  if (TLI.isOperationLegalOrCustom(ISD::ROTL, VT))
    return DAG.getNode(ISD::ROTL, SDLoc(N), VT, BSwap, ShAmt);
  if (TLI.isOperationLegalOrCustom(ISD::ROTR, VT))
    return DAG.getNode(ISD::ROTR, SDLoc(N), VT, BSwap, ShAmt);
  return DAG.getNode(ISD::OR, SDLoc(N), VT,
                     DAG.getNode(ISD::SHL, SDLoc(N), VT, BSwap, ShAmt),
                     DAG.getNode(ISD::SRL, SDLoc(N), VT, BSwap, ShAmt));
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

    // fold (or x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;

    // fold (or x, -1) -> -1, vector edition
    if (ISD::isBuildVectorAllOnes(N0.getNode()))
      return N0;
    if (ISD::isBuildVectorAllOnes(N1.getNode()))
      return N1;
  }

  // fold (or x, undef) -> -1
  if (!LegalOperations &&
      (N0.getOpcode() == ISD::UNDEF || N1.getOpcode() == ISD::UNDEF)) {
    EVT EltVT = VT.isVector() ? VT.getVectorElementType() : VT;
    return DAG.getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), VT);
  }
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.FoldConstantArithmetic(ISD::OR, VT, N0C, N1C);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N1, N0);
  // fold (or x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (or x, -1) -> -1
  if (N1C && N1C->isAllOnesValue())
    return N1;
  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && DAG.MaskedValueIsZero(N0, ~N1C->getAPIntValue()))
    return N1;

  // Recognize halfword bswaps as (bswap + rotl 16) or (bswap + shl 16)
  SDValue BSwap = MatchBSwapHWord(N, N0, N1);
  if (BSwap.getNode() != 0)
    return BSwap;
  BSwap = MatchBSwapHWordLow(N, N0, N1);
  if (BSwap.getNode() != 0)
    return BSwap;

  // reassociate or
  SDValue ROR = ReassociateOps(ISD::OR, SDLoc(N), N0, N1);
  if (ROR.getNode() != 0)
    return ROR;
  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  // iff (c1 & c2) == 0.
  if (N1C && N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    if ((C1->getAPIntValue() & N1C->getAPIntValue()) != 0)
      return DAG.getNode(ISD::AND, SDLoc(N), VT,
                         DAG.getNode(ISD::OR, SDLoc(N0), VT,
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
        SDValue ORNode = DAG.getNode(ISD::OR, SDLoc(LR),
                                     LR.getValueType(), LL, RL);
        AddToWorkList(ORNode.getNode());
        return DAG.getSetCC(SDLoc(N), VT, ORNode, LR, Op1);
      }
      // fold (or (setne X, -1), (setne Y, -1)) -> (setne (and X, Y), -1)
      // fold (or (setgt X, -1), (setgt Y  -1)) -> (setgt (and X, Y), -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() &&
          (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDValue ANDNode = DAG.getNode(ISD::AND, SDLoc(LR),
                                      LR.getValueType(), LL, RL);
        AddToWorkList(ANDNode.getNode());
        return DAG.getSetCC(SDLoc(N), VT, ANDNode, LR, Op1);
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
          (!LegalOperations ||
           (TLI.isCondCodeLegal(Result, LL.getSimpleValueType()) &&
            TLI.isOperationLegal(ISD::SETCC,
              getSetCCResultType(N0.getValueType())))))
        return DAG.getSetCC(SDLoc(N), N0.getValueType(),
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
      SDValue X = DAG.getNode(ISD::OR, SDLoc(N0), VT,
                              N0.getOperand(0), N1.getOperand(0));
      return DAG.getNode(ISD::AND, SDLoc(N), VT, X,
                         DAG.getConstant(LHSMask | RHSMask, VT));
    }
  }

  // See if this is some rotate idiom.
  if (SDNode *Rot = MatchRotate(N0, N1, SDLoc(N)))
    return SDValue(Rot, 0);

  // Simplify the operands using demanded-bits information.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

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
SDNode *DAGCombiner::MatchRotate(SDValue LHS, SDValue RHS, SDLoc DL) {
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

    SDValue Rot = DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, DL, VT,
                              LHSShiftArg, HasROTL ? LHSShiftAmt : RHSShiftAmt);

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

  // If the shift amount is sign/zext/any-extended just peel it off.
  SDValue LExtOp0 = LHSShiftAmt;
  SDValue RExtOp0 = RHSShiftAmt;
  if ((LHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::TRUNCATE) &&
      (RHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::TRUNCATE)) {
    LExtOp0 = LHSShiftAmt.getOperand(0);
    RExtOp0 = RHSShiftAmt.getOperand(0);
  }

  if (RExtOp0.getOpcode() == ISD::SUB && RExtOp0.getOperand(1) == LExtOp0) {
    // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
    //   (rotl x, y)
    // fold (or (shl x, (*ext y)), (srl x, (*ext (sub 32, y)))) ->
    //   (rotr x, (sub 32, y))
    if (ConstantSDNode *SUBC =
            dyn_cast<ConstantSDNode>(RExtOp0.getOperand(0)))
      if (SUBC->getAPIntValue() == OpSizeInBits)
        return DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, DL, VT, LHSShiftArg,
                           HasROTL ? LHSShiftAmt : RHSShiftAmt).getNode();
  } else if (LExtOp0.getOpcode() == ISD::SUB &&
             RExtOp0 == LExtOp0.getOperand(1)) {
    // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext y))) ->
    //   (rotr x, y)
    // fold (or (shl x, (*ext (sub 32, y))), (srl x, (*ext y))) ->
    //   (rotl x, (sub 32, y))
    if (ConstantSDNode *SUBC =
            dyn_cast<ConstantSDNode>(LExtOp0.getOperand(0)))
      if (SUBC->getAPIntValue() == OpSizeInBits)
        return DAG.getNode(HasROTR ? ISD::ROTR : ISD::ROTL, DL, VT, LHSShiftArg,
                           HasROTR ? RHSShiftAmt : LHSShiftAmt).getNode();
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

    // fold (xor x, 0) -> x, vector edition
    if (ISD::isBuildVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isBuildVectorAllZeros(N1.getNode()))
      return N0;
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
    return DAG.getNode(ISD::XOR, SDLoc(N), VT, N1, N0);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // reassociate xor
  SDValue RXOR = ReassociateOps(ISD::XOR, SDLoc(N), N0, N1);
  if (RXOR.getNode() != 0)
    return RXOR;

  // fold !(x cc y) -> (x !cc y)
  if (N1C && N1C->getAPIntValue() == 1 && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = LHS.getValueType().isInteger();
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);

    if (!LegalOperations ||
        TLI.isCondCodeLegal(NotCC, LHS.getSimpleValueType())) {
      switch (N0.getOpcode()) {
      default:
        llvm_unreachable("Unhandled SetCC Equivalent!");
      case ISD::SETCC:
        return DAG.getSetCC(SDLoc(N), VT, LHS, RHS, NotCC);
      case ISD::SELECT_CC:
        return DAG.getSelectCC(SDLoc(N), LHS, RHS, N0.getOperand(2),
                               N0.getOperand(3), NotCC);
      }
    }
  }

  // fold (not (zext (setcc x, y))) -> (zext (not (setcc x, y)))
  if (N1C && N1C->getAPIntValue() == 1 && N0.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getNode()->hasOneUse() &&
      isSetCCEquivalent(N0.getOperand(0), LHS, RHS, CC)){
    SDValue V = N0.getOperand(0);
    V = DAG.getNode(ISD::XOR, SDLoc(N0), V.getValueType(), V,
                    DAG.getConstant(1, V.getValueType()));
    AddToWorkList(V.getNode());
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, V);
  }

  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are setcc
  if (N1C && N1C->getAPIntValue() == 1 && VT == MVT::i1 &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, SDLoc(LHS), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, SDLoc(RHS), VT, RHS, N1); // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, SDLoc(N), VT, LHS, RHS);
    }
  }
  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are constants
  if (N1C && N1C->isAllOnesValue() &&
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, SDLoc(LHS), VT, LHS, N1); // LHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, SDLoc(RHS), VT, RHS, N1); // RHS = ~RHS
      AddToWorkList(LHS.getNode()); AddToWorkList(RHS.getNode());
      return DAG.getNode(NewOpcode, SDLoc(N), VT, LHS, RHS);
    }
  }
  // fold (xor (and x, y), y) -> (and (not x), y)
  if (N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
      N0->getOperand(1) == N1) {
    SDValue X = N0->getOperand(0);
    SDValue NotX = DAG.getNOT(SDLoc(X), X, VT);
    AddToWorkList(NotX.getNode());
    return DAG.getNode(ISD::AND, SDLoc(N), VT, NotX, N1);
  }
  // fold (xor (xor x, c1), c2) -> (xor x, (xor c1, c2))
  if (N1C && N0.getOpcode() == ISD::XOR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::XOR, SDLoc(N), VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N00C->getAPIntValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::XOR, SDLoc(N), VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getAPIntValue() ^
                                         N01C->getAPIntValue(), VT));
  }
  // fold (xor x, x) -> 0
  if (N0 == N1)
    return tryFoldToZero(SDLoc(N), TLI, VT, DAG, LegalOperations, LegalTypes);

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
  SDValue NewRHS = DAG.getNode(N->getOpcode(), SDLoc(LHS->getOperand(1)),
                               N->getValueType(0),
                               LHS->getOperand(1), N->getOperand(1));

  // Create the new shift.
  SDValue NewShift = DAG.getNode(N->getOpcode(),
                                 SDLoc(LHS->getOperand(0)),
                                 VT, LHS->getOperand(0), N->getOperand(1));

  // Create the new binop.
  return DAG.getNode(LHS->getOpcode(), SDLoc(N), VT, NewShift, NewRHS);
}

SDValue DAGCombiner::visitSHL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

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
  // fold (shl undef, x) -> 0
  if (N0.getOpcode() == ISD::UNDEF)
    return DAG.getConstant(0, VT);
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
      TruncC = TruncC.trunc(TruncVT.getSizeInBits());
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0,
                         DAG.getNode(ISD::AND, SDLoc(N), TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 SDLoc(N),
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
    if (c1 + c2 >= OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }

  // fold (shl (ext (shl x, c1)), c2) -> (ext (shl x, (add c1, c2)))
  // For this to be valid, the second form must not preserve any of the bits
  // that are shifted out by the inner shift in the first form.  This means
  // the outer shift size must be >= the number of bits added by the ext.
  // As a corollary, we don't care what kind of ext it is.
  if (N1C && (N0.getOpcode() == ISD::ZERO_EXTEND ||
              N0.getOpcode() == ISD::ANY_EXTEND ||
              N0.getOpcode() == ISD::SIGN_EXTEND) &&
      N0.getOperand(0).getOpcode() == ISD::SHL &&
      isa<ConstantSDNode>(N0.getOperand(0)->getOperand(1))) {
    uint64_t c1 =
      cast<ConstantSDNode>(N0.getOperand(0)->getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    EVT InnerShiftVT = N0.getOperand(0).getValueType();
    uint64_t InnerShiftSize = InnerShiftVT.getScalarType().getSizeInBits();
    if (c2 >= OpSizeInBits - InnerShiftSize) {
      if (c1 + c2 >= OpSizeInBits)
        return DAG.getConstant(0, VT);
      return DAG.getNode(ISD::SHL, SDLoc(N0), VT,
                         DAG.getNode(N0.getOpcode(), SDLoc(N0), VT,
                                     N0.getOperand(0)->getOperand(0)),
                         DAG.getConstant(c1 + c2, N1.getValueType()));
    }
  }

  // fold (shl (zext (srl x, C)), C) -> (zext (shl (srl x, C), C))
  // Only fold this if the inner zext has no other uses to avoid increasing
  // the total number of instructions.
  if (N1C && N0.getOpcode() == ISD::ZERO_EXTEND && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::SRL &&
      isa<ConstantSDNode>(N0.getOperand(0)->getOperand(1))) {
    uint64_t c1 =
      cast<ConstantSDNode>(N0.getOperand(0)->getOperand(1))->getZExtValue();
    if (c1 < VT.getSizeInBits()) {
      uint64_t c2 = N1C->getZExtValue();
      if (c1 == c2) {
        SDValue NewOp0 = N0.getOperand(0);
        EVT CountVT = NewOp0.getOperand(1).getValueType();
        SDValue NewSHL = DAG.getNode(ISD::SHL, SDLoc(N), NewOp0.getValueType(),
                                     NewOp0, DAG.getConstant(c2, CountVT));
        AddToWorkList(NewSHL.getNode());
        return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N0), VT, NewSHL);
      }
    }
  }

  // fold (shl (srl x, c1), c2) -> (and (shl x, (sub c2, c1), MASK) or
  //                               (and (srl x, (sub c1, c2), MASK)
  // Only fold this if the inner shift has no other uses -- if it does, folding
  // this will increase the total number of instructions.
  if (N1C && N0.getOpcode() == ISD::SRL && N0.hasOneUse() &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();
    if (c1 < VT.getSizeInBits()) {
      uint64_t c2 = N1C->getZExtValue();
      APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                         VT.getSizeInBits() - c1);
      SDValue Shift;
      if (c2 > c1) {
        Mask = Mask.shl(c2-c1);
        Shift = DAG.getNode(ISD::SHL, SDLoc(N), VT, N0.getOperand(0),
                            DAG.getConstant(c2-c1, N1.getValueType()));
      } else {
        Mask = Mask.lshr(c1-c2);
        Shift = DAG.getNode(ISD::SRL, SDLoc(N), VT, N0.getOperand(0),
                            DAG.getConstant(c1-c2, N1.getValueType()));
      }
      return DAG.getNode(ISD::AND, SDLoc(N0), VT, Shift,
                         DAG.getConstant(Mask, VT));
    }
  }
  // fold (shl (sra x, c1), c1) -> (and x, (shl -1, c1))
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1)) {
    SDValue HiBitsMask =
      DAG.getConstant(APInt::getHighBitsSet(VT.getSizeInBits(),
                                            VT.getSizeInBits() -
                                              N1C->getZExtValue()),
                      VT);
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0.getOperand(0),
                       HiBitsMask);
  }

  if (N1C) {
    SDValue NewSHL = visitShiftByConstant(N, N1C->getZExtValue());
    if (NewSHL.getNode())
      return NewSHL;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSRA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

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
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                         N0.getOperand(0), DAG.getValueType(ExtVT));
  }

  // fold (sra (sra x, c1), c2) -> (sra x, (add c1, c2))
  if (N1C && N0.getOpcode() == ISD::SRA) {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      unsigned Sum = N1C->getZExtValue() + C1->getZExtValue();
      if (Sum >= OpSizeInBits) Sum = OpSizeInBits-1;
      return DAG.getNode(ISD::SRA, SDLoc(N), VT, N0.getOperand(0),
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
        EVT::getIntegerVT(*DAG.getContext(),
                          OpSizeInBits - N1C->getZExtValue());
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

          SDValue Amt = DAG.getConstant(ShiftAmt,
              getShiftAmountTy(N0.getOperand(0).getValueType()));
          SDValue Shift = DAG.getNode(ISD::SRL, SDLoc(N0), VT,
                                      N0.getOperand(0), Amt);
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0), TruncVT,
                                      Shift);
          return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N),
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
      TruncC = TruncC.trunc(TruncVT.getScalarType().getSizeInBits());
      return DAG.getNode(ISD::SRA, SDLoc(N), VT, N0,
                         DAG.getNode(ISD::AND, SDLoc(N),
                                     TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 SDLoc(N),
                                                 TruncVT, N100),
                                     DAG.getConstant(TruncC, TruncVT)));
    }
  }

  // fold (sra (trunc (sr x, c1)), c2) -> (trunc (sra x, c1+c2))
  //      if c1 is equal to the number of bits the trunc removes
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (N0.getOperand(0).getOpcode() == ISD::SRL ||
       N0.getOperand(0).getOpcode() == ISD::SRA) &&
      N0.getOperand(0).hasOneUse() &&
      N0.getOperand(0).getOperand(1).hasOneUse() &&
      N1C && isa<ConstantSDNode>(N0.getOperand(0).getOperand(1))) {
    EVT LargeVT = N0.getOperand(0).getValueType();
    ConstantSDNode *LargeShiftAmt =
      cast<ConstantSDNode>(N0.getOperand(0).getOperand(1));

    if (LargeVT.getScalarType().getSizeInBits() - OpSizeInBits ==
        LargeShiftAmt->getZExtValue()) {
      SDValue Amt =
        DAG.getConstant(LargeShiftAmt->getZExtValue() + N1C->getZExtValue(),
              getShiftAmountTy(N0.getOperand(0).getOperand(0).getValueType()));
      SDValue SRA = DAG.getNode(ISD::SRA, SDLoc(N), LargeVT,
                                N0.getOperand(0).getOperand(0), Amt);
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, SRA);
    }
  }

  // Simplify, based on bits shifted out of the LHS.
  if (N1C && SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);


  // If the sign bit is known to be zero, switch this to a SRL.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, N1);

  if (N1C) {
    SDValue NewSRA = visitShiftByConstant(N, N1C->getZExtValue());
    if (NewSRA.getNode())
      return NewSRA;
  }

  return SDValue();
}

SDValue DAGCombiner::visitSRL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarType().getSizeInBits();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

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
    if (c1 + c2 >= OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }

  // fold (srl (trunc (srl x, c1)), c2) -> 0 or (trunc (srl x, (add c1, c2)))
  if (N1C && N0.getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(0).getOpcode() == ISD::SRL &&
      isa<ConstantSDNode>(N0.getOperand(0)->getOperand(1))) {
    uint64_t c1 =
      cast<ConstantSDNode>(N0.getOperand(0)->getOperand(1))->getZExtValue();
    uint64_t c2 = N1C->getZExtValue();
    EVT InnerShiftVT = N0.getOperand(0).getValueType();
    EVT ShiftCountVT = N0.getOperand(0)->getOperand(1).getValueType();
    uint64_t InnerShiftSize = InnerShiftVT.getScalarType().getSizeInBits();
    // This is only valid if the OpSizeInBits + c1 = size of inner shift.
    if (c1 + OpSizeInBits == InnerShiftSize) {
      if (c1 + c2 >= InnerShiftSize)
        return DAG.getConstant(0, VT);
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N0), VT,
                         DAG.getNode(ISD::SRL, SDLoc(N0), InnerShiftVT,
                                     N0.getOperand(0)->getOperand(0),
                                     DAG.getConstant(c1 + c2, ShiftCountVT)));
    }
  }

  // fold (srl (shl x, c), c) -> (and x, cst2)
  if (N1C && N0.getOpcode() == ISD::SHL && N0.getOperand(1) == N1 &&
      N0.getValueSizeInBits() <= 64) {
    uint64_t ShAmt = N1C->getZExtValue()+64-N0.getValueSizeInBits();
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getConstant(~0ULL >> ShAmt, VT));
  }

  // fold (srl (anyextend x), c) -> (and (anyextend (srl x, c)), mask)
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    EVT SmallVT = N0.getOperand(0).getValueType();
    if (N1C->getZExtValue() >= SmallVT.getSizeInBits())
      return DAG.getUNDEF(VT);

    if (!LegalTypes || TLI.isTypeDesirableForOp(ISD::SRL, SmallVT)) {
      uint64_t ShiftAmt = N1C->getZExtValue();
      SDValue SmallShift = DAG.getNode(ISD::SRL, SDLoc(N0), SmallVT,
                                       N0.getOperand(0),
                          DAG.getConstant(ShiftAmt, getShiftAmountTy(SmallVT)));
      AddToWorkList(SmallShift.getNode());
      APInt Mask = APInt::getAllOnesValue(VT.getSizeInBits()).lshr(ShiftAmt);
      return DAG.getNode(ISD::AND, SDLoc(N), VT,
                         DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, SmallShift),
                         DAG.getConstant(Mask, VT));
    }
  }

  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getZExtValue() + 1 == VT.getSizeInBits()) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0.getOperand(0), N1);
  }

  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ &&
      N1C->getAPIntValue() == Log2_32(VT.getSizeInBits())) {
    APInt KnownZero, KnownOne;
    DAG.ComputeMaskedBits(N0.getOperand(0), KnownZero, KnownOne);

    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (KnownOne.getBoolValue()) return DAG.getConstant(0, VT);

    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    APInt UnknownBits = ~KnownZero;
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
        Op = DAG.getNode(ISD::SRL, SDLoc(N0), VT, Op,
                  DAG.getConstant(ShAmt, getShiftAmountTy(Op.getValueType())));
        AddToWorkList(Op.getNode());
      }

      return DAG.getNode(ISD::XOR, SDLoc(N), VT,
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
      TruncC = TruncC.trunc(TruncVT.getSizeInBits());
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0,
                         DAG.getNode(ISD::AND, SDLoc(N),
                                     TruncVT,
                                     DAG.getNode(ISD::TRUNCATE,
                                                 SDLoc(N),
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

  // Attempt to convert a srl of a load into a narrower zero-extending load.
  SDValue NarrowLoad = ReduceLoadWidth(N);
  if (NarrowLoad.getNode())
    return NarrowLoad;

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
    return DAG.getNode(ISD::CTLZ, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTLZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz_zero_undef c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTTZ, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz_zero_undef c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTTZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTPOP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctpop c1) -> c2
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::CTPOP, SDLoc(N), VT, N0);
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
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N2);
  // fold (select C, 0, 1) -> (xor C, 1)
  if (VT.isInteger() &&
      (VT0 == MVT::i1 ||
       (VT0.isInteger() &&
        TLI.getBooleanContents(false) ==
        TargetLowering::ZeroOrOneBooleanContent)) &&
      N1C && N2C && N1C->isNullValue() && N2C->getAPIntValue() == 1) {
    SDValue XORNode;
    if (VT == VT0)
      return DAG.getNode(ISD::XOR, SDLoc(N), VT0,
                         N0, DAG.getConstant(1, VT0));
    XORNode = DAG.getNode(ISD::XOR, SDLoc(N0), VT0,
                          N0, DAG.getConstant(1, VT0));
    AddToWorkList(XORNode.getNode());
    if (VT.bitsGT(VT0))
      return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, XORNode);
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, XORNode);
  }
  // fold (select C, 0, X) -> (and (not C), X)
  if (VT == VT0 && VT == MVT::i1 && N1C && N1C->isNullValue()) {
    SDValue NOTNode = DAG.getNOT(SDLoc(N0), N0, VT);
    AddToWorkList(NOTNode.getNode());
    return DAG.getNode(ISD::AND, SDLoc(N), VT, NOTNode, N2);
  }
  // fold (select C, X, 1) -> (or (not C), X)
  if (VT == VT0 && VT == MVT::i1 && N2C && N2C->getAPIntValue() == 1) {
    SDValue NOTNode = DAG.getNOT(SDLoc(N0), N0, VT);
    AddToWorkList(NOTNode.getNode());
    return DAG.getNode(ISD::OR, SDLoc(N), VT, NOTNode, N1);
  }
  // fold (select C, X, 0) -> (and C, X)
  if (VT == MVT::i1 && N2C && N2C->isNullValue())
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0, N1);
  // fold (select X, X, Y) -> (or X, Y)
  // fold (select X, 1, Y) -> (or X, Y)
  if (VT == MVT::i1 && (N0 == N1 || (N1C && N1C->getAPIntValue() == 1)))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N0, N2);
  // fold (select X, Y, X) -> (and X, Y)
  // fold (select X, Y, 0) -> (and X, Y)
  if (VT == MVT::i1 && (N0 == N2 || (N2C && N2C->getAPIntValue() == 0)))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N0, N1);

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
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), VT,
                         N0.getOperand(0), N0.getOperand(1),
                         N1, N2, N0.getOperand(2));
    return SimplifySelect(SDLoc(N), N0, N1, N2);
  }

  return SDValue();
}

static
std::pair<SDValue, SDValue> SplitVSETCC(const SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  EVT LoVT, HiVT;
  llvm::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(N->getValueType(0));

  // Split the inputs.
  SDValue Lo, Hi, LL, LH, RL, RH;
  llvm::tie(LL, LH) = DAG.SplitVectorOperand(N, 0);
  llvm::tie(RL, RH) = DAG.SplitVectorOperand(N, 1);

  Lo = DAG.getNode(N->getOpcode(), DL, LoVT, LL, RL, N->getOperand(2));
  Hi = DAG.getNode(N->getOpcode(), DL, HiVT, LH, RH, N->getOperand(2));

  return std::make_pair(Lo, Hi);
}

SDValue DAGCombiner::visitVSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDLoc DL(N);

  // Canonicalize integer abs.
  // vselect (setg[te] X,  0),  X, -X ->
  // vselect (setgt    X, -1),  X, -X ->
  // vselect (setl[te] X,  0), -X,  X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
    bool isAbs = false;
    bool RHSIsAllZeros = ISD::isBuildVectorAllZeros(RHS.getNode());

    if (((RHSIsAllZeros && (CC == ISD::SETGT || CC == ISD::SETGE)) ||
         (ISD::isBuildVectorAllOnes(RHS.getNode()) && CC == ISD::SETGT)) &&
        N1 == LHS && N2.getOpcode() == ISD::SUB && N1 == N2.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N2.getOperand(0).getNode());
    else if ((RHSIsAllZeros && (CC == ISD::SETLT || CC == ISD::SETLE)) &&
             N2 == LHS && N1.getOpcode() == ISD::SUB && N2 == N1.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N1.getOperand(0).getNode());

    if (isAbs) {
      EVT VT = LHS.getValueType();
      SDValue Shift = DAG.getNode(
          ISD::SRA, DL, VT, LHS,
          DAG.getConstant(VT.getScalarType().getSizeInBits() - 1, VT));
      SDValue Add = DAG.getNode(ISD::ADD, DL, VT, LHS, Shift);
      AddToWorkList(Shift.getNode());
      AddToWorkList(Add.getNode());
      return DAG.getNode(ISD::XOR, DL, VT, Add, Shift);
    }
  }

  // If the VSELECT result requires splitting and the mask is provided by a
  // SETCC, then split both nodes and its operands before legalization. This
  // prevents the type legalizer from unrolling SETCC into scalar comparisons
  // and enables future optimizations (e.g. min/max pattern matching on X86).
  if (N0.getOpcode() == ISD::SETCC) {
    EVT VT = N->getValueType(0);

    // Check if any splitting is required.
    if (TLI.getTypeAction(*DAG.getContext(), VT) !=
        TargetLowering::TypeSplitVector)
      return SDValue();

    SDValue Lo, Hi, CCLo, CCHi, LL, LH, RL, RH;
    llvm::tie(CCLo, CCHi) = SplitVSETCC(N0.getNode(), DAG);
    llvm::tie(LL, LH) = DAG.SplitVectorOperand(N, 1);
    llvm::tie(RL, RH) = DAG.SplitVectorOperand(N, 2);

    Lo = DAG.getNode(N->getOpcode(), DL, LL.getValueType(), CCLo, LL, RL);
    Hi = DAG.getNode(N->getOpcode(), DL, LH.getValueType(), CCHi, LH, RH);

    // Add the new VSELECT nodes to the work list in case they need to be split
    // again.
    AddToWorkList(Lo.getNode());
    AddToWorkList(Hi.getNode());

    return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Hi);
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
  SDValue SCC = SimplifySetCC(getSetCCResultType(N0.getValueType()),
                              N0, N1, CC, SDLoc(N), false);
  if (SCC.getNode()) {
    AddToWorkList(SCC.getNode());

    if (ConstantSDNode *SCCC = dyn_cast<ConstantSDNode>(SCC.getNode())) {
      if (!SCCC->isNullValue())
        return N2;    // cond always true -> true val
      else
        return N3;    // cond always false -> false val
    }

    // Fold to a simpler select_cc
    if (SCC.getOpcode() == ISD::SETCC)
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), N2.getValueType(),
                         SCC.getOperand(0), SCC.getOperand(1), N2, N3,
                         SCC.getOperand(2));
  }

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDValue(N, 0);  // Don't revisit N.

  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(SDLoc(N), N0, N1, N2, N3, CC);
}

SDValue DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get(),
                       SDLoc(N));
}

// ExtendUsesToFormExtLoad - Trying to extend uses of a load to enable this:
// "fold ({s|z|a}ext (load x)) -> ({s|z|a}ext (truncate ({s|z|a}extload x)))"
// transformation. Returns true if extension are possible and the above
// mentioned transformation is profitable.
static bool ExtendUsesToFormExtLoad(SDNode *N, SDValue N0,
                                    unsigned ExtOpc,
                                    SmallVectorImpl<SDNode *> &ExtendNodes,
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
      // a good reason for the transformation.
      return ExtendNodes.size();
  }
  return true;
}

void DAGCombiner::ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                                  SDValue Trunc, SDValue ExtLoad, SDLoc DL,
                                  ISD::NodeType ExtType) {
  // Extend SetCC uses if necessary.
  for (unsigned i = 0, e = SetCCs.size(); i != e; ++i) {
    SDNode *SetCC = SetCCs[i];
    SmallVector<SDValue, 4> Ops;

    for (unsigned j = 0; j != 2; ++j) {
      SDValue SOp = SetCC->getOperand(j);
      if (SOp == Trunc)
        Ops.push_back(ExtLoad);
      else
        Ops.push_back(DAG.getNode(ExtType, DL, ExtLoad->getValueType(0), SOp));
    }

    Ops.push_back(SetCC->getOperand(2));
    CombineTo(SetCC, DAG.getNode(ISD::SETCC, DL, SetCC->getValueType(0),
                                 &Ops[0], Ops.size()));
  }
}

SDValue DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, N0);

  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT,
                       N0.getOperand(0));

  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (sext (truncate (load x))) -> (sext (smaller load x))
    // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorkList(oye);
      }
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
        return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, Op);
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);
    }

    // fold (sext (truncate x)) -> (sextinreg x).
    if (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                                 N0.getValueType())) {
      if (OpBits < DestBits)
        Op = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N0), VT, Op);
      else if (OpBits > DestBits)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(N0), VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, Op,
                         DAG.getValueType(N0.getValueType()));
    }
  }

  // fold (sext (load x)) -> (sext (truncate (sextload x)))
  // None of the supported targets knows how to perform load and sign extend
  // on vectors in one instruction.  We only perform this transformation on
  // scalars.
  if (ISD::isNON_EXTLoad(N0.getNode()) && !VT.isVector() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::SIGN_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));
      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::SIGN_EXTEND);
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), MemVT,
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                            N0.getValueType(), ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (sext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (sextload x), (sext cst))
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      TLI.isLoadExtLegal(ISD::SEXTLOAD, N0.getValueType()) &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0.getOperand(0));
    if (LN0->getExtensionType() != ISD::ZEXTLOAD) {
      bool DoXform = true;
      SmallVector<SDNode*, 4> SetCCs;
      if (!N0.hasOneUse())
        DoXform = ExtendUsesToFormExtLoad(N, N0.getOperand(0), ISD::SIGN_EXTEND,
                                          SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(LN0), VT,
                                         LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        Mask = Mask.sext(VT.getSizeInBits());
        SDValue And = DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                                  ExtLoad, DAG.getConstant(Mask, VT));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE,
                                    SDLoc(N0.getOperand(0)),
                                    N0.getOperand(0).getValueType(), ExtLoad);
        CombineTo(N, And);
        CombineTo(N0.getOperand(0).getNode(), Trunc, ExtLoad.getValue(1));
        ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                        ISD::SIGN_EXTEND);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    // sext(setcc) -> sext_in_reg(vsetcc) for vectors.
    // Only do this before legalize for now.
    if (VT.isVector() && !LegalOperations &&
        TLI.getBooleanContents(true) ==
          TargetLowering::ZeroOrNegativeOneBooleanContent) {
      EVT N0VT = N0.getOperand(0).getValueType();
      // On some architectures (such as SSE/NEON/etc) the SETCC result type is
      // of the same size as the compared operands. Only optimize sext(setcc())
      // if this is the case.
      EVT SVT = getSetCCResultType(N0VT);

      // We know that the # elements of the results is the same as the
      // # elements of the compare (and the # elements of the compare result
      // for that matter).  Check to see that they are the same size.  If so,
      // we know that the element size of the sext'd result matches the
      // element size of the compare operands.
      if (VT.getSizeInBits() == SVT.getSizeInBits())
        return DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1),
                             cast<CondCodeSDNode>(N0.getOperand(2))->get());

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/sign extend
      EVT MatchingVectorType = N0VT.changeVectorElementTypeToInteger();
      if (SVT == MatchingVectorType) {
        SDValue VsetCC = DAG.getSetCC(SDLoc(N), MatchingVectorType,
                               N0.getOperand(0), N0.getOperand(1),
                               cast<CondCodeSDNode>(N0.getOperand(2))->get());
        return DAG.getSExtOrTrunc(VsetCC, SDLoc(N), VT);
      }
    }

    // sext(setcc x, y, cc) -> (select_cc x, y, -1, 0, cc)
    unsigned ElementWidth = VT.getScalarType().getSizeInBits();
    SDValue NegOne =
      DAG.getConstant(APInt::getAllOnesValue(ElementWidth), VT);
    SDValue SCC =
      SimplifySelectCC(SDLoc(N), N0.getOperand(0), N0.getOperand(1),
                       NegOne, DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
    if (!VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::SETCC, getSetCCResultType(VT)))) {
      return DAG.getSelect(SDLoc(N), VT,
                           DAG.getSetCC(SDLoc(N),
                           getSetCCResultType(VT),
                           N0.getOperand(0), N0.getOperand(1),
                           cast<CondCodeSDNode>(N0.getOperand(2))->get()),
                           NegOne, DAG.getConstant(0, VT));
    }
  }

  // fold (sext x) -> (zext x) if the sign bit is known zero.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ZERO_EXTEND, VT)) &&
      DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, N0);

  return SDValue();
}

// isTruncateOf - If N is a truncate of some other value, return true, record
// the value being truncated in Op and which of Op's bits are zero in KnownZero.
// This function computes KnownZero to avoid a duplicated call to
// ComputeMaskedBits in the caller.
static bool isTruncateOf(SelectionDAG &DAG, SDValue N, SDValue &Op,
                         APInt &KnownZero) {
  APInt KnownOne;
  if (N->getOpcode() == ISD::TRUNCATE) {
    Op = N->getOperand(0);
    DAG.ComputeMaskedBits(Op, KnownZero, KnownOne);
    return true;
  }

  if (N->getOpcode() != ISD::SETCC || N->getValueType(0) != MVT::i1 ||
      cast<CondCodeSDNode>(N->getOperand(2))->get() != ISD::SETNE)
    return false;

  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  assert(Op0.getValueType() == Op1.getValueType());

  ConstantSDNode *COp0 = dyn_cast<ConstantSDNode>(Op0);
  ConstantSDNode *COp1 = dyn_cast<ConstantSDNode>(Op1);
  if (COp0 && COp0->isNullValue())
    Op = Op1;
  else if (COp1 && COp1->isNullValue())
    Op = Op0;
  else
    return false;

  DAG.ComputeMaskedBits(Op, KnownZero, KnownOne);

  if (!(KnownZero | APInt(Op.getValueSizeInBits(), 1)).isAllOnesValue())
    return false;

  return true;
}

SDValue DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (zext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, N0);
  // fold (zext (zext x)) -> (zext x)
  // fold (zext (aext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT,
                       N0.getOperand(0));

  // fold (zext (truncate x)) -> (zext x) or
  //      (zext (truncate x)) -> (truncate x)
  // This is valid when the truncated bits of x are already zero.
  // FIXME: We should extend this to work for vectors too.
  SDValue Op;
  APInt KnownZero;
  if (!VT.isVector() && isTruncateOf(DAG, N0, Op, KnownZero)) {
    APInt TruncatedBits =
      (Op.getValueSizeInBits() == N0.getValueSizeInBits()) ?
      APInt(Op.getValueSizeInBits(), 0) :
      APInt::getBitsSet(Op.getValueSizeInBits(),
                        N0.getValueSizeInBits(),
                        std::min(Op.getValueSizeInBits(),
                                 VT.getSizeInBits()));
    if (TruncatedBits == (KnownZero & TruncatedBits)) {
      if (VT.bitsGT(Op.getValueType()))
        return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, Op);
      if (VT.bitsLT(Op.getValueType()))
        return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);

      return Op;
    }
  }

  // fold (zext (truncate (load x))) -> (zext (smaller load x))
  // fold (zext (truncate (srl (load x), c))) -> (zext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorkList(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (!LegalOperations || TLI.isOperationLegal(ISD::AND, VT))) {

    // fold (zext (truncate (load x))) -> (zext (smaller load x))
    // fold (zext (truncate (srl (load x), c))) -> (zext (smaller load (x+c/n)))
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorkList(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }

    SDValue Op = N0.getOperand(0);
    if (Op.getValueType().bitsLT(VT)) {
      Op = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, Op);
      AddToWorkList(Op.getNode());
    } else if (Op.getValueType().bitsGT(VT)) {
      Op = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Op);
      AddToWorkList(Op.getNode());
    }
    return DAG.getZeroExtendInReg(Op, SDLoc(N),
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
      X = DAG.getNode(ISD::ANY_EXTEND, SDLoc(X), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, SDLoc(X), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask = Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, SDLoc(N), VT,
                       X, DAG.getConstant(Mask, VT));
  }

  // fold (zext (load x)) -> (zext (truncate (zextload x)))
  // None of the supported targets knows how to perform load and vector_zext
  // on vectors in one instruction.  We only perform this transformation on
  // scalars.
  if (ISD::isNON_EXTLoad(N0.getNode()) && !VT.isVector() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::ZEXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ZERO_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));

      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::ZERO_EXTEND);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (zext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (zextload x), (zext cst))
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      TLI.isLoadExtLegal(ISD::ZEXTLOAD, N0.getValueType()) &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0.getOperand(0));
    if (LN0->getExtensionType() != ISD::SEXTLOAD) {
      bool DoXform = true;
      SmallVector<SDNode*, 4> SetCCs;
      if (!N0.hasOneUse())
        DoXform = ExtendUsesToFormExtLoad(N, N0.getOperand(0), ISD::ZERO_EXTEND,
                                          SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN0), VT,
                                         LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        Mask = Mask.zext(VT.getSizeInBits());
        SDValue And = DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                                  ExtLoad, DAG.getConstant(Mask, VT));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE,
                                    SDLoc(N0.getOperand(0)),
                                    N0.getOperand(0).getValueType(), ExtLoad);
        CombineTo(N, And);
        CombineTo(N0.getOperand(0).getNode(), Trunc, ExtLoad.getValue(1));
        ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                        ISD::ZERO_EXTEND);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
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
      SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), MemVT,
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(),
                            ExtLoad),
                ExtLoad.getValue(1));
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    if (!LegalOperations && VT.isVector()) {
      // zext(setcc) -> (and (vsetcc), (1, 1, ...) for vectors.
      // Only do this before legalize for now.
      EVT N0VT = N0.getOperand(0).getValueType();
      EVT EltVT = VT.getVectorElementType();
      SmallVector<SDValue,8> OneOps(VT.getVectorNumElements(),
                                    DAG.getConstant(1, EltVT));
      if (VT.getSizeInBits() == N0VT.getSizeInBits())
        // We know that the # elements of the results is the same as the
        // # elements of the compare (and the # elements of the compare result
        // for that matter).  Check to see that they are the same size.  If so,
        // we know that the element size of the sext'd result matches the
        // element size of the compare operands.
        return DAG.getNode(ISD::AND, SDLoc(N), VT,
                           DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                                         N0.getOperand(1),
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get()),
                           DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT,
                                       &OneOps[0], OneOps.size()));

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/sign extend
      EVT MatchingElementType =
        EVT::getIntegerVT(*DAG.getContext(),
                          N0VT.getScalarType().getSizeInBits());
      EVT MatchingVectorType =
        EVT::getVectorVT(*DAG.getContext(), MatchingElementType,
                         N0VT.getVectorNumElements());
      SDValue VsetCC =
        DAG.getSetCC(SDLoc(N), MatchingVectorType, N0.getOperand(0),
                      N0.getOperand(1),
                      cast<CondCodeSDNode>(N0.getOperand(2))->get());
      return DAG.getNode(ISD::AND, SDLoc(N), VT,
                         DAG.getSExtOrTrunc(VsetCC, SDLoc(N), VT),
                         DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT,
                                     &OneOps[0], OneOps.size()));
    }

    // zext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
    SDValue SCC =
      SimplifySelectCC(SDLoc(N), N0.getOperand(0), N0.getOperand(1),
                       DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       cast<CondCodeSDNode>(N0.getOperand(2))->get(), true);
    if (SCC.getNode()) return SCC;
  }

  // (zext (shl (zext x), cst)) -> (shl (zext x), cst)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL) &&
      isa<ConstantSDNode>(N0.getOperand(1)) &&
      N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
      N0.hasOneUse()) {
    SDValue ShAmt = N0.getOperand(1);
    unsigned ShAmtVal = cast<ConstantSDNode>(ShAmt)->getZExtValue();
    if (N0.getOpcode() == ISD::SHL) {
      SDValue InnerZExt = N0.getOperand(0);
      // If the original shl may be shifting out bits, do not perform this
      // transformation.
      unsigned KnownZeroBits = InnerZExt.getValueType().getSizeInBits() -
        InnerZExt.getOperand(0).getValueType().getSizeInBits();
      if (ShAmtVal > KnownZeroBits)
        return SDValue();
    }

    SDLoc DL(N);

    // Ensure that the shift amount is wide enough for the shifted value.
    if (VT.getSizeInBits() >= 256)
      ShAmt = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, ShAmt);

    return DAG.getNode(N0.getOpcode(), DL, VT,
                       DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0)),
                       ShAmt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (aext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, N0);
  // fold (aext (aext x)) -> (aext x)
  // fold (aext (zext x)) -> (zext x)
  // fold (aext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::ANY_EXTEND  ||
      N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, N0.getOperand(0));

  // fold (aext (truncate (load x))) -> (aext (smaller load x))
  // fold (aext (truncate (srl (load x), c))) -> (aext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue NarrowLoad = ReduceLoadWidth(N0.getNode());
    if (NarrowLoad.getNode()) {
      SDNode* oye = N0.getNode()->getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorkList(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    SDValue TruncOp = N0.getOperand(0);
    if (TruncOp.getValueType() == VT)
      return TruncOp; // x iff x size == zext size.
    if (TruncOp.getValueType().bitsGT(VT))
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, TruncOp);
    return DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, TruncOp);
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
      X = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N), VT, X);
    } else if (X.getValueType().bitsGT(VT)) {
      X = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, X);
    }
    APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
    Mask = Mask.zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, SDLoc(N), VT,
                       X, DAG.getConstant(Mask, VT));
  }

  // fold (aext (load x)) -> (aext (truncate (extload x)))
  // None of the supported targets knows how to perform load and any_ext
  // on vectors in one instruction.  We only perform this transformation on
  // scalars.
  if (ISD::isNON_EXTLoad(N0.getNode()) && !VT.isVector() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    bool DoXform = true;
    SmallVector<SDNode*, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform = ExtendUsesToFormExtLoad(N, N0, ISD::ANY_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(),
                                       LN0->getBasePtr(), N0.getValueType(),
                                       LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                                  N0.getValueType(), ExtLoad);
      CombineTo(N0.getNode(), Trunc, ExtLoad.getValue(1));
      ExtendSetCCUses(SetCCs, Trunc, ExtLoad, SDLoc(N),
                      ISD::ANY_EXTEND);
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
    SDValue ExtLoad = DAG.getExtLoad(LN0->getExtensionType(), SDLoc(N),
                                     VT, LN0->getChain(), LN0->getBasePtr(),
                                     MemVT, LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::TRUNCATE, SDLoc(N0),
                          N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  if (N0.getOpcode() == ISD::SETCC) {
    // aext(setcc) -> sext_in_reg(vsetcc) for vectors.
    // Only do this before legalize for now.
    if (VT.isVector() && !LegalOperations) {
      EVT N0VT = N0.getOperand(0).getValueType();
        // We know that the # elements of the results is the same as the
        // # elements of the compare (and the # elements of the compare result
        // for that matter).  Check to see that they are the same size.  If so,
        // we know that the element size of the sext'd result matches the
        // element size of the compare operands.
      if (VT.getSizeInBits() == N0VT.getSizeInBits())
        return DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1),
                             cast<CondCodeSDNode>(N0.getOperand(2))->get());
      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/sign extend
      else {
        EVT MatchingElementType =
          EVT::getIntegerVT(*DAG.getContext(),
                            N0VT.getScalarType().getSizeInBits());
        EVT MatchingVectorType =
          EVT::getVectorVT(*DAG.getContext(), MatchingElementType,
                           N0VT.getVectorNumElements());
        SDValue VsetCC =
          DAG.getSetCC(SDLoc(N), MatchingVectorType, N0.getOperand(0),
                        N0.getOperand(1),
                        cast<CondCodeSDNode>(N0.getOperand(2))->get());
        return DAG.getSExtOrTrunc(VsetCC, SDLoc(N), VT);
      }
    }

    // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
    SDValue SCC =
      SimplifySelectCC(SDLoc(N), N0.getOperand(0), N0.getOperand(1),
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
  case ISD::Constant: {
    const ConstantSDNode *CV = cast<ConstantSDNode>(V.getNode());
    assert(CV != 0 && "Const value should be ConstSDNode.");
    const APInt &CVal = CV->getAPIntValue();
    APInt NewVal = CVal & Mask;
    if (NewVal != CVal)
      return DAG.getConstant(NewVal, V.getValueType());
    break;
  }
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
        return DAG.getNode(ISD::SRL, SDLoc(V), V.getValueType(),
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
  } else if (Opc == ISD::SRL) {
    // Another special-case: SRL is basically zero-extending a narrower value.
    ExtType = ISD::ZEXTLOAD;
    N0 = SDValue(N, 0);
    ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01) return SDValue();
    ExtVT = EVT::getIntegerVT(*DAG.getContext(),
                              VT.getSizeInBits() - N01->getZExtValue());
  }
  if (LegalOperations && !TLI.isLoadExtLegal(ExtType, ExtVT))
    return SDValue();

  unsigned EVTBits = ExtVT.getSizeInBits();

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (!ExtVT.isRound())
    return SDValue();

  unsigned ShAmt = 0;
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShAmt = N01->getZExtValue();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        // Is the load width a multiple of size of VT?
        if ((N0.getValueType().getSizeInBits() & (EVTBits-1)) != 0)
          return SDValue();
      }

      // At this point, we must have a load or else we can't do the transform.
      if (!isa<LoadSDNode>(N0)) return SDValue();

      // Because a SRL must be assumed to *need* to zero-extend the high bits
      // (as opposed to anyext the high bits), we can't combine the zextload
      // lowering of SRL and an sextload.
      if (cast<LoadSDNode>(N0)->getExtensionType() == ISD::SEXTLOAD)
        return SDValue();

      // If the shift amount is larger than the input type then we're not
      // accessing any of the loaded bytes.  If the load was a zextload/extload
      // then the result of the shift+trunc is zero/undef (handled elsewhere).
      if (ShAmt >= cast<LoadSDNode>(N0)->getMemoryVT().getSizeInBits())
        return SDValue();
    }
  }

  // If the load is shifted left (and the result isn't shifted back right),
  // we can fold the truncate through the shift.
  unsigned ShLeftAmt = 0;
  if (ShAmt == 0 && N0.getOpcode() == ISD::SHL && N0.hasOneUse() &&
      ExtVT == VT && TLI.isNarrowingProfitable(N0.getValueType(), VT)) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShLeftAmt = N01->getZExtValue();
      N0 = N0.getOperand(0);
    }
  }

  // If we haven't found a load, we can't narrow it.  Don't transform one with
  // multiple uses, this would require adding a new load.
  if (!isa<LoadSDNode>(N0) || !N0.hasOneUse())
    return SDValue();

  // Don't change the width of a volatile load.
  LoadSDNode *LN0 = cast<LoadSDNode>(N0);
  if (LN0->isVolatile())
    return SDValue();

  // Verify that we are actually reducing a load width here.
  if (LN0->getMemoryVT().getSizeInBits() < EVTBits)
    return SDValue();

  // For the transform to be legal, the load must produce only two values
  // (the value loaded and the chain).  Don't transform a pre-increment
  // load, for example, which produces an extra value.  Otherwise the
  // transformation is not equivalent, and the downstream logic to replace
  // uses gets things wrong.
  if (LN0->getNumValues() > 2)
    return SDValue();

  // If the load that we're shrinking is an extload and we're not just
  // discarding the extension we can't simply shrink the load. Bail.
  // TODO: It would be possible to merge the extensions in some cases.
  if (LN0->getExtensionType() != ISD::NON_EXTLOAD &&
      LN0->getMemoryVT().getSizeInBits() < ExtVT.getSizeInBits() + ShAmt)
    return SDValue();

  EVT PtrType = N0.getOperand(1).getValueType();

  if (PtrType == MVT::Untyped || PtrType.isExtended())
    // It's not possible to generate a constant of extended or untyped type.
    return SDValue();

  // For big endian targets, we need to adjust the offset to the pointer to
  // load the correct bytes.
  if (TLI.isBigEndian()) {
    unsigned LVTStoreBits = LN0->getMemoryVT().getStoreSizeInBits();
    unsigned EVTStoreBits = ExtVT.getStoreSizeInBits();
    ShAmt = LVTStoreBits - EVTStoreBits - ShAmt;
  }

  uint64_t PtrOff = ShAmt / 8;
  unsigned NewAlign = MinAlign(LN0->getAlignment(), PtrOff);
  SDValue NewPtr = DAG.getNode(ISD::ADD, SDLoc(LN0),
                               PtrType, LN0->getBasePtr(),
                               DAG.getConstant(PtrOff, PtrType));
  AddToWorkList(NewPtr.getNode());

  SDValue Load;
  if (ExtType == ISD::NON_EXTLOAD)
    Load =  DAG.getLoad(VT, SDLoc(N0), LN0->getChain(), NewPtr,
                        LN0->getPointerInfo().getWithOffset(PtrOff),
                        LN0->isVolatile(), LN0->isNonTemporal(),
                        LN0->isInvariant(), NewAlign, LN0->getTBAAInfo());
  else
    Load = DAG.getExtLoad(ExtType, SDLoc(N0), VT, LN0->getChain(),NewPtr,
                          LN0->getPointerInfo().getWithOffset(PtrOff),
                          ExtVT, LN0->isVolatile(), LN0->isNonTemporal(),
                          NewAlign, LN0->getTBAAInfo());

  // Replace the old load's chain with the new load's chain.
  WorkListRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1));

  // Shift the result left, if we've swallowed a left shift.
  SDValue Result = Load;
  if (ShLeftAmt != 0) {
    EVT ShImmTy = getShiftAmountTy(Result.getValueType());
    if (!isUIntN(ShImmTy.getSizeInBits(), ShLeftAmt))
      ShImmTy = VT;
    // If the shift amount is as large as the result size (but, presumably,
    // no larger than the source) then the useful bits of the result are
    // zero; we can't simply return the shortened shift, because the result
    // of that operation is undefined.
    if (ShLeftAmt >= VT.getSizeInBits())
      Result = DAG.getConstant(0, VT);
    else
      Result = DAG.getNode(ISD::SHL, SDLoc(N0), VT,
                          Result, DAG.getConstant(ShLeftAmt, ShImmTy));
  }

  // Return the new loaded value.
  return Result;
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
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, N0, N1);

  // If the input is already sign extended, just drop the extension.
  if (DAG.ComputeNumSignBits(N0) >= VTBits-EVTBits+1)
    return N0;

  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT.bitsLT(cast<VTSDNode>(N0.getOperand(1))->getVT()))
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                       N0.getOperand(0), N1);

  // fold (sext_in_reg (sext x)) -> (sext x)
  // fold (sext_in_reg (aext x)) -> (sext x)
  // if x is small enough.
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getValueType().getScalarType().getSizeInBits() <= EVTBits &&
        (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND, VT)))
      return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, N00, N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (DAG.MaskedValueIsZero(N0, APInt::getBitsSet(VTBits, EVTBits-1, EVTBits)))
    return DAG.getZeroExtendInReg(N0, SDLoc(N), EVT);

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
          return DAG.getNode(ISD::SRA, SDLoc(N), VT,
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
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), EVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    AddToWorkList(ExtLoad.getNode());
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }
  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (ISD::isZEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse() &&
      EVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, EVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), EVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  // Form (sext_inreg (bswap >> 16)) or (sext_inreg (rotl (bswap) 16))
  if (EVTBits <= 16 && N0.getOpcode() == ISD::OR) {
    SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                       N0.getOperand(1), false);
    if (BSwap.getNode() != 0)
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                         BSwap, N1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  bool isLE = TLI.isLittleEndian();

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND ||
      N0.getOpcode() == ISD::ANY_EXTEND) {
    if (N0.getOperand(0).getValueType().bitsLT(VT))
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), SDLoc(N), VT,
                         N0.getOperand(0));
    if (N0.getOperand(0).getValueType().bitsGT(VT))
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));
    // if the source and dest are the same type, we can drop both the extend
    // and the truncate.
    return N0.getOperand(0);
  }

  // Fold extract-and-trunc into a narrow extract. For example:
  //   i64 x = EXTRACT_VECTOR_ELT(v2i64 val, i32 1)
  //   i32 y = TRUNCATE(i64 x)
  //        -- becomes --
  //   v16i8 b = BITCAST (v2i64 val)
  //   i8 x = EXTRACT_VECTOR_ELT(v16i8 b, i32 8)
  //
  // Note: We only run this optimization after type legalization (which often
  // creates this pattern) and before operation legalization after which
  // we need to be more careful about the vector instructions that we generate.
  if (N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      LegalTypes && !LegalOperations && N0->hasOneUse()) {

    EVT VecTy = N0.getOperand(0).getValueType();
    EVT ExTy = N0.getValueType();
    EVT TrTy = N->getValueType(0);

    unsigned NumElem = VecTy.getVectorNumElements();
    unsigned SizeRatio = ExTy.getSizeInBits()/TrTy.getSizeInBits();

    EVT NVT = EVT::getVectorVT(*DAG.getContext(), TrTy, SizeRatio * NumElem);
    assert(NVT.getSizeInBits() == VecTy.getSizeInBits() && "Invalid Size");

    SDValue EltNo = N0->getOperand(1);
    if (isa<ConstantSDNode>(EltNo) && isTypeLegal(NVT)) {
      int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
      EVT IndexTy = TLI.getVectorIdxTy();
      int Index = isLE ? (Elt*SizeRatio) : (Elt*SizeRatio + (SizeRatio-1));

      SDValue V = DAG.getNode(ISD::BITCAST, SDLoc(N),
                              NVT, N0.getOperand(0));

      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
                         SDLoc(N), TrTy, V,
                         DAG.getConstant(Index, IndexTy));
    }
  }

  // Fold a series of buildvector, bitcast, and truncate if possible.
  // For example fold
  //   (2xi32 trunc (bitcast ((4xi32)buildvector x, x, y, y) 2xi64)) to
  //   (2xi32 (buildvector x, y)).
  if (Level == AfterLegalizeVectorOps && VT.isVector() &&
      N0.getOpcode() == ISD::BITCAST && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::BUILD_VECTOR &&
      N0.getOperand(0).hasOneUse()) {

    SDValue BuildVect = N0.getOperand(0);
    EVT BuildVectEltTy = BuildVect.getValueType().getVectorElementType();
    EVT TruncVecEltTy = VT.getVectorElementType();

    // Check that the element types match.
    if (BuildVectEltTy == TruncVecEltTy) {
      // Now we only need to compute the offset of the truncated elements.
      unsigned BuildVecNumElts =  BuildVect.getNumOperands();
      unsigned TruncVecNumElts = VT.getVectorNumElements();
      unsigned TruncEltOffset = BuildVecNumElts / TruncVecNumElts;

      assert((BuildVecNumElts % TruncVecNumElts) == 0 &&
             "Invalid number of elements");

      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = BuildVecNumElts; i != e; i += TruncEltOffset)
        Opnds.push_back(BuildVect.getOperand(i));

      return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N), VT, &Opnds[0],
                         Opnds.size());
    }
  }

  // See if we can simplify the input to this truncate through knowledge that
  // only the low bits are being used.
  // For example "trunc (or (shl x, 8), y)" // -> trunc y
  // Currently we only perform this optimization on scalars because vectors
  // may have different active low bits.
  if (!VT.isVector()) {
    SDValue Shorter =
      GetDemandedBits(N0, APInt::getLowBitsSet(N0.getValueSizeInBits(),
                                               VT.getSizeInBits()));
    if (Shorter.getNode())
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Shorter);
  }
  // fold (truncate (load x)) -> (smaller load x)
  // fold (truncate (srl (load x), c)) -> (smaller load (x+c/evtbits))
  if (!LegalTypes || TLI.isTypeDesirableForOp(N0.getOpcode(), VT)) {
    SDValue Reduced = ReduceLoadWidth(N);
    if (Reduced.getNode())
      return Reduced;
  }
  // fold (trunc (concat ... x ...)) -> (concat ..., (trunc x), ...)),
  // where ... are all 'undef'.
  if (N0.getOpcode() == ISD::CONCAT_VECTORS && !LegalTypes) {
    SmallVector<EVT, 8> VTs;
    SDValue V;
    unsigned Idx = 0;
    unsigned NumDefs = 0;

    for (unsigned i = 0, e = N0.getNumOperands(); i != e; ++i) {
      SDValue X = N0.getOperand(i);
      if (X.getOpcode() != ISD::UNDEF) {
        V = X;
        Idx = i;
        NumDefs++;
      }
      // Stop if more than one members are non-undef.
      if (NumDefs > 1)
        break;
      VTs.push_back(EVT::getVectorVT(*DAG.getContext(),
                                     VT.getVectorElementType(),
                                     X.getValueType().getVectorNumElements()));
    }

    if (NumDefs == 0)
      return DAG.getUNDEF(VT);

    if (NumDefs == 1) {
      assert(V.getNode() && "The single defined operand is empty!");
      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
        if (i != Idx) {
          Opnds.push_back(DAG.getUNDEF(VTs[i]));
          continue;
        }
        SDValue NV = DAG.getNode(ISD::TRUNCATE, SDLoc(V), VTs[i], V);
        AddToWorkList(NV.getNode());
        Opnds.push_back(NV);
      }
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT,
                         &Opnds[0], Opnds.size());
    }
  }

  // Simplify the operands using demanded-bits information.
  if (!VT.isVector() &&
      SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
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
  if (!LD1 || !LD2 || !ISD::isNON_EXTLoad(LD1) || !LD1->hasOneUse() ||
      LD1->getPointerInfo().getAddrSpace() !=
         LD2->getPointerInfo().getAddrSpace())
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
    unsigned NewAlign = TLI.getDataLayout()->
      getABITypeAlignment(VT.getTypeForEVT(*DAG.getContext()));

    if (NewAlign <= Align &&
        (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)))
      return DAG.getLoad(VT, SDLoc(N), LD1->getChain(),
                         LD1->getBasePtr(), LD1->getPointerInfo(),
                         false, false, false, Align);
  }

  return SDValue();
}

SDValue DAGCombiner::visitBITCAST(SDNode *N) {
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
      return ConstantFoldBITCASTofBUILD_VECTOR(N0.getNode(), DestEltVT);
  }

  // If the input is a constant, let getNode fold it.
  if (isa<ConstantSDNode>(N0) || isa<ConstantFPSDNode>(N0)) {
    SDValue Res = DAG.getNode(ISD::BITCAST, SDLoc(N), VT, N0);
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
  if (N0.getOpcode() == ISD::BITCAST)
    return DAG.getNode(ISD::BITCAST, SDLoc(N), VT,
                       N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a higher alignment than the original!
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile() &&
      (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)) &&
      TLI.isLoadBitCastBeneficial(N0.getValueType(), VT)) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    unsigned Align = TLI.getDataLayout()->
      getABITypeAlignment(VT.getTypeForEVT(*DAG.getContext()));
    unsigned OrigAlign = LN0->getAlignment();

    if (Align <= OrigAlign) {
      SDValue Load = DAG.getLoad(VT, SDLoc(N), LN0->getChain(),
                                 LN0->getBasePtr(), LN0->getPointerInfo(),
                                 LN0->isVolatile(), LN0->isNonTemporal(),
                                 LN0->isInvariant(), OrigAlign,
                                 LN0->getTBAAInfo());
      AddToWorkList(N);
      CombineTo(N0.getNode(),
                DAG.getNode(ISD::BITCAST, SDLoc(N0),
                            N0.getValueType(), Load),
                Load.getValue(1));
      return Load;
    }
  }

  // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
  // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
  // This often reduces constant pool loads.
  if (((N0.getOpcode() == ISD::FNEG && !TLI.isFNegFree(N0.getValueType())) ||
       (N0.getOpcode() == ISD::FABS && !TLI.isFAbsFree(N0.getValueType()))) &&
      N0.getNode()->hasOneUse() && VT.isInteger() &&
      !VT.isVector() && !N0.getValueType().isVector()) {
    SDValue NewConv = DAG.getNode(ISD::BITCAST, SDLoc(N0), VT,
                                  N0.getOperand(0));
    AddToWorkList(NewConv.getNode());

    APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
    if (N0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, SDLoc(N), VT,
                         NewConv, DAG.getConstant(SignBit, VT));
    assert(N0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, SDLoc(N), VT,
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
    if (isTypeLegal(IntXVT)) {
      SDValue X = DAG.getNode(ISD::BITCAST, SDLoc(N0),
                              IntXVT, N0.getOperand(1));
      AddToWorkList(X.getNode());

      // If X has a different width than the result/lhs, sext it or truncate it.
      unsigned VTWidth = VT.getSizeInBits();
      if (OrigXWidth < VTWidth) {
        X = DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, X);
        AddToWorkList(X.getNode());
      } else if (OrigXWidth > VTWidth) {
        // To get the sign bit in the right place, we have to shift it right
        // before truncating.
        X = DAG.getNode(ISD::SRL, SDLoc(X),
                        X.getValueType(), X,
                        DAG.getConstant(OrigXWidth-VTWidth, X.getValueType()));
        AddToWorkList(X.getNode());
        X = DAG.getNode(ISD::TRUNCATE, SDLoc(X), VT, X);
        AddToWorkList(X.getNode());
      }

      APInt SignBit = APInt::getSignBit(VT.getSizeInBits());
      X = DAG.getNode(ISD::AND, SDLoc(X), VT,
                      X, DAG.getConstant(SignBit, VT));
      AddToWorkList(X.getNode());

      SDValue Cst = DAG.getNode(ISD::BITCAST, SDLoc(N0),
                                VT, N0.getOperand(0));
      Cst = DAG.getNode(ISD::AND, SDLoc(Cst), VT,
                        Cst, DAG.getConstant(~SignBit, VT));
      AddToWorkList(Cst.getNode());

      return DAG.getNode(ISD::OR, SDLoc(N), VT, X, Cst);
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

/// ConstantFoldBITCASTofBUILD_VECTOR - We know that BV is a build_vector
/// node with Constant, ConstantFP or Undef operands.  DstEltVT indicates the
/// destination element value type.
SDValue DAGCombiner::
ConstantFoldBITCASTofBUILD_VECTOR(SDNode *BV, EVT DstEltVT) {
  EVT SrcEltVT = BV->getValueType(0).getVectorElementType();

  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDValue(BV, 0);

  unsigned SrcBitSize = SrcEltVT.getSizeInBits();
  unsigned DstBitSize = DstEltVT.getSizeInBits();

  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                              BV->getValueType(0).getVectorNumElements());

    // Due to the FP element handling below calling this routine recursively,
    // we can end up with a scalar-to-vector node here.
    if (BV->getOpcode() == ISD::SCALAR_TO_VECTOR)
      return DAG.getNode(ISD::SCALAR_TO_VECTOR, SDLoc(BV), VT,
                         DAG.getNode(ISD::BITCAST, SDLoc(BV),
                                     DstEltVT, BV->getOperand(0)));

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
      SDValue Op = BV->getOperand(i);
      // If the vector element type is not legal, the BUILD_VECTOR operands
      // are promoted and implicitly truncated.  Make that explicit here.
      if (Op.getValueType() != SrcEltVT)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(BV), SrcEltVT, Op);
      Ops.push_back(DAG.getNode(ISD::BITCAST, SDLoc(BV),
                                DstEltVT, Op));
      AddToWorkList(Ops.back().getNode());
    }
    return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(BV), VT,
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
    BV = ConstantFoldBITCASTofBUILD_VECTOR(BV, IntVT).getNode();
    SrcEltVT = IntVT;
  }

  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (DstEltVT.isFloatingPoint()) {
    assert((DstEltVT == MVT::f32 || DstEltVT == MVT::f64) && "Unknown FP VT!");
    EVT TmpVT = EVT::getIntegerVT(*DAG.getContext(), DstEltVT.getSizeInBits());
    SDNode *Tmp = ConstantFoldBITCASTofBUILD_VECTOR(BV, TmpVT).getNode();

    // Next, convert to FP elements of the same size.
    return ConstantFoldBITCASTofBUILD_VECTOR(Tmp, DstEltVT);
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

        NewBits |= cast<ConstantSDNode>(Op)->getAPIntValue().
                   zextOrTrunc(SrcBitSize).zext(DstBitSize);
      }

      if (EltIsUndef)
        Ops.push_back(DAG.getUNDEF(DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DstEltVT));
    }

    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT, Ops.size());
    return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(BV), VT,
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

    APInt OpVal = cast<ConstantSDNode>(BV->getOperand(i))->
                  getAPIntValue().zextOrTrunc(SrcBitSize);

    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      APInt ThisVal = OpVal.trunc(DstBitSize);
      Ops.push_back(DAG.getConstant(ThisVal, DstEltVT));
      if (isS2V && i == 0 && j == 0 && ThisVal.zext(SrcBitSize) == OpVal)
        // Simply turn this into a SCALAR_TO_VECTOR of the new type.
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, SDLoc(BV), VT,
                           Ops[0]);
      OpVal = OpVal.lshr(DstBitSize);
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (TLI.isBigEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(BV), VT,
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

  // fold (fadd c1, c2) -> c1 + c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N1, N0);
  // fold (fadd A, 0) -> A
  if (DAG.getTarget().Options.UnsafeFPMath && N1CFP &&
      N1CFP->getValueAPF().isZero())
    return N0;
  // fold (fadd A, (fneg B)) -> (fsub A, B)
  if ((!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT)) &&
    isNegatibleForFree(N1, LegalOperations, TLI, &DAG.getTarget().Options) == 2)
    return DAG.getNode(ISD::FSUB, SDLoc(N), VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));
  // fold (fadd (fneg A), B) -> (fsub B, A)
  if ((!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT)) &&
    isNegatibleForFree(N0, LegalOperations, TLI, &DAG.getTarget().Options) == 2)
    return DAG.getNode(ISD::FSUB, SDLoc(N), VT, N1,
                       GetNegatedExpression(N0, DAG, LegalOperations));

  // If allowed, fold (fadd (fadd x, c1), c2) -> (fadd x, (fadd c1, c2))
  if (DAG.getTarget().Options.UnsafeFPMath && N1CFP &&
      N0.getOpcode() == ISD::FADD && N0.getNode()->hasOneUse() &&
      isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                   N0.getOperand(1), N1));

  // No FP constant should be created after legalization as Instruction
  // Selection pass has hard time in dealing with FP constant.
  //
  // We don't need test this condition for transformation like following, as
  // the DAG being transformed implies it is legal to take FP constant as
  // operand.
  //
  //  (fadd (fmul c, x), x) -> (fmul c+1, x)
  //
  bool AllowNewFpConst = (Level < AfterLegalizeDAG);

  // If allow, fold (fadd (fneg x), x) -> 0.0
  if (AllowNewFpConst && DAG.getTarget().Options.UnsafeFPMath &&
      N0.getOpcode() == ISD::FNEG && N0.getOperand(0) == N1)
    return DAG.getConstantFP(0.0, VT);

    // If allow, fold (fadd x, (fneg x)) -> 0.0
  if (AllowNewFpConst && DAG.getTarget().Options.UnsafeFPMath &&
      N1.getOpcode() == ISD::FNEG && N1.getOperand(0) == N0)
    return DAG.getConstantFP(0.0, VT);

  // In unsafe math mode, we can fold chains of FADD's of the same value
  // into multiplications.  This transform is not safe in general because
  // we are reducing the number of rounding steps.
  if (DAG.getTarget().Options.UnsafeFPMath &&
      TLI.isOperationLegalOrCustom(ISD::FMUL, VT) &&
      !N0CFP && !N1CFP) {
    if (N0.getOpcode() == ISD::FMUL) {
      ConstantFPSDNode *CFP00 = dyn_cast<ConstantFPSDNode>(N0.getOperand(0));
      ConstantFPSDNode *CFP01 = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));

      // (fadd (fmul c, x), x) -> (fmul x, c+1)
      if (CFP00 && !CFP01 && N0.getOperand(1) == N1) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP00, 0),
                                     DAG.getConstantFP(1.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N1, NewCFP);
      }

      // (fadd (fmul x, c), x) -> (fmul x, c+1)
      if (CFP01 && !CFP00 && N0.getOperand(0) == N1) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP01, 0),
                                     DAG.getConstantFP(1.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N1, NewCFP);
      }

      // (fadd (fmul c, x), (fadd x, x)) -> (fmul x, c+2)
      if (CFP00 && !CFP01 && N1.getOpcode() == ISD::FADD &&
          N1.getOperand(0) == N1.getOperand(1) &&
          N0.getOperand(1) == N1.getOperand(0)) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP00, 0),
                                     DAG.getConstantFP(2.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N0.getOperand(1), NewCFP);
      }

      // (fadd (fmul x, c), (fadd x, x)) -> (fmul x, c+2)
      if (CFP01 && !CFP00 && N1.getOpcode() == ISD::FADD &&
          N1.getOperand(0) == N1.getOperand(1) &&
          N0.getOperand(0) == N1.getOperand(0)) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP01, 0),
                                     DAG.getConstantFP(2.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N0.getOperand(0), NewCFP);
      }
    }

    if (N1.getOpcode() == ISD::FMUL) {
      ConstantFPSDNode *CFP10 = dyn_cast<ConstantFPSDNode>(N1.getOperand(0));
      ConstantFPSDNode *CFP11 = dyn_cast<ConstantFPSDNode>(N1.getOperand(1));

      // (fadd x, (fmul c, x)) -> (fmul x, c+1)
      if (CFP10 && !CFP11 && N1.getOperand(1) == N0) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP10, 0),
                                     DAG.getConstantFP(1.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N0, NewCFP);
      }

      // (fadd x, (fmul x, c)) -> (fmul x, c+1)
      if (CFP11 && !CFP10 && N1.getOperand(0) == N0) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP11, 0),
                                     DAG.getConstantFP(1.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N0, NewCFP);
      }


      // (fadd (fadd x, x), (fmul c, x)) -> (fmul x, c+2)
      if (CFP10 && !CFP11 && N0.getOpcode() == ISD::FADD &&
          N0.getOperand(0) == N0.getOperand(1) &&
          N1.getOperand(1) == N0.getOperand(0)) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP10, 0),
                                     DAG.getConstantFP(2.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N1.getOperand(1), NewCFP);
      }

      // (fadd (fadd x, x), (fmul x, c)) -> (fmul x, c+2)
      if (CFP11 && !CFP10 && N0.getOpcode() == ISD::FADD &&
          N0.getOperand(0) == N0.getOperand(1) &&
          N1.getOperand(0) == N0.getOperand(0)) {
        SDValue NewCFP = DAG.getNode(ISD::FADD, SDLoc(N), VT,
                                     SDValue(CFP11, 0),
                                     DAG.getConstantFP(2.0, VT));
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N1.getOperand(0), NewCFP);
      }
    }

    if (N0.getOpcode() == ISD::FADD && AllowNewFpConst) {
      ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N0.getOperand(0));
      // (fadd (fadd x, x), x) -> (fmul x, 3.0)
      if (!CFP && N0.getOperand(0) == N0.getOperand(1) &&
          (N0.getOperand(0) == N1))
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N1, DAG.getConstantFP(3.0, VT));
    }

    if (N1.getOpcode() == ISD::FADD && AllowNewFpConst) {
      ConstantFPSDNode *CFP10 = dyn_cast<ConstantFPSDNode>(N1.getOperand(0));
      // (fadd x, (fadd x, x)) -> (fmul x, 3.0)
      if (!CFP10 && N1.getOperand(0) == N1.getOperand(1) &&
          N1.getOperand(0) == N0)
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           N0, DAG.getConstantFP(3.0, VT));
    }

    // (fadd (fadd x, x), (fadd x, x)) -> (fmul x, 4.0)
    if (AllowNewFpConst &&
        N0.getOpcode() == ISD::FADD && N1.getOpcode() == ISD::FADD &&
        N0.getOperand(0) == N0.getOperand(1) &&
        N1.getOperand(0) == N1.getOperand(1) &&
        N0.getOperand(0) == N1.getOperand(0))
      return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                         N0.getOperand(0),
                         DAG.getConstantFP(4.0, VT));
  }

  // FADD -> FMA combines:
  if ((DAG.getTarget().Options.AllowFPOpFusion == FPOpFusion::Fast ||
       DAG.getTarget().Options.UnsafeFPMath) &&
      DAG.getTarget().getTargetLowering()->isFMAFasterThanFMulAndFAdd(VT) &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FMA, VT))) {

    // fold (fadd (fmul x, y), z) -> (fma x, y, z)
    if (N0.getOpcode() == ISD::FMUL && N0->hasOneUse())
      return DAG.getNode(ISD::FMA, SDLoc(N), VT,
                         N0.getOperand(0), N0.getOperand(1), N1);

    // fold (fadd x, (fmul y, z)) -> (fma y, z, x)
    // Note: Commutes FADD operands.
    if (N1.getOpcode() == ISD::FMUL && N1->hasOneUse())
      return DAG.getNode(ISD::FMA, SDLoc(N), VT,
                         N1.getOperand(0), N1.getOperand(1), N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitFSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc dl(N);

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FSUB, SDLoc(N), VT, N0, N1);
  // fold (fsub A, 0) -> A
  if (DAG.getTarget().Options.UnsafeFPMath &&
      N1CFP && N1CFP->getValueAPF().isZero())
    return N0;
  // fold (fsub 0, B) -> -B
  if (DAG.getTarget().Options.UnsafeFPMath &&
      N0CFP && N0CFP->getValueAPF().isZero()) {
    if (isNegatibleForFree(N1, LegalOperations, TLI, &DAG.getTarget().Options))
      return GetNegatedExpression(N1, DAG, LegalOperations);
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, dl, VT, N1);
  }
  // fold (fsub A, (fneg B)) -> (fadd A, B)
  if (isNegatibleForFree(N1, LegalOperations, TLI, &DAG.getTarget().Options))
    return DAG.getNode(ISD::FADD, dl, VT, N0,
                       GetNegatedExpression(N1, DAG, LegalOperations));

  // If 'unsafe math' is enabled, fold
  //    (fsub x, x) -> 0.0 &
  //    (fsub x, (fadd x, y)) -> (fneg y) &
  //    (fsub x, (fadd y, x)) -> (fneg y)
  if (DAG.getTarget().Options.UnsafeFPMath) {
    if (N0 == N1)
      return DAG.getConstantFP(0.0f, VT);

    if (N1.getOpcode() == ISD::FADD) {
      SDValue N10 = N1->getOperand(0);
      SDValue N11 = N1->getOperand(1);

      if (N10 == N0 && isNegatibleForFree(N11, LegalOperations, TLI,
                                          &DAG.getTarget().Options))
        return GetNegatedExpression(N11, DAG, LegalOperations);

      if (N11 == N0 && isNegatibleForFree(N10, LegalOperations, TLI,
                                          &DAG.getTarget().Options))
        return GetNegatedExpression(N10, DAG, LegalOperations);
    }
  }

  // FSUB -> FMA combines:
  if ((DAG.getTarget().Options.AllowFPOpFusion == FPOpFusion::Fast ||
       DAG.getTarget().Options.UnsafeFPMath) &&
      DAG.getTarget().getTargetLowering()->isFMAFasterThanFMulAndFAdd(VT) &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FMA, VT))) {

    // fold (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
    if (N0.getOpcode() == ISD::FMUL && N0->hasOneUse())
      return DAG.getNode(ISD::FMA, dl, VT,
                         N0.getOperand(0), N0.getOperand(1),
                         DAG.getNode(ISD::FNEG, dl, VT, N1));

    // fold (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
    // Note: Commutes FSUB operands.
    if (N1.getOpcode() == ISD::FMUL && N1->hasOneUse())
      return DAG.getNode(ISD::FMA, dl, VT,
                         DAG.getNode(ISD::FNEG, dl, VT,
                         N1.getOperand(0)),
                         N1.getOperand(1), N0);

    // fold (fsub (fneg (fmul, x, y)), z) -> (fma (fneg x), y, (fneg z))
    if (N0.getOpcode() == ISD::FNEG &&
        N0.getOperand(0).getOpcode() == ISD::FMUL &&
        N0->hasOneUse() && N0.getOperand(0).hasOneUse()) {
      SDValue N00 = N0.getOperand(0).getOperand(0);
      SDValue N01 = N0.getOperand(0).getOperand(1);
      return DAG.getNode(ISD::FMA, dl, VT,
                         DAG.getNode(ISD::FNEG, dl, VT, N00), N01,
                         DAG.getNode(ISD::FNEG, dl, VT, N1));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FMUL, SDLoc(N), VT, N0, N1);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMUL, SDLoc(N), VT, N1, N0);
  // fold (fmul A, 0) -> 0
  if (DAG.getTarget().Options.UnsafeFPMath &&
      N1CFP && N1CFP->getValueAPF().isZero())
    return N1;
  // fold (fmul A, 0) -> 0, vector edition.
  if (DAG.getTarget().Options.UnsafeFPMath &&
      ISD::isBuildVectorAllZeros(N1.getNode()))
    return N1;
  // fold (fmul A, 1.0) -> A
  if (N1CFP && N1CFP->isExactlyValue(1.0))
    return N0;
  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0, N0);
  // fold (fmul X, -1.0) -> (fneg X)
  if (N1CFP && N1CFP->isExactlyValue(-1.0))
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, SDLoc(N), VT, N0);

  // fold (fmul (fneg X), (fneg Y)) -> (fmul X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations, TLI,
                                       &DAG.getTarget().Options)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations, TLI,
                                         &DAG.getTarget().Options)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                           GetNegatedExpression(N0, DAG, LegalOperations),
                           GetNegatedExpression(N1, DAG, LegalOperations));
    }
  }

  // If allowed, fold (fmul (fmul x, c1), c2) -> (fmul x, (fmul c1, c2))
  if (DAG.getTarget().Options.UnsafeFPMath &&
      N1CFP && N0.getOpcode() == ISD::FMUL &&
      N0.getNode()->hasOneUse() && isa<ConstantFPSDNode>(N0.getOperand(1)))
    return DAG.getNode(ISD::FMUL, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                                   N0.getOperand(1), N1));

  return SDValue();
}

SDValue DAGCombiner::visitFMA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc dl(N);

  if (DAG.getTarget().Options.UnsafeFPMath) {
    if (N0CFP && N0CFP->isZero())
      return N2;
    if (N1CFP && N1CFP->isZero())
      return N2;
  }
  if (N0CFP && N0CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N1, N2);
  if (N1CFP && N1CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0, N2);

  // Canonicalize (fma c, x, y) -> (fma x, c, y)
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMA, SDLoc(N), VT, N1, N0, N2);

  // (fma x, c1, (fmul x, c2)) -> (fmul x, c1+c2)
  if (DAG.getTarget().Options.UnsafeFPMath && N1CFP &&
      N2.getOpcode() == ISD::FMUL &&
      N0 == N2.getOperand(0) &&
      N2.getOperand(1).getOpcode() == ISD::ConstantFP) {
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT, N1, N2.getOperand(1)));
  }


  // (fma (fmul x, c1), c2, y) -> (fma x, c1*c2, y)
  if (DAG.getTarget().Options.UnsafeFPMath &&
      N0.getOpcode() == ISD::FMUL && N1CFP &&
      N0.getOperand(1).getOpcode() == ISD::ConstantFP) {
    return DAG.getNode(ISD::FMA, dl, VT,
                       N0.getOperand(0),
                       DAG.getNode(ISD::FMUL, dl, VT, N1, N0.getOperand(1)),
                       N2);
  }

  // (fma x, 1, y) -> (fadd x, y)
  // (fma x, -1, y) -> (fadd (fneg x), y)
  if (N1CFP) {
    if (N1CFP->isExactlyValue(1.0))
      return DAG.getNode(ISD::FADD, dl, VT, N0, N2);

    if (N1CFP->isExactlyValue(-1.0) &&
        (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))) {
      SDValue RHSNeg = DAG.getNode(ISD::FNEG, dl, VT, N0);
      AddToWorkList(RHSNeg.getNode());
      return DAG.getNode(ISD::FADD, dl, VT, N2, RHSNeg);
    }
  }

  // (fma x, c, x) -> (fmul x, (c+1))
  if (DAG.getTarget().Options.UnsafeFPMath && N1CFP && N0 == N2)
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT,
                                   N1, DAG.getConstantFP(1.0, VT)));

  // (fma x, c, (fneg x)) -> (fmul x, (c-1))
  if (DAG.getTarget().Options.UnsafeFPMath && N1CFP &&
      N2.getOpcode() == ISD::FNEG && N2.getOperand(0) == N0)
    return DAG.getNode(ISD::FMUL, dl, VT, N0,
                       DAG.getNode(ISD::FADD, dl, VT,
                                   N1, DAG.getConstantFP(-1.0, VT)));


  return SDValue();
}

SDValue DAGCombiner::visitFDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // fold vector ops
  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVBinOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FDIV, SDLoc(N), VT, N0, N1);

  // fold (fdiv X, c2) -> fmul X, 1/c2 if losing precision is acceptable.
  if (N1CFP && DAG.getTarget().Options.UnsafeFPMath) {
    // Compute the reciprocal 1.0 / c2.
    APFloat N1APF = N1CFP->getValueAPF();
    APFloat Recip(N1APF.getSemantics(), 1); // 1.0
    APFloat::opStatus st = Recip.divide(N1APF, APFloat::rmNearestTiesToEven);
    // Only do the transform if the reciprocal is a legal fp immediate that
    // isn't too nasty (eg NaN, denormal, ...).
    if ((st == APFloat::opOK || st == APFloat::opInexact) && // Not too nasty
        (!LegalOperations ||
         // FIXME: custom lowering of ConstantFP might fail (see e.g. ARM
         // backend)... we should handle this gracefully after Legalize.
         // TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT) ||
         TLI.isOperationLegal(llvm::ISD::ConstantFP, VT) ||
         TLI.isFPImmLegal(Recip, VT)))
      return DAG.getNode(ISD::FMUL, SDLoc(N), VT, N0,
                         DAG.getConstantFP(Recip, VT));
  }

  // (fdiv (fneg X), (fneg Y)) -> (fdiv X, Y)
  if (char LHSNeg = isNegatibleForFree(N0, LegalOperations, TLI,
                                       &DAG.getTarget().Options)) {
    if (char RHSNeg = isNegatibleForFree(N1, LegalOperations, TLI,
                                         &DAG.getTarget().Options)) {
      // Both can be negated for free, check to see if at least one is cheaper
      // negated.
      if (LHSNeg == 2 || RHSNeg == 2)
        return DAG.getNode(ISD::FDIV, SDLoc(N), VT,
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
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FREM, SDLoc(N), VT, N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);

  if (N0CFP && N1CFP)  // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0, N1);

  if (N1CFP) {
    const APFloat& V = N1CFP->getValueAPF();
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    if (!V.isNegative()) {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FABS, VT))
        return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);
    } else {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, SDLoc(N), VT,
                           DAG.getNode(ISD::FABS, SDLoc(N0), VT, N0));
    }
  }

  // copysign(fabs(x), y) -> copysign(x, y)
  // copysign(fneg(x), y) -> copysign(x, y)
  // copysign(copysign(x,z), y) -> copysign(x, y)
  if (N0.getOpcode() == ISD::FABS || N0.getOpcode() == ISD::FNEG ||
      N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0.getOperand(0), N1);

  // copysign(x, abs(y)) -> abs(x)
  if (N1.getOpcode() == ISD::FABS)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);

  // copysign(x, copysign(y,z)) -> copysign(x, z)
  if (N1.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0, N1.getOperand(1));

  // copysign(x, fp_extend(y)) -> copysign(x, y)
  // copysign(x, fp_round(y)) -> copysign(x, y)
  if (N1.getOpcode() == ISD::FP_EXTEND || N1.getOpcode() == ISD::FP_ROUND)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       N0, N1.getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (sint_to_fp c1) -> c1fp
  if (N0C &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and SINT_TO_FP is not legal on this target,
  // but UINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to UINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);
  }

  // The next optimizations are desireable only if SELECT_CC can be lowered.
  // Check against MVT::Other for SELECT_CC, which is a workaround for targets
  // having to say they don't support SELECT_CC on every type the DAG knows
  // about, since there is no way to mark an opcode illegal at all value types
  // (See also visitSELECT)
  if (TLI.isOperationLegalOrCustom(ISD::SELECT_CC, MVT::Other)) {
    // fold (sint_to_fp (setcc x, y, cc)) -> (select_cc x, y, -1.0, 0.0,, cc)
    if (N0.getOpcode() == ISD::SETCC && N0.getValueType() == MVT::i1 &&
        !VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDValue Ops[] =
        { N0.getOperand(0), N0.getOperand(1),
          DAG.getConstantFP(-1.0, VT) , DAG.getConstantFP(0.0, VT),
          N0.getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), VT, Ops, 5);
    }

    // fold (sint_to_fp (zext (setcc x, y, cc))) ->
    //      (select_cc x, y, 1.0, 0.0,, cc)
    if (N0.getOpcode() == ISD::ZERO_EXTEND &&
        N0.getOperand(0).getOpcode() == ISD::SETCC &&!VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDValue Ops[] =
        { N0.getOperand(0).getOperand(0), N0.getOperand(0).getOperand(1),
          DAG.getConstantFP(1.0, VT) , DAG.getConstantFP(0.0, VT),
          N0.getOperand(0).getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), VT, Ops, 5);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // fold (uint_to_fp c1) -> c1fp
  if (N0C &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and UINT_TO_FP is not legal on this target,
  // but SINT_TO_FP is legal on this target, try to convert.
  if (!TLI.isOperationLegalOrCustom(ISD::UINT_TO_FP, OpVT) &&
      TLI.isOperationLegalOrCustom(ISD::SINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to SINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);
  }

  // The next optimizations are desireable only if SELECT_CC can be lowered.
  // Check against MVT::Other for SELECT_CC, which is a workaround for targets
  // having to say they don't support SELECT_CC on every type the DAG knows
  // about, since there is no way to mark an opcode illegal at all value types
  // (See also visitSELECT)
  if (TLI.isOperationLegalOrCustom(ISD::SELECT_CC, MVT::Other)) {
    // fold (uint_to_fp (setcc x, y, cc)) -> (select_cc x, y, -1.0, 0.0,, cc)

    if (N0.getOpcode() == ISD::SETCC && !VT.isVector() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(llvm::ISD::ConstantFP, VT))) {
      SDValue Ops[] =
        { N0.getOperand(0), N0.getOperand(1),
          DAG.getConstantFP(1.0, VT),  DAG.getConstantFP(0.0, VT),
          N0.getOperand(2) };
      return DAG.getNode(ISD::SELECT_CC, SDLoc(N), VT, Ops, 5);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_SINT, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP)
    return DAG.getNode(ISD::FP_TO_UINT, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT, N0, N1);

  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);

  // fold (fp_round (fp_round x)) -> (fp_round x)
  if (N0.getOpcode() == ISD::FP_ROUND) {
    // This is a value preserving truncation if both round's are.
    bool IsTrunc = N->getConstantOperandVal(1) == 1 &&
                   N0.getNode()->getConstantOperandVal(1) == 1;
    return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT, N0.getOperand(0),
                       DAG.getIntPtrConstant(IsTrunc));
  }

  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse()) {
    SDValue Tmp = DAG.getNode(ISD::FP_ROUND, SDLoc(N0), VT,
                              N0.getOperand(0), N1);
    AddToWorkList(Tmp.getNode());
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
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
  if (N0CFP && isTypeLegal(EVT)) {
    SDValue Round = DAG.getConstantFP(*N0CFP->getConstantFPValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, Round);
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
  if (N0CFP)
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, N0);

  // Turn fp_extend(fp_round(X, 1)) -> x since the fp_round doesn't affect the
  // value of X.
  if (N0.getOpcode() == ISD::FP_ROUND
      && N0.getNode()->getConstantOperandVal(1) == 1) {
    SDValue In = N0.getOperand(0);
    if (In.getValueType() == VT) return In;
    if (VT.bitsLT(In.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT,
                         In, N0.getOperand(1));
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, In);
  }

  // fold (fpext (load x)) -> (fpext (fptrunc (extload x)))
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      ((!LegalOperations && !cast<LoadSDNode>(N0)->isVolatile()) ||
       TLI.isLoadExtLegal(ISD::EXTLOAD, N0.getValueType()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), N0.getValueType(),
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::FP_ROUND, SDLoc(N0),
                          N0.getValueType(), ExtLoad, DAG.getIntPtrConstant(1)),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  return SDValue();
}

SDValue DAGCombiner::visitFNEG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVUnaryOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  if (isNegatibleForFree(N0, LegalOperations, DAG.getTargetLoweringInfo(),
                         &DAG.getTarget().Options))
    return GetNegatedExpression(N0, DAG, LegalOperations);

  // Transform fneg(bitconvert(x)) -> bitconvert(x^sign) to avoid loading
  // constant pool values.
  if (!TLI.isFNegFree(VT) && N0.getOpcode() == ISD::BITCAST &&
      !VT.isVector() &&
      N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::XOR, SDLoc(N0), IntVT, Int,
              DAG.getConstant(APInt::getSignBit(IntVT.getSizeInBits()), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BITCAST, SDLoc(N),
                         VT, Int);
    }
  }

  // (fneg (fmul c, x)) -> (fmul -c, x)
  if (N0.getOpcode() == ISD::FMUL) {
    ConstantFPSDNode *CFP1 = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));
    if (CFP1)
      return DAG.getNode(ISD::FMUL, SDLoc(N), VT,
                         N0.getOperand(0),
                         DAG.getNode(ISD::FNEG, SDLoc(N), VT,
                                     N0.getOperand(1)));
  }

  return SDValue();
}

SDValue DAGCombiner::visitFCEIL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fceil c1) -> fceil(c1)
  if (N0CFP)
    return DAG.getNode(ISD::FCEIL, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFTRUNC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (ftrunc c1) -> ftrunc(c1)
  if (N0CFP)
    return DAG.getNode(ISD::FTRUNC, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFFLOOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (ffloor c1) -> ffloor(c1)
  if (N0CFP)
    return DAG.getNode(ISD::FFLOOR, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  if (VT.isVector()) {
    SDValue FoldedVOp = SimplifyVUnaryOp(N);
    if (FoldedVOp.getNode()) return FoldedVOp;
  }

  // fold (fabs c1) -> fabs(c1)
  if (N0CFP)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);
  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0.getOperand(0));

  // Transform fabs(bitconvert(x)) -> bitconvert(x&~sign) to avoid loading
  // constant pool values.
  if (!TLI.isFAbsFree(VT) &&
      N0.getOpcode() == ISD::BITCAST && N0.getNode()->hasOneUse() &&
      N0.getOperand(0).getValueType().isInteger() &&
      !N0.getOperand(0).getValueType().isVector()) {
    SDValue Int = N0.getOperand(0);
    EVT IntVT = Int.getValueType();
    if (IntVT.isInteger() && !IntVT.isVector()) {
      Int = DAG.getNode(ISD::AND, SDLoc(N0), IntVT, Int,
             DAG.getConstant(~APInt::getSignBit(IntVT.getSizeInBits()), IntVT));
      AddToWorkList(Int.getNode());
      return DAG.getNode(ISD::BITCAST, SDLoc(N),
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
      TLI.isOperationLegalOrCustom(ISD::BR_CC,
                                   N1.getOperand(0).getValueType())) {
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }

  if ((N1.hasOneUse() && N1.getOpcode() == ISD::SRL) ||
      ((N1.getOpcode() == ISD::TRUNCATE && N1.hasOneUse()) &&
       (N1.getOperand(0).hasOneUse() &&
        N1.getOperand(0).getOpcode() == ISD::SRL))) {
    SDNode *Trunc = 0;
    if (N1.getOpcode() == ISD::TRUNCATE) {
      // Look pass the truncate.
      Trunc = N1.getNode();
      N1 = N1.getOperand(0);
    }

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
            DAG.getSetCC(SDLoc(N),
                         getSetCCResultType(Op0.getValueType()),
                         Op0, DAG.getConstant(0, Op0.getValueType()),
                         ISD::SETNE);

          SDValue NewBRCond = DAG.getNode(ISD::BRCOND, SDLoc(N),
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
          DAG.ReplaceAllUsesOfValueWith(N1, SetCC);
          removeFromWorkList(N1.getNode());
          DAG.DeleteNode(N1.getNode());
          return SDValue(N, 0);   // Return N so it doesn't get rechecked!
        }
      }
    }

    if (Trunc)
      // Restore N1 if the above transformation doesn't match.
      N1 = N->getOperand(1);
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
        if (Tmp.getNode() != TheXor) {
          DEBUG(dbgs() << "\nReplacing.8 ";
                TheXor->dump(&DAG);
                dbgs() << "\nWith: ";
                Tmp.getNode()->dump(&DAG);
                dbgs() << '\n');
          WorkListRemover DeadNodes(*this);
          DAG.ReplaceAllUsesOfValueWith(N1, Tmp);
          removeFromWorkList(TheXor);
          DAG.DeleteNode(TheXor);
          return DAG.getNode(ISD::BRCOND, SDLoc(N),
                             MVT::Other, Chain, Tmp, N2);
        }

        // visitXOR has changed XOR's operands or replaced the XOR completely,
        // bail out.
        return SDValue(N, 0);
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

      EVT SetCCVT = N1.getValueType();
      if (LegalTypes)
        SetCCVT = getSetCCResultType(SetCCVT);
      SDValue SetCC = DAG.getSetCC(SDLoc(TheXor),
                                   SetCCVT,
                                   Op0, Op1,
                                   Equal ? ISD::SETEQ : ISD::SETNE);
      // Replace the uses of XOR with SETCC
      WorkListRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N1, SetCC);
      removeFromWorkList(N1.getNode());
      DAG.DeleteNode(N1.getNode());
      return DAG.getNode(ISD::BRCOND, SDLoc(N),
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
  SDValue Simp = SimplifySetCC(getSetCCResultType(CondLHS.getValueType()),
                               CondLHS, CondRHS, CC->get(), SDLoc(N),
                               false);
  if (Simp.getNode()) AddToWorkList(Simp.getNode());

  // fold to a simpler setcc
  if (Simp.getNode() && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       N->getOperand(0), Simp.getOperand(2),
                       Simp.getOperand(0), Simp.getOperand(1),
                       N->getOperand(4));

  return SDValue();
}

/// canFoldInAddressingMode - Return true if 'Use' is a load or a store that
/// uses N as its base pointer and that N may be folded in the load / store
/// addressing mode.
static bool canFoldInAddressingMode(SDNode *N, SDNode *Use,
                                    SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  EVT VT;
  if (LoadSDNode *LD  = dyn_cast<LoadSDNode>(Use)) {
    if (LD->isIndexed() || LD->getBasePtr().getNode() != N)
      return false;
    VT = Use->getValueType(0);
  } else if (StoreSDNode *ST  = dyn_cast<StoreSDNode>(Use)) {
    if (ST->isIndexed() || ST->getBasePtr().getNode() != N)
      return false;
    VT = ST->getValue().getValueType();
  } else
    return false;

  TargetLowering::AddrMode AM;
  if (N->getOpcode() == ISD::ADD) {
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else if (N->getOpcode() == ISD::SUB) {
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = -Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else
    return false;

  return TLI.isLegalAddressingMode(AM, VT.getTypeForEVT(*DAG.getContext()));
}

/// CombineToPreIndexedLoadStore - Try turning a load / store into a
/// pre-indexed load / store when the base pointer is an add or subtract
/// and it has other uses besides the load / store. After the
/// transformation, the new indexed load / store has effectively folded
/// the add / subtract in and all of its other uses are redirected to the
/// new load / store.
bool DAGCombiner::CombineToPreIndexedLoadStore(SDNode *N) {
  if (Level < AfterLegalizeDAG)
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

  // Backends without true r+i pre-indexed forms may need to pass a
  // constant base with a variable offset so that constant coercion
  // will work with the patterns in canonical form.
  bool Swapped = false;
  if (isa<ConstantSDNode>(BasePtr)) {
    std::swap(BasePtr, Offset);
    Swapped = true;
  }

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

  // If the offset is a constant, there may be other adds of constants that
  // can be folded with this one. We should do this to avoid having to keep
  // a copy of the original base pointer.
  SmallVector<SDNode *, 16> OtherUses;
  if (isa<ConstantSDNode>(Offset))
    for (SDNode::use_iterator I = BasePtr.getNode()->use_begin(),
         E = BasePtr.getNode()->use_end(); I != E; ++I) {
      SDNode *Use = *I;
      if (Use == Ptr.getNode())
        continue;

      if (Use->isPredecessorOf(N))
        continue;

      if (Use->getOpcode() != ISD::ADD && Use->getOpcode() != ISD::SUB) {
        OtherUses.clear();
        break;
      }

      SDValue Op0 = Use->getOperand(0), Op1 = Use->getOperand(1);
      if (Op1.getNode() == BasePtr.getNode())
        std::swap(Op0, Op1);
      assert(Op0.getNode() == BasePtr.getNode() &&
             "Use of ADD/SUB but not an operand");

      if (!isa<ConstantSDNode>(Op1)) {
        OtherUses.clear();
        break;
      }

      // FIXME: In some cases, we can be smarter about this.
      if (Op1.getValueType() != Offset.getValueType()) {
        OtherUses.clear();
        break;
      }

      OtherUses.push_back(Use);
    }

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Now check for #3 and #4.
  bool RealUse = false;

  // Caches for hasPredecessorHelper
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;

  for (SDNode::use_iterator I = Ptr.getNode()->use_begin(),
         E = Ptr.getNode()->use_end(); I != E; ++I) {
    SDNode *Use = *I;
    if (Use == N)
      continue;
    if (N->hasPredecessorHelper(Use, Visited, Worklist))
      return false;

    // If Ptr may be folded in addressing mode of other use, then it's
    // not profitable to do this transformation.
    if (!canFoldInAddressingMode(Ptr.getNode(), Use, DAG, TLI))
      RealUse = true;
  }

  if (!RealUse)
    return false;

  SDValue Result;
  if (isLoad)
    Result = DAG.getIndexedLoad(SDValue(N,0), SDLoc(N),
                                BasePtr, Offset, AM);
  else
    Result = DAG.getIndexedStore(SDValue(N,0), SDLoc(N),
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
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
  }

  // Finally, since the node is now dead, remove it from the graph.
  DAG.DeleteNode(N);

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Replace other uses of BasePtr that can be updated to use Ptr
  for (unsigned i = 0, e = OtherUses.size(); i != e; ++i) {
    unsigned OffsetIdx = 1;
    if (OtherUses[i]->getOperand(OffsetIdx).getNode() == BasePtr.getNode())
      OffsetIdx = 0;
    assert(OtherUses[i]->getOperand(!OffsetIdx).getNode() ==
           BasePtr.getNode() && "Expected BasePtr operand");

    // We need to replace ptr0 in the following expression:
    //   x0 * offset0 + y0 * ptr0 = t0
    // knowing that
    //   x1 * offset1 + y1 * ptr0 = t1 (the indexed load/store)
    //
    // where x0, x1, y0 and y1 in {-1, 1} are given by the types of the
    // indexed load/store and the expresion that needs to be re-written.
    //
    // Therefore, we have:
    //   t0 = (x0 * offset0 - x1 * y0 * y1 *offset1) + (y0 * y1) * t1

    ConstantSDNode *CN =
      cast<ConstantSDNode>(OtherUses[i]->getOperand(OffsetIdx));
    int X0, X1, Y0, Y1;
    APInt Offset0 = CN->getAPIntValue();
    APInt Offset1 = cast<ConstantSDNode>(Offset)->getAPIntValue();

    X0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 1) ? -1 : 1;
    Y0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 0) ? -1 : 1;
    X1 = (AM == ISD::PRE_DEC && !Swapped) ? -1 : 1;
    Y1 = (AM == ISD::PRE_DEC && Swapped) ? -1 : 1;

    unsigned Opcode = (Y0 * Y1 < 0) ? ISD::SUB : ISD::ADD;

    APInt CNV = Offset0;
    if (X0 < 0) CNV = -CNV;
    if (X1 * Y0 * Y1 < 0) CNV = CNV + Offset1;
    else CNV = CNV - Offset1;

    // We can now generate the new expression.
    SDValue NewOp1 = DAG.getConstant(CNV, CN->getValueType(0));
    SDValue NewOp2 = Result.getValue(isLoad ? 1 : 0);

    SDValue NewUse = DAG.getNode(Opcode,
                                 SDLoc(OtherUses[i]),
                                 OtherUses[i]->getValueType(0), NewOp1, NewOp2);
    DAG.ReplaceAllUsesOfValueWith(SDValue(OtherUses[i], 0), NewUse);
    removeFromWorkList(OtherUses[i]);
    DAG.DeleteNode(OtherUses[i]);
  }

  // Replace the uses of Ptr with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(Ptr, Result.getValue(isLoad ? 1 : 0));
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
  if (Level < AfterLegalizeDAG)
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
      // Don't create a indexed load / store with zero offset.
      if (isa<ConstantSDNode>(Offset) &&
          cast<ConstantSDNode>(Offset)->isNullValue())
        continue;

      // Try turning it into a post-indexed load / store except when
      // 1) All uses are load / store ops that use it as base ptr (and
      //    it may be folded as addressing mmode).
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
            if (!canFoldInAddressingMode(Use, UseUse, DAG, TLI))
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
          ? DAG.getIndexedLoad(SDValue(N,0), SDLoc(N),
                               BasePtr, Offset, AM)
          : DAG.getIndexedStore(SDValue(N,0), SDLoc(N),
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
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
        } else {
          DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
        }

        // Finally, since the node is now dead, remove it from the graph.
        DAG.DeleteNode(N);

        // Replace the uses of Use with uses of the updated base value.
        DAG.ReplaceAllUsesOfValueWith(SDValue(Op, 0),
                                      Result.getValue(isLoad ? 1 : 0));
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

  // If load is not volatile and there are no uses of the loaded value (and
  // the updated indexed value in case of indexed loads), change uses of the
  // chain value into uses of the chain input (i.e. delete the dead load).
  if (!LD->isVolatile()) {
    if (N->getValueType(1) == MVT::Other) {
      // Unindexed loads.
      if (!N->hasAnyUseOfValue(0)) {
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
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);

        if (N->use_empty()) {
          removeFromWorkList(N);
          DAG.DeleteNode(N);
        }

        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    } else {
      // Indexed loads.
      assert(N->getValueType(2) == MVT::Other && "Malformed indexed loads?");
      if (!N->hasAnyUseOfValue(0) && !N->hasAnyUseOfValue(1)) {
        SDValue Undef = DAG.getUNDEF(N->getValueType(0));
        DEBUG(dbgs() << "\nReplacing.7 ";
              N->dump(&DAG);
              dbgs() << "\nWith: ";
              Undef.getNode()->dump(&DAG);
              dbgs() << " and 2 other values\n");
        WorkListRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Undef);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1),
                                      DAG.getUNDEF(N->getValueType(1)));
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 2), Chain);
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
  if (ISD::isNormalLoad(N) && !LD->isVolatile()) {
    if (ISD::isNON_TRUNCStore(Chain.getNode())) {
      StoreSDNode *PrevST = cast<StoreSDNode>(Chain);
      if (PrevST->getBasePtr() == Ptr &&
          PrevST->getValue().getValueType() == N->getValueType(0))
      return CombineTo(N, Chain.getOperand(1), Chain);
    }
  }

  // Try to infer better alignment information than the load already has.
  if (OptLevel != CodeGenOpt::None && LD->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > LD->getMemOperand()->getBaseAlignment()) {
        SDValue NewLoad =
               DAG.getExtLoad(LD->getExtensionType(), SDLoc(N),
                              LD->getValueType(0),
                              Chain, Ptr, LD->getPointerInfo(),
                              LD->getMemoryVT(),
                              LD->isVolatile(), LD->isNonTemporal(), Align,
                              LD->getTBAAInfo());
        return CombineTo(N, NewLoad, SDValue(NewLoad.getNode(), 1), true);
      }
    }
  }

  bool UseAA = CombinerAA.getNumOccurrences() > 0 ? CombinerAA :
    TLI.getTargetMachine().getSubtarget<TargetSubtargetInfo>().useAA();
  if (UseAA) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplLoad;

      // Replace the chain to void dependency.
      if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
        ReplLoad = DAG.getLoad(N->getValueType(0), SDLoc(LD),
                               BetterChain, Ptr, LD->getMemOperand());
      } else {
        ReplLoad = DAG.getExtLoad(LD->getExtensionType(), SDLoc(LD),
                                  LD->getValueType(0),
                                  BetterChain, Ptr, LD->getMemoryVT(),
                                  LD->getMemOperand());
      }

      // Create token factor to keep old chain connected.
      SDValue Token = DAG.getNode(ISD::TokenFactor, SDLoc(N),
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

  // Try to slice up N to more direct loads if the slices are mapped to
  // different register banks or pairing can take place.
  if (SliceUpLoad(N))
    return SDValue(N, 0);

  return SDValue();
}

namespace {
/// \brief Helper structure used to slice a load in smaller loads.
/// Basically a slice is obtained from the following sequence:
/// Origin = load Ty1, Base
/// Shift = srl Ty1 Origin, CstTy Amount
/// Inst = trunc Shift to Ty2
///
/// Then, it will be rewriten into:
/// Slice = load SliceTy, Base + SliceOffset
/// [Inst = zext Slice to Ty2], only if SliceTy <> Ty2
///
/// SliceTy is deduced from the number of bits that are actually used to
/// build Inst.
struct LoadedSlice {
  /// \brief Helper structure used to compute the cost of a slice.
  struct Cost {
    /// Are we optimizing for code size.
    bool ForCodeSize;
    /// Various cost.
    unsigned Loads;
    unsigned Truncates;
    unsigned CrossRegisterBanksCopies;
    unsigned ZExts;
    unsigned Shift;

    Cost(bool ForCodeSize = false)
        : ForCodeSize(ForCodeSize), Loads(0), Truncates(0),
          CrossRegisterBanksCopies(0), ZExts(0), Shift(0) {}

    /// \brief Get the cost of one isolated slice.
    Cost(const LoadedSlice &LS, bool ForCodeSize = false)
        : ForCodeSize(ForCodeSize), Loads(1), Truncates(0),
          CrossRegisterBanksCopies(0), ZExts(0), Shift(0) {
      EVT TruncType = LS.Inst->getValueType(0);
      EVT LoadedType = LS.getLoadedType();
      if (TruncType != LoadedType &&
          !LS.DAG->getTargetLoweringInfo().isZExtFree(LoadedType, TruncType))
        ZExts = 1;
    }

    /// \brief Account for slicing gain in the current cost.
    /// Slicing provide a few gains like removing a shift or a
    /// truncate. This method allows to grow the cost of the original
    /// load with the gain from this slice.
    void addSliceGain(const LoadedSlice &LS) {
      // Each slice saves a truncate.
      const TargetLowering &TLI = LS.DAG->getTargetLoweringInfo();
      if (!TLI.isTruncateFree(LS.Inst->getValueType(0),
                              LS.Inst->getOperand(0).getValueType()))
        ++Truncates;
      // If there is a shift amount, this slice gets rid of it.
      if (LS.Shift)
        ++Shift;
      // If this slice can merge a cross register bank copy, account for it.
      if (LS.canMergeExpensiveCrossRegisterBankCopy())
        ++CrossRegisterBanksCopies;
    }

    Cost &operator+=(const Cost &RHS) {
      Loads += RHS.Loads;
      Truncates += RHS.Truncates;
      CrossRegisterBanksCopies += RHS.CrossRegisterBanksCopies;
      ZExts += RHS.ZExts;
      Shift += RHS.Shift;
      return *this;
    }

    bool operator==(const Cost &RHS) const {
      return Loads == RHS.Loads && Truncates == RHS.Truncates &&
             CrossRegisterBanksCopies == RHS.CrossRegisterBanksCopies &&
             ZExts == RHS.ZExts && Shift == RHS.Shift;
    }

    bool operator!=(const Cost &RHS) const { return !(*this == RHS); }

    bool operator<(const Cost &RHS) const {
      // Assume cross register banks copies are as expensive as loads.
      // FIXME: Do we want some more target hooks?
      unsigned ExpensiveOpsLHS = Loads + CrossRegisterBanksCopies;
      unsigned ExpensiveOpsRHS = RHS.Loads + RHS.CrossRegisterBanksCopies;
      // Unless we are optimizing for code size, consider the
      // expensive operation first.
      if (!ForCodeSize && ExpensiveOpsLHS != ExpensiveOpsRHS)
        return ExpensiveOpsLHS < ExpensiveOpsRHS;
      return (Truncates + ZExts + Shift + ExpensiveOpsLHS) <
             (RHS.Truncates + RHS.ZExts + RHS.Shift + ExpensiveOpsRHS);
    }

    bool operator>(const Cost &RHS) const { return RHS < *this; }

    bool operator<=(const Cost &RHS) const { return !(RHS < *this); }

    bool operator>=(const Cost &RHS) const { return !(*this < RHS); }
  };
  // The last instruction that represent the slice. This should be a
  // truncate instruction.
  SDNode *Inst;
  // The original load instruction.
  LoadSDNode *Origin;
  // The right shift amount in bits from the original load.
  unsigned Shift;
  // The DAG from which Origin came from.
  // This is used to get some contextual information about legal types, etc.
  SelectionDAG *DAG;

  LoadedSlice(SDNode *Inst = NULL, LoadSDNode *Origin = NULL,
              unsigned Shift = 0, SelectionDAG *DAG = NULL)
      : Inst(Inst), Origin(Origin), Shift(Shift), DAG(DAG) {}

  LoadedSlice(const LoadedSlice &LS)
      : Inst(LS.Inst), Origin(LS.Origin), Shift(LS.Shift), DAG(LS.DAG) {}

  /// \brief Get the bits used in a chunk of bits \p BitWidth large.
  /// \return Result is \p BitWidth and has used bits set to 1 and
  ///         not used bits set to 0.
  APInt getUsedBits() const {
    // Reproduce the trunc(lshr) sequence:
    // - Start from the truncated value.
    // - Zero extend to the desired bit width.
    // - Shift left.
    assert(Origin && "No original load to compare against.");
    unsigned BitWidth = Origin->getValueSizeInBits(0);
    assert(Inst && "This slice is not bound to an instruction");
    assert(Inst->getValueSizeInBits(0) <= BitWidth &&
           "Extracted slice is bigger than the whole type!");
    APInt UsedBits(Inst->getValueSizeInBits(0), 0);
    UsedBits.setAllBits();
    UsedBits = UsedBits.zext(BitWidth);
    UsedBits <<= Shift;
    return UsedBits;
  }

  /// \brief Get the size of the slice to be loaded in bytes.
  unsigned getLoadedSize() const {
    unsigned SliceSize = getUsedBits().countPopulation();
    assert(!(SliceSize & 0x7) && "Size is not a multiple of a byte.");
    return SliceSize / 8;
  }

  /// \brief Get the type that will be loaded for this slice.
  /// Note: This may not be the final type for the slice.
  EVT getLoadedType() const {
    assert(DAG && "Missing context");
    LLVMContext &Ctxt = *DAG->getContext();
    return EVT::getIntegerVT(Ctxt, getLoadedSize() * 8);
  }

  /// \brief Get the alignment of the load used for this slice.
  unsigned getAlignment() const {
    unsigned Alignment = Origin->getAlignment();
    unsigned Offset = getOffsetFromBase();
    if (Offset != 0)
      Alignment = MinAlign(Alignment, Alignment + Offset);
    return Alignment;
  }

  /// \brief Check if this slice can be rewritten with legal operations.
  bool isLegal() const {
    // An invalid slice is not legal.
    if (!Origin || !Inst || !DAG)
      return false;

    // Offsets are for indexed load only, we do not handle that.
    if (Origin->getOffset().getOpcode() != ISD::UNDEF)
      return false;

    const TargetLowering &TLI = DAG->getTargetLoweringInfo();

    // Check that the type is legal.
    EVT SliceType = getLoadedType();
    if (!TLI.isTypeLegal(SliceType))
      return false;

    // Check that the load is legal for this type.
    if (!TLI.isOperationLegal(ISD::LOAD, SliceType))
      return false;

    // Check that the offset can be computed.
    // 1. Check its type.
    EVT PtrType = Origin->getBasePtr().getValueType();
    if (PtrType == MVT::Untyped || PtrType.isExtended())
      return false;

    // 2. Check that it fits in the immediate.
    if (!TLI.isLegalAddImmediate(getOffsetFromBase()))
      return false;

    // 3. Check that the computation is legal.
    if (!TLI.isOperationLegal(ISD::ADD, PtrType))
      return false;

    // Check that the zext is legal if it needs one.
    EVT TruncateType = Inst->getValueType(0);
    if (TruncateType != SliceType &&
        !TLI.isOperationLegal(ISD::ZERO_EXTEND, TruncateType))
      return false;

    return true;
  }

  /// \brief Get the offset in bytes of this slice in the original chunk of
  /// bits.
  /// \pre DAG != NULL.
  uint64_t getOffsetFromBase() const {
    assert(DAG && "Missing context.");
    bool IsBigEndian =
        DAG->getTargetLoweringInfo().getDataLayout()->isBigEndian();
    assert(!(Shift & 0x7) && "Shifts not aligned on Bytes are not supported.");
    uint64_t Offset = Shift / 8;
    unsigned TySizeInBytes = Origin->getValueSizeInBits(0) / 8;
    assert(!(Origin->getValueSizeInBits(0) & 0x7) &&
           "The size of the original loaded type is not a multiple of a"
           " byte.");
    // If Offset is bigger than TySizeInBytes, it means we are loading all
    // zeros. This should have been optimized before in the process.
    assert(TySizeInBytes > Offset &&
           "Invalid shift amount for given loaded size");
    if (IsBigEndian)
      Offset = TySizeInBytes - Offset - getLoadedSize();
    return Offset;
  }

  /// \brief Generate the sequence of instructions to load the slice
  /// represented by this object and redirect the uses of this slice to
  /// this new sequence of instructions.
  /// \pre this->Inst && this->Origin are valid Instructions and this
  /// object passed the legal check: LoadedSlice::isLegal returned true.
  /// \return The last instruction of the sequence used to load the slice.
  SDValue loadSlice() const {
    assert(Inst && Origin && "Unable to replace a non-existing slice.");
    const SDValue &OldBaseAddr = Origin->getBasePtr();
    SDValue BaseAddr = OldBaseAddr;
    // Get the offset in that chunk of bytes w.r.t. the endianess.
    int64_t Offset = static_cast<int64_t>(getOffsetFromBase());
    assert(Offset >= 0 && "Offset too big to fit in int64_t!");
    if (Offset) {
      // BaseAddr = BaseAddr + Offset.
      EVT ArithType = BaseAddr.getValueType();
      BaseAddr = DAG->getNode(ISD::ADD, SDLoc(Origin), ArithType, BaseAddr,
                              DAG->getConstant(Offset, ArithType));
    }

    // Create the type of the loaded slice according to its size.
    EVT SliceType = getLoadedType();

    // Create the load for the slice.
    SDValue LastInst = DAG->getLoad(
        SliceType, SDLoc(Origin), Origin->getChain(), BaseAddr,
        Origin->getPointerInfo().getWithOffset(Offset), Origin->isVolatile(),
        Origin->isNonTemporal(), Origin->isInvariant(), getAlignment());
    // If the final type is not the same as the loaded type, this means that
    // we have to pad with zero. Create a zero extend for that.
    EVT FinalType = Inst->getValueType(0);
    if (SliceType != FinalType)
      LastInst =
          DAG->getNode(ISD::ZERO_EXTEND, SDLoc(LastInst), FinalType, LastInst);
    return LastInst;
  }

  /// \brief Check if this slice can be merged with an expensive cross register
  /// bank copy. E.g.,
  /// i = load i32
  /// f = bitcast i32 i to float
  bool canMergeExpensiveCrossRegisterBankCopy() const {
    if (!Inst || !Inst->hasOneUse())
      return false;
    SDNode *Use = *Inst->use_begin();
    if (Use->getOpcode() != ISD::BITCAST)
      return false;
    assert(DAG && "Missing context");
    const TargetLowering &TLI = DAG->getTargetLoweringInfo();
    EVT ResVT = Use->getValueType(0);
    const TargetRegisterClass *ResRC = TLI.getRegClassFor(ResVT.getSimpleVT());
    const TargetRegisterClass *ArgRC =
        TLI.getRegClassFor(Use->getOperand(0).getValueType().getSimpleVT());
    if (ArgRC == ResRC || !TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // At this point, we know that we perform a cross-register-bank copy.
    // Check if it is expensive.
    const TargetRegisterInfo *TRI = TLI.getTargetMachine().getRegisterInfo();
    // Assume bitcasts are cheap, unless both register classes do not
    // explicitly share a common sub class.
    if (!TRI || TRI->getCommonSubClass(ArgRC, ResRC))
      return false;

    // Check if it will be merged with the load.
    // 1. Check the alignment constraint.
    unsigned RequiredAlignment = TLI.getDataLayout()->getABITypeAlignment(
        ResVT.getTypeForEVT(*DAG->getContext()));

    if (RequiredAlignment > getAlignment())
      return false;

    // 2. Check that the load is a legal operation for that type.
    if (!TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // 3. Check that we do not have a zext in the way.
    if (Inst->getValueType(0) != getLoadedType())
      return false;

    return true;
  }
};
}

/// \brief Sorts LoadedSlice according to their offset.
struct LoadedSliceSorter {
  bool operator()(const LoadedSlice &LHS, const LoadedSlice &RHS) {
    assert(LHS.Origin == RHS.Origin && "Different bases not implemented.");
    return LHS.getOffsetFromBase() < RHS.getOffsetFromBase();
  }
};

/// \brief Check that all bits set in \p UsedBits form a dense region, i.e.,
/// \p UsedBits looks like 0..0 1..1 0..0.
static bool areUsedBitsDense(const APInt &UsedBits) {
  // If all the bits are one, this is dense!
  if (UsedBits.isAllOnesValue())
    return true;

  // Get rid of the unused bits on the right.
  APInt NarrowedUsedBits = UsedBits.lshr(UsedBits.countTrailingZeros());
  // Get rid of the unused bits on the left.
  if (NarrowedUsedBits.countLeadingZeros())
    NarrowedUsedBits = NarrowedUsedBits.trunc(NarrowedUsedBits.getActiveBits());
  // Check that the chunk of bits is completely used.
  return NarrowedUsedBits.isAllOnesValue();
}

/// \brief Check whether or not \p First and \p Second are next to each other
/// in memory. This means that there is no hole between the bits loaded
/// by \p First and the bits loaded by \p Second.
static bool areSlicesNextToEachOther(const LoadedSlice &First,
                                     const LoadedSlice &Second) {
  assert(First.Origin == Second.Origin && First.Origin &&
         "Unable to match different memory origins.");
  APInt UsedBits = First.getUsedBits();
  assert((UsedBits & Second.getUsedBits()) == 0 &&
         "Slices are not supposed to overlap.");
  UsedBits |= Second.getUsedBits();
  return areUsedBitsDense(UsedBits);
}

/// \brief Adjust the \p GlobalLSCost according to the target
/// paring capabilities and the layout of the slices.
/// \pre \p GlobalLSCost should account for at least as many loads as
/// there is in the slices in \p LoadedSlices.
static void adjustCostForPairing(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                 LoadedSlice::Cost &GlobalLSCost) {
  unsigned NumberOfSlices = LoadedSlices.size();
  // If there is less than 2 elements, no pairing is possible.
  if (NumberOfSlices < 2)
    return;

  // Sort the slices so that elements that are likely to be next to each
  // other in memory are next to each other in the list.
  std::sort(LoadedSlices.begin(), LoadedSlices.end(), LoadedSliceSorter());
  const TargetLowering &TLI = LoadedSlices[0].DAG->getTargetLoweringInfo();
  // First (resp. Second) is the first (resp. Second) potentially candidate
  // to be placed in a paired load.
  const LoadedSlice *First = NULL;
  const LoadedSlice *Second = NULL;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice,
                // Set the beginning of the pair.
                                                           First = Second) {

    Second = &LoadedSlices[CurrSlice];

    // If First is NULL, it means we start a new pair.
    // Get to the next slice.
    if (!First)
      continue;

    EVT LoadedType = First->getLoadedType();

    // If the types of the slices are different, we cannot pair them.
    if (LoadedType != Second->getLoadedType())
      continue;

    // Check if the target supplies paired loads for this type.
    unsigned RequiredAlignment = 0;
    if (!TLI.hasPairedLoad(LoadedType, RequiredAlignment)) {
      // move to the next pair, this type is hopeless.
      Second = NULL;
      continue;
    }
    // Check if we meet the alignment requirement.
    if (RequiredAlignment > First->getAlignment())
      continue;

    // Check that both loads are next to each other in memory.
    if (!areSlicesNextToEachOther(*First, *Second))
      continue;

    assert(GlobalLSCost.Loads > 0 && "We save more loads than we created!");
    --GlobalLSCost.Loads;
    // Move to the next pair.
    Second = NULL;
  }
}

/// \brief Check the profitability of all involved LoadedSlice.
/// Currently, it is considered profitable if there is exactly two
/// involved slices (1) which are (2) next to each other in memory, and
/// whose cost (\see LoadedSlice::Cost) is smaller than the original load (3).
///
/// Note: The order of the elements in \p LoadedSlices may be modified, but not
/// the elements themselves.
///
/// FIXME: When the cost model will be mature enough, we can relax
/// constraints (1) and (2).
static bool isSlicingProfitable(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                const APInt &UsedBits, bool ForCodeSize) {
  unsigned NumberOfSlices = LoadedSlices.size();
  if (StressLoadSlicing)
    return NumberOfSlices > 1;

  // Check (1).
  if (NumberOfSlices != 2)
    return false;

  // Check (2).
  if (!areUsedBitsDense(UsedBits))
    return false;

  // Check (3).
  LoadedSlice::Cost OrigCost(ForCodeSize), GlobalSlicingCost(ForCodeSize);
  // The original code has one big load.
  OrigCost.Loads = 1;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice) {
    const LoadedSlice &LS = LoadedSlices[CurrSlice];
    // Accumulate the cost of all the slices.
    LoadedSlice::Cost SliceCost(LS, ForCodeSize);
    GlobalSlicingCost += SliceCost;

    // Account as cost in the original configuration the gain obtained
    // with the current slices.
    OrigCost.addSliceGain(LS);
  }

  // If the target supports paired load, adjust the cost accordingly.
  adjustCostForPairing(LoadedSlices, GlobalSlicingCost);
  return OrigCost > GlobalSlicingCost;
}

/// \brief If the given load, \p LI, is used only by trunc or trunc(lshr)
/// operations, split it in the various pieces being extracted.
///
/// This sort of thing is introduced by SROA.
/// This slicing takes care not to insert overlapping loads.
/// \pre LI is a simple load (i.e., not an atomic or volatile load).
bool DAGCombiner::SliceUpLoad(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (LD->isVolatile() || !ISD::isNormalLoad(LD) ||
      !LD->getValueType(0).isInteger())
    return false;

  // Keep track of already used bits to detect overlapping values.
  // In that case, we will just abort the transformation.
  APInt UsedBits(LD->getValueSizeInBits(0), 0);

  SmallVector<LoadedSlice, 4> LoadedSlices;

  // Check if this load is used as several smaller chunks of bits.
  // Basically, look for uses in trunc or trunc(lshr) and record a new chain
  // of computation for each trunc.
  for (SDNode::use_iterator UI = LD->use_begin(), UIEnd = LD->use_end();
       UI != UIEnd; ++UI) {
    // Skip the uses of the chain.
    if (UI.getUse().getResNo() != 0)
      continue;

    SDNode *User = *UI;
    unsigned Shift = 0;

    // Check if this is a trunc(lshr).
    if (User->getOpcode() == ISD::SRL && User->hasOneUse() &&
        isa<ConstantSDNode>(User->getOperand(1))) {
      Shift = cast<ConstantSDNode>(User->getOperand(1))->getZExtValue();
      User = *User->use_begin();
    }

    // At this point, User is a Truncate, iff we encountered, trunc or
    // trunc(lshr).
    if (User->getOpcode() != ISD::TRUNCATE)
      return false;

    // The width of the type must be a power of 2 and greater than 8-bits.
    // Otherwise the load cannot be represented in LLVM IR.
    // Moreover, if we shifted with a non 8-bits multiple, the slice
    // will be accross several bytes. We do not support that.
    unsigned Width = User->getValueSizeInBits(0);
    if (Width < 8 || !isPowerOf2_32(Width) || (Shift & 0x7))
      return 0;

    // Build the slice for this chain of computations.
    LoadedSlice LS(User, LD, Shift, &DAG);
    APInt CurrentUsedBits = LS.getUsedBits();

    // Check if this slice overlaps with another.
    if ((CurrentUsedBits & UsedBits) != 0)
      return false;
    // Update the bits used globally.
    UsedBits |= CurrentUsedBits;

    // Check if the new slice would be legal.
    if (!LS.isLegal())
      return false;

    // Record the slice.
    LoadedSlices.push_back(LS);
  }

  // Abort slicing if it does not seem to be profitable.
  if (!isSlicingProfitable(LoadedSlices, UsedBits, ForCodeSize))
    return false;

  ++SlicedLoads;

  // Rewrite each chain to use an independent load.
  // By construction, each chain can be represented by a unique load.

  // Prepare the argument for the new token factor for all the slices.
  SmallVector<SDValue, 8> ArgChains;
  for (SmallVectorImpl<LoadedSlice>::const_iterator
           LSIt = LoadedSlices.begin(),
           LSItEnd = LoadedSlices.end();
       LSIt != LSItEnd; ++LSIt) {
    SDValue SliceInst = LSIt->loadSlice();
    CombineTo(LSIt->Inst, SliceInst, true);
    if (SliceInst.getNode()->getOpcode() != ISD::LOAD)
      SliceInst = SliceInst.getOperand(0);
    assert(SliceInst->getOpcode() == ISD::LOAD &&
           "It takes more than a zext to get to the loaded slice!!");
    ArgChains.push_back(SliceInst.getValue(1));
  }

  SDValue Chain = DAG.getNode(ISD::TokenFactor, SDLoc(LD), MVT::Other,
                              &ArgChains[0], ArgChains.size());
  DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);
  return true;
}

/// CheckForMaskedLoad - Check to see if V is (and load (ptr), imm), where the
/// load is having specific bytes cleared out.  If so, return the byte size
/// being masked out and the shift amount.
static std::pair<unsigned, unsigned>
CheckForMaskedLoad(SDValue V, SDValue Ptr, SDValue Chain) {
  std::pair<unsigned, unsigned> Result(0, 0);

  // Check for the structure we're looking for.
  if (V->getOpcode() != ISD::AND ||
      !isa<ConstantSDNode>(V->getOperand(1)) ||
      !ISD::isNormalLoad(V->getOperand(0).getNode()))
    return Result;

  // Check the chain and pointer.
  LoadSDNode *LD = cast<LoadSDNode>(V->getOperand(0));
  if (LD->getBasePtr() != Ptr) return Result;  // Not from same pointer.

  // The store should be chained directly to the load or be an operand of a
  // tokenfactor.
  if (LD == Chain.getNode())
    ; // ok.
  else if (Chain->getOpcode() != ISD::TokenFactor)
    return Result; // Fail.
  else {
    bool isOk = false;
    for (unsigned i = 0, e = Chain->getNumOperands(); i != e; ++i)
      if (Chain->getOperand(i).getNode() == LD) {
        isOk = true;
        break;
      }
    if (!isOk) return Result;
  }

  // This only handles simple types.
  if (V.getValueType() != MVT::i16 &&
      V.getValueType() != MVT::i32 &&
      V.getValueType() != MVT::i64)
    return Result;

  // Check the constant mask.  Invert it so that the bits being masked out are
  // 0 and the bits being kept are 1.  Use getSExtValue so that leading bits
  // follow the sign bit for uniformity.
  uint64_t NotMask = ~cast<ConstantSDNode>(V->getOperand(1))->getSExtValue();
  unsigned NotMaskLZ = countLeadingZeros(NotMask);
  if (NotMaskLZ & 7) return Result;  // Must be multiple of a byte.
  unsigned NotMaskTZ = countTrailingZeros(NotMask);
  if (NotMaskTZ & 7) return Result;  // Must be multiple of a byte.
  if (NotMaskLZ == 64) return Result;  // All zero mask.

  // See if we have a continuous run of bits.  If so, we have 0*1+0*
  if (CountTrailingOnes_64(NotMask >> NotMaskTZ)+NotMaskTZ+NotMaskLZ != 64)
    return Result;

  // Adjust NotMaskLZ down to be from the actual size of the int instead of i64.
  if (V.getValueType() != MVT::i64 && NotMaskLZ)
    NotMaskLZ -= 64-V.getValueSizeInBits();

  unsigned MaskedBytes = (V.getValueSizeInBits()-NotMaskLZ-NotMaskTZ)/8;
  switch (MaskedBytes) {
  case 1:
  case 2:
  case 4: break;
  default: return Result; // All one mask, or 5-byte mask.
  }

  // Verify that the first bit starts at a multiple of mask so that the access
  // is aligned the same as the access width.
  if (NotMaskTZ && NotMaskTZ/8 % MaskedBytes) return Result;

  Result.first = MaskedBytes;
  Result.second = NotMaskTZ/8;
  return Result;
}


/// ShrinkLoadReplaceStoreWithStore - Check to see if IVal is something that
/// provides a value as specified by MaskInfo.  If so, replace the specified
/// store with a narrower store of truncated IVal.
static SDNode *
ShrinkLoadReplaceStoreWithStore(const std::pair<unsigned, unsigned> &MaskInfo,
                                SDValue IVal, StoreSDNode *St,
                                DAGCombiner *DC) {
  unsigned NumBytes = MaskInfo.first;
  unsigned ByteShift = MaskInfo.second;
  SelectionDAG &DAG = DC->getDAG();

  // Check to see if IVal is all zeros in the part being masked in by the 'or'
  // that uses this.  If not, this is not a replacement.
  APInt Mask = ~APInt::getBitsSet(IVal.getValueSizeInBits(),
                                  ByteShift*8, (ByteShift+NumBytes)*8);
  if (!DAG.MaskedValueIsZero(IVal, Mask)) return 0;

  // Check that it is legal on the target to do this.  It is legal if the new
  // VT we're shrinking to (i8/i16/i32) is legal or we're still before type
  // legalization.
  MVT VT = MVT::getIntegerVT(NumBytes*8);
  if (!DC->isTypeLegal(VT))
    return 0;

  // Okay, we can do this!  Replace the 'St' store with a store of IVal that is
  // shifted by ByteShift and truncated down to NumBytes.
  if (ByteShift)
    IVal = DAG.getNode(ISD::SRL, SDLoc(IVal), IVal.getValueType(), IVal,
                       DAG.getConstant(ByteShift*8,
                                    DC->getShiftAmountTy(IVal.getValueType())));

  // Figure out the offset for the store and the alignment of the access.
  unsigned StOffset;
  unsigned NewAlign = St->getAlignment();

  if (DAG.getTargetLoweringInfo().isLittleEndian())
    StOffset = ByteShift;
  else
    StOffset = IVal.getValueType().getStoreSize() - ByteShift - NumBytes;

  SDValue Ptr = St->getBasePtr();
  if (StOffset) {
    Ptr = DAG.getNode(ISD::ADD, SDLoc(IVal), Ptr.getValueType(),
                      Ptr, DAG.getConstant(StOffset, Ptr.getValueType()));
    NewAlign = MinAlign(NewAlign, StOffset);
  }

  // Truncate down to the new size.
  IVal = DAG.getNode(ISD::TRUNCATE, SDLoc(IVal), VT, IVal);

  ++OpsNarrowed;
  return DAG.getStore(St->getChain(), SDLoc(St), IVal, Ptr,
                      St->getPointerInfo().getWithOffset(StOffset),
                      false, false, NewAlign).getNode();
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

  // If this is "store (or X, Y), P" and X is "(and (load P), cst)", where cst
  // is a byte mask indicating a consecutive number of bytes, check to see if
  // Y is known to provide just those bytes.  If so, we try to replace the
  // load + replace + store sequence with a single (narrower) store, which makes
  // the load dead.
  if (Opc == ISD::OR) {
    std::pair<unsigned, unsigned> MaskedLoad;
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(0), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDNode *NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(1), ST,this))
        return SDValue(NewST, 0);

    // Or is commutative, so try swapping X and Y.
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(1), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDNode *NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(0), ST,this))
        return SDValue(NewST, 0);
  }

  if ((Opc != ISD::OR && Opc != ISD::XOR && Opc != ISD::AND) ||
      Value.getOperand(1).getOpcode() != ISD::Constant)
    return SDValue();

  SDValue N0 = Value.getOperand(0);
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      Chain == SDValue(N0.getNode(), 1)) {
    LoadSDNode *LD = cast<LoadSDNode>(N0);
    if (LD->getBasePtr() != Ptr ||
        LD->getPointerInfo().getAddrSpace() !=
        ST->getPointerInfo().getAddrSpace())
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
    APInt Mask = APInt::getBitsSet(BitWidth, ShAmt,
                                   std::min(BitWidth, ShAmt + NewBW));
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
      Type *NewVTTy = NewVT.getTypeForEVT(*DAG.getContext());
      if (NewAlign < TLI.getDataLayout()->getABITypeAlignment(NewVTTy))
        return SDValue();

      SDValue NewPtr = DAG.getNode(ISD::ADD, SDLoc(LD),
                                   Ptr.getValueType(), Ptr,
                                   DAG.getConstant(PtrOff, Ptr.getValueType()));
      SDValue NewLD = DAG.getLoad(NewVT, SDLoc(N0),
                                  LD->getChain(), NewPtr,
                                  LD->getPointerInfo().getWithOffset(PtrOff),
                                  LD->isVolatile(), LD->isNonTemporal(),
                                  LD->isInvariant(), NewAlign,
                                  LD->getTBAAInfo());
      SDValue NewVal = DAG.getNode(Opc, SDLoc(Value), NewVT, NewLD,
                                   DAG.getConstant(NewImm, NewVT));
      SDValue NewST = DAG.getStore(Chain, SDLoc(N),
                                   NewVal, NewPtr,
                                   ST->getPointerInfo().getWithOffset(PtrOff),
                                   false, false, NewAlign);

      AddToWorkList(NewPtr.getNode());
      AddToWorkList(NewLD.getNode());
      AddToWorkList(NewVal.getNode());
      WorkListRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLD.getValue(1));
      ++OpsNarrowed;
      return NewST;
    }
  }

  return SDValue();
}

/// TransformFPLoadStorePair - For a given floating point load / store pair,
/// if the load value isn't used by any other operations, then consider
/// transforming the pair to integer load / store operations if the target
/// deems the transformation profitable.
SDValue DAGCombiner::TransformFPLoadStorePair(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  if (ISD::isNormalStore(ST) && ISD::isNormalLoad(Value.getNode()) &&
      Value.hasOneUse() &&
      Chain == SDValue(Value.getNode(), 1)) {
    LoadSDNode *LD = cast<LoadSDNode>(Value);
    EVT VT = LD->getMemoryVT();
    if (!VT.isFloatingPoint() ||
        VT != ST->getMemoryVT() ||
        LD->isNonTemporal() ||
        ST->isNonTemporal() ||
        LD->getPointerInfo().getAddrSpace() != 0 ||
        ST->getPointerInfo().getAddrSpace() != 0)
      return SDValue();

    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), VT.getSizeInBits());
    if (!TLI.isOperationLegal(ISD::LOAD, IntVT) ||
        !TLI.isOperationLegal(ISD::STORE, IntVT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::LOAD, VT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::STORE, VT))
      return SDValue();

    unsigned LDAlign = LD->getAlignment();
    unsigned STAlign = ST->getAlignment();
    Type *IntVTTy = IntVT.getTypeForEVT(*DAG.getContext());
    unsigned ABIAlign = TLI.getDataLayout()->getABITypeAlignment(IntVTTy);
    if (LDAlign < ABIAlign || STAlign < ABIAlign)
      return SDValue();

    SDValue NewLD = DAG.getLoad(IntVT, SDLoc(Value),
                                LD->getChain(), LD->getBasePtr(),
                                LD->getPointerInfo(),
                                false, false, false, LDAlign);

    SDValue NewST = DAG.getStore(NewLD.getValue(1), SDLoc(N),
                                 NewLD, ST->getBasePtr(),
                                 ST->getPointerInfo(),
                                 false, false, STAlign);

    AddToWorkList(NewLD.getNode());
    AddToWorkList(NewST.getNode());
    WorkListRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(Value.getValue(1), NewLD.getValue(1));
    ++LdStFP2Int;
    return NewST;
  }

  return SDValue();
}

/// Helper struct to parse and store a memory address as base + index + offset.
/// We ignore sign extensions when it is safe to do so.
/// The following two expressions are not equivalent. To differentiate we need
/// to store whether there was a sign extension involved in the index
/// computation.
///  (load (i64 add (i64 copyfromreg %c)
///                 (i64 signextend (add (i8 load %index)
///                                      (i8 1))))
/// vs
///
/// (load (i64 add (i64 copyfromreg %c)
///                (i64 signextend (i32 add (i32 signextend (i8 load %index))
///                                         (i32 1)))))
struct BaseIndexOffset {
  SDValue Base;
  SDValue Index;
  int64_t Offset;
  bool IsIndexSignExt;

  BaseIndexOffset() : Offset(0), IsIndexSignExt(false) {}

  BaseIndexOffset(SDValue Base, SDValue Index, int64_t Offset,
                  bool IsIndexSignExt) :
    Base(Base), Index(Index), Offset(Offset), IsIndexSignExt(IsIndexSignExt) {}

  bool equalBaseIndex(const BaseIndexOffset &Other) {
    return Other.Base == Base && Other.Index == Index &&
      Other.IsIndexSignExt == IsIndexSignExt;
  }

  /// Parses tree in Ptr for base, index, offset addresses.
  static BaseIndexOffset match(SDValue Ptr) {
    bool IsIndexSignExt = false;

    // We only can pattern match BASE + INDEX + OFFSET. If Ptr is not an ADD
    // instruction, then it could be just the BASE or everything else we don't
    // know how to handle. Just use Ptr as BASE and give up.
    if (Ptr->getOpcode() != ISD::ADD)
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // We know that we have at least an ADD instruction. Try to pattern match
    // the simple case of BASE + OFFSET.
    if (isa<ConstantSDNode>(Ptr->getOperand(1))) {
      int64_t Offset = cast<ConstantSDNode>(Ptr->getOperand(1))->getSExtValue();
      return  BaseIndexOffset(Ptr->getOperand(0), SDValue(), Offset,
                              IsIndexSignExt);
    }

    // Inside a loop the current BASE pointer is calculated using an ADD and a
    // MUL instruction. In this case Ptr is the actual BASE pointer.
    // (i64 add (i64 %array_ptr)
    //          (i64 mul (i64 %induction_var)
    //                   (i64 %element_size)))
    if (Ptr->getOperand(1)->getOpcode() == ISD::MUL)
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // Look at Base + Index + Offset cases.
    SDValue Base = Ptr->getOperand(0);
    SDValue IndexOffset = Ptr->getOperand(1);

    // Skip signextends.
    if (IndexOffset->getOpcode() == ISD::SIGN_EXTEND) {
      IndexOffset = IndexOffset->getOperand(0);
      IsIndexSignExt = true;
    }

    // Either the case of Base + Index (no offset) or something else.
    if (IndexOffset->getOpcode() != ISD::ADD)
      return BaseIndexOffset(Base, IndexOffset, 0, IsIndexSignExt);

    // Now we have the case of Base + Index + offset.
    SDValue Index = IndexOffset->getOperand(0);
    SDValue Offset = IndexOffset->getOperand(1);

    if (!isa<ConstantSDNode>(Offset))
      return BaseIndexOffset(Ptr, SDValue(), 0, IsIndexSignExt);

    // Ignore signextends.
    if (Index->getOpcode() == ISD::SIGN_EXTEND) {
      Index = Index->getOperand(0);
      IsIndexSignExt = true;
    } else IsIndexSignExt = false;

    int64_t Off = cast<ConstantSDNode>(Offset)->getSExtValue();
    return BaseIndexOffset(Base, Index, Off, IsIndexSignExt);
  }
};

/// Holds a pointer to an LSBaseSDNode as well as information on where it
/// is located in a sequence of memory operations connected by a chain.
struct MemOpLink {
  MemOpLink (LSBaseSDNode *N, int64_t Offset, unsigned Seq):
    MemNode(N), OffsetFromBase(Offset), SequenceNum(Seq) { }
  // Ptr to the mem node.
  LSBaseSDNode *MemNode;
  // Offset from the base ptr.
  int64_t OffsetFromBase;
  // What is the sequence number of this mem node.
  // Lowest mem operand in the DAG starts at zero.
  unsigned SequenceNum;
};

/// Sorts store nodes in a link according to their offset from a shared
// base ptr.
struct ConsecutiveMemoryChainSorter {
  bool operator()(MemOpLink LHS, MemOpLink RHS) {
    return LHS.OffsetFromBase < RHS.OffsetFromBase;
  }
};

bool DAGCombiner::MergeConsecutiveStores(StoreSDNode* St) {
  EVT MemVT = St->getMemoryVT();
  int64_t ElementSizeBytes = MemVT.getSizeInBits()/8;
  bool NoVectors = DAG.getMachineFunction().getFunction()->getAttributes().
    hasAttribute(AttributeSet::FunctionIndex, Attribute::NoImplicitFloat);

  // Don't merge vectors into wider inputs.
  if (MemVT.isVector() || !MemVT.isSimple())
    return false;

  // Perform an early exit check. Do not bother looking at stored values that
  // are not constants or loads.
  SDValue StoredVal = St->getValue();
  bool IsLoadSrc = isa<LoadSDNode>(StoredVal);
  if (!isa<ConstantSDNode>(StoredVal) && !isa<ConstantFPSDNode>(StoredVal) &&
      !IsLoadSrc)
    return false;

  // Only look at ends of store sequences.
  SDValue Chain = SDValue(St, 1);
  if (Chain->hasOneUse() && Chain->use_begin()->getOpcode() == ISD::STORE)
    return false;

  // This holds the base pointer, index, and the offset in bytes from the base
  // pointer.
  BaseIndexOffset BasePtr = BaseIndexOffset::match(St->getBasePtr());

  // We must have a base and an offset.
  if (!BasePtr.Base.getNode())
    return false;

  // Do not handle stores to undef base pointers.
  if (BasePtr.Base.getOpcode() == ISD::UNDEF)
    return false;

  // Save the LoadSDNodes that we find in the chain.
  // We need to make sure that these nodes do not interfere with
  // any of the store nodes.
  SmallVector<LSBaseSDNode*, 8> AliasLoadNodes;

  // Save the StoreSDNodes that we find in the chain.
  SmallVector<MemOpLink, 8> StoreNodes;

  // Walk up the chain and look for nodes with offsets from the same
  // base pointer. Stop when reaching an instruction with a different kind
  // or instruction which has a different base pointer.
  unsigned Seq = 0;
  StoreSDNode *Index = St;
  while (Index) {
    // If the chain has more than one use, then we can't reorder the mem ops.
    if (Index != St && !SDValue(Index, 1)->hasOneUse())
      break;

    // Find the base pointer and offset for this memory node.
    BaseIndexOffset Ptr = BaseIndexOffset::match(Index->getBasePtr());

    // Check that the base pointer is the same as the original one.
    if (!Ptr.equalBaseIndex(BasePtr))
      break;

    // Check that the alignment is the same.
    if (Index->getAlignment() != St->getAlignment())
      break;

    // The memory operands must not be volatile.
    if (Index->isVolatile() || Index->isIndexed())
      break;

    // No truncation.
    if (StoreSDNode *St = dyn_cast<StoreSDNode>(Index))
      if (St->isTruncatingStore())
        break;

    // The stored memory type must be the same.
    if (Index->getMemoryVT() != MemVT)
      break;

    // We do not allow unaligned stores because we want to prevent overriding
    // stores.
    if (Index->getAlignment()*8 != MemVT.getSizeInBits())
      break;

    // We found a potential memory operand to merge.
    StoreNodes.push_back(MemOpLink(Index, Ptr.Offset, Seq++));

    // Find the next memory operand in the chain. If the next operand in the
    // chain is a store then move up and continue the scan with the next
    // memory operand. If the next operand is a load save it and use alias
    // information to check if it interferes with anything.
    SDNode *NextInChain = Index->getChain().getNode();
    while (1) {
      if (StoreSDNode *STn = dyn_cast<StoreSDNode>(NextInChain)) {
        // We found a store node. Use it for the next iteration.
        Index = STn;
        break;
      } else if (LoadSDNode *Ldn = dyn_cast<LoadSDNode>(NextInChain)) {
        if (Ldn->isVolatile()) {
          Index = NULL;
          break;
        }

        // Save the load node for later. Continue the scan.
        AliasLoadNodes.push_back(Ldn);
        NextInChain = Ldn->getChain().getNode();
        continue;
      } else {
        Index = NULL;
        break;
      }
    }
  }

  // Check if there is anything to merge.
  if (StoreNodes.size() < 2)
    return false;

  // Sort the memory operands according to their distance from the base pointer.
  std::sort(StoreNodes.begin(), StoreNodes.end(),
            ConsecutiveMemoryChainSorter());

  // Scan the memory operations on the chain and find the first non-consecutive
  // store memory address.
  unsigned LastConsecutiveStore = 0;
  int64_t StartAddress = StoreNodes[0].OffsetFromBase;
  for (unsigned i = 0, e = StoreNodes.size(); i < e; ++i) {

    // Check that the addresses are consecutive starting from the second
    // element in the list of stores.
    if (i > 0) {
      int64_t CurrAddress = StoreNodes[i].OffsetFromBase;
      if (CurrAddress - StartAddress != (ElementSizeBytes * i))
        break;
    }

    bool Alias = false;
    // Check if this store interferes with any of the loads that we found.
    for (unsigned ld = 0, lde = AliasLoadNodes.size(); ld < lde; ++ld)
      if (isAlias(AliasLoadNodes[ld], StoreNodes[i].MemNode)) {
        Alias = true;
        break;
      }
    // We found a load that alias with this store. Stop the sequence.
    if (Alias)
      break;

    // Mark this node as useful.
    LastConsecutiveStore = i;
  }

  // The node with the lowest store address.
  LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;

  // Store the constants into memory as one consecutive store.
  if (!IsLoadSrc) {
    unsigned LastLegalType = 0;
    unsigned LastLegalVectorType = 0;
    bool NonZero = false;
    for (unsigned i=0; i<LastConsecutiveStore+1; ++i) {
      StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[i].MemNode);
      SDValue StoredVal = St->getValue();

      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(StoredVal)) {
        NonZero |= !C->isNullValue();
      } else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(StoredVal)) {
        NonZero |= !C->getConstantFPValue()->isNullValue();
      } else {
        // Non constant.
        break;
      }

      // Find a legal type for the constant store.
      unsigned StoreBW = (i+1) * ElementSizeBytes * 8;
      EVT StoreTy = EVT::getIntegerVT(*DAG.getContext(), StoreBW);
      if (TLI.isTypeLegal(StoreTy))
        LastLegalType = i+1;
      // Or check whether a truncstore is legal.
      else if (TLI.getTypeAction(*DAG.getContext(), StoreTy) ==
               TargetLowering::TypePromoteInteger) {
        EVT LegalizedStoredValueTy =
          TLI.getTypeToTransformTo(*DAG.getContext(), StoredVal.getValueType());
        if (TLI.isTruncStoreLegal(LegalizedStoredValueTy, StoreTy))
          LastLegalType = i+1;
      }

      // Find a legal type for the vector store.
      EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT, i+1);
      if (TLI.isTypeLegal(Ty))
        LastLegalVectorType = i + 1;
    }

    // We only use vectors if the constant is known to be zero and the
    // function is not marked with the noimplicitfloat attribute.
    if (NonZero || NoVectors)
      LastLegalVectorType = 0;

    // Check if we found a legal integer type to store.
    if (LastLegalType == 0 && LastLegalVectorType == 0)
      return false;

    bool UseVector = (LastLegalVectorType > LastLegalType) && !NoVectors;
    unsigned NumElem = UseVector ? LastLegalVectorType : LastLegalType;

    // Make sure we have something to merge.
    if (NumElem < 2)
      return false;

    unsigned EarliestNodeUsed = 0;
    for (unsigned i=0; i < NumElem; ++i) {
      // Find a chain for the new wide-store operand. Notice that some
      // of the store nodes that we found may not be selected for inclusion
      // in the wide store. The chain we use needs to be the chain of the
      // earliest store node which is *used* and replaced by the wide store.
      if (StoreNodes[i].SequenceNum > StoreNodes[EarliestNodeUsed].SequenceNum)
        EarliestNodeUsed = i;
    }

    // The earliest Node in the DAG.
    LSBaseSDNode *EarliestOp = StoreNodes[EarliestNodeUsed].MemNode;
    SDLoc DL(StoreNodes[0].MemNode);

    SDValue StoredVal;
    if (UseVector) {
      // Find a legal type for the vector store.
      EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT, NumElem);
      assert(TLI.isTypeLegal(Ty) && "Illegal vector store");
      StoredVal = DAG.getConstant(0, Ty);
    } else {
      unsigned StoreBW = NumElem * ElementSizeBytes * 8;
      APInt StoreInt(StoreBW, 0);

      // Construct a single integer constant which is made of the smaller
      // constant inputs.
      bool IsLE = TLI.isLittleEndian();
      for (unsigned i = 0; i < NumElem ; ++i) {
        unsigned Idx = IsLE ?(NumElem - 1 - i) : i;
        StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[Idx].MemNode);
        SDValue Val = St->getValue();
        StoreInt<<=ElementSizeBytes*8;
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val)) {
          StoreInt|=C->getAPIntValue().zext(StoreBW);
        } else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Val)) {
          StoreInt|= C->getValueAPF().bitcastToAPInt().zext(StoreBW);
        } else {
          assert(false && "Invalid constant element type");
        }
      }

      // Create the new Load and Store operations.
      EVT StoreTy = EVT::getIntegerVT(*DAG.getContext(), StoreBW);
      StoredVal = DAG.getConstant(StoreInt, StoreTy);
    }

    SDValue NewStore = DAG.getStore(EarliestOp->getChain(), DL, StoredVal,
                                    FirstInChain->getBasePtr(),
                                    FirstInChain->getPointerInfo(),
                                    false, false,
                                    FirstInChain->getAlignment());

    // Replace the first store with the new store
    CombineTo(EarliestOp, NewStore);
    // Erase all other stores.
    for (unsigned i = 0; i < NumElem ; ++i) {
      if (StoreNodes[i].MemNode == EarliestOp)
        continue;
      StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
      // ReplaceAllUsesWith will replace all uses that existed when it was
      // called, but graph optimizations may cause new ones to appear. For
      // example, the case in pr14333 looks like
      //
      //  St's chain -> St -> another store -> X
      //
      // And the only difference from St to the other store is the chain.
      // When we change it's chain to be St's chain they become identical,
      // get CSEed and the net result is that X is now a use of St.
      // Since we know that St is redundant, just iterate.
      while (!St->use_empty())
        DAG.ReplaceAllUsesWith(SDValue(St, 0), St->getChain());
      removeFromWorkList(St);
      DAG.DeleteNode(St);
    }

    return true;
  }

  // Below we handle the case of multiple consecutive stores that
  // come from multiple consecutive loads. We merge them into a single
  // wide load and a single wide store.

  // Look for load nodes which are used by the stored values.
  SmallVector<MemOpLink, 8> LoadNodes;

  // Find acceptable loads. Loads need to have the same chain (token factor),
  // must not be zext, volatile, indexed, and they must be consecutive.
  BaseIndexOffset LdBasePtr;
  for (unsigned i=0; i<LastConsecutiveStore+1; ++i) {
    StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[i].MemNode);
    LoadSDNode *Ld = dyn_cast<LoadSDNode>(St->getValue());
    if (!Ld) break;

    // Loads must only have one use.
    if (!Ld->hasNUsesOfValue(1, 0))
      break;

    // Check that the alignment is the same as the stores.
    if (Ld->getAlignment() != St->getAlignment())
      break;

    // The memory operands must not be volatile.
    if (Ld->isVolatile() || Ld->isIndexed())
      break;

    // We do not accept ext loads.
    if (Ld->getExtensionType() != ISD::NON_EXTLOAD)
      break;

    // The stored memory type must be the same.
    if (Ld->getMemoryVT() != MemVT)
      break;

    BaseIndexOffset LdPtr = BaseIndexOffset::match(Ld->getBasePtr());
    // If this is not the first ptr that we check.
    if (LdBasePtr.Base.getNode()) {
      // The base ptr must be the same.
      if (!LdPtr.equalBaseIndex(LdBasePtr))
        break;
    } else {
      // Check that all other base pointers are the same as this one.
      LdBasePtr = LdPtr;
    }

    // We found a potential memory operand to merge.
    LoadNodes.push_back(MemOpLink(Ld, LdPtr.Offset, 0));
  }

  if (LoadNodes.size() < 2)
    return false;

  // Scan the memory operations on the chain and find the first non-consecutive
  // load memory address. These variables hold the index in the store node
  // array.
  unsigned LastConsecutiveLoad = 0;
  // This variable refers to the size and not index in the array.
  unsigned LastLegalVectorType = 0;
  unsigned LastLegalIntegerType = 0;
  StartAddress = LoadNodes[0].OffsetFromBase;
  SDValue FirstChain = LoadNodes[0].MemNode->getChain();
  for (unsigned i = 1; i < LoadNodes.size(); ++i) {
    // All loads much share the same chain.
    if (LoadNodes[i].MemNode->getChain() != FirstChain)
      break;

    int64_t CurrAddress = LoadNodes[i].OffsetFromBase;
    if (CurrAddress - StartAddress != (ElementSizeBytes * i))
      break;
    LastConsecutiveLoad = i;

    // Find a legal type for the vector store.
    EVT StoreTy = EVT::getVectorVT(*DAG.getContext(), MemVT, i+1);
    if (TLI.isTypeLegal(StoreTy))
      LastLegalVectorType = i + 1;

    // Find a legal type for the integer store.
    unsigned StoreBW = (i+1) * ElementSizeBytes * 8;
    StoreTy = EVT::getIntegerVT(*DAG.getContext(), StoreBW);
    if (TLI.isTypeLegal(StoreTy))
      LastLegalIntegerType = i + 1;
    // Or check whether a truncstore and extload is legal.
    else if (TLI.getTypeAction(*DAG.getContext(), StoreTy) ==
             TargetLowering::TypePromoteInteger) {
      EVT LegalizedStoredValueTy =
        TLI.getTypeToTransformTo(*DAG.getContext(), StoreTy);
      if (TLI.isTruncStoreLegal(LegalizedStoredValueTy, StoreTy) &&
          TLI.isLoadExtLegal(ISD::ZEXTLOAD, StoreTy) &&
          TLI.isLoadExtLegal(ISD::SEXTLOAD, StoreTy) &&
          TLI.isLoadExtLegal(ISD::EXTLOAD, StoreTy))
        LastLegalIntegerType = i+1;
    }
  }

  // Only use vector types if the vector type is larger than the integer type.
  // If they are the same, use integers.
  bool UseVectorTy = LastLegalVectorType > LastLegalIntegerType && !NoVectors;
  unsigned LastLegalType = std::max(LastLegalVectorType, LastLegalIntegerType);

  // We add +1 here because the LastXXX variables refer to location while
  // the NumElem refers to array/index size.
  unsigned NumElem = std::min(LastConsecutiveStore, LastConsecutiveLoad) + 1;
  NumElem = std::min(LastLegalType, NumElem);

  if (NumElem < 2)
    return false;

  // The earliest Node in the DAG.
  unsigned EarliestNodeUsed = 0;
  LSBaseSDNode *EarliestOp = StoreNodes[EarliestNodeUsed].MemNode;
  for (unsigned i=1; i<NumElem; ++i) {
    // Find a chain for the new wide-store operand. Notice that some
    // of the store nodes that we found may not be selected for inclusion
    // in the wide store. The chain we use needs to be the chain of the
    // earliest store node which is *used* and replaced by the wide store.
    if (StoreNodes[i].SequenceNum > StoreNodes[EarliestNodeUsed].SequenceNum)
      EarliestNodeUsed = i;
  }

  // Find if it is better to use vectors or integers to load and store
  // to memory.
  EVT JointMemOpVT;
  if (UseVectorTy) {
    JointMemOpVT = EVT::getVectorVT(*DAG.getContext(), MemVT, NumElem);
  } else {
    unsigned StoreBW = NumElem * ElementSizeBytes * 8;
    JointMemOpVT = EVT::getIntegerVT(*DAG.getContext(), StoreBW);
  }

  SDLoc LoadDL(LoadNodes[0].MemNode);
  SDLoc StoreDL(StoreNodes[0].MemNode);

  LoadSDNode *FirstLoad = cast<LoadSDNode>(LoadNodes[0].MemNode);
  SDValue NewLoad = DAG.getLoad(JointMemOpVT, LoadDL,
                                FirstLoad->getChain(),
                                FirstLoad->getBasePtr(),
                                FirstLoad->getPointerInfo(),
                                false, false, false,
                                FirstLoad->getAlignment());

  SDValue NewStore = DAG.getStore(EarliestOp->getChain(), StoreDL, NewLoad,
                                  FirstInChain->getBasePtr(),
                                  FirstInChain->getPointerInfo(), false, false,
                                  FirstInChain->getAlignment());

  // Replace one of the loads with the new load.
  LoadSDNode *Ld = cast<LoadSDNode>(LoadNodes[0].MemNode);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1),
                                SDValue(NewLoad.getNode(), 1));

  // Remove the rest of the load chains.
  for (unsigned i = 1; i < NumElem ; ++i) {
    // Replace all chain users of the old load nodes with the chain of the new
    // load node.
    LoadSDNode *Ld = cast<LoadSDNode>(LoadNodes[i].MemNode);
    DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1), Ld->getChain());
  }

  // Replace the first store with the new store.
  CombineTo(EarliestOp, NewStore);
  // Erase all other stores.
  for (unsigned i = 0; i < NumElem ; ++i) {
    // Remove all Store nodes.
    if (StoreNodes[i].MemNode == EarliestOp)
      continue;
    StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
    DAG.ReplaceAllUsesOfValueWith(SDValue(St, 0), St->getChain());
    removeFromWorkList(St);
    DAG.DeleteNode(St);
  }

  return true;
}

SDValue DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();

  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BITCAST && !ST->isTruncatingStore() &&
      ST->isUnindexed()) {
    unsigned OrigAlign = ST->getAlignment();
    EVT SVT = Value.getOperand(0).getValueType();
    unsigned Align = TLI.getDataLayout()->
      getABITypeAlignment(SVT.getTypeForEVT(*DAG.getContext()));
    if (Align <= OrigAlign &&
        ((!LegalOperations && !ST->isVolatile()) ||
         TLI.isOperationLegalOrCustom(ISD::STORE, SVT)))
      return DAG.getStore(Chain, SDLoc(N), Value.getOperand(0),
                          Ptr, ST->getPointerInfo(), ST->isVolatile(),
                          ST->isNonTemporal(), OrigAlign,
                          ST->getTBAAInfo());
  }

  // Turn 'store undef, Ptr' -> nothing.
  if (Value.getOpcode() == ISD::UNDEF && ST->isUnindexed())
    return Chain;

  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Value)) {
    // NOTE: If the original store is volatile, this transform must not increase
    // the number of stores.  For example, on x86-32 an f64 can be stored in one
    // processor operation but an i64 (which is not legal) requires two.  So the
    // transform should not be done in this case.
    if (Value.getOpcode() != ISD::TargetConstantFP) {
      SDValue Tmp;
      switch (CFP->getSimpleValueType(0).SimpleTy) {
      default: llvm_unreachable("Unknown FP type");
      case MVT::f16:    // We don't do this for these yet.
      case MVT::f80:
      case MVT::f128:
      case MVT::ppcf128:
        break;
      case MVT::f32:
        if ((isTypeLegal(MVT::i32) && !LegalOperations && !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          Tmp = DAG.getConstant((uint32_t)CFP->getValueAPF().
                              bitcastToAPInt().getZExtValue(), MVT::i32);
          return DAG.getStore(Chain, SDLoc(N), Tmp,
                              Ptr, ST->getMemOperand());
        }
        break;
      case MVT::f64:
        if ((TLI.isTypeLegal(MVT::i64) && !LegalOperations &&
             !ST->isVolatile()) ||
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i64)) {
          Tmp = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                getZExtValue(), MVT::i64);
          return DAG.getStore(Chain, SDLoc(N), Tmp,
                              Ptr, ST->getMemOperand());
        }

        if (!ST->isVolatile() &&
            TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
          // Many FP stores are not made apparent until after legalize, e.g. for
          // argument passing.  Since this is so common, custom legalize the
          // 64-bit integer store into two 32-bit stores.
          uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
          SDValue Lo = DAG.getConstant(Val & 0xFFFFFFFF, MVT::i32);
          SDValue Hi = DAG.getConstant(Val >> 32, MVT::i32);
          if (TLI.isBigEndian()) std::swap(Lo, Hi);

          unsigned Alignment = ST->getAlignment();
          bool isVolatile = ST->isVolatile();
          bool isNonTemporal = ST->isNonTemporal();
          const MDNode *TBAAInfo = ST->getTBAAInfo();

          SDValue St0 = DAG.getStore(Chain, SDLoc(ST), Lo,
                                     Ptr, ST->getPointerInfo(),
                                     isVolatile, isNonTemporal,
                                     ST->getAlignment(), TBAAInfo);
          Ptr = DAG.getNode(ISD::ADD, SDLoc(N), Ptr.getValueType(), Ptr,
                            DAG.getConstant(4, Ptr.getValueType()));
          Alignment = MinAlign(Alignment, 4U);
          SDValue St1 = DAG.getStore(Chain, SDLoc(ST), Hi,
                                     Ptr, ST->getPointerInfo().getWithOffset(4),
                                     isVolatile, isNonTemporal,
                                     Alignment, TBAAInfo);
          return DAG.getNode(ISD::TokenFactor, SDLoc(N), MVT::Other,
                             St0, St1);
        }

        break;
      }
    }
  }

  // Try to infer better alignment information than the store already has.
  if (OptLevel != CodeGenOpt::None && ST->isUnindexed()) {
    if (unsigned Align = DAG.InferPtrAlignment(Ptr)) {
      if (Align > ST->getAlignment())
        return DAG.getTruncStore(Chain, SDLoc(N), Value,
                                 Ptr, ST->getPointerInfo(), ST->getMemoryVT(),
                                 ST->isVolatile(), ST->isNonTemporal(), Align,
                                 ST->getTBAAInfo());
    }
  }

  // Try transforming a pair floating point load / store ops to integer
  // load / store ops.
  SDValue NewST = TransformFPLoadStorePair(N);
  if (NewST.getNode())
    return NewST;

  bool UseAA = CombinerAA.getNumOccurrences() > 0 ? CombinerAA :
    TLI.getTargetMachine().getSubtarget<TargetSubtargetInfo>().useAA();
  if (UseAA) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(N, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplStore;

      // Replace the chain to avoid dependency.
      if (ST->isTruncatingStore()) {
        ReplStore = DAG.getTruncStore(BetterChain, SDLoc(N), Value, Ptr,
                                      ST->getMemoryVT(), ST->getMemOperand());
      } else {
        ReplStore = DAG.getStore(BetterChain, SDLoc(N), Value, Ptr,
                                 ST->getMemOperand());
      }

      // Create token to keep both nodes around.
      SDValue Token = DAG.getNode(ISD::TokenFactor, SDLoc(N),
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
                      APInt::getLowBitsSet(
                        Value.getValueType().getScalarType().getSizeInBits(),
                        ST->getMemoryVT().getScalarType().getSizeInBits()));
    AddToWorkList(Value.getNode());
    if (Shorter.getNode())
      return DAG.getTruncStore(Chain, SDLoc(N), Shorter,
                               Ptr, ST->getMemoryVT(), ST->getMemOperand());

    // Otherwise, see if we can simplify the operation with
    // SimplifyDemandedBits, which only works if the value has a single use.
    if (SimplifyDemandedBits(Value,
                        APInt::getLowBitsSet(
                          Value.getValueType().getScalarType().getSizeInBits(),
                          ST->getMemoryVT().getScalarType().getSizeInBits())))
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
    return DAG.getTruncStore(Chain, SDLoc(N), Value.getOperand(0),
                             Ptr, ST->getMemoryVT(), ST->getMemOperand());
  }

  // Only perform this optimization before the types are legal, because we
  // don't want to perform this optimization on every DAGCombine invocation.
  if (!LegalTypes) {
    bool EverChanged = false;

    do {
      // There can be multiple store sequences on the same chain.
      // Keep trying to merge store sequences until we are unable to do so
      // or until we merge the last store on the chain.
      bool Changed = MergeConsecutiveStores(ST);
      EverChanged |= Changed;
      if (!Changed) break;
    } while (ST->getOpcode() != ISD::DELETED_NODE);

    if (EverChanged)
      return SDValue(N, 0);
  }

  return ReduceLoadOpStoreWidth(N);
}

SDValue DAGCombiner::visitINSERT_VECTOR_ELT(SDNode *N) {
  SDValue InVec = N->getOperand(0);
  SDValue InVal = N->getOperand(1);
  SDValue EltNo = N->getOperand(2);
  SDLoc dl(N);

  // If the inserted element is an UNDEF, just use the input vector.
  if (InVal.getOpcode() == ISD::UNDEF)
    return InVec;

  EVT VT = InVec.getValueType();

  // If we can't generate a legal BUILD_VECTOR, exit
  if (LegalOperations && !TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return SDValue();

  // Check that we know which element is being inserted
  if (!isa<ConstantSDNode>(EltNo))
    return SDValue();
  unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();

  // Check that the operand is a BUILD_VECTOR (or UNDEF, which can essentially
  // be converted to a BUILD_VECTOR).  Fill in the Ops vector with the
  // vector elements.
  SmallVector<SDValue, 8> Ops;
  // Do not combine these two vectors if the output vector will not replace
  // the input vector.
  if (InVec.getOpcode() == ISD::BUILD_VECTOR && InVec.hasOneUse()) {
    Ops.append(InVec.getNode()->op_begin(),
               InVec.getNode()->op_end());
  } else if (InVec.getOpcode() == ISD::UNDEF) {
    unsigned NElts = VT.getVectorNumElements();
    Ops.append(NElts, DAG.getUNDEF(InVal.getValueType()));
  } else {
    return SDValue();
  }

  // Insert the element
  if (Elt < Ops.size()) {
    // All the operands of BUILD_VECTOR must have the same type;
    // we enforce that here.
    EVT OpVT = Ops[0].getValueType();
    if (InVal.getValueType() != OpVT)
      InVal = OpVT.bitsGT(InVal.getValueType()) ?
                DAG.getNode(ISD::ANY_EXTEND, dl, OpVT, InVal) :
                DAG.getNode(ISD::TRUNCATE, dl, OpVT, InVal);
    Ops[Elt] = InVal;
  }

  // Return the new vector
  return DAG.getNode(ISD::BUILD_VECTOR, dl,
                     VT, &Ops[0], Ops.size());
}

SDValue DAGCombiner::visitEXTRACT_VECTOR_ELT(SDNode *N) {
  // (vextract (scalar_to_vector val, 0) -> val
  SDValue InVec = N->getOperand(0);
  EVT VT = InVec.getValueType();
  EVT NVT = N->getValueType(0);

  if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    // Check if the result type doesn't match the inserted element type. A
    // SCALAR_TO_VECTOR may truncate the inserted element and the
    // EXTRACT_VECTOR_ELT may widen the extracted vector.
    SDValue InOp = InVec.getOperand(0);
    if (InOp.getValueType() != NVT) {
      assert(InOp.getValueType().isInteger() && NVT.isInteger());
      return DAG.getSExtOrTrunc(InOp, SDLoc(InVec), NVT);
    }
    return InOp;
  }

  SDValue EltNo = N->getOperand(1);
  bool ConstEltNo = isa<ConstantSDNode>(EltNo);

  // Transform: (EXTRACT_VECTOR_ELT( VECTOR_SHUFFLE )) -> EXTRACT_VECTOR_ELT.
  // We only perform this optimization before the op legalization phase because
  // we may introduce new vector instructions which are not backed by TD
  // patterns. For example on AVX, extracting elements from a wide vector
  // without using extract_subvector.
  if (InVec.getOpcode() == ISD::VECTOR_SHUFFLE
      && ConstEltNo && !LegalOperations) {
    int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    int NumElem = VT.getVectorNumElements();
    ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(InVec);
    // Find the new index to extract from.
    int OrigElt = SVOp->getMaskElt(Elt);

    // Extracting an undef index is undef.
    if (OrigElt == -1)
      return DAG.getUNDEF(NVT);

    // Select the right vector half to extract from.
    if (OrigElt < NumElem) {
      InVec = InVec->getOperand(0);
    } else {
      InVec = InVec->getOperand(1);
      OrigElt -= NumElem;
    }

    EVT IndexTy = TLI.getVectorIdxTy();
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SDLoc(N), NVT,
                       InVec, DAG.getConstant(OrigElt, IndexTy));
  }

  // Perform only after legalization to ensure build_vector / vector_shuffle
  // optimizations have already been done.
  if (!LegalOperations) return SDValue();

  // (vextract (v4f32 load $addr), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 s2v (f32 load $addr)), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 shuffle (load $addr), <1,u,u,u>), 0) -> (f32 load $addr)

  if (ConstEltNo) {
    int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
    bool NewLoad = false;
    bool BCNumEltsChanged = false;
    EVT ExtVT = VT.getVectorElementType();
    EVT LVT = ExtVT;

    // If the result of load has to be truncated, then it's not necessarily
    // profitable.
    if (NVT.bitsLT(LVT) && !TLI.isTruncateFree(LVT, NVT))
      return SDValue();

    if (InVec.getOpcode() == ISD::BITCAST) {
      // Don't duplicate a load with other uses.
      if (!InVec.hasOneUse())
        return SDValue();

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
      // Don't duplicate a load with other uses.
      if (!InVec.hasOneUse())
        return SDValue();

      LN0 = cast<LoadSDNode>(InVec.getOperand(0));
    } else if ((SVN = dyn_cast<ShuffleVectorSDNode>(InVec))) {
      // (vextract (vector_shuffle (load $addr), v2, <1, u, u, u>), 1)
      // =>
      // (load $addr+1*size)

      // Don't duplicate a load with other uses.
      if (!InVec.hasOneUse())
        return SDValue();

      // If the bit convert changed the number of elements, it is unsafe
      // to examine the mask.
      if (BCNumEltsChanged)
        return SDValue();

      // Select the input vector, guarding against out of range extract vector.
      unsigned NumElems = VT.getVectorNumElements();
      int Idx = (Elt > (int)NumElems) ? -1 : SVN->getMaskElt(Elt);
      InVec = (Idx < (int)NumElems) ? InVec.getOperand(0) : InVec.getOperand(1);

      if (InVec.getOpcode() == ISD::BITCAST) {
        // Don't duplicate a load with other uses.
        if (!InVec.hasOneUse())
          return SDValue();

        InVec = InVec.getOperand(0);
      }
      if (ISD::isNormalLoad(InVec.getNode())) {
        LN0 = cast<LoadSDNode>(InVec);
        Elt = (Idx < (int)NumElems) ? Idx : Idx - (int)NumElems;
      }
    }

    // Make sure we found a non-volatile load and the extractelement is
    // the only use.
    if (!LN0 || !LN0->hasNUsesOfValue(1,0) || LN0->isVolatile())
      return SDValue();

    // If Idx was -1 above, Elt is going to be -1, so just return undef.
    if (Elt == -1)
      return DAG.getUNDEF(LVT);

    unsigned Align = LN0->getAlignment();
    if (NewLoad) {
      // Check the resultant load doesn't need a higher alignment than the
      // original load.
      unsigned NewAlign =
        TLI.getDataLayout()
            ->getABITypeAlignment(LVT.getTypeForEVT(*DAG.getContext()));

      if (NewAlign > Align || !TLI.isOperationLegalOrCustom(ISD::LOAD, LVT))
        return SDValue();

      Align = NewAlign;
    }

    SDValue NewPtr = LN0->getBasePtr();
    unsigned PtrOff = 0;

    if (Elt) {
      PtrOff = LVT.getSizeInBits() * Elt / 8;
      EVT PtrType = NewPtr.getValueType();
      if (TLI.isBigEndian())
        PtrOff = VT.getSizeInBits() / 8 - PtrOff;
      NewPtr = DAG.getNode(ISD::ADD, SDLoc(N), PtrType, NewPtr,
                           DAG.getConstant(PtrOff, PtrType));
    }

    // The replacement we need to do here is a little tricky: we need to
    // replace an extractelement of a load with a load.
    // Use ReplaceAllUsesOfValuesWith to do the replacement.
    // Note that this replacement assumes that the extractvalue is the only
    // use of the load; that's okay because we don't want to perform this
    // transformation in other cases anyway.
    SDValue Load;
    SDValue Chain;
    if (NVT.bitsGT(LVT)) {
      // If the result type of vextract is wider than the load, then issue an
      // extending load instead.
      ISD::LoadExtType ExtType = TLI.isLoadExtLegal(ISD::ZEXTLOAD, LVT)
        ? ISD::ZEXTLOAD : ISD::EXTLOAD;
      Load = DAG.getExtLoad(ExtType, SDLoc(N), NVT, LN0->getChain(),
                            NewPtr, LN0->getPointerInfo().getWithOffset(PtrOff),
                            LVT, LN0->isVolatile(), LN0->isNonTemporal(),
                            Align, LN0->getTBAAInfo());
      Chain = Load.getValue(1);
    } else {
      Load = DAG.getLoad(LVT, SDLoc(N), LN0->getChain(), NewPtr,
                         LN0->getPointerInfo().getWithOffset(PtrOff),
                         LN0->isVolatile(), LN0->isNonTemporal(),
                         LN0->isInvariant(), Align, LN0->getTBAAInfo());
      Chain = Load.getValue(1);
      if (NVT.bitsLT(LVT))
        Load = DAG.getNode(ISD::TRUNCATE, SDLoc(N), NVT, Load);
      else
        Load = DAG.getNode(ISD::BITCAST, SDLoc(N), NVT, Load);
    }
    WorkListRemover DeadNodes(*this);
    SDValue From[] = { SDValue(N, 0), SDValue(LN0,1) };
    SDValue To[] = { Load, Chain };
    DAG.ReplaceAllUsesOfValuesWith(From, To, 2);
    // Since we're explcitly calling ReplaceAllUses, add the new node to the
    // worklist explicitly as well.
    AddToWorkList(Load.getNode());
    AddUsersToWorkList(Load.getNode()); // Add users too
    // Make sure to revisit this node to clean it up; it will usually be dead.
    AddToWorkList(N);
    return SDValue(N, 0);
  }

  return SDValue();
}

// Simplify (build_vec (ext )) to (bitcast (build_vec ))
SDValue DAGCombiner::reduceBuildVecExtToExtBuildVec(SDNode *N) {
  // We perform this optimization post type-legalization because
  // the type-legalizer often scalarizes integer-promoted vectors.
  // Performing this optimization before may create bit-casts which
  // will be type-legalized to complex code sequences.
  // We perform this optimization only before the operation legalizer because we
  // may introduce illegal operations.
  if (Level != AfterLegalizeVectorOps && Level != AfterLegalizeTypes)
    return SDValue();

  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  // Check to see if this is a BUILD_VECTOR of a bunch of values
  // which come from any_extend or zero_extend nodes. If so, we can create
  // a new BUILD_VECTOR using bit-casts which may enable other BUILD_VECTOR
  // optimizations. We do not handle sign-extend because we can't fill the sign
  // using shuffles.
  EVT SourceType = MVT::Other;
  bool AllAnyExt = true;

  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);
    // Ignore undef inputs.
    if (In.getOpcode() == ISD::UNDEF) continue;

    bool AnyExt  = In.getOpcode() == ISD::ANY_EXTEND;
    bool ZeroExt = In.getOpcode() == ISD::ZERO_EXTEND;

    // Abort if the element is not an extension.
    if (!ZeroExt && !AnyExt) {
      SourceType = MVT::Other;
      break;
    }

    // The input is a ZeroExt or AnyExt. Check the original type.
    EVT InTy = In.getOperand(0).getValueType();

    // Check that all of the widened source types are the same.
    if (SourceType == MVT::Other)
      // First time.
      SourceType = InTy;
    else if (InTy != SourceType) {
      // Multiple income types. Abort.
      SourceType = MVT::Other;
      break;
    }

    // Check if all of the extends are ANY_EXTENDs.
    AllAnyExt &= AnyExt;
  }

  // In order to have valid types, all of the inputs must be extended from the
  // same source type and all of the inputs must be any or zero extend.
  // Scalar sizes must be a power of two.
  EVT OutScalarTy = VT.getScalarType();
  bool ValidTypes = SourceType != MVT::Other &&
                 isPowerOf2_32(OutScalarTy.getSizeInBits()) &&
                 isPowerOf2_32(SourceType.getSizeInBits());

  // Create a new simpler BUILD_VECTOR sequence which other optimizations can
  // turn into a single shuffle instruction.
  if (!ValidTypes)
    return SDValue();

  bool isLE = TLI.isLittleEndian();
  unsigned ElemRatio = OutScalarTy.getSizeInBits()/SourceType.getSizeInBits();
  assert(ElemRatio > 1 && "Invalid element size ratio");
  SDValue Filler = AllAnyExt ? DAG.getUNDEF(SourceType):
                               DAG.getConstant(0, SourceType);

  unsigned NewBVElems = ElemRatio * VT.getVectorNumElements();
  SmallVector<SDValue, 8> Ops(NewBVElems, Filler);

  // Populate the new build_vector
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Cast = N->getOperand(i);
    assert((Cast.getOpcode() == ISD::ANY_EXTEND ||
            Cast.getOpcode() == ISD::ZERO_EXTEND ||
            Cast.getOpcode() == ISD::UNDEF) && "Invalid cast opcode");
    SDValue In;
    if (Cast.getOpcode() == ISD::UNDEF)
      In = DAG.getUNDEF(SourceType);
    else
      In = Cast->getOperand(0);
    unsigned Index = isLE ? (i * ElemRatio) :
                            (i * ElemRatio + (ElemRatio - 1));

    assert(Index < Ops.size() && "Invalid index");
    Ops[Index] = In;
  }

  // The type of the new BUILD_VECTOR node.
  EVT VecVT = EVT::getVectorVT(*DAG.getContext(), SourceType, NewBVElems);
  assert(VecVT.getSizeInBits() == VT.getSizeInBits() &&
         "Invalid vector size");
  // Check if the new vector type is legal.
  if (!isTypeLegal(VecVT)) return SDValue();

  // Make the new BUILD_VECTOR.
  SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, VecVT, &Ops[0], Ops.size());

  // The new BUILD_VECTOR node has the potential to be further optimized.
  AddToWorkList(BV.getNode());
  // Bitcast to the desired type.
  return DAG.getNode(ISD::BITCAST, dl, VT, BV);
}

SDValue DAGCombiner::reduceBuildVecConvertToConvertBuildVec(SDNode *N) {
  EVT VT = N->getValueType(0);

  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);

  EVT SrcVT = MVT::Other;
  unsigned Opcode = ISD::DELETED_NODE;
  unsigned NumDefs = 0;

  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);
    unsigned Opc = In.getOpcode();

    if (Opc == ISD::UNDEF)
      continue;

    // If all scalar values are floats and converted from integers.
    if (Opcode == ISD::DELETED_NODE &&
        (Opc == ISD::UINT_TO_FP || Opc == ISD::SINT_TO_FP)) {
      Opcode = Opc;
    }

    if (Opc != Opcode)
      return SDValue();

    EVT InVT = In.getOperand(0).getValueType();

    // If all scalar values are typed differently, bail out. It's chosen to
    // simplify BUILD_VECTOR of integer types.
    if (SrcVT == MVT::Other)
      SrcVT = InVT;
    if (SrcVT != InVT)
      return SDValue();
    NumDefs++;
  }

  // If the vector has just one element defined, it's not worth to fold it into
  // a vectorized one.
  if (NumDefs < 2)
    return SDValue();

  assert((Opcode == ISD::UINT_TO_FP || Opcode == ISD::SINT_TO_FP)
         && "Should only handle conversion from integer to float.");
  assert(SrcVT != MVT::Other && "Cannot determine source type!");

  EVT NVT = EVT::getVectorVT(*DAG.getContext(), SrcVT, NumInScalars);

  if (!TLI.isOperationLegalOrCustom(Opcode, NVT))
    return SDValue();

  SmallVector<SDValue, 8> Opnds;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);

    if (In.getOpcode() == ISD::UNDEF)
      Opnds.push_back(DAG.getUNDEF(SrcVT));
    else
      Opnds.push_back(In.getOperand(0));
  }
  SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, NVT,
                           &Opnds[0], Opnds.size());
  AddToWorkList(BV.getNode());

  return DAG.getNode(Opcode, dl, VT, BV);
}

SDValue DAGCombiner::visitBUILD_VECTOR(SDNode *N) {
  unsigned NumInScalars = N->getNumOperands();
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  // A vector built entirely of undefs is undef.
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  SDValue V = reduceBuildVecExtToExtBuildVec(N);
  if (V.getNode())
    return V;

  V = reduceBuildVecConvertToConvertBuildVec(N);
  if (V.getNode())
    return V;

  // Check to see if this is a BUILD_VECTOR of a bunch of EXTRACT_VECTOR_ELT
  // operations.  If so, and if the EXTRACT_VECTOR_ELT vector inputs come from
  // at most two distinct vectors, turn this into a shuffle node.

  // May only combine to shuffle after legalize if shuffle is legal.
  if (LegalOperations &&
      !TLI.isOperationLegalOrCustom(ISD::VECTOR_SHUFFLE, VT))
    return SDValue();

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

    // We allow up to two distinct input vectors.
    SDValue ExtractedFromVec = N->getOperand(i).getOperand(0);
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

    // We can't generate a shuffle node with mismatched input and output types.
    // Attempt to transform a single input vector to the correct type.
    if ((VT != VecIn1.getValueType())) {
      // We don't support shuffeling between TWO values of different types.
      if (VecIn2.getNode() != 0)
        return SDValue();

      // We only support widening of vectors which are half the size of the
      // output registers. For example XMM->YMM widening on X86 with AVX.
      if (VecIn1.getValueType().getSizeInBits()*2 != VT.getSizeInBits())
        return SDValue();

      // If the input vector type has a different base type to the output
      // vector type, bail out.
      if (VecIn1.getValueType().getVectorElementType() !=
          VT.getVectorElementType())
        return SDValue();

      // Widen the input vector by adding undef values.
      VecIn1 = DAG.getNode(ISD::CONCAT_VECTORS, dl, VT,
                           VecIn1, DAG.getUNDEF(VecIn1.getValueType()));
    }

    // If VecIn2 is unused then change it to undef.
    VecIn2 = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(VT);

    // Check that we were able to transform all incoming values to the same
    // type.
    if (VecIn2.getValueType() != VecIn1.getValueType() ||
        VecIn1.getValueType() != VT)
          return SDValue();

    // Only type-legal BUILD_VECTOR nodes are converted to shuffle nodes.
    if (!isTypeLegal(VT))
      return SDValue();

    // Return the new VECTOR_SHUFFLE node.
    SDValue Ops[2];
    Ops[0] = VecIn1;
    Ops[1] = VecIn2;
    return DAG.getVectorShuffle(VT, dl, Ops[0], Ops[1], &Mask[0]);
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

  // Check if all of the operands are undefs.
  EVT VT = N->getValueType(0);
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  // Optimize concat_vectors where one of the vectors is undef.
  if (N->getNumOperands() == 2 &&
      N->getOperand(1)->getOpcode() == ISD::UNDEF) {
    SDValue In = N->getOperand(0);
    assert(In.getValueType().isVector() && "Must concat vectors");

    // Transform: concat_vectors(scalar, undef) -> scalar_to_vector(sclr).
    if (In->getOpcode() == ISD::BITCAST &&
        !In->getOperand(0)->getValueType(0).isVector()) {
      SDValue Scalar = In->getOperand(0);
      EVT SclTy = Scalar->getValueType(0);

      if (!SclTy.isFloatingPoint() && !SclTy.isInteger())
        return SDValue();

      EVT NVT = EVT::getVectorVT(*DAG.getContext(), SclTy,
                                 VT.getSizeInBits() / SclTy.getSizeInBits());
      if (!TLI.isTypeLegal(NVT) || !TLI.isTypeLegal(Scalar.getValueType()))
        return SDValue();

      SDLoc dl = SDLoc(N);
      SDValue Res = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, NVT, Scalar);
      return DAG.getNode(ISD::BITCAST, dl, VT, Res);
    }
  }

  // Type legalization of vectors and DAG canonicalization of SHUFFLE_VECTOR
  // nodes often generate nop CONCAT_VECTOR nodes.
  // Scan the CONCAT_VECTOR operands and look for a CONCAT operations that
  // place the incoming vectors at the exact same location.
  SDValue SingleSource = SDValue();
  unsigned PartNumElem = N->getOperand(0).getValueType().getVectorNumElements();

  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Op = N->getOperand(i);

    if (Op.getOpcode() == ISD::UNDEF)
      continue;

    // Check if this is the identity extract:
    if (Op.getOpcode() != ISD::EXTRACT_SUBVECTOR)
      return SDValue();

    // Find the single incoming vector for the extract_subvector.
    if (SingleSource.getNode()) {
      if (Op.getOperand(0) != SingleSource)
        return SDValue();
    } else {
      SingleSource = Op.getOperand(0);

      // Check the source type is the same as the type of the result.
      // If not, this concat may extend the vector, so we can not
      // optimize it away.
      if (SingleSource.getValueType() != N->getValueType(0))
        return SDValue();
    }

    unsigned IdentityIndex = i * PartNumElem;
    ConstantSDNode *CS = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    // The extract index must be constant.
    if (!CS)
      return SDValue();

    // Check that we are reading from the identity index.
    if (CS->getZExtValue() != IdentityIndex)
      return SDValue();
  }

  if (SingleSource.getNode())
    return SingleSource;

  return SDValue();
}

SDValue DAGCombiner::visitEXTRACT_SUBVECTOR(SDNode* N) {
  EVT NVT = N->getValueType(0);
  SDValue V = N->getOperand(0);

  if (V->getOpcode() == ISD::CONCAT_VECTORS) {
    // Combine:
    //    (extract_subvec (concat V1, V2, ...), i)
    // Into:
    //    Vi if possible
    // Only operand 0 is checked as 'concat' assumes all inputs of the same
    // type.
    if (V->getOperand(0).getValueType() != NVT)
      return SDValue();
    unsigned Idx = dyn_cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    unsigned NumElems = NVT.getVectorNumElements();
    assert((Idx % NumElems) == 0 &&
           "IDX in concat is not a multiple of the result vector length.");
    return V->getOperand(Idx / NumElems);
  }

  // Skip bitcasting
  if (V->getOpcode() == ISD::BITCAST)
    V = V.getOperand(0);

  if (V->getOpcode() == ISD::INSERT_SUBVECTOR) {
    SDLoc dl(N);
    // Handle only simple case where vector being inserted and vector
    // being extracted are of same type, and are half size of larger vectors.
    EVT BigVT = V->getOperand(0).getValueType();
    EVT SmallVT = V->getOperand(1).getValueType();
    if (!NVT.bitsEq(SmallVT) || NVT.getSizeInBits()*2 != BigVT.getSizeInBits())
      return SDValue();

    // Only handle cases where both indexes are constants with the same type.
    ConstantSDNode *ExtIdx = dyn_cast<ConstantSDNode>(N->getOperand(1));
    ConstantSDNode *InsIdx = dyn_cast<ConstantSDNode>(V->getOperand(2));

    if (InsIdx && ExtIdx &&
        InsIdx->getValueType(0).getSizeInBits() <= 64 &&
        ExtIdx->getValueType(0).getSizeInBits() <= 64) {
      // Combine:
      //    (extract_subvec (insert_subvec V1, V2, InsIdx), ExtIdx)
      // Into:
      //    indices are equal or bit offsets are equal => V1
      //    otherwise => (extract_subvec V1, ExtIdx)
      if (InsIdx->getZExtValue() * SmallVT.getScalarType().getSizeInBits() ==
          ExtIdx->getZExtValue() * NVT.getScalarType().getSizeInBits())
        return DAG.getNode(ISD::BITCAST, dl, NVT, V->getOperand(1));
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NVT,
                         DAG.getNode(ISD::BITCAST, dl,
                                     N->getOperand(0).getValueType(),
                                     V->getOperand(0)), N->getOperand(1));
    }
  }

  return SDValue();
}

// Tries to turn a shuffle of two CONCAT_VECTORS into a single concat.
static SDValue partitionShuffleOfConcats(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);

  SmallVector<SDValue, 4> Ops;
  EVT ConcatVT = N0.getOperand(0).getValueType();
  unsigned NumElemsPerConcat = ConcatVT.getVectorNumElements();
  unsigned NumConcats = NumElts / NumElemsPerConcat;

  // Look at every vector that's inserted. We're looking for exact
  // subvector-sized copies from a concatenated vector
  for (unsigned I = 0; I != NumConcats; ++I) {
    // Make sure we're dealing with a copy.
    unsigned Begin = I * NumElemsPerConcat;
    bool AllUndef = true, NoUndef = true;
    for (unsigned J = Begin; J != Begin + NumElemsPerConcat; ++J) {
      if (SVN->getMaskElt(J) >= 0)
        AllUndef = false;
      else
        NoUndef = false;
    }

    if (NoUndef) {
      if (SVN->getMaskElt(Begin) % NumElemsPerConcat != 0)
        return SDValue();

      for (unsigned J = 1; J != NumElemsPerConcat; ++J)
        if (SVN->getMaskElt(Begin + J - 1) + 1 != SVN->getMaskElt(Begin + J))
          return SDValue();

      unsigned FirstElt = SVN->getMaskElt(Begin) / NumElemsPerConcat;
      if (FirstElt < N0.getNumOperands())
        Ops.push_back(N0.getOperand(FirstElt));
      else
        Ops.push_back(N1.getOperand(FirstElt - N0.getNumOperands()));

    } else if (AllUndef) {
      Ops.push_back(DAG.getUNDEF(N0.getOperand(0).getValueType()));
    } else { // Mixed with general masks and undefs, can't do optimization.
      return SDValue();
    }
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Ops.data(),
                     Ops.size());
}

SDValue DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  assert(N0.getValueType() == VT && "Vector shuffle must be normalized in DAG");

  // Canonicalize shuffle undef, undef -> undef
  if (N0.getOpcode() == ISD::UNDEF && N1.getOpcode() == ISD::UNDEF)
    return DAG.getUNDEF(VT);

  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);

  // Canonicalize shuffle v, v -> v, undef
  if (N0 == N1) {
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) Idx -= NumElts;
      NewMask.push_back(Idx);
    }
    return DAG.getVectorShuffle(VT, SDLoc(N), N0, DAG.getUNDEF(VT),
                                &NewMask[0]);
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N0.getOpcode() == ISD::UNDEF) {
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= 0) {
        if (Idx >= (int)NumElts)
          Idx -= NumElts;
        else
          Idx = -1; // remove reference to lhs
      }
      NewMask.push_back(Idx);
    }
    return DAG.getVectorShuffle(VT, SDLoc(N), N1, DAG.getUNDEF(VT),
                                &NewMask[0]);
  }

  // Remove references to rhs if it is undef
  if (N1.getOpcode() == ISD::UNDEF) {
    bool Changed = false;
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) {
        Idx = -1;
        Changed = true;
      }
      NewMask.push_back(Idx);
    }
    if (Changed)
      return DAG.getVectorShuffle(VT, SDLoc(N), N0, N1, &NewMask[0]);
  }

  // If it is a splat, check if the argument vector is another splat or a
  // build_vector with all scalar elements the same.
  if (SVN->isSplat() && SVN->getSplatIndex() < (int)NumElts) {
    SDNode *V = N0.getNode();

    // If this is a bit convert that changes the element type of the vector but
    // not the number of vector elements, look through it.  Be careful not to
    // look though conversions that change things like v4f32 to v2f64.
    if (V->getOpcode() == ISD::BITCAST) {
      SDValue ConvInput = V->getOperand(0);
      if (ConvInput.getValueType().isVector() &&
          ConvInput.getValueType().getVectorNumElements() == NumElts)
        V = ConvInput.getNode();
    }

    if (V->getOpcode() == ISD::BUILD_VECTOR) {
      assert(V->getNumOperands() == NumElts &&
             "BUILD_VECTOR has wrong number of operands");
      SDValue Base;
      bool AllSame = true;
      for (unsigned i = 0; i != NumElts; ++i) {
        if (V->getOperand(i).getOpcode() != ISD::UNDEF) {
          Base = V->getOperand(i);
          break;
        }
      }
      // Splat of <u, u, u, u>, return <u, u, u, u>
      if (!Base.getNode())
        return N0;
      for (unsigned i = 0; i != NumElts; ++i) {
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

  if (N0.getOpcode() == ISD::CONCAT_VECTORS &&
      Level < AfterLegalizeVectorOps &&
      (N1.getOpcode() == ISD::UNDEF ||
      (N1.getOpcode() == ISD::CONCAT_VECTORS &&
       N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()))) {
    SDValue V = partitionShuffleOfConcats(N, DAG);

    if (V.getNode())
      return V;
  }

  // If this shuffle node is simply a swizzle of another shuffle node,
  // and it reverses the swizzle of the previous shuffle then we can
  // optimize shuffle(shuffle(x, undef), undef) -> x.
  if (N0.getOpcode() == ISD::VECTOR_SHUFFLE && Level < AfterLegalizeDAG &&
      N1.getOpcode() == ISD::UNDEF) {

    ShuffleVectorSDNode *OtherSV = cast<ShuffleVectorSDNode>(N0);

    // Shuffle nodes can only reverse shuffles with a single non-undef value.
    if (N0.getOperand(1).getOpcode() != ISD::UNDEF)
      return SDValue();

    // The incoming shuffle must be of the same type as the result of the
    // current shuffle.
    assert(OtherSV->getOperand(0).getValueType() == VT &&
           "Shuffle types don't match");

    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      assert(Idx < (int)NumElts && "Index references undef operand");
      // Next, this index comes from the first value, which is the incoming
      // shuffle. Adopt the incoming index.
      if (Idx >= 0)
        Idx = OtherSV->getMaskElt(Idx);

      // The combined shuffle must map each index to itself.
      if (Idx >= 0 && (unsigned)Idx != i)
        return SDValue();
    }

    return OtherSV->getOperand(0);
  }

  return SDValue();
}

/// XformToShuffleWithZero - Returns a vector_shuffle if it able to transform
/// an AND to a vector_shuffle with the destination vector and a zero vector.
/// e.g. AND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDValue DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (N->getOpcode() == ISD::AND) {
    if (RHS.getOpcode() == ISD::BITCAST)
      RHS = RHS.getOperand(0);
    if (RHS.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<int, 8> Indices;
      unsigned NumElts = RHS.getNumOperands();
      for (unsigned i = 0; i != NumElts; ++i) {
        SDValue Elt = RHS.getOperand(i);
        if (!isa<ConstantSDNode>(Elt))
          return SDValue();

        if (cast<ConstantSDNode>(Elt)->isAllOnesValue())
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
      SDValue Zero = DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N),
                                 RVT, &ZeroOps[0], ZeroOps.size());
      LHS = DAG.getNode(ISD::BITCAST, dl, RVT, LHS);
      SDValue Shuf = DAG.getVectorShuffle(RVT, dl, LHS, Zero, &Indices[0]);
      return DAG.getNode(ISD::BITCAST, dl, VT, Shuf);
    }
  }

  return SDValue();
}

/// SimplifyVBinOp - Visit a binary vector operation, like ADD.
SDValue DAGCombiner::SimplifyVBinOp(SDNode *N) {
  assert(N->getValueType(0).isVector() &&
         "SimplifyVBinOp only works on vectors!");

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

      EVT VT = LHSOp.getValueType();
      EVT RVT = RHSOp.getValueType();
      if (RVT != VT) {
        // Integer BUILD_VECTOR operands may have types larger than the element
        // size (e.g., when the element type is not legal).  Prior to type
        // legalization, the types may not match between the two BUILD_VECTORS.
        // Truncate one of the operands to make them match.
        if (RVT.getSizeInBits() > VT.getSizeInBits()) {
          RHSOp = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, RHSOp);
        } else {
          LHSOp = DAG.getNode(ISD::TRUNCATE, SDLoc(N), RVT, LHSOp);
          VT = RVT;
        }
      }
      SDValue FoldOp = DAG.getNode(N->getOpcode(), SDLoc(LHS), VT,
                                   LHSOp, RHSOp);
      if (FoldOp.getOpcode() != ISD::UNDEF &&
          FoldOp.getOpcode() != ISD::Constant &&
          FoldOp.getOpcode() != ISD::ConstantFP)
        break;
      Ops.push_back(FoldOp);
      AddToWorkList(FoldOp.getNode());
    }

    if (Ops.size() == LHS.getNumOperands())
      return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N),
                         LHS.getValueType(), &Ops[0], Ops.size());
  }

  return SDValue();
}

/// SimplifyVUnaryOp - Visit a binary vector operation, like FABS/FNEG.
SDValue DAGCombiner::SimplifyVUnaryOp(SDNode *N) {
  assert(N->getValueType(0).isVector() &&
         "SimplifyVUnaryOp only works on vectors!");

  SDValue N0 = N->getOperand(0);

  if (N0.getOpcode() != ISD::BUILD_VECTOR)
    return SDValue();

  // Operand is a BUILD_VECTOR node, see if we can constant fold it.
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0, e = N0.getNumOperands(); i != e; ++i) {
    SDValue Op = N0.getOperand(i);
    if (Op.getOpcode() != ISD::UNDEF &&
        Op.getOpcode() != ISD::ConstantFP)
      break;
    EVT EltVT = Op.getValueType();
    SDValue FoldOp = DAG.getNode(N->getOpcode(), SDLoc(N0), EltVT, Op);
    if (FoldOp.getOpcode() != ISD::UNDEF &&
        FoldOp.getOpcode() != ISD::ConstantFP)
      break;
    Ops.push_back(FoldOp);
    AddToWorkList(FoldOp.getNode());
  }

  if (Ops.size() != N0.getNumOperands())
    return SDValue();

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(N),
                     N0.getValueType(), &Ops[0], Ops.size());
}

SDValue DAGCombiner::SimplifySelect(SDLoc DL, SDValue N0,
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
      SDValue SETCC = DAG.getNode(ISD::SETCC, SDLoc(N0),
                                  N0.getValueType(),
                                  SCC.getOperand(0), SCC.getOperand(1),
                                  SCC.getOperand(4));
      AddToWorkList(SETCC.getNode());
      return DAG.getSelect(SDLoc(SCC), SCC.getValueType(),
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

  // Cannot simplify select with vector condition
  if (TheSelect->getOperand(0).getValueType().isVector()) return false;

  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() != RHS.getOpcode() ||
      !LHS.hasOneUse() || !RHS.hasOneUse())
    return false;

  // If this is a load and the token chain is identical, replace the select
  // of two loads with a load through a select of the address to load from.
  // This triggers in things like "select bool X, 10.0, 123.0" after the FP
  // constants have been dropped into the constant pool.
  if (LHS.getOpcode() == ISD::LOAD) {
    LoadSDNode *LLD = cast<LoadSDNode>(LHS);
    LoadSDNode *RLD = cast<LoadSDNode>(RHS);

    // Token chains must be identical.
    if (LHS.getOperand(0) != RHS.getOperand(0) ||
        // Do not let this transformation reduce the number of volatile loads.
        LLD->isVolatile() || RLD->isVolatile() ||
        // If this is an EXTLOAD, the VT's must match.
        LLD->getMemoryVT() != RLD->getMemoryVT() ||
        // If this is an EXTLOAD, the kind of extension must match.
        (LLD->getExtensionType() != RLD->getExtensionType() &&
         // The only exception is if one of the extensions is anyext.
         LLD->getExtensionType() != ISD::EXTLOAD &&
         RLD->getExtensionType() != ISD::EXTLOAD) ||
        // FIXME: this discards src value information.  This is
        // over-conservative. It would be beneficial to be able to remember
        // both potential memory locations.  Since we are discarding
        // src value info, don't do the transformation if the memory
        // locations are not in the default address space.
        LLD->getPointerInfo().getAddrSpace() != 0 ||
        RLD->getPointerInfo().getAddrSpace() != 0 ||
        !TLI.isOperationLegalOrCustom(TheSelect->getOpcode(),
                                      LLD->getBasePtr().getValueType()))
      return false;

    // Check that the select condition doesn't reach either load.  If so,
    // folding this will induce a cycle into the DAG.  If not, this is safe to
    // xform, so create a select of the addresses.
    SDValue Addr;
    if (TheSelect->getOpcode() == ISD::SELECT) {
      SDNode *CondNode = TheSelect->getOperand(0).getNode();
      if ((LLD->hasAnyUseOfValue(1) && LLD->isPredecessorOf(CondNode)) ||
          (RLD->hasAnyUseOfValue(1) && RLD->isPredecessorOf(CondNode)))
        return false;
      // The loads must not depend on one another.
      if (LLD->isPredecessorOf(RLD) ||
          RLD->isPredecessorOf(LLD))
        return false;
      Addr = DAG.getSelect(SDLoc(TheSelect),
                           LLD->getBasePtr().getValueType(),
                           TheSelect->getOperand(0), LLD->getBasePtr(),
                           RLD->getBasePtr());
    } else {  // Otherwise SELECT_CC
      SDNode *CondLHS = TheSelect->getOperand(0).getNode();
      SDNode *CondRHS = TheSelect->getOperand(1).getNode();

      if ((LLD->hasAnyUseOfValue(1) &&
           (LLD->isPredecessorOf(CondLHS) || LLD->isPredecessorOf(CondRHS))) ||
          (RLD->hasAnyUseOfValue(1) &&
           (RLD->isPredecessorOf(CondLHS) || RLD->isPredecessorOf(CondRHS))))
        return false;

      Addr = DAG.getNode(ISD::SELECT_CC, SDLoc(TheSelect),
                         LLD->getBasePtr().getValueType(),
                         TheSelect->getOperand(0),
                         TheSelect->getOperand(1),
                         LLD->getBasePtr(), RLD->getBasePtr(),
                         TheSelect->getOperand(4));
    }

    SDValue Load;
    if (LLD->getExtensionType() == ISD::NON_EXTLOAD) {
      Load = DAG.getLoad(TheSelect->getValueType(0),
                         SDLoc(TheSelect),
                         // FIXME: Discards pointer and TBAA info.
                         LLD->getChain(), Addr, MachinePointerInfo(),
                         LLD->isVolatile(), LLD->isNonTemporal(),
                         LLD->isInvariant(), LLD->getAlignment());
    } else {
      Load = DAG.getExtLoad(LLD->getExtensionType() == ISD::EXTLOAD ?
                            RLD->getExtensionType() : LLD->getExtensionType(),
                            SDLoc(TheSelect),
                            TheSelect->getValueType(0),
                            // FIXME: Discards pointer and TBAA info.
                            LLD->getChain(), Addr, MachinePointerInfo(),
                            LLD->getMemoryVT(), LLD->isVolatile(),
                            LLD->isNonTemporal(), LLD->getAlignment());
    }

    // Users of the select now use the result of the load.
    CombineTo(TheSelect, Load);

    // Users of the old loads now use the new load's chain.  We know the
    // old-load value is dead now.
    CombineTo(LHS.getNode(), Load.getValue(0), Load.getValue(1));
    CombineTo(RHS.getNode(), Load.getValue(0), Load.getValue(1));
    return true;
  }

  return false;
}

/// SimplifySelectCC - Simplify an expression of the form (N0 cond N1) ? N2 : N3
/// where 'cond' is the comparison specified by CC.
SDValue DAGCombiner::SimplifySelectCC(SDLoc DL, SDValue N0, SDValue N1,
                                      SDValue N2, SDValue N3,
                                      ISD::CondCode CC, bool NotExtCompare) {
  // (x ? y : y) -> y.
  if (N2 == N3) return N2;

  EVT VT = N2.getValueType();
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.getNode());

  // Determine if the condition we're dealing with is constant
  SDValue SCC = SimplifySetCC(getSetCCResultType(N0.getValueType()),
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
        Type *FPTy = Elts[0]->getType();
        const DataLayout &TD = *TLI.getDataLayout();

        // Create a ConstantArray of the two constants.
        Constant *CA = ConstantArray::get(ArrayType::get(FPTy, 2), Elts);
        SDValue CPIdx = DAG.getConstantPool(CA, TLI.getPointerTy(),
                                            TD.getPrefTypeAlignment(FPTy));
        unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();

        // Get the offsets to the 0 and 1 element of the array so that we can
        // select between them.
        SDValue Zero = DAG.getIntPtrConstant(0);
        unsigned EltSize = (unsigned)TD.getTypeAllocSize(Elts[0]->getType());
        SDValue One = DAG.getIntPtrConstant(EltSize);

        SDValue Cond = DAG.getSetCC(DL,
                                    getSetCCResultType(N0.getValueType()),
                                    N0, N1, CC);
        AddToWorkList(Cond.getNode());
        SDValue CstOffset = DAG.getSelect(DL, Zero.getValueType(),
                                          Cond, One, Zero);
        AddToWorkList(CstOffset.getNode());
        CPIdx = DAG.getNode(ISD::ADD, DL, CPIdx.getValueType(), CPIdx,
                            CstOffset);
        AddToWorkList(CPIdx.getNode());
        return DAG.getLoad(TV->getValueType(0), DL, DAG.getEntryNode(), CPIdx,
                           MachinePointerInfo::getConstantPool(), false,
                           false, false, Alignment);

      }
    }

  // Check to see if we can perform the "gzip trick", transforming
  // (select_cc setlt X, 0, A, 0) -> (and (sra X, (sub size(X), 1), A)
  if (N1C && N3C && N3C->isNullValue() && CC == ISD::SETLT &&
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
        SDValue ShCt = DAG.getConstant(ShCtV,
                                       getShiftAmountTy(N0.getValueType()));
        SDValue Shift = DAG.getNode(ISD::SRL, SDLoc(N0),
                                    XType, N0, ShCt);
        AddToWorkList(Shift.getNode());

        if (XType.bitsGT(AType)) {
          Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
          AddToWorkList(Shift.getNode());
        }

        return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
      }

      SDValue Shift = DAG.getNode(ISD::SRA, SDLoc(N0),
                                  XType, N0,
                                  DAG.getConstant(XType.getSizeInBits()-1,
                                         getShiftAmountTy(N0.getValueType())));
      AddToWorkList(Shift.getNode());

      if (XType.bitsGT(AType)) {
        Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
        AddToWorkList(Shift.getNode());
      }

      return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
    }
  }

  // fold (select_cc seteq (and x, y), 0, 0, A) -> (and (shr (shl x)) A)
  // where y is has a single bit set.
  // A plaintext description would be, we can turn the SELECT_CC into an AND
  // when the condition can be materialized as an all-ones register.  Any
  // single bit-test can be materialized as an all-ones register with
  // shift-left and shift-right-arith.
  if (CC == ISD::SETEQ && N0->getOpcode() == ISD::AND &&
      N0->getValueType(0) == VT &&
      N1C && N1C->isNullValue() &&
      N2C && N2C->isNullValue()) {
    SDValue AndLHS = N0->getOperand(0);
    ConstantSDNode *ConstAndRHS = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (ConstAndRHS && ConstAndRHS->getAPIntValue().countPopulation() == 1) {
      // Shift the tested bit over the sign bit.
      APInt AndMask = ConstAndRHS->getAPIntValue();
      SDValue ShlAmt =
        DAG.getConstant(AndMask.countLeadingZeros(),
                        getShiftAmountTy(AndLHS.getValueType()));
      SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(N0), VT, AndLHS, ShlAmt);

      // Now arithmetic right shift it all the way over, so the result is either
      // all-ones, or zero.
      SDValue ShrAmt =
        DAG.getConstant(AndMask.getBitWidth()-1,
                        getShiftAmountTy(Shl.getValueType()));
      SDValue Shr = DAG.getNode(ISD::SRA, SDLoc(N0), VT, Shl, ShrAmt);

      return DAG.getNode(ISD::AND, DL, VT, Shr, N3);
    }
  }

  // fold select C, 16, 0 -> shl C, 4
  if (N2C && N3C && N3C->isNullValue() && N2C->getAPIntValue().isPowerOf2() &&
    TLI.getBooleanContents(N0.getValueType().isVector()) ==
      TargetLowering::ZeroOrOneBooleanContent) {

    // If the caller doesn't want us to simplify this into a zext of a compare,
    // don't do it.
    if (NotExtCompare && N2C->getAPIntValue() == 1)
      return SDValue();

    // Get a SetCC of the condition
    // NOTE: Don't create a SETCC if it's not legal on this target.
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC,
          LegalTypes ? getSetCCResultType(N0.getValueType()) : MVT::i1)) {
      SDValue Temp, SCC;
      // cast from setcc result type to select result type
      if (LegalTypes) {
        SCC  = DAG.getSetCC(DL, getSetCCResultType(N0.getValueType()),
                            N0, N1, CC);
        if (N2.getValueType().bitsLT(SCC.getValueType()))
          Temp = DAG.getZeroExtendInReg(SCC, SDLoc(N2),
                                        N2.getValueType());
        else
          Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2),
                             N2.getValueType(), SCC);
      } else {
        SCC  = DAG.getSetCC(SDLoc(N0), MVT::i1, N0, N1, CC);
        Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2),
                           N2.getValueType(), SCC);
      }

      AddToWorkList(SCC.getNode());
      AddToWorkList(Temp.getNode());

      if (N2C->getAPIntValue() == 1)
        return Temp;

      // shl setcc result by log2 n2c
      return DAG.getNode(
          ISD::SHL, DL, N2.getValueType(), Temp,
          DAG.getConstant(N2C->getAPIntValue().logBase2(),
                          getShiftAmountTy(Temp.getValueType())));
    }
  }

  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && N3C && N3C->isNullValue() && N2C && (N2C->getAPIntValue() == 1ULL)) {
    EVT XType = N0.getValueType();
    if (!LegalOperations ||
        TLI.isOperationLegal(ISD::SETCC, getSetCCResultType(XType))) {
      SDValue Res = DAG.getSetCC(DL, getSetCCResultType(XType), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Res);
      return Res;
    }

    // fold (seteq X, 0) -> (srl (ctlz X, log2(size(X))))
    if (N1C && N1C->isNullValue() && CC == ISD::SETEQ &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::CTLZ, XType))) {
      SDValue Ctlz = DAG.getNode(ISD::CTLZ, SDLoc(N0), XType, N0);
      return DAG.getNode(ISD::SRL, DL, XType, Ctlz,
                         DAG.getConstant(Log2_32(XType.getSizeInBits()),
                                       getShiftAmountTy(Ctlz.getValueType())));
    }
    // fold (setgt X, 0) -> (srl (and (-X, ~X), size(X)-1))
    if (N1C && N1C->isNullValue() && CC == ISD::SETGT) {
      SDValue NegN0 = DAG.getNode(ISD::SUB, SDLoc(N0),
                                  XType, DAG.getConstant(0, XType), N0);
      SDValue NotN0 = DAG.getNOT(SDLoc(N0), N0, XType);
      return DAG.getNode(ISD::SRL, DL, XType,
                         DAG.getNode(ISD::AND, DL, XType, NegN0, NotN0),
                         DAG.getConstant(XType.getSizeInBits()-1,
                                         getShiftAmountTy(XType)));
    }
    // fold (setgt X, -1) -> (xor (srl (X, size(X)-1), 1))
    if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT) {
      SDValue Sign = DAG.getNode(ISD::SRL, SDLoc(N0), XType, N0,
                                 DAG.getConstant(XType.getSizeInBits()-1,
                                         getShiftAmountTy(N0.getValueType())));
      return DAG.getNode(ISD::XOR, DL, XType, Sign, DAG.getConstant(1, XType));
    }
  }

  // Check to see if this is an integer abs.
  // select_cc setg[te] X,  0,  X, -X ->
  // select_cc setgt    X, -1,  X, -X ->
  // select_cc setl[te] X,  0, -X,  X ->
  // select_cc setlt    X,  1, -X,  X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C) {
    ConstantSDNode *SubC = NULL;
    if (((N1C->isNullValue() && (CC == ISD::SETGT || CC == ISD::SETGE)) ||
         (N1C->isAllOnesValue() && CC == ISD::SETGT)) &&
        N0 == N2 && N3.getOpcode() == ISD::SUB && N0 == N3.getOperand(1))
      SubC = dyn_cast<ConstantSDNode>(N3.getOperand(0));
    else if (((N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE)) ||
              (N1C->isOne() && CC == ISD::SETLT)) &&
             N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1))
      SubC = dyn_cast<ConstantSDNode>(N2.getOperand(0));

    EVT XType = N0.getValueType();
    if (SubC && SubC->isNullValue() && XType.isInteger()) {
      SDValue Shift = DAG.getNode(ISD::SRA, SDLoc(N0), XType,
                                  N0,
                                  DAG.getConstant(XType.getSizeInBits()-1,
                                         getShiftAmountTy(N0.getValueType())));
      SDValue Add = DAG.getNode(ISD::ADD, SDLoc(N0),
                                XType, N0, Shift);
      AddToWorkList(Shift.getNode());
      AddToWorkList(Add.getNode());
      return DAG.getNode(ISD::XOR, DL, XType, Add, Shift);
    }
  }

  return SDValue();
}

/// SimplifySetCC - This is a stub for TargetLowering::SimplifySetCC.
SDValue DAGCombiner::SimplifySetCC(EVT VT, SDValue N0,
                                   SDValue N1, ISD::CondCode Cond,
                                   SDLoc DL, bool foldBooleans) {
  TargetLowering::DAGCombinerInfo
    DagCombineInfo(DAG, Level, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo, DL);
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDValue DAGCombiner::BuildSDIV(SDNode *N) {
  std::vector<SDNode*> Built;
  SDValue S = TLI.BuildSDIV(N, DAG, LegalOperations, &Built);

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
  SDValue S = TLI.BuildUDIV(N, DAG, LegalOperations, &Built);

  for (std::vector<SDNode*>::iterator ii = Built.begin(), ee = Built.end();
       ii != ee; ++ii)
    AddToWorkList(*ii);
  return S;
}

/// FindBaseOffset - Return true if base is a frame index, which is known not
// to alias with anything but itself.  Provides base object and offset as
// results.
static bool FindBaseOffset(SDValue Ptr, SDValue &Base, int64_t &Offset,
                           const GlobalValue *&GV, const void *&CV) {
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
    CV = C->isMachineConstantPoolEntry() ? (const void *)C->getMachineCPVal()
                                         : (const void *)C->getConstVal();
    Offset += C->getOffset();
    return false;
  }
  // If it's any of the following then it can't alias with anything but itself.
  return isa<FrameIndexSDNode>(Base);
}

/// isAlias - Return true if there is any possibility that the two addresses
/// overlap.
bool DAGCombiner::isAlias(SDValue Ptr1, int64_t Size1, bool IsVolatile1,
                          const Value *SrcValue1, int SrcValueOffset1,
                          unsigned SrcValueAlign1,
                          const MDNode *TBAAInfo1,
                          SDValue Ptr2, int64_t Size2, bool IsVolatile2,
                          const Value *SrcValue2, int SrcValueOffset2,
                          unsigned SrcValueAlign2,
                          const MDNode *TBAAInfo2) const {
  // If they are the same then they must be aliases.
  if (Ptr1 == Ptr2) return true;

  // If they are both volatile then they cannot be reordered.
  if (IsVolatile1 && IsVolatile2) return true;

  // Gather base node and offset information.
  SDValue Base1, Base2;
  int64_t Offset1, Offset2;
  const GlobalValue *GV1, *GV2;
  const void *CV1, *CV2;
  bool isFrameIndex1 = FindBaseOffset(Ptr1, Base1, Offset1, GV1, CV1);
  bool isFrameIndex2 = FindBaseOffset(Ptr2, Base2, Offset2, GV2, CV2);

  // If they have a same base address then check to see if they overlap.
  if (Base1 == Base2 || (GV1 && (GV1 == GV2)) || (CV1 && (CV1 == CV2)))
    return !((Offset1 + Size1) <= Offset2 || (Offset2 + Size2) <= Offset1);

  // It is possible for different frame indices to alias each other, mostly
  // when tail call optimization reuses return address slots for arguments.
  // To catch this case, look up the actual index of frame indices to compute
  // the real alias relationship.
  if (isFrameIndex1 && isFrameIndex2) {
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    Offset1 += MFI->getObjectOffset(cast<FrameIndexSDNode>(Base1)->getIndex());
    Offset2 += MFI->getObjectOffset(cast<FrameIndexSDNode>(Base2)->getIndex());
    return !((Offset1 + Size1) <= Offset2 || (Offset2 + Size2) <= Offset1);
  }

  // Otherwise, if we know what the bases are, and they aren't identical, then
  // we know they cannot alias.
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

  bool UseAA = CombinerGlobalAA.getNumOccurrences() > 0 ? CombinerGlobalAA :
    TLI.getTargetMachine().getSubtarget<TargetSubtargetInfo>().useAA();
  if (UseAA && SrcValue1 && SrcValue2) {
    // Use alias analysis information.
    int64_t MinOffset = std::min(SrcValueOffset1, SrcValueOffset2);
    int64_t Overlap1 = Size1 + SrcValueOffset1 - MinOffset;
    int64_t Overlap2 = Size2 + SrcValueOffset2 - MinOffset;
    AliasAnalysis::AliasResult AAResult =
      AA.alias(AliasAnalysis::Location(SrcValue1, Overlap1, TBAAInfo1),
               AliasAnalysis::Location(SrcValue2, Overlap2, TBAAInfo2));
    if (AAResult == AliasAnalysis::NoAlias)
      return false;
  }

  // Otherwise we have to assume they alias.
  return true;
}

bool DAGCombiner::isAlias(LSBaseSDNode *Op0, LSBaseSDNode *Op1) {
  SDValue Ptr0, Ptr1;
  int64_t Size0, Size1;
  bool IsVolatile0, IsVolatile1;
  const Value *SrcValue0, *SrcValue1;
  int SrcValueOffset0, SrcValueOffset1;
  unsigned SrcValueAlign0, SrcValueAlign1;
  const MDNode *SrcTBAAInfo0, *SrcTBAAInfo1;
  FindAliasInfo(Op0, Ptr0, Size0, IsVolatile0, SrcValue0, SrcValueOffset0,
                SrcValueAlign0, SrcTBAAInfo0);
  FindAliasInfo(Op1, Ptr1, Size1, IsVolatile1, SrcValue1, SrcValueOffset1,
                SrcValueAlign1, SrcTBAAInfo1);
  return isAlias(Ptr0, Size0, IsVolatile0, SrcValue0, SrcValueOffset0,
                 SrcValueAlign0, SrcTBAAInfo0,
                 Ptr1, Size1, IsVolatile1, SrcValue1, SrcValueOffset1,
                 SrcValueAlign1, SrcTBAAInfo1);
}

/// FindAliasInfo - Extracts the relevant alias information from the memory
/// node.  Returns true if the operand was a nonvolatile load.
bool DAGCombiner::FindAliasInfo(SDNode *N,
                                SDValue &Ptr, int64_t &Size, bool &IsVolatile,
                                const Value *&SrcValue,
                                int &SrcValueOffset,
                                unsigned &SrcValueAlign,
                                const MDNode *&TBAAInfo) const {
  LSBaseSDNode *LS = cast<LSBaseSDNode>(N);

  Ptr = LS->getBasePtr();
  Size = LS->getMemoryVT().getSizeInBits() >> 3;
  IsVolatile = LS->isVolatile();
  SrcValue = LS->getSrcValue();
  SrcValueOffset = LS->getSrcValueOffset();
  SrcValueAlign = LS->getOriginalAlignment();
  TBAAInfo = LS->getTBAAInfo();
  return isa<LoadSDNode>(LS) && !IsVolatile;
}

/// GatherAllAliases - Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDValue OriginalChain,
                                   SmallVectorImpl<SDValue> &Aliases) {
  SmallVector<SDValue, 8> Chains;     // List of chains to visit.
  SmallPtrSet<SDNode *, 16> Visited;  // Visited node set.

  // Get alias information for node.
  SDValue Ptr;
  int64_t Size;
  bool IsVolatile;
  const Value *SrcValue;
  int SrcValueOffset;
  unsigned SrcValueAlign;
  const MDNode *SrcTBAAInfo;
  bool IsLoad = FindAliasInfo(N, Ptr, Size, IsVolatile, SrcValue,
                              SrcValueOffset, SrcValueAlign, SrcTBAAInfo);

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
      bool OpIsVolatile;
      const Value *OpSrcValue;
      int OpSrcValueOffset;
      unsigned OpSrcValueAlign;
      const MDNode *OpSrcTBAAInfo;
      bool IsOpLoad = FindAliasInfo(Chain.getNode(), OpPtr, OpSize,
                                    OpIsVolatile, OpSrcValue, OpSrcValueOffset,
                                    OpSrcValueAlign,
                                    OpSrcTBAAInfo);

      // If chain is alias then stop here.
      if (!(IsLoad && IsOpLoad) &&
          isAlias(Ptr, Size, IsVolatile, SrcValue, SrcValueOffset,
                  SrcValueAlign, SrcTBAAInfo,
                  OpPtr, OpSize, OpIsVolatile, OpSrcValue, OpSrcValueOffset,
                  OpSrcValueAlign, OpSrcTBAAInfo)) {
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

  // If no operands then chain to entry token.
  if (Aliases.size() == 0)
    return DAG.getEntryNode();

  // If a single operand then chain to it.  We don't need to revisit it.
  if (Aliases.size() == 1)
    return Aliases[0];

  // Construct a custom tailored token factor.
  return DAG.getNode(ISD::TokenFactor, SDLoc(N), MVT::Other,
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
