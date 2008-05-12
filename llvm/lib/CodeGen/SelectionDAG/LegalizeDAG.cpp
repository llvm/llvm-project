//===-- LegalizeDAG.cpp - Implement SelectionDAG::Legalize ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::Legalize method.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSubtarget.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool>
ViewLegalizeDAGs("view-legalize-dags", cl::Hidden,
                 cl::desc("Pop up a window to show dags before legalize"));
#else
static const bool ViewLegalizeDAGs = 0;
#endif

//===----------------------------------------------------------------------===//
/// SelectionDAGLegalize - This takes an arbitrary SelectionDAG as input and
/// hacks on it until the target machine can handle it.  This involves
/// eliminating value sizes the machine cannot handle (promoting small sizes to
/// large sizes or splitting up large values into small values) as well as
/// eliminating operations the machine cannot handle.
///
/// This code also does a small amount of optimization and recognition of idioms
/// as part of its processing.  For example, if a target does not support a
/// 'setcc' instruction efficiently, but does support 'brcc' instruction, this
/// will attempt merge setcc and brc instructions into brcc's.
///
namespace {
class VISIBILITY_HIDDEN SelectionDAGLegalize {
  TargetLowering &TLI;
  SelectionDAG &DAG;

  // Libcall insertion helpers.
  
  /// LastCALLSEQ_END - This keeps track of the CALLSEQ_END node that has been
  /// legalized.  We use this to ensure that calls are properly serialized
  /// against each other, including inserted libcalls.
  SDOperand LastCALLSEQ_END;
  
  /// IsLegalizingCall - This member is used *only* for purposes of providing
  /// helpful assertions that a libcall isn't created while another call is 
  /// being legalized (which could lead to non-serialized call sequences).
  bool IsLegalizingCall;
  
  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand      // Try to expand this to other ops, otherwise use a libcall.
  };
  
  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// value type, where the two bits correspond to the LegalizeAction enum.
  /// This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;

  /// LegalizedNodes - For nodes that are of legal width, and that have more
  /// than one use, this map indicates what regularized operand to use.  This
  /// allows us to avoid legalizing the same thing more than once.
  DenseMap<SDOperand, SDOperand> LegalizedNodes;

  /// PromotedNodes - For nodes that are below legal width, and that have more
  /// than one use, this map indicates what promoted value to use.  This allows
  /// us to avoid promoting the same thing more than once.
  DenseMap<SDOperand, SDOperand> PromotedNodes;

  /// ExpandedNodes - For nodes that need to be expanded this map indicates
  /// which which operands are the expanded version of the input.  This allows
  /// us to avoid expanding the same node more than once.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> > ExpandedNodes;

  /// SplitNodes - For vector nodes that need to be split, this map indicates
  /// which which operands are the split version of the input.  This allows us
  /// to avoid splitting the same node more than once.
  std::map<SDOperand, std::pair<SDOperand, SDOperand> > SplitNodes;
  
  /// ScalarizedNodes - For nodes that need to be converted from vector types to
  /// scalar types, this contains the mapping of ones we have already
  /// processed to the result.
  std::map<SDOperand, SDOperand> ScalarizedNodes;
  
  void AddLegalizedOperand(SDOperand From, SDOperand To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }
  void AddPromotedOperand(SDOperand From, SDOperand To) {
    bool isNew = PromotedNodes.insert(std::make_pair(From, To));
    assert(isNew && "Got into the map somehow?");
    // If someone requests legalization of the new node, return itself.
    LegalizedNodes.insert(std::make_pair(To, To));
  }

public:

  SelectionDAGLegalize(SelectionDAG &DAG);

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return (LegalizeAction)ValueTypeActions.getTypeAction(VT);
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT::ValueType VT) const {
    return getTypeAction(VT) == Legal;
  }

  void LegalizeDAG();

private:
  /// HandleOp - Legalize, Promote, or Expand the specified operand as
  /// appropriate for its type.
  void HandleOp(SDOperand Op);
    
  /// LegalizeOp - We know that the specified value has a legal type.
  /// Recursively ensure that the operands have legal types, then return the
  /// result.
  SDOperand LegalizeOp(SDOperand O);
  
  /// UnrollVectorOp - We know that the given vector has a legal type, however
  /// the operation it performs is not legal and is an operation that we have
  /// no way of lowering.  "Unroll" the vector, splitting out the scalars and
  /// operating on each element individually.
  SDOperand UnrollVectorOp(SDOperand O);
  
  /// PerformInsertVectorEltInMemory - Some target cannot handle a variable
  /// insertion index for the INSERT_VECTOR_ELT instruction.  In this case, it
  /// is necessary to spill the vector being inserted into to memory, perform
  /// the insert there, and then read the result back.
  SDOperand PerformInsertVectorEltInMemory(SDOperand Vec, SDOperand Val,
                                           SDOperand Idx);

  /// PromoteOp - Given an operation that produces a value in an invalid type,
  /// promote it to compute the value into a larger type.  The produced value
  /// will have the correct bits for the low portion of the register, but no
  /// guarantee is made about the top bits: it may be zero, sign-extended, or
  /// garbage.
  SDOperand PromoteOp(SDOperand O);

  /// ExpandOp - Expand the specified SDOperand into its two component pieces
  /// Lo&Hi.  Note that the Op MUST be an expanded type.  As a result of this,
  /// the LegalizeNodes map is filled in for any results that are not expanded,
  /// the ExpandedNodes map is filled in for any results that are expanded, and
  /// the Lo/Hi values are returned.   This applies to integer types and Vector
  /// types.
  void ExpandOp(SDOperand O, SDOperand &Lo, SDOperand &Hi);

  /// SplitVectorOp - Given an operand of vector type, break it down into
  /// two smaller values.
  void SplitVectorOp(SDOperand O, SDOperand &Lo, SDOperand &Hi);
  
  /// ScalarizeVectorOp - Given an operand of single-element vector type
  /// (e.g. v1f32), convert it into the equivalent operation that returns a
  /// scalar (e.g. f32) value.
  SDOperand ScalarizeVectorOp(SDOperand O);
  
  /// isShuffleLegal - Return true if a vector shuffle is legal with the
  /// specified mask and type.  Targets can specify exactly which masks they
  /// support and the code generator is tasked with not creating illegal masks.
  ///
  /// Note that this will also return true for shuffles that are promoted to a
  /// different type.
  ///
  /// If this is a legal shuffle, this method returns the (possibly promoted)
  /// build_vector Mask.  If it's not a legal shuffle, it returns null.
  SDNode *isShuffleLegal(MVT::ValueType VT, SDOperand Mask) const;
  
  bool LegalizeAllNodesNotLeadingTo(SDNode *N, SDNode *Dest,
                                    SmallPtrSet<SDNode*, 32> &NodesLeadingTo);

  void LegalizeSetCCOperands(SDOperand &LHS, SDOperand &RHS, SDOperand &CC);
    
  SDOperand ExpandLibCall(RTLIB::Libcall LC, SDNode *Node, bool isSigned,
                          SDOperand &Hi);
  SDOperand ExpandIntToFP(bool isSigned, MVT::ValueType DestTy,
                          SDOperand Source);

  SDOperand EmitStackConvert(SDOperand SrcOp, MVT::ValueType SlotVT, 
                             MVT::ValueType DestVT);
  SDOperand ExpandBUILD_VECTOR(SDNode *Node);
  SDOperand ExpandSCALAR_TO_VECTOR(SDNode *Node);
  SDOperand ExpandLegalINT_TO_FP(bool isSigned,
                                 SDOperand LegalOp,
                                 MVT::ValueType DestVT);
  SDOperand PromoteLegalINT_TO_FP(SDOperand LegalOp, MVT::ValueType DestVT,
                                  bool isSigned);
  SDOperand PromoteLegalFP_TO_INT(SDOperand LegalOp, MVT::ValueType DestVT,
                                  bool isSigned);

  SDOperand ExpandBSWAP(SDOperand Op);
  SDOperand ExpandBitCount(unsigned Opc, SDOperand Op);
  bool ExpandShift(unsigned Opc, SDOperand Op, SDOperand Amt,
                   SDOperand &Lo, SDOperand &Hi);
  void ExpandShiftParts(unsigned NodeOp, SDOperand Op, SDOperand Amt,
                        SDOperand &Lo, SDOperand &Hi);

  SDOperand ExpandEXTRACT_SUBVECTOR(SDOperand Op);
  SDOperand ExpandEXTRACT_VECTOR_ELT(SDOperand Op);
};
}

/// isVectorShuffleLegal - Return true if a vector shuffle is legal with the
/// specified mask and type.  Targets can specify exactly which masks they
/// support and the code generator is tasked with not creating illegal masks.
///
/// Note that this will also return true for shuffles that are promoted to a
/// different type.
SDNode *SelectionDAGLegalize::isShuffleLegal(MVT::ValueType VT, 
                                             SDOperand Mask) const {
  switch (TLI.getOperationAction(ISD::VECTOR_SHUFFLE, VT)) {
  default: return 0;
  case TargetLowering::Legal:
  case TargetLowering::Custom:
    break;
  case TargetLowering::Promote: {
    // If this is promoted to a different type, convert the shuffle mask and
    // ask if it is legal in the promoted type!
    MVT::ValueType NVT = TLI.getTypeToPromoteTo(ISD::VECTOR_SHUFFLE, VT);

    // If we changed # elements, change the shuffle mask.
    unsigned NumEltsGrowth =
      MVT::getVectorNumElements(NVT) / MVT::getVectorNumElements(VT);
    assert(NumEltsGrowth && "Cannot promote to vector type with fewer elts!");
    if (NumEltsGrowth > 1) {
      // Renumber the elements.
      SmallVector<SDOperand, 8> Ops;
      for (unsigned i = 0, e = Mask.getNumOperands(); i != e; ++i) {
        SDOperand InOp = Mask.getOperand(i);
        for (unsigned j = 0; j != NumEltsGrowth; ++j) {
          if (InOp.getOpcode() == ISD::UNDEF)
            Ops.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
          else {
            unsigned InEltNo = cast<ConstantSDNode>(InOp)->getValue();
            Ops.push_back(DAG.getConstant(InEltNo*NumEltsGrowth+j, MVT::i32));
          }
        }
      }
      Mask = DAG.getNode(ISD::BUILD_VECTOR, NVT, &Ops[0], Ops.size());
    }
    VT = NVT;
    break;
  }
  }
  return TLI.isShuffleMaskLegal(Mask, VT) ? Mask.Val : 0;
}

SelectionDAGLegalize::SelectionDAGLegalize(SelectionDAG &dag)
  : TLI(dag.getTargetLoweringInfo()), DAG(dag),
    ValueTypeActions(TLI.getValueTypeActions()) {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");
}

/// ComputeTopDownOrdering - Compute a top-down ordering of the dag, where Order
/// contains all of a nodes operands before it contains the node.
static void ComputeTopDownOrdering(SelectionDAG &DAG,
                                   SmallVector<SDNode*, 64> &Order) {

  DenseMap<SDNode*, unsigned> Visited;
  std::vector<SDNode*> Worklist;
  Worklist.reserve(128);
  
  // Compute ordering from all of the leaves in the graphs, those (like the
  // entry node) that have no operands.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNumOperands() == 0) {
      Visited[I] = 0 - 1U;
      Worklist.push_back(I);
    }
  }
  
  while (!Worklist.empty()) {
    SDNode *N = Worklist.back();
    Worklist.pop_back();
    
    if (++Visited[N] != N->getNumOperands())
      continue;  // Haven't visited all operands yet
    
    Order.push_back(N);

    // Now that we have N in, add anything that uses it if all of their operands
    // are now done.
    for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end();
         UI != E; ++UI)
      Worklist.push_back(UI->getUser());
  }

  assert(Order.size() == Visited.size() &&
         Order.size() == 
         (unsigned)std::distance(DAG.allnodes_begin(), DAG.allnodes_end()) &&
         "Error: DAG is cyclic!");
}


void SelectionDAGLegalize::LegalizeDAG() {
  LastCALLSEQ_END = DAG.getEntryNode();
  IsLegalizingCall = false;
  
  // The legalize process is inherently a bottom-up recursive process (users
  // legalize their uses before themselves).  Given infinite stack space, we
  // could just start legalizing on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, compute an ordering of the nodes where each
  // node is only legalized after all of its operands are legalized.
  SmallVector<SDNode*, 64> Order;
  ComputeTopDownOrdering(DAG, Order);
  
  for (unsigned i = 0, e = Order.size(); i != e; ++i)
    HandleOp(SDOperand(Order[i], 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDOperand OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  ExpandedNodes.clear();
  LegalizedNodes.clear();
  PromotedNodes.clear();
  SplitNodes.clear();
  ScalarizedNodes.clear();

  // Remove dead nodes now.
  DAG.RemoveDeadNodes();
}


/// FindCallEndFromCallStart - Given a chained node that is part of a call
/// sequence, find the CALLSEQ_END node that terminates the call sequence.
static SDNode *FindCallEndFromCallStart(SDNode *Node) {
  if (Node->getOpcode() == ISD::CALLSEQ_END)
    return Node;
  if (Node->use_empty())
    return 0;   // No CallSeqEnd
  
  // The chain is usually at the end.
  SDOperand TheChain(Node, Node->getNumValues()-1);
  if (TheChain.getValueType() != MVT::Other) {
    // Sometimes it's at the beginning.
    TheChain = SDOperand(Node, 0);
    if (TheChain.getValueType() != MVT::Other) {
      // Otherwise, hunt for it.
      for (unsigned i = 1, e = Node->getNumValues(); i != e; ++i)
        if (Node->getValueType(i) == MVT::Other) {
          TheChain = SDOperand(Node, i);
          break;
        }
          
      // Otherwise, we walked into a node without a chain.  
      if (TheChain.getValueType() != MVT::Other)
        return 0;
    }
  }
  
  for (SDNode::use_iterator UI = Node->use_begin(),
       E = Node->use_end(); UI != E; ++UI) {
    
    // Make sure to only follow users of our token chain.
    SDNode *User = UI->getUser();
    for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
      if (User->getOperand(i) == TheChain)
        if (SDNode *Result = FindCallEndFromCallStart(User))
          return Result;
  }
  return 0;
}

/// FindCallStartFromCallEnd - Given a chained node that is part of a call 
/// sequence, find the CALLSEQ_START node that initiates the call sequence.
static SDNode *FindCallStartFromCallEnd(SDNode *Node) {
  assert(Node && "Didn't find callseq_start for a call??");
  if (Node->getOpcode() == ISD::CALLSEQ_START) return Node;
  
  assert(Node->getOperand(0).getValueType() == MVT::Other &&
         "Node doesn't have a token chain argument!");
  return FindCallStartFromCallEnd(Node->getOperand(0).Val);
}

/// LegalizeAllNodesNotLeadingTo - Recursively walk the uses of N, looking to
/// see if any uses can reach Dest.  If no dest operands can get to dest, 
/// legalize them, legalize ourself, and return false, otherwise, return true.
///
/// Keep track of the nodes we fine that actually do lead to Dest in
/// NodesLeadingTo.  This avoids retraversing them exponential number of times.
///
bool SelectionDAGLegalize::LegalizeAllNodesNotLeadingTo(SDNode *N, SDNode *Dest,
                                     SmallPtrSet<SDNode*, 32> &NodesLeadingTo) {
  if (N == Dest) return true;  // N certainly leads to Dest :)
  
  // If we've already processed this node and it does lead to Dest, there is no
  // need to reprocess it.
  if (NodesLeadingTo.count(N)) return true;
  
  // If the first result of this node has been already legalized, then it cannot
  // reach N.
  switch (getTypeAction(N->getValueType(0))) {
  case Legal: 
    if (LegalizedNodes.count(SDOperand(N, 0))) return false;
    break;
  case Promote:
    if (PromotedNodes.count(SDOperand(N, 0))) return false;
    break;
  case Expand:
    if (ExpandedNodes.count(SDOperand(N, 0))) return false;
    break;
  }
  
  // Okay, this node has not already been legalized.  Check and legalize all
  // operands.  If none lead to Dest, then we can legalize this node.
  bool OperandsLeadToDest = false;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    OperandsLeadToDest |=     // If an operand leads to Dest, so do we.
      LegalizeAllNodesNotLeadingTo(N->getOperand(i).Val, Dest, NodesLeadingTo);

  if (OperandsLeadToDest) {
    NodesLeadingTo.insert(N);
    return true;
  }

  // Okay, this node looks safe, legalize it and return false.
  HandleOp(SDOperand(N, 0));
  return false;
}

/// HandleOp - Legalize, Promote, or Expand the specified operand as
/// appropriate for its type.
void SelectionDAGLegalize::HandleOp(SDOperand Op) {
  MVT::ValueType VT = Op.getValueType();
  switch (getTypeAction(VT)) {
  default: assert(0 && "Bad type action!");
  case Legal:   (void)LegalizeOp(Op); break;
  case Promote: (void)PromoteOp(Op); break;
  case Expand:
    if (!MVT::isVector(VT)) {
      // If this is an illegal scalar, expand it into its two component
      // pieces.
      SDOperand X, Y;
      if (Op.getOpcode() == ISD::TargetConstant)
        break;  // Allow illegal target nodes.
      ExpandOp(Op, X, Y);
    } else if (MVT::getVectorNumElements(VT) == 1) {
      // If this is an illegal single element vector, convert it to a
      // scalar operation.
      (void)ScalarizeVectorOp(Op);
    } else {
      // Otherwise, this is an illegal multiple element vector.
      // Split it in half and legalize both parts.
      SDOperand X, Y;
      SplitVectorOp(Op, X, Y);
    }
    break;
  }
}

/// ExpandConstantFP - Expands the ConstantFP node to an integer constant or
/// a load from the constant pool.
static SDOperand ExpandConstantFP(ConstantFPSDNode *CFP, bool UseCP,
                                  SelectionDAG &DAG, TargetLowering &TLI) {
  bool Extend = false;

  // If a FP immediate is precise when represented as a float and if the
  // target can do an extending load from float to double, we put it into
  // the constant pool as a float, even if it's is statically typed as a
  // double.  This shrinks FP constants and canonicalizes them for targets where
  // an FP extending load is the same cost as a normal load (such as on the x87
  // fp stack or PPC FP unit).
  MVT::ValueType VT = CFP->getValueType(0);
  ConstantFP *LLVMC = ConstantFP::get(CFP->getValueAPF());
  if (!UseCP) {
    if (VT!=MVT::f64 && VT!=MVT::f32)
      assert(0 && "Invalid type expansion");
    return DAG.getConstant(LLVMC->getValueAPF().convertToAPInt(),
                           (VT == MVT::f64) ? MVT::i64 : MVT::i32);
  }

  MVT::ValueType OrigVT = VT;
  MVT::ValueType SVT = VT;
  while (SVT != MVT::f32) {
    SVT = (unsigned)SVT - 1;
    if (CFP->isValueValidForType(SVT, CFP->getValueAPF()) &&
        // Only do this if the target has a native EXTLOAD instruction from
        // smaller type.
        TLI.isLoadXLegal(ISD::EXTLOAD, SVT) &&
        TLI.ShouldShrinkFPConstant(OrigVT)) {
      const Type *SType = MVT::getTypeForValueType(SVT);
      LLVMC = cast<ConstantFP>(ConstantExpr::getFPTrunc(LLVMC, SType));
      VT = SVT;
      Extend = true;
    }
  }

  SDOperand CPIdx = DAG.getConstantPool(LLVMC, TLI.getPointerTy());
  if (Extend)
    return DAG.getExtLoad(ISD::EXTLOAD, OrigVT, DAG.getEntryNode(),
                          CPIdx, PseudoSourceValue::getConstantPool(),
                          0, VT);
  return DAG.getLoad(OrigVT, DAG.getEntryNode(), CPIdx,
                     PseudoSourceValue::getConstantPool(), 0);
}


/// ExpandFCOPYSIGNToBitwiseOps - Expands fcopysign to a series of bitwise
/// operations.
static
SDOperand ExpandFCOPYSIGNToBitwiseOps(SDNode *Node, MVT::ValueType NVT,
                                      SelectionDAG &DAG, TargetLowering &TLI) {
  MVT::ValueType VT = Node->getValueType(0);
  MVT::ValueType SrcVT = Node->getOperand(1).getValueType();
  assert((SrcVT == MVT::f32 || SrcVT == MVT::f64) &&
         "fcopysign expansion only supported for f32 and f64");
  MVT::ValueType SrcNVT = (SrcVT == MVT::f64) ? MVT::i64 : MVT::i32;

  // First get the sign bit of second operand.
  SDOperand Mask1 = (SrcVT == MVT::f64)
    ? DAG.getConstantFP(BitsToDouble(1ULL << 63), SrcVT)
    : DAG.getConstantFP(BitsToFloat(1U << 31), SrcVT);
  Mask1 = DAG.getNode(ISD::BIT_CONVERT, SrcNVT, Mask1);
  SDOperand SignBit= DAG.getNode(ISD::BIT_CONVERT, SrcNVT, Node->getOperand(1));
  SignBit = DAG.getNode(ISD::AND, SrcNVT, SignBit, Mask1);
  // Shift right or sign-extend it if the two operands have different types.
  int SizeDiff = MVT::getSizeInBits(SrcNVT) - MVT::getSizeInBits(NVT);
  if (SizeDiff > 0) {
    SignBit = DAG.getNode(ISD::SRL, SrcNVT, SignBit,
                          DAG.getConstant(SizeDiff, TLI.getShiftAmountTy()));
    SignBit = DAG.getNode(ISD::TRUNCATE, NVT, SignBit);
  } else if (SizeDiff < 0)
    SignBit = DAG.getNode(ISD::SIGN_EXTEND, NVT, SignBit);

  // Clear the sign bit of first operand.
  SDOperand Mask2 = (VT == MVT::f64)
    ? DAG.getConstantFP(BitsToDouble(~(1ULL << 63)), VT)
    : DAG.getConstantFP(BitsToFloat(~(1U << 31)), VT);
  Mask2 = DAG.getNode(ISD::BIT_CONVERT, NVT, Mask2);
  SDOperand Result = DAG.getNode(ISD::BIT_CONVERT, NVT, Node->getOperand(0));
  Result = DAG.getNode(ISD::AND, NVT, Result, Mask2);

  // Or the value with the sign bit.
  Result = DAG.getNode(ISD::OR, NVT, Result, SignBit);
  return Result;
}

/// ExpandUnalignedStore - Expands an unaligned store to 2 half-size stores.
static
SDOperand ExpandUnalignedStore(StoreSDNode *ST, SelectionDAG &DAG,
                               TargetLowering &TLI) {
  SDOperand Chain = ST->getChain();
  SDOperand Ptr = ST->getBasePtr();
  SDOperand Val = ST->getValue();
  MVT::ValueType VT = Val.getValueType();
  int Alignment = ST->getAlignment();
  int SVOffset = ST->getSrcValueOffset();
  if (MVT::isFloatingPoint(ST->getMemoryVT()) || 
      MVT::isVector(ST->getMemoryVT())) {
    // Expand to a bitconvert of the value to the integer type of the 
    // same size, then a (misaligned) int store.
    MVT::ValueType intVT;
    if (MVT::is128BitVector(VT) || VT == MVT::ppcf128 || VT == MVT::f128)
      intVT = MVT::i128;
    else if (MVT::is64BitVector(VT) || VT==MVT::f64)
      intVT = MVT::i64;
    else if (VT==MVT::f32)
      intVT = MVT::i32;
    else
      assert(0 && "Unaligned store of unsupported type");

    SDOperand Result = DAG.getNode(ISD::BIT_CONVERT, intVT, Val);
    return DAG.getStore(Chain, Result, Ptr, ST->getSrcValue(),
                        SVOffset, ST->isVolatile(), Alignment);
  }
  assert(MVT::isInteger(ST->getMemoryVT()) &&
         !MVT::isVector(ST->getMemoryVT()) &&
         "Unaligned store of unknown type.");
  // Get the half-size VT
  MVT::ValueType NewStoredVT = ST->getMemoryVT() - 1;
  int NumBits = MVT::getSizeInBits(NewStoredVT);
  int IncrementSize = NumBits / 8;

  // Divide the stored value in two parts.
  SDOperand ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDOperand Lo = Val;
  SDOperand Hi = DAG.getNode(ISD::SRL, VT, Val, ShiftAmount);

  // Store the two parts
  SDOperand Store1, Store2;
  Store1 = DAG.getTruncStore(Chain, TLI.isLittleEndian()?Lo:Hi, Ptr,
                             ST->getSrcValue(), SVOffset, NewStoredVT,
                             ST->isVolatile(), Alignment);
  Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                    DAG.getConstant(IncrementSize, TLI.getPointerTy()));
  Alignment = MinAlign(Alignment, IncrementSize);
  Store2 = DAG.getTruncStore(Chain, TLI.isLittleEndian()?Hi:Lo, Ptr,
                             ST->getSrcValue(), SVOffset + IncrementSize,
                             NewStoredVT, ST->isVolatile(), Alignment);

  return DAG.getNode(ISD::TokenFactor, MVT::Other, Store1, Store2);
}

/// ExpandUnalignedLoad - Expands an unaligned load to 2 half-size loads.
static
SDOperand ExpandUnalignedLoad(LoadSDNode *LD, SelectionDAG &DAG,
                              TargetLowering &TLI) {
  int SVOffset = LD->getSrcValueOffset();
  SDOperand Chain = LD->getChain();
  SDOperand Ptr = LD->getBasePtr();
  MVT::ValueType VT = LD->getValueType(0);
  MVT::ValueType LoadedVT = LD->getMemoryVT();
  if (MVT::isFloatingPoint(VT) || MVT::isVector(VT)) {
    // Expand to a (misaligned) integer load of the same size,
    // then bitconvert to floating point or vector.
    MVT::ValueType intVT;
    if (MVT::is128BitVector(LoadedVT) || 
         LoadedVT == MVT::ppcf128 || LoadedVT == MVT::f128)
      intVT = MVT::i128;
    else if (MVT::is64BitVector(LoadedVT) || LoadedVT == MVT::f64)
      intVT = MVT::i64;
    else if (LoadedVT == MVT::f32)
      intVT = MVT::i32;
    else
      assert(0 && "Unaligned load of unsupported type");

    SDOperand newLoad = DAG.getLoad(intVT, Chain, Ptr, LD->getSrcValue(),
                                    SVOffset, LD->isVolatile(), 
                                    LD->getAlignment());
    SDOperand Result = DAG.getNode(ISD::BIT_CONVERT, LoadedVT, newLoad);
    if (MVT::isFloatingPoint(VT) && LoadedVT != VT)
      Result = DAG.getNode(ISD::FP_EXTEND, VT, Result);

    SDOperand Ops[] = { Result, Chain };
    return DAG.getNode(ISD::MERGE_VALUES, DAG.getVTList(VT, MVT::Other), 
                       Ops, 2);
  }
  assert(MVT::isInteger(LoadedVT) && !MVT::isVector(LoadedVT) &&
         "Unaligned load of unsupported type.");

  // Compute the new VT that is half the size of the old one.  This is an
  // integer MVT.
  unsigned NumBits = MVT::getSizeInBits(LoadedVT);
  MVT::ValueType NewLoadedVT;
  NewLoadedVT = MVT::getIntegerType(NumBits/2);
  NumBits >>= 1;
  
  unsigned Alignment = LD->getAlignment();
  unsigned IncrementSize = NumBits / 8;
  ISD::LoadExtType HiExtType = LD->getExtensionType();

  // If the original load is NON_EXTLOAD, the hi part load must be ZEXTLOAD.
  if (HiExtType == ISD::NON_EXTLOAD)
    HiExtType = ISD::ZEXTLOAD;

  // Load the value in two parts
  SDOperand Lo, Hi;
  if (TLI.isLittleEndian()) {
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset, NewLoadedVT, LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Hi = DAG.getExtLoad(HiExtType, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  } else {
    Hi = DAG.getExtLoad(HiExtType, VT, Chain, Ptr, LD->getSrcValue(), SVOffset,
                        NewLoadedVT,LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  }

  // aggregate the two parts
  SDOperand ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDOperand Result = DAG.getNode(ISD::SHL, VT, Hi, ShiftAmount);
  Result = DAG.getNode(ISD::OR, VT, Result, Lo);

  SDOperand TF = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                             Hi.getValue(1));

  SDOperand Ops[] = { Result, TF };
  return DAG.getNode(ISD::MERGE_VALUES, DAG.getVTList(VT, MVT::Other), Ops, 2);
}

/// UnrollVectorOp - We know that the given vector has a legal type, however
/// the operation it performs is not legal and is an operation that we have
/// no way of lowering.  "Unroll" the vector, splitting out the scalars and
/// operating on each element individually.
SDOperand SelectionDAGLegalize::UnrollVectorOp(SDOperand Op) {
  MVT::ValueType VT = Op.getValueType();
  assert(isTypeLegal(VT) &&
         "Caller should expand or promote operands that are not legal!");
  assert(Op.Val->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");
  unsigned NE = MVT::getVectorNumElements(VT);
  MVT::ValueType EltVT = MVT::getVectorElementType(VT);

  SmallVector<SDOperand, 8> Scalars;
  SmallVector<SDOperand, 4> Operands(Op.getNumOperands());
  for (unsigned i = 0; i != NE; ++i) {
    for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
      SDOperand Operand = Op.getOperand(j);
      MVT::ValueType OperandVT = Operand.getValueType();
      if (MVT::isVector(OperandVT)) {
        // A vector operand; extract a single element.
        MVT::ValueType OperandEltVT = MVT::getVectorElementType(OperandVT);
        Operands[j] = DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
                                  OperandEltVT,
                                  Operand,
                                  DAG.getConstant(i, MVT::i32));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }
    Scalars.push_back(DAG.getNode(Op.getOpcode(), EltVT,
                                  &Operands[0], Operands.size()));
  }

  return DAG.getNode(ISD::BUILD_VECTOR, VT, &Scalars[0], Scalars.size());
}

/// GetFPLibCall - Return the right libcall for the given floating point type.
static RTLIB::Libcall GetFPLibCall(MVT::ValueType VT,
                                   RTLIB::Libcall Call_F32,
                                   RTLIB::Libcall Call_F64,
                                   RTLIB::Libcall Call_F80,
                                   RTLIB::Libcall Call_PPCF128) {
  return
    VT == MVT::f32 ? Call_F32 :
    VT == MVT::f64 ? Call_F64 :
    VT == MVT::f80 ? Call_F80 :
    VT == MVT::ppcf128 ? Call_PPCF128 :
    RTLIB::UNKNOWN_LIBCALL;
}

/// PerformInsertVectorEltInMemory - Some target cannot handle a variable
/// insertion index for the INSERT_VECTOR_ELT instruction.  In this case, it
/// is necessary to spill the vector being inserted into to memory, perform
/// the insert there, and then read the result back.
SDOperand SelectionDAGLegalize::
PerformInsertVectorEltInMemory(SDOperand Vec, SDOperand Val, SDOperand Idx) {
  SDOperand Tmp1 = Vec;
  SDOperand Tmp2 = Val;
  SDOperand Tmp3 = Idx;
  
  // If the target doesn't support this, we have to spill the input vector
  // to a temporary stack slot, update the element, then reload it.  This is
  // badness.  We could also load the value into a vector register (either
  // with a "move to register" or "extload into register" instruction, then
  // permute it into place, if the idx is a constant and if the idx is
  // supported by the target.
  MVT::ValueType VT    = Tmp1.getValueType();
  MVT::ValueType EltVT = MVT::getVectorElementType(VT);
  MVT::ValueType IdxVT = Tmp3.getValueType();
  MVT::ValueType PtrVT = TLI.getPointerTy();
  SDOperand StackPtr = DAG.CreateStackTemporary(VT);

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(StackPtr.Val);
  int SPFI = StackPtrFI->getIndex();

  // Store the vector.
  SDOperand Ch = DAG.getStore(DAG.getEntryNode(), Tmp1, StackPtr,
                              PseudoSourceValue::getFixedStack(),
                              SPFI);

  // Truncate or zero extend offset to target pointer type.
  unsigned CastOpc = (IdxVT > PtrVT) ? ISD::TRUNCATE : ISD::ZERO_EXTEND;
  Tmp3 = DAG.getNode(CastOpc, PtrVT, Tmp3);
  // Add the offset to the index.
  unsigned EltSize = MVT::getSizeInBits(EltVT)/8;
  Tmp3 = DAG.getNode(ISD::MUL, IdxVT, Tmp3,DAG.getConstant(EltSize, IdxVT));
  SDOperand StackPtr2 = DAG.getNode(ISD::ADD, IdxVT, Tmp3, StackPtr);
  // Store the scalar value.
  Ch = DAG.getTruncStore(Ch, Tmp2, StackPtr2,
                         PseudoSourceValue::getFixedStack(), SPFI, EltVT);
  // Load the updated vector.
  return DAG.getLoad(VT, Ch, StackPtr, PseudoSourceValue::getFixedStack(),SPFI);
}

/// LegalizeOp - We know that the specified value has a legal type, and
/// that its operands are legal.  Now ensure that the operation itself
/// is legal, recursively ensuring that the operands' operations remain
/// legal.
SDOperand SelectionDAGLegalize::LegalizeOp(SDOperand Op) {
  if (Op.getOpcode() == ISD::TargetConstant) // Allow illegal target nodes.
    return Op;
  
  assert(isTypeLegal(Op.getValueType()) &&
         "Caller should expand or promote operands that are not legal!");
  SDNode *Node = Op.Val;

  // If this operation defines any values that cannot be represented in a
  // register on this target, make sure to expand or promote them.
  if (Node->getNumValues() > 1) {
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      if (getTypeAction(Node->getValueType(i)) != Legal) {
        HandleOp(Op.getValue(i));
        assert(LegalizedNodes.count(Op) &&
               "Handling didn't add legal operands!");
        return LegalizedNodes[Op];
      }
  }

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  DenseMap<SDOperand, SDOperand>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDOperand Tmp1, Tmp2, Tmp3, Tmp4;
  SDOperand Result = Op;
  bool isCustom = false;
  
  switch (Node->getOpcode()) {
  case ISD::FrameIndex:
  case ISD::EntryToken:
  case ISD::Register:
  case ISD::BasicBlock:
  case ISD::TargetFrameIndex:
  case ISD::TargetJumpTable:
  case ISD::TargetConstant:
  case ISD::TargetConstantFP:
  case ISD::TargetConstantPool:
  case ISD::TargetGlobalAddress:
  case ISD::TargetGlobalTLSAddress:
  case ISD::TargetExternalSymbol:
  case ISD::VALUETYPE:
  case ISD::SRCVALUE:
  case ISD::MEMOPERAND:
  case ISD::STRING:
  case ISD::CONDCODE:
  case ISD::ARG_FLAGS:
    // Primitives must all be legal.
    assert(TLI.isOperationLegal(Node->getOpcode(), Node->getValueType(0)) &&
           "This must be legal!");
    break;
  default:
    if (Node->getOpcode() >= ISD::BUILTIN_OP_END) {
      // If this is a target node, legalize it by legalizing the operands then
      // passing it through.
      SmallVector<SDOperand, 8> Ops;
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
        Ops.push_back(LegalizeOp(Node->getOperand(i)));

      Result = DAG.UpdateNodeOperands(Result.getValue(0), &Ops[0], Ops.size());

      for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
        AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
      return Result.getValue(Op.ResNo);
    }
    // Otherwise this is an unhandled builtin node.  splat.
#ifndef NDEBUG
    cerr << "NODE: "; Node->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to legalize this operator!");
    abort();
  case ISD::GLOBAL_OFFSET_TABLE:
  case ISD::GlobalAddress:
  case ISD::GlobalTLSAddress:
  case ISD::ExternalSymbol:
  case ISD::ConstantPool:
  case ISD::JumpTable: // Nothing to do.
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Op, DAG);
      if (Tmp1.Val) Result = Tmp1;
      // FALLTHROUGH if the target doesn't want to lower this op after all.
    case TargetLowering::Legal:
      break;
    }
    break;
  case ISD::FRAMEADDR:
  case ISD::RETURNADDR:
    // The only option for these nodes is to custom lower them.  If the target
    // does not custom lower them, then return zero.
    Tmp1 = TLI.LowerOperation(Op, DAG);
    if (Tmp1.Val) 
      Result = Tmp1;
    else
      Result = DAG.getConstant(0, TLI.getPointerTy());
    break;
  case ISD::FRAME_TO_ARGS_OFFSET: {
    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Legal:
      Result = DAG.getConstant(0, VT);
      break;
    }
    }
    break;
  case ISD::EXCEPTIONADDR: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
        unsigned Reg = TLI.getExceptionAddressRegister();
        Result = DAG.getCopyFromReg(Tmp1, Reg, VT);
      }
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Legal: {
      SDOperand Ops[] = { DAG.getConstant(0, VT), Tmp1 };
      Result = DAG.getNode(ISD::MERGE_VALUES, DAG.getVTList(VT, MVT::Other),
                           Ops, 2);
      break;
    }
    }
    }
    if (Result.Val->getNumValues() == 1) break;

    assert(Result.Val->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since we produced two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result);
    Tmp2 = LegalizeOp(Result.getValue(1));
    AddLegalizedOperand(Op.getValue(0), Tmp1);
    AddLegalizedOperand(Op.getValue(1), Tmp2);
    return Op.ResNo ? Tmp2 : Tmp1;
  case ISD::EHSELECTION: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
        unsigned Reg = TLI.getExceptionSelectorRegister();
        Result = DAG.getCopyFromReg(Tmp2, Reg, VT);
      }
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Legal: {
      SDOperand Ops[] = { DAG.getConstant(0, VT), Tmp2 };
      Result = DAG.getNode(ISD::MERGE_VALUES, DAG.getVTList(VT, MVT::Other),
                           Ops, 2);
      break;
    }
    }
    }
    if (Result.Val->getNumValues() == 1) break;

    assert(Result.Val->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since we produced two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result);
    Tmp2 = LegalizeOp(Result.getValue(1));
    AddLegalizedOperand(Op.getValue(0), Tmp1);
    AddLegalizedOperand(Op.getValue(1), Tmp2);
    return Op.ResNo ? Tmp2 : Tmp1;
  case ISD::EH_RETURN: {
    MVT::ValueType VT = Node->getValueType(0);
    // The only "good" option for this node is to custom lower it.
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported at all!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Legal:
      // Target does not know, how to lower this, lower to noop
      Result = LegalizeOp(Node->getOperand(0));
      break;
    }
    }
    break;
  case ISD::AssertSext:
  case ISD::AssertZext:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
    break;
  case ISD::MERGE_VALUES:
    // Legalize eliminates MERGE_VALUES nodes.
    Result = Node->getOperand(Op.ResNo);
    break;
  case ISD::CopyFromReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Result = Op.getValue(0);
    if (Node->getNumValues() == 2) {
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
    } else {
      assert(Node->getNumValues() == 3 && "Invalid copyfromreg!");
      if (Node->getNumOperands() == 3) {
        Tmp2 = LegalizeOp(Node->getOperand(2));
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1),Tmp2);
      } else {
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
      }
      AddLegalizedOperand(Op.getValue(2), Result.getValue(2));
    }
    // Since CopyFromReg produces two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(Op.getValue(0), Result);
    AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  case ISD::UNDEF: {
    MVT::ValueType VT = Op.getValueType();
    switch (TLI.getOperationAction(ISD::UNDEF, VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand:
      if (MVT::isInteger(VT))
        Result = DAG.getConstant(0, VT);
      else if (MVT::isFloatingPoint(VT))
        Result = DAG.getConstantFP(APFloat(APInt(MVT::getSizeInBits(VT), 0)),
                                   VT);
      else
        assert(0 && "Unknown value type!");
      break;
    case TargetLowering::Legal:
      break;
    }
    break;
  }
    
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID: {
    SmallVector<SDOperand, 8> Ops;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
    Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
    
    // Allow the target to custom lower its intrinsics if it wants to.
    if (TLI.getOperationAction(Node->getOpcode(), MVT::Other) == 
        TargetLowering::Custom) {
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) Result = Tmp3;
    }

    if (Result.Val->getNumValues() == 1) break;

    // Must have return value and chain result.
    assert(Result.Val->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since loads produce two values, make sure to remember that we 
    // legalized both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  }    

  case ISD::LOCATION:
    assert(Node->getNumOperands() == 5 && "Invalid LOCATION node!");
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the input chain.
    
    switch (TLI.getOperationAction(ISD::LOCATION, MVT::Other)) {
    case TargetLowering::Promote:
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
      MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
      bool useDEBUG_LOC = TLI.isOperationLegal(ISD::DEBUG_LOC, MVT::Other);
      bool useLABEL = TLI.isOperationLegal(ISD::LABEL, MVT::Other);
      
      if (MMI && (useDEBUG_LOC || useLABEL)) {
        const std::string &FName =
          cast<StringSDNode>(Node->getOperand(3))->getValue();
        const std::string &DirName = 
          cast<StringSDNode>(Node->getOperand(4))->getValue();
        unsigned SrcFile = MMI->RecordSource(DirName, FName);

        SmallVector<SDOperand, 8> Ops;
        Ops.push_back(Tmp1);  // chain
        SDOperand LineOp = Node->getOperand(1);
        SDOperand ColOp = Node->getOperand(2);
        
        if (useDEBUG_LOC) {
          Ops.push_back(LineOp);  // line #
          Ops.push_back(ColOp);  // col #
          Ops.push_back(DAG.getConstant(SrcFile, MVT::i32));  // source file id
          Result = DAG.getNode(ISD::DEBUG_LOC, MVT::Other, &Ops[0], Ops.size());
        } else {
          unsigned Line = cast<ConstantSDNode>(LineOp)->getValue();
          unsigned Col = cast<ConstantSDNode>(ColOp)->getValue();
          unsigned ID = MMI->RecordSourceLine(Line, Col, SrcFile);
          Ops.push_back(DAG.getConstant(ID, MVT::i32));
          Ops.push_back(DAG.getConstant(0, MVT::i32)); // a debug label
          Result = DAG.getNode(ISD::LABEL, MVT::Other, &Ops[0], Ops.size());
        }
      } else {
        Result = Tmp1;  // chain
      }
      break;
    }
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) ||
          getTypeAction(Node->getOperand(1).getValueType()) == Promote) {
        SmallVector<SDOperand, 8> Ops;
        Ops.push_back(Tmp1);
        if (getTypeAction(Node->getOperand(1).getValueType()) == Legal) {
          Ops.push_back(Node->getOperand(1));  // line # must be legal.
          Ops.push_back(Node->getOperand(2));  // col # must be legal.
        } else {
          // Otherwise promote them.
          Ops.push_back(PromoteOp(Node->getOperand(1)));
          Ops.push_back(PromoteOp(Node->getOperand(2)));
        }
        Ops.push_back(Node->getOperand(3));  // filename must be legal.
        Ops.push_back(Node->getOperand(4));  // working dir # must be legal.
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
      break;
    }
    break;

  case ISD::DECLARE:
    assert(Node->getNumOperands() == 3 && "Invalid DECLARE node!");
    switch (TLI.getOperationAction(ISD::DECLARE, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the address.
      Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the variable.
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      break;
    case TargetLowering::Expand:
      Result = LegalizeOp(Node->getOperand(0));
      break;
    }
    break;    
    
  case ISD::DEBUG_LOC:
    assert(Node->getNumOperands() == 4 && "Invalid DEBUG_LOC node!");
    switch (TLI.getOperationAction(ISD::DEBUG_LOC, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the line #.
      Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the col #.
      Tmp4 = LegalizeOp(Node->getOperand(3));  // Legalize the source file id.
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3, Tmp4);
      break;
    }
    break;    

  case ISD::LABEL:
    assert(Node->getNumOperands() == 3 && "Invalid LABEL node!");
    switch (TLI.getOperationAction(ISD::LABEL, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the label id.
      Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the "flavor" operand.
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      break;
    case TargetLowering::Expand:
      Result = LegalizeOp(Node->getOperand(0));
      break;
    }
    break;

  case ISD::PREFETCH:
    assert(Node->getNumOperands() == 4 && "Invalid Prefetch node!");
    switch (TLI.getOperationAction(ISD::PREFETCH, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the address.
      Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the rw specifier.
      Tmp4 = LegalizeOp(Node->getOperand(3));  // Legalize locality specifier.
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3, Tmp4);
      break;
    case TargetLowering::Expand:
      // It's a noop.
      Result = LegalizeOp(Node->getOperand(0));
      break;
    }
    break;

  case ISD::MEMBARRIER: {
    assert(Node->getNumOperands() == 6 && "Invalid MemBarrier node!");
    switch (TLI.getOperationAction(ISD::MEMBARRIER, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: {
      SDOperand Ops[6];
      Ops[0] = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      for (int x = 1; x < 6; ++x) {
        Ops[x] = Node->getOperand(x);
        if (!isTypeLegal(Ops[x].getValueType()))
          Ops[x] = PromoteOp(Ops[x]);
      }
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], 6);
      break;
    }
    case TargetLowering::Expand:
      //There is no libgcc call for this op
      Result = Node->getOperand(0);  // Noop
    break;
    }
    break;
  }

  case ISD::ATOMIC_LCS: {
    unsigned int num_operands = 4;
    assert(Node->getNumOperands() == num_operands && "Invalid Atomic node!");
    SDOperand Ops[4];
    for (unsigned int x = 0; x < num_operands; ++x)
      Ops[x] = LegalizeOp(Node->getOperand(x));
    Result = DAG.UpdateNodeOperands(Result, &Ops[0], num_operands);
    
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Custom:
        Result = TLI.LowerOperation(Result, DAG);
        break;
      case TargetLowering::Legal:
        break;
    }
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  }      
  case ISD::ATOMIC_LAS:
  case ISD::ATOMIC_LSS:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP: {
    unsigned int num_operands = 3;
    assert(Node->getNumOperands() == num_operands && "Invalid Atomic node!");
    SDOperand Ops[3];
    for (unsigned int x = 0; x < num_operands; ++x)
      Ops[x] = LegalizeOp(Node->getOperand(x));
    Result = DAG.UpdateNodeOperands(Result, &Ops[0], num_operands);
    
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Result, DAG);
      break;
    case TargetLowering::Expand:
      Result = SDOperand(TLI.ExpandOperationResult(Op.Val, DAG),0);
      break;
    case TargetLowering::Legal:
      break;
    }
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  }      
  case ISD::Constant: {
    ConstantSDNode *CN = cast<ConstantSDNode>(Node);
    unsigned opAction =
      TLI.getOperationAction(ISD::Constant, CN->getValueType(0));

    // We know we don't need to expand constants here, constants only have one
    // value and we check that it is fine above.

    if (opAction == TargetLowering::Custom) {
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val)
        Result = Tmp1;
    }
    break;
  }
  case ISD::ConstantFP: {
    // Spill FP immediates to the constant pool if the target cannot directly
    // codegen them.  Targets often have some immediate values that can be
    // efficiently generated into an FP register without a load.  We explicitly
    // leave these constants as ConstantFP nodes for the target to deal with.
    ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Node);

    switch (TLI.getOperationAction(ISD::ConstantFP, CFP->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      // Check to see if this FP immediate is already legal.
      bool isLegal = false;
      for (TargetLowering::legal_fpimm_iterator I = TLI.legal_fpimm_begin(),
             E = TLI.legal_fpimm_end(); I != E; ++I) {
        if (CFP->isExactlyValue(*I)) {
          isLegal = true;
          break;
        }
      }
      // If this is a legal constant, turn it into a TargetConstantFP node.
      if (isLegal)
        break;
      Result = ExpandConstantFP(CFP, true, DAG, TLI);
    }
    }
    break;
  }
  case ISD::TokenFactor:
    if (Node->getNumOperands() == 2) {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Tmp2 = LegalizeOp(Node->getOperand(1));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    } else if (Node->getNumOperands() == 3) {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Tmp2 = LegalizeOp(Node->getOperand(1));
      Tmp3 = LegalizeOp(Node->getOperand(2));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    } else {
      SmallVector<SDOperand, 8> Ops;
      // Legalize the operands.
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
        Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
    }
    break;
    
  case ISD::FORMAL_ARGUMENTS:
  case ISD::CALL:
    // The only option for this is to custom lower it.
    Tmp3 = TLI.LowerOperation(Result.getValue(0), DAG);
    assert(Tmp3.Val && "Target didn't custom lower this node!");
    // A call within a calling sequence must be legalized to something
    // other than the normal CALLSEQ_END.  Violating this gets Legalize
    // into an infinite loop.
    assert ((!IsLegalizingCall ||
             Node->getOpcode() != ISD::CALL ||
             Tmp3.Val->getOpcode() != ISD::CALLSEQ_END) &&
            "Nested CALLSEQ_START..CALLSEQ_END not supported.");

    // The number of incoming and outgoing values should match; unless the final
    // outgoing value is a flag.
    assert((Tmp3.Val->getNumValues() == Result.Val->getNumValues() ||
            (Tmp3.Val->getNumValues() == Result.Val->getNumValues() + 1 &&
             Tmp3.Val->getValueType(Tmp3.Val->getNumValues() - 1) ==
               MVT::Flag)) &&
           "Lowering call/formal_arguments produced unexpected # results!");
    
    // Since CALL/FORMAL_ARGUMENTS nodes produce multiple values, make sure to
    // remember that we legalized all of them, so it doesn't get relegalized.
    for (unsigned i = 0, e = Tmp3.Val->getNumValues(); i != e; ++i) {
      if (Tmp3.Val->getValueType(i) == MVT::Flag)
        continue;
      Tmp1 = LegalizeOp(Tmp3.getValue(i));
      if (Op.ResNo == i)
        Tmp2 = Tmp1;
      AddLegalizedOperand(SDOperand(Node, i), Tmp1);
    }
    return Tmp2;
   case ISD::EXTRACT_SUBREG: {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      ConstantSDNode *idx = dyn_cast<ConstantSDNode>(Node->getOperand(1));
      assert(idx && "Operand must be a constant");
      Tmp2 = DAG.getTargetConstant(idx->getValue(), idx->getValueType(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    }
    break;
  case ISD::INSERT_SUBREG: {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Tmp2 = LegalizeOp(Node->getOperand(1));      
      ConstantSDNode *idx = dyn_cast<ConstantSDNode>(Node->getOperand(2));
      assert(idx && "Operand must be a constant");
      Tmp3 = DAG.getTargetConstant(idx->getValue(), idx->getValueType(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    }
    break;      
  case ISD::BUILD_VECTOR:
    switch (TLI.getOperationAction(ISD::BUILD_VECTOR, Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand:
      Result = ExpandBUILD_VECTOR(Result.Val);
      break;
    }
    break;
  case ISD::INSERT_VECTOR_ELT:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // InVec
    Tmp3 = LegalizeOp(Node->getOperand(2));  // InEltNo

    // The type of the value to insert may not be legal, even though the vector
    // type is legal.  Legalize/Promote accordingly.  We do not handle Expand
    // here.
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    default: assert(0 && "Cannot expand insert element operand");
    case Legal:   Tmp2 = LegalizeOp(Node->getOperand(1)); break;
    case Promote: Tmp2 = PromoteOp(Node->getOperand(1));  break;
    }
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    
    switch (TLI.getOperationAction(ISD::INSERT_VECTOR_ELT,
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp4 = TLI.LowerOperation(Result, DAG);
      if (Tmp4.Val) {
        Result = Tmp4;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      // If the insert index is a constant, codegen this as a scalar_to_vector,
      // then a shuffle that inserts it into the right position in the vector.
      if (ConstantSDNode *InsertPos = dyn_cast<ConstantSDNode>(Tmp3)) {
        // SCALAR_TO_VECTOR requires that the type of the value being inserted
        // match the element type of the vector being created.
        if (Tmp2.getValueType() == 
            MVT::getVectorElementType(Op.getValueType())) {
          SDOperand ScVec = DAG.getNode(ISD::SCALAR_TO_VECTOR, 
                                        Tmp1.getValueType(), Tmp2);
          
          unsigned NumElts = MVT::getVectorNumElements(Tmp1.getValueType());
          MVT::ValueType ShufMaskVT = MVT::getIntVectorWithNumElements(NumElts);
          MVT::ValueType ShufMaskEltVT = MVT::getVectorElementType(ShufMaskVT);
          
          // We generate a shuffle of InVec and ScVec, so the shuffle mask
          // should be 0,1,2,3,4,5... with the appropriate element replaced with
          // elt 0 of the RHS.
          SmallVector<SDOperand, 8> ShufOps;
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertPos->getValue())
              ShufOps.push_back(DAG.getConstant(i, ShufMaskEltVT));
            else
              ShufOps.push_back(DAG.getConstant(NumElts, ShufMaskEltVT));
          }
          SDOperand ShufMask = DAG.getNode(ISD::BUILD_VECTOR, ShufMaskVT,
                                           &ShufOps[0], ShufOps.size());
          
          Result = DAG.getNode(ISD::VECTOR_SHUFFLE, Tmp1.getValueType(),
                               Tmp1, ScVec, ShufMask);
          Result = LegalizeOp(Result);
          break;
        }
      }
      Result = PerformInsertVectorEltInMemory(Tmp1, Tmp2, Tmp3);
      break;
    }
    }
    break;
  case ISD::SCALAR_TO_VECTOR:
    if (!TLI.isTypeLegal(Node->getOperand(0).getValueType())) {
      Result = LegalizeOp(ExpandSCALAR_TO_VECTOR(Node));
      break;
    }
    
    Tmp1 = LegalizeOp(Node->getOperand(0));  // InVal
    Result = DAG.UpdateNodeOperands(Result, Tmp1);
    switch (TLI.getOperationAction(ISD::SCALAR_TO_VECTOR,
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand:
      Result = LegalizeOp(ExpandSCALAR_TO_VECTOR(Node));
      break;
    }
    break;
  case ISD::VECTOR_SHUFFLE:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Legalize the input vectors,
    Tmp2 = LegalizeOp(Node->getOperand(1));   // but not the shuffle mask.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));

    // Allow targets to custom lower the SHUFFLEs they support.
    switch (TLI.getOperationAction(ISD::VECTOR_SHUFFLE,Result.getValueType())) {
    default: assert(0 && "Unknown operation action!");
    case TargetLowering::Legal:
      assert(isShuffleLegal(Result.getValueType(), Node->getOperand(2)) &&
             "vector shuffle should not be created if not legal!");
      break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      MVT::ValueType VT = Node->getValueType(0);
      MVT::ValueType EltVT = MVT::getVectorElementType(VT);
      MVT::ValueType PtrVT = TLI.getPointerTy();
      SDOperand Mask = Node->getOperand(2);
      unsigned NumElems = Mask.getNumOperands();
      SmallVector<SDOperand,8> Ops;
      for (unsigned i = 0; i != NumElems; ++i) {
        SDOperand Arg = Mask.getOperand(i);
        if (Arg.getOpcode() == ISD::UNDEF) {
          Ops.push_back(DAG.getNode(ISD::UNDEF, EltVT));
        } else {
          assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
          unsigned Idx = cast<ConstantSDNode>(Arg)->getValue();
          if (Idx < NumElems)
            Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, EltVT, Tmp1,
                                      DAG.getConstant(Idx, PtrVT)));
          else
            Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, EltVT, Tmp2,
                                      DAG.getConstant(Idx - NumElems, PtrVT)));
        }
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
      break;
    }
    case TargetLowering::Promote: {
      // Change base type to a different vector type.
      MVT::ValueType OVT = Node->getValueType(0);
      MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Cast the two input vectors.
      Tmp1 = DAG.getNode(ISD::BIT_CONVERT, NVT, Tmp1);
      Tmp2 = DAG.getNode(ISD::BIT_CONVERT, NVT, Tmp2);
      
      // Convert the shuffle mask to the right # elements.
      Tmp3 = SDOperand(isShuffleLegal(OVT, Node->getOperand(2)), 0);
      assert(Tmp3.Val && "Shuffle not legal?");
      Result = DAG.getNode(ISD::VECTOR_SHUFFLE, NVT, Tmp1, Tmp2, Tmp3);
      Result = DAG.getNode(ISD::BIT_CONVERT, OVT, Result);
      break;
    }
    }
    break;
  
  case ISD::EXTRACT_VECTOR_ELT:
    Tmp1 = Node->getOperand(0);
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    Result = ExpandEXTRACT_VECTOR_ELT(Result);
    break;

  case ISD::EXTRACT_SUBVECTOR: 
    Tmp1 = Node->getOperand(0);
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    Result = ExpandEXTRACT_SUBVECTOR(Result);
    break;
    
  case ISD::CALLSEQ_START: {
    SDNode *CallEnd = FindCallEndFromCallStart(Node);
    
    // Recursively Legalize all of the inputs of the call end that do not lead
    // to this call start.  This ensures that any libcalls that need be inserted
    // are inserted *before* the CALLSEQ_START.
    {SmallPtrSet<SDNode*, 32> NodesLeadingTo;
    for (unsigned i = 0, e = CallEnd->getNumOperands(); i != e; ++i)
      LegalizeAllNodesNotLeadingTo(CallEnd->getOperand(i).Val, Node,
                                   NodesLeadingTo);
    }

    // Now that we legalized all of the inputs (which may have inserted
    // libcalls) create the new CALLSEQ_START node.
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    // Merge in the last call, to ensure that this call start after the last
    // call ended.
    if (LastCALLSEQ_END.getOpcode() != ISD::EntryToken) {
      Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
      Tmp1 = LegalizeOp(Tmp1);
    }
      
    // Do not try to legalize the target-specific arguments (#1+).
    if (Tmp1 != Node->getOperand(0)) {
      SmallVector<SDOperand, 8> Ops(Node->op_begin(), Node->op_end());
      Ops[0] = Tmp1;
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
    }
    
    // Remember that the CALLSEQ_START is legalized.
    AddLegalizedOperand(Op.getValue(0), Result);
    if (Node->getNumValues() == 2)    // If this has a flag result, remember it.
      AddLegalizedOperand(Op.getValue(1), Result.getValue(1));
    
    // Now that the callseq_start and all of the non-call nodes above this call
    // sequence have been legalized, legalize the call itself.  During this 
    // process, no libcalls can/will be inserted, guaranteeing that no calls
    // can overlap.
    assert(!IsLegalizingCall && "Inconsistent sequentialization of calls!");
    SDOperand InCallSEQ = LastCALLSEQ_END;
    // Note that we are selecting this call!
    LastCALLSEQ_END = SDOperand(CallEnd, 0);
    IsLegalizingCall = true;
    
    // Legalize the call, starting from the CALLSEQ_END.
    LegalizeOp(LastCALLSEQ_END);
    assert(!IsLegalizingCall && "CALLSEQ_END should have cleared this!");
    return Result;
  }
  case ISD::CALLSEQ_END:
    // If the CALLSEQ_START node hasn't been legalized first, legalize it.  This
    // will cause this node to be legalized as well as handling libcalls right.
    if (LastCALLSEQ_END.Val != Node) {
      LegalizeOp(SDOperand(FindCallStartFromCallEnd(Node), 0));
      DenseMap<SDOperand, SDOperand>::iterator I = LegalizedNodes.find(Op);
      assert(I != LegalizedNodes.end() &&
             "Legalizing the call start should have legalized this node!");
      return I->second;
    }
    
    // Otherwise, the call start has been legalized and everything is going 
    // according to plan.  Just legalize ourselves normally here.
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Do not try to legalize the target-specific arguments (#1+), except for
    // an optional flag input.
    if (Node->getOperand(Node->getNumOperands()-1).getValueType() != MVT::Flag){
      if (Tmp1 != Node->getOperand(0)) {
        SmallVector<SDOperand, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    } else {
      Tmp2 = LegalizeOp(Node->getOperand(Node->getNumOperands()-1));
      if (Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(Node->getNumOperands()-1)) {
        SmallVector<SDOperand, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Ops.back() = Tmp2;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    }
    assert(IsLegalizingCall && "Call sequence imbalance between start/end?");
    // This finishes up call legalization.
    IsLegalizingCall = false;
    
    // If the CALLSEQ_END node has a flag, remember that we legalized it.
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    if (Node->getNumValues() == 2)
      AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  case ISD::DYNAMIC_STACKALLOC: {
    MVT::ValueType VT = Node->getValueType(0);
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the size.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the alignment.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);

    Tmp1 = Result.getValue(0);
    Tmp2 = Result.getValue(1);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
      unsigned SPReg = TLI.getStackPointerRegisterToSaveRestore();
      assert(SPReg && "Target cannot require DYNAMIC_STACKALLOC expansion and"
             " not tell us which reg is the stack pointer!");
      SDOperand Chain = Tmp1.getOperand(0);

      // Chain the dynamic stack allocation so that it doesn't modify the stack
      // pointer when other instructions are using the stack.
      Chain = DAG.getCALLSEQ_START(Chain,
                                   DAG.getConstant(0, TLI.getPointerTy()));

      SDOperand Size  = Tmp2.getOperand(1);
      SDOperand SP = DAG.getCopyFromReg(Chain, SPReg, VT);
      Chain = SP.getValue(1);
      unsigned Align = cast<ConstantSDNode>(Tmp3)->getValue();
      unsigned StackAlign =
        TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
      if (Align > StackAlign)
        SP = DAG.getNode(ISD::AND, VT, SP,
                         DAG.getConstant(-(uint64_t)Align, VT));
      Tmp1 = DAG.getNode(ISD::SUB, VT, SP, Size);       // Value
      Chain = DAG.getCopyToReg(Chain, SPReg, Tmp1);     // Output chain

      Tmp2 =
        DAG.getCALLSEQ_END(Chain,
                           DAG.getConstant(0, TLI.getPointerTy()),
                           DAG.getConstant(0, TLI.getPointerTy()),
                           SDOperand());

      Tmp1 = LegalizeOp(Tmp1);
      Tmp2 = LegalizeOp(Tmp2);
      break;
    }
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Tmp1, DAG);
      if (Tmp3.Val) {
        Tmp1 = LegalizeOp(Tmp3);
        Tmp2 = LegalizeOp(Tmp3.getValue(1));
      }
      break;
    case TargetLowering::Legal:
      break;
    }
    // Since this op produce two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Tmp1);
    AddLegalizedOperand(SDOperand(Node, 1), Tmp2);
    return Op.ResNo ? Tmp2 : Tmp1;
  }
  case ISD::INLINEASM: {
    SmallVector<SDOperand, 8> Ops(Node->op_begin(), Node->op_end());
    bool Changed = false;
    // Legalize all of the operands of the inline asm, in case they are nodes
    // that need to be expanded or something.  Note we skip the asm string and
    // all of the TargetConstant flags.
    SDOperand Op = LegalizeOp(Ops[0]);
    Changed = Op != Ops[0];
    Ops[0] = Op;

    bool HasInFlag = Ops.back().getValueType() == MVT::Flag;
    for (unsigned i = 2, e = Ops.size()-HasInFlag; i < e; ) {
      unsigned NumVals = cast<ConstantSDNode>(Ops[i])->getValue() >> 3;
      for (++i; NumVals; ++i, --NumVals) {
        SDOperand Op = LegalizeOp(Ops[i]);
        if (Op != Ops[i]) {
          Changed = true;
          Ops[i] = Op;
        }
      }
    }

    if (HasInFlag) {
      Op = LegalizeOp(Ops.back());
      Changed |= Op != Ops.back();
      Ops.back() = Op;
    }
    
    if (Changed)
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      
    // INLINE asm returns a chain and flag, make sure to add both to the map.
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result.getValue(Op.ResNo);
  }
  case ISD::BR:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();
    
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
    break;
  case ISD::BRIND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();
    
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    default: assert(0 && "Indirect target must be legal type (pointer)!");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the condition.
      break;
    }
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    break;
  case ISD::BR_JT:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();

    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the jumptable node.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));

    switch (TLI.getOperationAction(ISD::BR_JT, MVT::Other)) {  
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Expand: {
      SDOperand Chain = Result.getOperand(0);
      SDOperand Table = Result.getOperand(1);
      SDOperand Index = Result.getOperand(2);

      MVT::ValueType PTy = TLI.getPointerTy();
      MachineFunction &MF = DAG.getMachineFunction();
      unsigned EntrySize = MF.getJumpTableInfo()->getEntrySize();
      Index= DAG.getNode(ISD::MUL, PTy, Index, DAG.getConstant(EntrySize, PTy));
      SDOperand Addr = DAG.getNode(ISD::ADD, PTy, Index, Table);
      
      SDOperand LD;
      switch (EntrySize) {
      default: assert(0 && "Size of jump table not supported yet."); break;
      case 4: LD = DAG.getLoad(MVT::i32, Chain, Addr,
                               PseudoSourceValue::getJumpTable(), 0); break;
      case 8: LD = DAG.getLoad(MVT::i64, Chain, Addr,
                               PseudoSourceValue::getJumpTable(), 0); break;
      }

      Addr = LD;
      if (TLI.getTargetMachine().getRelocationModel() == Reloc::PIC_) {
        // For PIC, the sequence is:
        // BRIND(load(Jumptable + index) + RelocBase)
        // RelocBase can be JumpTable, GOT or some sort of global base.
        if (PTy != MVT::i32)
          Addr = DAG.getNode(ISD::SIGN_EXTEND, PTy, Addr);
        Addr = DAG.getNode(ISD::ADD, PTy, Addr,
                           TLI.getPICJumpTableRelocBase(Table, DAG));
      }
      Result = DAG.getNode(ISD::BRIND, MVT::Other, LD.getValue(1), Addr);
    }
    }
    break;
  case ISD::BRCOND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a return.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();

    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the condition.
      break;
    case Promote: {
      Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the condition.
      
      // The top bits of the promoted condition are not necessarily zero, ensure
      // that the value is properly zero extended.
      unsigned BitWidth = Tmp2.getValueSizeInBits();
      if (!DAG.MaskedValueIsZero(Tmp2, 
                                 APInt::getHighBitsSet(BitWidth, BitWidth-1)))
        Tmp2 = DAG.getZeroExtendInReg(Tmp2, MVT::i1);
      break;
    }
    }

    // Basic block destination (Op#2) is always legal.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));
      
    switch (TLI.getOperationAction(ISD::BRCOND, MVT::Other)) {  
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Expand:
      // Expand brcond's setcc into its constituent parts and create a BR_CC
      // Node.
      if (Tmp2.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, Tmp2.getOperand(2),
                             Tmp2.getOperand(0), Tmp2.getOperand(1),
                             Node->getOperand(2));
      } else {
        Result = DAG.getNode(ISD::BR_CC, MVT::Other, Tmp1, 
                             DAG.getCondCode(ISD::SETNE), Tmp2,
                             DAG.getConstant(0, Tmp2.getValueType()),
                             Node->getOperand(2));
      }
      break;
    }
    break;
  case ISD::BR_CC:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    Tmp2 = Node->getOperand(2);              // LHS 
    Tmp3 = Node->getOperand(3);              // RHS
    Tmp4 = Node->getOperand(1);              // CC

    LegalizeSetCCOperands(Tmp2, Tmp3, Tmp4);
    LastCALLSEQ_END = DAG.getEntryNode();

    // If we didn't get both a LHS and RHS back from LegalizeSetCCOperands,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp3.Val == 0) {
      Tmp3 = DAG.getConstant(0, Tmp2.getValueType());
      Tmp4 = DAG.getCondCode(ISD::SETNE);
    }
    
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp4, Tmp2, Tmp3, 
                                    Node->getOperand(4));
      
    switch (TLI.getOperationAction(ISD::BR_CC, Tmp3.getValueType())) {
    default: assert(0 && "Unexpected action for BR_CC!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp4 = TLI.LowerOperation(Result, DAG);
      if (Tmp4.Val) Result = Tmp4;
      break;
    }
    break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    Tmp1 = LegalizeOp(LD->getChain());   // Legalize the chain.
    Tmp2 = LegalizeOp(LD->getBasePtr()); // Legalize the base pointer.

    ISD::LoadExtType ExtType = LD->getExtensionType();
    if (ExtType == ISD::NON_EXTLOAD) {
      MVT::ValueType VT = Node->getValueType(0);
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, LD->getOffset());
      Tmp3 = Result.getValue(0);
      Tmp4 = Result.getValue(1);
    
      switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Legal:
        // If this is an unaligned load and the target doesn't support it,
        // expand it.
        if (!TLI.allowsUnalignedMemoryAccesses()) {
          unsigned ABIAlignment = TLI.getTargetData()->
            getABITypeAlignment(MVT::getTypeForValueType(LD->getMemoryVT()));
          if (LD->getAlignment() < ABIAlignment){
            Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.Val), DAG,
                                         TLI);
            Tmp3 = Result.getOperand(0);
            Tmp4 = Result.getOperand(1);
            Tmp3 = LegalizeOp(Tmp3);
            Tmp4 = LegalizeOp(Tmp4);
          }
        }
        break;
      case TargetLowering::Custom:
        Tmp1 = TLI.LowerOperation(Tmp3, DAG);
        if (Tmp1.Val) {
          Tmp3 = LegalizeOp(Tmp1);
          Tmp4 = LegalizeOp(Tmp1.getValue(1));
        }
        break;
      case TargetLowering::Promote: {
        // Only promote a load of vector type to another.
        assert(MVT::isVector(VT) && "Cannot promote this load!");
        // Change base type to a different vector type.
        MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), VT);

        Tmp1 = DAG.getLoad(NVT, Tmp1, Tmp2, LD->getSrcValue(),
                           LD->getSrcValueOffset(),
                           LD->isVolatile(), LD->getAlignment());
        Tmp3 = LegalizeOp(DAG.getNode(ISD::BIT_CONVERT, VT, Tmp1));
        Tmp4 = LegalizeOp(Tmp1.getValue(1));
        break;
      }
      }
      // Since loads produce two values, make sure to remember that we 
      // legalized both of them.
      AddLegalizedOperand(SDOperand(Node, 0), Tmp3);
      AddLegalizedOperand(SDOperand(Node, 1), Tmp4);
      return Op.ResNo ? Tmp4 : Tmp3;
    } else {
      MVT::ValueType SrcVT = LD->getMemoryVT();
      unsigned SrcWidth = MVT::getSizeInBits(SrcVT);
      int SVOffset = LD->getSrcValueOffset();
      unsigned Alignment = LD->getAlignment();
      bool isVolatile = LD->isVolatile();

      if (SrcWidth != MVT::getStoreSizeInBits(SrcVT) &&
          // Some targets pretend to have an i1 loading operation, and actually
          // load an i8.  This trick is correct for ZEXTLOAD because the top 7
          // bits are guaranteed to be zero; it helps the optimizers understand
          // that these bits are zero.  It is also useful for EXTLOAD, since it
          // tells the optimizers that those bits are undefined.  It would be
          // nice to have an effective generic way of getting these benefits...
          // Until such a way is found, don't insist on promoting i1 here.
          (SrcVT != MVT::i1 ||
           TLI.getLoadXAction(ExtType, MVT::i1) == TargetLowering::Promote)) {
        // Promote to a byte-sized load if not loading an integral number of
        // bytes.  For example, promote EXTLOAD:i20 -> EXTLOAD:i24.
        unsigned NewWidth = MVT::getStoreSizeInBits(SrcVT);
        MVT::ValueType NVT = MVT::getIntegerType(NewWidth);
        SDOperand Ch;

        // The extra bits are guaranteed to be zero, since we stored them that
        // way.  A zext load from NVT thus automatically gives zext from SrcVT.

        ISD::LoadExtType NewExtType =
          ExtType == ISD::ZEXTLOAD ? ISD::ZEXTLOAD : ISD::EXTLOAD;

        Result = DAG.getExtLoad(NewExtType, Node->getValueType(0),
                                Tmp1, Tmp2, LD->getSrcValue(), SVOffset,
                                NVT, isVolatile, Alignment);

        Ch = Result.getValue(1); // The chain.

        if (ExtType == ISD::SEXTLOAD)
          // Having the top bits zero doesn't help when sign extending.
          Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                               Result, DAG.getValueType(SrcVT));
        else if (ExtType == ISD::ZEXTLOAD || NVT == Result.getValueType())
          // All the top bits are guaranteed to be zero - inform the optimizers.
          Result = DAG.getNode(ISD::AssertZext, Result.getValueType(), Result,
                               DAG.getValueType(SrcVT));

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else if (SrcWidth & (SrcWidth - 1)) {
        // If not loading a power-of-2 number of bits, expand as two loads.
        assert(MVT::isExtendedVT(SrcVT) && !MVT::isVector(SrcVT) &&
               "Unsupported extload!");
        unsigned RoundWidth = 1 << Log2_32(SrcWidth);
        assert(RoundWidth < SrcWidth);
        unsigned ExtraWidth = SrcWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Load size not an integral number of bytes!");
        MVT::ValueType RoundVT = MVT::getIntegerType(RoundWidth);
        MVT::ValueType ExtraVT = MVT::getIntegerType(ExtraWidth);
        SDOperand Lo, Hi, Ch;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // EXTLOAD:i24 -> ZEXTLOAD:i16 | (shl EXTLOAD@+2:i8, 16)
          // Load the bottom RoundWidth bits.
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getExtLoad(ExtType, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, Hi.getValueType(), Hi,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, Node->getValueType(0), Lo, Hi);
        } else {
          // Big endian - avoid unaligned loads.
          // EXTLOAD:i24 -> (shl EXTLOAD:i16, 8) | ZEXTLOAD@+2:i8
          // Load the top RoundWidth bits.
          Hi = DAG.getExtLoad(ExtType, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, Hi.getValueType(), Hi,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, Node->getValueType(0), Lo, Hi);
        }

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else {
        switch (TLI.getLoadXAction(ExtType, SrcVT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Custom:
          isCustom = true;
          // FALLTHROUGH
        case TargetLowering::Legal:
          Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, LD->getOffset());
          Tmp1 = Result.getValue(0);
          Tmp2 = Result.getValue(1);

          if (isCustom) {
            Tmp3 = TLI.LowerOperation(Result, DAG);
            if (Tmp3.Val) {
              Tmp1 = LegalizeOp(Tmp3);
              Tmp2 = LegalizeOp(Tmp3.getValue(1));
            }
          } else {
            // If this is an unaligned load and the target doesn't support it,
            // expand it.
            if (!TLI.allowsUnalignedMemoryAccesses()) {
              unsigned ABIAlignment = TLI.getTargetData()->
                getABITypeAlignment(MVT::getTypeForValueType(LD->getMemoryVT()));
              if (LD->getAlignment() < ABIAlignment){
                Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.Val), DAG,
                                             TLI);
                Tmp1 = Result.getOperand(0);
                Tmp2 = Result.getOperand(1);
                Tmp1 = LegalizeOp(Tmp1);
                Tmp2 = LegalizeOp(Tmp2);
              }
            }
          }
          break;
        case TargetLowering::Expand:
          // f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
          if (SrcVT == MVT::f32 && Node->getValueType(0) == MVT::f64) {
            SDOperand Load = DAG.getLoad(SrcVT, Tmp1, Tmp2, LD->getSrcValue(),
                                         LD->getSrcValueOffset(),
                                         LD->isVolatile(), LD->getAlignment());
            Result = DAG.getNode(ISD::FP_EXTEND, Node->getValueType(0), Load);
            Tmp1 = LegalizeOp(Result);  // Relegalize new nodes.
            Tmp2 = LegalizeOp(Load.getValue(1));
            break;
          }
          assert(ExtType != ISD::EXTLOAD &&"EXTLOAD should always be supported!");
          // Turn the unsupported load into an EXTLOAD followed by an explicit
          // zero/sign extend inreg.
          Result = DAG.getExtLoad(ISD::EXTLOAD, Node->getValueType(0),
                                  Tmp1, Tmp2, LD->getSrcValue(),
                                  LD->getSrcValueOffset(), SrcVT,
                                  LD->isVolatile(), LD->getAlignment());
          SDOperand ValRes;
          if (ExtType == ISD::SEXTLOAD)
            ValRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                                 Result, DAG.getValueType(SrcVT));
          else
            ValRes = DAG.getZeroExtendInReg(Result, SrcVT);
          Tmp1 = LegalizeOp(ValRes);  // Relegalize new nodes.
          Tmp2 = LegalizeOp(Result.getValue(1));  // Relegalize new nodes.
          break;
        }
      }

      // Since loads produce two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDOperand(Node, 0), Tmp1);
      AddLegalizedOperand(SDOperand(Node, 1), Tmp2);
      return Op.ResNo ? Tmp2 : Tmp1;
    }
  }
  case ISD::EXTRACT_ELEMENT: {
    MVT::ValueType OpTy = Node->getOperand(0).getValueType();
    switch (getTypeAction(OpTy)) {
    default: assert(0 && "EXTRACT_ELEMENT action for type unimplemented!");
    case Legal:
      if (cast<ConstantSDNode>(Node->getOperand(1))->getValue()) {
        // 1 -> Hi
        Result = DAG.getNode(ISD::SRL, OpTy, Node->getOperand(0),
                             DAG.getConstant(MVT::getSizeInBits(OpTy)/2, 
                                             TLI.getShiftAmountTy()));
        Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), Result);
      } else {
        // 0 -> Lo
        Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), 
                             Node->getOperand(0));
      }
      break;
    case Expand:
      // Get both the low and high parts.
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      if (cast<ConstantSDNode>(Node->getOperand(1))->getValue())
        Result = Tmp2;  // 1 -> Hi
      else
        Result = Tmp1;  // 0 -> Lo
      break;
    }
    break;
  }

  case ISD::CopyToReg:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    assert(isTypeLegal(Node->getOperand(2).getValueType()) &&
           "Register type must be legal!");
    // Legalize the incoming value (must be a legal type).
    Tmp2 = LegalizeOp(Node->getOperand(2));
    if (Node->getNumValues() == 1) {
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1), Tmp2);
    } else {
      assert(Node->getNumValues() == 2 && "Unknown CopyToReg");
      if (Node->getNumOperands() == 4) {
        Tmp3 = LegalizeOp(Node->getOperand(3));
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1), Tmp2,
                                        Tmp3);
      } else {
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1),Tmp2);
      }
      
      // Since this produces two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
      AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
      return Result;
    }
    break;

  case ISD::RET:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    // Ensure that libcalls are emitted before a return.
    Tmp1 = DAG.getNode(ISD::TokenFactor, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();
      
    switch (Node->getNumOperands()) {
    case 3:  // ret val
      Tmp2 = Node->getOperand(1);
      Tmp3 = Node->getOperand(2);  // Signness
      switch (getTypeAction(Tmp2.getValueType())) {
      case Legal:
        Result = DAG.UpdateNodeOperands(Result, Tmp1, LegalizeOp(Tmp2), Tmp3);
        break;
      case Expand:
        if (!MVT::isVector(Tmp2.getValueType())) {
          SDOperand Lo, Hi;
          ExpandOp(Tmp2, Lo, Hi);

          // Big endian systems want the hi reg first.
          if (TLI.isBigEndian())
            std::swap(Lo, Hi);
          
          if (Hi.Val)
            Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Lo, Tmp3, Hi,Tmp3);
          else
            Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Lo, Tmp3);
          Result = LegalizeOp(Result);
        } else {
          SDNode *InVal = Tmp2.Val;
          int InIx = Tmp2.ResNo;
          unsigned NumElems = MVT::getVectorNumElements(InVal->getValueType(InIx));
          MVT::ValueType EVT = MVT::getVectorElementType(InVal->getValueType(InIx));
          
          // Figure out if there is a simple type corresponding to this Vector
          // type.  If so, convert to the vector type.
          MVT::ValueType TVT = MVT::getVectorType(EVT, NumElems);
          if (TLI.isTypeLegal(TVT)) {
            // Turn this into a return of the vector type.
            Tmp2 = LegalizeOp(Tmp2);
            Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
          } else if (NumElems == 1) {
            // Turn this into a return of the scalar type.
            Tmp2 = ScalarizeVectorOp(Tmp2);
            Tmp2 = LegalizeOp(Tmp2);
            Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
            
            // FIXME: Returns of gcc generic vectors smaller than a legal type
            // should be returned in integer registers!
            
            // The scalarized value type may not be legal, e.g. it might require
            // promotion or expansion.  Relegalize the return.
            Result = LegalizeOp(Result);
          } else {
            // FIXME: Returns of gcc generic vectors larger than a legal vector
            // type should be returned by reference!
            SDOperand Lo, Hi;
            SplitVectorOp(Tmp2, Lo, Hi);
            Result = DAG.getNode(ISD::RET, MVT::Other, Tmp1, Lo, Tmp3, Hi,Tmp3);
            Result = LegalizeOp(Result);
          }
        }
        break;
      case Promote:
        Tmp2 = PromoteOp(Node->getOperand(1));
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
        Result = LegalizeOp(Result);
        break;
      }
      break;
    case 1:  // ret void
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    default: { // ret <values>
      SmallVector<SDOperand, 8> NewValues;
      NewValues.push_back(Tmp1);
      for (unsigned i = 1, e = Node->getNumOperands(); i < e; i += 2)
        switch (getTypeAction(Node->getOperand(i).getValueType())) {
        case Legal:
          NewValues.push_back(LegalizeOp(Node->getOperand(i)));
          NewValues.push_back(Node->getOperand(i+1));
          break;
        case Expand: {
          SDOperand Lo, Hi;
          assert(!MVT::isExtendedVT(Node->getOperand(i).getValueType()) &&
                 "FIXME: TODO: implement returning non-legal vector types!");
          ExpandOp(Node->getOperand(i), Lo, Hi);
          NewValues.push_back(Lo);
          NewValues.push_back(Node->getOperand(i+1));
          if (Hi.Val) {
            NewValues.push_back(Hi);
            NewValues.push_back(Node->getOperand(i+1));
          }
          break;
        }
        case Promote:
          assert(0 && "Can't promote multiple return value yet!");
        }
          
      if (NewValues.size() == Node->getNumOperands())
        Result = DAG.UpdateNodeOperands(Result, &NewValues[0],NewValues.size());
      else
        Result = DAG.getNode(ISD::RET, MVT::Other,
                             &NewValues[0], NewValues.size());
      break;
    }
    }

    if (Result.getOpcode() == ISD::RET) {
      switch (TLI.getOperationAction(Result.getOpcode(), MVT::Other)) {
      default: assert(0 && "This action is not supported yet!");
      case TargetLowering::Legal: break;
      case TargetLowering::Custom:
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) Result = Tmp1;
        break;
      }
    }
    break;
  case ISD::STORE: {
    StoreSDNode *ST = cast<StoreSDNode>(Node);
    Tmp1 = LegalizeOp(ST->getChain());    // Legalize the chain.
    Tmp2 = LegalizeOp(ST->getBasePtr());  // Legalize the pointer.
    int SVOffset = ST->getSrcValueOffset();
    unsigned Alignment = ST->getAlignment();
    bool isVolatile = ST->isVolatile();

    if (!ST->isTruncatingStore()) {
      // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
      // FIXME: We shouldn't do this for TargetConstantFP's.
      // FIXME: move this to the DAG Combiner!  Note that we can't regress due
      // to phase ordering between legalized code and the dag combiner.  This
      // probably means that we need to integrate dag combiner and legalizer
      // together.
      // We generally can't do this one for long doubles.
      if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(ST->getValue())) {
        if (CFP->getValueType(0) == MVT::f32 && 
            getTypeAction(MVT::i32) == Legal) {
          Tmp3 = DAG.getConstant(CFP->getValueAPF().
                                          convertToAPInt().zextOrTrunc(32),
                                  MVT::i32);
          Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                SVOffset, isVolatile, Alignment);
          break;
        } else if (CFP->getValueType(0) == MVT::f64) {
          // If this target supports 64-bit registers, do a single 64-bit store.
          if (getTypeAction(MVT::i64) == Legal) {
            Tmp3 = DAG.getConstant(CFP->getValueAPF().convertToAPInt().
                                     zextOrTrunc(64), MVT::i64);
            Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            break;
          } else if (getTypeAction(MVT::i32) == Legal) {
            // Otherwise, if the target supports 32-bit registers, use 2 32-bit
            // stores.  If the target supports neither 32- nor 64-bits, this
            // xform is certainly not worth it.
            const APInt &IntVal =CFP->getValueAPF().convertToAPInt();
            SDOperand Lo = DAG.getConstant(APInt(IntVal).trunc(32), MVT::i32);
            SDOperand Hi = DAG.getConstant(IntVal.lshr(32).trunc(32), MVT::i32);
            if (TLI.isBigEndian()) std::swap(Lo, Hi);

            Lo = DAG.getStore(Tmp1, Lo, Tmp2, ST->getSrcValue(),
                              SVOffset, isVolatile, Alignment);
            Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                               DAG.getIntPtrConstant(4));
            Hi = DAG.getStore(Tmp1, Hi, Tmp2, ST->getSrcValue(), SVOffset+4,
                              isVolatile, MinAlign(Alignment, 4U));

            Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
            break;
          }
        }
      }
      
      switch (getTypeAction(ST->getMemoryVT())) {
      case Legal: {
        Tmp3 = LegalizeOp(ST->getValue());
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp3, Tmp2, 
                                        ST->getOffset());

        MVT::ValueType VT = Tmp3.getValueType();
        switch (TLI.getOperationAction(ISD::STORE, VT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Legal:
          // If this is an unaligned store and the target doesn't support it,
          // expand it.
          if (!TLI.allowsUnalignedMemoryAccesses()) {
            unsigned ABIAlignment = TLI.getTargetData()->
              getABITypeAlignment(MVT::getTypeForValueType(ST->getMemoryVT()));
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.Val), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.Val) Result = Tmp1;
          break;
        case TargetLowering::Promote:
          assert(MVT::isVector(VT) && "Unknown legal promote case!");
          Tmp3 = DAG.getNode(ISD::BIT_CONVERT, 
                             TLI.getTypeToPromoteTo(ISD::STORE, VT), Tmp3);
          Result = DAG.getStore(Tmp1, Tmp3, Tmp2,
                                ST->getSrcValue(), SVOffset, isVolatile,
                                Alignment);
          break;
        }
        break;
      }
      case Promote:
        // Truncate the value and store the result.
        Tmp3 = PromoteOp(ST->getValue());
        Result = DAG.getTruncStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                   SVOffset, ST->getMemoryVT(),
                                   isVolatile, Alignment);
        break;

      case Expand:
        unsigned IncrementSize = 0;
        SDOperand Lo, Hi;
      
        // If this is a vector type, then we have to calculate the increment as
        // the product of the element size in bytes, and the number of elements
        // in the high half of the vector.
        if (MVT::isVector(ST->getValue().getValueType())) {
          SDNode *InVal = ST->getValue().Val;
          int InIx = ST->getValue().ResNo;
          MVT::ValueType InVT = InVal->getValueType(InIx);
          unsigned NumElems = MVT::getVectorNumElements(InVT);
          MVT::ValueType EVT = MVT::getVectorElementType(InVT);

          // Figure out if there is a simple type corresponding to this Vector
          // type.  If so, convert to the vector type.
          MVT::ValueType TVT = MVT::getVectorType(EVT, NumElems);
          if (TLI.isTypeLegal(TVT)) {
            // Turn this into a normal store of the vector type.
            Tmp3 = LegalizeOp(ST->getValue());
            Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            Result = LegalizeOp(Result);
            break;
          } else if (NumElems == 1) {
            // Turn this into a normal store of the scalar type.
            Tmp3 = ScalarizeVectorOp(ST->getValue());
            Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            // The scalarized value type may not be legal, e.g. it might require
            // promotion or expansion.  Relegalize the scalar store.
            Result = LegalizeOp(Result);
            break;
          } else {
            SplitVectorOp(ST->getValue(), Lo, Hi);
            IncrementSize = MVT::getVectorNumElements(Lo.Val->getValueType(0)) * 
                            MVT::getSizeInBits(EVT)/8;
          }
        } else {
          ExpandOp(ST->getValue(), Lo, Hi);
          IncrementSize = Hi.Val ? MVT::getSizeInBits(Hi.getValueType())/8 : 0;

          if (TLI.isBigEndian())
            std::swap(Lo, Hi);
        }

        Lo = DAG.getStore(Tmp1, Lo, Tmp2, ST->getSrcValue(),
                          SVOffset, isVolatile, Alignment);

        if (Hi.Val == NULL) {
          // Must be int <-> float one-to-one expansion.
          Result = Lo;
          break;
        }

        Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                           DAG.getIntPtrConstant(IncrementSize));
        assert(isTypeLegal(Tmp2.getValueType()) &&
               "Pointers must be legal!");
        SVOffset += IncrementSize;
        Alignment = MinAlign(Alignment, IncrementSize);
        Hi = DAG.getStore(Tmp1, Hi, Tmp2, ST->getSrcValue(),
                          SVOffset, isVolatile, Alignment);
        Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
        break;
      }
    } else {
      switch (getTypeAction(ST->getValue().getValueType())) {
      case Legal:
        Tmp3 = LegalizeOp(ST->getValue());
        break;
      case Promote:
        // We can promote the value, the truncstore will still take care of it.
        Tmp3 = PromoteOp(ST->getValue());
        break;
      case Expand:
        // Just store the low part.  This may become a non-trunc store, so make
        // sure to use getTruncStore, not UpdateNodeOperands below.
        ExpandOp(ST->getValue(), Tmp3, Tmp4);
        return DAG.getTruncStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset, MVT::i8, isVolatile, Alignment);
      }

      MVT::ValueType StVT = ST->getMemoryVT();
      unsigned StWidth = MVT::getSizeInBits(StVT);

      if (StWidth != MVT::getStoreSizeInBits(StVT)) {
        // Promote to a byte-sized store with upper bits zero if not
        // storing an integral number of bytes.  For example, promote
        // TRUNCSTORE:i1 X -> TRUNCSTORE:i8 (and X, 1)
        MVT::ValueType NVT = MVT::getIntegerType(MVT::getStoreSizeInBits(StVT));
        Tmp3 = DAG.getZeroExtendInReg(Tmp3, StVT);
        Result = DAG.getTruncStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                   SVOffset, NVT, isVolatile, Alignment);
      } else if (StWidth & (StWidth - 1)) {
        // If not storing a power-of-2 number of bits, expand as two stores.
        assert(MVT::isExtendedVT(StVT) && !MVT::isVector(StVT) &&
               "Unsupported truncstore!");
        unsigned RoundWidth = 1 << Log2_32(StWidth);
        assert(RoundWidth < StWidth);
        unsigned ExtraWidth = StWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Store size not an integral number of bytes!");
        MVT::ValueType RoundVT = MVT::getIntegerType(RoundWidth);
        MVT::ValueType ExtraVT = MVT::getIntegerType(ExtraWidth);
        SDOperand Lo, Hi;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 X, TRUNCSTORE@+2:i8 (srl X, 16)
          // Store the bottom RoundWidth bits.
          Lo = DAG.getTruncStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset, RoundVT,
                                 isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getNode(ISD::SRL, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, Hi, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        } else {
          // Big endian - avoid unaligned stores.
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 (srl X, 8), TRUNCSTORE@+2:i8 X
          // Store the top RoundWidth bits.
          Hi = DAG.getNode(ISD::SRL, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, Hi, Tmp2, ST->getSrcValue(), SVOffset,
                                 RoundVT, isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getTruncStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        }

        // The order of the stores doesn't matter.
        Result = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
      } else {
        if (Tmp1 != ST->getChain() || Tmp3 != ST->getValue() ||
            Tmp2 != ST->getBasePtr())
          Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp3, Tmp2,
                                          ST->getOffset());

        switch (TLI.getTruncStoreAction(ST->getValue().getValueType(), StVT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Legal:
          // If this is an unaligned store and the target doesn't support it,
          // expand it.
          if (!TLI.allowsUnalignedMemoryAccesses()) {
            unsigned ABIAlignment = TLI.getTargetData()->
              getABITypeAlignment(MVT::getTypeForValueType(ST->getMemoryVT()));
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.Val), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Result = TLI.LowerOperation(Result, DAG);
          break;
        case Expand:
          // TRUNCSTORE:i16 i32 -> STORE i16
          assert(isTypeLegal(StVT) && "Do not know how to expand this store!");
          Tmp3 = DAG.getNode(ISD::TRUNCATE, StVT, Tmp3);
          Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(), SVOffset,
                                isVolatile, Alignment);
          break;
        }
      }
    }
    break;
  }
  case ISD::PCMARKER:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
    break;
  case ISD::STACKSAVE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Result = DAG.UpdateNodeOperands(Result, Tmp1);
    Tmp1 = Result.getValue(0);
    Tmp2 = Result.getValue(1);
    
    switch (TLI.getOperationAction(ISD::STACKSAVE, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.Val) {
        Tmp1 = LegalizeOp(Tmp3);
        Tmp2 = LegalizeOp(Tmp3.getValue(1));
      }
      break;
    case TargetLowering::Expand:
      // Expand to CopyFromReg if the target set 
      // StackPointerRegisterToSaveRestore.
      if (unsigned SP = TLI.getStackPointerRegisterToSaveRestore()) {
        Tmp1 = DAG.getCopyFromReg(Result.getOperand(0), SP,
                                  Node->getValueType(0));
        Tmp2 = Tmp1.getValue(1);
      } else {
        Tmp1 = DAG.getNode(ISD::UNDEF, Node->getValueType(0));
        Tmp2 = Node->getOperand(0);
      }
      break;
    }

    // Since stacksave produce two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Tmp1);
    AddLegalizedOperand(SDOperand(Node, 1), Tmp2);
    return Op.ResNo ? Tmp2 : Tmp1;

  case ISD::STACKRESTORE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
      
    switch (TLI.getOperationAction(ISD::STACKRESTORE, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Expand:
      // Expand to CopyToReg if the target set 
      // StackPointerRegisterToSaveRestore.
      if (unsigned SP = TLI.getStackPointerRegisterToSaveRestore()) {
        Result = DAG.getCopyToReg(Tmp1, SP, Tmp2);
      } else {
        Result = Tmp1;
      }
      break;
    }
    break;

  case ISD::READCYCLECOUNTER:
    Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the chain
    Result = DAG.UpdateNodeOperands(Result, Tmp1);
    switch (TLI.getOperationAction(ISD::READCYCLECOUNTER,
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = Result.getValue(0);
      Tmp2 = Result.getValue(1);
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Result, DAG);
      Tmp1 = LegalizeOp(Result.getValue(0));
      Tmp2 = LegalizeOp(Result.getValue(1));
      break;
    }

    // Since rdcc produce two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Tmp1);
    AddLegalizedOperand(SDOperand(Node, 1), Tmp2);
    return Result;

  case ISD::SELECT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the condition.
      break;
    case Promote: {
      Tmp1 = PromoteOp(Node->getOperand(0));  // Promote the condition.
      // Make sure the condition is either zero or one.
      unsigned BitWidth = Tmp1.getValueSizeInBits();
      if (!DAG.MaskedValueIsZero(Tmp1,
                                 APInt::getHighBitsSet(BitWidth, BitWidth-1)))
        Tmp1 = DAG.getZeroExtendInReg(Tmp1, MVT::i1);
      break;
    }
    }
    Tmp2 = LegalizeOp(Node->getOperand(1));   // TrueVal
    Tmp3 = LegalizeOp(Node->getOperand(2));   // FalseVal

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      
    switch (TLI.getOperationAction(ISD::SELECT, Tmp2.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom: {
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    }
    case TargetLowering::Expand:
      if (Tmp1.getOpcode() == ISD::SETCC) {
        Result = DAG.getSelectCC(Tmp1.getOperand(0), Tmp1.getOperand(1), 
                              Tmp2, Tmp3,
                              cast<CondCodeSDNode>(Tmp1.getOperand(2))->get());
      } else {
        Result = DAG.getSelectCC(Tmp1, 
                                 DAG.getConstant(0, Tmp1.getValueType()),
                                 Tmp2, Tmp3, ISD::SETNE);
      }
      break;
    case TargetLowering::Promote: {
      MVT::ValueType NVT =
        TLI.getTypeToPromoteTo(ISD::SELECT, Tmp2.getValueType());
      unsigned ExtOp, TruncOp;
      if (MVT::isVector(Tmp2.getValueType())) {
        ExtOp   = ISD::BIT_CONVERT;
        TruncOp = ISD::BIT_CONVERT;
      } else if (MVT::isInteger(Tmp2.getValueType())) {
        ExtOp   = ISD::ANY_EXTEND;
        TruncOp = ISD::TRUNCATE;
      } else {
        ExtOp   = ISD::FP_EXTEND;
        TruncOp = ISD::FP_ROUND;
      }
      // Promote each of the values to the new type.
      Tmp2 = DAG.getNode(ExtOp, NVT, Tmp2);
      Tmp3 = DAG.getNode(ExtOp, NVT, Tmp3);
      // Perform the larger operation, then round down.
      Result = DAG.getNode(ISD::SELECT, NVT, Tmp1, Tmp2,Tmp3);
      if (TruncOp != ISD::FP_ROUND)
        Result = DAG.getNode(TruncOp, Node->getValueType(0), Result);
      else
        Result = DAG.getNode(TruncOp, Node->getValueType(0), Result,
                             DAG.getIntPtrConstant(0));
      break;
    }
    }
    break;
  case ISD::SELECT_CC: {
    Tmp1 = Node->getOperand(0);               // LHS
    Tmp2 = Node->getOperand(1);               // RHS
    Tmp3 = LegalizeOp(Node->getOperand(2));   // True
    Tmp4 = LegalizeOp(Node->getOperand(3));   // False
    SDOperand CC = Node->getOperand(4);
    
    LegalizeSetCCOperands(Tmp1, Tmp2, CC);
    
    // If we didn't get both a LHS and RHS back from LegalizeSetCCOperands,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp2.Val == 0) {
      Tmp2 = DAG.getConstant(0, Tmp1.getValueType());
      CC = DAG.getCondCode(ISD::SETNE);
    }
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3, Tmp4, CC);

    // Everything is legal, see if we should expand this op or something.
    switch (TLI.getOperationAction(ISD::SELECT_CC, Tmp3.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    }
    break;
  }
  case ISD::SETCC:
    Tmp1 = Node->getOperand(0);
    Tmp2 = Node->getOperand(1);
    Tmp3 = Node->getOperand(2);
    LegalizeSetCCOperands(Tmp1, Tmp2, Tmp3);
    
    // If we had to Expand the SetCC operands into a SELECT node, then it may 
    // not always be possible to return a true LHS & RHS.  In this case, just 
    // return the value we legalized, returned in the LHS
    if (Tmp2.Val == 0) {
      Result = Tmp1;
      break;
    }

    switch (TLI.getOperationAction(ISD::SETCC, Tmp1.getValueType())) {
    default: assert(0 && "Cannot handle this action for SETCC yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH.
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      if (isCustom) {
        Tmp4 = TLI.LowerOperation(Result, DAG);
        if (Tmp4.Val) Result = Tmp4;
      }
      break;
    case TargetLowering::Promote: {
      // First step, figure out the appropriate operation to use.
      // Allow SETCC to not be supported for all legal data types
      // Mostly this targets FP
      MVT::ValueType NewInTy = Node->getOperand(0).getValueType();
      MVT::ValueType OldVT = NewInTy; OldVT = OldVT;

      // Scan for the appropriate larger type to use.
      while (1) {
        NewInTy = (MVT::ValueType)(NewInTy+1);

        assert(MVT::isInteger(NewInTy) == MVT::isInteger(OldVT) &&
               "Fell off of the edge of the integer world");
        assert(MVT::isFloatingPoint(NewInTy) == MVT::isFloatingPoint(OldVT) &&
               "Fell off of the edge of the floating point world");
          
        // If the target supports SETCC of this type, use it.
        if (TLI.isOperationLegal(ISD::SETCC, NewInTy))
          break;
      }
      if (MVT::isInteger(NewInTy))
        assert(0 && "Cannot promote Legal Integer SETCC yet");
      else {
        Tmp1 = DAG.getNode(ISD::FP_EXTEND, NewInTy, Tmp1);
        Tmp2 = DAG.getNode(ISD::FP_EXTEND, NewInTy, Tmp2);
      }
      Tmp1 = LegalizeOp(Tmp1);
      Tmp2 = LegalizeOp(Tmp2);
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
      Result = LegalizeOp(Result);
      break;
    }
    case TargetLowering::Expand:
      // Expand a setcc node into a select_cc of the same condition, lhs, and
      // rhs that selects between const 1 (true) and const 0 (false).
      MVT::ValueType VT = Node->getValueType(0);
      Result = DAG.getNode(ISD::SELECT_CC, VT, Tmp1, Tmp2, 
                           DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                           Tmp3);
      break;
    }
    break;

  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    SmallVector<SDOperand, 8> Ops;
    bool Changed = false;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Changed |= Ops.back() != Node->getOperand(i);
    }
    if (Changed)
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());

    switch (TLI.getOperationAction(Node->getOpcode(),
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) {
        SDOperand Tmp2, RetVal(0, 0);
        for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i) {
          Tmp2 = LegalizeOp(Tmp1.getValue(i));
          AddLegalizedOperand(SDOperand(Node, i), Tmp2);
          if (i == Op.ResNo)
            RetVal = Tmp2;
        }
        assert(RetVal.Val && "Illegal result number");
        return RetVal;
      }
      break;
    }

    // Since these produce multiple values, make sure to remember that we
    // legalized all of them.
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      AddLegalizedOperand(SDOperand(Node, i), Result.getValue(i));
    return Result.getValue(Op.ResNo);
  }

    // Binary operators
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
  case ISD::UDIV:
  case ISD::SDIV:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FPOW:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "Not possible");
    case Legal:
      Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the RHS.
      break;
    case Promote:
      Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the RHS.
      break;
    }
    
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
      
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "BinOp legalize operation not supported");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Expand: {
      MVT::ValueType VT = Op.getValueType();
 
      // See if multiply or divide can be lowered using two-result operations.
      SDVTList VTs = DAG.getVTList(VT, VT);
      if (Node->getOpcode() == ISD::MUL) {
        // We just need the low half of the multiply; try both the signed
        // and unsigned forms. If the target supports both SMUL_LOHI and
        // UMUL_LOHI, form a preference by checking which forms of plain
        // MULH it supports.
        bool HasSMUL_LOHI = TLI.isOperationLegal(ISD::SMUL_LOHI, VT);
        bool HasUMUL_LOHI = TLI.isOperationLegal(ISD::UMUL_LOHI, VT);
        bool HasMULHS = TLI.isOperationLegal(ISD::MULHS, VT);
        bool HasMULHU = TLI.isOperationLegal(ISD::MULHU, VT);
        unsigned OpToUse = 0;
        if (HasSMUL_LOHI && !HasMULHS) {
          OpToUse = ISD::SMUL_LOHI;
        } else if (HasUMUL_LOHI && !HasMULHU) {
          OpToUse = ISD::UMUL_LOHI;
        } else if (HasSMUL_LOHI) {
          OpToUse = ISD::SMUL_LOHI;
        } else if (HasUMUL_LOHI) {
          OpToUse = ISD::UMUL_LOHI;
        }
        if (OpToUse) {
          Result = SDOperand(DAG.getNode(OpToUse, VTs, Tmp1, Tmp2).Val, 0);
          break;
        }
      }
      if (Node->getOpcode() == ISD::MULHS &&
          TLI.isOperationLegal(ISD::SMUL_LOHI, VT)) {
        Result = SDOperand(DAG.getNode(ISD::SMUL_LOHI, VTs, Tmp1, Tmp2).Val, 1);
        break;
      }
      if (Node->getOpcode() == ISD::MULHU && 
          TLI.isOperationLegal(ISD::UMUL_LOHI, VT)) {
        Result = SDOperand(DAG.getNode(ISD::UMUL_LOHI, VTs, Tmp1, Tmp2).Val, 1);
        break;
      }
      if (Node->getOpcode() == ISD::SDIV &&
          TLI.isOperationLegal(ISD::SDIVREM, VT)) {
        Result = SDOperand(DAG.getNode(ISD::SDIVREM, VTs, Tmp1, Tmp2).Val, 0);
        break;
      }
      if (Node->getOpcode() == ISD::UDIV &&
          TLI.isOperationLegal(ISD::UDIVREM, VT)) {
        Result = SDOperand(DAG.getNode(ISD::UDIVREM, VTs, Tmp1, Tmp2).Val, 0);
        break;
      }

      // Check to see if we have a libcall for this operator.
      RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
      bool isSigned = false;
      switch (Node->getOpcode()) {
      case ISD::UDIV:
      case ISD::SDIV:
        if (VT == MVT::i32) {
          LC = Node->getOpcode() == ISD::UDIV
            ? RTLIB::UDIV_I32 : RTLIB::SDIV_I32;
          isSigned = Node->getOpcode() == ISD::SDIV;
        }
        break;
      case ISD::FPOW:
        LC = GetFPLibCall(VT, RTLIB::POW_F32, RTLIB::POW_F64, RTLIB::POW_F80,
                          RTLIB::POW_PPCF128);
        break;
      default: break;
      }
      if (LC != RTLIB::UNKNOWN_LIBCALL) {
        SDOperand Dummy;
        Result = ExpandLibCall(LC, Node, isSigned, Dummy);
        break;
      }

      assert(MVT::isVector(Node->getValueType(0)) &&
             "Cannot expand this binary operator!");
      // Expand the operation into a bunch of nasty scalar code.
      Result = LegalizeOp(UnrollVectorOp(Op));
      break;
    }
    case TargetLowering::Promote: {
      switch (Node->getOpcode()) {
      default:  assert(0 && "Do not know how to promote this BinOp!");
      case ISD::AND:
      case ISD::OR:
      case ISD::XOR: {
        MVT::ValueType OVT = Node->getValueType(0);
        MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);
        assert(MVT::isVector(OVT) && "Cannot promote this BinOp!");
        // Bit convert each of the values to the new type.
        Tmp1 = DAG.getNode(ISD::BIT_CONVERT, NVT, Tmp1);
        Tmp2 = DAG.getNode(ISD::BIT_CONVERT, NVT, Tmp2);
        Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
        // Bit convert the result back the original type.
        Result = DAG.getNode(ISD::BIT_CONVERT, OVT, Result);
        break;
      }
      }
    }
    }
    break;
    
  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
  case ISD::SDIVREM:
  case ISD::UDIVREM:
    // These nodes will only be produced by target-specific lowering, so
    // they shouldn't be here if they aren't legal.
    assert(TLI.isOperationLegal(Node->getOpcode(), Node->getValueType(0)) &&
           "This must be legal!");

    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    break;

  case ISD::FCOPYSIGN:  // FCOPYSIGN does not require LHS/RHS to match type!
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
      case Expand: assert(0 && "Not possible");
      case Legal:
        Tmp2 = LegalizeOp(Node->getOperand(1)); // Legalize the RHS.
        break;
      case Promote:
        Tmp2 = PromoteOp(Node->getOperand(1));  // Promote the RHS.
        break;
    }
      
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "Operation not supported");
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Legal: break;
    case TargetLowering::Expand: {
      // If this target supports fabs/fneg natively and select is cheap,
      // do this efficiently.
      if (!TLI.isSelectExpensive() &&
          TLI.getOperationAction(ISD::FABS, Tmp1.getValueType()) ==
          TargetLowering::Legal &&
          TLI.getOperationAction(ISD::FNEG, Tmp1.getValueType()) ==
          TargetLowering::Legal) {
        // Get the sign bit of the RHS.
        MVT::ValueType IVT = 
          Tmp2.getValueType() == MVT::f32 ? MVT::i32 : MVT::i64;
        SDOperand SignBit = DAG.getNode(ISD::BIT_CONVERT, IVT, Tmp2);
        SignBit = DAG.getSetCC(TLI.getSetCCResultType(SignBit),
                               SignBit, DAG.getConstant(0, IVT), ISD::SETLT);
        // Get the absolute value of the result.
        SDOperand AbsVal = DAG.getNode(ISD::FABS, Tmp1.getValueType(), Tmp1);
        // Select between the nabs and abs value based on the sign bit of
        // the input.
        Result = DAG.getNode(ISD::SELECT, AbsVal.getValueType(), SignBit,
                             DAG.getNode(ISD::FNEG, AbsVal.getValueType(), 
                                         AbsVal),
                             AbsVal);
        Result = LegalizeOp(Result);
        break;
      }
      
      // Otherwise, do bitwise ops!
      MVT::ValueType NVT = 
        Node->getValueType(0) == MVT::f32 ? MVT::i32 : MVT::i64;
      Result = ExpandFCOPYSIGNToBitwiseOps(Node, NVT, DAG, TLI);
      Result = DAG.getNode(ISD::BIT_CONVERT, Node->getValueType(0), Result);
      Result = LegalizeOp(Result);
      break;
    }
    }
    break;
    
  case ISD::ADDC:
  case ISD::SUBC:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    // Since this produces two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result;

  case ISD::ADDE:
  case ISD::SUBE:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Tmp3 = LegalizeOp(Node->getOperand(2));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    // Since this produces two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDOperand(Node, 1), Result.getValue(1));
    return Result;
    
  case ISD::BUILD_PAIR: {
    MVT::ValueType PairTy = Node->getValueType(0);
    // TODO: handle the case where the Lo and Hi operands are not of legal type
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Lo
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Hi
    switch (TLI.getOperationAction(ISD::BUILD_PAIR, PairTy)) {
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom this yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::BUILD_PAIR, PairTy, Tmp1, Tmp2);
      break;
    case TargetLowering::Expand:
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, PairTy, Tmp1);
      Tmp2 = DAG.getNode(ISD::ANY_EXTEND, PairTy, Tmp2);
      Tmp2 = DAG.getNode(ISD::SHL, PairTy, Tmp2,
                         DAG.getConstant(MVT::getSizeInBits(PairTy)/2, 
                                         TLI.getShiftAmountTy()));
      Result = DAG.getNode(ISD::OR, PairTy, Tmp1, Tmp2);
      break;
    }
    break;
  }

  case ISD::UREM:
  case ISD::SREM:
  case ISD::FREM:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Promote: assert(0 && "Cannot promote this yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand: {
      unsigned DivOpc= (Node->getOpcode() == ISD::UREM) ? ISD::UDIV : ISD::SDIV;
      bool isSigned = DivOpc == ISD::SDIV;
      MVT::ValueType VT = Node->getValueType(0);
 
      // See if remainder can be lowered using two-result operations.
      SDVTList VTs = DAG.getVTList(VT, VT);
      if (Node->getOpcode() == ISD::SREM &&
          TLI.isOperationLegal(ISD::SDIVREM, VT)) {
        Result = SDOperand(DAG.getNode(ISD::SDIVREM, VTs, Tmp1, Tmp2).Val, 1);
        break;
      }
      if (Node->getOpcode() == ISD::UREM &&
          TLI.isOperationLegal(ISD::UDIVREM, VT)) {
        Result = SDOperand(DAG.getNode(ISD::UDIVREM, VTs, Tmp1, Tmp2).Val, 1);
        break;
      }

      if (MVT::isInteger(VT)) {
        if (TLI.getOperationAction(DivOpc, VT) ==
            TargetLowering::Legal) {
          // X % Y -> X-X/Y*Y
          Result = DAG.getNode(DivOpc, VT, Tmp1, Tmp2);
          Result = DAG.getNode(ISD::MUL, VT, Result, Tmp2);
          Result = DAG.getNode(ISD::SUB, VT, Tmp1, Result);
        } else if (MVT::isVector(VT)) {
          Result = LegalizeOp(UnrollVectorOp(Op));
        } else {
          assert(VT == MVT::i32 &&
                 "Cannot expand this binary operator!");
          RTLIB::Libcall LC = Node->getOpcode() == ISD::UREM
            ? RTLIB::UREM_I32 : RTLIB::SREM_I32;
          SDOperand Dummy;
          Result = ExpandLibCall(LC, Node, isSigned, Dummy);
        }
      } else {
        assert(MVT::isFloatingPoint(VT) &&
               "remainder op must have integer or floating-point type");
        if (MVT::isVector(VT)) {
          Result = LegalizeOp(UnrollVectorOp(Op));
        } else {
          // Floating point mod -> fmod libcall.
          RTLIB::Libcall LC = GetFPLibCall(VT, RTLIB::REM_F32, RTLIB::REM_F64,
                                           RTLIB::REM_F80, RTLIB::REM_PPCF128);
          SDOperand Dummy;
          Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
        }
      }
      break;
    }
    }
    break;
  case ISD::VAARG: {
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));
      Result = Result.getValue(0);
      Tmp1 = Result.getValue(1);

      if (isCustom) {
        Tmp2 = TLI.LowerOperation(Result, DAG);
        if (Tmp2.Val) {
          Result = LegalizeOp(Tmp2);
          Tmp1 = LegalizeOp(Tmp2.getValue(1));
        }
      }
      break;
    case TargetLowering::Expand: {
      const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
      SDOperand VAList = DAG.getLoad(TLI.getPointerTy(), Tmp1, Tmp2, V, 0);
      // Increment the pointer, VAList, to the next vaarg
      Tmp3 = DAG.getNode(ISD::ADD, TLI.getPointerTy(), VAList, 
                         DAG.getConstant(MVT::getSizeInBits(VT)/8, 
                                         TLI.getPointerTy()));
      // Store the incremented VAList to the legalized pointer
      Tmp3 = DAG.getStore(VAList.getValue(1), Tmp3, Tmp2, V, 0);
      // Load the actual argument out of the pointer VAList
      Result = DAG.getLoad(VT, Tmp3, VAList, NULL, 0);
      Tmp1 = LegalizeOp(Result.getValue(1));
      Result = LegalizeOp(Result);
      break;
    }
    }
    // Since VAARG produces two values, make sure to remember that we 
    // legalized both of them.
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Tmp1);
    return Op.ResNo ? Tmp1 : Result;
  }
    
  case ISD::VACOPY: 
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the dest pointer.
    Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the source pointer.

    switch (TLI.getOperationAction(ISD::VACOPY, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3,
                                      Node->getOperand(3), Node->getOperand(4));
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      // This defaults to loading a pointer from the input and storing it to the
      // output, returning the chain.
      const Value *VD = cast<SrcValueSDNode>(Node->getOperand(3))->getValue();
      const Value *VS = cast<SrcValueSDNode>(Node->getOperand(4))->getValue();
      Tmp4 = DAG.getLoad(TLI.getPointerTy(), Tmp1, Tmp3, VS, 0);
      Result = DAG.getStore(Tmp4.getValue(1), Tmp4, Tmp2, VD, 0);
      break;
    }
    break;

  case ISD::VAEND: 
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    switch (TLI.getOperationAction(ISD::VAEND, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Tmp1, DAG);
        if (Tmp1.Val) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      Result = Tmp1; // Default to a no-op, return the chain
      break;
    }
    break;
    
  case ISD::VASTART: 
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));
    
    switch (TLI.getOperationAction(ISD::VASTART, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    }
    break;
    
  case ISD::ROTL:
  case ISD::ROTR:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default:
      assert(0 && "ROTL/ROTR legalize operation not supported");
      break;
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.Val) Result = Tmp1;
      break;
    case TargetLowering::Promote:
      assert(0 && "Do not know how to promote ROTL/ROTR");
      break;
    case TargetLowering::Expand:
      assert(0 && "Do not know how to expand ROTL/ROTR");
      break;
    }
    break;
    
  case ISD::BSWAP:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Op
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Custom:
      assert(0 && "Cannot custom legalize this yet!");
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    case TargetLowering::Promote: {
      MVT::ValueType OVT = Tmp1.getValueType();
      MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);
      unsigned DiffBits = MVT::getSizeInBits(NVT) - MVT::getSizeInBits(OVT);

      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Tmp1);
      Tmp1 = DAG.getNode(ISD::BSWAP, NVT, Tmp1);
      Result = DAG.getNode(ISD::SRL, NVT, Tmp1,
                           DAG.getConstant(DiffBits, TLI.getShiftAmountTy()));
      break;
    }
    case TargetLowering::Expand:
      Result = ExpandBSWAP(Tmp1);
      break;
    }
    break;
    
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Op
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Custom:
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      if (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0)) ==
          TargetLowering::Custom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) {
          Result = Tmp1;
        }
      }
      break;
    case TargetLowering::Promote: {
      MVT::ValueType OVT = Tmp1.getValueType();
      MVT::ValueType NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Zero extend the argument.
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Tmp1);
      // Perform the larger operation, then subtract if needed.
      Tmp1 = DAG.getNode(Node->getOpcode(), Node->getValueType(0), Tmp1);
      switch (Node->getOpcode()) {
      case ISD::CTPOP:
        Result = Tmp1;
        break;
      case ISD::CTTZ:
        //if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
        Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(Tmp1), Tmp1,
                            DAG.getConstant(MVT::getSizeInBits(NVT), NVT),
                            ISD::SETEQ);
        Result = DAG.getNode(ISD::SELECT, NVT, Tmp2,
                             DAG.getConstant(MVT::getSizeInBits(OVT),NVT), Tmp1);
        break;
      case ISD::CTLZ:
        // Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
        Result = DAG.getNode(ISD::SUB, NVT, Tmp1,
                             DAG.getConstant(MVT::getSizeInBits(NVT) -
                                             MVT::getSizeInBits(OVT), NVT));
        break;
      }
      break;
    }
    case TargetLowering::Expand:
      Result = ExpandBitCount(Node->getOpcode(), Tmp1);
      break;
    }
    break;

    // Unary operators
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    case TargetLowering::Promote:
    case TargetLowering::Custom:
     isCustom = true;
     // FALLTHROUGH
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      switch (Node->getOpcode()) {
      default: assert(0 && "Unreachable!");
      case ISD::FNEG:
        // Expand Y = FNEG(X) ->  Y = SUB -0.0, X
        Tmp2 = DAG.getConstantFP(-0.0, Node->getValueType(0));
        Result = DAG.getNode(ISD::FSUB, Node->getValueType(0), Tmp2, Tmp1);
        break;
      case ISD::FABS: {
        // Expand Y = FABS(X) -> Y = (X >u 0.0) ? X : fneg(X).
        MVT::ValueType VT = Node->getValueType(0);
        Tmp2 = DAG.getConstantFP(0.0, VT);
        Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(Tmp1), Tmp1, Tmp2,
                            ISD::SETUGT);
        Tmp3 = DAG.getNode(ISD::FNEG, VT, Tmp1);
        Result = DAG.getNode(ISD::SELECT, VT, Tmp2, Tmp1, Tmp3);
        break;
      }
      case ISD::FSQRT:
      case ISD::FSIN:
      case ISD::FCOS: {
        MVT::ValueType VT = Node->getValueType(0);

        // Expand unsupported unary vector operators by unrolling them.
        if (MVT::isVector(VT)) {
          Result = LegalizeOp(UnrollVectorOp(Op));
          break;
        }

        RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
        switch(Node->getOpcode()) {
        case ISD::FSQRT:
          LC = GetFPLibCall(VT, RTLIB::SQRT_F32, RTLIB::SQRT_F64,
                            RTLIB::SQRT_F80, RTLIB::SQRT_PPCF128);
          break;
        case ISD::FSIN:
          LC = GetFPLibCall(VT, RTLIB::SIN_F32, RTLIB::SIN_F64,
                            RTLIB::SIN_F80, RTLIB::SIN_PPCF128);
          break;
        case ISD::FCOS:
          LC = GetFPLibCall(VT, RTLIB::COS_F32, RTLIB::COS_F64,
                            RTLIB::COS_F80, RTLIB::COS_PPCF128);
          break;
        default: assert(0 && "Unreachable!");
        }
        SDOperand Dummy;
        Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
        break;
      }
      }
      break;
    }
    break;
  case ISD::FPOWI: {
    MVT::ValueType VT = Node->getValueType(0);

    // Expand unsupported unary vector operators by unrolling them.
    if (MVT::isVector(VT)) {
      Result = LegalizeOp(UnrollVectorOp(Op));
      break;
    }

    // We always lower FPOWI into a libcall.  No target support for it yet.
    RTLIB::Libcall LC = GetFPLibCall(VT, RTLIB::POWI_F32, RTLIB::POWI_F64,
                                     RTLIB::POWI_F80, RTLIB::POWI_PPCF128);
    SDOperand Dummy;
    Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
    break;
  }
  case ISD::BIT_CONVERT:
    if (!isTypeLegal(Node->getOperand(0).getValueType())) {
      Result = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                                Node->getValueType(0));
    } else if (MVT::isVector(Op.getOperand(0).getValueType())) {
      // The input has to be a vector type, we have to either scalarize it, pack
      // it, or convert it based on whether the input vector type is legal.
      SDNode *InVal = Node->getOperand(0).Val;
      int InIx = Node->getOperand(0).ResNo;
      unsigned NumElems = MVT::getVectorNumElements(InVal->getValueType(InIx));
      MVT::ValueType EVT = MVT::getVectorElementType(InVal->getValueType(InIx));
    
      // Figure out if there is a simple type corresponding to this Vector
      // type.  If so, convert to the vector type.
      MVT::ValueType TVT = MVT::getVectorType(EVT, NumElems);
      if (TLI.isTypeLegal(TVT)) {
        // Turn this into a bit convert of the vector input.
        Result = DAG.getNode(ISD::BIT_CONVERT, Node->getValueType(0), 
                             LegalizeOp(Node->getOperand(0)));
        break;
      } else if (NumElems == 1) {
        // Turn this into a bit convert of the scalar input.
        Result = DAG.getNode(ISD::BIT_CONVERT, Node->getValueType(0), 
                             ScalarizeVectorOp(Node->getOperand(0)));
        break;
      } else {
        // FIXME: UNIMP!  Store then reload
        assert(0 && "Cast from unsupported vector type not implemented yet!");
      }
    } else {
      switch (TLI.getOperationAction(ISD::BIT_CONVERT,
                                     Node->getOperand(0).getValueType())) {
      default: assert(0 && "Unknown operation action!");
      case TargetLowering::Expand:
        Result = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                                  Node->getValueType(0));
        break;
      case TargetLowering::Legal:
        Tmp1 = LegalizeOp(Node->getOperand(0));
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
        break;
      }
    }
    break;
      
    // Conversion operators.  The source and destination have different types.
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    bool isSigned = Node->getOpcode() == ISD::SINT_TO_FP;
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      switch (TLI.getOperationAction(Node->getOpcode(),
                                     Node->getOperand(0).getValueType())) {
      default: assert(0 && "Unknown operation action!");
      case TargetLowering::Custom:
        isCustom = true;
        // FALLTHROUGH
      case TargetLowering::Legal:
        Tmp1 = LegalizeOp(Node->getOperand(0));
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
        if (isCustom) {
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.Val) Result = Tmp1;
        }
        break;
      case TargetLowering::Expand:
        Result = ExpandLegalINT_TO_FP(isSigned,
                                      LegalizeOp(Node->getOperand(0)),
                                      Node->getValueType(0));
        break;
      case TargetLowering::Promote:
        Result = PromoteLegalINT_TO_FP(LegalizeOp(Node->getOperand(0)),
                                       Node->getValueType(0),
                                       isSigned);
        break;
      }
      break;
    case Expand:
      Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP,
                             Node->getValueType(0), Node->getOperand(0));
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));
      if (isSigned) {
        Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, Tmp1.getValueType(),
                 Tmp1, DAG.getValueType(Node->getOperand(0).getValueType()));
      } else {
        Tmp1 = DAG.getZeroExtendInReg(Tmp1,
                                      Node->getOperand(0).getValueType());
      }
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      Result = LegalizeOp(Result);  // The 'op' is not necessarily legal!
      break;
    }
    break;
  }
  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);

      // Since the result is legal, we should just be able to truncate the low
      // part of the source.
      Result = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), Tmp1);
      break;
    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::TRUNCATE, Op.getValueType(), Result);
      break;
    }
    break;

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));

      switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))){
      default: assert(0 && "Unknown operation action!");
      case TargetLowering::Custom:
        isCustom = true;
        // FALLTHROUGH
      case TargetLowering::Legal:
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
        if (isCustom) {
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.Val) Result = Tmp1;
        }
        break;
      case TargetLowering::Promote:
        Result = PromoteLegalFP_TO_INT(Tmp1, Node->getValueType(0),
                                       Node->getOpcode() == ISD::FP_TO_SINT);
        break;
      case TargetLowering::Expand:
        if (Node->getOpcode() == ISD::FP_TO_UINT) {
          SDOperand True, False;
          MVT::ValueType VT =  Node->getOperand(0).getValueType();
          MVT::ValueType NVT = Node->getValueType(0);
          const uint64_t zero[] = {0, 0};
          APFloat apf = APFloat(APInt(MVT::getSizeInBits(VT), 2, zero));
          APInt x = APInt::getSignBit(MVT::getSizeInBits(NVT));
          (void)apf.convertFromAPInt(x, false, APFloat::rmNearestTiesToEven);
          Tmp2 = DAG.getConstantFP(apf, VT);
          Tmp3 = DAG.getSetCC(TLI.getSetCCResultType(Node->getOperand(0)),
                            Node->getOperand(0), Tmp2, ISD::SETLT);
          True = DAG.getNode(ISD::FP_TO_SINT, NVT, Node->getOperand(0));
          False = DAG.getNode(ISD::FP_TO_SINT, NVT,
                              DAG.getNode(ISD::FSUB, VT, Node->getOperand(0),
                                          Tmp2));
          False = DAG.getNode(ISD::XOR, NVT, False, 
                              DAG.getConstant(x, NVT));
          Result = DAG.getNode(ISD::SELECT, NVT, Tmp3, True, False);
          break;
        } else {
          assert(0 && "Do not know how to expand FP_TO_SINT yet!");
        }
        break;
      }
      break;
    case Expand: {
      MVT::ValueType VT = Op.getValueType();
      MVT::ValueType OVT = Node->getOperand(0).getValueType();
      // Convert ppcf128 to i32
      if (OVT == MVT::ppcf128 && VT == MVT::i32) {
        if (Node->getOpcode() == ISD::FP_TO_SINT) {
          Result = DAG.getNode(ISD::FP_ROUND_INREG, MVT::ppcf128, 
                               Node->getOperand(0), DAG.getValueType(MVT::f64));
          Result = DAG.getNode(ISD::FP_ROUND, MVT::f64, Result, 
                               DAG.getIntPtrConstant(1));
          Result = DAG.getNode(ISD::FP_TO_SINT, VT, Result);
        } else {
          const uint64_t TwoE31[] = {0x41e0000000000000LL, 0};
          APFloat apf = APFloat(APInt(128, 2, TwoE31));
          Tmp2 = DAG.getConstantFP(apf, OVT);
          //  X>=2^31 ? (int)(X-2^31)+0x80000000 : (int)X
          // FIXME: generated code sucks.
          Result = DAG.getNode(ISD::SELECT_CC, VT, Node->getOperand(0), Tmp2,
                               DAG.getNode(ISD::ADD, MVT::i32,
                                 DAG.getNode(ISD::FP_TO_SINT, VT,
                                   DAG.getNode(ISD::FSUB, OVT,
                                                 Node->getOperand(0), Tmp2)),
                                 DAG.getConstant(0x80000000, MVT::i32)),
                               DAG.getNode(ISD::FP_TO_SINT, VT, 
                                           Node->getOperand(0)),
                               DAG.getCondCode(ISD::SETGE));
        }
        break;
      }
      // Convert f32 / f64 to i32 / i64 / i128.
      RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
      switch (Node->getOpcode()) {
      case ISD::FP_TO_SINT: {
        if (VT == MVT::i32) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOSINT_F32_I32;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOSINT_F64_I32;
          else
            assert(0 && "Unexpected i32-to-fp conversion!");
        } else if (VT == MVT::i64) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOSINT_F32_I64;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOSINT_F64_I64;
          else if (OVT == MVT::f80)
            LC = RTLIB::FPTOSINT_F80_I64;
          else if (OVT == MVT::ppcf128)
            LC = RTLIB::FPTOSINT_PPCF128_I64;
          else
            assert(0 && "Unexpected i64-to-fp conversion!");
        } else if (VT == MVT::i128) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOSINT_F32_I128;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOSINT_F64_I128;
          else if (OVT == MVT::f80)
            LC = RTLIB::FPTOSINT_F80_I128;
          else if (OVT == MVT::ppcf128)
            LC = RTLIB::FPTOSINT_PPCF128_I128;
          else
            assert(0 && "Unexpected i128-to-fp conversion!");
        } else {
          assert(0 && "Unexpectd int-to-fp conversion!");
        }
        break;
      }
      case ISD::FP_TO_UINT: {
        if (VT == MVT::i32) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOUINT_F32_I32;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOUINT_F64_I32;
          else if (OVT == MVT::f80)
            LC = RTLIB::FPTOUINT_F80_I32;
          else
            assert(0 && "Unexpected i32-to-fp conversion!");
        } else if (VT == MVT::i64) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOUINT_F32_I64;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOUINT_F64_I64;
          else if (OVT == MVT::f80)
            LC = RTLIB::FPTOUINT_F80_I64;
          else if (OVT == MVT::ppcf128)
            LC = RTLIB::FPTOUINT_PPCF128_I64;
          else
            assert(0 && "Unexpected i64-to-fp conversion!");
        } else if (VT == MVT::i128) {
          if (OVT == MVT::f32)
            LC = RTLIB::FPTOUINT_F32_I128;
          else if (OVT == MVT::f64)
            LC = RTLIB::FPTOUINT_F64_I128;
          else if (OVT == MVT::f80)
            LC = RTLIB::FPTOUINT_F80_I128;
          else if (OVT == MVT::ppcf128)
            LC = RTLIB::FPTOUINT_PPCF128_I128;
          else
            assert(0 && "Unexpected i128-to-fp conversion!");
        } else {
          assert(0 && "Unexpectd int-to-fp conversion!");
        }
        break;
      }
      default: assert(0 && "Unreachable!");
      }
      SDOperand Dummy;
      Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
      break;
    }
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, LegalizeOp(Tmp1));
      Result = LegalizeOp(Result);
      break;
    }
    break;

  case ISD::FP_EXTEND: {
    MVT::ValueType DstVT = Op.getValueType();
    MVT::ValueType SrcVT = Op.getOperand(0).getValueType();
    if (TLI.getConvertAction(SrcVT, DstVT) == TargetLowering::Expand) {
      // The only other way we can lower this is to turn it into a STORE,
      // LOAD pair, targetting a temporary location (a stack slot).
      Result = EmitStackConvert(Node->getOperand(0), SrcVT, DstVT);
      break;
    }
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "Shouldn't need to expand other operators here!");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::FP_EXTEND, Op.getValueType(), Tmp1);
      break;
    }
    break;
  }
  case ISD::FP_ROUND: {
    MVT::ValueType DstVT = Op.getValueType();
    MVT::ValueType SrcVT = Op.getOperand(0).getValueType();
    if (TLI.getConvertAction(SrcVT, DstVT) == TargetLowering::Expand) {
      if (SrcVT == MVT::ppcf128) {
        SDOperand Lo;
        ExpandOp(Node->getOperand(0), Lo, Result);
        // Round it the rest of the way (e.g. to f32) if needed.
        if (DstVT!=MVT::f64)
          Result = DAG.getNode(ISD::FP_ROUND, DstVT, Result, Op.getOperand(1));
        break;
      }
      // The only other way we can lower this is to turn it into a STORE,
      // LOAD pair, targetting a temporary location (a stack slot).
      Result = EmitStackConvert(Node->getOperand(0), DstVT, DstVT);
      break;
    }
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "Shouldn't need to expand other operators here!");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
      break;
    case Promote:
      Tmp1 = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::FP_ROUND, Op.getValueType(), Tmp1,
                           Node->getOperand(1));
      break;
    }
    break;
  }
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "Shouldn't need to expand other operators here!");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      if (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0)) ==
          TargetLowering::Custom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.Val) Result = Tmp1;
      }
      break;
    case Promote:
      switch (Node->getOpcode()) {
      case ISD::ANY_EXTEND:
        Tmp1 = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Tmp1);
        break;
      case ISD::ZERO_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Result);
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
        break;
      case ISD::SIGN_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, Op.getValueType(), Result);
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                          DAG.getValueType(Node->getOperand(0).getValueType()));
        break;
      }
    }
    break;
  case ISD::FP_ROUND_INREG:
  case ISD::SIGN_EXTEND_INREG: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    MVT::ValueType ExtraVT = cast<VTSDNode>(Node->getOperand(1))->getVT();

    // If this operation is not supported, convert it to a shl/shr or load/store
    // pair.
    switch (TLI.getOperationAction(Node->getOpcode(), ExtraVT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Legal:
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
      break;
    case TargetLowering::Expand:
      // If this is an integer extend and shifts are supported, do that.
      if (Node->getOpcode() == ISD::SIGN_EXTEND_INREG) {
        // NOTE: we could fall back on load/store here too for targets without
        // SAR.  However, it is doubtful that any exist.
        unsigned BitsDiff = MVT::getSizeInBits(Node->getValueType(0)) -
                            MVT::getSizeInBits(ExtraVT);
        SDOperand ShiftCst = DAG.getConstant(BitsDiff, TLI.getShiftAmountTy());
        Result = DAG.getNode(ISD::SHL, Node->getValueType(0),
                             Node->getOperand(0), ShiftCst);
        Result = DAG.getNode(ISD::SRA, Node->getValueType(0),
                             Result, ShiftCst);
      } else if (Node->getOpcode() == ISD::FP_ROUND_INREG) {
        // The only way we can lower this is to turn it into a TRUNCSTORE,
        // EXTLOAD pair, targetting a temporary location (a stack slot).

        // NOTE: there is a choice here between constantly creating new stack
        // slots and always reusing the same one.  We currently always create
        // new ones, as reuse may inhibit scheduling.
        Result = EmitStackConvert(Node->getOperand(0), ExtraVT, 
                                  Node->getValueType(0));
      } else {
        assert(0 && "Unknown op");
      }
      break;
    }
    break;
  }
  case ISD::TRAMPOLINE: {
    SDOperand Ops[6];
    for (unsigned i = 0; i != 6; ++i)
      Ops[i] = LegalizeOp(Node->getOperand(i));
    Result = DAG.UpdateNodeOperands(Result, Ops, 6);
    // The only option for this node is to custom lower it.
    Result = TLI.LowerOperation(Result, DAG);
    assert(Result.Val && "Should always custom lower!");

    // Since trampoline produces two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result.getValue(1));
    Result = LegalizeOp(Result);
    AddLegalizedOperand(SDOperand(Node, 0), Result);
    AddLegalizedOperand(SDOperand(Node, 1), Tmp1);
    return Op.ResNo ? Tmp1 : Result;
  }
   case ISD::FLT_ROUNDS_: {
    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Legal:
      // If this operation is not supported, lower it to constant 1
      Result = DAG.getConstant(1, VT);
      break;
    }
    break;
  }
  case ISD::TRAP: {
    MVT::ValueType VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.Val) break;
      // Fall Thru
    case TargetLowering::Expand:
      // If this operation is not supported, lower it to 'abort()' call
      Tmp1 = LegalizeOp(Node->getOperand(0));
      TargetLowering::ArgListTy Args;
      std::pair<SDOperand,SDOperand> CallResult =
        TLI.LowerCallTo(Tmp1, Type::VoidTy,
                        false, false, false, CallingConv::C, false,
                        DAG.getExternalSymbol("abort", TLI.getPointerTy()),
                        Args, DAG);
      Result = CallResult.second;
      break;
    }
    break;
  }
  }
  
  assert(Result.getValueType() == Op.getValueType() &&
         "Bad legalization!");
  
  // Make sure that the generated code is itself legal.
  if (Result != Op)
    Result = LegalizeOp(Result);

  // Note that LegalizeOp may be reentered even from single-use nodes, which
  // means that we always must cache transformed nodes.
  AddLegalizedOperand(Op, Result);
  return Result;
}

/// PromoteOp - Given an operation that produces a value in an invalid type,
/// promote it to compute the value into a larger type.  The produced value will
/// have the correct bits for the low portion of the register, but no guarantee
/// is made about the top bits: it may be zero, sign-extended, or garbage.
SDOperand SelectionDAGLegalize::PromoteOp(SDOperand Op) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  assert(getTypeAction(VT) == Promote &&
         "Caller should expand or legalize operands that are not promotable!");
  assert(NVT > VT && MVT::isInteger(NVT) == MVT::isInteger(VT) &&
         "Cannot promote to smaller type!");

  SDOperand Tmp1, Tmp2, Tmp3;
  SDOperand Result;
  SDNode *Node = Op.Val;

  DenseMap<SDOperand, SDOperand>::iterator I = PromotedNodes.find(Op);
  if (I != PromotedNodes.end()) return I->second;

  switch (Node->getOpcode()) {
  case ISD::CopyFromReg:
    assert(0 && "CopyFromReg must be legal!");
  default:
#ifndef NDEBUG
    cerr << "NODE: "; Node->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::UNDEF:
    Result = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::Constant:
    if (VT != MVT::i1)
      Result = DAG.getNode(ISD::SIGN_EXTEND, NVT, Op);
    else
      Result = DAG.getNode(ISD::ZERO_EXTEND, NVT, Op);
    assert(isa<ConstantSDNode>(Result) && "Didn't constant fold zext?");
    break;
  case ISD::ConstantFP:
    Result = DAG.getNode(ISD::FP_EXTEND, NVT, Op);
    assert(isa<ConstantFPSDNode>(Result) && "Didn't constant fold fp_extend?");
    break;

  case ISD::SETCC:
    assert(isTypeLegal(TLI.getSetCCResultType(Node->getOperand(0)))
           && "SetCC type is not legal??");
    Result = DAG.getNode(ISD::SETCC,
                         TLI.getSetCCResultType(Node->getOperand(0)),
                         Node->getOperand(0), Node->getOperand(1),
                         Node->getOperand(2));
    break;
    
  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Result = LegalizeOp(Node->getOperand(0));
      assert(Result.getValueType() >= NVT &&
             "This truncation doesn't make sense!");
      if (Result.getValueType() > NVT)    // Truncate to NVT instead of VT
        Result = DAG.getNode(ISD::TRUNCATE, NVT, Result);
      break;
    case Promote:
      // The truncation is not required, because we don't guarantee anything
      // about high bits anyway.
      Result = PromoteOp(Node->getOperand(0));
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      // Truncate the low part of the expanded value to the result type
      Result = DAG.getNode(ISD::TRUNCATE, NVT, Tmp1);
    }
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "BUG: Smaller reg should have been promoted!");
    case Legal:
      // Input is legal?  Just do extend all the way to the larger type.
      Result = DAG.getNode(Node->getOpcode(), NVT, Node->getOperand(0));
      break;
    case Promote:
      // Promote the reg if it's smaller.
      Result = PromoteOp(Node->getOperand(0));
      // The high bits are not guaranteed to be anything.  Insert an extend.
      if (Node->getOpcode() == ISD::SIGN_EXTEND)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else if (Node->getOpcode() == ISD::ZERO_EXTEND)
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
      break;
    }
    break;
  case ISD::BIT_CONVERT:
    Result = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                              Node->getValueType(0));
    Result = PromoteOp(Result);
    break;
    
  case ISD::FP_EXTEND:
    assert(0 && "Case not implemented.  Dynamically dead with 2 FP types!");
  case ISD::FP_ROUND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "BUG: Cannot expand FP regs!");
    case Promote:  assert(0 && "Unreachable with 2 FP types!");
    case Legal:
      if (Node->getConstantOperandVal(1) == 0) {
        // Input is legal?  Do an FP_ROUND_INREG.
        Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Node->getOperand(0),
                             DAG.getValueType(VT));
      } else {
        // Just remove the truncate, it isn't affecting the value.
        Result = DAG.getNode(ISD::FP_ROUND, NVT, Node->getOperand(0), 
                             Node->getOperand(1));
      }
      break;
    }
    break;
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      // No extra round required here.
      Result = DAG.getNode(Node->getOpcode(), NVT, Node->getOperand(0));
      break;

    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      if (Node->getOpcode() == ISD::SINT_TO_FP)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, Result.getValueType(),
                             Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else
        Result = DAG.getZeroExtendInReg(Result,
                                        Node->getOperand(0).getValueType());
      // No extra round required here.
      Result = DAG.getNode(Node->getOpcode(), NVT, Result);
      break;
    case Expand:
      Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP, NVT,
                             Node->getOperand(0));
      // Round if we cannot tolerate excess precision.
      if (NoExcessFPPrecision)
        Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                             DAG.getValueType(VT));
      break;
    }
    break;

  case ISD::SIGN_EXTEND_INREG:
    Result = PromoteOp(Node->getOperand(0));
    Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Result, 
                         Node->getOperand(1));
    break;
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
    case Expand:
      Tmp1 = Node->getOperand(0);
      break;
    case Promote:
      // The input result is prerounded, so we don't have to do anything
      // special.
      Tmp1 = PromoteOp(Node->getOperand(0));
      break;
    }
    // If we're promoting a UINT to a larger size, check to see if the new node
    // will be legal.  If it isn't, check to see if FP_TO_SINT is legal, since
    // we can use that instead.  This allows us to generate better code for
    // FP_TO_UINT for small destination sizes on targets where FP_TO_UINT is not
    // legal, such as PowerPC.
    if (Node->getOpcode() == ISD::FP_TO_UINT && 
        !TLI.isOperationLegal(ISD::FP_TO_UINT, NVT) &&
        (TLI.isOperationLegal(ISD::FP_TO_SINT, NVT) ||
         TLI.getOperationAction(ISD::FP_TO_SINT, NVT)==TargetLowering::Custom)){
      Result = DAG.getNode(ISD::FP_TO_SINT, NVT, Tmp1);
    } else {
      Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    }
    break;

  case ISD::FABS:
  case ISD::FNEG:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    // NOTE: we do not have to do any extra rounding here for
    // NoExcessFPPrecision, because we know the input will have the appropriate
    // precision, and these operations don't modify precision at all.
    break;

  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::FPOWI: {
    // Promote f32 powi to f64 powi.  Note that this could insert a libcall
    // directly as well, which may be better.
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(ISD::FPOWI, NVT, Tmp1, Node->getOperand(1));
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;
  }
    
  case ISD::ATOMIC_LCS: {
    Tmp2 = PromoteOp(Node->getOperand(2));
    Tmp3 = PromoteOp(Node->getOperand(3));
    Result = DAG.getAtomic(Node->getOpcode(), Node->getOperand(0), 
                           Node->getOperand(1), Tmp2, Tmp3,
                           cast<AtomicSDNode>(Node)->getVT());
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }
  case ISD::ATOMIC_LAS:
  case ISD::ATOMIC_LSS:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP: {
    Tmp2 = PromoteOp(Node->getOperand(2));
    Result = DAG.getAtomic(Node->getOpcode(), Node->getOperand(0), 
                           Node->getOperand(1), Tmp2,
                           cast<AtomicSDNode>(Node)->getVT());
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
    // The input may have strange things in the top bits of the registers, but
    // these operations don't care.  They may have weird bits going out, but
    // that too is okay if they are integer operations.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    
    // Floating point operations will give excess precision that we may not be
    // able to tolerate.  If we DO allow excess precision, just leave it,
    // otherwise excise it.
    // FIXME: Why would we need to round FP ops more than integer ones?
    //     Is Round(Add(Add(A,B),C)) != Round(Add(Round(Add(A,B)), C))
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::SDIV:
  case ISD::SREM:
    // These operators require that their input be sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    if (MVT::isInteger(NVT)) {
      Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                         DAG.getValueType(VT));
      Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp2,
                         DAG.getValueType(VT));
    }
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);

    // Perform FP_ROUND: this is probably overly pessimistic.
    if (MVT::isFloatingPoint(NVT) && NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::FCOPYSIGN:
    // These operators require that their input be fp extended.
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "not implemented");
    case Legal:   Tmp1 = LegalizeOp(Node->getOperand(0)); break;
    case Promote: Tmp1 = PromoteOp(Node->getOperand(0));  break;
    }
    switch (getTypeAction(Node->getOperand(1).getValueType())) {
    case Expand: assert(0 && "not implemented");
    case Legal:   Tmp2 = LegalizeOp(Node->getOperand(1)); break;
    case Promote: Tmp2 = PromoteOp(Node->getOperand(1)); break;
    }
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    
    // Perform FP_ROUND: this is probably overly pessimistic.
    if (NoExcessFPPrecision && Node->getOpcode() != ISD::FCOPYSIGN)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::UDIV:
  case ISD::UREM:
    // These operators require that their input be zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(MVT::isInteger(NVT) && "Operators don't apply to FP!");
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
    Tmp2 = DAG.getZeroExtendInReg(Tmp2, VT);
    Result = DAG.getNode(Node->getOpcode(), NVT, Tmp1, Tmp2);
    break;

  case ISD::SHL:
    Tmp1 = PromoteOp(Node->getOperand(0));
    Result = DAG.getNode(ISD::SHL, NVT, Tmp1, Node->getOperand(1));
    break;
  case ISD::SRA:
    // The input value must be properly sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                       DAG.getValueType(VT));
    Result = DAG.getNode(ISD::SRA, NVT, Tmp1, Node->getOperand(1));
    break;
  case ISD::SRL:
    // The input value must be properly zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
    Result = DAG.getNode(ISD::SRL, NVT, Tmp1, Node->getOperand(1));
    break;

  case ISD::VAARG:
    Tmp1 = Node->getOperand(0);   // Get the chain.
    Tmp2 = Node->getOperand(1);   // Get the pointer.
    if (TLI.getOperationAction(ISD::VAARG, VT) == TargetLowering::Custom) {
      Tmp3 = DAG.getVAArg(VT, Tmp1, Tmp2, Node->getOperand(2));
      Result = TLI.CustomPromoteOperation(Tmp3, DAG);
    } else {
      const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
      SDOperand VAList = DAG.getLoad(TLI.getPointerTy(), Tmp1, Tmp2, V, 0);
      // Increment the pointer, VAList, to the next vaarg
      Tmp3 = DAG.getNode(ISD::ADD, TLI.getPointerTy(), VAList, 
                         DAG.getConstant(MVT::getSizeInBits(VT)/8, 
                                         TLI.getPointerTy()));
      // Store the incremented VAList to the legalized pointer
      Tmp3 = DAG.getStore(VAList.getValue(1), Tmp3, Tmp2, V, 0);
      // Load the actual argument out of the pointer VAList
      Result = DAG.getExtLoad(ISD::EXTLOAD, NVT, Tmp3, VAList, NULL, 0, VT);
    }
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;

  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(Node)
      ? ISD::EXTLOAD : LD->getExtensionType();
    Result = DAG.getExtLoad(ExtType, NVT,
                            LD->getChain(), LD->getBasePtr(),
                            LD->getSrcValue(), LD->getSrcValueOffset(),
                            LD->getMemoryVT(),
                            LD->isVolatile(),
                            LD->getAlignment());
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }
  case ISD::SELECT:
    Tmp2 = PromoteOp(Node->getOperand(1));   // Legalize the op0
    Tmp3 = PromoteOp(Node->getOperand(2));   // Legalize the op1
    Result = DAG.getNode(ISD::SELECT, NVT, Node->getOperand(0), Tmp2, Tmp3);
    break;
  case ISD::SELECT_CC:
    Tmp2 = PromoteOp(Node->getOperand(2));   // True
    Tmp3 = PromoteOp(Node->getOperand(3));   // False
    Result = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                         Node->getOperand(1), Tmp2, Tmp3, Node->getOperand(4));
    break;
  case ISD::BSWAP:
    Tmp1 = Node->getOperand(0);
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Tmp1);
    Tmp1 = DAG.getNode(ISD::BSWAP, NVT, Tmp1);
    Result = DAG.getNode(ISD::SRL, NVT, Tmp1,
                         DAG.getConstant(MVT::getSizeInBits(NVT) -
                                         MVT::getSizeInBits(VT),
                                         TLI.getShiftAmountTy()));
    break;
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    // Zero extend the argument
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, NVT, Node->getOperand(0));
    // Perform the larger operation, then subtract if needed.
    Tmp1 = DAG.getNode(Node->getOpcode(), NVT, Tmp1);
    switch(Node->getOpcode()) {
    case ISD::CTPOP:
      Result = Tmp1;
      break;
    case ISD::CTTZ:
      // if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
      Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(Tmp1), Tmp1,
                          DAG.getConstant(MVT::getSizeInBits(NVT), NVT),
                          ISD::SETEQ);
      Result = DAG.getNode(ISD::SELECT, NVT, Tmp2,
                           DAG.getConstant(MVT::getSizeInBits(VT), NVT), Tmp1);
      break;
    case ISD::CTLZ:
      //Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
      Result = DAG.getNode(ISD::SUB, NVT, Tmp1,
                           DAG.getConstant(MVT::getSizeInBits(NVT) -
                                           MVT::getSizeInBits(VT), NVT));
      break;
    }
    break;
  case ISD::EXTRACT_SUBVECTOR:
    Result = PromoteOp(ExpandEXTRACT_SUBVECTOR(Op));
    break;
  case ISD::EXTRACT_VECTOR_ELT:
    Result = PromoteOp(ExpandEXTRACT_VECTOR_ELT(Op));
    break;
  }

  assert(Result.Val && "Didn't set a result!");

  // Make sure the result is itself legal.
  Result = LegalizeOp(Result);
  
  // Remember that we promoted this!
  AddPromotedOperand(Op, Result);
  return Result;
}

/// ExpandEXTRACT_VECTOR_ELT - Expand an EXTRACT_VECTOR_ELT operation into
/// a legal EXTRACT_VECTOR_ELT operation, scalar code, or memory traffic,
/// based on the vector type. The return type of this matches the element type
/// of the vector, which may not be legal for the target.
SDOperand SelectionDAGLegalize::ExpandEXTRACT_VECTOR_ELT(SDOperand Op) {
  // We know that operand #0 is the Vec vector.  If the index is a constant
  // or if the invec is a supported hardware type, we can use it.  Otherwise,
  // lower to a store then an indexed load.
  SDOperand Vec = Op.getOperand(0);
  SDOperand Idx = Op.getOperand(1);
  
  MVT::ValueType TVT = Vec.getValueType();
  unsigned NumElems = MVT::getVectorNumElements(TVT);
  
  switch (TLI.getOperationAction(ISD::EXTRACT_VECTOR_ELT, TVT)) {
  default: assert(0 && "This action is not supported yet!");
  case TargetLowering::Custom: {
    Vec = LegalizeOp(Vec);
    Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
    SDOperand Tmp3 = TLI.LowerOperation(Op, DAG);
    if (Tmp3.Val)
      return Tmp3;
    break;
  }
  case TargetLowering::Legal:
    if (isTypeLegal(TVT)) {
      Vec = LegalizeOp(Vec);
      Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
      return Op;
    }
    break;
  case TargetLowering::Expand:
    break;
  }

  if (NumElems == 1) {
    // This must be an access of the only element.  Return it.
    Op = ScalarizeVectorOp(Vec);
  } else if (!TLI.isTypeLegal(TVT) && isa<ConstantSDNode>(Idx)) {
    unsigned NumLoElts =  1 << Log2_32(NumElems-1);
    ConstantSDNode *CIdx = cast<ConstantSDNode>(Idx);
    SDOperand Lo, Hi;
    SplitVectorOp(Vec, Lo, Hi);
    if (CIdx->getValue() < NumLoElts) {
      Vec = Lo;
    } else {
      Vec = Hi;
      Idx = DAG.getConstant(CIdx->getValue() - NumLoElts,
                            Idx.getValueType());
    }
  
    // It's now an extract from the appropriate high or low part.  Recurse.
    Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
    Op = ExpandEXTRACT_VECTOR_ELT(Op);
  } else {
    // Store the value to a temporary stack slot, then LOAD the scalar
    // element back out.
    SDOperand StackPtr = DAG.CreateStackTemporary(Vec.getValueType());
    SDOperand Ch = DAG.getStore(DAG.getEntryNode(), Vec, StackPtr, NULL, 0);

    // Add the offset to the index.
    unsigned EltSize = MVT::getSizeInBits(Op.getValueType())/8;
    Idx = DAG.getNode(ISD::MUL, Idx.getValueType(), Idx,
                      DAG.getConstant(EltSize, Idx.getValueType()));

    if (MVT::getSizeInBits(Idx.getValueType()) >
        MVT::getSizeInBits(TLI.getPointerTy()))
      Idx = DAG.getNode(ISD::TRUNCATE, TLI.getPointerTy(), Idx);
    else
      Idx = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(), Idx);

    StackPtr = DAG.getNode(ISD::ADD, Idx.getValueType(), Idx, StackPtr);

    Op = DAG.getLoad(Op.getValueType(), Ch, StackPtr, NULL, 0);
  }
  return Op;
}

/// ExpandEXTRACT_SUBVECTOR - Expand a EXTRACT_SUBVECTOR operation.  For now
/// we assume the operation can be split if it is not already legal.
SDOperand SelectionDAGLegalize::ExpandEXTRACT_SUBVECTOR(SDOperand Op) {
  // We know that operand #0 is the Vec vector.  For now we assume the index
  // is a constant and that the extracted result is a supported hardware type.
  SDOperand Vec = Op.getOperand(0);
  SDOperand Idx = LegalizeOp(Op.getOperand(1));
  
  unsigned NumElems = MVT::getVectorNumElements(Vec.getValueType());
  
  if (NumElems == MVT::getVectorNumElements(Op.getValueType())) {
    // This must be an access of the desired vector length.  Return it.
    return Vec;
  }

  ConstantSDNode *CIdx = cast<ConstantSDNode>(Idx);
  SDOperand Lo, Hi;
  SplitVectorOp(Vec, Lo, Hi);
  if (CIdx->getValue() < NumElems/2) {
    Vec = Lo;
  } else {
    Vec = Hi;
    Idx = DAG.getConstant(CIdx->getValue() - NumElems/2, Idx.getValueType());
  }
  
  // It's now an extract from the appropriate high or low part.  Recurse.
  Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
  return ExpandEXTRACT_SUBVECTOR(Op);
}

/// LegalizeSetCCOperands - Attempts to create a legal LHS and RHS for a SETCC
/// with condition CC on the current target.  This usually involves legalizing
/// or promoting the arguments.  In the case where LHS and RHS must be expanded,
/// there may be no choice but to create a new SetCC node to represent the
/// legalized value of setcc lhs, rhs.  In this case, the value is returned in
/// LHS, and the SDOperand returned in RHS has a nil SDNode value.
void SelectionDAGLegalize::LegalizeSetCCOperands(SDOperand &LHS,
                                                 SDOperand &RHS,
                                                 SDOperand &CC) {
  SDOperand Tmp1, Tmp2, Tmp3, Result;    
  
  switch (getTypeAction(LHS.getValueType())) {
  case Legal:
    Tmp1 = LegalizeOp(LHS);   // LHS
    Tmp2 = LegalizeOp(RHS);   // RHS
    break;
  case Promote:
    Tmp1 = PromoteOp(LHS);   // LHS
    Tmp2 = PromoteOp(RHS);   // RHS

    // If this is an FP compare, the operands have already been extended.
    if (MVT::isInteger(LHS.getValueType())) {
      MVT::ValueType VT = LHS.getValueType();
      MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);

      // Otherwise, we have to insert explicit sign or zero extends.  Note
      // that we could insert sign extends for ALL conditions, but zero extend
      // is cheaper on many machines (an AND instead of two shifts), so prefer
      // it.
      switch (cast<CondCodeSDNode>(CC)->get()) {
      default: assert(0 && "Unknown integer comparison!");
      case ISD::SETEQ:
      case ISD::SETNE:
      case ISD::SETUGE:
      case ISD::SETUGT:
      case ISD::SETULE:
      case ISD::SETULT:
        // ALL of these operations will work if we either sign or zero extend
        // the operands (including the unsigned comparisons!).  Zero extend is
        // usually a simpler/cheaper operation, so prefer it.
        Tmp1 = DAG.getZeroExtendInReg(Tmp1, VT);
        Tmp2 = DAG.getZeroExtendInReg(Tmp2, VT);
        break;
      case ISD::SETGE:
      case ISD::SETGT:
      case ISD::SETLT:
      case ISD::SETLE:
        Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp1,
                           DAG.getValueType(VT));
        Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Tmp2,
                           DAG.getValueType(VT));
        break;
      }
    }
    break;
  case Expand: {
    MVT::ValueType VT = LHS.getValueType();
    if (VT == MVT::f32 || VT == MVT::f64) {
      // Expand into one or more soft-fp libcall(s).
      RTLIB::Libcall LC1, LC2 = RTLIB::UNKNOWN_LIBCALL;
      switch (cast<CondCodeSDNode>(CC)->get()) {
      case ISD::SETEQ:
      case ISD::SETOEQ:
        LC1 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
        break;
      case ISD::SETNE:
      case ISD::SETUNE:
        LC1 = (VT == MVT::f32) ? RTLIB::UNE_F32 : RTLIB::UNE_F64;
        break;
      case ISD::SETGE:
      case ISD::SETOGE:
        LC1 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
        break;
      case ISD::SETLT:
      case ISD::SETOLT:
        LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
        break;
      case ISD::SETLE:
      case ISD::SETOLE:
        LC1 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
        break;
      case ISD::SETGT:
      case ISD::SETOGT:
        LC1 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
        break;
      case ISD::SETUO:
        LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
        break;
      case ISD::SETO:
        LC1 = (VT == MVT::f32) ? RTLIB::O_F32 : RTLIB::O_F64;
        break;
      default:
        LC1 = (VT == MVT::f32) ? RTLIB::UO_F32 : RTLIB::UO_F64;
        switch (cast<CondCodeSDNode>(CC)->get()) {
        case ISD::SETONE:
          // SETONE = SETOLT | SETOGT
          LC1 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
          // Fallthrough
        case ISD::SETUGT:
          LC2 = (VT == MVT::f32) ? RTLIB::OGT_F32 : RTLIB::OGT_F64;
          break;
        case ISD::SETUGE:
          LC2 = (VT == MVT::f32) ? RTLIB::OGE_F32 : RTLIB::OGE_F64;
          break;
        case ISD::SETULT:
          LC2 = (VT == MVT::f32) ? RTLIB::OLT_F32 : RTLIB::OLT_F64;
          break;
        case ISD::SETULE:
          LC2 = (VT == MVT::f32) ? RTLIB::OLE_F32 : RTLIB::OLE_F64;
          break;
        case ISD::SETUEQ:
          LC2 = (VT == MVT::f32) ? RTLIB::OEQ_F32 : RTLIB::OEQ_F64;
          break;
        default: assert(0 && "Unsupported FP setcc!");
        }
      }
      
      SDOperand Dummy;
      Tmp1 = ExpandLibCall(LC1,
                           DAG.getNode(ISD::MERGE_VALUES, VT, LHS, RHS).Val, 
                           false /*sign irrelevant*/, Dummy);
      Tmp2 = DAG.getConstant(0, MVT::i32);
      CC = DAG.getCondCode(TLI.getCmpLibcallCC(LC1));
      if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
        Tmp1 = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(Tmp1), Tmp1, Tmp2,
                           CC);
        LHS = ExpandLibCall(LC2,
                            DAG.getNode(ISD::MERGE_VALUES, VT, LHS, RHS).Val,
                            false /*sign irrelevant*/, Dummy);
        Tmp2 = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(LHS), LHS, Tmp2,
                           DAG.getCondCode(TLI.getCmpLibcallCC(LC2)));
        Tmp1 = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp2);
        Tmp2 = SDOperand();
      }
      LHS = Tmp1;
      RHS = Tmp2;
      return;
    }

    SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
    ExpandOp(LHS, LHSLo, LHSHi);
    ExpandOp(RHS, RHSLo, RHSHi);
    ISD::CondCode CCCode = cast<CondCodeSDNode>(CC)->get();

    if (VT==MVT::ppcf128) {
      // FIXME:  This generated code sucks.  We want to generate
      //         FCMP crN, hi1, hi2
      //         BNE crN, L:
      //         FCMP crN, lo1, lo2
      // The following can be improved, but not that much.
      Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETEQ);
      Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSLo), LHSLo, RHSLo, CCCode);
      Tmp3 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
      Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, ISD::SETNE);
      Tmp2 = DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi, CCCode);
      Tmp1 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
      Tmp1 = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp3);
      Tmp2 = SDOperand();
      break;
    }

    switch (CCCode) {
    case ISD::SETEQ:
    case ISD::SETNE:
      if (RHSLo == RHSHi)
        if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
          if (RHSCST->isAllOnesValue()) {
            // Comparison to -1.
            Tmp1 = DAG.getNode(ISD::AND, LHSLo.getValueType(), LHSLo, LHSHi);
            Tmp2 = RHSLo;
            break;
          }

      Tmp1 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSLo, RHSLo);
      Tmp2 = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSHi, RHSHi);
      Tmp1 = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp2);
      Tmp2 = DAG.getConstant(0, Tmp1.getValueType());
      break;
    default:
      // If this is a comparison of the sign bit, just look at the top part.
      // X > -1,  x < 0
      if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(RHS))
        if ((cast<CondCodeSDNode>(CC)->get() == ISD::SETLT && 
             CST->isNullValue()) ||               // X < 0
            (cast<CondCodeSDNode>(CC)->get() == ISD::SETGT &&
             CST->isAllOnesValue())) {            // X > -1
          Tmp1 = LHSHi;
          Tmp2 = RHSHi;
          break;
        }

      // FIXME: This generated code sucks.
      ISD::CondCode LowCC;
      switch (CCCode) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETLT:
      case ISD::SETULT: LowCC = ISD::SETULT; break;
      case ISD::SETGT:
      case ISD::SETUGT: LowCC = ISD::SETUGT; break;
      case ISD::SETLE:
      case ISD::SETULE: LowCC = ISD::SETULE; break;
      case ISD::SETGE:
      case ISD::SETUGE: LowCC = ISD::SETUGE; break;
      }

      // Tmp1 = lo(op1) < lo(op2)   // Always unsigned comparison
      // Tmp2 = hi(op1) < hi(op2)   // Signedness depends on operands
      // dest = hi(op1) == hi(op2) ? Tmp1 : Tmp2;

      // NOTE: on targets without efficient SELECT of bools, we can always use
      // this identity: (B1 ? B2 : B3) --> (B1 & B2)|(!B1&B3)
      TargetLowering::DAGCombinerInfo DagCombineInfo(DAG, false, true, NULL);
      Tmp1 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSLo), LHSLo, RHSLo,
                               LowCC, false, DagCombineInfo);
      if (!Tmp1.Val)
        Tmp1 = DAG.getSetCC(TLI.getSetCCResultType(LHSLo), LHSLo, RHSLo, LowCC);
      Tmp2 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi,
                               CCCode, false, DagCombineInfo);
      if (!Tmp2.Val)
        Tmp2 = DAG.getNode(ISD::SETCC, TLI.getSetCCResultType(LHSHi), LHSHi,
                           RHSHi,CC);
      
      ConstantSDNode *Tmp1C = dyn_cast<ConstantSDNode>(Tmp1.Val);
      ConstantSDNode *Tmp2C = dyn_cast<ConstantSDNode>(Tmp2.Val);
      if ((Tmp1C && Tmp1C->isNullValue()) ||
          (Tmp2C && Tmp2C->isNullValue() &&
           (CCCode == ISD::SETLE || CCCode == ISD::SETGE ||
            CCCode == ISD::SETUGE || CCCode == ISD::SETULE)) ||
          (Tmp2C && Tmp2C->getAPIntValue() == 1 &&
           (CCCode == ISD::SETLT || CCCode == ISD::SETGT ||
            CCCode == ISD::SETUGT || CCCode == ISD::SETULT))) {
        // low part is known false, returns high part.
        // For LE / GE, if high part is known false, ignore the low part.
        // For LT / GT, if high part is known true, ignore the low part.
        Tmp1 = Tmp2;
        Tmp2 = SDOperand();
      } else {
        Result = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi,
                                   ISD::SETEQ, false, DagCombineInfo);
        if (!Result.Val)
          Result=DAG.getSetCC(TLI.getSetCCResultType(LHSHi), LHSHi, RHSHi,
                              ISD::SETEQ);
        Result = LegalizeOp(DAG.getNode(ISD::SELECT, Tmp1.getValueType(),
                                        Result, Tmp1, Tmp2));
        Tmp1 = Result;
        Tmp2 = SDOperand();
      }
    }
  }
  }
  LHS = Tmp1;
  RHS = Tmp2;
}

/// EmitStackConvert - Emit a store/load combination to the stack.  This stores
/// SrcOp to a stack slot of type SlotVT, truncating it if needed.  It then does
/// a load from the stack slot to DestVT, extending it if needed.
/// The resultant code need not be legal.
SDOperand SelectionDAGLegalize::EmitStackConvert(SDOperand SrcOp,
                                                 MVT::ValueType SlotVT, 
                                                 MVT::ValueType DestVT) {
  // Create the stack frame object.
  SDOperand FIPtr = DAG.CreateStackTemporary(SlotVT);

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(FIPtr);
  int SPFI = StackPtrFI->getIndex();

  unsigned SrcSize = MVT::getSizeInBits(SrcOp.getValueType());
  unsigned SlotSize = MVT::getSizeInBits(SlotVT);
  unsigned DestSize = MVT::getSizeInBits(DestVT);
  
  // Emit a store to the stack slot.  Use a truncstore if the input value is
  // later than DestVT.
  SDOperand Store;
  if (SrcSize > SlotSize)
    Store = DAG.getTruncStore(DAG.getEntryNode(), SrcOp, FIPtr,
                              PseudoSourceValue::getFixedStack(),
                              SPFI, SlotVT);
  else {
    assert(SrcSize == SlotSize && "Invalid store");
    Store = DAG.getStore(DAG.getEntryNode(), SrcOp, FIPtr,
                         PseudoSourceValue::getFixedStack(),
                         SPFI, SlotVT);
  }
  
  // Result is a load from the stack slot.
  if (SlotSize == DestSize)
    return DAG.getLoad(DestVT, Store, FIPtr, NULL, 0);
  
  assert(SlotSize < DestSize && "Unknown extension!");
  return DAG.getExtLoad(ISD::EXTLOAD, DestVT, Store, FIPtr, NULL, 0, SlotVT);
}

SDOperand SelectionDAGLegalize::ExpandSCALAR_TO_VECTOR(SDNode *Node) {
  // Create a vector sized/aligned stack slot, store the value to element #0,
  // then load the whole vector back out.
  SDOperand StackPtr = DAG.CreateStackTemporary(Node->getValueType(0));

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(StackPtr);
  int SPFI = StackPtrFI->getIndex();

  SDOperand Ch = DAG.getStore(DAG.getEntryNode(), Node->getOperand(0), StackPtr,
                              PseudoSourceValue::getFixedStack(), SPFI);
  return DAG.getLoad(Node->getValueType(0), Ch, StackPtr,
                     PseudoSourceValue::getFixedStack(), SPFI);
}


/// ExpandBUILD_VECTOR - Expand a BUILD_VECTOR node on targets that don't
/// support the operation, but do support the resultant vector type.
SDOperand SelectionDAGLegalize::ExpandBUILD_VECTOR(SDNode *Node) {
  
  // If the only non-undef value is the low element, turn this into a 
  // SCALAR_TO_VECTOR node.  If this is { X, X, X, X }, determine X.
  unsigned NumElems = Node->getNumOperands();
  bool isOnlyLowElement = true;
  SDOperand SplatValue = Node->getOperand(0);
  
  // FIXME: it would be far nicer to change this into map<SDOperand,uint64_t>
  // and use a bitmask instead of a list of elements.
  std::map<SDOperand, std::vector<unsigned> > Values;
  Values[SplatValue].push_back(0);
  bool isConstant = true;
  if (!isa<ConstantFPSDNode>(SplatValue) && !isa<ConstantSDNode>(SplatValue) &&
      SplatValue.getOpcode() != ISD::UNDEF)
    isConstant = false;
  
  for (unsigned i = 1; i < NumElems; ++i) {
    SDOperand V = Node->getOperand(i);
    Values[V].push_back(i);
    if (V.getOpcode() != ISD::UNDEF)
      isOnlyLowElement = false;
    if (SplatValue != V)
      SplatValue = SDOperand(0,0);

    // If this isn't a constant element or an undef, we can't use a constant
    // pool load.
    if (!isa<ConstantFPSDNode>(V) && !isa<ConstantSDNode>(V) &&
        V.getOpcode() != ISD::UNDEF)
      isConstant = false;
  }
  
  if (isOnlyLowElement) {
    // If the low element is an undef too, then this whole things is an undef.
    if (Node->getOperand(0).getOpcode() == ISD::UNDEF)
      return DAG.getNode(ISD::UNDEF, Node->getValueType(0));
    // Otherwise, turn this into a scalar_to_vector node.
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, Node->getValueType(0),
                       Node->getOperand(0));
  }
  
  // If all elements are constants, create a load from the constant pool.
  if (isConstant) {
    MVT::ValueType VT = Node->getValueType(0);
    std::vector<Constant*> CV;
    for (unsigned i = 0, e = NumElems; i != e; ++i) {
      if (ConstantFPSDNode *V = 
          dyn_cast<ConstantFPSDNode>(Node->getOperand(i))) {
        CV.push_back(ConstantFP::get(V->getValueAPF()));
      } else if (ConstantSDNode *V = 
                   dyn_cast<ConstantSDNode>(Node->getOperand(i))) {
        CV.push_back(ConstantInt::get(V->getAPIntValue()));
      } else {
        assert(Node->getOperand(i).getOpcode() == ISD::UNDEF);
        const Type *OpNTy = 
          MVT::getTypeForValueType(Node->getOperand(0).getValueType());
        CV.push_back(UndefValue::get(OpNTy));
      }
    }
    Constant *CP = ConstantVector::get(CV);
    SDOperand CPIdx = DAG.getConstantPool(CP, TLI.getPointerTy());
    return DAG.getLoad(VT, DAG.getEntryNode(), CPIdx,
                       PseudoSourceValue::getConstantPool(), 0);
  }
  
  if (SplatValue.Val) {   // Splat of one value?
    // Build the shuffle constant vector: <0, 0, 0, 0>
    MVT::ValueType MaskVT = 
      MVT::getIntVectorWithNumElements(NumElems);
    SDOperand Zero = DAG.getConstant(0, MVT::getVectorElementType(MaskVT));
    std::vector<SDOperand> ZeroVec(NumElems, Zero);
    SDOperand SplatMask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                      &ZeroVec[0], ZeroVec.size());

    // If the target supports VECTOR_SHUFFLE and this shuffle mask, use it.
    if (isShuffleLegal(Node->getValueType(0), SplatMask)) {
      // Get the splatted value into the low element of a vector register.
      SDOperand LowValVec = 
        DAG.getNode(ISD::SCALAR_TO_VECTOR, Node->getValueType(0), SplatValue);
    
      // Return shuffle(LowValVec, undef, <0,0,0,0>)
      return DAG.getNode(ISD::VECTOR_SHUFFLE, Node->getValueType(0), LowValVec,
                         DAG.getNode(ISD::UNDEF, Node->getValueType(0)),
                         SplatMask);
    }
  }
  
  // If there are only two unique elements, we may be able to turn this into a
  // vector shuffle.
  if (Values.size() == 2) {
    // Get the two values in deterministic order.
    SDOperand Val1 = Node->getOperand(1);
    SDOperand Val2;
    std::map<SDOperand, std::vector<unsigned> >::iterator MI = Values.begin();
    if (MI->first != Val1)
      Val2 = MI->first;
    else
      Val2 = (++MI)->first;
    
    // If Val1 is an undef, make sure end ends up as Val2, to ensure that our 
    // vector shuffle has the undef vector on the RHS.
    if (Val1.getOpcode() == ISD::UNDEF)
      std::swap(Val1, Val2);
    
    // Build the shuffle constant vector: e.g. <0, 4, 0, 4>
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
    MVT::ValueType MaskEltVT = MVT::getVectorElementType(MaskVT);
    std::vector<SDOperand> MaskVec(NumElems);

    // Set elements of the shuffle mask for Val1.
    std::vector<unsigned> &Val1Elts = Values[Val1];
    for (unsigned i = 0, e = Val1Elts.size(); i != e; ++i)
      MaskVec[Val1Elts[i]] = DAG.getConstant(0, MaskEltVT);

    // Set elements of the shuffle mask for Val2.
    std::vector<unsigned> &Val2Elts = Values[Val2];
    for (unsigned i = 0, e = Val2Elts.size(); i != e; ++i)
      if (Val2.getOpcode() != ISD::UNDEF)
        MaskVec[Val2Elts[i]] = DAG.getConstant(NumElems, MaskEltVT);
      else
        MaskVec[Val2Elts[i]] = DAG.getNode(ISD::UNDEF, MaskEltVT);
    
    SDOperand ShuffleMask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                        &MaskVec[0], MaskVec.size());

    // If the target supports SCALAR_TO_VECTOR and this shuffle mask, use it.
    if (TLI.isOperationLegal(ISD::SCALAR_TO_VECTOR, Node->getValueType(0)) &&
        isShuffleLegal(Node->getValueType(0), ShuffleMask)) {
      Val1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, Node->getValueType(0), Val1);
      Val2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, Node->getValueType(0), Val2);
      SDOperand Ops[] = { Val1, Val2, ShuffleMask };

      // Return shuffle(LoValVec, HiValVec, <0,1,0,1>)
      return DAG.getNode(ISD::VECTOR_SHUFFLE, Node->getValueType(0), Ops, 3);
    }
  }
  
  // Otherwise, we can't handle this case efficiently.  Allocate a sufficiently
  // aligned object on the stack, store each element into it, then load
  // the result as a vector.
  MVT::ValueType VT = Node->getValueType(0);
  // Create the stack frame object.
  SDOperand FIPtr = DAG.CreateStackTemporary(VT);
  
  // Emit a store of each element to the stack slot.
  SmallVector<SDOperand, 8> Stores;
  unsigned TypeByteSize = 
    MVT::getSizeInBits(Node->getOperand(0).getValueType())/8;
  // Store (in the right endianness) the elements to memory.
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    // Ignore undef elements.
    if (Node->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    
    unsigned Offset = TypeByteSize*i;
    
    SDOperand Idx = DAG.getConstant(Offset, FIPtr.getValueType());
    Idx = DAG.getNode(ISD::ADD, FIPtr.getValueType(), FIPtr, Idx);
    
    Stores.push_back(DAG.getStore(DAG.getEntryNode(), Node->getOperand(i), Idx, 
                                  NULL, 0));
  }
  
  SDOperand StoreChain;
  if (!Stores.empty())    // Not all undef elements?
    StoreChain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                             &Stores[0], Stores.size());
  else
    StoreChain = DAG.getEntryNode();
  
  // Result is a load from the stack slot.
  return DAG.getLoad(VT, StoreChain, FIPtr, NULL, 0);
}

void SelectionDAGLegalize::ExpandShiftParts(unsigned NodeOp,
                                            SDOperand Op, SDOperand Amt,
                                            SDOperand &Lo, SDOperand &Hi) {
  // Expand the subcomponents.
  SDOperand LHSL, LHSH;
  ExpandOp(Op, LHSL, LHSH);

  SDOperand Ops[] = { LHSL, LHSH, Amt };
  MVT::ValueType VT = LHSL.getValueType();
  Lo = DAG.getNode(NodeOp, DAG.getNodeValueTypes(VT, VT), 2, Ops, 3);
  Hi = Lo.getValue(1);
}


/// ExpandShift - Try to find a clever way to expand this shift operation out to
/// smaller elements.  If we can't find a way that is more efficient than a
/// libcall on this target, return false.  Otherwise, return true with the
/// low-parts expanded into Lo and Hi.
bool SelectionDAGLegalize::ExpandShift(unsigned Opc, SDOperand Op,SDOperand Amt,
                                       SDOperand &Lo, SDOperand &Hi) {
  assert((Opc == ISD::SHL || Opc == ISD::SRA || Opc == ISD::SRL) &&
         "This is not a shift!");

  MVT::ValueType NVT = TLI.getTypeToTransformTo(Op.getValueType());
  SDOperand ShAmt = LegalizeOp(Amt);
  MVT::ValueType ShTy = ShAmt.getValueType();
  unsigned ShBits = MVT::getSizeInBits(ShTy);
  unsigned VTBits = MVT::getSizeInBits(Op.getValueType());
  unsigned NVTBits = MVT::getSizeInBits(NVT);

  // Handle the case when Amt is an immediate.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Amt.Val)) {
    unsigned Cst = CN->getValue();
    // Expand the incoming operand to be shifted, so that we have its parts
    SDOperand InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Cst-NVTBits,ShTy));
      } else if (Cst == NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = InL;
      } else {
        Lo = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Cst, ShTy));
        Hi = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(NVTBits-Cst, ShTy)));
      }
      return true;
    case ISD::SRL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Cst-NVTBits,ShTy));
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getConstant(0, NVT);
      } else {
        Lo = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Cst, ShTy));
      }
      return true;
    case ISD::SRA:
      if (Cst > VTBits) {
        Hi = Lo = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRA, NVT, InH,
                           DAG.getConstant(Cst-NVTBits, ShTy));
        Hi = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getNode(ISD::SRA, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else {
        Lo = DAG.getNode(ISD::OR, NVT,
           DAG.getNode(ISD::SRL, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, NVT, InH, DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRA, NVT, InH, DAG.getConstant(Cst, ShTy));
      }
      return true;
    }
  }
  
  // Okay, the shift amount isn't constant.  However, if we can tell that it is
  // >= 32 or < 32, we can still simplify it, without knowing the actual value.
  APInt Mask = APInt::getHighBitsSet(ShBits, ShBits - Log2_32(NVTBits));
  APInt KnownZero, KnownOne;
  DAG.ComputeMaskedBits(Amt, Mask, KnownZero, KnownOne);
  
  // If we know that if any of the high bits of the shift amount are one, then
  // we can do this as a couple of simple shifts.
  if (KnownOne.intersects(Mask)) {
    // Mask out the high bit, which we know is set.
    Amt = DAG.getNode(ISD::AND, Amt.getValueType(), Amt,
                      DAG.getConstant(~Mask, Amt.getValueType()));
    
    // Expand the incoming operand to be shifted, so that we have its parts
    SDOperand InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      Lo = DAG.getConstant(0, NVT);              // Low part is zero.
      Hi = DAG.getNode(ISD::SHL, NVT, InL, Amt); // High part from Lo part.
      return true;
    case ISD::SRL:
      Hi = DAG.getConstant(0, NVT);              // Hi part is zero.
      Lo = DAG.getNode(ISD::SRL, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, NVT, InH,       // Sign extend high part.
                       DAG.getConstant(NVTBits-1, Amt.getValueType()));
      Lo = DAG.getNode(ISD::SRA, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    }
  }
  
  // If we know that the high bits of the shift amount are all zero, then we can
  // do this as a couple of simple shifts.
  if ((KnownZero & Mask) == Mask) {
    // Compute 32-amt.
    SDOperand Amt2 = DAG.getNode(ISD::SUB, Amt.getValueType(),
                                 DAG.getConstant(NVTBits, Amt.getValueType()),
                                 Amt);
    
    // Expand the incoming operand to be shifted, so that we have its parts
    SDOperand InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      Lo = DAG.getNode(ISD::SHL, NVT, InL, Amt);
      Hi = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SHL, NVT, InH, Amt),
                       DAG.getNode(ISD::SRL, NVT, InL, Amt2));
      return true;
    case ISD::SRL:
      Hi = DAG.getNode(ISD::SRL, NVT, InH, Amt);
      Lo = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SRL, NVT, InL, Amt),
                       DAG.getNode(ISD::SHL, NVT, InH, Amt2));
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, NVT, InH, Amt);
      Lo = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SRL, NVT, InL, Amt),
                       DAG.getNode(ISD::SHL, NVT, InH, Amt2));
      return true;
    }
  }
  
  return false;
}


// ExpandLibCall - Expand a node into a call to a libcall.  If the result value
// does not fit into a register, return the lo part and set the hi part to the
// by-reg argument.  If it does fit into a single register, return the result
// and leave the Hi part unset.
SDOperand SelectionDAGLegalize::ExpandLibCall(RTLIB::Libcall LC, SDNode *Node,
                                              bool isSigned, SDOperand &Hi) {
  assert(!IsLegalizingCall && "Cannot overlap legalization of calls!");
  // The input chain to this libcall is the entry node of the function. 
  // Legalizing the call will automatically add the previous call to the
  // dependence.
  SDOperand InChain = DAG.getEntryNode();
  
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    MVT::ValueType ArgVT = Node->getOperand(i).getValueType();
    const Type *ArgTy = MVT::getTypeForValueType(ArgVT);
    Entry.Node = Node->getOperand(i); Entry.Ty = ArgTy; 
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDOperand Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                           TLI.getPointerTy());

  // Splice the libcall in wherever FindInputOutputChains tells us to.
  const Type *RetTy = MVT::getTypeForValueType(Node->getValueType(0));
  std::pair<SDOperand,SDOperand> CallInfo =
    TLI.LowerCallTo(InChain, RetTy, isSigned, !isSigned, false, CallingConv::C,
                    false, Callee, Args, DAG);

  // Legalize the call sequence, starting with the chain.  This will advance
  // the LastCALLSEQ_END to the legalized version of the CALLSEQ_END node that
  // was added by LowerCallTo (guaranteeing proper serialization of calls).
  LegalizeOp(CallInfo.second);
  SDOperand Result;
  switch (getTypeAction(CallInfo.first.getValueType())) {
  default: assert(0 && "Unknown thing");
  case Legal:
    Result = CallInfo.first;
    break;
  case Expand:
    ExpandOp(CallInfo.first, Result, Hi);
    break;
  }
  return Result;
}


/// ExpandIntToFP - Expand a [US]INT_TO_FP operation.
///
SDOperand SelectionDAGLegalize::
ExpandIntToFP(bool isSigned, MVT::ValueType DestTy, SDOperand Source) {
  MVT::ValueType SourceVT = Source.getValueType();
  bool ExpandSource = getTypeAction(SourceVT) == Expand;

  // Special case for i32 source to take advantage of UINTTOFP_I32_F32, etc.
  if (!isSigned && SourceVT != MVT::i32) {
    // The integer value loaded will be incorrectly if the 'sign bit' of the
    // incoming integer is set.  To handle this, we dynamically test to see if
    // it is set, and, if so, add a fudge factor.
    SDOperand Hi;
    if (ExpandSource) {
      SDOperand Lo;
      ExpandOp(Source, Lo, Hi);
      Source = DAG.getNode(ISD::BUILD_PAIR, SourceVT, Lo, Hi);
    } else {
      // The comparison for the sign bit will use the entire operand.
      Hi = Source;
    }

    // If this is unsigned, and not supported, first perform the conversion to
    // signed, then adjust the result if the sign bit is set.
    SDOperand SignedConv = ExpandIntToFP(true, DestTy, Source);

    SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultType(Hi), Hi,
                                     DAG.getConstant(0, Hi.getValueType()),
                                     ISD::SETLT);
    SDOperand Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
    SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                      SignSet, Four, Zero);
    uint64_t FF = 0x5f800000ULL;
    if (TLI.isLittleEndian()) FF <<= 32;
    static Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

    SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
    CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
    SDOperand FudgeInReg;
    if (DestTy == MVT::f32)
      FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                               PseudoSourceValue::getConstantPool(), 0);
    else if (MVT::getSizeInBits(DestTy) > MVT::getSizeInBits(MVT::f32))
      // FIXME: Avoid the extend by construction the right constantpool?
      FudgeInReg = DAG.getExtLoad(ISD::EXTLOAD, DestTy, DAG.getEntryNode(),
                                  CPIdx,
                                  PseudoSourceValue::getConstantPool(), 0,
                                  MVT::f32);
    else 
      assert(0 && "Unexpected conversion");

    MVT::ValueType SCVT = SignedConv.getValueType();
    if (SCVT != DestTy) {
      // Destination type needs to be expanded as well. The FADD now we are
      // constructing will be expanded into a libcall.
      if (MVT::getSizeInBits(SCVT) != MVT::getSizeInBits(DestTy)) {
        assert(MVT::getSizeInBits(SCVT) * 2 == MVT::getSizeInBits(DestTy));
        SignedConv = DAG.getNode(ISD::BUILD_PAIR, DestTy,
                                 SignedConv, SignedConv.getValue(1));
      }
      SignedConv = DAG.getNode(ISD::BIT_CONVERT, DestTy, SignedConv);
    }
    return DAG.getNode(ISD::FADD, DestTy, SignedConv, FudgeInReg);
  }

  // Check to see if the target has a custom way to lower this.  If so, use it.
  switch (TLI.getOperationAction(ISD::SINT_TO_FP, SourceVT)) {
  default: assert(0 && "This action not implemented for this operation!");
  case TargetLowering::Legal:
  case TargetLowering::Expand:
    break;   // This case is handled below.
  case TargetLowering::Custom: {
    SDOperand NV = TLI.LowerOperation(DAG.getNode(ISD::SINT_TO_FP, DestTy,
                                                  Source), DAG);
    if (NV.Val)
      return LegalizeOp(NV);
    break;   // The target decided this was legal after all
  }
  }

  // Expand the source, then glue it back together for the call.  We must expand
  // the source in case it is shared (this pass of legalize must traverse it).
  if (ExpandSource) {
    SDOperand SrcLo, SrcHi;
    ExpandOp(Source, SrcLo, SrcHi);
    Source = DAG.getNode(ISD::BUILD_PAIR, SourceVT, SrcLo, SrcHi);
  }

  RTLIB::Libcall LC;
  if (SourceVT == MVT::i32) {
    if (DestTy == MVT::f32)
      LC = isSigned ? RTLIB::SINTTOFP_I32_F32 : RTLIB::UINTTOFP_I32_F32;
    else {
      assert(DestTy == MVT::f64 && "Unknown fp value type!");
      LC = isSigned ? RTLIB::SINTTOFP_I32_F64 : RTLIB::UINTTOFP_I32_F64;
    }
  } else if (SourceVT == MVT::i64) {
    if (DestTy == MVT::f32)
      LC = RTLIB::SINTTOFP_I64_F32;
    else if (DestTy == MVT::f64)
      LC = RTLIB::SINTTOFP_I64_F64;
    else if (DestTy == MVT::f80)
      LC = RTLIB::SINTTOFP_I64_F80;
    else {
      assert(DestTy == MVT::ppcf128 && "Unknown fp value type!");
      LC = RTLIB::SINTTOFP_I64_PPCF128;
    }
  } else if (SourceVT == MVT::i128) {
    if (DestTy == MVT::f32)
      LC = RTLIB::SINTTOFP_I128_F32;
    else if (DestTy == MVT::f64)
      LC = RTLIB::SINTTOFP_I128_F64;
    else if (DestTy == MVT::f80)
      LC = RTLIB::SINTTOFP_I128_F80;
    else {
      assert(DestTy == MVT::ppcf128 && "Unknown fp value type!");
      LC = RTLIB::SINTTOFP_I128_PPCF128;
    }
  } else {
    assert(0 && "Unknown int value type");
  }
  
  assert(TLI.getLibcallName(LC) && "Don't know how to expand this SINT_TO_FP!");
  Source = DAG.getNode(ISD::SINT_TO_FP, DestTy, Source);
  SDOperand HiPart;
  SDOperand Result = ExpandLibCall(LC, Source.Val, isSigned, HiPart);
  if (Result.getValueType() != DestTy && HiPart.Val)
    Result = DAG.getNode(ISD::BUILD_PAIR, DestTy, Result, HiPart);
  return Result;
}

/// ExpandLegalINT_TO_FP - This function is responsible for legalizing a
/// INT_TO_FP operation of the specified operand when the target requests that
/// we expand it.  At this point, we know that the result and operand types are
/// legal for the target.
SDOperand SelectionDAGLegalize::ExpandLegalINT_TO_FP(bool isSigned,
                                                     SDOperand Op0,
                                                     MVT::ValueType DestVT) {
  if (Op0.getValueType() == MVT::i32) {
    // simple 32-bit [signed|unsigned] integer to float/double expansion
    
    // Get the stack frame index of a 8 byte buffer.
    SDOperand StackSlot = DAG.CreateStackTemporary(MVT::f64);
    
    // word offset constant for Hi/Lo address computation
    SDOperand WordOff = DAG.getConstant(sizeof(int), TLI.getPointerTy());
    // set up Hi and Lo (into buffer) address based on endian
    SDOperand Hi = StackSlot;
    SDOperand Lo = DAG.getNode(ISD::ADD, TLI.getPointerTy(), StackSlot,WordOff);
    if (TLI.isLittleEndian())
      std::swap(Hi, Lo);
    
    // if signed map to unsigned space
    SDOperand Op0Mapped;
    if (isSigned) {
      // constant used to invert sign bit (signed to unsigned mapping)
      SDOperand SignBit = DAG.getConstant(0x80000000u, MVT::i32);
      Op0Mapped = DAG.getNode(ISD::XOR, MVT::i32, Op0, SignBit);
    } else {
      Op0Mapped = Op0;
    }
    // store the lo of the constructed double - based on integer input
    SDOperand Store1 = DAG.getStore(DAG.getEntryNode(),
                                    Op0Mapped, Lo, NULL, 0);
    // initial hi portion of constructed double
    SDOperand InitialHi = DAG.getConstant(0x43300000u, MVT::i32);
    // store the hi of the constructed double - biased exponent
    SDOperand Store2=DAG.getStore(Store1, InitialHi, Hi, NULL, 0);
    // load the constructed double
    SDOperand Load = DAG.getLoad(MVT::f64, Store2, StackSlot, NULL, 0);
    // FP constant to bias correct the final result
    SDOperand Bias = DAG.getConstantFP(isSigned ?
                                            BitsToDouble(0x4330000080000000ULL)
                                          : BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    // subtract the bias
    SDOperand Sub = DAG.getNode(ISD::FSUB, MVT::f64, Load, Bias);
    // final result
    SDOperand Result;
    // handle final rounding
    if (DestVT == MVT::f64) {
      // do nothing
      Result = Sub;
    } else if (MVT::getSizeInBits(DestVT) < MVT::getSizeInBits(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_ROUND, DestVT, Sub,
                           DAG.getIntPtrConstant(0));
    } else if (MVT::getSizeInBits(DestVT) > MVT::getSizeInBits(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_EXTEND, DestVT, Sub);
    }
    return Result;
  }
  assert(!isSigned && "Legalize cannot Expand SINT_TO_FP for i64 yet");
  SDOperand Tmp1 = DAG.getNode(ISD::SINT_TO_FP, DestVT, Op0);

  SDOperand SignSet = DAG.getSetCC(TLI.getSetCCResultType(Op0), Op0,
                                   DAG.getConstant(0, Op0.getValueType()),
                                   ISD::SETLT);
  SDOperand Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
  SDOperand CstOffset = DAG.getNode(ISD::SELECT, Zero.getValueType(),
                                    SignSet, Four, Zero);

  // If the sign bit of the integer is set, the large number will be treated
  // as a negative number.  To counteract this, the dynamic code adds an
  // offset depending on the data type.
  uint64_t FF;
  switch (Op0.getValueType()) {
  default: assert(0 && "Unsupported integer type!");
  case MVT::i8 : FF = 0x43800000ULL; break;  // 2^8  (as a float)
  case MVT::i16: FF = 0x47800000ULL; break;  // 2^16 (as a float)
  case MVT::i32: FF = 0x4F800000ULL; break;  // 2^32 (as a float)
  case MVT::i64: FF = 0x5F800000ULL; break;  // 2^64 (as a float)
  }
  if (TLI.isLittleEndian()) FF <<= 32;
  static Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

  SDOperand CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  CPIdx = DAG.getNode(ISD::ADD, TLI.getPointerTy(), CPIdx, CstOffset);
  SDOperand FudgeInReg;
  if (DestVT == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0);
  else {
    FudgeInReg =
      LegalizeOp(DAG.getExtLoad(ISD::EXTLOAD, DestVT,
                                DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                MVT::f32));
  }

  return DAG.getNode(ISD::FADD, DestVT, Tmp1, FudgeInReg);
}

/// PromoteLegalINT_TO_FP - This function is responsible for legalizing a
/// *INT_TO_FP operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal UINT_TO_FP or SINT_TO_FP
/// operation that takes a larger input.
SDOperand SelectionDAGLegalize::PromoteLegalINT_TO_FP(SDOperand LegalOp,
                                                      MVT::ValueType DestVT,
                                                      bool isSigned) {
  // First step, figure out the appropriate *INT_TO_FP operation to use.
  MVT::ValueType NewInTy = LegalOp.getValueType();

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewInTy = (MVT::ValueType)(NewInTy+1);
    assert(MVT::isInteger(NewInTy) && "Ran out of possibilities!");

    // If the target supports SINT_TO_FP of this type, use it.
    switch (TLI.getOperationAction(ISD::SINT_TO_FP, NewInTy)) {
      default: break;
      case TargetLowering::Legal:
        if (!TLI.isTypeLegal(NewInTy))
          break;  // Can't use this datatype.
        // FALL THROUGH.
      case TargetLowering::Custom:
        OpToUse = ISD::SINT_TO_FP;
        break;
    }
    if (OpToUse) break;
    if (isSigned) continue;

    // If the target supports UINT_TO_FP of this type, use it.
    switch (TLI.getOperationAction(ISD::UINT_TO_FP, NewInTy)) {
      default: break;
      case TargetLowering::Legal:
        if (!TLI.isTypeLegal(NewInTy))
          break;  // Can't use this datatype.
        // FALL THROUGH.
      case TargetLowering::Custom:
        OpToUse = ISD::UINT_TO_FP;
        break;
    }
    if (OpToUse) break;

    // Otherwise, try a larger type.
  }

  // Okay, we found the operation and type to use.  Zero extend our input to the
  // desired type then run the operation on it.
  return DAG.getNode(OpToUse, DestVT,
                     DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND,
                                 NewInTy, LegalOp));
}

/// PromoteLegalFP_TO_INT - This function is responsible for legalizing a
/// FP_TO_*INT operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal FP_TO_UINT or FP_TO_SINT
/// operation that returns a larger result.
SDOperand SelectionDAGLegalize::PromoteLegalFP_TO_INT(SDOperand LegalOp,
                                                      MVT::ValueType DestVT,
                                                      bool isSigned) {
  // First step, figure out the appropriate FP_TO*INT operation to use.
  MVT::ValueType NewOutTy = DestVT;

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewOutTy = (MVT::ValueType)(NewOutTy+1);
    assert(MVT::isInteger(NewOutTy) && "Ran out of possibilities!");

    // If the target supports FP_TO_SINT returning this type, use it.
    switch (TLI.getOperationAction(ISD::FP_TO_SINT, NewOutTy)) {
    default: break;
    case TargetLowering::Legal:
      if (!TLI.isTypeLegal(NewOutTy))
        break;  // Can't use this datatype.
      // FALL THROUGH.
    case TargetLowering::Custom:
      OpToUse = ISD::FP_TO_SINT;
      break;
    }
    if (OpToUse) break;

    // If the target supports FP_TO_UINT of this type, use it.
    switch (TLI.getOperationAction(ISD::FP_TO_UINT, NewOutTy)) {
    default: break;
    case TargetLowering::Legal:
      if (!TLI.isTypeLegal(NewOutTy))
        break;  // Can't use this datatype.
      // FALL THROUGH.
    case TargetLowering::Custom:
      OpToUse = ISD::FP_TO_UINT;
      break;
    }
    if (OpToUse) break;

    // Otherwise, try a larger type.
  }

  
  // Okay, we found the operation and type to use.
  SDOperand Operation = DAG.getNode(OpToUse, NewOutTy, LegalOp);
  
  // If the operation produces an invalid type, it must be custom lowered.  Use
  // the target lowering hooks to expand it.  Just keep the low part of the
  // expanded operation, we know that we're truncating anyway.
  if (getTypeAction(NewOutTy) == Expand) {
    Operation = SDOperand(TLI.ExpandOperationResult(Operation.Val, DAG), 0);
    assert(Operation.Val && "Didn't return anything");
  }
  
  // Truncate the result of the extended FP_TO_*INT operation to the desired
  // size.
  return DAG.getNode(ISD::TRUNCATE, DestVT, Operation);
}

/// ExpandBSWAP - Open code the operations for BSWAP of the specified operation.
///
SDOperand SelectionDAGLegalize::ExpandBSWAP(SDOperand Op) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType SHVT = TLI.getShiftAmountTy();
  SDOperand Tmp1, Tmp2, Tmp3, Tmp4, Tmp5, Tmp6, Tmp7, Tmp8;
  switch (VT) {
  default: assert(0 && "Unhandled Expand type in BSWAP!"); abort();
  case MVT::i16:
    Tmp2 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(8, SHVT));
    return DAG.getNode(ISD::OR, VT, Tmp1, Tmp2);
  case MVT::i32:
    Tmp4 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(8, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::AND, VT, Tmp3, DAG.getConstant(0xFF0000, VT));
    Tmp2 = DAG.getNode(ISD::AND, VT, Tmp2, DAG.getConstant(0xFF00, VT));
    Tmp4 = DAG.getNode(ISD::OR, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, VT, Tmp2, Tmp1);
    return DAG.getNode(ISD::OR, VT, Tmp4, Tmp2);
  case MVT::i64:
    Tmp8 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(40, SHVT));
    Tmp6 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(24, SHVT));
    Tmp5 = DAG.getNode(ISD::SHL, VT, Op, DAG.getConstant(8, SHVT));
    Tmp4 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(8, SHVT));
    Tmp3 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(24, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(40, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::AND, VT, Tmp7, DAG.getConstant(255ULL<<48, VT));
    Tmp6 = DAG.getNode(ISD::AND, VT, Tmp6, DAG.getConstant(255ULL<<40, VT));
    Tmp5 = DAG.getNode(ISD::AND, VT, Tmp5, DAG.getConstant(255ULL<<32, VT));
    Tmp4 = DAG.getNode(ISD::AND, VT, Tmp4, DAG.getConstant(255ULL<<24, VT));
    Tmp3 = DAG.getNode(ISD::AND, VT, Tmp3, DAG.getConstant(255ULL<<16, VT));
    Tmp2 = DAG.getNode(ISD::AND, VT, Tmp2, DAG.getConstant(255ULL<<8 , VT));
    Tmp8 = DAG.getNode(ISD::OR, VT, Tmp8, Tmp7);
    Tmp6 = DAG.getNode(ISD::OR, VT, Tmp6, Tmp5);
    Tmp4 = DAG.getNode(ISD::OR, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, VT, Tmp2, Tmp1);
    Tmp8 = DAG.getNode(ISD::OR, VT, Tmp8, Tmp6);
    Tmp4 = DAG.getNode(ISD::OR, VT, Tmp4, Tmp2);
    return DAG.getNode(ISD::OR, VT, Tmp8, Tmp4);
  }
}

/// ExpandBitCount - Expand the specified bitcount instruction into operations.
///
SDOperand SelectionDAGLegalize::ExpandBitCount(unsigned Opc, SDOperand Op) {
  switch (Opc) {
  default: assert(0 && "Cannot expand this yet!");
  case ISD::CTPOP: {
    static const uint64_t mask[6] = {
      0x5555555555555555ULL, 0x3333333333333333ULL,
      0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
      0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
    };
    MVT::ValueType VT = Op.getValueType();
    MVT::ValueType ShVT = TLI.getShiftAmountTy();
    unsigned len = MVT::getSizeInBits(VT);
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      //x = (x & mask[i][len/8]) + (x >> (1 << i) & mask[i][len/8])
      SDOperand Tmp2 = DAG.getConstant(mask[i], VT);
      SDOperand Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::ADD, VT, DAG.getNode(ISD::AND, VT, Op, Tmp2),
                       DAG.getNode(ISD::AND, VT,
                                   DAG.getNode(ISD::SRL, VT, Op, Tmp3),Tmp2));
    }
    return Op;
  }
  case ISD::CTLZ: {
    // for now, we do this:
    // x = x | (x >> 1);
    // x = x | (x >> 2);
    // ...
    // x = x | (x >>16);
    // x = x | (x >>32); // for 64-bit input
    // return popcount(~x);
    //
    // but see also: http://www.hackersdelight.org/HDcode/nlz.cc
    MVT::ValueType VT = Op.getValueType();
    MVT::ValueType ShVT = TLI.getShiftAmountTy();
    unsigned len = MVT::getSizeInBits(VT);
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      SDOperand Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::OR, VT, Op, DAG.getNode(ISD::SRL, VT, Op, Tmp3));
    }
    Op = DAG.getNode(ISD::XOR, VT, Op, DAG.getConstant(~0ULL, VT));
    return DAG.getNode(ISD::CTPOP, VT, Op);
  }
  case ISD::CTTZ: {
    // for now, we use: { return popcount(~x & (x - 1)); }
    // unless the target has ctlz but not ctpop, in which case we use:
    // { return 32 - nlz(~x & (x-1)); }
    // see also http://www.hackersdelight.org/HDcode/ntz.cc
    MVT::ValueType VT = Op.getValueType();
    SDOperand Tmp2 = DAG.getConstant(~0ULL, VT);
    SDOperand Tmp3 = DAG.getNode(ISD::AND, VT,
                       DAG.getNode(ISD::XOR, VT, Op, Tmp2),
                       DAG.getNode(ISD::SUB, VT, Op, DAG.getConstant(1, VT)));
    // If ISD::CTLZ is legal and CTPOP isn't, then do that instead.
    if (!TLI.isOperationLegal(ISD::CTPOP, VT) &&
        TLI.isOperationLegal(ISD::CTLZ, VT))
      return DAG.getNode(ISD::SUB, VT,
                         DAG.getConstant(MVT::getSizeInBits(VT), VT),
                         DAG.getNode(ISD::CTLZ, VT, Tmp3));
    return DAG.getNode(ISD::CTPOP, VT, Tmp3);
  }
  }
}

/// ExpandOp - Expand the specified SDOperand into its two component pieces
/// Lo&Hi.  Note that the Op MUST be an expanded type.  As a result of this, the
/// LegalizeNodes map is filled in for any results that are not expanded, the
/// ExpandedNodes map is filled in for any results that are expanded, and the
/// Lo/Hi values are returned.
void SelectionDAGLegalize::ExpandOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi){
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDNode *Node = Op.Val;
  assert(getTypeAction(VT) == Expand && "Not an expanded type!");
  assert(((MVT::isInteger(NVT) && NVT < VT) || MVT::isFloatingPoint(VT) ||
         MVT::isVector(VT)) &&
         "Cannot expand to FP value or to larger int value!");

  // See if we already expanded it.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> >::iterator I
    = ExpandedNodes.find(Op);
  if (I != ExpandedNodes.end()) {
    Lo = I->second.first;
    Hi = I->second.second;
    return;
  }

  switch (Node->getOpcode()) {
  case ISD::CopyFromReg:
    assert(0 && "CopyFromReg must be legal!");
  case ISD::FP_ROUND_INREG:
    if (VT == MVT::ppcf128 && 
        TLI.getOperationAction(ISD::FP_ROUND_INREG, VT) == 
            TargetLowering::Custom) {
      SDOperand SrcLo, SrcHi, Src;
      ExpandOp(Op.getOperand(0), SrcLo, SrcHi);
      Src = DAG.getNode(ISD::BUILD_PAIR, VT, SrcLo, SrcHi);
      SDOperand Result = TLI.LowerOperation(
        DAG.getNode(ISD::FP_ROUND_INREG, VT, Src, Op.getOperand(1)), DAG);
      assert(Result.Val->getOpcode() == ISD::BUILD_PAIR);
      Lo = Result.Val->getOperand(0);
      Hi = Result.Val->getOperand(1);
      break;
    }
    // fall through
  default:
#ifndef NDEBUG
    cerr << "NODE: "; Node->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand this operator!");
    abort();
  case ISD::EXTRACT_ELEMENT:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    if (cast<ConstantSDNode>(Node->getOperand(1))->getValue())
      return ExpandOp(Hi, Lo, Hi);
    return ExpandOp(Lo, Lo, Hi);
  case ISD::EXTRACT_VECTOR_ELT:
    assert(VT==MVT::i64 && "Do not know how to expand this operator!");
    // ExpandEXTRACT_VECTOR_ELT tolerates invalid result types.
    Lo  = ExpandEXTRACT_VECTOR_ELT(Op);
    return ExpandOp(Lo, Lo, Hi);
  case ISD::UNDEF:
    NVT = TLI.getTypeToExpandTo(VT);
    Lo = DAG.getNode(ISD::UNDEF, NVT);
    Hi = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::Constant: {
    unsigned NVTBits = MVT::getSizeInBits(NVT);
    const APInt &Cst = cast<ConstantSDNode>(Node)->getAPIntValue();
    Lo = DAG.getConstant(APInt(Cst).trunc(NVTBits), NVT);
    Hi = DAG.getConstant(Cst.lshr(NVTBits).trunc(NVTBits), NVT);
    break;
  }
  case ISD::ConstantFP: {
    ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Node);
    if (CFP->getValueType(0) == MVT::ppcf128) {
      APInt api = CFP->getValueAPF().convertToAPInt();
      Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &api.getRawData()[1])),
                             MVT::f64);
      Hi = DAG.getConstantFP(APFloat(APInt(64, 1, &api.getRawData()[0])), 
                             MVT::f64);
      break;
    }
    Lo = ExpandConstantFP(CFP, false, DAG, TLI);
    if (getTypeAction(Lo.getValueType()) == Expand)
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  case ISD::BUILD_PAIR:
    // Return the operands.
    Lo = Node->getOperand(0);
    Hi = Node->getOperand(1);
    break;
      
  case ISD::MERGE_VALUES:
    if (Node->getNumValues() == 1) {
      ExpandOp(Op.getOperand(0), Lo, Hi);
      break;
    }
    // FIXME: For now only expand i64,chain = MERGE_VALUES (x, y)
    assert(Op.ResNo == 0 && Node->getNumValues() == 2 &&
           Op.getValue(1).getValueType() == MVT::Other &&
           "unhandled MERGE_VALUES");
    ExpandOp(Op.getOperand(0), Lo, Hi);
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Op.getOperand(1)));
    break;
    
  case ISD::SIGN_EXTEND_INREG:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    // sext_inreg the low part if needed.
    Lo = DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Lo, Node->getOperand(1));
    
    // The high part gets the sign extension from the lo-part.  This handles
    // things like sextinreg V:i64 from i8.
    Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                     DAG.getConstant(MVT::getSizeInBits(NVT)-1,
                                     TLI.getShiftAmountTy()));
    break;

  case ISD::BSWAP: {
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDOperand TempLo = DAG.getNode(ISD::BSWAP, NVT, Hi);
    Hi = DAG.getNode(ISD::BSWAP, NVT, Lo);
    Lo = TempLo;
    break;
  }
    
  case ISD::CTPOP:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    Lo = DAG.getNode(ISD::ADD, NVT,          // ctpop(HL) -> ctpop(H)+ctpop(L)
                     DAG.getNode(ISD::CTPOP, NVT, Lo),
                     DAG.getNode(ISD::CTPOP, NVT, Hi));
    Hi = DAG.getConstant(0, NVT);
    break;

  case ISD::CTLZ: {
    // ctlz (HL) -> ctlz(H) != 32 ? ctlz(H) : (ctlz(L)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDOperand BitsC = DAG.getConstant(MVT::getSizeInBits(NVT), NVT);
    SDOperand HLZ = DAG.getNode(ISD::CTLZ, NVT, Hi);
    SDOperand TopNotZero = DAG.getSetCC(TLI.getSetCCResultType(HLZ), HLZ, BitsC,
                                        ISD::SETNE);
    SDOperand LowPart = DAG.getNode(ISD::CTLZ, NVT, Lo);
    LowPart = DAG.getNode(ISD::ADD, NVT, LowPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, NVT, TopNotZero, HLZ, LowPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::CTTZ: {
    // cttz (HL) -> cttz(L) != 32 ? cttz(L) : (cttz(H)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDOperand BitsC = DAG.getConstant(MVT::getSizeInBits(NVT), NVT);
    SDOperand LTZ = DAG.getNode(ISD::CTTZ, NVT, Lo);
    SDOperand BotNotZero = DAG.getSetCC(TLI.getSetCCResultType(LTZ), LTZ, BitsC,
                                        ISD::SETNE);
    SDOperand HiPart = DAG.getNode(ISD::CTTZ, NVT, Hi);
    HiPart = DAG.getNode(ISD::ADD, NVT, HiPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, NVT, BotNotZero, LTZ, HiPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::VAARG: {
    SDOperand Ch = Node->getOperand(0);   // Legalize the chain.
    SDOperand Ptr = Node->getOperand(1);  // Legalize the pointer.
    Lo = DAG.getVAArg(NVT, Ch, Ptr, Node->getOperand(2));
    Hi = DAG.getVAArg(NVT, Lo.getValue(1), Ptr, Node->getOperand(2));

    // Remember that we legalized the chain.
    Hi = LegalizeOp(Hi);
    AddLegalizedOperand(Op.getValue(1), Hi.getValue(1));
    if (TLI.isBigEndian())
      std::swap(Lo, Hi);
    break;
  }
    
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    SDOperand Ch  = LD->getChain();    // Legalize the chain.
    SDOperand Ptr = LD->getBasePtr();  // Legalize the pointer.
    ISD::LoadExtType ExtType = LD->getExtensionType();
    int SVOffset = LD->getSrcValueOffset();
    unsigned Alignment = LD->getAlignment();
    bool isVolatile = LD->isVolatile();

    if (ExtType == ISD::NON_EXTLOAD) {
      Lo = DAG.getLoad(NVT, Ch, Ptr, LD->getSrcValue(), SVOffset,
                       isVolatile, Alignment);
      if (VT == MVT::f32 || VT == MVT::f64) {
        // f32->i32 or f64->i64 one to one expansion.
        // Remember that we legalized the chain.
        AddLegalizedOperand(SDOperand(Node, 1), LegalizeOp(Lo.getValue(1)));
        // Recursively expand the new load.
        if (getTypeAction(NVT) == Expand)
          ExpandOp(Lo, Lo, Hi);
        break;
      }

      // Increment the pointer to the other half.
      unsigned IncrementSize = MVT::getSizeInBits(Lo.getValueType())/8;
      Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                        DAG.getIntPtrConstant(IncrementSize));
      SVOffset += IncrementSize;
      Alignment = MinAlign(Alignment, IncrementSize);
      Hi = DAG.getLoad(NVT, Ch, Ptr, LD->getSrcValue(), SVOffset,
                       isVolatile, Alignment);

      // Build a factor node to remember that this load is independent of the
      // other one.
      SDOperand TF = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                                 Hi.getValue(1));

      // Remember that we legalized the chain.
      AddLegalizedOperand(Op.getValue(1), LegalizeOp(TF));
      if (TLI.isBigEndian())
        std::swap(Lo, Hi);
    } else {
      MVT::ValueType EVT = LD->getMemoryVT();

      if ((VT == MVT::f64 && EVT == MVT::f32) ||
          (VT == MVT::ppcf128 && (EVT==MVT::f64 || EVT==MVT::f32))) {
        // f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
        SDOperand Load = DAG.getLoad(EVT, Ch, Ptr, LD->getSrcValue(),
                                     SVOffset, isVolatile, Alignment);
        // Remember that we legalized the chain.
        AddLegalizedOperand(SDOperand(Node, 1), LegalizeOp(Load.getValue(1)));
        ExpandOp(DAG.getNode(ISD::FP_EXTEND, VT, Load), Lo, Hi);
        break;
      }
    
      if (EVT == NVT)
        Lo = DAG.getLoad(NVT, Ch, Ptr, LD->getSrcValue(),
                         SVOffset, isVolatile, Alignment);
      else
        Lo = DAG.getExtLoad(ExtType, NVT, Ch, Ptr, LD->getSrcValue(),
                            SVOffset, EVT, isVolatile,
                            Alignment);
    
      // Remember that we legalized the chain.
      AddLegalizedOperand(SDOperand(Node, 1), LegalizeOp(Lo.getValue(1)));

      if (ExtType == ISD::SEXTLOAD) {
        // The high part is obtained by SRA'ing all but one of the bits of the
        // lo part.
        unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
        Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                         DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
      } else if (ExtType == ISD::ZEXTLOAD) {
        // The high part is just a zero.
        Hi = DAG.getConstant(0, NVT);
      } else /* if (ExtType == ISD::EXTLOAD) */ {
        // The high part is undefined.
        Hi = DAG.getNode(ISD::UNDEF, NVT);
      }
    }
    break;
  }
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {   // Simple logical operators -> two trivial pieces.
    SDOperand LL, LH, RL, RH;
    ExpandOp(Node->getOperand(0), LL, LH);
    ExpandOp(Node->getOperand(1), RL, RH);
    Lo = DAG.getNode(Node->getOpcode(), NVT, LL, RL);
    Hi = DAG.getNode(Node->getOpcode(), NVT, LH, RH);
    break;
  }
  case ISD::SELECT: {
    SDOperand LL, LH, RL, RH;
    ExpandOp(Node->getOperand(1), LL, LH);
    ExpandOp(Node->getOperand(2), RL, RH);
    if (getTypeAction(NVT) == Expand)
      NVT = TLI.getTypeToExpandTo(NVT);
    Lo = DAG.getNode(ISD::SELECT, NVT, Node->getOperand(0), LL, RL);
    if (VT != MVT::f32)
      Hi = DAG.getNode(ISD::SELECT, NVT, Node->getOperand(0), LH, RH);
    break;
  }
  case ISD::SELECT_CC: {
    SDOperand TL, TH, FL, FH;
    ExpandOp(Node->getOperand(2), TL, TH);
    ExpandOp(Node->getOperand(3), FL, FH);
    if (getTypeAction(NVT) == Expand)
      NVT = TLI.getTypeToExpandTo(NVT);
    Lo = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                     Node->getOperand(1), TL, FL, Node->getOperand(4));
    if (VT != MVT::f32)
      Hi = DAG.getNode(ISD::SELECT_CC, NVT, Node->getOperand(0),
                       Node->getOperand(1), TH, FH, Node->getOperand(4));
    break;
  }
  case ISD::ANY_EXTEND:
    // The low part is any extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ANY_EXTEND, NVT, Node->getOperand(0));
    // The high part is undefined.
    Hi = DAG.getNode(ISD::UNDEF, NVT);
    break;
  case ISD::SIGN_EXTEND: {
    // The low part is just a sign extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::SIGN_EXTEND, NVT, Node->getOperand(0));

    // The high part is obtained by SRA'ing all but one of the bits of the lo
    // part.
    unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
    Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                     DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
    break;
  }
  case ISD::ZERO_EXTEND:
    // The low part is just a zero extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, Node->getOperand(0));

    // The high part is just a zero.
    Hi = DAG.getConstant(0, NVT);
    break;
    
  case ISD::TRUNCATE: {
    // The input value must be larger than this value.  Expand *it*.
    SDOperand NewLo;
    ExpandOp(Node->getOperand(0), NewLo, Hi);
    
    // The low part is now either the right size, or it is closer.  If not the
    // right size, make an illegal truncate so we recursively expand it.
    if (NewLo.getValueType() != Node->getValueType(0))
      NewLo = DAG.getNode(ISD::TRUNCATE, Node->getValueType(0), NewLo);
    ExpandOp(NewLo, Lo, Hi);
    break;
  }
    
  case ISD::BIT_CONVERT: {
    SDOperand Tmp;
    if (TLI.getOperationAction(ISD::BIT_CONVERT, VT) == TargetLowering::Custom){
      // If the target wants to, allow it to lower this itself.
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "cannot expand FP!");
      case Legal:   Tmp = LegalizeOp(Node->getOperand(0)); break;
      case Promote: Tmp = PromoteOp (Node->getOperand(0)); break;
      }
      Tmp = TLI.LowerOperation(DAG.getNode(ISD::BIT_CONVERT, VT, Tmp), DAG);
    }

    // f32 / f64 must be expanded to i32 / i64.
    if (VT == MVT::f32 || VT == MVT::f64) {
      Lo = DAG.getNode(ISD::BIT_CONVERT, NVT, Node->getOperand(0));
      if (getTypeAction(NVT) == Expand)
        ExpandOp(Lo, Lo, Hi);
      break;
    }

    // If source operand will be expanded to the same type as VT, i.e.
    // i64 <- f64, i32 <- f32, expand the source operand instead.
    MVT::ValueType VT0 = Node->getOperand(0).getValueType();
    if (getTypeAction(VT0) == Expand && TLI.getTypeToTransformTo(VT0) == VT) {
      ExpandOp(Node->getOperand(0), Lo, Hi);
      break;
    }

    // Turn this into a load/store pair by default.
    if (Tmp.Val == 0)
      Tmp = EmitStackConvert(Node->getOperand(0), VT, VT);
    
    ExpandOp(Tmp, Lo, Hi);
    break;
  }

  case ISD::READCYCLECOUNTER: {
    assert(TLI.getOperationAction(ISD::READCYCLECOUNTER, VT) == 
                 TargetLowering::Custom &&
           "Must custom expand ReadCycleCounter");
    SDOperand Tmp = TLI.LowerOperation(Op, DAG);
    assert(Tmp.Val && "Node must be custom expanded!");
    ExpandOp(Tmp.getValue(0), Lo, Hi);
    AddLegalizedOperand(SDOperand(Node, 1), // Remember we legalized the chain.
                        LegalizeOp(Tmp.getValue(1)));
    break;
  }

  case ISD::ATOMIC_LCS: {
    SDOperand Tmp = TLI.LowerOperation(Op, DAG);
    assert(Tmp.Val && "Node must be custom expanded!");
    ExpandOp(Tmp.getValue(0), Lo, Hi);
    AddLegalizedOperand(SDOperand(Node, 1), // Remember we legalized the chain.
                        LegalizeOp(Tmp.getValue(1)));
    break;
  }



    // These operators cannot be expanded directly, emit them as calls to
    // library functions.
  case ISD::FP_TO_SINT: {
    if (TLI.getOperationAction(ISD::FP_TO_SINT, VT) == TargetLowering::Custom) {
      SDOperand Op;
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "cannot expand FP!");
      case Legal:   Op = LegalizeOp(Node->getOperand(0)); break;
      case Promote: Op = PromoteOp (Node->getOperand(0)); break;
      }

      Op = TLI.LowerOperation(DAG.getNode(ISD::FP_TO_SINT, VT, Op), DAG);

      // Now that the custom expander is done, expand the result, which is still
      // VT.
      if (Op.Val) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
    if (VT == MVT::i64) {
      if (Node->getOperand(0).getValueType() == MVT::f32)
        LC = RTLIB::FPTOSINT_F32_I64;
      else if (Node->getOperand(0).getValueType() == MVT::f64)
        LC = RTLIB::FPTOSINT_F64_I64;
      else if (Node->getOperand(0).getValueType() == MVT::f80)
        LC = RTLIB::FPTOSINT_F80_I64;
      else if (Node->getOperand(0).getValueType() == MVT::ppcf128)
        LC = RTLIB::FPTOSINT_PPCF128_I64;
      Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    } else if (VT == MVT::i128) {
      if (Node->getOperand(0).getValueType() == MVT::f32)
        LC = RTLIB::FPTOSINT_F32_I128;
      else if (Node->getOperand(0).getValueType() == MVT::f64)
        LC = RTLIB::FPTOSINT_F64_I128;
      else if (Node->getOperand(0).getValueType() == MVT::f80)
        LC = RTLIB::FPTOSINT_F80_I128;
      else if (Node->getOperand(0).getValueType() == MVT::ppcf128)
        LC = RTLIB::FPTOSINT_PPCF128_I128;
      Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    } else {
      assert(0 && "Unexpected uint-to-fp conversion!");
    }
    break;
  }

  case ISD::FP_TO_UINT: {
    if (TLI.getOperationAction(ISD::FP_TO_UINT, VT) == TargetLowering::Custom) {
      SDOperand Op;
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
        case Expand: assert(0 && "cannot expand FP!");
        case Legal:   Op = LegalizeOp(Node->getOperand(0)); break;
        case Promote: Op = PromoteOp (Node->getOperand(0)); break;
      }
        
      Op = TLI.LowerOperation(DAG.getNode(ISD::FP_TO_UINT, VT, Op), DAG);

      // Now that the custom expander is done, expand the result.
      if (Op.Val) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
    if (VT == MVT::i64) {
      if (Node->getOperand(0).getValueType() == MVT::f32)
        LC = RTLIB::FPTOUINT_F32_I64;
      else if (Node->getOperand(0).getValueType() == MVT::f64)
        LC = RTLIB::FPTOUINT_F64_I64;
      else if (Node->getOperand(0).getValueType() == MVT::f80)
        LC = RTLIB::FPTOUINT_F80_I64;
      else if (Node->getOperand(0).getValueType() == MVT::ppcf128)
        LC = RTLIB::FPTOUINT_PPCF128_I64;
      Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    } else if (VT == MVT::i128) {
      if (Node->getOperand(0).getValueType() == MVT::f32)
        LC = RTLIB::FPTOUINT_F32_I128;
      else if (Node->getOperand(0).getValueType() == MVT::f64)
        LC = RTLIB::FPTOUINT_F64_I128;
      else if (Node->getOperand(0).getValueType() == MVT::f80)
        LC = RTLIB::FPTOUINT_F80_I128;
      else if (Node->getOperand(0).getValueType() == MVT::ppcf128)
        LC = RTLIB::FPTOUINT_PPCF128_I128;
      Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    } else {
      assert(0 && "Unexpected uint-to-fp conversion!");
    }
    break;
  }

  case ISD::SHL: {
    // If the target wants custom lowering, do so.
    SDOperand ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SHL, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SHL, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If ADDC/ADDE are supported and if the shift amount is a constant 1, emit 
    // this X << 1 as X+X.
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(ShiftAmt)) {
      if (ShAmt->getAPIntValue() == 1 && TLI.isOperationLegal(ISD::ADDC, NVT) && 
          TLI.isOperationLegal(ISD::ADDE, NVT)) {
        SDOperand LoOps[2], HiOps[3];
        ExpandOp(Node->getOperand(0), LoOps[0], HiOps[0]);
        SDVTList VTList = DAG.getVTList(LoOps[0].getValueType(), MVT::Flag);
        LoOps[1] = LoOps[0];
        Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);

        HiOps[1] = HiOps[0];
        HiOps[2] = Lo.getValue(1);
        Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SHL, Node->getOperand(0), ShiftAmt, Lo, Hi))
      break;

    // If this target supports SHL_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SHL_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SHL_PARTS, Node->getOperand(0), ShiftAmt, Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall(RTLIB::SHL_I64, Node, false/*left shift=unsigned*/, Hi);
    break;
  }

  case ISD::SRA: {
    // If the target wants custom lowering, do so.
    SDOperand ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SRA, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SRA, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRA, Node->getOperand(0), ShiftAmt, Lo, Hi))
      break;

    // If this target supports SRA_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SRA_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SRA_PARTS, Node->getOperand(0), ShiftAmt, Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall(RTLIB::SRA_I64, Node, true/*ashr is signed*/, Hi);
    break;
  }

  case ISD::SRL: {
    // If the target wants custom lowering, do so.
    SDOperand ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SRL, VT) == TargetLowering::Custom) {
      SDOperand Op = DAG.getNode(ISD::SRL, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRL, Node->getOperand(0), ShiftAmt, Lo, Hi))
      break;

    // If this target supports SRL_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SRL_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SRL_PARTS, Node->getOperand(0), ShiftAmt, Lo, Hi);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall(RTLIB::SRL_I64, Node, false/*lshr is unsigned*/, Hi);
    break;
  }

  case ISD::ADD:
  case ISD::SUB: {
    // If the target wants to custom expand this, let them.
    if (TLI.getOperationAction(Node->getOpcode(), VT) ==
            TargetLowering::Custom) {
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.Val) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // Expand the subcomponents.
    SDOperand LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
    SDOperand LoOps[2], HiOps[3];
    LoOps[0] = LHSL;
    LoOps[1] = RHSL;
    HiOps[0] = LHSH;
    HiOps[1] = RHSH;
    if (Node->getOpcode() == ISD::ADD) {
      Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
    } else {
      Lo = DAG.getNode(ISD::SUBC, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::SUBE, VTList, HiOps, 3);
    }
    break;
  }
    
  case ISD::ADDC:
  case ISD::SUBC: {
    // Expand the subcomponents.
    SDOperand LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
    SDOperand LoOps[2] = { LHSL, RHSL };
    SDOperand HiOps[3] = { LHSH, RHSH };
    
    if (Node->getOpcode() == ISD::ADDC) {
      Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
    } else {
      Lo = DAG.getNode(ISD::SUBC, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::SUBE, VTList, HiOps, 3);
    }
    // Remember that we legalized the flag.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Hi.getValue(1)));
    break;
  }
  case ISD::ADDE:
  case ISD::SUBE: {
    // Expand the subcomponents.
    SDOperand LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
    SDOperand LoOps[3] = { LHSL, RHSL, Node->getOperand(2) };
    SDOperand HiOps[3] = { LHSH, RHSH };
    
    Lo = DAG.getNode(Node->getOpcode(), VTList, LoOps, 3);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(Node->getOpcode(), VTList, HiOps, 3);
    
    // Remember that we legalized the flag.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Hi.getValue(1)));
    break;
  }
  case ISD::MUL: {
    // If the target wants to custom expand this, let them.
    if (TLI.getOperationAction(ISD::MUL, VT) == TargetLowering::Custom) {
      SDOperand New = TLI.LowerOperation(Op, DAG);
      if (New.Val) {
        ExpandOp(New, Lo, Hi);
        break;
      }
    }
    
    bool HasMULHS = TLI.isOperationLegal(ISD::MULHS, NVT);
    bool HasMULHU = TLI.isOperationLegal(ISD::MULHU, NVT);
    bool HasSMUL_LOHI = TLI.isOperationLegal(ISD::SMUL_LOHI, NVT);
    bool HasUMUL_LOHI = TLI.isOperationLegal(ISD::UMUL_LOHI, NVT);
    if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
      SDOperand LL, LH, RL, RH;
      ExpandOp(Node->getOperand(0), LL, LH);
      ExpandOp(Node->getOperand(1), RL, RH);
      unsigned OuterBitSize = Op.getValueSizeInBits();
      unsigned InnerBitSize = RH.getValueSizeInBits();
      unsigned LHSSB = DAG.ComputeNumSignBits(Op.getOperand(0));
      unsigned RHSSB = DAG.ComputeNumSignBits(Op.getOperand(1));
      APInt HighMask = APInt::getHighBitsSet(OuterBitSize, InnerBitSize);
      if (DAG.MaskedValueIsZero(Node->getOperand(0), HighMask) &&
          DAG.MaskedValueIsZero(Node->getOperand(1), HighMask)) {
        // The inputs are both zero-extended.
        if (HasUMUL_LOHI) {
          // We can emit a umul_lohi.
          Lo = DAG.getNode(ISD::UMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
          Hi = SDOperand(Lo.Val, 1);
          break;
        }
        if (HasMULHU) {
          // We can emit a mulhu+mul.
          Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
          Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
          break;
        }
      }
      if (LHSSB > InnerBitSize && RHSSB > InnerBitSize) {
        // The input values are both sign-extended.
        if (HasSMUL_LOHI) {
          // We can emit a smul_lohi.
          Lo = DAG.getNode(ISD::SMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
          Hi = SDOperand(Lo.Val, 1);
          break;
        }
        if (HasMULHS) {
          // We can emit a mulhs+mul.
          Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
          Hi = DAG.getNode(ISD::MULHS, NVT, LL, RL);
          break;
        }
      }
      if (HasUMUL_LOHI) {
        // Lo,Hi = umul LHS, RHS.
        SDOperand UMulLOHI = DAG.getNode(ISD::UMUL_LOHI,
                                         DAG.getVTList(NVT, NVT), LL, RL);
        Lo = UMulLOHI;
        Hi = UMulLOHI.getValue(1);
        RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
        LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
        break;
      }
      if (HasMULHU) {
        Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
        RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
        LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
        Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
        break;
      }
    }

    // If nothing else, we can make a libcall.
    Lo = ExpandLibCall(RTLIB::MUL_I64, Node, false/*sign irrelevant*/, Hi);
    break;
  }
  case ISD::SDIV:
    Lo = ExpandLibCall(RTLIB::SDIV_I64, Node, true, Hi);
    break;
  case ISD::UDIV:
    Lo = ExpandLibCall(RTLIB::UDIV_I64, Node, true, Hi);
    break;
  case ISD::SREM:
    Lo = ExpandLibCall(RTLIB::SREM_I64, Node, true, Hi);
    break;
  case ISD::UREM:
    Lo = ExpandLibCall(RTLIB::UREM_I64, Node, true, Hi);
    break;

  case ISD::FADD:
    Lo = ExpandLibCall(GetFPLibCall(VT, RTLIB::ADD_F32,
                                        RTLIB::ADD_F64,
                                        RTLIB::ADD_F80,
                                        RTLIB::ADD_PPCF128),
                       Node, false, Hi);
    break;
  case ISD::FSUB:
    Lo = ExpandLibCall(GetFPLibCall(VT, RTLIB::SUB_F32,
                                        RTLIB::SUB_F64,
                                        RTLIB::SUB_F80,
                                        RTLIB::SUB_PPCF128),
                       Node, false, Hi);
    break;
  case ISD::FMUL:
    Lo = ExpandLibCall(GetFPLibCall(VT, RTLIB::MUL_F32,
                                        RTLIB::MUL_F64,
                                        RTLIB::MUL_F80,
                                        RTLIB::MUL_PPCF128),
                       Node, false, Hi);
    break;
  case ISD::FDIV:
    Lo = ExpandLibCall(GetFPLibCall(VT, RTLIB::DIV_F32,
                                        RTLIB::DIV_F64,
                                        RTLIB::DIV_F80,
                                        RTLIB::DIV_PPCF128),
                       Node, false, Hi);
    break;
  case ISD::FP_EXTEND:
    if (VT == MVT::ppcf128) {
      assert(Node->getOperand(0).getValueType()==MVT::f32 ||
             Node->getOperand(0).getValueType()==MVT::f64);
      const uint64_t zero = 0;
      if (Node->getOperand(0).getValueType()==MVT::f32)
        Hi = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Node->getOperand(0));
      else
        Hi = Node->getOperand(0);
      Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
      break;
    }
    Lo = ExpandLibCall(RTLIB::FPEXT_F32_F64, Node, true, Hi);
    break;
  case ISD::FP_ROUND:
    Lo = ExpandLibCall(RTLIB::FPROUND_F64_F32, Node, true, Hi);
    break;
  case ISD::FPOWI:
    Lo = ExpandLibCall(GetFPLibCall(VT, RTLIB::POWI_F32,
                                        RTLIB::POWI_F64,
                                        RTLIB::POWI_F80,
                                        RTLIB::POWI_PPCF128),
                       Node, false, Hi);
    break;
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS: {
    RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
    switch(Node->getOpcode()) {
    case ISD::FSQRT:
      LC = GetFPLibCall(VT, RTLIB::SQRT_F32, RTLIB::SQRT_F64,
                        RTLIB::SQRT_F80, RTLIB::SQRT_PPCF128);
      break;
    case ISD::FSIN:
      LC = GetFPLibCall(VT, RTLIB::SIN_F32, RTLIB::SIN_F64,
                        RTLIB::SIN_F80, RTLIB::SIN_PPCF128);
      break;
    case ISD::FCOS:
      LC = GetFPLibCall(VT, RTLIB::COS_F32, RTLIB::COS_F64,
                        RTLIB::COS_F80, RTLIB::COS_PPCF128);
      break;
    default: assert(0 && "Unreachable!");
    }
    Lo = ExpandLibCall(LC, Node, false, Hi);
    break;
  }
  case ISD::FABS: {
    if (VT == MVT::ppcf128) {
      SDOperand Tmp;
      ExpandOp(Node->getOperand(0), Lo, Tmp);
      Hi = DAG.getNode(ISD::FABS, NVT, Tmp);
      // lo = hi==fabs(hi) ? lo : -lo;
      Lo = DAG.getNode(ISD::SELECT_CC, NVT, Hi, Tmp,
                    Lo, DAG.getNode(ISD::FNEG, NVT, Lo),
                    DAG.getCondCode(ISD::SETEQ));
      break;
    }
    SDOperand Mask = (VT == MVT::f64)
      ? DAG.getConstantFP(BitsToDouble(~(1ULL << 63)), VT)
      : DAG.getConstantFP(BitsToFloat(~(1U << 31)), VT);
    Mask = DAG.getNode(ISD::BIT_CONVERT, NVT, Mask);
    Lo = DAG.getNode(ISD::BIT_CONVERT, NVT, Node->getOperand(0));
    Lo = DAG.getNode(ISD::AND, NVT, Lo, Mask);
    if (getTypeAction(NVT) == Expand)
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  case ISD::FNEG: {
    if (VT == MVT::ppcf128) {
      ExpandOp(Node->getOperand(0), Lo, Hi);
      Lo = DAG.getNode(ISD::FNEG, MVT::f64, Lo);
      Hi = DAG.getNode(ISD::FNEG, MVT::f64, Hi);
      break;
    }
    SDOperand Mask = (VT == MVT::f64)
      ? DAG.getConstantFP(BitsToDouble(1ULL << 63), VT)
      : DAG.getConstantFP(BitsToFloat(1U << 31), VT);
    Mask = DAG.getNode(ISD::BIT_CONVERT, NVT, Mask);
    Lo = DAG.getNode(ISD::BIT_CONVERT, NVT, Node->getOperand(0));
    Lo = DAG.getNode(ISD::XOR, NVT, Lo, Mask);
    if (getTypeAction(NVT) == Expand)
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  case ISD::FCOPYSIGN: {
    Lo = ExpandFCOPYSIGNToBitwiseOps(Node, NVT, DAG, TLI);
    if (getTypeAction(NVT) == Expand)
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    bool isSigned = Node->getOpcode() == ISD::SINT_TO_FP;
    MVT::ValueType SrcVT = Node->getOperand(0).getValueType();

    // Promote the operand if needed.  Do this before checking for
    // ppcf128 so conversions of i16 and i8 work.
    if (getTypeAction(SrcVT) == Promote) {
      SDOperand Tmp = PromoteOp(Node->getOperand(0));
      Tmp = isSigned
        ? DAG.getNode(ISD::SIGN_EXTEND_INREG, Tmp.getValueType(), Tmp,
                      DAG.getValueType(SrcVT))
        : DAG.getZeroExtendInReg(Tmp, SrcVT);
      Node = DAG.UpdateNodeOperands(Op, Tmp).Val;
      SrcVT = Node->getOperand(0).getValueType();
    }

    if (VT == MVT::ppcf128 && SrcVT == MVT::i32) {
      static const uint64_t zero = 0;
      if (isSigned) {
        Hi = LegalizeOp(DAG.getNode(ISD::SINT_TO_FP, MVT::f64, 
                                    Node->getOperand(0)));
        Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
      } else {
        static const uint64_t TwoE32[] = { 0x41f0000000000000LL, 0 };
        Hi = LegalizeOp(DAG.getNode(ISD::SINT_TO_FP, MVT::f64, 
                                    Node->getOperand(0)));
        Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
        Hi = DAG.getNode(ISD::BUILD_PAIR, VT, Lo, Hi);
        // X>=0 ? {(f64)x, 0} : {(f64)x, 0} + 2^32
        ExpandOp(DAG.getNode(ISD::SELECT_CC, MVT::ppcf128, Node->getOperand(0),
                             DAG.getConstant(0, MVT::i32), 
                             DAG.getNode(ISD::FADD, MVT::ppcf128, Hi,
                                         DAG.getConstantFP(
                                            APFloat(APInt(128, 2, TwoE32)),
                                            MVT::ppcf128)),
                             Hi,
                             DAG.getCondCode(ISD::SETLT)),
                 Lo, Hi);
      }
      break;
    }
    if (VT == MVT::ppcf128 && SrcVT == MVT::i64 && !isSigned) {
      // si64->ppcf128 done by libcall, below
      static const uint64_t TwoE64[] = { 0x43f0000000000000LL, 0 };
      ExpandOp(DAG.getNode(ISD::SINT_TO_FP, MVT::ppcf128, Node->getOperand(0)),
               Lo, Hi);
      Hi = DAG.getNode(ISD::BUILD_PAIR, VT, Lo, Hi);
      // x>=0 ? (ppcf128)(i64)x : (ppcf128)(i64)x + 2^64
      ExpandOp(DAG.getNode(ISD::SELECT_CC, MVT::ppcf128, Node->getOperand(0),
                           DAG.getConstant(0, MVT::i64), 
                           DAG.getNode(ISD::FADD, MVT::ppcf128, Hi,
                                       DAG.getConstantFP(
                                          APFloat(APInt(128, 2, TwoE64)),
                                          MVT::ppcf128)),
                           Hi,
                           DAG.getCondCode(ISD::SETLT)),
               Lo, Hi);
      break;
    }

    Lo = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP, VT,
                       Node->getOperand(0));
    if (getTypeAction(Lo.getValueType()) == Expand)
      // float to i32 etc. can be 'expanded' to a single node.
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  }

  // Make sure the resultant values have been legalized themselves, unless this
  // is a type that requires multi-step expansion.
  if (getTypeAction(NVT) != Expand && NVT != MVT::isVoid) {
    Lo = LegalizeOp(Lo);
    if (Hi.Val)
      // Don't legalize the high part if it is expanded to a single node.
      Hi = LegalizeOp(Hi);
  }

  // Remember in a map if the values will be reused later.
  bool isNew = ExpandedNodes.insert(std::make_pair(Op, std::make_pair(Lo, Hi)));
  assert(isNew && "Value already expanded?!?");
}

/// SplitVectorOp - Given an operand of vector type, break it down into
/// two smaller values, still of vector type.
void SelectionDAGLegalize::SplitVectorOp(SDOperand Op, SDOperand &Lo,
                                         SDOperand &Hi) {
  assert(MVT::isVector(Op.getValueType()) && "Cannot split non-vector type!");
  SDNode *Node = Op.Val;
  unsigned NumElements = MVT::getVectorNumElements(Op.getValueType());
  assert(NumElements > 1 && "Cannot split a single element vector!");

  MVT::ValueType NewEltVT = MVT::getVectorElementType(Op.getValueType());

  unsigned NewNumElts_Lo = 1 << Log2_32(NumElements-1);
  unsigned NewNumElts_Hi = NumElements - NewNumElts_Lo;

  MVT::ValueType NewVT_Lo = MVT::getVectorType(NewEltVT, NewNumElts_Lo);
  MVT::ValueType NewVT_Hi = MVT::getVectorType(NewEltVT, NewNumElts_Hi);

  // See if we already split it.
  std::map<SDOperand, std::pair<SDOperand, SDOperand> >::iterator I
    = SplitNodes.find(Op);
  if (I != SplitNodes.end()) {
    Lo = I->second.first;
    Hi = I->second.second;
    return;
  }
  
  switch (Node->getOpcode()) {
  default: 
#ifndef NDEBUG
    Node->dump(&DAG);
#endif
    assert(0 && "Unhandled operation in SplitVectorOp!");
  case ISD::UNDEF:
    Lo = DAG.getNode(ISD::UNDEF, NewVT_Lo);
    Hi = DAG.getNode(ISD::UNDEF, NewVT_Hi);
    break;
  case ISD::BUILD_PAIR:
    Lo = Node->getOperand(0);
    Hi = Node->getOperand(1);
    break;
  case ISD::INSERT_VECTOR_ELT: {
    if (ConstantSDNode *Idx = dyn_cast<ConstantSDNode>(Node->getOperand(2))) {
      SplitVectorOp(Node->getOperand(0), Lo, Hi);
      unsigned Index = Idx->getValue();
      SDOperand ScalarOp = Node->getOperand(1);
      if (Index < NewNumElts_Lo)
        Lo = DAG.getNode(ISD::INSERT_VECTOR_ELT, NewVT_Lo, Lo, ScalarOp,
                         DAG.getIntPtrConstant(Index));
      else
        Hi = DAG.getNode(ISD::INSERT_VECTOR_ELT, NewVT_Hi, Hi, ScalarOp,
                         DAG.getIntPtrConstant(Index - NewNumElts_Lo));
      break;
    }
    SDOperand Tmp = PerformInsertVectorEltInMemory(Node->getOperand(0),
                                                   Node->getOperand(1),
                                                   Node->getOperand(2));
    SplitVectorOp(Tmp, Lo, Hi);
    break;
  }
  case ISD::VECTOR_SHUFFLE: {
    // Build the low part.
    SDOperand Mask = Node->getOperand(2);
    SmallVector<SDOperand, 8> Ops;
    MVT::ValueType PtrVT = TLI.getPointerTy();
    
    // Insert all of the elements from the input that are needed.  We use 
    // buildvector of extractelement here because the input vectors will have
    // to be legalized, so this makes the code simpler.
    for (unsigned i = 0; i != NewNumElts_Lo; ++i) {
      SDOperand IdxNode = Mask.getOperand(i);
      if (IdxNode.getOpcode() == ISD::UNDEF) {
        Ops.push_back(DAG.getNode(ISD::UNDEF, NewEltVT));
        continue;
      }
      unsigned Idx = cast<ConstantSDNode>(IdxNode)->getValue();
      SDOperand InVec = Node->getOperand(0);
      if (Idx >= NumElements) {
        InVec = Node->getOperand(1);
        Idx -= NumElements;
      }
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, NewEltVT, InVec,
                                DAG.getConstant(Idx, PtrVT)));
    }
    Lo = DAG.getNode(ISD::BUILD_VECTOR, NewVT_Lo, &Ops[0], Ops.size());
    Ops.clear();
    
    for (unsigned i = NewNumElts_Lo; i != NumElements; ++i) {
      SDOperand IdxNode = Mask.getOperand(i);
      if (IdxNode.getOpcode() == ISD::UNDEF) {
        Ops.push_back(DAG.getNode(ISD::UNDEF, NewEltVT));
        continue;
      }
      unsigned Idx = cast<ConstantSDNode>(IdxNode)->getValue();
      SDOperand InVec = Node->getOperand(0);
      if (Idx >= NumElements) {
        InVec = Node->getOperand(1);
        Idx -= NumElements;
      }
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, NewEltVT, InVec,
                                DAG.getConstant(Idx, PtrVT)));
    }
    Hi = DAG.getNode(ISD::BUILD_VECTOR, NewVT_Lo, &Ops[0], Ops.size());
    break;
  }
  case ISD::BUILD_VECTOR: {
    SmallVector<SDOperand, 8> LoOps(Node->op_begin(), 
                                    Node->op_begin()+NewNumElts_Lo);
    Lo = DAG.getNode(ISD::BUILD_VECTOR, NewVT_Lo, &LoOps[0], LoOps.size());

    SmallVector<SDOperand, 8> HiOps(Node->op_begin()+NewNumElts_Lo, 
                                    Node->op_end());
    Hi = DAG.getNode(ISD::BUILD_VECTOR, NewVT_Hi, &HiOps[0], HiOps.size());
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // FIXME: Handle non-power-of-two vectors?
    unsigned NewNumSubvectors = Node->getNumOperands() / 2;
    if (NewNumSubvectors == 1) {
      Lo = Node->getOperand(0);
      Hi = Node->getOperand(1);
    } else {
      SmallVector<SDOperand, 8> LoOps(Node->op_begin(), 
                                      Node->op_begin()+NewNumSubvectors);
      Lo = DAG.getNode(ISD::CONCAT_VECTORS, NewVT_Lo, &LoOps[0], LoOps.size());

      SmallVector<SDOperand, 8> HiOps(Node->op_begin()+NewNumSubvectors, 
                                      Node->op_end());
      Hi = DAG.getNode(ISD::CONCAT_VECTORS, NewVT_Hi, &HiOps[0], HiOps.size());
    }
    break;
  }
  case ISD::SELECT: {
    SDOperand Cond = Node->getOperand(0);

    SDOperand LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(1), LL, LH);
    SplitVectorOp(Node->getOperand(2), RL, RH);

    if (MVT::isVector(Cond.getValueType())) {
      // Handle a vector merge.
      SDOperand CL, CH;
      SplitVectorOp(Cond, CL, CH);
      Lo = DAG.getNode(Node->getOpcode(), NewVT_Lo, CL, LL, RL);
      Hi = DAG.getNode(Node->getOpcode(), NewVT_Hi, CH, LH, RH);
    } else {
      // Handle a simple select with vector operands.
      Lo = DAG.getNode(Node->getOpcode(), NewVT_Lo, Cond, LL, RL);
      Hi = DAG.getNode(Node->getOpcode(), NewVT_Hi, Cond, LH, RH);
    }
    break;
  }
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::UREM:
  case ISD::SREM:
  case ISD::FREM: {
    SDOperand LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(0), LL, LH);
    SplitVectorOp(Node->getOperand(1), RL, RH);
    
    Lo = DAG.getNode(Node->getOpcode(), NewVT_Lo, LL, RL);
    Hi = DAG.getNode(Node->getOpcode(), NewVT_Hi, LH, RH);
    break;
  }
  case ISD::FPOWI: {
    SDOperand L, H;
    SplitVectorOp(Node->getOperand(0), L, H);

    Lo = DAG.getNode(Node->getOpcode(), NewVT_Lo, L, Node->getOperand(1));
    Hi = DAG.getNode(Node->getOpcode(), NewVT_Hi, H, Node->getOperand(1));
    break;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    SDOperand L, H;
    SplitVectorOp(Node->getOperand(0), L, H);

    Lo = DAG.getNode(Node->getOpcode(), NewVT_Lo, L);
    Hi = DAG.getNode(Node->getOpcode(), NewVT_Hi, H);
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    SDOperand Ch = LD->getChain();
    SDOperand Ptr = LD->getBasePtr();
    const Value *SV = LD->getSrcValue();
    int SVOffset = LD->getSrcValueOffset();
    unsigned Alignment = LD->getAlignment();
    bool isVolatile = LD->isVolatile();

    Lo = DAG.getLoad(NewVT_Lo, Ch, Ptr, SV, SVOffset, isVolatile, Alignment);
    unsigned IncrementSize = NewNumElts_Lo * MVT::getSizeInBits(NewEltVT)/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    SVOffset += IncrementSize;
    Alignment = MinAlign(Alignment, IncrementSize);
    Hi = DAG.getLoad(NewVT_Hi, Ch, Ptr, SV, SVOffset, isVolatile, Alignment);
    
    // Build a factor node to remember that this load is independent of the
    // other one.
    SDOperand TF = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                               Hi.getValue(1));
    
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(TF));
    break;
  }
  case ISD::BIT_CONVERT: {
    // We know the result is a vector.  The input may be either a vector or a
    // scalar value.
    SDOperand InOp = Node->getOperand(0);
    if (!MVT::isVector(InOp.getValueType()) ||
        MVT::getVectorNumElements(InOp.getValueType()) == 1) {
      // The input is a scalar or single-element vector.
      // Lower to a store/load so that it can be split.
      // FIXME: this could be improved probably.
      SDOperand Ptr = DAG.CreateStackTemporary(InOp.getValueType());
      FrameIndexSDNode *FI = cast<FrameIndexSDNode>(Ptr.Val);

      SDOperand St = DAG.getStore(DAG.getEntryNode(),
                                  InOp, Ptr,
                                  PseudoSourceValue::getFixedStack(),
                                  FI->getIndex());
      InOp = DAG.getLoad(Op.getValueType(), St, Ptr,
                         PseudoSourceValue::getFixedStack(),
                         FI->getIndex());
    }
    // Split the vector and convert each of the pieces now.
    SplitVectorOp(InOp, Lo, Hi);
    Lo = DAG.getNode(ISD::BIT_CONVERT, NewVT_Lo, Lo);
    Hi = DAG.getNode(ISD::BIT_CONVERT, NewVT_Hi, Hi);
    break;
  }
  }
      
  // Remember in a map if the values will be reused later.
  bool isNew = 
    SplitNodes.insert(std::make_pair(Op, std::make_pair(Lo, Hi))).second;
  assert(isNew && "Value already split?!?");
}


/// ScalarizeVectorOp - Given an operand of single-element vector type
/// (e.g. v1f32), convert it into the equivalent operation that returns a
/// scalar (e.g. f32) value.
SDOperand SelectionDAGLegalize::ScalarizeVectorOp(SDOperand Op) {
  assert(MVT::isVector(Op.getValueType()) &&
         "Bad ScalarizeVectorOp invocation!");
  SDNode *Node = Op.Val;
  MVT::ValueType NewVT = MVT::getVectorElementType(Op.getValueType());
  assert(MVT::getVectorNumElements(Op.getValueType()) == 1);
  
  // See if we already scalarized it.
  std::map<SDOperand, SDOperand>::iterator I = ScalarizedNodes.find(Op);
  if (I != ScalarizedNodes.end()) return I->second;
  
  SDOperand Result;
  switch (Node->getOpcode()) {
  default: 
#ifndef NDEBUG
    Node->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Unknown vector operation in ScalarizeVectorOp!");
  case ISD::ADD:
  case ISD::FADD:
  case ISD::SUB:
  case ISD::FSUB:
  case ISD::MUL:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FREM:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    Result = DAG.getNode(Node->getOpcode(),
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)),
                         ScalarizeVectorOp(Node->getOperand(1)));
    break;
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
    Result = DAG.getNode(Node->getOpcode(),
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)));
    break;
  case ISD::FPOWI:
    Result = DAG.getNode(Node->getOpcode(),
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)),
                         Node->getOperand(1));
    break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    SDOperand Ch = LegalizeOp(LD->getChain());     // Legalize the chain.
    SDOperand Ptr = LegalizeOp(LD->getBasePtr());  // Legalize the pointer.
    
    const Value *SV = LD->getSrcValue();
    int SVOffset = LD->getSrcValueOffset();
    Result = DAG.getLoad(NewVT, Ch, Ptr, SV, SVOffset,
                         LD->isVolatile(), LD->getAlignment());

    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }
  case ISD::BUILD_VECTOR:
    Result = Node->getOperand(0);
    break;
  case ISD::INSERT_VECTOR_ELT:
    // Returning the inserted scalar element.
    Result = Node->getOperand(1);
    break;
  case ISD::CONCAT_VECTORS:
    assert(Node->getOperand(0).getValueType() == NewVT &&
           "Concat of non-legal vectors not yet supported!");
    Result = Node->getOperand(0);
    break;
  case ISD::VECTOR_SHUFFLE: {
    // Figure out if the scalar is the LHS or RHS and return it.
    SDOperand EltNum = Node->getOperand(2).getOperand(0);
    if (cast<ConstantSDNode>(EltNum)->getValue())
      Result = ScalarizeVectorOp(Node->getOperand(1));
    else
      Result = ScalarizeVectorOp(Node->getOperand(0));
    break;
  }
  case ISD::EXTRACT_SUBVECTOR:
    Result = Node->getOperand(0);
    assert(Result.getValueType() == NewVT);
    break;
  case ISD::BIT_CONVERT:
    Result = DAG.getNode(ISD::BIT_CONVERT, NewVT, Op.getOperand(0));
    break;
  case ISD::SELECT:
    Result = DAG.getNode(ISD::SELECT, NewVT, Op.getOperand(0),
                         ScalarizeVectorOp(Op.getOperand(1)),
                         ScalarizeVectorOp(Op.getOperand(2)));
    break;
  }

  if (TLI.isTypeLegal(NewVT))
    Result = LegalizeOp(Result);
  bool isNew = ScalarizedNodes.insert(std::make_pair(Op, Result)).second;
  assert(isNew && "Value already scalarized?");
  return Result;
}


// SelectionDAG::Legalize - This is the entry point for the file.
//
void SelectionDAG::Legalize() {
  if (ViewLegalizeDAGs) viewGraph();

  /// run - This is the main entry point to this class.
  ///
  SelectionDAGLegalize(*this).LegalizeDAG();
}

