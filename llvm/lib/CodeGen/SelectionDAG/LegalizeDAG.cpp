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
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Analysis/DebugInfo.h"
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
#include "llvm/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
using namespace llvm;

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
  bool TypesNeedLegalizing;

  // Libcall insertion helpers.
  
  /// LastCALLSEQ_END - This keeps track of the CALLSEQ_END node that has been
  /// legalized.  We use this to ensure that calls are properly serialized
  /// against each other, including inserted libcalls.
  SDValue LastCALLSEQ_END;
  
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
  DenseMap<SDValue, SDValue> LegalizedNodes;

  /// PromotedNodes - For nodes that are below legal width, and that have more
  /// than one use, this map indicates what promoted value to use.  This allows
  /// us to avoid promoting the same thing more than once.
  DenseMap<SDValue, SDValue> PromotedNodes;

  /// ExpandedNodes - For nodes that need to be expanded this map indicates
  /// which operands are the expanded version of the input.  This allows
  /// us to avoid expanding the same node more than once.
  DenseMap<SDValue, std::pair<SDValue, SDValue> > ExpandedNodes;

  /// SplitNodes - For vector nodes that need to be split, this map indicates
  /// which operands are the split version of the input.  This allows us
  /// to avoid splitting the same node more than once.
  std::map<SDValue, std::pair<SDValue, SDValue> > SplitNodes;
  
  /// ScalarizedNodes - For nodes that need to be converted from vector types to
  /// scalar types, this contains the mapping of ones we have already
  /// processed to the result.
  std::map<SDValue, SDValue> ScalarizedNodes;
  
  /// WidenNodes - For nodes that need to be widened from one vector type to
  /// another, this contains the mapping of those that we have already widen.
  /// This allows us to avoid widening more than once.
  std::map<SDValue, SDValue> WidenNodes;

  void AddLegalizedOperand(SDValue From, SDValue To) {
    LegalizedNodes.insert(std::make_pair(From, To));
    // If someone requests legalization of the new node, return itself.
    if (From != To)
      LegalizedNodes.insert(std::make_pair(To, To));
  }
  void AddPromotedOperand(SDValue From, SDValue To) {
    bool isNew = PromotedNodes.insert(std::make_pair(From, To)).second;
    assert(isNew && "Got into the map somehow?");
    isNew = isNew;
    // If someone requests legalization of the new node, return itself.
    LegalizedNodes.insert(std::make_pair(To, To));
  }
  void AddWidenedOperand(SDValue From, SDValue To) {
    bool isNew = WidenNodes.insert(std::make_pair(From, To)).second;
    assert(isNew && "Got into the map somehow?");
    isNew = isNew;
    // If someone requests legalization of the new node, return itself.
    LegalizedNodes.insert(std::make_pair(To, To));
  }

public:
  explicit SelectionDAGLegalize(SelectionDAG &DAG, bool TypesNeedLegalizing);

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT VT) const {
    return (LegalizeAction)ValueTypeActions.getTypeAction(VT);
  }

  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT VT) const {
    return getTypeAction(VT) == Legal;
  }

  void LegalizeDAG();

private:
  /// HandleOp - Legalize, Promote, or Expand the specified operand as
  /// appropriate for its type.
  void HandleOp(SDValue Op);
    
  /// LegalizeOp - We know that the specified value has a legal type.
  /// Recursively ensure that the operands have legal types, then return the
  /// result.
  SDValue LegalizeOp(SDValue O);
  
  /// UnrollVectorOp - We know that the given vector has a legal type, however
  /// the operation it performs is not legal and is an operation that we have
  /// no way of lowering.  "Unroll" the vector, splitting out the scalars and
  /// operating on each element individually.
  SDValue UnrollVectorOp(SDValue O);
  
  /// PerformInsertVectorEltInMemory - Some target cannot handle a variable
  /// insertion index for the INSERT_VECTOR_ELT instruction.  In this case, it
  /// is necessary to spill the vector being inserted into to memory, perform
  /// the insert there, and then read the result back.
  SDValue PerformInsertVectorEltInMemory(SDValue Vec, SDValue Val,
                                           SDValue Idx, DebugLoc dl);

  /// PromoteOp - Given an operation that produces a value in an invalid type,
  /// promote it to compute the value into a larger type.  The produced value
  /// will have the correct bits for the low portion of the register, but no
  /// guarantee is made about the top bits: it may be zero, sign-extended, or
  /// garbage.
  SDValue PromoteOp(SDValue O);

  /// ExpandOp - Expand the specified SDValue into its two component pieces
  /// Lo&Hi.  Note that the Op MUST be an expanded type.  As a result of this,
  /// the LegalizedNodes map is filled in for any results that are not expanded,
  /// the ExpandedNodes map is filled in for any results that are expanded, and
  /// the Lo/Hi values are returned.   This applies to integer types and Vector
  /// types.
  void ExpandOp(SDValue O, SDValue &Lo, SDValue &Hi);

  /// WidenVectorOp - Widen a vector operation to a wider type given by WidenVT 
  /// (e.g., v3i32 to v4i32).  The produced value will have the correct value
  /// for the existing elements but no guarantee is made about the new elements
  /// at the end of the vector: it may be zero, ones, or garbage. This is useful
  /// when we have an instruction operating on an illegal vector type and we
  /// want to widen it to do the computation on a legal wider vector type.
  SDValue WidenVectorOp(SDValue Op, MVT WidenVT);

  /// SplitVectorOp - Given an operand of vector type, break it down into
  /// two smaller values.
  void SplitVectorOp(SDValue O, SDValue &Lo, SDValue &Hi);
  
  /// ScalarizeVectorOp - Given an operand of single-element vector type
  /// (e.g. v1f32), convert it into the equivalent operation that returns a
  /// scalar (e.g. f32) value.
  SDValue ScalarizeVectorOp(SDValue O);
  
  /// Useful 16 element vector type that is used to pass operands for widening.
  typedef SmallVector<SDValue, 16> SDValueVector;  
  
  /// LoadWidenVectorOp - Load a vector for a wider type. Returns true if
  /// the LdChain contains a single load and false if it contains a token
  /// factor for multiple loads. It takes
  ///   Result:  location to return the result
  ///   LdChain: location to return the load chain
  ///   Op:      load operation to widen
  ///   NVT:     widen vector result type we want for the load
  bool LoadWidenVectorOp(SDValue& Result, SDValue& LdChain, 
                         SDValue Op, MVT NVT);
                        
  /// Helper genWidenVectorLoads - Helper function to generate a set of
  /// loads to load a vector with a resulting wider type. It takes
  ///   LdChain: list of chains for the load we have generated
  ///   Chain:   incoming chain for the ld vector
  ///   BasePtr: base pointer to load from
  ///   SV:      memory disambiguation source value
  ///   SVOffset:  memory disambiugation offset
  ///   Alignment: alignment of the memory
  ///   isVolatile: volatile load
  ///   LdWidth:    width of memory that we want to load 
  ///   ResType:    the wider result result type for the resulting loaded vector
  SDValue genWidenVectorLoads(SDValueVector& LdChain, SDValue Chain,
                                SDValue BasePtr, const Value *SV,
                                int SVOffset, unsigned Alignment,
                                bool isVolatile, unsigned LdWidth,
                                MVT ResType, DebugLoc dl);
  
  /// StoreWidenVectorOp - Stores a widen vector into non widen memory
  /// location. It takes
  ///     ST:      store node that we want to replace
  ///     Chain:   incoming store chain
  ///     BasePtr: base address of where we want to store into
  SDValue StoreWidenVectorOp(StoreSDNode *ST, SDValue Chain, 
                               SDValue BasePtr);
  
  /// Helper genWidenVectorStores - Helper function to generate a set of
  /// stores to store a widen vector into non widen memory
  // It takes
  //   StChain: list of chains for the stores we have generated
  //   Chain:   incoming chain for the ld vector
  //   BasePtr: base pointer to load from
  //   SV:      memory disambiguation source value
  //   SVOffset:   memory disambiugation offset
  //   Alignment:  alignment of the memory
  //   isVolatile: volatile lod
  //   ValOp:   value to store  
  //   StWidth: width of memory that we want to store 
  void genWidenVectorStores(SDValueVector& StChain, SDValue Chain,
                            SDValue BasePtr, const Value *SV,
                            int SVOffset, unsigned Alignment,
                            bool isVolatile, SDValue ValOp,
                            unsigned StWidth, DebugLoc dl);
 
  /// isShuffleLegal - Return non-null if a vector shuffle is legal with the
  /// specified mask and type.  Targets can specify exactly which masks they
  /// support and the code generator is tasked with not creating illegal masks.
  ///
  /// Note that this will also return true for shuffles that are promoted to a
  /// different type.
  ///
  /// If this is a legal shuffle, this method returns the (possibly promoted)
  /// build_vector Mask.  If it's not a legal shuffle, it returns null.
  SDNode *isShuffleLegal(MVT VT, SDValue Mask) const;
  
  bool LegalizeAllNodesNotLeadingTo(SDNode *N, SDNode *Dest,
                                    SmallPtrSet<SDNode*, 32> &NodesLeadingTo);

  void LegalizeSetCCOperands(SDValue &LHS, SDValue &RHS, SDValue &CC,
                             DebugLoc dl);
  void LegalizeSetCCCondCode(MVT VT, SDValue &LHS, SDValue &RHS, SDValue &CC,
                             DebugLoc dl);
  void LegalizeSetCC(MVT VT, SDValue &LHS, SDValue &RHS, SDValue &CC,
                     DebugLoc dl) {
    LegalizeSetCCOperands(LHS, RHS, CC, dl);
    LegalizeSetCCCondCode(VT, LHS, RHS, CC, dl);
  }
    
  SDValue ExpandLibCall(RTLIB::Libcall LC, SDNode *Node, bool isSigned,
                          SDValue &Hi);
  SDValue ExpandIntToFP(bool isSigned, MVT DestTy, SDValue Source, DebugLoc dl);

  SDValue EmitStackConvert(SDValue SrcOp, MVT SlotVT, MVT DestVT, DebugLoc dl);
  SDValue ExpandBUILD_VECTOR(SDNode *Node);
  SDValue ExpandSCALAR_TO_VECTOR(SDNode *Node);
  SDValue LegalizeINT_TO_FP(SDValue Result, bool isSigned, MVT DestTy, 
                            SDValue Op, DebugLoc dl);
  SDValue ExpandLegalINT_TO_FP(bool isSigned, SDValue LegalOp, MVT DestVT,
                               DebugLoc dl);
  SDValue PromoteLegalINT_TO_FP(SDValue LegalOp, MVT DestVT, bool isSigned,
                                DebugLoc dl);
  SDValue PromoteLegalFP_TO_INT(SDValue LegalOp, MVT DestVT, bool isSigned,
                                DebugLoc dl);

  SDValue ExpandBSWAP(SDValue Op, DebugLoc dl);
  SDValue ExpandBitCount(unsigned Opc, SDValue Op, DebugLoc dl);
  bool ExpandShift(unsigned Opc, SDValue Op, SDValue Amt,
                   SDValue &Lo, SDValue &Hi, DebugLoc dl);
  void ExpandShiftParts(unsigned NodeOp, SDValue Op, SDValue Amt,
                        SDValue &Lo, SDValue &Hi, DebugLoc dl);

  SDValue ExpandEXTRACT_SUBVECTOR(SDValue Op);
  SDValue ExpandEXTRACT_VECTOR_ELT(SDValue Op);
};
}

/// isVectorShuffleLegal - Return true if a vector shuffle is legal with the
/// specified mask and type.  Targets can specify exactly which masks they
/// support and the code generator is tasked with not creating illegal masks.
///
/// Note that this will also return true for shuffles that are promoted to a
/// different type.
SDNode *SelectionDAGLegalize::isShuffleLegal(MVT VT, SDValue Mask) const {
  switch (TLI.getOperationAction(ISD::VECTOR_SHUFFLE, VT)) {
  default: return 0;
  case TargetLowering::Legal:
  case TargetLowering::Custom:
    break;
  case TargetLowering::Promote: {
    // If this is promoted to a different type, convert the shuffle mask and
    // ask if it is legal in the promoted type!
    MVT NVT = TLI.getTypeToPromoteTo(ISD::VECTOR_SHUFFLE, VT);
    MVT EltVT = NVT.getVectorElementType();

    // If we changed # elements, change the shuffle mask.
    unsigned NumEltsGrowth =
      NVT.getVectorNumElements() / VT.getVectorNumElements();
    assert(NumEltsGrowth && "Cannot promote to vector type with fewer elts!");
    if (NumEltsGrowth > 1) {
      // Renumber the elements.
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0, e = Mask.getNumOperands(); i != e; ++i) {
        SDValue InOp = Mask.getOperand(i);
        for (unsigned j = 0; j != NumEltsGrowth; ++j) {
          if (InOp.getOpcode() == ISD::UNDEF)
            Ops.push_back(DAG.getNode(ISD::UNDEF, 
                                      InOp.getNode()->getDebugLoc(), EltVT));
          else {
            unsigned InEltNo = cast<ConstantSDNode>(InOp)->getZExtValue();
            Ops.push_back(DAG.getConstant(InEltNo*NumEltsGrowth+j, EltVT));
          }
        }
      }
      Mask = DAG.getNode(ISD::BUILD_VECTOR, Mask.getNode()->getDebugLoc(),
                         NVT, &Ops[0], Ops.size());
    }
    VT = NVT;
    break;
  }
  }
  return TLI.isShuffleMaskLegal(Mask, VT) ? Mask.getNode() : 0;
}

SelectionDAGLegalize::SelectionDAGLegalize(SelectionDAG &dag, bool types)
  : TLI(dag.getTargetLoweringInfo()), DAG(dag), TypesNeedLegalizing(types),
    ValueTypeActions(TLI.getValueTypeActions()) {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");
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
  DAG.AssignTopologicalOrder();
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = prior(DAG.allnodes_end()); I != next(E); ++I)
    HandleOp(SDValue(I, 0));

  // Finally, it's possible the root changed.  Get the new root.
  SDValue OldRoot = DAG.getRoot();
  assert(LegalizedNodes.count(OldRoot) && "Root didn't get legalized?");
  DAG.setRoot(LegalizedNodes[OldRoot]);

  ExpandedNodes.clear();
  LegalizedNodes.clear();
  PromotedNodes.clear();
  SplitNodes.clear();
  ScalarizedNodes.clear();
  WidenNodes.clear();

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
  SDValue TheChain(Node, Node->getNumValues()-1);
  if (TheChain.getValueType() != MVT::Other) {
    // Sometimes it's at the beginning.
    TheChain = SDValue(Node, 0);
    if (TheChain.getValueType() != MVT::Other) {
      // Otherwise, hunt for it.
      for (unsigned i = 1, e = Node->getNumValues(); i != e; ++i)
        if (Node->getValueType(i) == MVT::Other) {
          TheChain = SDValue(Node, i);
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
    SDNode *User = *UI;
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
  return FindCallStartFromCallEnd(Node->getOperand(0).getNode());
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
    if (LegalizedNodes.count(SDValue(N, 0))) return false;
    break;
  case Promote:
    if (PromotedNodes.count(SDValue(N, 0))) return false;
    break;
  case Expand:
    if (ExpandedNodes.count(SDValue(N, 0))) return false;
    break;
  }
  
  // Okay, this node has not already been legalized.  Check and legalize all
  // operands.  If none lead to Dest, then we can legalize this node.
  bool OperandsLeadToDest = false;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    OperandsLeadToDest |=     // If an operand leads to Dest, so do we.
      LegalizeAllNodesNotLeadingTo(N->getOperand(i).getNode(), Dest, NodesLeadingTo);

  if (OperandsLeadToDest) {
    NodesLeadingTo.insert(N);
    return true;
  }

  // Okay, this node looks safe, legalize it and return false.
  HandleOp(SDValue(N, 0));
  return false;
}

/// HandleOp - Legalize, Promote, Widen, or Expand the specified operand as
/// appropriate for its type.
void SelectionDAGLegalize::HandleOp(SDValue Op) {
  MVT VT = Op.getValueType();
  // If the type legalizer was run then we should never see any illegal result
  // types here except for target constants (the type legalizer does not touch
  // those) or for build vector used as a mask for a vector shuffle.
  // FIXME: We can removed the BUILD_VECTOR case when we fix PR2957.
  assert((TypesNeedLegalizing || getTypeAction(VT) == Legal ||
          Op.getOpcode() == ISD::TargetConstant ||
          Op.getOpcode() == ISD::BUILD_VECTOR) &&
         "Illegal type introduced after type legalization?");
  switch (getTypeAction(VT)) {
  default: assert(0 && "Bad type action!");
  case Legal:   (void)LegalizeOp(Op); break;
  case Promote:
    if (!VT.isVector()) {
      (void)PromoteOp(Op);
      break;
    }
    else  {
      // See if we can widen otherwise use Expand to either scalarize or split
      MVT WidenVT = TLI.getWidenVectorType(VT);
      if (WidenVT != MVT::Other) {
        (void) WidenVectorOp(Op, WidenVT);
        break;
      }
      // else fall thru to expand since we can't widen the vector
    }
  case Expand:
    if (!VT.isVector()) {
      // If this is an illegal scalar, expand it into its two component
      // pieces.
      SDValue X, Y;
      if (Op.getOpcode() == ISD::TargetConstant)
        break;  // Allow illegal target nodes.
      ExpandOp(Op, X, Y);
    } else if (VT.getVectorNumElements() == 1) {
      // If this is an illegal single element vector, convert it to a
      // scalar operation.
      (void)ScalarizeVectorOp(Op);
    } else {
      // This is an illegal multiple element vector.
      // Split it in half and legalize both parts.
      SDValue X, Y;
      SplitVectorOp(Op, X, Y);
    }
    break;
  }
}

/// ExpandConstantFP - Expands the ConstantFP node to an integer constant or
/// a load from the constant pool.
static SDValue ExpandConstantFP(ConstantFPSDNode *CFP, bool UseCP,
                                SelectionDAG &DAG, const TargetLowering &TLI) {
  bool Extend = false;

  // If a FP immediate is precise when represented as a float and if the
  // target can do an extending load from float to double, we put it into
  // the constant pool as a float, even if it's is statically typed as a
  // double.  This shrinks FP constants and canonicalizes them for targets where
  // an FP extending load is the same cost as a normal load (such as on the x87
  // fp stack or PPC FP unit).
  MVT VT = CFP->getValueType(0);
  ConstantFP *LLVMC = const_cast<ConstantFP*>(CFP->getConstantFPValue());
  if (!UseCP) {
    if (VT!=MVT::f64 && VT!=MVT::f32)
      assert(0 && "Invalid type expansion");
    return DAG.getConstant(LLVMC->getValueAPF().bitcastToAPInt(),
                           (VT == MVT::f64) ? MVT::i64 : MVT::i32);
  }

  MVT OrigVT = VT;
  MVT SVT = VT;
  while (SVT != MVT::f32) {
    SVT = (MVT::SimpleValueType)(SVT.getSimpleVT() - 1);
    if (CFP->isValueValidForType(SVT, CFP->getValueAPF()) &&
        // Only do this if the target has a native EXTLOAD instruction from
        // smaller type.
        TLI.isLoadExtLegal(ISD::EXTLOAD, SVT) &&
        TLI.ShouldShrinkFPConstant(OrigVT)) {
      const Type *SType = SVT.getTypeForMVT();
      LLVMC = cast<ConstantFP>(ConstantExpr::getFPTrunc(LLVMC, SType));
      VT = SVT;
      Extend = true;
    }
  }

  SDValue CPIdx = DAG.getConstantPool(LLVMC, TLI.getPointerTy());
  unsigned Alignment = 1 << cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  if (Extend)
    return DAG.getExtLoad(ISD::EXTLOAD, OrigVT, DAG.getEntryNode(),
                          CPIdx, PseudoSourceValue::getConstantPool(),
                          0, VT, false, Alignment);
  return DAG.getLoad(OrigVT, DAG.getEntryNode(), CPIdx,
                     PseudoSourceValue::getConstantPool(), 0, false, Alignment);
}


/// ExpandFCOPYSIGNToBitwiseOps - Expands fcopysign to a series of bitwise
/// operations.
static
SDValue ExpandFCOPYSIGNToBitwiseOps(SDNode *Node, MVT NVT,
                                    SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  DebugLoc dl = Node->getDebugLoc();
  MVT VT = Node->getValueType(0);
  MVT SrcVT = Node->getOperand(1).getValueType();
  assert((SrcVT == MVT::f32 || SrcVT == MVT::f64) &&
         "fcopysign expansion only supported for f32 and f64");
  MVT SrcNVT = (SrcVT == MVT::f64) ? MVT::i64 : MVT::i32;

  // First get the sign bit of second operand.
  SDValue Mask1 = (SrcVT == MVT::f64)
    ? DAG.getConstantFP(BitsToDouble(1ULL << 63), SrcVT)
    : DAG.getConstantFP(BitsToFloat(1U << 31), SrcVT);
  Mask1 = DAG.getNode(ISD::BIT_CONVERT, dl, SrcNVT, Mask1);
  SDValue SignBit= DAG.getNode(ISD::BIT_CONVERT, dl, SrcNVT,  
                               Node->getOperand(1));
  SignBit = DAG.getNode(ISD::AND, dl, SrcNVT, SignBit, Mask1);
  // Shift right or sign-extend it if the two operands have different types.
  int SizeDiff = SrcNVT.getSizeInBits() - NVT.getSizeInBits();
  if (SizeDiff > 0) {
    SignBit = DAG.getNode(ISD::SRL, dl, SrcNVT, SignBit,
                          DAG.getConstant(SizeDiff, TLI.getShiftAmountTy()));
    SignBit = DAG.getNode(ISD::TRUNCATE, dl, NVT, SignBit);
  } else if (SizeDiff < 0) {
    SignBit = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, SignBit);
    SignBit = DAG.getNode(ISD::SHL, dl, NVT, SignBit,
                          DAG.getConstant(-SizeDiff, TLI.getShiftAmountTy()));
  }

  // Clear the sign bit of first operand.
  SDValue Mask2 = (VT == MVT::f64)
    ? DAG.getConstantFP(BitsToDouble(~(1ULL << 63)), VT)
    : DAG.getConstantFP(BitsToFloat(~(1U << 31)), VT);
  Mask2 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Mask2);
  SDValue Result = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(0));
  Result = DAG.getNode(ISD::AND, dl, NVT, Result, Mask2);

  // Or the value with the sign bit.
  Result = DAG.getNode(ISD::OR, dl, NVT, Result, SignBit);
  return Result;
}

/// ExpandUnalignedStore - Expands an unaligned store to 2 half-size stores.
static
SDValue ExpandUnalignedStore(StoreSDNode *ST, SelectionDAG &DAG,
                             const TargetLowering &TLI) {
  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();
  SDValue Val = ST->getValue();
  MVT VT = Val.getValueType();
  int Alignment = ST->getAlignment();
  int SVOffset = ST->getSrcValueOffset();
  DebugLoc dl = ST->getDebugLoc();
  if (ST->getMemoryVT().isFloatingPoint() ||
      ST->getMemoryVT().isVector()) {
    MVT intVT = MVT::getIntegerVT(VT.getSizeInBits());
    if (TLI.isTypeLegal(intVT)) {
      // Expand to a bitconvert of the value to the integer type of the
      // same size, then a (misaligned) int store.
      // FIXME: Does not handle truncating floating point stores!
      SDValue Result = DAG.getNode(ISD::BIT_CONVERT, dl, intVT, Val);
      return DAG.getStore(Chain, dl, Result, Ptr, ST->getSrcValue(),
                          SVOffset, ST->isVolatile(), Alignment);
    } else {
      // Do a (aligned) store to a stack slot, then copy from the stack slot
      // to the final destination using (unaligned) integer loads and stores.
      MVT StoredVT = ST->getMemoryVT();
      MVT RegVT =
        TLI.getRegisterType(MVT::getIntegerVT(StoredVT.getSizeInBits()));
      unsigned StoredBytes = StoredVT.getSizeInBits() / 8;
      unsigned RegBytes = RegVT.getSizeInBits() / 8;
      unsigned NumRegs = (StoredBytes + RegBytes - 1) / RegBytes;

      // Make sure the stack slot is also aligned for the register type.
      SDValue StackPtr = DAG.CreateStackTemporary(StoredVT, RegVT);

      // Perform the original store, only redirected to the stack slot.
      SDValue Store = DAG.getTruncStore(Chain, dl, 
                                        Val, StackPtr, NULL, 0,StoredVT);
      SDValue Increment = DAG.getConstant(RegBytes, TLI.getPointerTy());
      SmallVector<SDValue, 8> Stores;
      unsigned Offset = 0;

      // Do all but one copies using the full register width.
      for (unsigned i = 1; i < NumRegs; i++) {
        // Load one integer register's worth from the stack slot.
        SDValue Load = DAG.getLoad(RegVT, dl, Store, StackPtr, NULL, 0);
        // Store it to the final location.  Remember the store.
        Stores.push_back(DAG.getStore(Load.getValue(1), dl, Load, Ptr,
                                      ST->getSrcValue(), SVOffset + Offset,
                                      ST->isVolatile(),
                                      MinAlign(ST->getAlignment(), Offset)));
        // Increment the pointers.
        Offset += RegBytes;
        StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                               Increment);
        Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
      }

      // The last store may be partial.  Do a truncating store.  On big-endian
      // machines this requires an extending load from the stack slot to ensure
      // that the bits are in the right place.
      MVT MemVT = MVT::getIntegerVT(8 * (StoredBytes - Offset));

      // Load from the stack slot.
      SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, RegVT, Store, StackPtr,
                                    NULL, 0, MemVT);

      Stores.push_back(DAG.getTruncStore(Load.getValue(1), dl, Load, Ptr,
                                         ST->getSrcValue(), SVOffset + Offset,
                                         MemVT, ST->isVolatile(),
                                         MinAlign(ST->getAlignment(), Offset)));
      // The order of the stores doesn't matter - say it with a TokenFactor.
      return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Stores[0],
                         Stores.size());
    }
  }
  assert(ST->getMemoryVT().isInteger() &&
         !ST->getMemoryVT().isVector() &&
         "Unaligned store of unknown type.");
  // Get the half-size VT
  MVT NewStoredVT =
    (MVT::SimpleValueType)(ST->getMemoryVT().getSimpleVT() - 1);
  int NumBits = NewStoredVT.getSizeInBits();
  int IncrementSize = NumBits / 8;

  // Divide the stored value in two parts.
  SDValue ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDValue Lo = Val;
  SDValue Hi = DAG.getNode(ISD::SRL, dl, VT, Val, ShiftAmount);

  // Store the two parts
  SDValue Store1, Store2;
  Store1 = DAG.getTruncStore(Chain, dl, TLI.isLittleEndian()?Lo:Hi, Ptr,
                             ST->getSrcValue(), SVOffset, NewStoredVT,
                             ST->isVolatile(), Alignment);
  Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                    DAG.getConstant(IncrementSize, TLI.getPointerTy()));
  Alignment = MinAlign(Alignment, IncrementSize);
  Store2 = DAG.getTruncStore(Chain, dl, TLI.isLittleEndian()?Hi:Lo, Ptr,
                             ST->getSrcValue(), SVOffset + IncrementSize,
                             NewStoredVT, ST->isVolatile(), Alignment);

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2);
}

/// ExpandUnalignedLoad - Expands an unaligned load to 2 half-size loads.
static
SDValue ExpandUnalignedLoad(LoadSDNode *LD, SelectionDAG &DAG,
                            const TargetLowering &TLI) {
  int SVOffset = LD->getSrcValueOffset();
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  MVT VT = LD->getValueType(0);
  MVT LoadedVT = LD->getMemoryVT();
  DebugLoc dl = LD->getDebugLoc();
  if (VT.isFloatingPoint() || VT.isVector()) {
    MVT intVT = MVT::getIntegerVT(LoadedVT.getSizeInBits());
    if (TLI.isTypeLegal(intVT)) {
      // Expand to a (misaligned) integer load of the same size,
      // then bitconvert to floating point or vector.
      SDValue newLoad = DAG.getLoad(intVT, dl, Chain, Ptr, LD->getSrcValue(),
                                    SVOffset, LD->isVolatile(),
                                    LD->getAlignment());
      SDValue Result = DAG.getNode(ISD::BIT_CONVERT, dl, LoadedVT, newLoad);
      if (VT.isFloatingPoint() && LoadedVT != VT)
        Result = DAG.getNode(ISD::FP_EXTEND, dl, VT, Result);

      SDValue Ops[] = { Result, Chain };
      return DAG.getMergeValues(Ops, 2, dl);
    } else {
      // Copy the value to a (aligned) stack slot using (unaligned) integer
      // loads and stores, then do a (aligned) load from the stack slot.
      MVT RegVT = TLI.getRegisterType(intVT);
      unsigned LoadedBytes = LoadedVT.getSizeInBits() / 8;
      unsigned RegBytes = RegVT.getSizeInBits() / 8;
      unsigned NumRegs = (LoadedBytes + RegBytes - 1) / RegBytes;

      // Make sure the stack slot is also aligned for the register type.
      SDValue StackBase = DAG.CreateStackTemporary(LoadedVT, RegVT);

      SDValue Increment = DAG.getConstant(RegBytes, TLI.getPointerTy());
      SmallVector<SDValue, 8> Stores;
      SDValue StackPtr = StackBase;
      unsigned Offset = 0;

      // Do all but one copies using the full register width.
      for (unsigned i = 1; i < NumRegs; i++) {
        // Load one integer register's worth from the original location.
        SDValue Load = DAG.getLoad(RegVT, dl, Chain, Ptr, LD->getSrcValue(),
                                   SVOffset + Offset, LD->isVolatile(),
                                   MinAlign(LD->getAlignment(), Offset));
        // Follow the load with a store to the stack slot.  Remember the store.
        Stores.push_back(DAG.getStore(Load.getValue(1), dl, Load, StackPtr,
                                      NULL, 0));
        // Increment the pointers.
        Offset += RegBytes;
        Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
        StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                               Increment);
      }

      // The last copy may be partial.  Do an extending load.
      MVT MemVT = MVT::getIntegerVT(8 * (LoadedBytes - Offset));
      SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, RegVT, Chain, Ptr,
                                    LD->getSrcValue(), SVOffset + Offset,
                                    MemVT, LD->isVolatile(),
                                    MinAlign(LD->getAlignment(), Offset));
      // Follow the load with a store to the stack slot.  Remember the store.
      // On big-endian machines this requires a truncating store to ensure
      // that the bits end up in the right place.
      Stores.push_back(DAG.getTruncStore(Load.getValue(1), dl, Load, StackPtr,
                                         NULL, 0, MemVT));

      // The order of the stores doesn't matter - say it with a TokenFactor.
      SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Stores[0],
                               Stores.size());

      // Finally, perform the original load only redirected to the stack slot.
      Load = DAG.getExtLoad(LD->getExtensionType(), dl, VT, TF, StackBase,
                            NULL, 0, LoadedVT);

      // Callers expect a MERGE_VALUES node.
      SDValue Ops[] = { Load, TF };
      return DAG.getMergeValues(Ops, 2, dl);
    }
  }
  assert(LoadedVT.isInteger() && !LoadedVT.isVector() &&
         "Unaligned load of unsupported type.");

  // Compute the new VT that is half the size of the old one.  This is an
  // integer MVT.
  unsigned NumBits = LoadedVT.getSizeInBits();
  MVT NewLoadedVT;
  NewLoadedVT = MVT::getIntegerVT(NumBits/2);
  NumBits >>= 1;
  
  unsigned Alignment = LD->getAlignment();
  unsigned IncrementSize = NumBits / 8;
  ISD::LoadExtType HiExtType = LD->getExtensionType();

  // If the original load is NON_EXTLOAD, the hi part load must be ZEXTLOAD.
  if (HiExtType == ISD::NON_EXTLOAD)
    HiExtType = ISD::ZEXTLOAD;

  // Load the value in two parts
  SDValue Lo, Hi;
  if (TLI.isLittleEndian()) {
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset, NewLoadedVT, LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Hi = DAG.getExtLoad(HiExtType, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  } else {
    Hi = DAG.getExtLoad(HiExtType, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset, NewLoadedVT,LD->isVolatile(), Alignment);
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getConstant(IncrementSize, TLI.getPointerTy()));
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, VT, Chain, Ptr, LD->getSrcValue(),
                        SVOffset + IncrementSize, NewLoadedVT, LD->isVolatile(),
                        MinAlign(Alignment, IncrementSize));
  }

  // aggregate the two parts
  SDValue ShiftAmount = DAG.getConstant(NumBits, TLI.getShiftAmountTy());
  SDValue Result = DAG.getNode(ISD::SHL, dl, VT, Hi, ShiftAmount);
  Result = DAG.getNode(ISD::OR, dl, VT, Result, Lo);

  SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                             Hi.getValue(1));

  SDValue Ops[] = { Result, TF };
  return DAG.getMergeValues(Ops, 2, dl);
}

/// UnrollVectorOp - We know that the given vector has a legal type, however
/// the operation it performs is not legal and is an operation that we have
/// no way of lowering.  "Unroll" the vector, splitting out the scalars and
/// operating on each element individually.
SDValue SelectionDAGLegalize::UnrollVectorOp(SDValue Op) {
  MVT VT = Op.getValueType();
  assert(isTypeLegal(VT) &&
         "Caller should expand or promote operands that are not legal!");
  assert(Op.getNode()->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");
  unsigned NE = VT.getVectorNumElements();
  MVT EltVT = VT.getVectorElementType();
  DebugLoc dl = Op.getNode()->getDebugLoc();

  SmallVector<SDValue, 8> Scalars;
  SmallVector<SDValue, 4> Operands(Op.getNumOperands());
  for (unsigned i = 0; i != NE; ++i) {
    for (unsigned j = 0; j != Op.getNumOperands(); ++j) {
      SDValue Operand = Op.getOperand(j);
      MVT OperandVT = Operand.getValueType();
      if (OperandVT.isVector()) {
        // A vector operand; extract a single element.
        MVT OperandEltVT = OperandVT.getVectorElementType();
        Operands[j] = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                                  OperandEltVT,
                                  Operand,
                                  DAG.getConstant(i, MVT::i32));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }

    switch (Op.getOpcode()) {
    default:
      Scalars.push_back(DAG.getNode(Op.getOpcode(), dl, EltVT,
                                    &Operands[0], Operands.size()));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
    case ISD::ROTL:
    case ISD::ROTR:
      Scalars.push_back(DAG.getNode(Op.getOpcode(), dl, EltVT, Operands[0],
                                    DAG.getShiftAmountOperand(Operands[1])));
      break;
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Scalars[0], Scalars.size());
}

/// GetFPLibCall - Return the right libcall for the given floating point type.
static RTLIB::Libcall GetFPLibCall(MVT VT,
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
SDValue SelectionDAGLegalize::
PerformInsertVectorEltInMemory(SDValue Vec, SDValue Val, SDValue Idx,
                               DebugLoc dl) {
  SDValue Tmp1 = Vec;
  SDValue Tmp2 = Val;
  SDValue Tmp3 = Idx;
  
  // If the target doesn't support this, we have to spill the input vector
  // to a temporary stack slot, update the element, then reload it.  This is
  // badness.  We could also load the value into a vector register (either
  // with a "move to register" or "extload into register" instruction, then
  // permute it into place, if the idx is a constant and if the idx is
  // supported by the target.
  MVT VT    = Tmp1.getValueType();
  MVT EltVT = VT.getVectorElementType();
  MVT IdxVT = Tmp3.getValueType();
  MVT PtrVT = TLI.getPointerTy();
  SDValue StackPtr = DAG.CreateStackTemporary(VT);

  int SPFI = cast<FrameIndexSDNode>(StackPtr.getNode())->getIndex();

  // Store the vector.
  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, Tmp1, StackPtr,
                            PseudoSourceValue::getFixedStack(SPFI), 0);

  // Truncate or zero extend offset to target pointer type.
  unsigned CastOpc = IdxVT.bitsGT(PtrVT) ? ISD::TRUNCATE : ISD::ZERO_EXTEND;
  Tmp3 = DAG.getNode(CastOpc, dl, PtrVT, Tmp3);
  // Add the offset to the index.
  unsigned EltSize = EltVT.getSizeInBits()/8;
  Tmp3 = DAG.getNode(ISD::MUL, dl, IdxVT, Tmp3,DAG.getConstant(EltSize, IdxVT));
  SDValue StackPtr2 = DAG.getNode(ISD::ADD, dl, IdxVT, Tmp3, StackPtr);
  // Store the scalar value.
  Ch = DAG.getTruncStore(Ch, dl, Tmp2, StackPtr2,
                         PseudoSourceValue::getFixedStack(SPFI), 0, EltVT);
  // Load the updated vector.
  return DAG.getLoad(VT, dl, Ch, StackPtr,
                     PseudoSourceValue::getFixedStack(SPFI), 0);
}


/// LegalizeOp - We know that the specified value has a legal type, and
/// that its operands are legal.  Now ensure that the operation itself
/// is legal, recursively ensuring that the operands' operations remain
/// legal.
SDValue SelectionDAGLegalize::LegalizeOp(SDValue Op) {
  if (Op.getOpcode() == ISD::TargetConstant) // Allow illegal target nodes.
    return Op;
  
  assert(isTypeLegal(Op.getValueType()) &&
         "Caller should expand or promote operands that are not legal!");
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();

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
  DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
  if (I != LegalizedNodes.end()) return I->second;

  SDValue Tmp1, Tmp2, Tmp3, Tmp4;
  SDValue Result = Op;
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
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
        Ops.push_back(LegalizeOp(Node->getOperand(i)));

      Result = DAG.UpdateNodeOperands(Result.getValue(0), &Ops[0], Ops.size());

      for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
        AddLegalizedOperand(Op.getValue(i), Result.getValue(i));
      return Result.getValue(Op.getResNo());
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
      if (Tmp1.getNode()) Result = Tmp1;
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
    if (Tmp1.getNode()) 
      Result = Tmp1;
    else
      Result = DAG.getConstant(0, TLI.getPointerTy());
    break;
  case ISD::FRAME_TO_ARGS_OFFSET: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Legal:
      Result = DAG.getConstant(0, VT);
      break;
    }
    }
    break;
  case ISD::EXCEPTIONADDR: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
        unsigned Reg = TLI.getExceptionAddressRegister();
        Result = DAG.getCopyFromReg(Tmp1, Reg, VT);
      }
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Legal: {
      SDValue Ops[] = { DAG.getConstant(0, VT), Tmp1 };
      Result = DAG.getMergeValues(Ops, 2, dl);
      break;
    }
    }
    }
    if (Result.getNode()->getNumValues() == 1) break;

    assert(Result.getNode()->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since we produced two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result);
    Tmp2 = LegalizeOp(Result.getValue(1));
    AddLegalizedOperand(Op.getValue(0), Tmp1);
    AddLegalizedOperand(Op.getValue(1), Tmp2);
    return Op.getResNo() ? Tmp2 : Tmp1;
  case ISD::EHSELECTION: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
        unsigned Reg = TLI.getExceptionSelectorRegister();
        Result = DAG.getCopyFromReg(Tmp2, Reg, VT);
      }
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Legal: {
      SDValue Ops[] = { DAG.getConstant(0, VT), Tmp2 };
      Result = DAG.getMergeValues(Ops, 2, dl);
      break;
    }
    }
    }
    if (Result.getNode()->getNumValues() == 1) break;

    assert(Result.getNode()->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since we produced two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result);
    Tmp2 = LegalizeOp(Result.getValue(1));
    AddLegalizedOperand(Op.getValue(0), Tmp1);
    AddLegalizedOperand(Op.getValue(1), Tmp2);
    return Op.getResNo() ? Tmp2 : Tmp1;
  case ISD::EH_RETURN: {
    MVT VT = Node->getValueType(0);
    // The only "good" option for this node is to custom lower it.
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported at all!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
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
    Result = Node->getOperand(Op.getResNo());
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
    return Result.getValue(Op.getResNo());
  case ISD::UNDEF: {
    MVT VT = Op.getValueType();
    switch (TLI.getOperationAction(ISD::UNDEF, VT)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand:
      if (VT.isInteger())
        Result = DAG.getConstant(0, VT);
      else if (VT.isFloatingPoint())
        Result = DAG.getConstantFP(APFloat(APInt(VT.getSizeInBits(), 0)),
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
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
    Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
    
    // Allow the target to custom lower its intrinsics if it wants to.
    if (TLI.getOperationAction(Node->getOpcode(), MVT::Other) == 
        TargetLowering::Custom) {
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) Result = Tmp3;
    }

    if (Result.getNode()->getNumValues() == 1) break;

    // Must have return value and chain result.
    assert(Result.getNode()->getNumValues() == 2 &&
           "Cannot return more than two values!");

    // Since loads produce two values, make sure to remember that we 
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  }    

  case ISD::DBG_STOPPOINT:
    assert(Node->getNumOperands() == 1 && "Invalid DBG_STOPPOINT node!");
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the input chain.
    
    switch (TLI.getOperationAction(ISD::DBG_STOPPOINT, MVT::Other)) {
    case TargetLowering::Promote:
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Expand: {
      DwarfWriter *DW = DAG.getDwarfWriter();
      bool useDEBUG_LOC = TLI.isOperationLegalOrCustom(ISD::DEBUG_LOC,
                                                       MVT::Other);
      bool useLABEL = TLI.isOperationLegalOrCustom(ISD::DBG_LABEL, MVT::Other);
      
      const DbgStopPointSDNode *DSP = cast<DbgStopPointSDNode>(Node);
      GlobalVariable *CU_GV = cast<GlobalVariable>(DSP->getCompileUnit());
      if (DW && (useDEBUG_LOC || useLABEL) && !CU_GV->isDeclaration()) {
        DICompileUnit CU(cast<GlobalVariable>(DSP->getCompileUnit()));
        unsigned SrcFile = DW->RecordSource(CU.getDirectory(),
                                            CU.getFilename());
        
        unsigned Line = DSP->getLine();
        unsigned Col = DSP->getColumn();
        
        // A bit self-referential to have DebugLoc on Debug_Loc nodes, but
        // it won't hurt anything.
        if (useDEBUG_LOC) {
          SDValue Ops[] = { Tmp1, DAG.getConstant(Line, MVT::i32),
                              DAG.getConstant(Col, MVT::i32),
                              DAG.getConstant(SrcFile, MVT::i32) };
          Result = DAG.getNode(ISD::DEBUG_LOC, dl, MVT::Other, Ops, 4);
        } else {
          unsigned ID = DW->RecordSourceLine(Line, Col, SrcFile);
          Result = DAG.getLabel(ISD::DBG_LABEL, dl, Tmp1, ID);
        }
      } else {
        Result = Tmp1;  // chain
      }
      break;
    }
    case TargetLowering::Legal: {
      LegalizeAction Action = getTypeAction(Node->getOperand(1).getValueType());
      if (Action == Legal && Tmp1 == Node->getOperand(0))
        break;

      SmallVector<SDValue, 8> Ops;
      Ops.push_back(Tmp1);
      if (Action == Legal) {
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
      break;
    }
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
    case TargetLowering::Legal: {
      LegalizeAction Action = getTypeAction(Node->getOperand(1).getValueType());
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      if (Action == Legal && Tmp1 == Node->getOperand(0))
        break;
      if (Action == Legal) {
        Tmp2 = Node->getOperand(1);
        Tmp3 = Node->getOperand(2);
        Tmp4 = Node->getOperand(3);
      } else {
        Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the line #.
        Tmp3 = LegalizeOp(Node->getOperand(2));  // Legalize the col #.
        Tmp4 = LegalizeOp(Node->getOperand(3));  // Legalize the source file id.
      }
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3, Tmp4);
      break;
    }
    }
    break;    

  case ISD::DBG_LABEL:
  case ISD::EH_LABEL:
    assert(Node->getNumOperands() == 1 && "Invalid LABEL node!");
    switch (TLI.getOperationAction(Node->getOpcode(), MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
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
      SDValue Ops[6];
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

  case ISD::ATOMIC_CMP_SWAP: {
    unsigned int num_operands = 4;
    assert(Node->getNumOperands() == num_operands && "Invalid Atomic node!");
    SDValue Ops[4];
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
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  }
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP: {
    unsigned int num_operands = 3;
    assert(Node->getNumOperands() == num_operands && "Invalid Atomic node!");
    SDValue Ops[3];
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
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  }
  case ISD::Constant: {
    ConstantSDNode *CN = cast<ConstantSDNode>(Node);
    unsigned opAction =
      TLI.getOperationAction(ISD::Constant, CN->getValueType(0));

    // We know we don't need to expand constants here, constants only have one
    // value and we check that it is fine above.

    if (opAction == TargetLowering::Custom) {
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode())
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
      if (Tmp3.getNode()) {
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
      SmallVector<SDValue, 8> Ops;
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
    assert(Tmp3.getNode() && "Target didn't custom lower this node!");
    // A call within a calling sequence must be legalized to something
    // other than the normal CALLSEQ_END.  Violating this gets Legalize
    // into an infinite loop.
    assert ((!IsLegalizingCall ||
             Node->getOpcode() != ISD::CALL ||
             Tmp3.getNode()->getOpcode() != ISD::CALLSEQ_END) &&
            "Nested CALLSEQ_START..CALLSEQ_END not supported.");

    // The number of incoming and outgoing values should match; unless the final
    // outgoing value is a flag.
    assert((Tmp3.getNode()->getNumValues() == Result.getNode()->getNumValues() ||
            (Tmp3.getNode()->getNumValues() == Result.getNode()->getNumValues() + 1 &&
             Tmp3.getNode()->getValueType(Tmp3.getNode()->getNumValues() - 1) ==
               MVT::Flag)) &&
           "Lowering call/formal_arguments produced unexpected # results!");
    
    // Since CALL/FORMAL_ARGUMENTS nodes produce multiple values, make sure to
    // remember that we legalized all of them, so it doesn't get relegalized.
    for (unsigned i = 0, e = Tmp3.getNode()->getNumValues(); i != e; ++i) {
      if (Tmp3.getNode()->getValueType(i) == MVT::Flag)
        continue;
      Tmp1 = LegalizeOp(Tmp3.getValue(i));
      if (Op.getResNo() == i)
        Tmp2 = Tmp1;
      AddLegalizedOperand(SDValue(Node, i), Tmp1);
    }
    return Tmp2;
   case ISD::EXTRACT_SUBREG: {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      ConstantSDNode *idx = dyn_cast<ConstantSDNode>(Node->getOperand(1));
      assert(idx && "Operand must be a constant");
      Tmp2 = DAG.getTargetConstant(idx->getAPIntValue(), idx->getValueType(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    }
    break;
  case ISD::INSERT_SUBREG: {
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Tmp2 = LegalizeOp(Node->getOperand(1));      
      ConstantSDNode *idx = dyn_cast<ConstantSDNode>(Node->getOperand(2));
      assert(idx && "Operand must be a constant");
      Tmp3 = DAG.getTargetConstant(idx->getAPIntValue(), idx->getValueType(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    }
    break;      
  case ISD::BUILD_VECTOR:
    switch (TLI.getOperationAction(ISD::BUILD_VECTOR, Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Result, DAG);
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand:
      Result = ExpandBUILD_VECTOR(Result.getNode());
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
    case Expand:
      // FIXME: An alternative would be to check to see if the target is not
      // going to custom lower this operation, we could bitcast to half elt 
      // width and perform two inserts at that width, if that is legal.
      Tmp2 = Node->getOperand(1);
      break;
    }
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    
    switch (TLI.getOperationAction(ISD::INSERT_VECTOR_ELT,
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp4 = TLI.LowerOperation(Result, DAG);
      if (Tmp4.getNode()) {
        Result = Tmp4;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Promote:
      // Fall thru for vector case
    case TargetLowering::Expand: {
      // If the insert index is a constant, codegen this as a scalar_to_vector,
      // then a shuffle that inserts it into the right position in the vector.
      if (ConstantSDNode *InsertPos = dyn_cast<ConstantSDNode>(Tmp3)) {
        // SCALAR_TO_VECTOR requires that the type of the value being inserted
        // match the element type of the vector being created.
        if (Tmp2.getValueType() == 
            Op.getValueType().getVectorElementType()) {
          SDValue ScVec = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                        Tmp1.getValueType(), Tmp2);
          
          unsigned NumElts = Tmp1.getValueType().getVectorNumElements();
          MVT ShufMaskVT =
            MVT::getIntVectorWithNumElements(NumElts);
          MVT ShufMaskEltVT = ShufMaskVT.getVectorElementType();
          
          // We generate a shuffle of InVec and ScVec, so the shuffle mask
          // should be 0,1,2,3,4,5... with the appropriate element replaced with
          // elt 0 of the RHS.
          SmallVector<SDValue, 8> ShufOps;
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertPos->getZExtValue())
              ShufOps.push_back(DAG.getConstant(i, ShufMaskEltVT));
            else
              ShufOps.push_back(DAG.getConstant(NumElts, ShufMaskEltVT));
          }
          SDValue ShufMask = DAG.getNode(ISD::BUILD_VECTOR, dl, ShufMaskVT,
                                           &ShufOps[0], ShufOps.size());
          
          Result = DAG.getNode(ISD::VECTOR_SHUFFLE, dl, Tmp1.getValueType(),
                               Tmp1, ScVec, ShufMask);
          Result = LegalizeOp(Result);
          break;
        }
      }
      Result = PerformInsertVectorEltInMemory(Tmp1, Tmp2, Tmp3, dl);
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
      if (Tmp3.getNode()) {
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
      if (Tmp3.getNode()) {
        Result = Tmp3;
        break;
      }
      // FALLTHROUGH
    case TargetLowering::Expand: {
      MVT VT = Node->getValueType(0);
      MVT EltVT = VT.getVectorElementType();
      MVT PtrVT = TLI.getPointerTy();
      SDValue Mask = Node->getOperand(2);
      unsigned NumElems = Mask.getNumOperands();
      SmallVector<SDValue,8> Ops;
      for (unsigned i = 0; i != NumElems; ++i) {
        SDValue Arg = Mask.getOperand(i);
        if (Arg.getOpcode() == ISD::UNDEF) {
          Ops.push_back(DAG.getNode(ISD::UNDEF, dl, EltVT));
        } else {
          assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
          unsigned Idx = cast<ConstantSDNode>(Arg)->getZExtValue();
          if (Idx < NumElems)
            Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Tmp1,
                                      DAG.getConstant(Idx, PtrVT)));
          else
            Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Tmp2,
                                      DAG.getConstant(Idx - NumElems, PtrVT)));
        }
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], Ops.size());
      break;
    }
    case TargetLowering::Promote: {
      // Change base type to a different vector type.
      MVT OVT = Node->getValueType(0);
      MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Cast the two input vectors.
      Tmp1 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp1);
      Tmp2 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp2);
      
      // Convert the shuffle mask to the right # elements.
      Tmp3 = SDValue(isShuffleLegal(OVT, Node->getOperand(2)), 0);
      assert(Tmp3.getNode() && "Shuffle not legal?");
      Result = DAG.getNode(ISD::VECTOR_SHUFFLE, dl, NVT, Tmp1, Tmp2, Tmp3);
      Result = DAG.getNode(ISD::BIT_CONVERT, dl, OVT, Result);
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
    
  case ISD::CONCAT_VECTORS: {
    // Use extract/insert/build vector for now. We might try to be
    // more clever later.
    MVT PtrVT = TLI.getPointerTy();
    SmallVector<SDValue, 8> Ops;
    unsigned NumOperands = Node->getNumOperands();
    for (unsigned i=0; i < NumOperands; ++i) {
      SDValue SubOp = Node->getOperand(i);
      MVT VVT = SubOp.getNode()->getValueType(0);
      MVT EltVT = VVT.getVectorElementType();
      unsigned NumSubElem = VVT.getVectorNumElements();
      for (unsigned j=0; j < NumSubElem; ++j) {
        Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, SubOp,
                                  DAG.getConstant(j, PtrVT)));
      }
    }
    return LegalizeOp(DAG.getNode(ISD::BUILD_VECTOR, dl, Node->getValueType(0),
                      &Ops[0], Ops.size()));
  }

  case ISD::CALLSEQ_START: {
    SDNode *CallEnd = FindCallEndFromCallStart(Node);
    
    // Recursively Legalize all of the inputs of the call end that do not lead
    // to this call start.  This ensures that any libcalls that need be inserted
    // are inserted *before* the CALLSEQ_START.
    {SmallPtrSet<SDNode*, 32> NodesLeadingTo;
    for (unsigned i = 0, e = CallEnd->getNumOperands(); i != e; ++i)
      LegalizeAllNodesNotLeadingTo(CallEnd->getOperand(i).getNode(), Node,
                                   NodesLeadingTo);
    }

    // Now that we legalized all of the inputs (which may have inserted
    // libcalls) create the new CALLSEQ_START node.
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    // Merge in the last call, to ensure that this call start after the last
    // call ended.
    if (LastCALLSEQ_END.getOpcode() != ISD::EntryToken) {
      Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                         Tmp1, LastCALLSEQ_END);
      Tmp1 = LegalizeOp(Tmp1);
    }
      
    // Do not try to legalize the target-specific arguments (#1+).
    if (Tmp1 != Node->getOperand(0)) {
      SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
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
    // Note that we are selecting this call!
    LastCALLSEQ_END = SDValue(CallEnd, 0);
    IsLegalizingCall = true;
    
    // Legalize the call, starting from the CALLSEQ_END.
    LegalizeOp(LastCALLSEQ_END);
    assert(!IsLegalizingCall && "CALLSEQ_END should have cleared this!");
    return Result;
  }
  case ISD::CALLSEQ_END:
    // If the CALLSEQ_START node hasn't been legalized first, legalize it.  This
    // will cause this node to be legalized as well as handling libcalls right.
    if (LastCALLSEQ_END.getNode() != Node) {
      LegalizeOp(SDValue(FindCallStartFromCallEnd(Node), 0));
      DenseMap<SDValue, SDValue>::iterator I = LegalizedNodes.find(Op);
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
        SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    } else {
      Tmp2 = LegalizeOp(Node->getOperand(Node->getNumOperands()-1));
      if (Tmp1 != Node->getOperand(0) ||
          Tmp2 != Node->getOperand(Node->getNumOperands()-1)) {
        SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
        Ops[0] = Tmp1;
        Ops.back() = Tmp2;
        Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());
      }
    }
    assert(IsLegalizingCall && "Call sequence imbalance between start/end?");
    // This finishes up call legalization.
    IsLegalizingCall = false;
    
    // If the CALLSEQ_END node has a flag, remember that we legalized it.
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    if (Node->getNumValues() == 2)
      AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  case ISD::DYNAMIC_STACKALLOC: {
    MVT VT = Node->getValueType(0);
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
      SDValue Chain = Tmp1.getOperand(0);

      // Chain the dynamic stack allocation so that it doesn't modify the stack
      // pointer when other instructions are using the stack.
      Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(0, true));

      SDValue Size  = Tmp2.getOperand(1);
      SDValue SP = DAG.getCopyFromReg(Chain, SPReg, VT);
      Chain = SP.getValue(1);
      unsigned Align = cast<ConstantSDNode>(Tmp3)->getZExtValue();
      unsigned StackAlign =
        TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
      if (Align > StackAlign)
        SP = DAG.getNode(ISD::AND, dl, VT, SP,
                         DAG.getConstant(-(uint64_t)Align, VT));
      Tmp1 = DAG.getNode(ISD::SUB, dl, VT, SP, Size);       // Value
      Chain = DAG.getCopyToReg(Chain, SPReg, Tmp1);     // Output chain

      Tmp2 = DAG.getCALLSEQ_END(Chain,  DAG.getIntPtrConstant(0, true),
                                DAG.getIntPtrConstant(0, true), SDValue());

      Tmp1 = LegalizeOp(Tmp1);
      Tmp2 = LegalizeOp(Tmp2);
      break;
    }
    case TargetLowering::Custom:
      Tmp3 = TLI.LowerOperation(Tmp1, DAG);
      if (Tmp3.getNode()) {
        Tmp1 = LegalizeOp(Tmp3);
        Tmp2 = LegalizeOp(Tmp3.getValue(1));
      }
      break;
    case TargetLowering::Legal:
      break;
    }
    // Since this op produce two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Tmp1);
    AddLegalizedOperand(SDValue(Node, 1), Tmp2);
    return Op.getResNo() ? Tmp2 : Tmp1;
  }
  case ISD::INLINEASM: {
    SmallVector<SDValue, 8> Ops(Node->op_begin(), Node->op_end());
    bool Changed = false;
    // Legalize all of the operands of the inline asm, in case they are nodes
    // that need to be expanded or something.  Note we skip the asm string and
    // all of the TargetConstant flags.
    SDValue Op = LegalizeOp(Ops[0]);
    Changed = Op != Ops[0];
    Ops[0] = Op;

    bool HasInFlag = Ops.back().getValueType() == MVT::Flag;
    for (unsigned i = 2, e = Ops.size()-HasInFlag; i < e; ) {
      unsigned NumVals = cast<ConstantSDNode>(Ops[i])->getZExtValue() >> 3;
      for (++i; NumVals; ++i, --NumVals) {
        SDValue Op = LegalizeOp(Ops[i]);
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
    AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
    AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
    return Result.getValue(Op.getResNo());
  }
  case ISD::BR:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();
    
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Node->getOperand(1));
    break;
  case ISD::BRIND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a branch.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
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
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    LastCALLSEQ_END = DAG.getEntryNode();

    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the jumptable node.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Node->getOperand(2));

    switch (TLI.getOperationAction(ISD::BR_JT, MVT::Other)) {  
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Expand: {
      SDValue Chain = Result.getOperand(0);
      SDValue Table = Result.getOperand(1);
      SDValue Index = Result.getOperand(2);

      MVT PTy = TLI.getPointerTy();
      MachineFunction &MF = DAG.getMachineFunction();
      unsigned EntrySize = MF.getJumpTableInfo()->getEntrySize();
      Index= DAG.getNode(ISD::MUL, dl, PTy, 
                         Index, DAG.getConstant(EntrySize, PTy));
      SDValue Addr = DAG.getNode(ISD::ADD, dl, PTy, Index, Table);

      MVT MemVT = MVT::getIntegerVT(EntrySize * 8);
      SDValue LD = DAG.getExtLoad(ISD::SEXTLOAD, dl, PTy, Chain, Addr,
                                  PseudoSourceValue::getJumpTable(), 0, MemVT);
      Addr = LD;
      if (TLI.getTargetMachine().getRelocationModel() == Reloc::PIC_) {
        // For PIC, the sequence is:
        // BRIND(load(Jumptable + index) + RelocBase)
        // RelocBase can be JumpTable, GOT or some sort of global base.
        Addr = DAG.getNode(ISD::ADD, dl, PTy, Addr,
                           TLI.getPICJumpTableRelocBase(Table, DAG));
      }
      Result = DAG.getNode(ISD::BRIND, dl, MVT::Other, LD.getValue(1), Addr);
    }
    }
    break;
  case ISD::BRCOND:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    // Ensure that libcalls are emitted before a return.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
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
        Tmp2 = DAG.getZeroExtendInReg(Tmp2, dl, MVT::i1);
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
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Expand:
      // Expand brcond's setcc into its constituent parts and create a BR_CC
      // Node.
      if (Tmp2.getOpcode() == ISD::SETCC) {
        Result = DAG.getNode(ISD::BR_CC, dl, MVT::Other, 
                             Tmp1, Tmp2.getOperand(2),
                             Tmp2.getOperand(0), Tmp2.getOperand(1),
                             Node->getOperand(2));
      } else {
        Result = DAG.getNode(ISD::BR_CC, dl, MVT::Other, Tmp1, 
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
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
    Tmp1 = LegalizeOp(Tmp1);
    Tmp2 = Node->getOperand(2);              // LHS 
    Tmp3 = Node->getOperand(3);              // RHS
    Tmp4 = Node->getOperand(1);              // CC

    LegalizeSetCC(TLI.getSetCCResultType(Tmp2.getValueType()), 
                  Tmp2, Tmp3, Tmp4, dl);
    LastCALLSEQ_END = DAG.getEntryNode();

    // If we didn't get both a LHS and RHS back from LegalizeSetCC,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp3.getNode() == 0) {
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
      if (Tmp4.getNode()) Result = Tmp4;
      break;
    }
    break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    Tmp1 = LegalizeOp(LD->getChain());   // Legalize the chain.
    Tmp2 = LegalizeOp(LD->getBasePtr()); // Legalize the base pointer.

    ISD::LoadExtType ExtType = LD->getExtensionType();
    if (ExtType == ISD::NON_EXTLOAD) {
      MVT VT = Node->getValueType(0);
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
            getABITypeAlignment(LD->getMemoryVT().getTypeForMVT());
          if (LD->getAlignment() < ABIAlignment){
            Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.getNode()), DAG,
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
        if (Tmp1.getNode()) {
          Tmp3 = LegalizeOp(Tmp1);
          Tmp4 = LegalizeOp(Tmp1.getValue(1));
        }
        break;
      case TargetLowering::Promote: {
        // Only promote a load of vector type to another.
        assert(VT.isVector() && "Cannot promote this load!");
        // Change base type to a different vector type.
        MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), VT);

        Tmp1 = DAG.getLoad(NVT, dl, Tmp1, Tmp2, LD->getSrcValue(),
                           LD->getSrcValueOffset(),
                           LD->isVolatile(), LD->getAlignment());
        Tmp3 = LegalizeOp(DAG.getNode(ISD::BIT_CONVERT, VT, Tmp1));
        Tmp4 = LegalizeOp(Tmp1.getValue(1));
        break;
      }
      }
      // Since loads produce two values, make sure to remember that we 
      // legalized both of them.
      AddLegalizedOperand(SDValue(Node, 0), Tmp3);
      AddLegalizedOperand(SDValue(Node, 1), Tmp4);
      return Op.getResNo() ? Tmp4 : Tmp3;
    } else {
      MVT SrcVT = LD->getMemoryVT();
      unsigned SrcWidth = SrcVT.getSizeInBits();
      int SVOffset = LD->getSrcValueOffset();
      unsigned Alignment = LD->getAlignment();
      bool isVolatile = LD->isVolatile();

      if (SrcWidth != SrcVT.getStoreSizeInBits() &&
          // Some targets pretend to have an i1 loading operation, and actually
          // load an i8.  This trick is correct for ZEXTLOAD because the top 7
          // bits are guaranteed to be zero; it helps the optimizers understand
          // that these bits are zero.  It is also useful for EXTLOAD, since it
          // tells the optimizers that those bits are undefined.  It would be
          // nice to have an effective generic way of getting these benefits...
          // Until such a way is found, don't insist on promoting i1 here.
          (SrcVT != MVT::i1 ||
           TLI.getLoadExtAction(ExtType, MVT::i1) == TargetLowering::Promote)) {
        // Promote to a byte-sized load if not loading an integral number of
        // bytes.  For example, promote EXTLOAD:i20 -> EXTLOAD:i24.
        unsigned NewWidth = SrcVT.getStoreSizeInBits();
        MVT NVT = MVT::getIntegerVT(NewWidth);
        SDValue Ch;

        // The extra bits are guaranteed to be zero, since we stored them that
        // way.  A zext load from NVT thus automatically gives zext from SrcVT.

        ISD::LoadExtType NewExtType =
          ExtType == ISD::ZEXTLOAD ? ISD::ZEXTLOAD : ISD::EXTLOAD;

        Result = DAG.getExtLoad(NewExtType, dl, Node->getValueType(0),
                                Tmp1, Tmp2, LD->getSrcValue(), SVOffset,
                                NVT, isVolatile, Alignment);

        Ch = Result.getValue(1); // The chain.

        if (ExtType == ISD::SEXTLOAD)
          // Having the top bits zero doesn't help when sign extending.
          Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, 
                               Result.getValueType(),
                               Result, DAG.getValueType(SrcVT));
        else if (ExtType == ISD::ZEXTLOAD || NVT == Result.getValueType())
          // All the top bits are guaranteed to be zero - inform the optimizers.
          Result = DAG.getNode(ISD::AssertZext, dl, 
                               Result.getValueType(), Result,
                               DAG.getValueType(SrcVT));

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else if (SrcWidth & (SrcWidth - 1)) {
        // If not loading a power-of-2 number of bits, expand as two loads.
        assert(SrcVT.isExtended() && !SrcVT.isVector() &&
               "Unsupported extload!");
        unsigned RoundWidth = 1 << Log2_32(SrcWidth);
        assert(RoundWidth < SrcWidth);
        unsigned ExtraWidth = SrcWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Load size not an integral number of bytes!");
        MVT RoundVT = MVT::getIntegerVT(RoundWidth);
        MVT ExtraVT = MVT::getIntegerVT(ExtraWidth);
        SDValue Lo, Hi, Ch;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // EXTLOAD:i24 -> ZEXTLOAD:i16 | (shl EXTLOAD@+2:i8, 16)
          // Load the bottom RoundWidth bits.
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl,
                              Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getExtLoad(ExtType, dl, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, dl, Hi.getValueType(), Hi,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, dl, Node->getValueType(0), Lo, Hi);
        } else {
          // Big endian - avoid unaligned loads.
          // EXTLOAD:i24 -> (shl EXTLOAD:i16, 8) | ZEXTLOAD@+2:i8
          // Load the top RoundWidth bits.
          Hi = DAG.getExtLoad(ExtType, dl, Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset, RoundVT, isVolatile,
                              Alignment);

          // Load the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, 
                              Node->getValueType(0), Tmp1, Tmp2,
                              LD->getSrcValue(), SVOffset + IncrementSize,
                              ExtraVT, isVolatile,
                              MinAlign(Alignment, IncrementSize));

          // Build a factor node to remember that this load is independent of the
          // other one.
          Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                           Hi.getValue(1));

          // Move the top bits to the right place.
          Hi = DAG.getNode(ISD::SHL, dl, Hi.getValueType(), Hi,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));

          // Join the hi and lo parts.
          Result = DAG.getNode(ISD::OR, dl, Node->getValueType(0), Lo, Hi);
        }

        Tmp1 = LegalizeOp(Result);
        Tmp2 = LegalizeOp(Ch);
      } else {
        switch (TLI.getLoadExtAction(ExtType, SrcVT)) {
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
            if (Tmp3.getNode()) {
              Tmp1 = LegalizeOp(Tmp3);
              Tmp2 = LegalizeOp(Tmp3.getValue(1));
            }
          } else {
            // If this is an unaligned load and the target doesn't support it,
            // expand it.
            if (!TLI.allowsUnalignedMemoryAccesses()) {
              unsigned ABIAlignment = TLI.getTargetData()->
                getABITypeAlignment(LD->getMemoryVT().getTypeForMVT());
              if (LD->getAlignment() < ABIAlignment){
                Result = ExpandUnalignedLoad(cast<LoadSDNode>(Result.getNode()), DAG,
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
            SDValue Load = DAG.getLoad(SrcVT, dl, Tmp1, Tmp2, LD->getSrcValue(),
                                         LD->getSrcValueOffset(),
                                         LD->isVolatile(), LD->getAlignment());
            Result = DAG.getNode(ISD::FP_EXTEND, dl, 
                                 Node->getValueType(0), Load);
            Tmp1 = LegalizeOp(Result);  // Relegalize new nodes.
            Tmp2 = LegalizeOp(Load.getValue(1));
            break;
          }
          assert(ExtType != ISD::EXTLOAD &&"EXTLOAD should always be supported!");
          // Turn the unsupported load into an EXTLOAD followed by an explicit
          // zero/sign extend inreg.
          Result = DAG.getExtLoad(ISD::EXTLOAD, dl, Node->getValueType(0),
                                  Tmp1, Tmp2, LD->getSrcValue(),
                                  LD->getSrcValueOffset(), SrcVT,
                                  LD->isVolatile(), LD->getAlignment());
          SDValue ValRes;
          if (ExtType == ISD::SEXTLOAD)
            ValRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl,
                                 Result.getValueType(),
                                 Result, DAG.getValueType(SrcVT));
          else
            ValRes = DAG.getZeroExtendInReg(Result, dl, SrcVT);
          Tmp1 = LegalizeOp(ValRes);  // Relegalize new nodes.
          Tmp2 = LegalizeOp(Result.getValue(1));  // Relegalize new nodes.
          break;
        }
      }

      // Since loads produce two values, make sure to remember that we legalized
      // both of them.
      AddLegalizedOperand(SDValue(Node, 0), Tmp1);
      AddLegalizedOperand(SDValue(Node, 1), Tmp2);
      return Op.getResNo() ? Tmp2 : Tmp1;
    }
  }
  case ISD::EXTRACT_ELEMENT: {
    MVT OpTy = Node->getOperand(0).getValueType();
    switch (getTypeAction(OpTy)) {
    default: assert(0 && "EXTRACT_ELEMENT action for type unimplemented!");
    case Legal:
      if (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue()) {
        // 1 -> Hi
        Result = DAG.getNode(ISD::SRL, dl, OpTy, Node->getOperand(0),
                             DAG.getConstant(OpTy.getSizeInBits()/2,
                                             TLI.getShiftAmountTy()));
        Result = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0), Result);
      } else {
        // 0 -> Lo
        Result = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0), 
                             Node->getOperand(0));
      }
      break;
    case Expand:
      // Get both the low and high parts.
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      if (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue())
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
      AddLegalizedOperand(SDValue(Node, 0), Result.getValue(0));
      AddLegalizedOperand(SDValue(Node, 1), Result.getValue(1));
      return Result;
    }
    break;

  case ISD::RET:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.

    // Ensure that libcalls are emitted before a return.
    Tmp1 = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Tmp1, LastCALLSEQ_END);
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
        if (!Tmp2.getValueType().isVector()) {
          SDValue Lo, Hi;
          ExpandOp(Tmp2, Lo, Hi);

          // Big endian systems want the hi reg first.
          if (TLI.isBigEndian())
            std::swap(Lo, Hi);
          
          if (Hi.getNode())
            Result = DAG.getNode(ISD::RET, dl, MVT::Other, 
                                 Tmp1, Lo, Tmp3, Hi,Tmp3);
          else
            Result = DAG.getNode(ISD::RET, dl, MVT::Other, Tmp1, Lo, Tmp3);
          Result = LegalizeOp(Result);
        } else {
          SDNode *InVal = Tmp2.getNode();
          int InIx = Tmp2.getResNo();
          unsigned NumElems = InVal->getValueType(InIx).getVectorNumElements();
          MVT EVT = InVal->getValueType(InIx).getVectorElementType();
          
          // Figure out if there is a simple type corresponding to this Vector
          // type.  If so, convert to the vector type.
          MVT TVT = MVT::getVectorVT(EVT, NumElems);
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
            SDValue Lo, Hi;
            SplitVectorOp(Tmp2, Lo, Hi);
            Result = DAG.getNode(ISD::RET, dl, MVT::Other, 
                                 Tmp1, Lo, Tmp3, Hi,Tmp3);
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
      SmallVector<SDValue, 8> NewValues;
      NewValues.push_back(Tmp1);
      for (unsigned i = 1, e = Node->getNumOperands(); i < e; i += 2)
        switch (getTypeAction(Node->getOperand(i).getValueType())) {
        case Legal:
          NewValues.push_back(LegalizeOp(Node->getOperand(i)));
          NewValues.push_back(Node->getOperand(i+1));
          break;
        case Expand: {
          SDValue Lo, Hi;
          assert(!Node->getOperand(i).getValueType().isExtended() &&
                 "FIXME: TODO: implement returning non-legal vector types!");
          ExpandOp(Node->getOperand(i), Lo, Hi);
          NewValues.push_back(Lo);
          NewValues.push_back(Node->getOperand(i+1));
          if (Hi.getNode()) {
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
        Result = DAG.getNode(ISD::RET, dl, MVT::Other,
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
        if (Tmp1.getNode()) Result = Tmp1;
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
                                          bitcastToAPInt().zextOrTrunc(32),
                                  MVT::i32);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                SVOffset, isVolatile, Alignment);
          break;
        } else if (CFP->getValueType(0) == MVT::f64) {
          // If this target supports 64-bit registers, do a single 64-bit store.
          if (getTypeAction(MVT::i64) == Legal) {
            Tmp3 = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                                     zextOrTrunc(64), MVT::i64);
            Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            break;
          } else if (getTypeAction(MVT::i32) == Legal && !ST->isVolatile()) {
            // Otherwise, if the target supports 32-bit registers, use 2 32-bit
            // stores.  If the target supports neither 32- nor 64-bits, this
            // xform is certainly not worth it.
            const APInt &IntVal =CFP->getValueAPF().bitcastToAPInt();
            SDValue Lo = DAG.getConstant(APInt(IntVal).trunc(32), MVT::i32);
            SDValue Hi = DAG.getConstant(IntVal.lshr(32).trunc(32), MVT::i32);
            if (TLI.isBigEndian()) std::swap(Lo, Hi);

            Lo = DAG.getStore(Tmp1, dl, Lo, Tmp2, ST->getSrcValue(),
                              SVOffset, isVolatile, Alignment);
            Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                               DAG.getIntPtrConstant(4));
            Hi = DAG.getStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(), SVOffset+4,
                              isVolatile, MinAlign(Alignment, 4U));

            Result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
            break;
          }
        }
      }
      
      switch (getTypeAction(ST->getMemoryVT())) {
      case Legal: {
        Tmp3 = LegalizeOp(ST->getValue());
        Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp3, Tmp2, 
                                        ST->getOffset());

        MVT VT = Tmp3.getValueType();
        switch (TLI.getOperationAction(ISD::STORE, VT)) {
        default: assert(0 && "This action is not supported yet!");
        case TargetLowering::Legal:
          // If this is an unaligned store and the target doesn't support it,
          // expand it.
          if (!TLI.allowsUnalignedMemoryAccesses()) {
            unsigned ABIAlignment = TLI.getTargetData()->
              getABITypeAlignment(ST->getMemoryVT().getTypeForMVT());
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.getNode()), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.getNode()) Result = Tmp1;
          break;
        case TargetLowering::Promote:
          assert(VT.isVector() && "Unknown legal promote case!");
          Tmp3 = DAG.getNode(ISD::BIT_CONVERT, dl, 
                             TLI.getTypeToPromoteTo(ISD::STORE, VT), Tmp3);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2,
                                ST->getSrcValue(), SVOffset, isVolatile,
                                Alignment);
          break;
        }
        break;
      }
      case Promote:
        if (!ST->getMemoryVT().isVector()) {
          // Truncate the value and store the result.
          Tmp3 = PromoteOp(ST->getValue());
          Result = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                     SVOffset, ST->getMemoryVT(),
                                     isVolatile, Alignment);
          break;
        }
        // Fall thru to expand for vector
      case Expand: {
        unsigned IncrementSize = 0;
        SDValue Lo, Hi;
      
        // If this is a vector type, then we have to calculate the increment as
        // the product of the element size in bytes, and the number of elements
        // in the high half of the vector.
        if (ST->getValue().getValueType().isVector()) {
          SDNode *InVal = ST->getValue().getNode();
          int InIx = ST->getValue().getResNo();
          MVT InVT = InVal->getValueType(InIx);
          unsigned NumElems = InVT.getVectorNumElements();
          MVT EVT = InVT.getVectorElementType();

          // Figure out if there is a simple type corresponding to this Vector
          // type.  If so, convert to the vector type.
          MVT TVT = MVT::getVectorVT(EVT, NumElems);
          if (TLI.isTypeLegal(TVT)) {
            // Turn this into a normal store of the vector type.
            Tmp3 = LegalizeOp(ST->getValue());
            Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            Result = LegalizeOp(Result);
            break;
          } else if (NumElems == 1) {
            // Turn this into a normal store of the scalar type.
            Tmp3 = ScalarizeVectorOp(ST->getValue());
            Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                  SVOffset, isVolatile, Alignment);
            // The scalarized value type may not be legal, e.g. it might require
            // promotion or expansion.  Relegalize the scalar store.
            Result = LegalizeOp(Result);
            break;
          } else {
            // Check if we have widen this node with another value
            std::map<SDValue, SDValue>::iterator I =
              WidenNodes.find(ST->getValue());
            if (I != WidenNodes.end()) {
              Result = StoreWidenVectorOp(ST, Tmp1, Tmp2);
              break;
            }
            else {
              SplitVectorOp(ST->getValue(), Lo, Hi);
              IncrementSize = Lo.getNode()->getValueType(0).getVectorNumElements() *
                              EVT.getSizeInBits()/8;
            }
          }
        } else {
          ExpandOp(ST->getValue(), Lo, Hi);
          IncrementSize = Hi.getNode() ? Hi.getValueType().getSizeInBits()/8 : 0;

          if (Hi.getNode() && TLI.isBigEndian())
            std::swap(Lo, Hi);
        }

        Lo = DAG.getStore(Tmp1, dl, Lo, Tmp2, ST->getSrcValue(),
                          SVOffset, isVolatile, Alignment);

        if (Hi.getNode() == NULL) {
          // Must be int <-> float one-to-one expansion.
          Result = Lo;
          break;
        }

        Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                           DAG.getIntPtrConstant(IncrementSize));
        assert(isTypeLegal(Tmp2.getValueType()) &&
               "Pointers must be legal!");
        SVOffset += IncrementSize;
        Alignment = MinAlign(Alignment, IncrementSize);
        Hi = DAG.getStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(),
                          SVOffset, isVolatile, Alignment);
        Result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
        break;
      }  // case Expand
      }
    } else {
      switch (getTypeAction(ST->getValue().getValueType())) {
      case Legal:
        Tmp3 = LegalizeOp(ST->getValue());
        break;
      case Promote:
        if (!ST->getValue().getValueType().isVector()) {
          // We can promote the value, the truncstore will still take care of it.
          Tmp3 = PromoteOp(ST->getValue());
          break;
        }
        // Vector case falls through to expand
      case Expand:
        // Just store the low part.  This may become a non-trunc store, so make
        // sure to use getTruncStore, not UpdateNodeOperands below.
        ExpandOp(ST->getValue(), Tmp3, Tmp4);
        return DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset, MVT::i8, isVolatile, Alignment);
      }

      MVT StVT = ST->getMemoryVT();
      unsigned StWidth = StVT.getSizeInBits();

      if (StWidth != StVT.getStoreSizeInBits()) {
        // Promote to a byte-sized store with upper bits zero if not
        // storing an integral number of bytes.  For example, promote
        // TRUNCSTORE:i1 X -> TRUNCSTORE:i8 (and X, 1)
        MVT NVT = MVT::getIntegerVT(StVT.getStoreSizeInBits());
        Tmp3 = DAG.getZeroExtendInReg(Tmp3, dl, StVT);
        Result = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                   SVOffset, NVT, isVolatile, Alignment);
      } else if (StWidth & (StWidth - 1)) {
        // If not storing a power-of-2 number of bits, expand as two stores.
        assert(StVT.isExtended() && !StVT.isVector() &&
               "Unsupported truncstore!");
        unsigned RoundWidth = 1 << Log2_32(StWidth);
        assert(RoundWidth < StWidth);
        unsigned ExtraWidth = StWidth - RoundWidth;
        assert(ExtraWidth < RoundWidth);
        assert(!(RoundWidth % 8) && !(ExtraWidth % 8) &&
               "Store size not an integral number of bytes!");
        MVT RoundVT = MVT::getIntegerVT(RoundWidth);
        MVT ExtraVT = MVT::getIntegerVT(ExtraWidth);
        SDValue Lo, Hi;
        unsigned IncrementSize;

        if (TLI.isLittleEndian()) {
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 X, TRUNCSTORE@+2:i8 (srl X, 16)
          // Store the bottom RoundWidth bits.
          Lo = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset, RoundVT,
                                 isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Hi = DAG.getNode(ISD::SRL, dl, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(RoundWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        } else {
          // Big endian - avoid unaligned stores.
          // TRUNCSTORE:i24 X -> TRUNCSTORE:i16 (srl X, 8), TRUNCSTORE@+2:i8 X
          // Store the top RoundWidth bits.
          Hi = DAG.getNode(ISD::SRL, dl, Tmp3.getValueType(), Tmp3,
                           DAG.getConstant(ExtraWidth, TLI.getShiftAmountTy()));
          Hi = DAG.getTruncStore(Tmp1, dl, Hi, Tmp2, ST->getSrcValue(),
                                 SVOffset, RoundVT, isVolatile, Alignment);

          // Store the remaining ExtraWidth bits.
          IncrementSize = RoundWidth / 8;
          Tmp2 = DAG.getNode(ISD::ADD, dl, Tmp2.getValueType(), Tmp2,
                             DAG.getIntPtrConstant(IncrementSize));
          Lo = DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                 SVOffset + IncrementSize, ExtraVT, isVolatile,
                                 MinAlign(Alignment, IncrementSize));
        }

        // The order of the stores doesn't matter.
        Result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
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
              getABITypeAlignment(ST->getMemoryVT().getTypeForMVT());
            if (ST->getAlignment() < ABIAlignment)
              Result = ExpandUnalignedStore(cast<StoreSDNode>(Result.getNode()), DAG,
                                            TLI);
          }
          break;
        case TargetLowering::Custom:
          Result = TLI.LowerOperation(Result, DAG);
          break;
        case Expand:
          // TRUNCSTORE:i16 i32 -> STORE i16
          assert(isTypeLegal(StVT) && "Do not know how to expand this store!");
          Tmp3 = DAG.getNode(ISD::TRUNCATE, dl, StVT, Tmp3);
          Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2, ST->getSrcValue(),
                                SVOffset, isVolatile, Alignment);
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
      if (Tmp3.getNode()) {
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
        Tmp1 = DAG.getNode(ISD::UNDEF, dl, Node->getValueType(0));
        Tmp2 = Node->getOperand(0);
      }
      break;
    }

    // Since stacksave produce two values, make sure to remember that we
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Tmp1);
    AddLegalizedOperand(SDValue(Node, 1), Tmp2);
    return Op.getResNo() ? Tmp2 : Tmp1;

  case ISD::STACKRESTORE:
    Tmp1 = LegalizeOp(Node->getOperand(0));  // Legalize the chain.
    Tmp2 = LegalizeOp(Node->getOperand(1));  // Legalize the pointer.
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
      
    switch (TLI.getOperationAction(ISD::STACKRESTORE, MVT::Other)) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
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
    AddLegalizedOperand(SDValue(Node, 0), Tmp1);
    AddLegalizedOperand(SDValue(Node, 1), Tmp2);
    return Result;

  case ISD::SELECT:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "It's impossible to expand bools");
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0)); // Legalize the condition.
      break;
    case Promote: {
      assert(!Node->getOperand(0).getValueType().isVector() && "not possible");
      Tmp1 = PromoteOp(Node->getOperand(0));  // Promote the condition.
      // Make sure the condition is either zero or one.
      unsigned BitWidth = Tmp1.getValueSizeInBits();
      if (!DAG.MaskedValueIsZero(Tmp1,
                                 APInt::getHighBitsSet(BitWidth, BitWidth-1)))
        Tmp1 = DAG.getZeroExtendInReg(Tmp1, dl, MVT::i1);
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
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    }
    case TargetLowering::Expand:
      if (Tmp1.getOpcode() == ISD::SETCC) {
        Result = DAG.getSelectCC(dl, Tmp1.getOperand(0), Tmp1.getOperand(1), 
                              Tmp2, Tmp3,
                              cast<CondCodeSDNode>(Tmp1.getOperand(2))->get());
      } else {
        Result = DAG.getSelectCC(dl, Tmp1, 
                                 DAG.getConstant(0, Tmp1.getValueType()),
                                 Tmp2, Tmp3, ISD::SETNE);
      }
      break;
    case TargetLowering::Promote: {
      MVT NVT =
        TLI.getTypeToPromoteTo(ISD::SELECT, Tmp2.getValueType());
      unsigned ExtOp, TruncOp;
      if (Tmp2.getValueType().isVector()) {
        ExtOp   = ISD::BIT_CONVERT;
        TruncOp = ISD::BIT_CONVERT;
      } else if (Tmp2.getValueType().isInteger()) {
        ExtOp   = ISD::ANY_EXTEND;
        TruncOp = ISD::TRUNCATE;
      } else {
        ExtOp   = ISD::FP_EXTEND;
        TruncOp = ISD::FP_ROUND;
      }
      // Promote each of the values to the new type.
      Tmp2 = DAG.getNode(ExtOp, dl, NVT, Tmp2);
      Tmp3 = DAG.getNode(ExtOp, dl, NVT, Tmp3);
      // Perform the larger operation, then round down.
      Result = DAG.getNode(ISD::SELECT, dl, NVT, Tmp1, Tmp2,Tmp3);
      if (TruncOp != ISD::FP_ROUND)
        Result = DAG.getNode(TruncOp, dl, Node->getValueType(0), Result);
      else
        Result = DAG.getNode(TruncOp, dl, Node->getValueType(0), Result,
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
    SDValue CC = Node->getOperand(4);
    
    LegalizeSetCC(TLI.getSetCCResultType(Tmp1.getValueType()), 
                  Tmp1, Tmp2, CC, dl);
    
    // If we didn't get both a LHS and RHS back from LegalizeSetCC,
    // the LHS is a legal SETCC itself.  In this case, we need to compare
    // the result against zero to select between true and false values.
    if (Tmp2.getNode() == 0) {
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
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    }
    break;
  }
  case ISD::SETCC:
    Tmp1 = Node->getOperand(0);
    Tmp2 = Node->getOperand(1);
    Tmp3 = Node->getOperand(2);
    LegalizeSetCC(Node->getValueType(0), Tmp1, Tmp2, Tmp3, dl);
    
    // If we had to Expand the SetCC operands into a SELECT node, then it may 
    // not always be possible to return a true LHS & RHS.  In this case, just 
    // return the value we legalized, returned in the LHS
    if (Tmp2.getNode() == 0) {
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
        if (Tmp4.getNode()) Result = Tmp4;
      }
      break;
    case TargetLowering::Promote: {
      // First step, figure out the appropriate operation to use.
      // Allow SETCC to not be supported for all legal data types
      // Mostly this targets FP
      MVT NewInTy = Node->getOperand(0).getValueType();
      MVT OldVT = NewInTy; OldVT = OldVT;

      // Scan for the appropriate larger type to use.
      while (1) {
        NewInTy = (MVT::SimpleValueType)(NewInTy.getSimpleVT()+1);

        assert(NewInTy.isInteger() == OldVT.isInteger() &&
               "Fell off of the edge of the integer world");
        assert(NewInTy.isFloatingPoint() == OldVT.isFloatingPoint() &&
               "Fell off of the edge of the floating point world");
          
        // If the target supports SETCC of this type, use it.
        if (TLI.isOperationLegalOrCustom(ISD::SETCC, NewInTy))
          break;
      }
      if (NewInTy.isInteger())
        assert(0 && "Cannot promote Legal Integer SETCC yet");
      else {
        Tmp1 = DAG.getNode(ISD::FP_EXTEND, dl, NewInTy, Tmp1);
        Tmp2 = DAG.getNode(ISD::FP_EXTEND, dl, NewInTy, Tmp2);
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
      MVT VT = Node->getValueType(0);
      Result = DAG.getNode(ISD::SELECT_CC, dl, VT, Tmp1, Tmp2, 
                           DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                           Tmp3);
      break;
    }
    break;
  case ISD::VSETCC: {
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS
    SDValue CC = Node->getOperand(2);
    
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, CC);

    // Everything is legal, see if we should expand this op or something.
    switch (TLI.getOperationAction(ISD::VSETCC, Tmp1.getValueType())) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    case TargetLowering::Expand: {
      // Unroll into a nasty set of scalar code for now.
      MVT VT = Node->getValueType(0);
      unsigned NumElems = VT.getVectorNumElements();
      MVT EltVT = VT.getVectorElementType();
      MVT TmpEltVT = Tmp1.getValueType().getVectorElementType();
      SmallVector<SDValue, 8> Ops(NumElems);
      for (unsigned i = 0; i < NumElems; ++i) {
        SDValue In1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, TmpEltVT,
                                  Tmp1, DAG.getIntPtrConstant(i));
        Ops[i] = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(TmpEltVT),
                             In1, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                                              TmpEltVT, Tmp2,
                                              DAG.getIntPtrConstant(i)),
                             CC);
        Ops[i] = DAG.getNode(ISD::SELECT, dl, EltVT, Ops[i], DAG.getConstant(
                                  APInt::getAllOnesValue(EltVT.getSizeInBits()),
                                  EltVT), DAG.getConstant(0, EltVT));
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], NumElems);
      break;
    }
    }
    break;
  }

  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: {
    SmallVector<SDValue, 8> Ops;
    bool Changed = false;
    unsigned N = Node->getNumOperands();
    for (unsigned i = 0; i + 1 < N; ++i) {
      Ops.push_back(LegalizeOp(Node->getOperand(i)));
      Changed |= Ops.back() != Node->getOperand(i);
    }
    Ops.push_back(LegalizeOp(DAG.getShiftAmountOperand(Node->getOperand(N-1))));
    Changed |= Ops.back() != Node->getOperand(N-1);
    if (Changed)
      Result = DAG.UpdateNodeOperands(Result, &Ops[0], Ops.size());

    switch (TLI.getOperationAction(Node->getOpcode(),
                                   Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) {
        SDValue Tmp2, RetVal(0, 0);
        for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i) {
          Tmp2 = LegalizeOp(Tmp1.getValue(i));
          AddLegalizedOperand(SDValue(Node, i), Tmp2);
          if (i == Op.getResNo())
            RetVal = Tmp2;
        }
        assert(RetVal.getNode() && "Illegal result number");
        return RetVal;
      }
      break;
    }

    // Since these produce multiple values, make sure to remember that we
    // legalized all of them.
    for (unsigned i = 0, e = Node->getNumValues(); i != e; ++i)
      AddLegalizedOperand(SDValue(Node, i), Result.getValue(i));
    return Result.getValue(Op.getResNo());
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
    Tmp2 = LegalizeOp(Node->getOperand(1));   // RHS

    if ((Node->getOpcode() == ISD::SHL ||
         Node->getOpcode() == ISD::SRL ||
         Node->getOpcode() == ISD::SRA) &&
        !Node->getValueType(0).isVector())
      Tmp2 = DAG.getShiftAmountOperand(Tmp2);

    switch (getTypeAction(Tmp2.getValueType())) {
    case Expand: assert(0 && "Not possible");
    case Legal:
      Tmp2 = LegalizeOp(Tmp2); // Legalize the RHS.
      break;
    case Promote:
      Tmp2 = PromoteOp(Tmp2);  // Promote the RHS.
      break;
    }

    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "BinOp legalize operation not supported");
    case TargetLowering::Legal: break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) {
        Result = Tmp1;
        break;
      }
      // Fall through if the custom lower can't deal with the operation
    case TargetLowering::Expand: {
      MVT VT = Op.getValueType();
      
      // See if multiply or divide can be lowered using two-result operations.
      SDVTList VTs = DAG.getVTList(VT, VT);
      if (Node->getOpcode() == ISD::MUL) {
        // We just need the low half of the multiply; try both the signed
        // and unsigned forms. If the target supports both SMUL_LOHI and
        // UMUL_LOHI, form a preference by checking which forms of plain
        // MULH it supports.
        bool HasSMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::SMUL_LOHI, VT);
        bool HasUMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::UMUL_LOHI, VT);
        bool HasMULHS = TLI.isOperationLegalOrCustom(ISD::MULHS, VT);
        bool HasMULHU = TLI.isOperationLegalOrCustom(ISD::MULHU, VT);
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
          Result = SDValue(DAG.getNode(OpToUse, dl, VTs, Tmp1, Tmp2).getNode(),
                           0);
          break;
        }
      }
      if (Node->getOpcode() == ISD::MULHS &&
          TLI.isOperationLegalOrCustom(ISD::SMUL_LOHI, VT)) {
        Result = SDValue(DAG.getNode(ISD::SMUL_LOHI, dl, 
                                     VTs, Tmp1, Tmp2).getNode(),
                         1);
        break;
      }
      if (Node->getOpcode() == ISD::MULHU && 
          TLI.isOperationLegalOrCustom(ISD::UMUL_LOHI, VT)) {
        Result = SDValue(DAG.getNode(ISD::UMUL_LOHI, dl,
                                     VTs, Tmp1, Tmp2).getNode(),
                         1);
        break;
      }
      if (Node->getOpcode() == ISD::SDIV &&
          TLI.isOperationLegalOrCustom(ISD::SDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::SDIVREM, dl, 
                                     VTs, Tmp1, Tmp2).getNode(),
                         0);
        break;
      }
      if (Node->getOpcode() == ISD::UDIV &&
          TLI.isOperationLegalOrCustom(ISD::UDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::UDIVREM, dl,
                                     VTs, Tmp1, Tmp2).getNode(),
                         0);
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
      case ISD::MUL:
        if (VT == MVT::i32)
          LC = RTLIB::MUL_I32;
        else if (VT == MVT::i64)
          LC = RTLIB::MUL_I64;
        break;
      case ISD::FPOW:
        LC = GetFPLibCall(VT, RTLIB::POW_F32, RTLIB::POW_F64, RTLIB::POW_F80,
                          RTLIB::POW_PPCF128);
        break;
      case ISD::FDIV:
        LC = GetFPLibCall(VT, RTLIB::DIV_F32, RTLIB::DIV_F64, RTLIB::DIV_F80,
                          RTLIB::DIV_PPCF128);
        break;
      default: break;
      }
      if (LC != RTLIB::UNKNOWN_LIBCALL) {
        SDValue Dummy;
        Result = ExpandLibCall(LC, Node, isSigned, Dummy);
        break;
      }
      
      assert(Node->getValueType(0).isVector() &&
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
        MVT OVT = Node->getValueType(0);
        MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);
        assert(OVT.isVector() && "Cannot promote this BinOp!");
        // Bit convert each of the values to the new type.
        Tmp1 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp1);
        Tmp2 = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Tmp2);
        Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
        // Bit convert the result back the original type.
        Result = DAG.getNode(ISD::BIT_CONVERT, dl, OVT, Result);
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
      if (Tmp1.getNode()) Result = Tmp1;
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
        MVT IVT =
          Tmp2.getValueType() == MVT::f32 ? MVT::i32 : MVT::i64;
        SDValue SignBit = DAG.getNode(ISD::BIT_CONVERT, dl, IVT, Tmp2);
        SignBit = DAG.getSetCC(dl, TLI.getSetCCResultType(IVT),
                               SignBit, DAG.getConstant(0, IVT), ISD::SETLT);
        // Get the absolute value of the result.
        SDValue AbsVal = DAG.getNode(ISD::FABS, dl, Tmp1.getValueType(), Tmp1);
        // Select between the nabs and abs value based on the sign bit of
        // the input.
        Result = DAG.getNode(ISD::SELECT, dl, AbsVal.getValueType(), SignBit,
                             DAG.getNode(ISD::FNEG, AbsVal.getValueType(), 
                                         AbsVal),
                             AbsVal);
        Result = LegalizeOp(Result);
        break;
      }
      
      // Otherwise, do bitwise ops!
      MVT NVT =
        Node->getValueType(0) == MVT::f32 ? MVT::i32 : MVT::i64;
      Result = ExpandFCOPYSIGNToBitwiseOps(Node, NVT, DAG, TLI);
      Result = DAG.getNode(ISD::BIT_CONVERT, dl, Node->getValueType(0), Result);
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
    Tmp3 = Result.getValue(0);
    Tmp4 = Result.getValue(1);

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Tmp3, DAG);
      if (Tmp1.getNode() != NULL) {
        Tmp3 = LegalizeOp(Tmp1);
        Tmp4 = LegalizeOp(Tmp1.getValue(1));
      }
      break;
    }
    // Since this produces two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDValue(Node, 0), Tmp3);
    AddLegalizedOperand(SDValue(Node, 1), Tmp4);
    return Op.getResNo() ? Tmp4 : Tmp3;

  case ISD::ADDE:
  case ISD::SUBE:
    Tmp1 = LegalizeOp(Node->getOperand(0));
    Tmp2 = LegalizeOp(Node->getOperand(1));
    Tmp3 = LegalizeOp(Node->getOperand(2));
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2, Tmp3);
    Tmp3 = Result.getValue(0);
    Tmp4 = Result.getValue(1);

    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default: assert(0 && "This action is not supported yet!");
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Tmp3, DAG);
      if (Tmp1.getNode() != NULL) {
        Tmp3 = LegalizeOp(Tmp1);
        Tmp4 = LegalizeOp(Tmp1.getValue(1));
      }
      break;
    }
    // Since this produces two values, make sure to remember that we legalized
    // both of them.
    AddLegalizedOperand(SDValue(Node, 0), Tmp3);
    AddLegalizedOperand(SDValue(Node, 1), Tmp4);
    return Op.getResNo() ? Tmp4 : Tmp3;
    
  case ISD::BUILD_PAIR: {
    MVT PairTy = Node->getValueType(0);
    // TODO: handle the case where the Lo and Hi operands are not of legal type
    Tmp1 = LegalizeOp(Node->getOperand(0));   // Lo
    Tmp2 = LegalizeOp(Node->getOperand(1));   // Hi
    switch (TLI.getOperationAction(ISD::BUILD_PAIR, PairTy)) {
    case TargetLowering::Promote:
    case TargetLowering::Custom:
      assert(0 && "Cannot promote/custom this yet!");
    case TargetLowering::Legal:
      if (Tmp1 != Node->getOperand(0) || Tmp2 != Node->getOperand(1))
        Result = DAG.getNode(ISD::BUILD_PAIR, dl, PairTy, Tmp1, Tmp2);
      break;
    case TargetLowering::Expand:
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, PairTy, Tmp1);
      Tmp2 = DAG.getNode(ISD::ANY_EXTEND, dl, PairTy, Tmp2);
      Tmp2 = DAG.getNode(ISD::SHL, dl, PairTy, Tmp2,
                         DAG.getConstant(PairTy.getSizeInBits()/2,
                                         TLI.getShiftAmountTy()));
      Result = DAG.getNode(ISD::OR, dl, PairTy, Tmp1, Tmp2);
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
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand: {
      unsigned DivOpc= (Node->getOpcode() == ISD::UREM) ? ISD::UDIV : ISD::SDIV;
      bool isSigned = DivOpc == ISD::SDIV;
      MVT VT = Node->getValueType(0);
 
      // See if remainder can be lowered using two-result operations.
      SDVTList VTs = DAG.getVTList(VT, VT);
      if (Node->getOpcode() == ISD::SREM &&
          TLI.isOperationLegalOrCustom(ISD::SDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::SDIVREM, dl, 
                                     VTs, Tmp1, Tmp2).getNode(), 1);
        break;
      }
      if (Node->getOpcode() == ISD::UREM &&
          TLI.isOperationLegalOrCustom(ISD::UDIVREM, VT)) {
        Result = SDValue(DAG.getNode(ISD::UDIVREM, dl,
                                     VTs, Tmp1, Tmp2).getNode(), 1);
        break;
      }

      if (VT.isInteger()) {
        if (TLI.getOperationAction(DivOpc, VT) ==
            TargetLowering::Legal) {
          // X % Y -> X-X/Y*Y
          Result = DAG.getNode(DivOpc, dl, VT, Tmp1, Tmp2);
          Result = DAG.getNode(ISD::MUL, dl, VT, Result, Tmp2);
          Result = DAG.getNode(ISD::SUB, dl, VT, Tmp1, Result);
        } else if (VT.isVector()) {
          Result = LegalizeOp(UnrollVectorOp(Op));
        } else {
          assert(VT == MVT::i32 &&
                 "Cannot expand this binary operator!");
          RTLIB::Libcall LC = Node->getOpcode() == ISD::UREM
            ? RTLIB::UREM_I32 : RTLIB::SREM_I32;
          SDValue Dummy;
          Result = ExpandLibCall(LC, Node, isSigned, Dummy);
        }
      } else {
        assert(VT.isFloatingPoint() &&
               "remainder op must have integer or floating-point type");
        if (VT.isVector()) {
          Result = LegalizeOp(UnrollVectorOp(Op));
        } else {
          // Floating point mod -> fmod libcall.
          RTLIB::Libcall LC = GetFPLibCall(VT, RTLIB::REM_F32, RTLIB::REM_F64,
                                           RTLIB::REM_F80, RTLIB::REM_PPCF128);
          SDValue Dummy;
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

    MVT VT = Node->getValueType(0);
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
        if (Tmp2.getNode()) {
          Result = LegalizeOp(Tmp2);
          Tmp1 = LegalizeOp(Tmp2.getValue(1));
        }
      }
      break;
    case TargetLowering::Expand: {
      const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
      SDValue VAList = DAG.getLoad(TLI.getPointerTy(), dl, Tmp1, Tmp2, V, 0);
      // Increment the pointer, VAList, to the next vaarg
      Tmp3 = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), VAList,
                         DAG.getConstant(TLI.getTargetData()->
                                         getTypePaddedSize(VT.getTypeForMVT()),
                                         TLI.getPointerTy()));
      // Store the incremented VAList to the legalized pointer
      Tmp3 = DAG.getStore(VAList.getValue(1), dl, Tmp3, Tmp2, V, 0);
      // Load the actual argument out of the pointer VAList
      Result = DAG.getLoad(VT, dl, Tmp3, VAList, NULL, 0);
      Tmp1 = LegalizeOp(Result.getValue(1));
      Result = LegalizeOp(Result);
      break;
    }
    }
    // Since VAARG produces two values, make sure to remember that we 
    // legalized both of them.
    AddLegalizedOperand(SDValue(Node, 0), Result);
    AddLegalizedOperand(SDValue(Node, 1), Tmp1);
    return Op.getResNo() ? Tmp1 : Result;
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
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      // This defaults to loading a pointer from the input and storing it to the
      // output, returning the chain.
      const Value *VD = cast<SrcValueSDNode>(Node->getOperand(3))->getValue();
      const Value *VS = cast<SrcValueSDNode>(Node->getOperand(4))->getValue();
      Tmp4 = DAG.getLoad(TLI.getPointerTy(), dl, Tmp1, Tmp3, VS, 0);
      Result = DAG.getStore(Tmp4.getValue(1), dl, Tmp4, Tmp2, VD, 0);
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
        if (Tmp1.getNode()) Result = Tmp1;
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
      if (Tmp1.getNode()) Result = Tmp1;
      break;
    }
    break;
    
  case ISD::ROTL:
  case ISD::ROTR:
    Tmp1 = LegalizeOp(Node->getOperand(0));   // LHS
    Tmp2 = LegalizeOp(DAG.getShiftAmountOperand(Node->getOperand(1)));   // RHS
    Result = DAG.UpdateNodeOperands(Result, Tmp1, Tmp2);
    switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
    default:
      assert(0 && "ROTL/ROTR legalize operation not supported");
      break;
    case TargetLowering::Legal:
      break;
    case TargetLowering::Custom:
      Tmp1 = TLI.LowerOperation(Result, DAG);
      if (Tmp1.getNode()) Result = Tmp1;
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
      MVT OVT = Tmp1.getValueType();
      MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);
      unsigned DiffBits = NVT.getSizeInBits() - OVT.getSizeInBits();

      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Tmp1);
      Tmp1 = DAG.getNode(ISD::BSWAP, dl, NVT, Tmp1);
      Result = DAG.getNode(ISD::SRL, dl, NVT, Tmp1,
                           DAG.getConstant(DiffBits, TLI.getShiftAmountTy()));
      break;
    }
    case TargetLowering::Expand:
      Result = ExpandBSWAP(Tmp1, dl);
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
        if (Tmp1.getNode()) {
          Result = Tmp1;
        }
      }
      break;
    case TargetLowering::Promote: {
      MVT OVT = Tmp1.getValueType();
      MVT NVT = TLI.getTypeToPromoteTo(Node->getOpcode(), OVT);

      // Zero extend the argument.
      Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Tmp1);
      // Perform the larger operation, then subtract if needed.
      Tmp1 = DAG.getNode(Node->getOpcode(), dl, Node->getValueType(0), Tmp1);
      switch (Node->getOpcode()) {
      case ISD::CTPOP:
        Result = Tmp1;
        break;
      case ISD::CTTZ:
        //if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
        Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(Tmp1.getValueType()),
                            Tmp1, DAG.getConstant(NVT.getSizeInBits(), NVT),
                            ISD::SETEQ);
        Result = DAG.getNode(ISD::SELECT, dl, NVT, Tmp2,
                             DAG.getConstant(OVT.getSizeInBits(), NVT), Tmp1);
        break;
      case ISD::CTLZ:
        // Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
        Result = DAG.getNode(ISD::SUB, dl, NVT, Tmp1,
                             DAG.getConstant(NVT.getSizeInBits() -
                                             OVT.getSizeInBits(), NVT));
        break;
      }
      break;
    }
    case TargetLowering::Expand:
      Result = ExpandBitCount(Node->getOpcode(), Tmp1, dl);
      break;
    }
    break;

    // Unary operators
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FTRUNC:
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
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
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      switch (Node->getOpcode()) {
      default: assert(0 && "Unreachable!");
      case ISD::FNEG:
        // Expand Y = FNEG(X) ->  Y = SUB -0.0, X
        Tmp2 = DAG.getConstantFP(-0.0, Node->getValueType(0));
        Result = DAG.getNode(ISD::FSUB, dl, Node->getValueType(0), Tmp2, Tmp1);
        break;
      case ISD::FABS: {
        // Expand Y = FABS(X) -> Y = (X >u 0.0) ? X : fneg(X).
        MVT VT = Node->getValueType(0);
        Tmp2 = DAG.getConstantFP(0.0, VT);
        Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(Tmp1.getValueType()),
                            Tmp1, Tmp2, ISD::SETUGT);
        Tmp3 = DAG.getNode(ISD::FNEG, dl, VT, Tmp1);
        Result = DAG.getNode(ISD::SELECT, dl, VT, Tmp2, Tmp1, Tmp3);
        break;
      }
      case ISD::FSQRT:
      case ISD::FSIN:
      case ISD::FCOS: 
      case ISD::FLOG:
      case ISD::FLOG2:
      case ISD::FLOG10:
      case ISD::FEXP:
      case ISD::FEXP2:
      case ISD::FTRUNC:
      case ISD::FFLOOR:
      case ISD::FCEIL:
      case ISD::FRINT:
      case ISD::FNEARBYINT: {
        MVT VT = Node->getValueType(0);

        // Expand unsupported unary vector operators by unrolling them.
        if (VT.isVector()) {
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
        case ISD::FLOG:
          LC = GetFPLibCall(VT, RTLIB::LOG_F32, RTLIB::LOG_F64,
                            RTLIB::LOG_F80, RTLIB::LOG_PPCF128);
          break;
        case ISD::FLOG2:
          LC = GetFPLibCall(VT, RTLIB::LOG2_F32, RTLIB::LOG2_F64,
                            RTLIB::LOG2_F80, RTLIB::LOG2_PPCF128);
          break;
        case ISD::FLOG10:
          LC = GetFPLibCall(VT, RTLIB::LOG10_F32, RTLIB::LOG10_F64,
                            RTLIB::LOG10_F80, RTLIB::LOG10_PPCF128);
          break;
        case ISD::FEXP:
          LC = GetFPLibCall(VT, RTLIB::EXP_F32, RTLIB::EXP_F64,
                            RTLIB::EXP_F80, RTLIB::EXP_PPCF128);
          break;
        case ISD::FEXP2:
          LC = GetFPLibCall(VT, RTLIB::EXP2_F32, RTLIB::EXP2_F64,
                            RTLIB::EXP2_F80, RTLIB::EXP2_PPCF128);
          break;
        case ISD::FTRUNC:
          LC = GetFPLibCall(VT, RTLIB::TRUNC_F32, RTLIB::TRUNC_F64,
                            RTLIB::TRUNC_F80, RTLIB::TRUNC_PPCF128);
          break;
        case ISD::FFLOOR:
          LC = GetFPLibCall(VT, RTLIB::FLOOR_F32, RTLIB::FLOOR_F64,
                            RTLIB::FLOOR_F80, RTLIB::FLOOR_PPCF128);
          break;
        case ISD::FCEIL:
          LC = GetFPLibCall(VT, RTLIB::CEIL_F32, RTLIB::CEIL_F64,
                            RTLIB::CEIL_F80, RTLIB::CEIL_PPCF128);
          break;
        case ISD::FRINT:
          LC = GetFPLibCall(VT, RTLIB::RINT_F32, RTLIB::RINT_F64,
                            RTLIB::RINT_F80, RTLIB::RINT_PPCF128);
          break;
        case ISD::FNEARBYINT:
          LC = GetFPLibCall(VT, RTLIB::NEARBYINT_F32, RTLIB::NEARBYINT_F64,
                            RTLIB::NEARBYINT_F80, RTLIB::NEARBYINT_PPCF128);
          break;
      break;
        default: assert(0 && "Unreachable!");
        }
        SDValue Dummy;
        Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
        break;
      }
      }
      break;
    }
    break;
  case ISD::FPOWI: {
    MVT VT = Node->getValueType(0);

    // Expand unsupported unary vector operators by unrolling them.
    if (VT.isVector()) {
      Result = LegalizeOp(UnrollVectorOp(Op));
      break;
    }

    // We always lower FPOWI into a libcall.  No target support for it yet.
    RTLIB::Libcall LC = GetFPLibCall(VT, RTLIB::POWI_F32, RTLIB::POWI_F64,
                                     RTLIB::POWI_F80, RTLIB::POWI_PPCF128);
    SDValue Dummy;
    Result = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Dummy);
    break;
  }
  case ISD::BIT_CONVERT:
    if (!isTypeLegal(Node->getOperand(0).getValueType())) {
      Result = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                                Node->getValueType(0), dl);
    } else if (Op.getOperand(0).getValueType().isVector()) {
      // The input has to be a vector type, we have to either scalarize it, pack
      // it, or convert it based on whether the input vector type is legal.
      SDNode *InVal = Node->getOperand(0).getNode();
      int InIx = Node->getOperand(0).getResNo();
      unsigned NumElems = InVal->getValueType(InIx).getVectorNumElements();
      MVT EVT = InVal->getValueType(InIx).getVectorElementType();
    
      // Figure out if there is a simple type corresponding to this Vector
      // type.  If so, convert to the vector type.
      MVT TVT = MVT::getVectorVT(EVT, NumElems);
      if (TLI.isTypeLegal(TVT)) {
        // Turn this into a bit convert of the vector input.
        Result = DAG.getNode(ISD::BIT_CONVERT, dl, Node->getValueType(0), 
                             LegalizeOp(Node->getOperand(0)));
        break;
      } else if (NumElems == 1) {
        // Turn this into a bit convert of the scalar input.
        Result = DAG.getNode(ISD::BIT_CONVERT, dl, Node->getValueType(0), 
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
                                  Node->getValueType(0), dl);
        break;
      case TargetLowering::Legal:
        Tmp1 = LegalizeOp(Node->getOperand(0));
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
        break;
      }
    }
    break;
  case ISD::CONVERT_RNDSAT: {
    ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(Node)->getCvtCode();
    switch (CvtCode) {
    default: assert(0 && "Unknown cvt code!");
    case ISD::CVT_SF:
    case ISD::CVT_UF:
    case ISD::CVT_FF:
      break;
    case ISD::CVT_FS:
    case ISD::CVT_FU:
    case ISD::CVT_SS:
    case ISD::CVT_SU:
    case ISD::CVT_US:
    case ISD::CVT_UU: {
      SDValue DTyOp = Node->getOperand(1);
      SDValue STyOp = Node->getOperand(2);
      SDValue RndOp = Node->getOperand(3);
      SDValue SatOp = Node->getOperand(4);
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "Shouldn't need to expand other operators here!");
      case Legal:
        Tmp1 = LegalizeOp(Node->getOperand(0));
        Result = DAG.UpdateNodeOperands(Result, Tmp1, DTyOp, STyOp,
                                        RndOp, SatOp);
        if (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0)) ==
            TargetLowering::Custom) {
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.getNode()) Result = Tmp1;
        }
        break;
      case Promote:
        Result = PromoteOp(Node->getOperand(0));
        // For FP, make Op1 a i32
        
        Result = DAG.getConvertRndSat(Op.getValueType(), Result,
                                      DTyOp, STyOp, RndOp, SatOp, CvtCode);
        break;
      }
      break;
    }
    } // end switch CvtCode
    break;
  }
    // Conversion operators.  The source and destination have different types.
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    bool isSigned = Node->getOpcode() == ISD::SINT_TO_FP;
    Result = LegalizeINT_TO_FP(Result, isSigned,
                               Node->getValueType(0), Node->getOperand(0), dl);
    break;
  }
  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      switch (TLI.getOperationAction(Node->getOpcode(), Node->getValueType(0))) {
      default: assert(0 && "Unknown TRUNCATE legalization operation action!");
      case TargetLowering::Custom:
        isCustom = true;
        // FALLTHROUGH
      case TargetLowering::Legal:
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
        if (isCustom) {
          Tmp1 = TLI.LowerOperation(Result, DAG);
          if (Tmp1.getNode()) Result = Tmp1;
        }
        break;
      case TargetLowering::Expand:
        assert(Result.getValueType().isVector() && "must be vector type");
        // Unroll the truncate.  We should do better.
        Result = LegalizeOp(UnrollVectorOp(Result));
      }
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);

      // Since the result is legal, we should just be able to truncate the low
      // part of the source.
      Result = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0), Tmp1);
      break;
    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      Result = DAG.getNode(ISD::TRUNCATE, dl, Op.getValueType(), Result);
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
          if (Tmp1.getNode()) Result = Tmp1;
        }
        break;
      case TargetLowering::Promote:
        Result = PromoteLegalFP_TO_INT(Tmp1, Node->getValueType(0),
                                       Node->getOpcode() == ISD::FP_TO_SINT, 
                                       dl);
        break;
      case TargetLowering::Expand:
        if (Node->getOpcode() == ISD::FP_TO_UINT) {
          SDValue True, False;
          MVT VT =  Node->getOperand(0).getValueType();
          MVT NVT = Node->getValueType(0);
          const uint64_t zero[] = {0, 0};
          APFloat apf = APFloat(APInt(VT.getSizeInBits(), 2, zero));
          APInt x = APInt::getSignBit(NVT.getSizeInBits());
          (void)apf.convertFromAPInt(x, false, APFloat::rmNearestTiesToEven);
          Tmp2 = DAG.getConstantFP(apf, VT);
          Tmp3 = DAG.getSetCC(dl, TLI.getSetCCResultType(VT), 
                              Node->getOperand(0),
                              Tmp2, ISD::SETLT);
          True = DAG.getNode(ISD::FP_TO_SINT, dl, NVT, Node->getOperand(0));
          False = DAG.getNode(ISD::FP_TO_SINT, dl, NVT,
                              DAG.getNode(ISD::FSUB, dl, VT, 
                                          Node->getOperand(0), Tmp2));
          False = DAG.getNode(ISD::XOR, dl, NVT, False,
                              DAG.getConstant(x, NVT));
          Result = DAG.getNode(ISD::SELECT, dl, NVT, Tmp3, True, False);
          break;
        } else {
          assert(0 && "Do not know how to expand FP_TO_SINT yet!");
        }
        break;
      }
      break;
    case Expand: {
      MVT VT = Op.getValueType();
      MVT OVT = Node->getOperand(0).getValueType();
      // Convert ppcf128 to i32
      if (OVT == MVT::ppcf128 && VT == MVT::i32) {
        if (Node->getOpcode() == ISD::FP_TO_SINT) {
          Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, MVT::ppcf128, 
                               Node->getOperand(0), DAG.getValueType(MVT::f64));
          Result = DAG.getNode(ISD::FP_ROUND, dl, MVT::f64, Result, 
                               DAG.getIntPtrConstant(1));
          Result = DAG.getNode(ISD::FP_TO_SINT, dl, VT, Result);
        } else {
          const uint64_t TwoE31[] = {0x41e0000000000000LL, 0};
          APFloat apf = APFloat(APInt(128, 2, TwoE31));
          Tmp2 = DAG.getConstantFP(apf, OVT);
          //  X>=2^31 ? (int)(X-2^31)+0x80000000 : (int)X
          // FIXME: generated code sucks.
          Result = DAG.getNode(ISD::SELECT_CC, dl, VT, Node->getOperand(0), 
                               Tmp2,
                               DAG.getNode(ISD::ADD, dl, MVT::i32,
                                 DAG.getNode(ISD::FP_TO_SINT, dl, VT,
                                   DAG.getNode(ISD::FSUB, dl, OVT,
                                                 Node->getOperand(0), Tmp2)),
                                 DAG.getConstant(0x80000000, MVT::i32)),
                               DAG.getNode(ISD::FP_TO_SINT, dl, VT, 
                                           Node->getOperand(0)),
                               DAG.getCondCode(ISD::SETGE));
        }
        break;
      }
      // Convert f32 / f64 to i32 / i64 / i128.
      RTLIB::Libcall LC = (Node->getOpcode() == ISD::FP_TO_SINT) ?
        RTLIB::getFPTOSINT(OVT, VT) : RTLIB::getFPTOUINT(OVT, VT);
      assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpectd fp-to-int conversion!");
      SDValue Dummy;
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
    MVT DstVT = Op.getValueType();
    MVT SrcVT = Op.getOperand(0).getValueType();
    if (TLI.getConvertAction(SrcVT, DstVT) == TargetLowering::Expand) {
      // The only other way we can lower this is to turn it into a STORE,
      // LOAD pair, targetting a temporary location (a stack slot).
      Result = EmitStackConvert(Node->getOperand(0), SrcVT, DstVT, dl);
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
      Result = DAG.getNode(ISD::FP_EXTEND, dl, Op.getValueType(), Tmp1);
      break;
    }
    break;
  }
  case ISD::FP_ROUND: {
    MVT DstVT = Op.getValueType();
    MVT SrcVT = Op.getOperand(0).getValueType();
    if (TLI.getConvertAction(SrcVT, DstVT) == TargetLowering::Expand) {
      if (SrcVT == MVT::ppcf128) {
        SDValue Lo;
        ExpandOp(Node->getOperand(0), Lo, Result);
        // Round it the rest of the way (e.g. to f32) if needed.
        if (DstVT!=MVT::f64)
          Result = DAG.getNode(ISD::FP_ROUND, dl,
                               DstVT, Result, Op.getOperand(1));
        break;
      }
      // The only other way we can lower this is to turn it into a STORE,
      // LOAD pair, targetting a temporary location (a stack slot).
      Result = EmitStackConvert(Node->getOperand(0), DstVT, DstVT, dl);
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
      Result = DAG.getNode(ISD::FP_ROUND, dl, Op.getValueType(), Tmp1,
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
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case Promote:
      switch (Node->getOpcode()) {
      case ISD::ANY_EXTEND:
        Tmp1 = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, dl, Op.getValueType(), Tmp1);
        break;
      case ISD::ZERO_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, dl, Op.getValueType(), Result);
        Result = DAG.getZeroExtendInReg(Result, dl,
                                        Node->getOperand(0).getValueType());
        break;
      case ISD::SIGN_EXTEND:
        Result = PromoteOp(Node->getOperand(0));
        Result = DAG.getNode(ISD::ANY_EXTEND, dl, Op.getValueType(), Result);
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Result.getValueType(),
                             Result,
                          DAG.getValueType(Node->getOperand(0).getValueType()));
        break;
      }
    }
    break;
  case ISD::FP_ROUND_INREG:
  case ISD::SIGN_EXTEND_INREG: {
    Tmp1 = LegalizeOp(Node->getOperand(0));
    MVT ExtraVT = cast<VTSDNode>(Node->getOperand(1))->getVT();

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
        unsigned BitsDiff = Node->getValueType(0).getSizeInBits() -
                            ExtraVT.getSizeInBits();
        SDValue ShiftCst = DAG.getConstant(BitsDiff, TLI.getShiftAmountTy());
        Result = DAG.getNode(ISD::SHL, dl, Node->getValueType(0),
                             Node->getOperand(0), ShiftCst);
        Result = DAG.getNode(ISD::SRA, dl, Node->getValueType(0),
                             Result, ShiftCst);
      } else if (Node->getOpcode() == ISD::FP_ROUND_INREG) {
        // The only way we can lower this is to turn it into a TRUNCSTORE,
        // EXTLOAD pair, targetting a temporary location (a stack slot).

        // NOTE: there is a choice here between constantly creating new stack
        // slots and always reusing the same one.  We currently always create
        // new ones, as reuse may inhibit scheduling.
        Result = EmitStackConvert(Node->getOperand(0), ExtraVT, 
                                  Node->getValueType(0), dl);
      } else {
        assert(0 && "Unknown op");
      }
      break;
    }
    break;
  }
  case ISD::TRAMPOLINE: {
    SDValue Ops[6];
    for (unsigned i = 0; i != 6; ++i)
      Ops[i] = LegalizeOp(Node->getOperand(i));
    Result = DAG.UpdateNodeOperands(Result, Ops, 6);
    // The only option for this node is to custom lower it.
    Result = TLI.LowerOperation(Result, DAG);
    assert(Result.getNode() && "Should always custom lower!");

    // Since trampoline produces two values, make sure to remember that we
    // legalized both of them.
    Tmp1 = LegalizeOp(Result.getValue(1));
    Result = LegalizeOp(Result);
    AddLegalizedOperand(SDValue(Node, 0), Result);
    AddLegalizedOperand(SDValue(Node, 1), Tmp1);
    return Op.getResNo() ? Tmp1 : Result;
  }
  case ISD::FLT_ROUNDS_: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Legal:
      // If this operation is not supported, lower it to constant 1
      Result = DAG.getConstant(1, VT);
      break;
    }
    break;
  }
  case ISD::TRAP: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Node->getOperand(0));
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
      break;
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Expand:
      // If this operation is not supported, lower it to 'abort()' call
      Tmp1 = LegalizeOp(Node->getOperand(0));
      TargetLowering::ArgListTy Args;
      std::pair<SDValue,SDValue> CallResult =
        TLI.LowerCallTo(Tmp1, Type::VoidTy,
                        false, false, false, false, CallingConv::C, false,
                        DAG.getExternalSymbol("abort", TLI.getPointerTy()),
                        Args, DAG, dl);
      Result = CallResult.second;
      break;
    }
    break;
  }

  case ISD::SADDO:
  case ISD::SSUBO: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // FALLTHROUGH
    case TargetLowering::Legal: {
      SDValue LHS = LegalizeOp(Node->getOperand(0));
      SDValue RHS = LegalizeOp(Node->getOperand(1));

      SDValue Sum = DAG.getNode(Node->getOpcode() == ISD::SADDO ? 
                                ISD::ADD : ISD::SUB, dl, LHS.getValueType(),
                                LHS, RHS);
      MVT OType = Node->getValueType(1);

      SDValue Zero = DAG.getConstant(0, LHS.getValueType());

      //   LHSSign -> LHS >= 0
      //   RHSSign -> RHS >= 0
      //   SumSign -> Sum >= 0
      //
      //   Add:
      //   Overflow -> (LHSSign == RHSSign) && (LHSSign != SumSign)
      //   Sub:
      //   Overflow -> (LHSSign != RHSSign) && (LHSSign != SumSign)
      //
      SDValue LHSSign = DAG.getSetCC(dl, OType, LHS, Zero, ISD::SETGE);
      SDValue RHSSign = DAG.getSetCC(dl, OType, RHS, Zero, ISD::SETGE);
      SDValue SignsMatch = DAG.getSetCC(dl, OType, LHSSign, RHSSign, 
                                        Node->getOpcode() == ISD::SADDO ? 
                                        ISD::SETEQ : ISD::SETNE);

      SDValue SumSign = DAG.getSetCC(dl, OType, Sum, Zero, ISD::SETGE);
      SDValue SumSignNE = DAG.getSetCC(dl, OType, LHSSign, SumSign, ISD::SETNE);

      SDValue Cmp = DAG.getNode(ISD::AND, dl, OType, SignsMatch, SumSignNE);

      MVT ValueVTs[] = { LHS.getValueType(), OType };
      SDValue Ops[] = { Sum, Cmp };

      Result = DAG.getNode(ISD::MERGE_VALUES, dl, 
                           DAG.getVTList(&ValueVTs[0], 2),
                           &Ops[0], 2);
      SDNode *RNode = Result.getNode();
      DAG.ReplaceAllUsesOfValueWith(SDValue(Node, 0), SDValue(RNode, 0));
      DAG.ReplaceAllUsesOfValueWith(SDValue(Node, 1), SDValue(RNode, 1));
      break;
    }
    }

    break;
  }
  case ISD::UADDO:
  case ISD::USUBO: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action not supported for this op yet!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // FALLTHROUGH
    case TargetLowering::Legal: {
      SDValue LHS = LegalizeOp(Node->getOperand(0));
      SDValue RHS = LegalizeOp(Node->getOperand(1));

      SDValue Sum = DAG.getNode(Node->getOpcode() == ISD::UADDO ?
                                ISD::ADD : ISD::SUB, dl, LHS.getValueType(),
                                LHS, RHS);
      MVT OType = Node->getValueType(1);
      SDValue Cmp = DAG.getSetCC(dl, OType, Sum, LHS,
                                 Node->getOpcode () == ISD::UADDO ? 
                                 ISD::SETULT : ISD::SETUGT);

      MVT ValueVTs[] = { LHS.getValueType(), OType };
      SDValue Ops[] = { Sum, Cmp };

      Result = DAG.getNode(ISD::MERGE_VALUES, dl, 
                           DAG.getVTList(&ValueVTs[0], 2),
                           &Ops[0], 2);
      SDNode *RNode = Result.getNode();
      DAG.ReplaceAllUsesOfValueWith(SDValue(Node, 0), SDValue(RNode, 0));
      DAG.ReplaceAllUsesOfValueWith(SDValue(Node, 1), SDValue(RNode, 1));
      break;
    }
    }

    break;
  }
  case ISD::SMULO:
  case ISD::UMULO: {
    MVT VT = Node->getValueType(0);
    switch (TLI.getOperationAction(Node->getOpcode(), VT)) {
    default: assert(0 && "This action is not supported at all!");
    case TargetLowering::Custom:
      Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) break;
      // Fall Thru
    case TargetLowering::Legal:
      // FIXME: According to Hacker's Delight, this can be implemented in
      // target independent lowering, but it would be inefficient, since it
      // requires a division + a branch.
      assert(0 && "Target independent lowering is not supported for SMULO/UMULO!");	
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
SDValue SelectionDAGLegalize::PromoteOp(SDValue Op) {
  MVT VT = Op.getValueType();
  MVT NVT = TLI.getTypeToTransformTo(VT);
  assert(getTypeAction(VT) == Promote &&
         "Caller should expand or legalize operands that are not promotable!");
  assert(NVT.bitsGT(VT) && NVT.isInteger() == VT.isInteger() &&
         "Cannot promote to smaller type!");

  SDValue Tmp1, Tmp2, Tmp3;
  SDValue Result;
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();

  DenseMap<SDValue, SDValue>::iterator I = PromotedNodes.find(Op);
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
    Result = DAG.getNode(ISD::UNDEF, dl, NVT);
    break;
  case ISD::Constant:
    if (VT != MVT::i1)
      Result = DAG.getNode(ISD::SIGN_EXTEND, dl, NVT, Op);
    else
      Result = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Op);
    assert(isa<ConstantSDNode>(Result) && "Didn't constant fold zext?");
    break;
  case ISD::ConstantFP:
    Result = DAG.getNode(ISD::FP_EXTEND, dl, NVT, Op);
    assert(isa<ConstantFPSDNode>(Result) && "Didn't constant fold fp_extend?");
    break;

  case ISD::SETCC: {
    MVT VT0 = Node->getOperand(0).getValueType();
    assert(isTypeLegal(TLI.getSetCCResultType(VT0))
           && "SetCC type is not legal??");
    Result = DAG.getNode(ISD::SETCC, dl, TLI.getSetCCResultType(VT0),
                         Node->getOperand(0), Node->getOperand(1),
                         Node->getOperand(2));
    break;
  }
  case ISD::TRUNCATE:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Legal:
      Result = LegalizeOp(Node->getOperand(0));
      assert(Result.getValueType().bitsGE(NVT) &&
             "This truncation doesn't make sense!");
      if (Result.getValueType().bitsGT(NVT))    // Truncate to NVT instead of VT
        Result = DAG.getNode(ISD::TRUNCATE, dl, NVT, Result);
      break;
    case Promote:
      // The truncation is not required, because we don't guarantee anything
      // about high bits anyway.
      Result = PromoteOp(Node->getOperand(0));
      break;
    case Expand:
      ExpandOp(Node->getOperand(0), Tmp1, Tmp2);
      // Truncate the low part of the expanded value to the result type
      Result = DAG.getNode(ISD::TRUNCATE, dl, NVT, Tmp1);
    }
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
    switch (getTypeAction(Node->getOperand(0).getValueType())) {
    case Expand: assert(0 && "BUG: Smaller reg should have been promoted!");
    case Legal:
      // Input is legal?  Just do extend all the way to the larger type.
      Result = DAG.getNode(Node->getOpcode(), dl, NVT, Node->getOperand(0));
      break;
    case Promote:
      // Promote the reg if it's smaller.
      Result = PromoteOp(Node->getOperand(0));
      // The high bits are not guaranteed to be anything.  Insert an extend.
      if (Node->getOpcode() == ISD::SIGN_EXTEND)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else if (Node->getOpcode() == ISD::ZERO_EXTEND)
        Result = DAG.getZeroExtendInReg(Result, dl,
                                        Node->getOperand(0).getValueType());
      break;
    }
    break;
  case ISD::CONVERT_RNDSAT: {
    ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(Node)->getCvtCode();
    assert ((CvtCode == ISD::CVT_SS || CvtCode == ISD::CVT_SU ||
             CvtCode == ISD::CVT_US || CvtCode == ISD::CVT_UU ||
             CvtCode == ISD::CVT_SF || CvtCode == ISD::CVT_UF) &&
            "can only promote integers");
    Result = DAG.getConvertRndSat(NVT, Node->getOperand(0),
                                  Node->getOperand(1), Node->getOperand(2),
                                  Node->getOperand(3), Node->getOperand(4),
                                  CvtCode);
    break;

  }
  case ISD::BIT_CONVERT:
    Result = EmitStackConvert(Node->getOperand(0), Node->getValueType(0),
                              Node->getValueType(0), dl);
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
        Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Node->getOperand(0),
                             DAG.getValueType(VT));
      } else {
        // Just remove the truncate, it isn't affecting the value.
        Result = DAG.getNode(ISD::FP_ROUND, dl, NVT, Node->getOperand(0), 
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
      Result = DAG.getNode(Node->getOpcode(), dl, NVT, Node->getOperand(0));
      break;

    case Promote:
      Result = PromoteOp(Node->getOperand(0));
      if (Node->getOpcode() == ISD::SINT_TO_FP)
        Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Result.getValueType(),
                             Result,
                         DAG.getValueType(Node->getOperand(0).getValueType()));
      else
        Result = DAG.getZeroExtendInReg(Result, dl,
                                        Node->getOperand(0).getValueType());
      // No extra round required here.
      Result = DAG.getNode(Node->getOpcode(), dl, NVT, Result);
      break;
    case Expand:
      Result = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP, NVT,
                             Node->getOperand(0), dl);
      // Round if we cannot tolerate excess precision.
      if (NoExcessFPPrecision)
        Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
                             DAG.getValueType(VT));
      break;
    }
    break;

  case ISD::SIGN_EXTEND_INREG:
    Result = PromoteOp(Node->getOperand(0));
    Result = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Result, 
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
        !TLI.isOperationLegalOrCustom(ISD::FP_TO_UINT, NVT) &&
        (TLI.isOperationLegalOrCustom(ISD::FP_TO_SINT, NVT) ||
         TLI.getOperationAction(ISD::FP_TO_SINT, NVT)==TargetLowering::Custom)){
      Result = DAG.getNode(ISD::FP_TO_SINT, dl, NVT, Tmp1);
    } else {
      Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1);
    }
    break;

  case ISD::FABS:
  case ISD::FNEG:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1);
    // NOTE: we do not have to do any extra rounding here for
    // NoExcessFPPrecision, because we know the input will have the appropriate
    // precision, and these operations don't modify precision at all.
    break;

  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FTRUNC:
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
    Tmp1 = PromoteOp(Node->getOperand(0));
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1);
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::FPOW:
  case ISD::FPOWI: {
    // Promote f32 pow(i) to f64 pow(i).  Note that this could insert a libcall
    // directly as well, which may be better.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = Node->getOperand(1);
    if (Node->getOpcode() == ISD::FPOW)
      Tmp2 = PromoteOp(Tmp2);
    assert(Tmp1.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
                           DAG.getValueType(VT));
    break;
  }
    
  case ISD::ATOMIC_CMP_SWAP: {
    AtomicSDNode* AtomNode = cast<AtomicSDNode>(Node);
    Tmp2 = PromoteOp(Node->getOperand(2));
    Tmp3 = PromoteOp(Node->getOperand(3));
    Result = DAG.getAtomic(Node->getOpcode(), dl, AtomNode->getMemoryVT(), 
                           AtomNode->getChain(), 
                           AtomNode->getBasePtr(), Tmp2, Tmp3,
                           AtomNode->getSrcValue(),
                           AtomNode->getAlignment());
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP: {
    AtomicSDNode* AtomNode = cast<AtomicSDNode>(Node);
    Tmp2 = PromoteOp(Node->getOperand(2));
    Result = DAG.getAtomic(Node->getOpcode(), dl, AtomNode->getMemoryVT(),
                           AtomNode->getChain(), 
                           AtomNode->getBasePtr(), Tmp2,
                           AtomNode->getSrcValue(),
                           AtomNode->getAlignment());
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
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(Tmp1.getValueType() == NVT && Tmp2.getValueType() == NVT);
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    
    // Floating point operations will give excess precision that we may not be
    // able to tolerate.  If we DO allow excess precision, just leave it,
    // otherwise excise it.
    // FIXME: Why would we need to round FP ops more than integer ones?
    //     Is Round(Add(Add(A,B),C)) != Round(Add(Round(Add(A,B)), C))
    if (NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::SDIV:
  case ISD::SREM:
    // These operators require that their input be sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    if (NVT.isInteger()) {
      Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Tmp1,
                         DAG.getValueType(VT));
      Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Tmp2,
                         DAG.getValueType(VT));
    }
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);

    // Perform FP_ROUND: this is probably overly pessimistic.
    if (NVT.isFloatingPoint() && NoExcessFPPrecision)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
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
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    
    // Perform FP_ROUND: this is probably overly pessimistic.
    if (NoExcessFPPrecision && Node->getOpcode() != ISD::FCOPYSIGN)
      Result = DAG.getNode(ISD::FP_ROUND_INREG, dl, NVT, Result,
                           DAG.getValueType(VT));
    break;

  case ISD::UDIV:
  case ISD::UREM:
    // These operators require that their input be zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp2 = PromoteOp(Node->getOperand(1));
    assert(NVT.isInteger() && "Operators don't apply to FP!");
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, dl, VT);
    Tmp2 = DAG.getZeroExtendInReg(Tmp2, dl, VT);
    Result = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1, Tmp2);
    break;

  case ISD::SHL:
    Tmp1 = PromoteOp(Node->getOperand(0));
    Result = DAG.getNode(ISD::SHL, dl, NVT, Tmp1, Node->getOperand(1));
    break;
  case ISD::SRA:
    // The input value must be properly sign extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Tmp1,
                       DAG.getValueType(VT));
    Result = DAG.getNode(ISD::SRA, dl, NVT, Tmp1, Node->getOperand(1));
    break;
  case ISD::SRL:
    // The input value must be properly zero extended.
    Tmp1 = PromoteOp(Node->getOperand(0));
    Tmp1 = DAG.getZeroExtendInReg(Tmp1, dl, VT);
    Result = DAG.getNode(ISD::SRL, dl, NVT, Tmp1, Node->getOperand(1));
    break;

  case ISD::VAARG:
    Tmp1 = Node->getOperand(0);   // Get the chain.
    Tmp2 = Node->getOperand(1);   // Get the pointer.
    if (TLI.getOperationAction(ISD::VAARG, VT) == TargetLowering::Custom) {
      Tmp3 = DAG.getVAArg(VT, Tmp1, Tmp2, Node->getOperand(2));
      Result = TLI.LowerOperation(Tmp3, DAG);
    } else {
      const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
      SDValue VAList = DAG.getLoad(TLI.getPointerTy(), Tmp1, Tmp2, V, 0);
      // Increment the pointer, VAList, to the next vaarg
      Tmp3 = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), VAList, 
                         DAG.getConstant(VT.getSizeInBits()/8,
                                         TLI.getPointerTy()));
      // Store the incremented VAList to the legalized pointer
      Tmp3 = DAG.getStore(VAList.getValue(1), dl, Tmp3, Tmp2, V, 0);
      // Load the actual argument out of the pointer VAList
      Result = DAG.getExtLoad(ISD::EXTLOAD, dl, NVT, Tmp3, VAList, NULL, 0, VT);
    }
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;

  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(Node)
      ? ISD::EXTLOAD : LD->getExtensionType();
    Result = DAG.getExtLoad(ExtType, dl, NVT,
                            LD->getChain(), LD->getBasePtr(),
                            LD->getSrcValue(), LD->getSrcValueOffset(),
                            LD->getMemoryVT(),
                            LD->isVolatile(),
                            LD->getAlignment());
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Result.getValue(1)));
    break;
  }
  case ISD::SELECT: {
    Tmp2 = PromoteOp(Node->getOperand(1));   // Legalize the op0
    Tmp3 = PromoteOp(Node->getOperand(2));   // Legalize the op1

    MVT VT2 = Tmp2.getValueType();
    assert(VT2 == Tmp3.getValueType()
           && "PromoteOp SELECT: Operands 2 and 3 ValueTypes don't match");
    // Ensure that the resulting node is at least the same size as the operands'
    // value types, because we cannot assume that TLI.getSetCCValueType() is
    // constant.
    Result = DAG.getNode(ISD::SELECT, dl, VT2, Node->getOperand(0), Tmp2, Tmp3);
    break;
  }
  case ISD::SELECT_CC:
    Tmp2 = PromoteOp(Node->getOperand(2));   // True
    Tmp3 = PromoteOp(Node->getOperand(3));   // False
    Result = DAG.getNode(ISD::SELECT_CC, dl, NVT, Node->getOperand(0),
                         Node->getOperand(1), Tmp2, Tmp3, Node->getOperand(4));
    break;
  case ISD::BSWAP:
    Tmp1 = Node->getOperand(0);
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Tmp1);
    Tmp1 = DAG.getNode(ISD::BSWAP, dl, NVT, Tmp1);
    Result = DAG.getNode(ISD::SRL, dl, NVT, Tmp1,
                         DAG.getConstant(NVT.getSizeInBits() -
                                         VT.getSizeInBits(),
                                         TLI.getShiftAmountTy()));
    break;
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ:
    // Zero extend the argument
    Tmp1 = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Node->getOperand(0));
    // Perform the larger operation, then subtract if needed.
    Tmp1 = DAG.getNode(Node->getOpcode(), dl, NVT, Tmp1);
    switch(Node->getOpcode()) {
    case ISD::CTPOP:
      Result = Tmp1;
      break;
    case ISD::CTTZ:
      // if Tmp1 == sizeinbits(NVT) then Tmp1 = sizeinbits(Old VT)
      Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(Tmp1.getValueType()), Tmp1,
                          DAG.getConstant(NVT.getSizeInBits(), NVT),
                          ISD::SETEQ);
      Result = DAG.getNode(ISD::SELECT, dl, NVT, Tmp2,
                           DAG.getConstant(VT.getSizeInBits(), NVT), Tmp1);
      break;
    case ISD::CTLZ:
      //Tmp1 = Tmp1 - (sizeinbits(NVT) - sizeinbits(Old VT))
      Result = DAG.getNode(ISD::SUB, dl, NVT, Tmp1,
                           DAG.getConstant(NVT.getSizeInBits() -
                                           VT.getSizeInBits(), NVT));
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

  assert(Result.getNode() && "Didn't set a result!");

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
SDValue SelectionDAGLegalize::ExpandEXTRACT_VECTOR_ELT(SDValue Op) {
  // We know that operand #0 is the Vec vector.  If the index is a constant
  // or if the invec is a supported hardware type, we can use it.  Otherwise,
  // lower to a store then an indexed load.
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);
  DebugLoc dl = Op.getNode()->getDebugLoc();
  
  MVT TVT = Vec.getValueType();
  unsigned NumElems = TVT.getVectorNumElements();
  
  switch (TLI.getOperationAction(ISD::EXTRACT_VECTOR_ELT, TVT)) {
  default: assert(0 && "This action is not supported yet!");
  case TargetLowering::Custom: {
    Vec = LegalizeOp(Vec);
    Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
    SDValue Tmp3 = TLI.LowerOperation(Op, DAG);
    if (Tmp3.getNode())
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
  case TargetLowering::Promote:
    assert(TVT.isVector() && "not vector type");
    // fall thru to expand since vectors are by default are promote
  case TargetLowering::Expand:
    break;
  }

  if (NumElems == 1) {
    // This must be an access of the only element.  Return it.
    Op = ScalarizeVectorOp(Vec);
  } else if (!TLI.isTypeLegal(TVT) && isa<ConstantSDNode>(Idx)) {
    unsigned NumLoElts =  1 << Log2_32(NumElems-1);
    ConstantSDNode *CIdx = cast<ConstantSDNode>(Idx);
    SDValue Lo, Hi;
    SplitVectorOp(Vec, Lo, Hi);
    if (CIdx->getZExtValue() < NumLoElts) {
      Vec = Lo;
    } else {
      Vec = Hi;
      Idx = DAG.getConstant(CIdx->getZExtValue() - NumLoElts,
                            Idx.getValueType());
    }
  
    // It's now an extract from the appropriate high or low part.  Recurse.
    Op = DAG.UpdateNodeOperands(Op, Vec, Idx);
    Op = ExpandEXTRACT_VECTOR_ELT(Op);
  } else {
    // Store the value to a temporary stack slot, then LOAD the scalar
    // element back out.
    SDValue StackPtr = DAG.CreateStackTemporary(Vec.getValueType());
    SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, Vec, StackPtr, NULL, 0);

    // Add the offset to the index.
    unsigned EltSize = Op.getValueType().getSizeInBits()/8;
    Idx = DAG.getNode(ISD::MUL, dl, Idx.getValueType(), Idx,
                      DAG.getConstant(EltSize, Idx.getValueType()));

    if (Idx.getValueType().bitsGT(TLI.getPointerTy()))
      Idx = DAG.getNode(ISD::TRUNCATE, dl, TLI.getPointerTy(), Idx);
    else
      Idx = DAG.getNode(ISD::ZERO_EXTEND, dl, TLI.getPointerTy(), Idx);

    StackPtr = DAG.getNode(ISD::ADD, dl, Idx.getValueType(), Idx, StackPtr);

    Op = DAG.getLoad(Op.getValueType(), dl, Ch, StackPtr, NULL, 0);
  }
  return Op;
}

/// ExpandEXTRACT_SUBVECTOR - Expand a EXTRACT_SUBVECTOR operation.  For now
/// we assume the operation can be split if it is not already legal.
SDValue SelectionDAGLegalize::ExpandEXTRACT_SUBVECTOR(SDValue Op) {
  // We know that operand #0 is the Vec vector.  For now we assume the index
  // is a constant and that the extracted result is a supported hardware type.
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = LegalizeOp(Op.getOperand(1));
  
  unsigned NumElems = Vec.getValueType().getVectorNumElements();
  
  if (NumElems == Op.getValueType().getVectorNumElements()) {
    // This must be an access of the desired vector length.  Return it.
    return Vec;
  }

  ConstantSDNode *CIdx = cast<ConstantSDNode>(Idx);
  SDValue Lo, Hi;
  SplitVectorOp(Vec, Lo, Hi);
  if (CIdx->getZExtValue() < NumElems/2) {
    Vec = Lo;
  } else {
    Vec = Hi;
    Idx = DAG.getConstant(CIdx->getZExtValue() - NumElems/2,
                          Idx.getValueType());
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
/// LHS, and the SDValue returned in RHS has a nil SDNode value.
void SelectionDAGLegalize::LegalizeSetCCOperands(SDValue &LHS,
                                                 SDValue &RHS,
                                                 SDValue &CC,
                                                 DebugLoc dl) {
  SDValue Tmp1, Tmp2, Tmp3, Result;    
  
  switch (getTypeAction(LHS.getValueType())) {
  case Legal:
    Tmp1 = LegalizeOp(LHS);   // LHS
    Tmp2 = LegalizeOp(RHS);   // RHS
    break;
  case Promote:
    Tmp1 = PromoteOp(LHS);   // LHS
    Tmp2 = PromoteOp(RHS);   // RHS

    // If this is an FP compare, the operands have already been extended.
    if (LHS.getValueType().isInteger()) {
      MVT VT = LHS.getValueType();
      MVT NVT = TLI.getTypeToTransformTo(VT);

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
        Tmp1 = DAG.getZeroExtendInReg(Tmp1, dl, VT);
        Tmp2 = DAG.getZeroExtendInReg(Tmp2, dl, VT);
        break;
      case ISD::SETGE:
      case ISD::SETGT:
      case ISD::SETLT:
      case ISD::SETLE:
        Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Tmp1,
                           DAG.getValueType(VT));
        Tmp2 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Tmp2,
                           DAG.getValueType(VT));
        Tmp1 = LegalizeOp(Tmp1); // Relegalize new nodes.
        Tmp2 = LegalizeOp(Tmp2); // Relegalize new nodes.
        break;
      }
    }
    break;
  case Expand: {
    MVT VT = LHS.getValueType();
    if (VT == MVT::f32 || VT == MVT::f64) {
      // Expand into one or more soft-fp libcall(s).
      RTLIB::Libcall LC1 = RTLIB::UNKNOWN_LIBCALL, LC2 = RTLIB::UNKNOWN_LIBCALL;
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

      SDValue Dummy;
      SDValue Ops[2] = { LHS, RHS };
      Tmp1 = ExpandLibCall(LC1, DAG.getMergeValues(Ops, 2, dl).getNode(),
                           false /*sign irrelevant*/, Dummy);
      Tmp2 = DAG.getConstant(0, MVT::i32);
      CC = DAG.getCondCode(TLI.getCmpLibcallCC(LC1));
      if (LC2 != RTLIB::UNKNOWN_LIBCALL) {
        Tmp1 = DAG.getNode(ISD::SETCC, dl,
                           TLI.getSetCCResultType(Tmp1.getValueType()),
                           Tmp1, Tmp2, CC);
        LHS = ExpandLibCall(LC2, DAG.getMergeValues(Ops, 2, dl).getNode(),
                            false /*sign irrelevant*/, Dummy);
        Tmp2 = DAG.getNode(ISD::SETCC, dl,
                           TLI.getSetCCResultType(LHS.getValueType()), LHS,
                           Tmp2, DAG.getCondCode(TLI.getCmpLibcallCC(LC2)));
        Tmp1 = DAG.getNode(ISD::OR, dl, Tmp1.getValueType(), Tmp1, Tmp2);
        Tmp2 = SDValue();
      }
      LHS = LegalizeOp(Tmp1);
      RHS = Tmp2;
      return;
    }

    SDValue LHSLo, LHSHi, RHSLo, RHSHi;
    ExpandOp(LHS, LHSLo, LHSHi);
    ExpandOp(RHS, RHSLo, RHSHi);
    ISD::CondCode CCCode = cast<CondCodeSDNode>(CC)->get();

    if (VT==MVT::ppcf128) {
      // FIXME:  This generated code sucks.  We want to generate
      //         FCMPU crN, hi1, hi2
      //         BNE crN, L:
      //         FCMPU crN, lo1, lo2
      // The following can be improved, but not that much.
      Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                          LHSHi, RHSHi, ISD::SETOEQ);
      Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSLo.getValueType()),
                          LHSLo, RHSLo, CCCode);
      Tmp3 = DAG.getNode(ISD::AND, dl, Tmp1.getValueType(), Tmp1, Tmp2);
      Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                          LHSHi, RHSHi, ISD::SETUNE);
      Tmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                          LHSHi, RHSHi, CCCode);
      Tmp1 = DAG.getNode(ISD::AND, dl, Tmp1.getValueType(), Tmp1, Tmp2);
      Tmp1 = DAG.getNode(ISD::OR, dl, Tmp1.getValueType(), Tmp1, Tmp3);
      Tmp2 = SDValue();
      break;
    }

    switch (CCCode) {
    case ISD::SETEQ:
    case ISD::SETNE:
      if (RHSLo == RHSHi)
        if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
          if (RHSCST->isAllOnesValue()) {
            // Comparison to -1.
            Tmp1 = DAG.getNode(ISD::AND, dl,LHSLo.getValueType(), LHSLo, LHSHi);
            Tmp2 = RHSLo;
            break;
          }

      Tmp1 = DAG.getNode(ISD::XOR, dl, LHSLo.getValueType(), LHSLo, RHSLo);
      Tmp2 = DAG.getNode(ISD::XOR, dl, LHSLo.getValueType(), LHSHi, RHSHi);
      Tmp1 = DAG.getNode(ISD::OR, dl, Tmp1.getValueType(), Tmp1, Tmp2);
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
      Tmp1 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSLo.getValueType()),
                               LHSLo, RHSLo, LowCC, false, DagCombineInfo, dl);
      if (!Tmp1.getNode())
        Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSLo.getValueType()),
                            LHSLo, RHSLo, LowCC);
      Tmp2 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi.getValueType()),
                               LHSHi, RHSHi, CCCode, false, DagCombineInfo, dl);
      if (!Tmp2.getNode())
        Tmp2 = DAG.getNode(ISD::SETCC, dl,
                           TLI.getSetCCResultType(LHSHi.getValueType()),
                           LHSHi, RHSHi,CC);
      
      ConstantSDNode *Tmp1C = dyn_cast<ConstantSDNode>(Tmp1.getNode());
      ConstantSDNode *Tmp2C = dyn_cast<ConstantSDNode>(Tmp2.getNode());
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
        Tmp2 = SDValue();
      } else {
        Result = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi.getValueType()),
                                   LHSHi, RHSHi, ISD::SETEQ, false,
                                   DagCombineInfo, dl);
        if (!Result.getNode())
          Result=DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                              LHSHi, RHSHi, ISD::SETEQ);
        Result = LegalizeOp(DAG.getNode(ISD::SELECT, dl, Tmp1.getValueType(),
                                        Result, Tmp1, Tmp2));
        Tmp1 = Result;
        Tmp2 = SDValue();
      }
    }
  }
  }
  LHS = Tmp1;
  RHS = Tmp2;
}

/// LegalizeSetCCCondCode - Legalize a SETCC with given LHS and RHS and
/// condition code CC on the current target. This routine assumes LHS and rHS
/// have already been legalized by LegalizeSetCCOperands. It expands SETCC with
/// illegal condition code into AND / OR of multiple SETCC values.
void SelectionDAGLegalize::LegalizeSetCCCondCode(MVT VT,
                                                 SDValue &LHS, SDValue &RHS,
                                                 SDValue &CC,
                                                 DebugLoc dl) {
  MVT OpVT = LHS.getValueType();
  ISD::CondCode CCCode = cast<CondCodeSDNode>(CC)->get();
  switch (TLI.getCondCodeAction(CCCode, OpVT)) {
  default: assert(0 && "Unknown condition code action!");
  case TargetLowering::Legal:
    // Nothing to do.
    break;
  case TargetLowering::Expand: {
    ISD::CondCode CC1 = ISD::SETCC_INVALID, CC2 = ISD::SETCC_INVALID;
    unsigned Opc = 0;
    switch (CCCode) {
    default: assert(0 && "Don't know how to expand this condition!"); abort();
    case ISD::SETOEQ: CC1 = ISD::SETEQ; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOGT: CC1 = ISD::SETGT; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOGE: CC1 = ISD::SETGE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOLT: CC1 = ISD::SETLT; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETOLE: CC1 = ISD::SETLE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETONE: CC1 = ISD::SETNE; CC2 = ISD::SETO;  Opc = ISD::AND; break;
    case ISD::SETUEQ: CC1 = ISD::SETEQ; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUGT: CC1 = ISD::SETGT; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUGE: CC1 = ISD::SETGE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETULT: CC1 = ISD::SETLT; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETULE: CC1 = ISD::SETLE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    case ISD::SETUNE: CC1 = ISD::SETNE; CC2 = ISD::SETUO; Opc = ISD::OR;  break;
    // FIXME: Implement more expansions.
    }

    SDValue SetCC1 = DAG.getSetCC(dl, VT, LHS, RHS, CC1);
    SDValue SetCC2 = DAG.getSetCC(dl, VT, LHS, RHS, CC2);
    LHS = DAG.getNode(Opc, dl, VT, SetCC1, SetCC2);
    RHS = SDValue();
    CC  = SDValue();
    break;
  }
  }
}

/// EmitStackConvert - Emit a store/load combination to the stack.  This stores
/// SrcOp to a stack slot of type SlotVT, truncating it if needed.  It then does
/// a load from the stack slot to DestVT, extending it if needed.
/// The resultant code need not be legal.
SDValue SelectionDAGLegalize::EmitStackConvert(SDValue SrcOp,
                                               MVT SlotVT,
                                               MVT DestVT,
                                               DebugLoc dl) {
  // Create the stack frame object.
  unsigned SrcAlign = TLI.getTargetData()->getPrefTypeAlignment(
                                          SrcOp.getValueType().getTypeForMVT());
  SDValue FIPtr = DAG.CreateStackTemporary(SlotVT, SrcAlign);
  
  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(FIPtr);
  int SPFI = StackPtrFI->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(SPFI);

  unsigned SrcSize = SrcOp.getValueType().getSizeInBits();
  unsigned SlotSize = SlotVT.getSizeInBits();
  unsigned DestSize = DestVT.getSizeInBits();
  unsigned DestAlign = TLI.getTargetData()->getPrefTypeAlignment(
                                                        DestVT.getTypeForMVT());
  
  // Emit a store to the stack slot.  Use a truncstore if the input value is
  // later than DestVT.
  SDValue Store;
  
  if (SrcSize > SlotSize)
    Store = DAG.getTruncStore(DAG.getEntryNode(), dl, SrcOp, FIPtr,
                              SV, 0, SlotVT, false, SrcAlign);
  else {
    assert(SrcSize == SlotSize && "Invalid store");
    Store = DAG.getStore(DAG.getEntryNode(), dl, SrcOp, FIPtr,
                         SV, 0, false, SrcAlign);
  }
  
  // Result is a load from the stack slot.
  if (SlotSize == DestSize)
    return DAG.getLoad(DestVT, dl, Store, FIPtr, SV, 0, false, DestAlign);
  
  assert(SlotSize < DestSize && "Unknown extension!");
  return DAG.getExtLoad(ISD::EXTLOAD, dl, DestVT, Store, FIPtr, SV, 0, SlotVT,
                        false, DestAlign);
}

SDValue SelectionDAGLegalize::ExpandSCALAR_TO_VECTOR(SDNode *Node) {
  DebugLoc dl = Node->getDebugLoc();
  // Create a vector sized/aligned stack slot, store the value to element #0,
  // then load the whole vector back out.
  SDValue StackPtr = DAG.CreateStackTemporary(Node->getValueType(0));

  FrameIndexSDNode *StackPtrFI = cast<FrameIndexSDNode>(StackPtr);
  int SPFI = StackPtrFI->getIndex();

  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, Node->getOperand(0), 
                              StackPtr,
                              PseudoSourceValue::getFixedStack(SPFI), 0);
  return DAG.getLoad(Node->getValueType(0), dl, Ch, StackPtr,
                     PseudoSourceValue::getFixedStack(SPFI), 0);
}


/// ExpandBUILD_VECTOR - Expand a BUILD_VECTOR node on targets that don't
/// support the operation, but do support the resultant vector type.
SDValue SelectionDAGLegalize::ExpandBUILD_VECTOR(SDNode *Node) {
  
  // If the only non-undef value is the low element, turn this into a 
  // SCALAR_TO_VECTOR node.  If this is { X, X, X, X }, determine X.
  unsigned NumElems = Node->getNumOperands();
  bool isOnlyLowElement = true;
  SDValue SplatValue = Node->getOperand(0);
  DebugLoc dl = Node->getDebugLoc();
  
  // FIXME: it would be far nicer to change this into map<SDValue,uint64_t>
  // and use a bitmask instead of a list of elements.
  std::map<SDValue, std::vector<unsigned> > Values;
  Values[SplatValue].push_back(0);
  bool isConstant = true;
  if (!isa<ConstantFPSDNode>(SplatValue) && !isa<ConstantSDNode>(SplatValue) &&
      SplatValue.getOpcode() != ISD::UNDEF)
    isConstant = false;
  
  for (unsigned i = 1; i < NumElems; ++i) {
    SDValue V = Node->getOperand(i);
    Values[V].push_back(i);
    if (V.getOpcode() != ISD::UNDEF)
      isOnlyLowElement = false;
    if (SplatValue != V)
      SplatValue = SDValue(0,0);

    // If this isn't a constant element or an undef, we can't use a constant
    // pool load.
    if (!isa<ConstantFPSDNode>(V) && !isa<ConstantSDNode>(V) &&
        V.getOpcode() != ISD::UNDEF)
      isConstant = false;
  }
  
  if (isOnlyLowElement) {
    // If the low element is an undef too, then this whole things is an undef.
    if (Node->getOperand(0).getOpcode() == ISD::UNDEF)
      return DAG.getNode(ISD::UNDEF, dl, Node->getValueType(0));
    // Otherwise, turn this into a scalar_to_vector node.
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, Node->getValueType(0),
                       Node->getOperand(0));
  }
  
  // If all elements are constants, create a load from the constant pool.
  if (isConstant) {
    MVT VT = Node->getValueType(0);
    std::vector<Constant*> CV;
    for (unsigned i = 0, e = NumElems; i != e; ++i) {
      if (ConstantFPSDNode *V = 
          dyn_cast<ConstantFPSDNode>(Node->getOperand(i))) {
        CV.push_back(const_cast<ConstantFP *>(V->getConstantFPValue()));
      } else if (ConstantSDNode *V = 
                   dyn_cast<ConstantSDNode>(Node->getOperand(i))) {
        CV.push_back(const_cast<ConstantInt *>(V->getConstantIntValue()));
      } else {
        assert(Node->getOperand(i).getOpcode() == ISD::UNDEF);
        const Type *OpNTy = 
          Node->getOperand(0).getValueType().getTypeForMVT();
        CV.push_back(UndefValue::get(OpNTy));
      }
    }
    Constant *CP = ConstantVector::get(CV);
    SDValue CPIdx = DAG.getConstantPool(CP, TLI.getPointerTy());
    unsigned Alignment = 1 << cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
    return DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                       PseudoSourceValue::getConstantPool(), 0,
                       false, Alignment);
  }
  
  if (SplatValue.getNode()) {   // Splat of one value?
    // Build the shuffle constant vector: <0, 0, 0, 0>
    MVT MaskVT = MVT::getIntVectorWithNumElements(NumElems);
    SDValue Zero = DAG.getConstant(0, MaskVT.getVectorElementType());
    std::vector<SDValue> ZeroVec(NumElems, Zero);
    SDValue SplatMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MaskVT,
                                      &ZeroVec[0], ZeroVec.size());

    // If the target supports VECTOR_SHUFFLE and this shuffle mask, use it.
    if (isShuffleLegal(Node->getValueType(0), SplatMask)) {
      // Get the splatted value into the low element of a vector register.
      SDValue LowValVec = 
        DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, 
                    Node->getValueType(0), SplatValue);
    
      // Return shuffle(LowValVec, undef, <0,0,0,0>)
      return DAG.getNode(ISD::VECTOR_SHUFFLE, dl, 
                         Node->getValueType(0), LowValVec,
                         DAG.getNode(ISD::UNDEF, Node->getValueType(0)),
                         SplatMask);
    }
  }
  
  // If there are only two unique elements, we may be able to turn this into a
  // vector shuffle.
  if (Values.size() == 2) {
    // Get the two values in deterministic order.
    SDValue Val1 = Node->getOperand(1);
    SDValue Val2;
    std::map<SDValue, std::vector<unsigned> >::iterator MI = Values.begin();
    if (MI->first != Val1)
      Val2 = MI->first;
    else
      Val2 = (++MI)->first;
    
    // If Val1 is an undef, make sure end ends up as Val2, to ensure that our 
    // vector shuffle has the undef vector on the RHS.
    if (Val1.getOpcode() == ISD::UNDEF)
      std::swap(Val1, Val2);
    
    // Build the shuffle constant vector: e.g. <0, 4, 0, 4>
    MVT MaskVT = MVT::getIntVectorWithNumElements(NumElems);
    MVT MaskEltVT = MaskVT.getVectorElementType();
    std::vector<SDValue> MaskVec(NumElems);

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
        MaskVec[Val2Elts[i]] = DAG.getNode(ISD::UNDEF, dl, MaskEltVT);
    
    SDValue ShuffleMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MaskVT,
                                        &MaskVec[0], MaskVec.size());

    // If the target supports SCALAR_TO_VECTOR and this shuffle mask, use it.
    if (TLI.isOperationLegalOrCustom(ISD::SCALAR_TO_VECTOR,
                                     Node->getValueType(0)) &&
        isShuffleLegal(Node->getValueType(0), ShuffleMask)) {
      Val1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,Node->getValueType(0), Val1);
      Val2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,Node->getValueType(0), Val2);
      SDValue Ops[] = { Val1, Val2, ShuffleMask };

      // Return shuffle(LoValVec, HiValVec, <0,1,0,1>)
      return DAG.getNode(ISD::VECTOR_SHUFFLE, dl,Node->getValueType(0), Ops, 3);
    }
  }
  
  // Otherwise, we can't handle this case efficiently.  Allocate a sufficiently
  // aligned object on the stack, store each element into it, then load
  // the result as a vector.
  MVT VT = Node->getValueType(0);
  // Create the stack frame object.
  SDValue FIPtr = DAG.CreateStackTemporary(VT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(FI);

  // Emit a store of each element to the stack slot.
  SmallVector<SDValue, 8> Stores;
  unsigned TypeByteSize = Node->getOperand(0).getValueType().getSizeInBits()/8;
  // Store (in the right endianness) the elements to memory.
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    // Ignore undef elements.
    if (Node->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    
    unsigned Offset = TypeByteSize*i;
    
    SDValue Idx = DAG.getConstant(Offset, FIPtr.getValueType());
    Idx = DAG.getNode(ISD::ADD, dl, FIPtr.getValueType(), FIPtr, Idx);
    
    Stores.push_back(DAG.getStore(DAG.getEntryNode(), dl, Node->getOperand(i),
                                  Idx, SV, Offset));
  }
  
  SDValue StoreChain;
  if (!Stores.empty())    // Not all undef elements?
    StoreChain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                             &Stores[0], Stores.size());
  else
    StoreChain = DAG.getEntryNode();
  
  // Result is a load from the stack slot.
  return DAG.getLoad(VT, dl, StoreChain, FIPtr, SV, 0);
}

void SelectionDAGLegalize::ExpandShiftParts(unsigned NodeOp,
                                            SDValue Op, SDValue Amt,
                                            SDValue &Lo, SDValue &Hi,
                                            DebugLoc dl) {
  // Expand the subcomponents.
  SDValue LHSL, LHSH;
  ExpandOp(Op, LHSL, LHSH);

  SDValue Ops[] = { LHSL, LHSH, Amt };
  MVT VT = LHSL.getValueType();
  Lo = DAG.getNode(NodeOp, dl, DAG.getNodeValueTypes(VT, VT), 2, Ops, 3);
  Hi = Lo.getValue(1);
}


/// ExpandShift - Try to find a clever way to expand this shift operation out to
/// smaller elements.  If we can't find a way that is more efficient than a
/// libcall on this target, return false.  Otherwise, return true with the
/// low-parts expanded into Lo and Hi.
bool SelectionDAGLegalize::ExpandShift(unsigned Opc, SDValue Op,SDValue Amt,
                                       SDValue &Lo, SDValue &Hi,
                                       DebugLoc dl) {
  assert((Opc == ISD::SHL || Opc == ISD::SRA || Opc == ISD::SRL) &&
         "This is not a shift!");

  MVT NVT = TLI.getTypeToTransformTo(Op.getValueType());
  SDValue ShAmt = LegalizeOp(Amt);
  MVT ShTy = ShAmt.getValueType();
  unsigned ShBits = ShTy.getSizeInBits();
  unsigned VTBits = Op.getValueType().getSizeInBits();
  unsigned NVTBits = NVT.getSizeInBits();

  // Handle the case when Amt is an immediate.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Amt.getNode())) {
    unsigned Cst = CN->getZExtValue();
    // Expand the incoming operand to be shifted, so that we have its parts
    SDValue InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getNode(ISD::SHL, dl,
                         NVT, InL, DAG.getConstant(Cst-NVTBits,ShTy));
      } else if (Cst == NVTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = InL;
      } else {
        Lo = DAG.getNode(ISD::SHL, dl, NVT, InL, DAG.getConstant(Cst, ShTy));
        Hi = DAG.getNode(ISD::OR, dl, NVT,
           DAG.getNode(ISD::SHL, dl, NVT, InH, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SRL, dl, NVT, InL, 
                       DAG.getConstant(NVTBits-Cst, ShTy)));
      }
      return true;
    case ISD::SRL:
      if (Cst > VTBits) {
        Lo = DAG.getConstant(0, NVT);
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRL, dl, NVT, 
                         InH, DAG.getConstant(Cst-NVTBits,ShTy));
        Hi = DAG.getConstant(0, NVT);
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getConstant(0, NVT);
      } else {
        Lo = DAG.getNode(ISD::OR, dl, NVT,
           DAG.getNode(ISD::SRL, dl, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, dl, NVT, InH, 
                       DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRL, dl, NVT, InH, DAG.getConstant(Cst, ShTy));
      }
      return true;
    case ISD::SRA:
      if (Cst > VTBits) {
        Hi = Lo = DAG.getNode(ISD::SRA, dl, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst > NVTBits) {
        Lo = DAG.getNode(ISD::SRA, dl, NVT, InH,
                           DAG.getConstant(Cst-NVTBits, ShTy));
        Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else if (Cst == NVTBits) {
        Lo = InH;
        Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,
                              DAG.getConstant(NVTBits-1, ShTy));
      } else {
        Lo = DAG.getNode(ISD::OR, dl, NVT,
           DAG.getNode(ISD::SRL, dl, NVT, InL, DAG.getConstant(Cst, ShTy)),
           DAG.getNode(ISD::SHL, dl, 
                       NVT, InH, DAG.getConstant(NVTBits-Cst, ShTy)));
        Hi = DAG.getNode(ISD::SRA, dl, NVT, InH, DAG.getConstant(Cst, ShTy));
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
    Amt = DAG.getNode(ISD::AND, dl, Amt.getValueType(), Amt,
                      DAG.getConstant(~Mask, Amt.getValueType()));
    
    // Expand the incoming operand to be shifted, so that we have its parts
    SDValue InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      Lo = DAG.getConstant(0, NVT);              // Low part is zero.
      Hi = DAG.getNode(ISD::SHL, dl, NVT, InL, Amt); // High part from Lo part.
      return true;
    case ISD::SRL:
      Hi = DAG.getConstant(0, NVT);              // Hi part is zero.
      Lo = DAG.getNode(ISD::SRL, dl, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,       // Sign extend high part.
                       DAG.getConstant(NVTBits-1, Amt.getValueType()));
      Lo = DAG.getNode(ISD::SRA, dl, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    }
  }
  
  // If we know that the high bits of the shift amount are all zero, then we can
  // do this as a couple of simple shifts.
  if ((KnownZero & Mask) == Mask) {
    // Compute 32-amt.
    SDValue Amt2 = DAG.getNode(ISD::SUB, dl, Amt.getValueType(),
                                 DAG.getConstant(NVTBits, Amt.getValueType()),
                                 Amt);
    
    // Expand the incoming operand to be shifted, so that we have its parts
    SDValue InL, InH;
    ExpandOp(Op, InL, InH);
    switch(Opc) {
    case ISD::SHL:
      Lo = DAG.getNode(ISD::SHL, dl, NVT, InL, Amt);
      Hi = DAG.getNode(ISD::OR, dl, NVT,
                       DAG.getNode(ISD::SHL, dl, NVT, InH, Amt),
                       DAG.getNode(ISD::SRL, dl, NVT, InL, Amt2));
      return true;
    case ISD::SRL:
      Hi = DAG.getNode(ISD::SRL, dl, NVT, InH, Amt);
      Lo = DAG.getNode(ISD::OR, dl, NVT,
                       DAG.getNode(ISD::SRL, dl, NVT, InL, Amt),
                       DAG.getNode(ISD::SHL, dl, NVT, InH, Amt2));
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, dl, NVT, InH, Amt);
      Lo = DAG.getNode(ISD::OR, dl, NVT,
                       DAG.getNode(ISD::SRL, dl, NVT, InL, Amt),
                       DAG.getNode(ISD::SHL, dl, NVT, InH, Amt2));
      return true;
    }
  }
  
  return false;
}


// ExpandLibCall - Expand a node into a call to a libcall.  If the result value
// does not fit into a register, return the lo part and set the hi part to the
// by-reg argument.  If it does fit into a single register, return the result
// and leave the Hi part unset.
SDValue SelectionDAGLegalize::ExpandLibCall(RTLIB::Libcall LC, SDNode *Node,
                                            bool isSigned, SDValue &Hi) {
  assert(!IsLegalizingCall && "Cannot overlap legalization of calls!");
  // The input chain to this libcall is the entry node of the function. 
  // Legalizing the call will automatically add the previous call to the
  // dependence.
  SDValue InChain = DAG.getEntryNode();
  
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i) {
    MVT ArgVT = Node->getOperand(i).getValueType();
    const Type *ArgTy = ArgVT.getTypeForMVT();
    Entry.Node = Node->getOperand(i); Entry.Ty = ArgTy; 
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                         TLI.getPointerTy());

  // Splice the libcall in wherever FindInputOutputChains tells us to.
  const Type *RetTy = Node->getValueType(0).getTypeForMVT();
  std::pair<SDValue,SDValue> CallInfo =
    TLI.LowerCallTo(InChain, RetTy, isSigned, !isSigned, false, false,
                    CallingConv::C, false, Callee, Args, DAG,
                    Node->getDebugLoc());

  // Legalize the call sequence, starting with the chain.  This will advance
  // the LastCALLSEQ_END to the legalized version of the CALLSEQ_END node that
  // was added by LowerCallTo (guaranteeing proper serialization of calls).
  LegalizeOp(CallInfo.second);
  SDValue Result;
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

/// LegalizeINT_TO_FP - Legalize a [US]INT_TO_FP operation.
///
SDValue SelectionDAGLegalize::
LegalizeINT_TO_FP(SDValue Result, bool isSigned, MVT DestTy, SDValue Op,
                  DebugLoc dl) {
  bool isCustom = false;
  SDValue Tmp1;
  switch (getTypeAction(Op.getValueType())) {
  case Legal:
    switch (TLI.getOperationAction(isSigned ? ISD::SINT_TO_FP : ISD::UINT_TO_FP,
                                   Op.getValueType())) {
    default: assert(0 && "Unknown operation action!");
    case TargetLowering::Custom:
      isCustom = true;
      // FALLTHROUGH
    case TargetLowering::Legal:
      Tmp1 = LegalizeOp(Op);
      if (Result.getNode())
        Result = DAG.UpdateNodeOperands(Result, Tmp1);
      else
        Result = DAG.getNode(isSigned ? ISD::SINT_TO_FP : ISD::UINT_TO_FP, dl,
                             DestTy, Tmp1);
      if (isCustom) {
        Tmp1 = TLI.LowerOperation(Result, DAG);
        if (Tmp1.getNode()) Result = Tmp1;
      }
      break;
    case TargetLowering::Expand:
      Result = ExpandLegalINT_TO_FP(isSigned, LegalizeOp(Op), DestTy, dl);
      break;
    case TargetLowering::Promote:
      Result = PromoteLegalINT_TO_FP(LegalizeOp(Op), DestTy, isSigned, dl);
      break;
    }
    break;
  case Expand:
    Result = ExpandIntToFP(isSigned, DestTy, Op, dl) ;
    break;
  case Promote:
    Tmp1 = PromoteOp(Op);
    if (isSigned) {
      Tmp1 = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Tmp1.getValueType(),
               Tmp1, DAG.getValueType(Op.getValueType()));
    } else {
      Tmp1 = DAG.getZeroExtendInReg(Tmp1, dl, 
                                    Op.getValueType());
    }
    if (Result.getNode())
      Result = DAG.UpdateNodeOperands(Result, Tmp1);
    else
      Result = DAG.getNode(isSigned ? ISD::SINT_TO_FP : ISD::UINT_TO_FP, dl,
                           DestTy, Tmp1);
    Result = LegalizeOp(Result);  // The 'op' is not necessarily legal!
    break;
  }
  return Result;
}

/// ExpandIntToFP - Expand a [US]INT_TO_FP operation.
///
SDValue SelectionDAGLegalize::
ExpandIntToFP(bool isSigned, MVT DestTy, SDValue Source, DebugLoc dl) {
  MVT SourceVT = Source.getValueType();
  bool ExpandSource = getTypeAction(SourceVT) == Expand;

  // Expand unsupported int-to-fp vector casts by unrolling them.
  if (DestTy.isVector()) {
    if (!ExpandSource)
      return LegalizeOp(UnrollVectorOp(Source));
    MVT DestEltTy = DestTy.getVectorElementType();
    if (DestTy.getVectorNumElements() == 1) {
      SDValue Scalar = ScalarizeVectorOp(Source);
      SDValue Result = LegalizeINT_TO_FP(SDValue(), isSigned,
                                         DestEltTy, Scalar, dl);
      return DAG.getNode(ISD::BUILD_VECTOR, dl, DestTy, Result);
    }
    SDValue Lo, Hi;
    SplitVectorOp(Source, Lo, Hi);
    MVT SplitDestTy = MVT::getVectorVT(DestEltTy,
                                       DestTy.getVectorNumElements() / 2);
    SDValue LoResult = LegalizeINT_TO_FP(SDValue(), isSigned, SplitDestTy, 
                                         Lo, dl);
    SDValue HiResult = LegalizeINT_TO_FP(SDValue(), isSigned, SplitDestTy, 
                                         Hi, dl);
    return LegalizeOp(DAG.getNode(ISD::CONCAT_VECTORS, dl, DestTy, LoResult,
                                  HiResult));
  }

  // Special case for i32 source to take advantage of UINTTOFP_I32_F32, etc.
  if (!isSigned && SourceVT != MVT::i32) {
    // The integer value loaded will be incorrectly if the 'sign bit' of the
    // incoming integer is set.  To handle this, we dynamically test to see if
    // it is set, and, if so, add a fudge factor.
    SDValue Hi;
    if (ExpandSource) {
      SDValue Lo;
      ExpandOp(Source, Lo, Hi);
      Source = DAG.getNode(ISD::BUILD_PAIR, dl, SourceVT, Lo, Hi);
    } else {
      // The comparison for the sign bit will use the entire operand.
      Hi = Source;
    }

    // Check to see if the target has a custom way to lower this.  If so, use
    // it.  (Note we've already expanded the operand in this case.)
    switch (TLI.getOperationAction(ISD::UINT_TO_FP, SourceVT)) {
    default: assert(0 && "This action not implemented for this operation!");
    case TargetLowering::Legal:
    case TargetLowering::Expand:
      break;   // This case is handled below.
    case TargetLowering::Custom: {
      SDValue NV = TLI.LowerOperation(DAG.getNode(ISD::UINT_TO_FP, DestTy,
                                                    Source), DAG);
      if (NV.getNode())
        return LegalizeOp(NV);
      break;   // The target decided this was legal after all
    }
    }

    // If this is unsigned, and not supported, first perform the conversion to
    // signed, then adjust the result if the sign bit is set.
    SDValue SignedConv = ExpandIntToFP(true, DestTy, Source, dl);

    SDValue SignSet = DAG.getSetCC(dl, 
                                   TLI.getSetCCResultType(Hi.getValueType()),
                                   Hi, DAG.getConstant(0, Hi.getValueType()),
                                   ISD::SETLT);
    SDValue Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
    SDValue CstOffset = DAG.getNode(ISD::SELECT, dl, Zero.getValueType(),
                                      SignSet, Four, Zero);
    uint64_t FF = 0x5f800000ULL;
    if (TLI.isLittleEndian()) FF <<= 32;
    static Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

    SDValue CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
    unsigned Alignment = 1 << cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
    CPIdx = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), CPIdx, CstOffset);
    Alignment = std::min(Alignment, 4u);
    SDValue FudgeInReg;
    if (DestTy == MVT::f32)
      FudgeInReg = DAG.getLoad(MVT::f32, dl, DAG.getEntryNode(), CPIdx,
                               PseudoSourceValue::getConstantPool(), 0,
                               false, Alignment);
    else if (DestTy.bitsGT(MVT::f32))
      // FIXME: Avoid the extend by construction the right constantpool?
      FudgeInReg = DAG.getExtLoad(ISD::EXTLOAD, dl, DestTy, DAG.getEntryNode(),
                                  CPIdx,
                                  PseudoSourceValue::getConstantPool(), 0,
                                  MVT::f32, false, Alignment);
    else 
      assert(0 && "Unexpected conversion");

    MVT SCVT = SignedConv.getValueType();
    if (SCVT != DestTy) {
      // Destination type needs to be expanded as well. The FADD now we are
      // constructing will be expanded into a libcall.
      if (SCVT.getSizeInBits() != DestTy.getSizeInBits()) {
        assert(SCVT.getSizeInBits() * 2 == DestTy.getSizeInBits());
        SignedConv = DAG.getNode(ISD::BUILD_PAIR, dl, DestTy,
                                 SignedConv, SignedConv.getValue(1));
      }
      SignedConv = DAG.getNode(ISD::BIT_CONVERT, dl, DestTy, SignedConv);
    }
    return DAG.getNode(ISD::FADD, dl, DestTy, SignedConv, FudgeInReg);
  }

  // Check to see if the target has a custom way to lower this.  If so, use it.
  switch (TLI.getOperationAction(ISD::SINT_TO_FP, SourceVT)) {
  default: assert(0 && "This action not implemented for this operation!");
  case TargetLowering::Legal:
  case TargetLowering::Expand:
    break;   // This case is handled below.
  case TargetLowering::Custom: {
    SDValue NV = TLI.LowerOperation(DAG.getNode(ISD::SINT_TO_FP, dl, DestTy,
                                                  Source), DAG);
    if (NV.getNode())
      return LegalizeOp(NV);
    break;   // The target decided this was legal after all
  }
  }

  // Expand the source, then glue it back together for the call.  We must expand
  // the source in case it is shared (this pass of legalize must traverse it).
  if (ExpandSource) {
    SDValue SrcLo, SrcHi;
    ExpandOp(Source, SrcLo, SrcHi);
    Source = DAG.getNode(ISD::BUILD_PAIR, dl, SourceVT, SrcLo, SrcHi);
  }

  RTLIB::Libcall LC = isSigned ?
    RTLIB::getSINTTOFP(SourceVT, DestTy) :
    RTLIB::getUINTTOFP(SourceVT, DestTy);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unknown int value type");

  Source = DAG.getNode(ISD::SINT_TO_FP, dl, DestTy, Source);
  SDValue HiPart;
  SDValue Result = ExpandLibCall(LC, Source.getNode(), isSigned, HiPart);
  if (Result.getValueType() != DestTy && HiPart.getNode())
    Result = DAG.getNode(ISD::BUILD_PAIR, dl, DestTy, Result, HiPart);
  return Result;
}

/// ExpandLegalINT_TO_FP - This function is responsible for legalizing a
/// INT_TO_FP operation of the specified operand when the target requests that
/// we expand it.  At this point, we know that the result and operand types are
/// legal for the target.
SDValue SelectionDAGLegalize::ExpandLegalINT_TO_FP(bool isSigned,
                                                   SDValue Op0,
                                                   MVT DestVT,
                                                   DebugLoc dl) {
  if (Op0.getValueType() == MVT::i32) {
    // simple 32-bit [signed|unsigned] integer to float/double expansion
    
    // Get the stack frame index of a 8 byte buffer.
    SDValue StackSlot = DAG.CreateStackTemporary(MVT::f64);
    
    // word offset constant for Hi/Lo address computation
    SDValue WordOff = DAG.getConstant(sizeof(int), TLI.getPointerTy());
    // set up Hi and Lo (into buffer) address based on endian
    SDValue Hi = StackSlot;
    SDValue Lo = DAG.getNode(ISD::ADD, dl, 
                             TLI.getPointerTy(), StackSlot,WordOff);
    if (TLI.isLittleEndian())
      std::swap(Hi, Lo);
    
    // if signed map to unsigned space
    SDValue Op0Mapped;
    if (isSigned) {
      // constant used to invert sign bit (signed to unsigned mapping)
      SDValue SignBit = DAG.getConstant(0x80000000u, MVT::i32);
      Op0Mapped = DAG.getNode(ISD::XOR, dl, MVT::i32, Op0, SignBit);
    } else {
      Op0Mapped = Op0;
    }
    // store the lo of the constructed double - based on integer input
    SDValue Store1 = DAG.getStore(DAG.getEntryNode(), dl,
                                    Op0Mapped, Lo, NULL, 0);
    // initial hi portion of constructed double
    SDValue InitialHi = DAG.getConstant(0x43300000u, MVT::i32);
    // store the hi of the constructed double - biased exponent
    SDValue Store2=DAG.getStore(Store1, dl, InitialHi, Hi, NULL, 0);
    // load the constructed double
    SDValue Load = DAG.getLoad(MVT::f64, dl, Store2, StackSlot, NULL, 0);
    // FP constant to bias correct the final result
    SDValue Bias = DAG.getConstantFP(isSigned ?
                                            BitsToDouble(0x4330000080000000ULL)
                                          : BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    // subtract the bias
    SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::f64, Load, Bias);
    // final result
    SDValue Result;
    // handle final rounding
    if (DestVT == MVT::f64) {
      // do nothing
      Result = Sub;
    } else if (DestVT.bitsLT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_ROUND, dl, DestVT, Sub,
                           DAG.getIntPtrConstant(0));
    } else if (DestVT.bitsGT(MVT::f64)) {
      Result = DAG.getNode(ISD::FP_EXTEND, dl, DestVT, Sub);
    }
    return Result;
  }
  assert(!isSigned && "Legalize cannot Expand SINT_TO_FP for i64 yet");
  SDValue Tmp1 = DAG.getNode(ISD::SINT_TO_FP, dl, DestVT, Op0);

  SDValue SignSet = DAG.getSetCC(dl, TLI.getSetCCResultType(Op0.getValueType()),
                                 Op0, DAG.getConstant(0, Op0.getValueType()),
                                 ISD::SETLT);
  SDValue Zero = DAG.getIntPtrConstant(0), Four = DAG.getIntPtrConstant(4);
  SDValue CstOffset = DAG.getNode(ISD::SELECT, dl, Zero.getValueType(),
                                    SignSet, Four, Zero);

  // If the sign bit of the integer is set, the large number will be treated
  // as a negative number.  To counteract this, the dynamic code adds an
  // offset depending on the data type.
  uint64_t FF;
  switch (Op0.getValueType().getSimpleVT()) {
  default: assert(0 && "Unsupported integer type!");
  case MVT::i8 : FF = 0x43800000ULL; break;  // 2^8  (as a float)
  case MVT::i16: FF = 0x47800000ULL; break;  // 2^16 (as a float)
  case MVT::i32: FF = 0x4F800000ULL; break;  // 2^32 (as a float)
  case MVT::i64: FF = 0x5F800000ULL; break;  // 2^64 (as a float)
  }
  if (TLI.isLittleEndian()) FF <<= 32;
  static Constant *FudgeFactor = ConstantInt::get(Type::Int64Ty, FF);

  SDValue CPIdx = DAG.getConstantPool(FudgeFactor, TLI.getPointerTy());
  unsigned Alignment = 1 << cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  CPIdx = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), CPIdx, CstOffset);
  Alignment = std::min(Alignment, 4u);
  SDValue FudgeInReg;
  if (DestVT == MVT::f32)
    FudgeInReg = DAG.getLoad(MVT::f32, dl, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0,
                             false, Alignment);
  else {
    FudgeInReg =
      LegalizeOp(DAG.getExtLoad(ISD::EXTLOAD, dl, DestVT,
                                DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                MVT::f32, false, Alignment));
  }

  return DAG.getNode(ISD::FADD, dl, DestVT, Tmp1, FudgeInReg);
}

/// PromoteLegalINT_TO_FP - This function is responsible for legalizing a
/// *INT_TO_FP operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal UINT_TO_FP or SINT_TO_FP
/// operation that takes a larger input.
SDValue SelectionDAGLegalize::PromoteLegalINT_TO_FP(SDValue LegalOp,
                                                    MVT DestVT,
                                                    bool isSigned,
                                                    DebugLoc dl) {
  // First step, figure out the appropriate *INT_TO_FP operation to use.
  MVT NewInTy = LegalOp.getValueType();

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewInTy = (MVT::SimpleValueType)(NewInTy.getSimpleVT()+1);
    assert(NewInTy.isInteger() && "Ran out of possibilities!");

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
  return DAG.getNode(OpToUse, dl, DestVT,
                     DAG.getNode(isSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND,
                                 dl, NewInTy, LegalOp));
}

/// PromoteLegalFP_TO_INT - This function is responsible for legalizing a
/// FP_TO_*INT operation of the specified operand when the target requests that
/// we promote it.  At this point, we know that the result and operand types are
/// legal for the target, and that there is a legal FP_TO_UINT or FP_TO_SINT
/// operation that returns a larger result.
SDValue SelectionDAGLegalize::PromoteLegalFP_TO_INT(SDValue LegalOp,
                                                    MVT DestVT,
                                                    bool isSigned,
                                                    DebugLoc dl) {
  // First step, figure out the appropriate FP_TO*INT operation to use.
  MVT NewOutTy = DestVT;

  unsigned OpToUse = 0;

  // Scan for the appropriate larger type to use.
  while (1) {
    NewOutTy = (MVT::SimpleValueType)(NewOutTy.getSimpleVT()+1);
    assert(NewOutTy.isInteger() && "Ran out of possibilities!");

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
  SDValue Operation = DAG.getNode(OpToUse, dl, NewOutTy, LegalOp);

  // If the operation produces an invalid type, it must be custom lowered.  Use
  // the target lowering hooks to expand it.  Just keep the low part of the
  // expanded operation, we know that we're truncating anyway.
  if (getTypeAction(NewOutTy) == Expand) {
    SmallVector<SDValue, 2> Results;
    TLI.ReplaceNodeResults(Operation.getNode(), Results, DAG);
    assert(Results.size() == 1 && "Incorrect FP_TO_XINT lowering!");
    Operation = Results[0];
  }

  // Truncate the result of the extended FP_TO_*INT operation to the desired
  // size.
  return DAG.getNode(ISD::TRUNCATE, dl, DestVT, Operation);
}

/// ExpandBSWAP - Open code the operations for BSWAP of the specified operation.
///
SDValue SelectionDAGLegalize::ExpandBSWAP(SDValue Op, DebugLoc dl) {
  MVT VT = Op.getValueType();
  MVT SHVT = TLI.getShiftAmountTy();
  SDValue Tmp1, Tmp2, Tmp3, Tmp4, Tmp5, Tmp6, Tmp7, Tmp8;
  switch (VT.getSimpleVT()) {
  default: assert(0 && "Unhandled Expand type in BSWAP!"); abort();
  case MVT::i16:
    Tmp2 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    return DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
  case MVT::i32:
    Tmp4 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp3 = DAG.getNode(ISD::AND, dl, VT, Tmp3, DAG.getConstant(0xFF0000, VT));
    Tmp2 = DAG.getNode(ISD::AND, dl, VT, Tmp2, DAG.getConstant(0xFF00, VT));
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp1);
    return DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp2);
  case MVT::i64:
    Tmp8 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(40, SHVT));
    Tmp6 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp5 = DAG.getNode(ISD::SHL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp4 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(8, SHVT));
    Tmp3 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(24, SHVT));
    Tmp2 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(40, SHVT));
    Tmp1 = DAG.getNode(ISD::SRL, dl, VT, Op, DAG.getConstant(56, SHVT));
    Tmp7 = DAG.getNode(ISD::AND, dl, VT, Tmp7, DAG.getConstant(255ULL<<48, VT));
    Tmp6 = DAG.getNode(ISD::AND, dl, VT, Tmp6, DAG.getConstant(255ULL<<40, VT));
    Tmp5 = DAG.getNode(ISD::AND, dl, VT, Tmp5, DAG.getConstant(255ULL<<32, VT));
    Tmp4 = DAG.getNode(ISD::AND, dl, VT, Tmp4, DAG.getConstant(255ULL<<24, VT));
    Tmp3 = DAG.getNode(ISD::AND, dl, VT, Tmp3, DAG.getConstant(255ULL<<16, VT));
    Tmp2 = DAG.getNode(ISD::AND, dl, VT, Tmp2, DAG.getConstant(255ULL<<8 , VT));
    Tmp8 = DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp7);
    Tmp6 = DAG.getNode(ISD::OR, dl, VT, Tmp6, Tmp5);
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp3);
    Tmp2 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp1);
    Tmp8 = DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp6);
    Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp2);
    return DAG.getNode(ISD::OR, dl, VT, Tmp8, Tmp4);
  }
}

/// ExpandBitCount - Expand the specified bitcount instruction into operations.
///
SDValue SelectionDAGLegalize::ExpandBitCount(unsigned Opc, SDValue Op, 
                                             DebugLoc dl) {
  switch (Opc) {
  default: assert(0 && "Cannot expand this yet!");
  case ISD::CTPOP: {
    static const uint64_t mask[6] = {
      0x5555555555555555ULL, 0x3333333333333333ULL,
      0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
      0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
    };
    MVT VT = Op.getValueType();
    MVT ShVT = TLI.getShiftAmountTy();
    unsigned len = VT.getSizeInBits();
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      //x = (x & mask[i][len/8]) + (x >> (1 << i) & mask[i][len/8])
      unsigned EltSize = VT.isVector() ?
        VT.getVectorElementType().getSizeInBits() : len;
      SDValue Tmp2 = DAG.getConstant(APInt(EltSize, mask[i]), VT);
      SDValue Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::ADD, dl, VT, DAG.getNode(ISD::AND, VT, Op, Tmp2),
                       DAG.getNode(ISD::AND, dl, VT,
                                   DAG.getNode(ISD::SRL, dl, VT, Op, Tmp3),
                                   Tmp2));
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
    MVT VT = Op.getValueType();
    MVT ShVT = TLI.getShiftAmountTy();
    unsigned len = VT.getSizeInBits();
    for (unsigned i = 0; (1U << i) <= (len / 2); ++i) {
      SDValue Tmp3 = DAG.getConstant(1ULL << i, ShVT);
      Op = DAG.getNode(ISD::OR, dl, VT, Op, 
                       DAG.getNode(ISD::SRL, VT, Op, Tmp3));
    }
    Op = DAG.getNOT(dl, Op, VT);
    return DAG.getNode(ISD::CTPOP, dl, VT, Op);
  }
  case ISD::CTTZ: {
    // for now, we use: { return popcount(~x & (x - 1)); }
    // unless the target has ctlz but not ctpop, in which case we use:
    // { return 32 - nlz(~x & (x-1)); }
    // see also http://www.hackersdelight.org/HDcode/ntz.cc
    MVT VT = Op.getValueType();
    SDValue Tmp3 = DAG.getNode(ISD::AND, dl, VT,
                               DAG.getNOT(dl, Op, VT),
                               DAG.getNode(ISD::SUB, dl, VT, Op,
                                           DAG.getConstant(1, VT)));
    // If ISD::CTLZ is legal and CTPOP isn't, then do that instead.
    if (!TLI.isOperationLegalOrCustom(ISD::CTPOP, VT) &&
        TLI.isOperationLegalOrCustom(ISD::CTLZ, VT))
      return DAG.getNode(ISD::SUB, dl, VT,
                         DAG.getConstant(VT.getSizeInBits(), VT),
                         DAG.getNode(ISD::CTLZ, dl, VT, Tmp3));
    return DAG.getNode(ISD::CTPOP, dl, VT, Tmp3);
  }
  }
}

/// ExpandOp - Expand the specified SDValue into its two component pieces
/// Lo&Hi.  Note that the Op MUST be an expanded type.  As a result of this, the
/// LegalizedNodes map is filled in for any results that are not expanded, the
/// ExpandedNodes map is filled in for any results that are expanded, and the
/// Lo/Hi values are returned.
void SelectionDAGLegalize::ExpandOp(SDValue Op, SDValue &Lo, SDValue &Hi){
  MVT VT = Op.getValueType();
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  assert(getTypeAction(VT) == Expand && "Not an expanded type!");
  assert(((NVT.isInteger() && NVT.bitsLT(VT)) || VT.isFloatingPoint() ||
         VT.isVector()) && "Cannot expand to FP value or to larger int value!");

  // See if we already expanded it.
  DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator I
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
      SDValue SrcLo, SrcHi, Src;
      ExpandOp(Op.getOperand(0), SrcLo, SrcHi);
      Src = DAG.getNode(ISD::BUILD_PAIR, dl, VT, SrcLo, SrcHi);
      SDValue Result = TLI.LowerOperation(
        DAG.getNode(ISD::FP_ROUND_INREG, dl, VT, Src, Op.getOperand(1)), DAG);
      assert(Result.getNode()->getOpcode() == ISD::BUILD_PAIR);
      Lo = Result.getNode()->getOperand(0);
      Hi = Result.getNode()->getOperand(1);
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
    if (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue())
      return ExpandOp(Hi, Lo, Hi);
    return ExpandOp(Lo, Lo, Hi);
  case ISD::EXTRACT_VECTOR_ELT:
    // ExpandEXTRACT_VECTOR_ELT tolerates invalid result types.
    Lo  = ExpandEXTRACT_VECTOR_ELT(Op);
    return ExpandOp(Lo, Lo, Hi);
  case ISD::UNDEF:
    Lo = DAG.getNode(ISD::UNDEF, dl, NVT);
    Hi = DAG.getNode(ISD::UNDEF, dl, NVT);
    break;
  case ISD::Constant: {
    unsigned NVTBits = NVT.getSizeInBits();
    const APInt &Cst = cast<ConstantSDNode>(Node)->getAPIntValue();
    Lo = DAG.getConstant(APInt(Cst).trunc(NVTBits), NVT);
    Hi = DAG.getConstant(Cst.lshr(NVTBits).trunc(NVTBits), NVT);
    break;
  }
  case ISD::ConstantFP: {
    ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Node);
    if (CFP->getValueType(0) == MVT::ppcf128) {
      APInt api = CFP->getValueAPF().bitcastToAPInt();
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
    assert(Op.getResNo() == 0 && Node->getNumValues() == 2 &&
           Op.getValue(1).getValueType() == MVT::Other &&
           "unhandled MERGE_VALUES");
    ExpandOp(Op.getOperand(0), Lo, Hi);
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Op.getOperand(1)));
    break;
    
  case ISD::SIGN_EXTEND_INREG:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    // sext_inreg the low part if needed.
    Lo = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Lo, Node->getOperand(1));
    
    // The high part gets the sign extension from the lo-part.  This handles
    // things like sextinreg V:i64 from i8.
    Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                     DAG.getConstant(NVT.getSizeInBits()-1,
                                     TLI.getShiftAmountTy()));
    break;

  case ISD::BSWAP: {
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDValue TempLo = DAG.getNode(ISD::BSWAP, dl, NVT, Hi);
    Hi = DAG.getNode(ISD::BSWAP, dl, NVT, Lo);
    Lo = TempLo;
    break;
  }
    
  case ISD::CTPOP:
    ExpandOp(Node->getOperand(0), Lo, Hi);
    Lo = DAG.getNode(ISD::ADD, dl, NVT,      // ctpop(HL) -> ctpop(H)+ctpop(L)
                     DAG.getNode(ISD::CTPOP, dl, NVT, Lo),
                     DAG.getNode(ISD::CTPOP, dl, NVT, Hi));
    Hi = DAG.getConstant(0, NVT);
    break;

  case ISD::CTLZ: {
    // ctlz (HL) -> ctlz(H) != 32 ? ctlz(H) : (ctlz(L)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDValue BitsC = DAG.getConstant(NVT.getSizeInBits(), NVT);
    SDValue HLZ = DAG.getNode(ISD::CTLZ, dl, NVT, Hi);
    SDValue TopNotZero = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), HLZ,
                                      BitsC, ISD::SETNE);
    SDValue LowPart = DAG.getNode(ISD::CTLZ, dl, NVT, Lo);
    LowPart = DAG.getNode(ISD::ADD, dl, NVT, LowPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, dl, NVT, TopNotZero, HLZ, LowPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::CTTZ: {
    // cttz (HL) -> cttz(L) != 32 ? cttz(L) : (cttz(H)+32)
    ExpandOp(Node->getOperand(0), Lo, Hi);
    SDValue BitsC = DAG.getConstant(NVT.getSizeInBits(), NVT);
    SDValue LTZ = DAG.getNode(ISD::CTTZ, dl, NVT, Lo);
    SDValue BotNotZero = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), LTZ,
                                      BitsC, ISD::SETNE);
    SDValue HiPart = DAG.getNode(ISD::CTTZ, dl, NVT, Hi);
    HiPart = DAG.getNode(ISD::ADD, dl, NVT, HiPart, BitsC);

    Lo = DAG.getNode(ISD::SELECT, dl, NVT, BotNotZero, LTZ, HiPart);
    Hi = DAG.getConstant(0, NVT);
    break;
  }

  case ISD::VAARG: {
    SDValue Ch = Node->getOperand(0);   // Legalize the chain.
    SDValue Ptr = Node->getOperand(1);  // Legalize the pointer.
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
    SDValue Ch  = LD->getChain();    // Legalize the chain.
    SDValue Ptr = LD->getBasePtr();  // Legalize the pointer.
    ISD::LoadExtType ExtType = LD->getExtensionType();
    const Value *SV = LD->getSrcValue();
    int SVOffset = LD->getSrcValueOffset();
    unsigned Alignment = LD->getAlignment();
    bool isVolatile = LD->isVolatile();

    if (ExtType == ISD::NON_EXTLOAD) {
      Lo = DAG.getLoad(NVT, dl, Ch, Ptr, SV, SVOffset,
                       isVolatile, Alignment);
      if (VT == MVT::f32 || VT == MVT::f64) {
        // f32->i32 or f64->i64 one to one expansion.
        // Remember that we legalized the chain.
        AddLegalizedOperand(SDValue(Node, 1), LegalizeOp(Lo.getValue(1)));
        // Recursively expand the new load.
        if (getTypeAction(NVT) == Expand)
          ExpandOp(Lo, Lo, Hi);
        break;
      }

      // Increment the pointer to the other half.
      unsigned IncrementSize = Lo.getValueType().getSizeInBits()/8;
      Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                        DAG.getIntPtrConstant(IncrementSize));
      SVOffset += IncrementSize;
      Alignment = MinAlign(Alignment, IncrementSize);
      Hi = DAG.getLoad(NVT, dl, Ch, Ptr, SV, SVOffset,
                       isVolatile, Alignment);

      // Build a factor node to remember that this load is independent of the
      // other one.
      SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                                 Hi.getValue(1));

      // Remember that we legalized the chain.
      AddLegalizedOperand(Op.getValue(1), LegalizeOp(TF));
      if (TLI.isBigEndian())
        std::swap(Lo, Hi);
    } else {
      MVT EVT = LD->getMemoryVT();

      if ((VT == MVT::f64 && EVT == MVT::f32) ||
          (VT == MVT::ppcf128 && (EVT==MVT::f64 || EVT==MVT::f32))) {
        // f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
        SDValue Load = DAG.getLoad(EVT, dl, Ch, Ptr, SV,
                                     SVOffset, isVolatile, Alignment);
        // Remember that we legalized the chain.
        AddLegalizedOperand(SDValue(Node, 1), LegalizeOp(Load.getValue(1)));
        ExpandOp(DAG.getNode(ISD::FP_EXTEND, dl, VT, Load), Lo, Hi);
        break;
      }
    
      if (EVT == NVT)
        Lo = DAG.getLoad(NVT, dl, Ch, Ptr, SV,
                         SVOffset, isVolatile, Alignment);
      else
        Lo = DAG.getExtLoad(ExtType, dl, NVT, Ch, Ptr, SV,
                            SVOffset, EVT, isVolatile,
                            Alignment);
    
      // Remember that we legalized the chain.
      AddLegalizedOperand(SDValue(Node, 1), LegalizeOp(Lo.getValue(1)));

      if (ExtType == ISD::SEXTLOAD) {
        // The high part is obtained by SRA'ing all but one of the bits of the
        // lo part.
        unsigned LoSize = Lo.getValueType().getSizeInBits();
        Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                         DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
      } else if (ExtType == ISD::ZEXTLOAD) {
        // The high part is just a zero.
        Hi = DAG.getConstant(0, NVT);
      } else /* if (ExtType == ISD::EXTLOAD) */ {
        // The high part is undefined.
        Hi = DAG.getNode(ISD::UNDEF, dl, NVT);
      }
    }
    break;
  }
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {   // Simple logical operators -> two trivial pieces.
    SDValue LL, LH, RL, RH;
    ExpandOp(Node->getOperand(0), LL, LH);
    ExpandOp(Node->getOperand(1), RL, RH);
    Lo = DAG.getNode(Node->getOpcode(), dl, NVT, LL, RL);
    Hi = DAG.getNode(Node->getOpcode(), dl, NVT, LH, RH);
    break;
  }
  case ISD::SELECT: {
    SDValue LL, LH, RL, RH;
    ExpandOp(Node->getOperand(1), LL, LH);
    ExpandOp(Node->getOperand(2), RL, RH);
    if (getTypeAction(NVT) == Expand)
      NVT = TLI.getTypeToExpandTo(NVT);
    Lo = DAG.getNode(ISD::SELECT, dl, NVT, Node->getOperand(0), LL, RL);
    if (VT != MVT::f32)
      Hi = DAG.getNode(ISD::SELECT, dl, NVT, Node->getOperand(0), LH, RH);
    break;
  }
  case ISD::SELECT_CC: {
    SDValue TL, TH, FL, FH;
    ExpandOp(Node->getOperand(2), TL, TH);
    ExpandOp(Node->getOperand(3), FL, FH);
    if (getTypeAction(NVT) == Expand)
      NVT = TLI.getTypeToExpandTo(NVT);
    Lo = DAG.getNode(ISD::SELECT_CC, dl, NVT, Node->getOperand(0),
                     Node->getOperand(1), TL, FL, Node->getOperand(4));
    if (VT != MVT::f32)
      Hi = DAG.getNode(ISD::SELECT_CC, dl, NVT, Node->getOperand(0),
                       Node->getOperand(1), TH, FH, Node->getOperand(4));
    break;
  }
  case ISD::ANY_EXTEND:
    // The low part is any extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ANY_EXTEND, dl, NVT, Node->getOperand(0));
    // The high part is undefined.
    Hi = DAG.getNode(ISD::UNDEF, dl, NVT);
    break;
  case ISD::SIGN_EXTEND: {
    // The low part is just a sign extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::SIGN_EXTEND, dl, NVT, Node->getOperand(0));

    // The high part is obtained by SRA'ing all but one of the bits of the lo
    // part.
    unsigned LoSize = Lo.getValueType().getSizeInBits();
    Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                     DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
    break;
  }
  case ISD::ZERO_EXTEND:
    // The low part is just a zero extension of the input (which degenerates to
    // a copy).
    Lo = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Node->getOperand(0));

    // The high part is just a zero.
    Hi = DAG.getConstant(0, NVT);
    break;
    
  case ISD::TRUNCATE: {
    // The input value must be larger than this value.  Expand *it*.
    SDValue NewLo;
    ExpandOp(Node->getOperand(0), NewLo, Hi);
    
    // The low part is now either the right size, or it is closer.  If not the
    // right size, make an illegal truncate so we recursively expand it.
    if (NewLo.getValueType() != Node->getValueType(0))
      NewLo = DAG.getNode(ISD::TRUNCATE, dl, Node->getValueType(0), NewLo);
    ExpandOp(NewLo, Lo, Hi);
    break;
  }
    
  case ISD::BIT_CONVERT: {
    SDValue Tmp;
    if (TLI.getOperationAction(ISD::BIT_CONVERT, VT) == TargetLowering::Custom){
      // If the target wants to, allow it to lower this itself.
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "cannot expand FP!");
      case Legal:   Tmp = LegalizeOp(Node->getOperand(0)); break;
      case Promote: Tmp = PromoteOp (Node->getOperand(0)); break;
      }
      Tmp = TLI.LowerOperation(DAG.getNode(ISD::BIT_CONVERT, dl, VT, Tmp), DAG);
    }

    // f32 / f64 must be expanded to i32 / i64.
    if (VT == MVT::f32 || VT == MVT::f64) {
      Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(0));
      if (getTypeAction(NVT) == Expand)
        ExpandOp(Lo, Lo, Hi);
      break;
    }

    // If source operand will be expanded to the same type as VT, i.e.
    // i64 <- f64, i32 <- f32, expand the source operand instead.
    MVT VT0 = Node->getOperand(0).getValueType();
    if (getTypeAction(VT0) == Expand && TLI.getTypeToTransformTo(VT0) == VT) {
      ExpandOp(Node->getOperand(0), Lo, Hi);
      break;
    }

    // Turn this into a load/store pair by default.
    if (Tmp.getNode() == 0)
      Tmp = EmitStackConvert(Node->getOperand(0), VT, VT, dl);
    
    ExpandOp(Tmp, Lo, Hi);
    break;
  }

  case ISD::READCYCLECOUNTER: {
    assert(TLI.getOperationAction(ISD::READCYCLECOUNTER, VT) == 
                 TargetLowering::Custom &&
           "Must custom expand ReadCycleCounter");
    SDValue Tmp = TLI.LowerOperation(Op, DAG);
    assert(Tmp.getNode() && "Node must be custom expanded!");
    ExpandOp(Tmp.getValue(0), Lo, Hi);
    AddLegalizedOperand(SDValue(Node, 1), // Remember we legalized the chain.
                        LegalizeOp(Tmp.getValue(1)));
    break;
  }

  case ISD::ATOMIC_CMP_SWAP: {
    // This operation does not need a loop.
    SDValue Tmp = TLI.LowerOperation(Op, DAG);
    assert(Tmp.getNode() && "Node must be custom expanded!");
    ExpandOp(Tmp.getValue(0), Lo, Hi);
    AddLegalizedOperand(SDValue(Node, 1), // Remember we legalized the chain.
                        LegalizeOp(Tmp.getValue(1)));
    break;
  }

  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_SWAP: {
    // These operations require a loop to be generated.  We can't do that yet,
    // so substitute a target-dependent pseudo and expand that later.
    SDValue In2Lo, In2Hi, In2;
    ExpandOp(Op.getOperand(2), In2Lo, In2Hi);
    In2 = DAG.getNode(ISD::BUILD_PAIR, dl, VT, In2Lo, In2Hi);
    AtomicSDNode* Anode = cast<AtomicSDNode>(Node);
    SDValue Replace = 
      DAG.getAtomic(Op.getOpcode(), dl, Anode->getMemoryVT(),
                    Op.getOperand(0), Op.getOperand(1), In2,
                    Anode->getSrcValue(), Anode->getAlignment());
    SDValue Result = TLI.LowerOperation(Replace, DAG);
    ExpandOp(Result.getValue(0), Lo, Hi);
    // Remember that we legalized the chain.
    AddLegalizedOperand(SDValue(Node,1), LegalizeOp(Result.getValue(1)));
    break;
  }

    // These operators cannot be expanded directly, emit them as calls to
    // library functions.
  case ISD::FP_TO_SINT: {
    if (TLI.getOperationAction(ISD::FP_TO_SINT, VT) == TargetLowering::Custom) {
      SDValue Op;
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
      case Expand: assert(0 && "cannot expand FP!");
      case Legal:   Op = LegalizeOp(Node->getOperand(0)); break;
      case Promote: Op = PromoteOp (Node->getOperand(0)); break;
      }

      Op = TLI.LowerOperation(DAG.getNode(ISD::FP_TO_SINT, dl, VT, Op), DAG);

      // Now that the custom expander is done, expand the result, which is still
      // VT.
      if (Op.getNode()) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    RTLIB::Libcall LC = RTLIB::getFPTOSINT(Node->getOperand(0).getValueType(),
                                           VT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpected uint-to-fp conversion!");
    Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    break;
  }

  case ISD::FP_TO_UINT: {
    if (TLI.getOperationAction(ISD::FP_TO_UINT, VT) == TargetLowering::Custom) {
      SDValue Op;
      switch (getTypeAction(Node->getOperand(0).getValueType())) {
        case Expand: assert(0 && "cannot expand FP!");
        case Legal:   Op = LegalizeOp(Node->getOperand(0)); break;
        case Promote: Op = PromoteOp (Node->getOperand(0)); break;
      }
        
      Op = TLI.LowerOperation(DAG.getNode(ISD::FP_TO_UINT, dl, VT, Op), DAG);

      // Now that the custom expander is done, expand the result.
      if (Op.getNode()) {
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    RTLIB::Libcall LC = RTLIB::getFPTOUINT(Node->getOperand(0).getValueType(),
                                           VT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpected fp-to-uint conversion!");
    Lo = ExpandLibCall(LC, Node, false/*sign irrelevant*/, Hi);
    break;
  }

  case ISD::SHL: {
    // If the target wants custom lowering, do so.
    SDValue ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SHL, VT) == TargetLowering::Custom) {
      SDValue Op = DAG.getNode(ISD::SHL, dl, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.getNode()) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If ADDC/ADDE are supported and if the shift amount is a constant 1, emit 
    // this X << 1 as X+X.
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(ShiftAmt)) {
      if (ShAmt->getAPIntValue() == 1 &&
          TLI.isOperationLegalOrCustom(ISD::ADDC, NVT) && 
          TLI.isOperationLegalOrCustom(ISD::ADDE, NVT)) {
        SDValue LoOps[2], HiOps[3];
        ExpandOp(Node->getOperand(0), LoOps[0], HiOps[0]);
        SDVTList VTList = DAG.getVTList(LoOps[0].getValueType(), MVT::Flag);
        LoOps[1] = LoOps[0];
        Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);

        HiOps[1] = HiOps[0];
        HiOps[2] = Lo.getValue(1);
        Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SHL, Node->getOperand(0), ShiftAmt, Lo, Hi, dl))
      break;

    // If this target supports SHL_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SHL_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SHL_PARTS, Node->getOperand(0), 
                       ShiftAmt, Lo, Hi, dl);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall(RTLIB::SHL_I64, Node, false/*left shift=unsigned*/, Hi);
    break;
  }

  case ISD::SRA: {
    // If the target wants custom lowering, do so.
    SDValue ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SRA, VT) == TargetLowering::Custom) {
      SDValue Op = DAG.getNode(ISD::SRA, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.getNode()) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }
    
    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRA,  Node->getOperand(0), ShiftAmt, Lo, Hi, dl))
      break;

    // If this target supports SRA_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SRA_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SRA_PARTS, Node->getOperand(0), 
                       ShiftAmt, Lo, Hi, dl);
      break;
    }

    // Otherwise, emit a libcall.
    Lo = ExpandLibCall(RTLIB::SRA_I64, Node, true/*ashr is signed*/, Hi);
    break;
  }

  case ISD::SRL: {
    // If the target wants custom lowering, do so.
    SDValue ShiftAmt = LegalizeOp(Node->getOperand(1));
    if (TLI.getOperationAction(ISD::SRL, VT) == TargetLowering::Custom) {
      SDValue Op = DAG.getNode(ISD::SRL, dl, VT, Node->getOperand(0), ShiftAmt);
      Op = TLI.LowerOperation(Op, DAG);
      if (Op.getNode()) {
        // Now that the custom expander is done, expand the result, which is
        // still VT.
        ExpandOp(Op, Lo, Hi);
        break;
      }
    }

    // If we can emit an efficient shift operation, do so now.
    if (ExpandShift(ISD::SRL, Node->getOperand(0), ShiftAmt, Lo, Hi, dl))
      break;

    // If this target supports SRL_PARTS, use it.
    TargetLowering::LegalizeAction Action =
      TLI.getOperationAction(ISD::SRL_PARTS, NVT);
    if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
        Action == TargetLowering::Custom) {
      ExpandShiftParts(ISD::SRL_PARTS,
                       Node->getOperand(0), ShiftAmt, Lo, Hi, dl);
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
      SDValue Result = TLI.LowerOperation(Op, DAG);
      if (Result.getNode()) {
        ExpandOp(Result, Lo, Hi);
        break;
      }
    }
    // Expand the subcomponents.
    SDValue LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDValue LoOps[2], HiOps[3];
    LoOps[0] = LHSL;
    LoOps[1] = RHSL;
    HiOps[0] = LHSH;
    HiOps[1] = RHSH;

    //cascaded check to see if any smaller size has a a carry flag.
    unsigned OpV = Node->getOpcode() == ISD::ADD ? ISD::ADDC : ISD::SUBC;
    bool hasCarry = false;
    for (unsigned BitSize = NVT.getSizeInBits(); BitSize != 0; BitSize /= 2) {
      MVT AVT = MVT::getIntegerVT(BitSize);
      if (TLI.isOperationLegalOrCustom(OpV, AVT)) {
        hasCarry = true;
        break;
      }
    }

    if(hasCarry) {
      SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
      if (Node->getOpcode() == ISD::ADD) {
        Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);
        HiOps[2] = Lo.getValue(1);
        Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
      } else {
        Lo = DAG.getNode(ISD::SUBC, dl, VTList, LoOps, 2);
        HiOps[2] = Lo.getValue(1);
        Hi = DAG.getNode(ISD::SUBE, dl, VTList, HiOps, 3);
      }
      break;
    } else {
      if (Node->getOpcode() == ISD::ADD) {
        Lo = DAG.getNode(ISD::ADD, dl, NVT, LoOps, 2);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, HiOps, 2);
        SDValue Cmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT),
                                    Lo, LoOps[0], ISD::SETULT);
        SDValue Carry1 = DAG.getNode(ISD::SELECT, dl, NVT, Cmp1,
                                     DAG.getConstant(1, NVT), 
                                     DAG.getConstant(0, NVT));
        SDValue Cmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT),
                                    Lo, LoOps[1], ISD::SETULT);
        SDValue Carry2 = DAG.getNode(ISD::SELECT, dl, NVT, Cmp2,
                                    DAG.getConstant(1, NVT), 
                                    Carry1);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, Carry2);
      } else {
        Lo = DAG.getNode(ISD::SUB, dl, NVT, LoOps, 2);
        Hi = DAG.getNode(ISD::SUB, dl, NVT, HiOps, 2);
        SDValue Cmp = DAG.getSetCC(dl, NVT, LoOps[0], LoOps[1], ISD::SETULT);
        SDValue Borrow = DAG.getNode(ISD::SELECT, dl, NVT, Cmp,
                                     DAG.getConstant(1, NVT), 
                                     DAG.getConstant(0, NVT));
        Hi = DAG.getNode(ISD::SUB, dl, NVT, Hi, Borrow);
      }
      break;
    }
  }
    
  case ISD::ADDC:
  case ISD::SUBC: {
    // Expand the subcomponents.
    SDValue LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
    SDValue LoOps[2] = { LHSL, RHSL };
    SDValue HiOps[3] = { LHSH, RHSH };
    
    if (Node->getOpcode() == ISD::ADDC) {
      Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
    } else {
      Lo = DAG.getNode(ISD::SUBC, dl, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::SUBE, dl, VTList, HiOps, 3);
    }
    // Remember that we legalized the flag.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Hi.getValue(1)));
    break;
  }
  case ISD::ADDE:
  case ISD::SUBE: {
    // Expand the subcomponents.
    SDValue LHSL, LHSH, RHSL, RHSH;
    ExpandOp(Node->getOperand(0), LHSL, LHSH);
    ExpandOp(Node->getOperand(1), RHSL, RHSH);
    SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
    SDValue LoOps[3] = { LHSL, RHSL, Node->getOperand(2) };
    SDValue HiOps[3] = { LHSH, RHSH };
    
    Lo = DAG.getNode(Node->getOpcode(), dl, VTList, LoOps, 3);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(Node->getOpcode(), dl, VTList, HiOps, 3);
    
    // Remember that we legalized the flag.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(Hi.getValue(1)));
    break;
  }
  case ISD::MUL: {
    // If the target wants to custom expand this, let them.
    if (TLI.getOperationAction(ISD::MUL, VT) == TargetLowering::Custom) {
      SDValue New = TLI.LowerOperation(Op, DAG);
      if (New.getNode()) {
        ExpandOp(New, Lo, Hi);
        break;
      }
    }
    
    bool HasMULHS = TLI.isOperationLegalOrCustom(ISD::MULHS, NVT);
    bool HasMULHU = TLI.isOperationLegalOrCustom(ISD::MULHU, NVT);
    bool HasSMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::SMUL_LOHI, NVT);
    bool HasUMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::UMUL_LOHI, NVT);
    if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
      SDValue LL, LH, RL, RH;
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
          Lo = DAG.getNode(ISD::UMUL_LOHI, dl, DAG.getVTList(NVT, NVT), LL, RL);
          Hi = SDValue(Lo.getNode(), 1);
          break;
        }
        if (HasMULHU) {
          // We can emit a mulhu+mul.
          Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
          Hi = DAG.getNode(ISD::MULHU, dl, NVT, LL, RL);
          break;
        }
      }
      if (LHSSB > InnerBitSize && RHSSB > InnerBitSize) {
        // The input values are both sign-extended.
        if (HasSMUL_LOHI) {
          // We can emit a smul_lohi.
          Lo = DAG.getNode(ISD::SMUL_LOHI, dl, DAG.getVTList(NVT, NVT), LL, RL);
          Hi = SDValue(Lo.getNode(), 1);
          break;
        }
        if (HasMULHS) {
          // We can emit a mulhs+mul.
          Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
          Hi = DAG.getNode(ISD::MULHS, dl, NVT, LL, RL);
          break;
        }
      }
      if (HasUMUL_LOHI) {
        // Lo,Hi = umul LHS, RHS.
        SDValue UMulLOHI = DAG.getNode(ISD::UMUL_LOHI, dl,
                                         DAG.getVTList(NVT, NVT), LL, RL);
        Lo = UMulLOHI;
        Hi = UMulLOHI.getValue(1);
        RH = DAG.getNode(ISD::MUL, dl, NVT, LL, RH);
        LH = DAG.getNode(ISD::MUL, dl, NVT, LH, RL);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, RH);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, LH);
        break;
      }
      if (HasMULHU) {
        Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, dl, NVT, LL, RL);
        RH = DAG.getNode(ISD::MUL, dl, NVT, LL, RH);
        LH = DAG.getNode(ISD::MUL, dl, NVT, LH, RL);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, RH);
        Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, LH);
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
  case ISD::FP_EXTEND: {
    if (VT == MVT::ppcf128) {
      assert(Node->getOperand(0).getValueType()==MVT::f32 ||
             Node->getOperand(0).getValueType()==MVT::f64);
      const uint64_t zero = 0;
      if (Node->getOperand(0).getValueType()==MVT::f32)
        Hi = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Node->getOperand(0));
      else
        Hi = Node->getOperand(0);
      Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
      break;
    }
    RTLIB::Libcall LC = RTLIB::getFPEXT(Node->getOperand(0).getValueType(), VT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_EXTEND!");
    Lo = ExpandLibCall(LC, Node, true, Hi);
    break;
  }
  case ISD::FP_ROUND: {
    RTLIB::Libcall LC = RTLIB::getFPROUND(Node->getOperand(0).getValueType(),
                                          VT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported FP_ROUND!");
    Lo = ExpandLibCall(LC, Node, true, Hi);
    break;
  }
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS: 
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FTRUNC:
  case ISD::FFLOOR:
  case ISD::FCEIL:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FPOW:
  case ISD::FPOWI: {
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
    case ISD::FLOG:
      LC = GetFPLibCall(VT, RTLIB::LOG_F32, RTLIB::LOG_F64,
                        RTLIB::LOG_F80, RTLIB::LOG_PPCF128);
      break;
    case ISD::FLOG2:
      LC = GetFPLibCall(VT, RTLIB::LOG2_F32, RTLIB::LOG2_F64,
                        RTLIB::LOG2_F80, RTLIB::LOG2_PPCF128);
      break;
    case ISD::FLOG10:
      LC = GetFPLibCall(VT, RTLIB::LOG10_F32, RTLIB::LOG10_F64,
                        RTLIB::LOG10_F80, RTLIB::LOG10_PPCF128);
      break;
    case ISD::FEXP:
      LC = GetFPLibCall(VT, RTLIB::EXP_F32, RTLIB::EXP_F64,
                        RTLIB::EXP_F80, RTLIB::EXP_PPCF128);
      break;
    case ISD::FEXP2:
      LC = GetFPLibCall(VT, RTLIB::EXP2_F32, RTLIB::EXP2_F64,
                        RTLIB::EXP2_F80, RTLIB::EXP2_PPCF128);
      break;
    case ISD::FTRUNC:
      LC = GetFPLibCall(VT, RTLIB::TRUNC_F32, RTLIB::TRUNC_F64,
                        RTLIB::TRUNC_F80, RTLIB::TRUNC_PPCF128);
      break;
    case ISD::FFLOOR:
      LC = GetFPLibCall(VT, RTLIB::FLOOR_F32, RTLIB::FLOOR_F64,
                        RTLIB::FLOOR_F80, RTLIB::FLOOR_PPCF128);
      break;
    case ISD::FCEIL:
      LC = GetFPLibCall(VT, RTLIB::CEIL_F32, RTLIB::CEIL_F64,
                        RTLIB::CEIL_F80, RTLIB::CEIL_PPCF128);
      break;
    case ISD::FRINT:
      LC = GetFPLibCall(VT, RTLIB::RINT_F32, RTLIB::RINT_F64,
                        RTLIB::RINT_F80, RTLIB::RINT_PPCF128);
      break;
    case ISD::FNEARBYINT:
      LC = GetFPLibCall(VT, RTLIB::NEARBYINT_F32, RTLIB::NEARBYINT_F64,
                        RTLIB::NEARBYINT_F80, RTLIB::NEARBYINT_PPCF128);
      break;
    case ISD::FPOW:
      LC = GetFPLibCall(VT, RTLIB::POW_F32, RTLIB::POW_F64, RTLIB::POW_F80,
                        RTLIB::POW_PPCF128);
      break;
    case ISD::FPOWI:
      LC = GetFPLibCall(VT, RTLIB::POWI_F32, RTLIB::POWI_F64, RTLIB::POWI_F80,
                        RTLIB::POWI_PPCF128);
      break;
    default: assert(0 && "Unreachable!");
    }
    Lo = ExpandLibCall(LC, Node, false, Hi);
    break;
  }
  case ISD::FABS: {
    if (VT == MVT::ppcf128) {
      SDValue Tmp;
      ExpandOp(Node->getOperand(0), Lo, Tmp);
      Hi = DAG.getNode(ISD::FABS, dl, NVT, Tmp);
      // lo = hi==fabs(hi) ? lo : -lo;
      Lo = DAG.getNode(ISD::SELECT_CC, dl, NVT, Hi, Tmp,
                    Lo, DAG.getNode(ISD::FNEG, dl, NVT, Lo),
                    DAG.getCondCode(ISD::SETEQ));
      break;
    }
    SDValue Mask = (VT == MVT::f64)
      ? DAG.getConstantFP(BitsToDouble(~(1ULL << 63)), VT)
      : DAG.getConstantFP(BitsToFloat(~(1U << 31)), VT);
    Mask = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Mask);
    Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(0));
    Lo = DAG.getNode(ISD::AND, dl, NVT, Lo, Mask);
    if (getTypeAction(NVT) == Expand)
      ExpandOp(Lo, Lo, Hi);
    break;
  }
  case ISD::FNEG: {
    if (VT == MVT::ppcf128) {
      ExpandOp(Node->getOperand(0), Lo, Hi);
      Lo = DAG.getNode(ISD::FNEG, dl, MVT::f64, Lo);
      Hi = DAG.getNode(ISD::FNEG, dl, MVT::f64, Hi);
      break;
    }
    SDValue Mask = (VT == MVT::f64)
      ? DAG.getConstantFP(BitsToDouble(1ULL << 63), VT)
      : DAG.getConstantFP(BitsToFloat(1U << 31), VT);
    Mask = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Mask);
    Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NVT, Node->getOperand(0));
    Lo = DAG.getNode(ISD::XOR, dl, NVT, Lo, Mask);
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
    MVT SrcVT = Node->getOperand(0).getValueType();

    // Promote the operand if needed.  Do this before checking for
    // ppcf128 so conversions of i16 and i8 work.
    if (getTypeAction(SrcVT) == Promote) {
      SDValue Tmp = PromoteOp(Node->getOperand(0));
      Tmp = isSigned
        ? DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Tmp.getValueType(), Tmp,
                      DAG.getValueType(SrcVT))
        : DAG.getZeroExtendInReg(Tmp, dl, SrcVT);
      Node = DAG.UpdateNodeOperands(Op, Tmp).getNode();
      SrcVT = Node->getOperand(0).getValueType();
    }

    if (VT == MVT::ppcf128 && SrcVT == MVT::i32) {
      static const uint64_t zero = 0;
      if (isSigned) {
        Hi = LegalizeOp(DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f64, 
                                    Node->getOperand(0)));
        Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
      } else {
        static const uint64_t TwoE32[] = { 0x41f0000000000000LL, 0 };
        Hi = LegalizeOp(DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f64, 
                                    Node->getOperand(0)));
        Lo = DAG.getConstantFP(APFloat(APInt(64, 1, &zero)), MVT::f64);
        Hi = DAG.getNode(ISD::BUILD_PAIR, dl, VT, Lo, Hi);
        // X>=0 ? {(f64)x, 0} : {(f64)x, 0} + 2^32
        ExpandOp(DAG.getNode(ISD::SELECT_CC, dl,
                             MVT::ppcf128, Node->getOperand(0),
                             DAG.getConstant(0, MVT::i32), 
                             DAG.getNode(ISD::FADD, dl, MVT::ppcf128, Hi,
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
      ExpandOp(DAG.getNode(ISD::SINT_TO_FP, dl, MVT::ppcf128,
               Node->getOperand(0)), Lo, Hi);
      Hi = DAG.getNode(ISD::BUILD_PAIR, dl, VT, Lo, Hi);
      // x>=0 ? (ppcf128)(i64)x : (ppcf128)(i64)x + 2^64
      ExpandOp(DAG.getNode(ISD::SELECT_CC, dl, MVT::ppcf128,
                           Node->getOperand(0),
                           DAG.getConstant(0, MVT::i64), 
                           DAG.getNode(ISD::FADD, dl, MVT::ppcf128, Hi,
                                       DAG.getConstantFP(
                                          APFloat(APInt(128, 2, TwoE64)),
                                          MVT::ppcf128)),
                           Hi,
                           DAG.getCondCode(ISD::SETLT)),
               Lo, Hi);
      break;
    }

    Lo = ExpandIntToFP(Node->getOpcode() == ISD::SINT_TO_FP, VT,
                       Node->getOperand(0), dl);
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
    if (Hi.getNode())
      // Don't legalize the high part if it is expanded to a single node.
      Hi = LegalizeOp(Hi);
  }

  // Remember in a map if the values will be reused later.
  bool isNew =
    ExpandedNodes.insert(std::make_pair(Op, std::make_pair(Lo, Hi))).second;
  assert(isNew && "Value already expanded?!?");
  isNew = isNew;
}

/// SplitVectorOp - Given an operand of vector type, break it down into
/// two smaller values, still of vector type.
void SelectionDAGLegalize::SplitVectorOp(SDValue Op, SDValue &Lo,
                                         SDValue &Hi) {
  assert(Op.getValueType().isVector() && "Cannot split non-vector type!");
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  unsigned NumElements = Op.getValueType().getVectorNumElements();
  assert(NumElements > 1 && "Cannot split a single element vector!");

  MVT NewEltVT = Op.getValueType().getVectorElementType();

  unsigned NewNumElts_Lo = 1 << Log2_32(NumElements-1);
  unsigned NewNumElts_Hi = NumElements - NewNumElts_Lo;

  MVT NewVT_Lo = MVT::getVectorVT(NewEltVT, NewNumElts_Lo);
  MVT NewVT_Hi = MVT::getVectorVT(NewEltVT, NewNumElts_Hi);

  // See if we already split it.
  std::map<SDValue, std::pair<SDValue, SDValue> >::iterator I
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
    Lo = DAG.getNode(ISD::UNDEF, dl, NewVT_Lo);
    Hi = DAG.getNode(ISD::UNDEF, dl, NewVT_Hi);
    break;
  case ISD::BUILD_PAIR:
    Lo = Node->getOperand(0);
    Hi = Node->getOperand(1);
    break;
  case ISD::INSERT_VECTOR_ELT: {
    if (ConstantSDNode *Idx = dyn_cast<ConstantSDNode>(Node->getOperand(2))) {
      SplitVectorOp(Node->getOperand(0), Lo, Hi);
      unsigned Index = Idx->getZExtValue();
      SDValue ScalarOp = Node->getOperand(1);
      if (Index < NewNumElts_Lo)
        Lo = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, NewVT_Lo, Lo, ScalarOp,
                         DAG.getIntPtrConstant(Index));
      else
        Hi = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, NewVT_Hi, Hi, ScalarOp,
                         DAG.getIntPtrConstant(Index - NewNumElts_Lo));
      break;
    }
    SDValue Tmp = PerformInsertVectorEltInMemory(Node->getOperand(0),
                                                   Node->getOperand(1),
                                                   Node->getOperand(2), dl);
    SplitVectorOp(Tmp, Lo, Hi);
    break;
  }
  case ISD::VECTOR_SHUFFLE: {
    // Build the low part.
    SDValue Mask = Node->getOperand(2);
    SmallVector<SDValue, 8> Ops;
    MVT PtrVT = TLI.getPointerTy();
    
    // Insert all of the elements from the input that are needed.  We use 
    // buildvector of extractelement here because the input vectors will have
    // to be legalized, so this makes the code simpler.
    for (unsigned i = 0; i != NewNumElts_Lo; ++i) {
      SDValue IdxNode = Mask.getOperand(i);
      if (IdxNode.getOpcode() == ISD::UNDEF) {
        Ops.push_back(DAG.getNode(ISD::UNDEF, dl, NewEltVT));
        continue;
      }
      unsigned Idx = cast<ConstantSDNode>(IdxNode)->getZExtValue();
      SDValue InVec = Node->getOperand(0);
      if (Idx >= NumElements) {
        InVec = Node->getOperand(1);
        Idx -= NumElements;
      }
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewEltVT, InVec,
                                DAG.getConstant(Idx, PtrVT)));
    }
    Lo = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT_Lo, &Ops[0], Ops.size());
    Ops.clear();
    
    for (unsigned i = NewNumElts_Lo; i != NumElements; ++i) {
      SDValue IdxNode = Mask.getOperand(i);
      if (IdxNode.getOpcode() == ISD::UNDEF) {
        Ops.push_back(DAG.getNode(ISD::UNDEF, dl, NewEltVT));
        continue;
      }
      unsigned Idx = cast<ConstantSDNode>(IdxNode)->getZExtValue();
      SDValue InVec = Node->getOperand(0);
      if (Idx >= NumElements) {
        InVec = Node->getOperand(1);
        Idx -= NumElements;
      }
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewEltVT, InVec,
                                DAG.getConstant(Idx, PtrVT)));
    }
    Hi = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT_Hi, &Ops[0], Ops.size());
    break;
  }
  case ISD::BUILD_VECTOR: {
    SmallVector<SDValue, 8> LoOps(Node->op_begin(), 
                                    Node->op_begin()+NewNumElts_Lo);
    Lo = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT_Lo, &LoOps[0], LoOps.size());

    SmallVector<SDValue, 8> HiOps(Node->op_begin()+NewNumElts_Lo, 
                                    Node->op_end());
    Hi = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT_Hi, &HiOps[0], HiOps.size());
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // FIXME: Handle non-power-of-two vectors?
    unsigned NewNumSubvectors = Node->getNumOperands() / 2;
    if (NewNumSubvectors == 1) {
      Lo = Node->getOperand(0);
      Hi = Node->getOperand(1);
    } else {
      SmallVector<SDValue, 8> LoOps(Node->op_begin(),
                                    Node->op_begin()+NewNumSubvectors);
      Lo = DAG.getNode(ISD::CONCAT_VECTORS, dl, NewVT_Lo, 
                       &LoOps[0], LoOps.size());

      SmallVector<SDValue, 8> HiOps(Node->op_begin()+NewNumSubvectors,
                                      Node->op_end());
      Hi = DAG.getNode(ISD::CONCAT_VECTORS, dl, NewVT_Hi, 
                       &HiOps[0], HiOps.size());
    }
    break;
  }
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue Vec = Op.getOperand(0);
    SDValue Idx = Op.getOperand(1);
    MVT     IdxVT = Idx.getValueType();

    Lo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NewVT_Lo, Vec, Idx);
    ConstantSDNode *CIdx = dyn_cast<ConstantSDNode>(Idx);
    if (CIdx) {
      Hi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NewVT_Hi, Vec, 
                       DAG.getConstant(CIdx->getZExtValue() + NewNumElts_Lo,
                                       IdxVT));
    } else {
      Idx = DAG.getNode(ISD::ADD, dl, IdxVT, Idx,
                        DAG.getConstant(NewNumElts_Lo, IdxVT));
      Hi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NewVT_Hi, Vec, Idx);
    }
    break;
  }
  case ISD::SELECT: {
    SDValue Cond = Node->getOperand(0);

    SDValue LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(1), LL, LH);
    SplitVectorOp(Node->getOperand(2), RL, RH);

    if (Cond.getValueType().isVector()) {
      // Handle a vector merge.
      SDValue CL, CH;
      SplitVectorOp(Cond, CL, CH);
      Lo = DAG.getNode(Node->getOpcode(), dl, NewVT_Lo, CL, LL, RL);
      Hi = DAG.getNode(Node->getOpcode(), dl, NewVT_Hi, CH, LH, RH);
    } else {
      // Handle a simple select with vector operands.
      Lo = DAG.getNode(Node->getOpcode(), dl, NewVT_Lo, Cond, LL, RL);
      Hi = DAG.getNode(Node->getOpcode(), dl, NewVT_Hi, Cond, LH, RH);
    }
    break;
  }
  case ISD::SELECT_CC: {
    SDValue CondLHS = Node->getOperand(0);
    SDValue CondRHS = Node->getOperand(1);
    SDValue CondCode = Node->getOperand(4);
    
    SDValue LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(2), LL, LH);
    SplitVectorOp(Node->getOperand(3), RL, RH);
    
    // Handle a simple select with vector operands.
    Lo = DAG.getNode(ISD::SELECT_CC, dl, NewVT_Lo, CondLHS, CondRHS,
                     LL, RL, CondCode);
    Hi = DAG.getNode(ISD::SELECT_CC, dl, NewVT_Hi, CondLHS, CondRHS, 
                     LH, RH, CondCode);
    break;
  }
  case ISD::VSETCC: {
    SDValue LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(0), LL, LH);
    SplitVectorOp(Node->getOperand(1), RL, RH);
    Lo = DAG.getNode(ISD::VSETCC, dl, NewVT_Lo, LL, RL, Node->getOperand(2));
    Hi = DAG.getNode(ISD::VSETCC, dl, NewVT_Hi, LH, RH, Node->getOperand(2));
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
  case ISD::FREM:
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL: {
    SDValue LL, LH, RL, RH;
    SplitVectorOp(Node->getOperand(0), LL, LH);
    SplitVectorOp(Node->getOperand(1), RL, RH);
    
    Lo = DAG.getNode(Node->getOpcode(), dl, NewVT_Lo, LL, RL);
    Hi = DAG.getNode(Node->getOpcode(), dl, NewVT_Hi, LH, RH);
    break;
  }
  case ISD::FP_ROUND:
  case ISD::FPOWI: {
    SDValue L, H;
    SplitVectorOp(Node->getOperand(0), L, H);

    Lo = DAG.getNode(Node->getOpcode(), dl, NewVT_Lo, L, Node->getOperand(1));
    Hi = DAG.getNode(Node->getOpcode(), dl, NewVT_Hi, H, Node->getOperand(1));
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
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::TRUNCATE:
  case ISD::ANY_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::FP_EXTEND: {
    SDValue L, H;
    SplitVectorOp(Node->getOperand(0), L, H);

    Lo = DAG.getNode(Node->getOpcode(), dl, NewVT_Lo, L);
    Hi = DAG.getNode(Node->getOpcode(), dl, NewVT_Hi, H);
    break;
  }
  case ISD::CONVERT_RNDSAT: {
    ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(Node)->getCvtCode();
    SDValue L, H;
    SplitVectorOp(Node->getOperand(0), L, H);
    SDValue DTyOpL =  DAG.getValueType(NewVT_Lo);
    SDValue DTyOpH =  DAG.getValueType(NewVT_Hi);
    SDValue STyOpL =  DAG.getValueType(L.getValueType());
    SDValue STyOpH =  DAG.getValueType(H.getValueType());

    SDValue RndOp = Node->getOperand(3);
    SDValue SatOp = Node->getOperand(4);

    Lo = DAG.getConvertRndSat(NewVT_Lo, L, DTyOpL, STyOpL,
                              RndOp, SatOp, CvtCode);
    Hi = DAG.getConvertRndSat(NewVT_Hi, H, DTyOpH, STyOpH,
                              RndOp, SatOp, CvtCode);
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    SDValue Ch = LD->getChain();
    SDValue Ptr = LD->getBasePtr();
    ISD::LoadExtType ExtType = LD->getExtensionType();
    const Value *SV = LD->getSrcValue();
    int SVOffset = LD->getSrcValueOffset();
    MVT MemoryVT = LD->getMemoryVT();
    unsigned Alignment = LD->getAlignment();
    bool isVolatile = LD->isVolatile();

    assert(LD->isUnindexed() && "Indexed vector loads are not supported yet!");
    SDValue Offset = DAG.getNode(ISD::UNDEF, dl, Ptr.getValueType());

    MVT MemNewEltVT = MemoryVT.getVectorElementType();
    MVT MemNewVT_Lo = MVT::getVectorVT(MemNewEltVT, NewNumElts_Lo);
    MVT MemNewVT_Hi = MVT::getVectorVT(MemNewEltVT, NewNumElts_Hi);

    Lo = DAG.getLoad(ISD::UNINDEXED, dl, ExtType,
                     NewVT_Lo, Ch, Ptr, Offset,
                     SV, SVOffset, MemNewVT_Lo, isVolatile, Alignment);
    unsigned IncrementSize = NewNumElts_Lo * MemNewEltVT.getSizeInBits()/8;
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    SVOffset += IncrementSize;
    Alignment = MinAlign(Alignment, IncrementSize);
    Hi = DAG.getLoad(ISD::UNINDEXED, dl, ExtType,
                     NewVT_Hi, Ch, Ptr, Offset,
                     SV, SVOffset, MemNewVT_Hi, isVolatile, Alignment);
    
    // Build a factor node to remember that this load is independent of the
    // other one.
    SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                               Hi.getValue(1));
    
    // Remember that we legalized the chain.
    AddLegalizedOperand(Op.getValue(1), LegalizeOp(TF));
    break;
  }
  case ISD::BIT_CONVERT: {
    // We know the result is a vector.  The input may be either a vector or a
    // scalar value.
    SDValue InOp = Node->getOperand(0);
    if (!InOp.getValueType().isVector() ||
        InOp.getValueType().getVectorNumElements() == 1) {
      // The input is a scalar or single-element vector.
      // Lower to a store/load so that it can be split.
      // FIXME: this could be improved probably.
      unsigned LdAlign = TLI.getTargetData()->getPrefTypeAlignment(
                                            Op.getValueType().getTypeForMVT());
      SDValue Ptr = DAG.CreateStackTemporary(InOp.getValueType(), LdAlign);
      int FI = cast<FrameIndexSDNode>(Ptr.getNode())->getIndex();

      SDValue St = DAG.getStore(DAG.getEntryNode(), dl,
                                  InOp, Ptr,
                                  PseudoSourceValue::getFixedStack(FI), 0);
      InOp = DAG.getLoad(Op.getValueType(), dl, St, Ptr,
                         PseudoSourceValue::getFixedStack(FI), 0);
    }
    // Split the vector and convert each of the pieces now.
    SplitVectorOp(InOp, Lo, Hi);
    Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NewVT_Lo, Lo);
    Hi = DAG.getNode(ISD::BIT_CONVERT, dl, NewVT_Hi, Hi);
    break;
  }
  }
      
  // Remember in a map if the values will be reused later.
  bool isNew = 
    SplitNodes.insert(std::make_pair(Op, std::make_pair(Lo, Hi))).second;
  assert(isNew && "Value already split?!?");
  isNew = isNew;
}


/// ScalarizeVectorOp - Given an operand of single-element vector type
/// (e.g. v1f32), convert it into the equivalent operation that returns a
/// scalar (e.g. f32) value.
SDValue SelectionDAGLegalize::ScalarizeVectorOp(SDValue Op) {
  assert(Op.getValueType().isVector() && "Bad ScalarizeVectorOp invocation!");
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  MVT NewVT = Op.getValueType().getVectorElementType();
  assert(Op.getValueType().getVectorNumElements() == 1);
  
  // See if we already scalarized it.
  std::map<SDValue, SDValue>::iterator I = ScalarizedNodes.find(Op);
  if (I != ScalarizedNodes.end()) return I->second;
  
  SDValue Result;
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
    Result = DAG.getNode(Node->getOpcode(), dl,
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)),
                         ScalarizeVectorOp(Node->getOperand(1)));
    break;
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::TRUNCATE:
  case ISD::FP_EXTEND:
    Result = DAG.getNode(Node->getOpcode(), dl,
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)));
    break;
  case ISD::CONVERT_RNDSAT: {
    SDValue Op0 = ScalarizeVectorOp(Node->getOperand(0));
    Result = DAG.getConvertRndSat(NewVT, Op0,
                                  DAG.getValueType(NewVT),
                                  DAG.getValueType(Op0.getValueType()),
                                  Node->getOperand(3),
                                  Node->getOperand(4),
                                  cast<CvtRndSatSDNode>(Node)->getCvtCode());
    break;
  }
  case ISD::FPOWI:
  case ISD::FP_ROUND:
    Result = DAG.getNode(Node->getOpcode(), dl,
                         NewVT, 
                         ScalarizeVectorOp(Node->getOperand(0)),
                         Node->getOperand(1));
    break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    SDValue Ch = LegalizeOp(LD->getChain());     // Legalize the chain.
    SDValue Ptr = LegalizeOp(LD->getBasePtr());  // Legalize the pointer.
    ISD::LoadExtType ExtType = LD->getExtensionType();
    const Value *SV = LD->getSrcValue();
    int SVOffset = LD->getSrcValueOffset();
    MVT MemoryVT = LD->getMemoryVT();
    unsigned Alignment = LD->getAlignment();
    bool isVolatile = LD->isVolatile();

    assert(LD->isUnindexed() && "Indexed vector loads are not supported yet!");
    SDValue Offset = DAG.getNode(ISD::UNDEF, dl, Ptr.getValueType());
    
    Result = DAG.getLoad(ISD::UNINDEXED, dl, ExtType,
                         NewVT, Ch, Ptr, Offset, SV, SVOffset,
                         MemoryVT.getVectorElementType(),
                         isVolatile, Alignment);

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
    SDValue EltNum = Node->getOperand(2).getOperand(0);
    if (cast<ConstantSDNode>(EltNum)->getZExtValue())
      Result = ScalarizeVectorOp(Node->getOperand(1));
    else
      Result = ScalarizeVectorOp(Node->getOperand(0));
    break;
  }
  case ISD::EXTRACT_SUBVECTOR:
    Result = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewVT, 
                         Node->getOperand(0), Node->getOperand(1));
    break;
  case ISD::BIT_CONVERT: {
    SDValue Op0 = Op.getOperand(0);
    if (Op0.getValueType().getVectorNumElements() == 1)
      Op0 = ScalarizeVectorOp(Op0);
    Result = DAG.getNode(ISD::BIT_CONVERT, dl, NewVT, Op0);
    break;
  }
  case ISD::SELECT:
    Result = DAG.getNode(ISD::SELECT, dl, NewVT, Op.getOperand(0),
                         ScalarizeVectorOp(Op.getOperand(1)),
                         ScalarizeVectorOp(Op.getOperand(2)));
    break;
  case ISD::SELECT_CC:
    Result = DAG.getNode(ISD::SELECT_CC, dl, NewVT, Node->getOperand(0), 
                         Node->getOperand(1),
                         ScalarizeVectorOp(Op.getOperand(2)),
                         ScalarizeVectorOp(Op.getOperand(3)),
                         Node->getOperand(4));
    break;
  case ISD::VSETCC: {
    SDValue Op0 = ScalarizeVectorOp(Op.getOperand(0));
    SDValue Op1 = ScalarizeVectorOp(Op.getOperand(1));
    Result = DAG.getNode(ISD::SETCC, dl,
                         TLI.getSetCCResultType(Op0.getValueType()),
                         Op0, Op1, Op.getOperand(2));
    Result = DAG.getNode(ISD::SELECT, dl, NewVT, Result,
                         DAG.getConstant(-1ULL, NewVT),
                         DAG.getConstant(0ULL, NewVT));
    break;
  }
  }

  if (TLI.isTypeLegal(NewVT))
    Result = LegalizeOp(Result);
  bool isNew = ScalarizedNodes.insert(std::make_pair(Op, Result)).second;
  assert(isNew && "Value already scalarized?");
  isNew = isNew;
  return Result;
}


SDValue SelectionDAGLegalize::WidenVectorOp(SDValue Op, MVT WidenVT) {
  std::map<SDValue, SDValue>::iterator I = WidenNodes.find(Op);
  if (I != WidenNodes.end()) return I->second;
  
  MVT VT = Op.getValueType();
  assert(VT.isVector() && "Cannot widen non-vector type!");

  SDValue Result;
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  MVT EVT = VT.getVectorElementType();

  unsigned NumElts = VT.getVectorNumElements();
  unsigned NewNumElts = WidenVT.getVectorNumElements();
  assert(NewNumElts > NumElts  && "Cannot widen to smaller type!");
  assert(NewNumElts < 17);

  // When widen is called, it is assumed that it is more efficient to use a
  // wide type.  The default action is to widen to operation to a wider legal
  // vector type and then do the operation if it is legal by calling LegalizeOp
  // again.  If there is no vector equivalent, we will unroll the operation, do
  // it, and rebuild the vector.  If most of the operations are vectorizible to
  // the legal type, the resulting code will be more efficient.  If this is not
  // the case, the resulting code will preform badly as we end up generating
  // code to pack/unpack the results. It is the function that calls widen
  // that is responsible for seeing this doesn't happen.
  switch (Node->getOpcode()) {
  default: 
#ifndef NDEBUG
      Node->dump(&DAG);
#endif
      assert(0 && "Unexpected operation in WidenVectorOp!");
      break;
  case ISD::CopyFromReg:
    assert(0 && "CopyFromReg doesn't need widening!");
  case ISD::Constant:
  case ISD::ConstantFP:
    // To build a vector of these elements, clients should call BuildVector
    // and with each element instead of creating a node with a vector type
    assert(0 && "Unexpected operation in WidenVectorOp!");
  case ISD::VAARG:
    // Variable Arguments with vector types doesn't make any sense to me
    assert(0 && "Unexpected operation in WidenVectorOp!");
    break;
  case ISD::UNDEF:
    Result = DAG.getNode(ISD::UNDEF, dl, WidenVT);
    break;
  case ISD::BUILD_VECTOR: {
    // Build a vector with undefined for the new nodes
    SDValueVector NewOps(Node->op_begin(), Node->op_end());
    for (unsigned i = NumElts; i < NewNumElts; ++i) {
      NewOps.push_back(DAG.getNode(ISD::UNDEF, dl, EVT));
    }
    Result = DAG.getNode(ISD::BUILD_VECTOR, dl, WidenVT, 
                         &NewOps[0], NewOps.size());    
    break;
  }
  case ISD::INSERT_VECTOR_ELT: {
    SDValue Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    Result = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, WidenVT, Tmp1,
                         Node->getOperand(1), Node->getOperand(2));
    break;
  }
  case ISD::VECTOR_SHUFFLE: {
    SDValue Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    SDValue Tmp2 = WidenVectorOp(Node->getOperand(1), WidenVT);
    // VECTOR_SHUFFLE 3rd operand must be a constant build vector that is
    // used as permutation array. We build the vector here instead of widening
    // because we don't want to legalize and have it turned to something else.
    SDValue PermOp = Node->getOperand(2);
    SDValueVector NewOps;
    MVT PVT = PermOp.getValueType().getVectorElementType();
    for (unsigned i = 0; i < NumElts; ++i) {
      if (PermOp.getOperand(i).getOpcode() == ISD::UNDEF) {
        NewOps.push_back(PermOp.getOperand(i));
      } else {
        unsigned Idx =
          cast<ConstantSDNode>(PermOp.getOperand(i))->getZExtValue();
        if (Idx < NumElts) {
          NewOps.push_back(PermOp.getOperand(i));
        }
        else {
          NewOps.push_back(DAG.getConstant(Idx + NewNumElts - NumElts,
                                           PermOp.getOperand(i).getValueType()));
        } 
      }
    }
    for (unsigned i = NumElts; i < NewNumElts; ++i) {
      NewOps.push_back(DAG.getNode(ISD::UNDEF, dl, PVT));
    }
    
    SDValue Tmp3 = DAG.getNode(ISD::BUILD_VECTOR, dl, 
                               MVT::getVectorVT(PVT, NewOps.size()),
                               &NewOps[0], NewOps.size()); 
    
    Result = DAG.getNode(ISD::VECTOR_SHUFFLE, dl, WidenVT, Tmp1, Tmp2, Tmp3);
    break;
  }
  case ISD::LOAD: {
    // If the load widen returns true, we can use a single load for the
    // vector.  Otherwise, it is returning a token factor for multiple
    // loads.
    SDValue TFOp;
    if (LoadWidenVectorOp(Result, TFOp, Op, WidenVT))
      AddLegalizedOperand(Op.getValue(1), LegalizeOp(TFOp.getValue(1)));
    else
      AddLegalizedOperand(Op.getValue(1), LegalizeOp(TFOp.getValue(0)));
    break;
  }

  case ISD::BIT_CONVERT: {
    SDValue Tmp1 = Node->getOperand(0);
    // Converts between two different types so we need to determine
    // the correct widen type for the input operand.
    MVT InVT = Tmp1.getValueType();
    unsigned WidenSize = WidenVT.getSizeInBits();    
    if (InVT.isVector()) {
      MVT InEltVT = InVT.getVectorElementType();
      unsigned InEltSize = InEltVT.getSizeInBits();
      assert(WidenSize % InEltSize == 0 &&
             "can not widen bit convert that are not multiple of element type");
      MVT NewInWidenVT = MVT::getVectorVT(InEltVT, WidenSize / InEltSize);
      Tmp1 = WidenVectorOp(Tmp1, NewInWidenVT);
      assert(Tmp1.getValueType().getSizeInBits() == WidenVT.getSizeInBits());
      Result = DAG.getNode(ISD::BIT_CONVERT, dl, WidenVT, Tmp1);
    } else {
      // If the result size is a multiple of the input size, widen the input
      // and then convert.
      unsigned InSize = InVT.getSizeInBits();
      assert(WidenSize % InSize == 0 &&
             "can not widen bit convert that are not multiple of element type");
      unsigned NewNumElts = WidenSize / InSize;
      SmallVector<SDValue, 16> Ops(NewNumElts);
      SDValue UndefVal = DAG.getNode(ISD::UNDEF, dl, InVT);
      Ops[0] = Tmp1;
      for (unsigned i = 1; i < NewNumElts; ++i)
        Ops[i] = UndefVal;

      MVT NewInVT = MVT::getVectorVT(InVT, NewNumElts);
      Result = DAG.getNode(ISD::BUILD_VECTOR, dl, NewInVT, &Ops[0], NewNumElts);
      Result = DAG.getNode(ISD::BIT_CONVERT, dl, WidenVT, Result);
    }
    break;
  }

  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::FP_ROUND: {
    SDValue Tmp1 = Node->getOperand(0);
    // Converts between two different types so we need to determine
    // the correct widen type for the input operand.
    MVT TVT = Tmp1.getValueType();
    assert(TVT.isVector() && "can not widen non vector type");
    MVT TEVT = TVT.getVectorElementType();
    MVT TWidenVT =  MVT::getVectorVT(TEVT, NewNumElts);
    Tmp1 = WidenVectorOp(Tmp1, TWidenVT);
    assert(Tmp1.getValueType().getVectorNumElements() == NewNumElts);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Tmp1);
    break;
  }

  case ISD::FP_EXTEND:
    assert(0 && "Case not implemented.  Dynamically dead with 2 FP types!");
  case ISD::TRUNCATE:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::SIGN_EXTEND_INREG:
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::CTPOP:
  case ISD::CTTZ:
  case ISD::CTLZ: {
    // Unary op widening
    SDValue Tmp1;
    Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    assert(Tmp1.getValueType() == WidenVT);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Tmp1);
    break;
  }
  case ISD::CONVERT_RNDSAT: {
    SDValue RndOp = Node->getOperand(3);
    SDValue SatOp = Node->getOperand(4);
    SDValue SrcOp = Node->getOperand(0);

    // Converts between two different types so we need to determine
    // the correct widen type for the input operand.
    MVT SVT = SrcOp.getValueType();
    assert(SVT.isVector() && "can not widen non vector type");
    MVT SEVT = SVT.getVectorElementType();
    MVT SWidenVT =  MVT::getVectorVT(SEVT, NewNumElts);

    SrcOp = WidenVectorOp(SrcOp, SWidenVT);
    assert(SrcOp.getValueType() == WidenVT);
    SDValue DTyOp = DAG.getValueType(WidenVT);
    SDValue STyOp = DAG.getValueType(SrcOp.getValueType());
    ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(Node)->getCvtCode();

    Result = DAG.getConvertRndSat(WidenVT, SrcOp, DTyOp, STyOp,
                                  RndOp, SatOp, CvtCode);
    break;
  }
  case ISD::FPOW:
  case ISD::FPOWI: 
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::SREM:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::FCOPYSIGN:
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::BSWAP: {
    // Binary op widening
    SDValue Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    SDValue Tmp2 = WidenVectorOp(Node->getOperand(1), WidenVT);
    assert(Tmp1.getValueType() == WidenVT && Tmp2.getValueType() == WidenVT);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Tmp1, Tmp2);
    break;
  }

  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL: {
    SDValue Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    assert(Tmp1.getValueType() == WidenVT);
    SDValue ShOp = Node->getOperand(1);
    MVT ShVT = ShOp.getValueType();
    MVT NewShVT = MVT::getVectorVT(ShVT.getVectorElementType(),
                                   WidenVT.getVectorNumElements());
    ShOp = WidenVectorOp(ShOp, NewShVT);
    assert(ShOp.getValueType() == NewShVT);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Tmp1, ShOp);
    break;
  }

  case ISD::EXTRACT_VECTOR_ELT: {
    SDValue Tmp1 = WidenVectorOp(Node->getOperand(0), WidenVT);
    assert(Tmp1.getValueType() == WidenVT);
    Result = DAG.getNode(Node->getOpcode(), dl, EVT, Tmp1, Node->getOperand(1));
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // We concurrently support only widen on a multiple of the incoming vector.
    // We could widen on a multiple of the incoming operand if necessary.
    unsigned NumConcat = NewNumElts / NumElts;
    assert(NewNumElts % NumElts == 0 && "Can widen only a multiple of vector");
    SDValue UndefVal = DAG.getNode(ISD::UNDEF, dl, VT);
    SmallVector<SDValue, 8> MOps;
    MOps.push_back(Op);
    for (unsigned i = 1; i != NumConcat; ++i) {
      MOps.push_back(UndefVal);
    }
    Result = LegalizeOp(DAG.getNode(ISD::CONCAT_VECTORS, dl, WidenVT,
                                    &MOps[0], MOps.size()));
    break;
  }
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue Tmp1 = Node->getOperand(0);
    SDValue Idx = Node->getOperand(1);
    ConstantSDNode *CIdx = dyn_cast<ConstantSDNode>(Idx);
    if (CIdx && CIdx->getZExtValue() == 0) {
      // Since we are access the start of the vector, the incoming
      // vector type might be the proper.
      MVT Tmp1VT = Tmp1.getValueType();
      if (Tmp1VT == WidenVT)
        return Tmp1;
      else {
        unsigned Tmp1VTNumElts = Tmp1VT.getVectorNumElements();
        if (Tmp1VTNumElts < NewNumElts)
          Result = WidenVectorOp(Tmp1, WidenVT);
        else
          Result = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, WidenVT, Tmp1, Idx);
      }
    } else if (NewNumElts % NumElts == 0) {
      // Widen the extracted subvector.
      unsigned NumConcat = NewNumElts / NumElts;
      SDValue UndefVal = DAG.getNode(ISD::UNDEF, dl, VT);
      SmallVector<SDValue, 8> MOps;
      MOps.push_back(Op);
      for (unsigned i = 1; i != NumConcat; ++i) {
        MOps.push_back(UndefVal);
      }
      Result = LegalizeOp(DAG.getNode(ISD::CONCAT_VECTORS, dl, WidenVT,
                                      &MOps[0], MOps.size()));
    } else {
      assert(0 && "can not widen extract subvector");
     // This could be implemented using insert and build vector but I would
     // like to see when this happens.
    }
    break;
  }

  case ISD::SELECT: {
    // Determine new condition widen type and widen
    SDValue Cond1 = Node->getOperand(0);
    MVT CondVT = Cond1.getValueType();
    assert(CondVT.isVector() && "can not widen non vector type");
    MVT CondEVT = CondVT.getVectorElementType();
    MVT CondWidenVT =  MVT::getVectorVT(CondEVT, NewNumElts);
    Cond1 = WidenVectorOp(Cond1, CondWidenVT);
    assert(Cond1.getValueType() == CondWidenVT && "Condition not widen");

    SDValue Tmp1 = WidenVectorOp(Node->getOperand(1), WidenVT);
    SDValue Tmp2 = WidenVectorOp(Node->getOperand(2), WidenVT);
    assert(Tmp1.getValueType() == WidenVT && Tmp2.getValueType() == WidenVT);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Cond1, Tmp1, Tmp2);
    break;
  }
  
  case ISD::SELECT_CC: {
    // Determine new condition widen type and widen
    SDValue Cond1 = Node->getOperand(0);
    SDValue Cond2 = Node->getOperand(1);
    MVT CondVT = Cond1.getValueType();
    assert(CondVT.isVector() && "can not widen non vector type");
    assert(CondVT == Cond2.getValueType() && "mismatch lhs/rhs");
    MVT CondEVT = CondVT.getVectorElementType();
    MVT CondWidenVT =  MVT::getVectorVT(CondEVT, NewNumElts);
    Cond1 = WidenVectorOp(Cond1, CondWidenVT);
    Cond2 = WidenVectorOp(Cond2, CondWidenVT);
    assert(Cond1.getValueType() == CondWidenVT &&
           Cond2.getValueType() == CondWidenVT && "condition not widen");

    SDValue Tmp1 = WidenVectorOp(Node->getOperand(2), WidenVT);
    SDValue Tmp2 = WidenVectorOp(Node->getOperand(3), WidenVT);
    assert(Tmp1.getValueType() == WidenVT && Tmp2.getValueType() == WidenVT &&
           "operands not widen");
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Cond1, Cond2, Tmp1,
                         Tmp2, Node->getOperand(4));
    break;
  }
  case ISD::VSETCC: {
    // Determine widen for the operand
    SDValue Tmp1 = Node->getOperand(0);
    MVT TmpVT = Tmp1.getValueType();
    assert(TmpVT.isVector() && "can not widen non vector type");
    MVT TmpEVT = TmpVT.getVectorElementType();
    MVT TmpWidenVT =  MVT::getVectorVT(TmpEVT, NewNumElts);
    Tmp1 = WidenVectorOp(Tmp1, TmpWidenVT);
    SDValue Tmp2 = WidenVectorOp(Node->getOperand(1), TmpWidenVT);
    Result = DAG.getNode(Node->getOpcode(), dl, WidenVT, Tmp1, Tmp2,
                         Node->getOperand(2));
    break;
  }
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP: {
    // For now, we assume that using vectors for these operations don't make
    // much sense so we just split it.  We return an empty result
    SDValue X, Y;
    SplitVectorOp(Op, X, Y);
    return Result;
    break;
  }

  } // end switch (Node->getOpcode())

  assert(Result.getNode() && "Didn't set a result!");  
  if (Result != Op)
    Result = LegalizeOp(Result);

  AddWidenedOperand(Op, Result);
  return Result;
}

// Utility function to find a legal vector type and its associated element
// type from a preferred width and whose vector type must be the same size
// as the VVT.
//  TLI:   Target lowering used to determine legal types
//  Width: Preferred width of element type
//  VVT:   Vector value type whose size we must match.
// Returns VecEVT and EVT - the vector type and its associated element type
static void FindWidenVecType(const TargetLowering &TLI, unsigned Width, MVT VVT,
                             MVT& EVT, MVT& VecEVT) {
  // We start with the preferred width, make it a power of 2 and see if
  // we can find a vector type of that width. If not, we reduce it by
  // another power of 2.  If we have widen the type, a vector of bytes should
  // always be legal.
  assert(TLI.isTypeLegal(VVT));
  unsigned EWidth = Width + 1;
  do {
    assert(EWidth > 0);
    EWidth =  (1 << Log2_32(EWidth-1));
    EVT = MVT::getIntegerVT(EWidth);
    unsigned NumEVT = VVT.getSizeInBits()/EWidth;
    VecEVT = MVT::getVectorVT(EVT, NumEVT);
  } while (!TLI.isTypeLegal(VecEVT) ||
           VVT.getSizeInBits() != VecEVT.getSizeInBits());
}

SDValue SelectionDAGLegalize::genWidenVectorLoads(SDValueVector& LdChain,
                                                    SDValue   Chain,
                                                    SDValue   BasePtr,
                                                    const Value *SV,
                                                    int         SVOffset,
                                                    unsigned    Alignment,
                                                    bool        isVolatile,
                                                    unsigned    LdWidth,
                                                    MVT         ResType,
                                                    DebugLoc    dl) {
  // We assume that we have good rules to handle loading power of two loads so
  // we break down the operations to power of 2 loads.  The strategy is to
  // load the largest power of 2 that we can easily transform to a legal vector
  // and then insert into that vector, and the cast the result into the legal
  // vector that we want.  This avoids unnecessary stack converts.
  // TODO: If the Ldwidth is legal, alignment is the same as the LdWidth, and
  //       the load is nonvolatile, we an use a wider load for the value.
  // Find a vector length we can load a large chunk
  MVT EVT, VecEVT;
  unsigned EVTWidth;
  FindWidenVecType(TLI, LdWidth, ResType, EVT, VecEVT);
  EVTWidth = EVT.getSizeInBits();

  SDValue LdOp = DAG.getLoad(EVT, dl, Chain, BasePtr, SV, SVOffset,
                               isVolatile, Alignment);
  SDValue VecOp = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VecEVT, LdOp);
  LdChain.push_back(LdOp.getValue(1));
  
  // Check if we can load the element with one instruction
  if (LdWidth == EVTWidth) {
    return DAG.getNode(ISD::BIT_CONVERT, dl, ResType, VecOp);
  }

  // The vector element order is endianness dependent.
  unsigned Idx = 1;
  LdWidth -= EVTWidth;
  unsigned Offset = 0;
    
  while (LdWidth > 0) {
    unsigned Increment = EVTWidth / 8;
    Offset += Increment;
    BasePtr = DAG.getNode(ISD::ADD, dl, BasePtr.getValueType(), BasePtr,
                          DAG.getIntPtrConstant(Increment));

    if (LdWidth < EVTWidth) {
      // Our current type we are using is too large, use a smaller size by
      // using a smaller power of 2
      unsigned oEVTWidth = EVTWidth;
      FindWidenVecType(TLI, LdWidth, ResType, EVT, VecEVT);
      EVTWidth = EVT.getSizeInBits();
      // Readjust position and vector position based on new load type
      Idx = Idx * (oEVTWidth/EVTWidth);
      VecOp = DAG.getNode(ISD::BIT_CONVERT, dl, VecEVT, VecOp);
    }
      
    SDValue LdOp = DAG.getLoad(EVT, dl, Chain, BasePtr, SV,
                                 SVOffset+Offset, isVolatile,
                                 MinAlign(Alignment, Offset));
    LdChain.push_back(LdOp.getValue(1));
    VecOp = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VecEVT, VecOp, LdOp,
                        DAG.getIntPtrConstant(Idx++));
    
    LdWidth -= EVTWidth;
  }

  return DAG.getNode(ISD::BIT_CONVERT, dl, ResType, VecOp);
}

bool SelectionDAGLegalize::LoadWidenVectorOp(SDValue& Result,
                                             SDValue& TFOp,
                                             SDValue Op,
                                             MVT NVT) {
  // TODO: Add support for ConcatVec and the ability to load many vector
  //       types (e.g., v4i8).  This will not work when a vector register
  //       to memory mapping is strange (e.g., vector elements are not
  //       stored in some sequential order).

  // It must be true that the widen vector type is bigger than where 
  // we need to load from.
  LoadSDNode *LD = cast<LoadSDNode>(Op.getNode());
  MVT LdVT = LD->getMemoryVT();
  DebugLoc dl = LD->getDebugLoc();
  assert(LdVT.isVector() && NVT.isVector());
  assert(LdVT.getVectorElementType() == NVT.getVectorElementType());
  
  // Load information
  SDValue Chain = LD->getChain();
  SDValue BasePtr = LD->getBasePtr();
  int       SVOffset = LD->getSrcValueOffset();
  unsigned  Alignment = LD->getAlignment();
  bool      isVolatile = LD->isVolatile();
  const Value *SV = LD->getSrcValue();
  unsigned int LdWidth = LdVT.getSizeInBits();
  
  // Load value as a large register
  SDValueVector LdChain;
  Result = genWidenVectorLoads(LdChain, Chain, BasePtr, SV, SVOffset,
                               Alignment, isVolatile, LdWidth, NVT, dl);

  if (LdChain.size() == 1) {
    TFOp = LdChain[0];
    return true;
  }
  else {
    TFOp=DAG.getNode(ISD::TokenFactor, dl, MVT::Other, 
                     &LdChain[0], LdChain.size());
    return false;
  }
}


void SelectionDAGLegalize::genWidenVectorStores(SDValueVector& StChain,
                                                SDValue   Chain,
                                                SDValue   BasePtr,
                                                const Value *SV,
                                                int         SVOffset,
                                                unsigned    Alignment,
                                                bool        isVolatile,
                                                SDValue     ValOp,
                                                unsigned    StWidth,
                                                DebugLoc    dl) {
  // Breaks the stores into a series of power of 2 width stores.  For any
  // width, we convert the vector to the vector of element size that we
  // want to store.  This avoids requiring a stack convert.
  
  // Find a width of the element type we can store with
  MVT VVT = ValOp.getValueType();
  MVT EVT, VecEVT;
  unsigned EVTWidth;
  FindWidenVecType(TLI, StWidth, VVT, EVT, VecEVT);
  EVTWidth = EVT.getSizeInBits();

  SDValue VecOp = DAG.getNode(ISD::BIT_CONVERT, dl, VecEVT, ValOp);
  SDValue EOp = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EVT, VecOp,
                            DAG.getIntPtrConstant(0));
  SDValue StOp = DAG.getStore(Chain, dl, EOp, BasePtr, SV, SVOffset,
                               isVolatile, Alignment);
  StChain.push_back(StOp);

  // Check if we are done
  if (StWidth == EVTWidth) {
    return;
  }
  
  unsigned Idx = 1;
  StWidth -= EVTWidth;
  unsigned Offset = 0;
    
  while (StWidth > 0) {
    unsigned Increment = EVTWidth / 8;
    Offset += Increment;
    BasePtr = DAG.getNode(ISD::ADD, dl, BasePtr.getValueType(), BasePtr,
                          DAG.getIntPtrConstant(Increment));
                          
    if (StWidth < EVTWidth) {
      // Our current type we are using is too large, use a smaller size by
      // using a smaller power of 2
      unsigned oEVTWidth = EVTWidth;
      FindWidenVecType(TLI, StWidth, VVT, EVT, VecEVT);
      EVTWidth = EVT.getSizeInBits();
      // Readjust position and vector position based on new load type
      Idx = Idx * (oEVTWidth/EVTWidth);
      VecOp = DAG.getNode(ISD::BIT_CONVERT, dl, VecEVT, VecOp);
    }
    
    EOp = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EVT, VecOp,
                      DAG.getIntPtrConstant(Idx++));
    StChain.push_back(DAG.getStore(Chain, dl, EOp, BasePtr, SV,
                                   SVOffset + Offset, isVolatile,
                                   MinAlign(Alignment, Offset)));
    StWidth -= EVTWidth;
  }
}


SDValue SelectionDAGLegalize::StoreWidenVectorOp(StoreSDNode *ST,
                                                   SDValue Chain,
                                                   SDValue BasePtr) {
  // TODO: It might be cleaner if we can use SplitVector and have more legal
  //        vector types that can be stored into memory (e.g., v4xi8 can
  //        be stored as a word). This will not work when a vector register
  //        to memory mapping is strange (e.g., vector elements are not
  //        stored in some sequential order).
  
  MVT StVT = ST->getMemoryVT();
  SDValue ValOp = ST->getValue();
  DebugLoc dl = ST->getDebugLoc();

  // Check if we have widen this node with another value
  std::map<SDValue, SDValue>::iterator I = WidenNodes.find(ValOp);
  if (I != WidenNodes.end())
    ValOp = I->second;
    
  MVT VVT = ValOp.getValueType();

  // It must be true that we the widen vector type is bigger than where
  // we need to store.
  assert(StVT.isVector() && VVT.isVector());
  assert(StVT.bitsLT(VVT));
  assert(StVT.getVectorElementType() == VVT.getVectorElementType());

  // Store value
  SDValueVector StChain;
  genWidenVectorStores(StChain, Chain, BasePtr, ST->getSrcValue(),
                       ST->getSrcValueOffset(), ST->getAlignment(),
                       ST->isVolatile(), ValOp, StVT.getSizeInBits(), dl);
  if (StChain.size() == 1)
    return StChain[0];
  else 
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                       &StChain[0], StChain.size());
}


// SelectionDAG::Legalize - This is the entry point for the file.
//
void SelectionDAG::Legalize(bool TypesNeedLegalizing) {
  /// run - This is the main entry point to this class.
  ///
  SelectionDAGLegalize(*this, TypesNeedLegalizing).LegalizeDAG();
}

