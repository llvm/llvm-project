//===-- SelectionDAG.cpp - Implement the SelectionDAG data structures -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "SDNodeOrdering.h"
#include "SDNodeDbgValue.h"
#include "llvm/Constants.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Function.h"
#include "llvm/GlobalAlias.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

/// makeVTList - Return an instance of the SDVTList struct initialized with the
/// specified members.
static SDVTList makeVTList(const EVT *VTs, unsigned NumVTs) {
  SDVTList Res = {VTs, NumVTs};
  return Res;
}

static const fltSemantics *EVTToAPFloatSemantics(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Unknown FP format");
  case MVT::f32:     return &APFloat::IEEEsingle;
  case MVT::f64:     return &APFloat::IEEEdouble;
  case MVT::f80:     return &APFloat::x87DoubleExtended;
  case MVT::f128:    return &APFloat::IEEEquad;
  case MVT::ppcf128: return &APFloat::PPCDoubleDouble;
  }
}

SelectionDAG::DAGUpdateListener::~DAGUpdateListener() {}

//===----------------------------------------------------------------------===//
//                              ConstantFPSDNode Class
//===----------------------------------------------------------------------===//

/// isExactlyValue - We don't rely on operator== working on double values, as
/// it returns true for things that are clearly not equal, like -0.0 and 0.0.
/// As such, this method can be used to do an exact bit-for-bit comparison of
/// two floating point values.
bool ConstantFPSDNode::isExactlyValue(const APFloat& V) const {
  return getValueAPF().bitwiseIsEqual(V);
}

bool ConstantFPSDNode::isValueValidForType(EVT VT,
                                           const APFloat& Val) {
  assert(VT.isFloatingPoint() && "Can only convert between FP types");

  // PPC long double cannot be converted to any other type.
  if (VT == MVT::ppcf128 ||
      &Val.getSemantics() == &APFloat::PPCDoubleDouble)
    return false;

  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  (void) Val2.convert(*EVTToAPFloatSemantics(VT), APFloat::rmNearestTiesToEven,
                      &losesInfo);
  return !losesInfo;
}

//===----------------------------------------------------------------------===//
//                              ISD Namespace
//===----------------------------------------------------------------------===//

/// isBuildVectorAllOnes - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are ~0 or undef.
bool ISD::isBuildVectorAllOnes(const SDNode *N) {
  // Look through a bit convert.
  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  unsigned i = 0, e = N->getNumOperands();

  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;

  // Do not accept an all-undef vector.
  if (i == e) return false;

  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements.
  SDValue NotZero = N->getOperand(i);
  if (isa<ConstantSDNode>(NotZero)) {
    if (!cast<ConstantSDNode>(NotZero)->isAllOnesValue())
      return false;
  } else if (isa<ConstantFPSDNode>(NotZero)) {
    if (!cast<ConstantFPSDNode>(NotZero)->getValueAPF().
                bitcastToAPInt().isAllOnesValue())
      return false;
  } else
    return false;

  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != NotZero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
  return true;
}


/// isBuildVectorAllZeros - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are 0 or undef.
bool ISD::isBuildVectorAllZeros(const SDNode *N) {
  // Look through a bit convert.
  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  unsigned i = 0, e = N->getNumOperands();

  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;

  // Do not accept an all-undef vector.
  if (i == e) return false;

  // Do not accept build_vectors that aren't all constants or which have non-0
  // elements.
  SDValue Zero = N->getOperand(i);
  if (isa<ConstantSDNode>(Zero)) {
    if (!cast<ConstantSDNode>(Zero)->isNullValue())
      return false;
  } else if (isa<ConstantFPSDNode>(Zero)) {
    if (!cast<ConstantFPSDNode>(Zero)->getValueAPF().isPosZero())
      return false;
  } else
    return false;

  // Okay, we have at least one 0 value, check to see if the rest match or are
  // undefs.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != Zero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
  return true;
}

/// isScalarToVector - Return true if the specified node is a
/// ISD::SCALAR_TO_VECTOR node or a BUILD_VECTOR node where only the low
/// element is not an undef.
bool ISD::isScalarToVector(const SDNode *N) {
  if (N->getOpcode() == ISD::SCALAR_TO_VECTOR)
    return true;

  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;
  if (N->getOperand(0).getOpcode() == ISD::UNDEF)
    return false;
  unsigned NumElems = N->getNumOperands();
  if (NumElems == 1)
    return false;
  for (unsigned i = 1; i < NumElems; ++i) {
    SDValue V = N->getOperand(i);
    if (V.getOpcode() != ISD::UNDEF)
      return false;
  }
  return true;
}

/// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
/// when given the operation for (X op Y).
ISD::CondCode ISD::getSetCCSwappedOperands(ISD::CondCode Operation) {
  // To perform this operation, we just need to swap the L and G bits of the
  // operation.
  unsigned OldL = (Operation >> 2) & 1;
  unsigned OldG = (Operation >> 1) & 1;
  return ISD::CondCode((Operation & ~6) |  // Keep the N, U, E bits
                       (OldL << 1) |       // New G bit
                       (OldG << 2));       // New L bit.
}

/// getSetCCInverse - Return the operation corresponding to !(X op Y), where
/// 'op' is a valid SetCC operation.
ISD::CondCode ISD::getSetCCInverse(ISD::CondCode Op, bool isInteger) {
  unsigned Operation = Op;
  if (isInteger)
    Operation ^= 7;   // Flip L, G, E bits, but not U.
  else
    Operation ^= 15;  // Flip all of the condition bits.

  if (Operation > ISD::SETTRUE2)
    Operation &= ~8;  // Don't let N and U bits get set.

  return ISD::CondCode(Operation);
}


/// isSignedOp - For an integer comparison, return 1 if the comparison is a
/// signed operation and 2 if the result is an unsigned comparison.  Return zero
/// if the operation does not depend on the sign of the input (setne and seteq).
static int isSignedOp(ISD::CondCode Opcode) {
  switch (Opcode) {
  default: llvm_unreachable("Illegal integer setcc operation!");
  case ISD::SETEQ:
  case ISD::SETNE: return 0;
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETGT:
  case ISD::SETGE: return 1;
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETUGT:
  case ISD::SETUGE: return 2;
  }
}

/// getSetCCOrOperation - Return the result of a logical OR between different
/// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This function
/// returns SETCC_INVALID if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCOrOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                       bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed integer setcc with an unsigned integer setcc.
    return ISD::SETCC_INVALID;

  unsigned Op = Op1 | Op2;  // Combine all of the condition bits.

  // If the N and U bits get set then the resultant comparison DOES suddenly
  // care about orderedness, and is true when ordered.
  if (Op > ISD::SETTRUE2)
    Op &= ~16;     // Clear the U bit if the N bit is set.

  // Canonicalize illegal integer setcc's.
  if (isInteger && Op == ISD::SETUNE)  // e.g. SETUGT | SETULT
    Op = ISD::SETNE;

  return ISD::CondCode(Op);
}

/// getSetCCAndOperation - Return the result of a logical AND between different
/// comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
/// function returns zero if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCAndOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                        bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed setcc with an unsigned setcc.
    return ISD::SETCC_INVALID;

  // Combine all of the condition bits.
  ISD::CondCode Result = ISD::CondCode(Op1 & Op2);

  // Canonicalize illegal integer setcc's.
  if (isInteger) {
    switch (Result) {
    default: break;
    case ISD::SETUO : Result = ISD::SETFALSE; break;  // SETUGT & SETULT
    case ISD::SETOEQ:                                 // SETEQ  & SETU[LG]E
    case ISD::SETUEQ: Result = ISD::SETEQ   ; break;  // SETUGE & SETULE
    case ISD::SETOLT: Result = ISD::SETULT  ; break;  // SETULT & SETNE
    case ISD::SETOGT: Result = ISD::SETUGT  ; break;  // SETUGT & SETNE
    }
  }

  return Result;
}

//===----------------------------------------------------------------------===//
//                           SDNode Profile Support
//===----------------------------------------------------------------------===//

/// AddNodeIDOpcode - Add the node opcode to the NodeID data.
///
static void AddNodeIDOpcode(FoldingSetNodeID &ID, unsigned OpC)  {
  ID.AddInteger(OpC);
}

/// AddNodeIDValueTypes - Value type lists are intern'd so we can represent them
/// solely with their pointer.
static void AddNodeIDValueTypes(FoldingSetNodeID &ID, SDVTList VTList) {
  ID.AddPointer(VTList.VTs);
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              const SDValue *Ops, unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops) {
    ID.AddPointer(Ops->getNode());
    ID.AddInteger(Ops->getResNo());
  }
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              const SDUse *Ops, unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops) {
    ID.AddPointer(Ops->getNode());
    ID.AddInteger(Ops->getResNo());
  }
}

static void AddNodeIDNode(FoldingSetNodeID &ID,
                          unsigned short OpC, SDVTList VTList,
                          const SDValue *OpList, unsigned N) {
  AddNodeIDOpcode(ID, OpC);
  AddNodeIDValueTypes(ID, VTList);
  AddNodeIDOperands(ID, OpList, N);
}

/// AddNodeIDCustom - If this is an SDNode with special info, add this info to
/// the NodeID data.
static void AddNodeIDCustom(FoldingSetNodeID &ID, const SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::TargetExternalSymbol:
  case ISD::ExternalSymbol:
    llvm_unreachable("Should only be used on nodes with operands");
  default: break;  // Normal nodes don't need extra info.
  case ISD::TargetConstant:
  case ISD::Constant:
    ID.AddPointer(cast<ConstantSDNode>(N)->getConstantIntValue());
    break;
  case ISD::TargetConstantFP:
  case ISD::ConstantFP: {
    ID.AddPointer(cast<ConstantFPSDNode>(N)->getConstantFPValue());
    break;
  }
  case ISD::TargetGlobalAddress:
  case ISD::GlobalAddress:
  case ISD::TargetGlobalTLSAddress:
  case ISD::GlobalTLSAddress: {
    const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);
    ID.AddPointer(GA->getGlobal());
    ID.AddInteger(GA->getOffset());
    ID.AddInteger(GA->getTargetFlags());
    break;
  }
  case ISD::BasicBlock:
    ID.AddPointer(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::Register:
    ID.AddInteger(cast<RegisterSDNode>(N)->getReg());
    break;

  case ISD::SRCVALUE:
    ID.AddPointer(cast<SrcValueSDNode>(N)->getValue());
    break;
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    ID.AddInteger(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::JumpTable:
  case ISD::TargetJumpTable:
    ID.AddInteger(cast<JumpTableSDNode>(N)->getIndex());
    ID.AddInteger(cast<JumpTableSDNode>(N)->getTargetFlags());
    break;
  case ISD::ConstantPool:
  case ISD::TargetConstantPool: {
    const ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(N);
    ID.AddInteger(CP->getAlignment());
    ID.AddInteger(CP->getOffset());
    if (CP->isMachineConstantPoolEntry())
      CP->getMachineCPVal()->addSelectionDAGCSEId(ID);
    else
      ID.AddPointer(CP->getConstVal());
    ID.AddInteger(CP->getTargetFlags());
    break;
  }
  case ISD::LOAD: {
    const LoadSDNode *LD = cast<LoadSDNode>(N);
    ID.AddInteger(LD->getMemoryVT().getRawBits());
    ID.AddInteger(LD->getRawSubclassData());
    break;
  }
  case ISD::STORE: {
    const StoreSDNode *ST = cast<StoreSDNode>(N);
    ID.AddInteger(ST->getMemoryVT().getRawBits());
    ID.AddInteger(ST->getRawSubclassData());
    break;
  }
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_SWAP:
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
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_STORE: {
    const AtomicSDNode *AT = cast<AtomicSDNode>(N);
    ID.AddInteger(AT->getMemoryVT().getRawBits());
    ID.AddInteger(AT->getRawSubclassData());
    break;
  }
  case ISD::VECTOR_SHUFFLE: {
    const ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);
    for (unsigned i = 0, e = N->getValueType(0).getVectorNumElements();
         i != e; ++i)
      ID.AddInteger(SVN->getMaskElt(i));
    break;
  }
  case ISD::TargetBlockAddress:
  case ISD::BlockAddress: {
    ID.AddPointer(cast<BlockAddressSDNode>(N)->getBlockAddress());
    ID.AddInteger(cast<BlockAddressSDNode>(N)->getTargetFlags());
    break;
  }
  } // end switch (N->getOpcode())
}

/// AddNodeIDNode - Generic routine for adding a nodes info to the NodeID
/// data.
static void AddNodeIDNode(FoldingSetNodeID &ID, const SDNode *N) {
  AddNodeIDOpcode(ID, N->getOpcode());
  // Add the return value info.
  AddNodeIDValueTypes(ID, N->getVTList());
  // Add the operand info.
  AddNodeIDOperands(ID, N->op_begin(), N->getNumOperands());

  // Handle SDNode leafs with special info.
  AddNodeIDCustom(ID, N);
}

/// encodeMemSDNodeFlags - Generic routine for computing a value for use in
/// the CSE map that carries volatility, temporalness, indexing mode, and
/// extension/truncation information.
///
static inline unsigned
encodeMemSDNodeFlags(int ConvType, ISD::MemIndexedMode AM, bool isVolatile,
                     bool isNonTemporal) {
  assert((ConvType & 3) == ConvType &&
         "ConvType may not require more than 2 bits!");
  assert((AM & 7) == AM &&
         "AM may not require more than 3 bits!");
  return ConvType |
         (AM << 2) |
         (isVolatile << 5) |
         (isNonTemporal << 6);
}

//===----------------------------------------------------------------------===//
//                              SelectionDAG Class
//===----------------------------------------------------------------------===//

/// doNotCSE - Return true if CSE should not be performed for this node.
static bool doNotCSE(SDNode *N) {
  if (N->getValueType(0) == MVT::Glue)
    return true; // Never CSE anything that produces a flag.

  switch (N->getOpcode()) {
  default: break;
  case ISD::HANDLENODE:
  case ISD::EH_LABEL:
    return true;   // Never CSE these nodes.
  }

  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Glue)
      return true; // Never CSE anything that produces a flag.

  return false;
}

/// RemoveDeadNodes - This method deletes all unreachable nodes in the
/// SelectionDAG.
void SelectionDAG::RemoveDeadNodes() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted.
  HandleSDNode Dummy(getRoot());

  SmallVector<SDNode*, 128> DeadNodes;

  // Add all obviously-dead nodes to the DeadNodes worklist.
  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I)
    if (I->use_empty())
      DeadNodes.push_back(I);

  RemoveDeadNodes(DeadNodes);

  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

/// RemoveDeadNodes - This method deletes the unreachable nodes in the
/// given list, and any nodes that become unreachable as a result.
void SelectionDAG::RemoveDeadNodes(SmallVectorImpl<SDNode *> &DeadNodes,
                                   DAGUpdateListener *UpdateListener) {

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.pop_back_val();

    if (UpdateListener)
      UpdateListener->NodeDeleted(N, 0);

    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ) {
      SDUse &Use = *I++;
      SDNode *Operand = Use.getNode();
      Use.set(SDValue());

      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }

    DeallocateNode(N);
  }
}

void SelectionDAG::RemoveDeadNode(SDNode *N, DAGUpdateListener *UpdateListener){
  SmallVector<SDNode*, 16> DeadNodes(1, N);
  RemoveDeadNodes(DeadNodes, UpdateListener);
}

void SelectionDAG::DeleteNode(SDNode *N) {
  // First take this out of the appropriate CSE map.
  RemoveNodeFromCSEMaps(N);

  // Finally, remove uses due to operands of this node, remove from the
  // AllNodes list, and delete the node.
  DeleteNodeNotInCSEMaps(N);
}

void SelectionDAG::DeleteNodeNotInCSEMaps(SDNode *N) {
  assert(N != AllNodes.begin() && "Cannot delete the entry node!");
  assert(N->use_empty() && "Cannot delete a node that is not dead!");

  // Drop all of the operands and decrement used node's use counts.
  N->DropOperands();

  DeallocateNode(N);
}

void SelectionDAG::DeallocateNode(SDNode *N) {
  if (N->OperandsNeedDelete)
    delete[] N->OperandList;

  // Set the opcode to DELETED_NODE to help catch bugs when node
  // memory is reallocated.
  N->NodeType = ISD::DELETED_NODE;

  NodeAllocator.Deallocate(AllNodes.remove(N));

  // Remove the ordering of this node.
  Ordering->remove(N);

  // If any of the SDDbgValue nodes refer to this SDNode, invalidate them.
  ArrayRef<SDDbgValue*> DbgVals = DbgInfo->getSDDbgValues(N);
  for (unsigned i = 0, e = DbgVals.size(); i != e; ++i)
    DbgVals[i]->setIsInvalidated();
}

/// RemoveNodeFromCSEMaps - Take the specified node out of the CSE map that
/// correspond to it.  This is useful when we're about to delete or repurpose
/// the node.  We don't want future request for structurally identical nodes
/// to return N anymore.
bool SelectionDAG::RemoveNodeFromCSEMaps(SDNode *N) {
  bool Erased = false;
  switch (N->getOpcode()) {
  case ISD::HANDLENODE: return false;  // noop.
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != 0;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = 0;
    break;
  case ISD::ExternalSymbol:
    Erased = ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::TargetExternalSymbol: {
    ExternalSymbolSDNode *ESN = cast<ExternalSymbolSDNode>(N);
    Erased = TargetExternalSymbols.erase(
               std::pair<std::string,unsigned char>(ESN->getSymbol(),
                                                    ESN->getTargetFlags()));
    break;
  }
  case ISD::VALUETYPE: {
    EVT VT = cast<VTSDNode>(N)->getVT();
    if (VT.isExtended()) {
      Erased = ExtendedValueTypeNodes.erase(VT);
    } else {
      Erased = ValueTypeNodes[VT.getSimpleVT().SimpleTy] != 0;
      ValueTypeNodes[VT.getSimpleVT().SimpleTy] = 0;
    }
    break;
  }
  default:
    // Remove it from the CSE Map.
    assert(N->getOpcode() != ISD::DELETED_NODE && "DELETED_NODE in CSEMap!");
    assert(N->getOpcode() != ISD::EntryToken && "EntryToken in CSEMap!");
    Erased = CSEMap.RemoveNode(N);
    break;
  }
#ifndef NDEBUG
  // Verify that the node was actually in one of the CSE maps, unless it has a
  // flag result (which cannot be CSE'd) or is one of the special cases that are
  // not subject to CSE.
  if (!Erased && N->getValueType(N->getNumValues()-1) != MVT::Glue &&
      !N->isMachineOpcode() && !doNotCSE(N)) {
    N->dump(this);
    dbgs() << "\n";
    llvm_unreachable("Node is not in map!");
  }
#endif
  return Erased;
}

/// AddModifiedNodeToCSEMaps - The specified node has been removed from the CSE
/// maps and modified in place. Add it back to the CSE maps, unless an identical
/// node already exists, in which case transfer all its users to the existing
/// node. This transfer can potentially trigger recursive merging.
///
void
SelectionDAG::AddModifiedNodeToCSEMaps(SDNode *N,
                                       DAGUpdateListener *UpdateListener) {
  // For node types that aren't CSE'd, just act as if no identical node
  // already exists.
  if (!doNotCSE(N)) {
    SDNode *Existing = CSEMap.GetOrInsertNode(N);
    if (Existing != N) {
      // If there was already an existing matching node, use ReplaceAllUsesWith
      // to replace the dead one with the existing one.  This can cause
      // recursive merging of other unrelated nodes down the line.
      ReplaceAllUsesWith(N, Existing, UpdateListener);

      // N is now dead.  Inform the listener if it exists and delete it.
      if (UpdateListener)
        UpdateListener->NodeDeleted(N, Existing);
      DeleteNodeNotInCSEMaps(N);
      return;
    }
  }

  // If the node doesn't already exist, we updated it.  Inform a listener if
  // it exists.
  if (UpdateListener)
    UpdateListener->NodeUpdated(N);
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, SDValue Op,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return 0;

  SDValue Ops[] = { Op };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 1);
  AddNodeIDCustom(ID, N);
  SDNode *Node = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  return Node;
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N,
                                           SDValue Op1, SDValue Op2,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return 0;

  SDValue Ops[] = { Op1, Op2 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 2);
  AddNodeIDCustom(ID, N);
  SDNode *Node = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  return Node;
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N,
                                           const SDValue *Ops,unsigned NumOps,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return 0;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, NumOps);
  AddNodeIDCustom(ID, N);
  SDNode *Node = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  return Node;
}

#ifndef NDEBUG
/// VerifyNodeCommon - Sanity check the given node.  Aborts if it is invalid.
static void VerifyNodeCommon(SDNode *N) {
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::BUILD_PAIR: {
    EVT VT = N->getValueType(0);
    assert(N->getNumValues() == 1 && "Too many results!");
    assert(!VT.isVector() && (VT.isInteger() || VT.isFloatingPoint()) &&
           "Wrong return type!");
    assert(N->getNumOperands() == 2 && "Wrong number of operands!");
    assert(N->getOperand(0).getValueType() == N->getOperand(1).getValueType() &&
           "Mismatched operand types!");
    assert(N->getOperand(0).getValueType().isInteger() == VT.isInteger() &&
           "Wrong operand type!");
    assert(VT.getSizeInBits() == 2 * N->getOperand(0).getValueSizeInBits() &&
           "Wrong return type size");
    break;
  }
  case ISD::BUILD_VECTOR: {
    assert(N->getNumValues() == 1 && "Too many results!");
    assert(N->getValueType(0).isVector() && "Wrong return type!");
    assert(N->getNumOperands() == N->getValueType(0).getVectorNumElements() &&
           "Wrong number of operands!");
    EVT EltVT = N->getValueType(0).getVectorElementType();
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      assert((I->getValueType() == EltVT ||
             (EltVT.isInteger() && I->getValueType().isInteger() &&
              EltVT.bitsLE(I->getValueType()))) &&
            "Wrong operand type!");
      assert(I->getValueType() == N->getOperand(0).getValueType() &&
             "Operands must all have the same type");
    }
    break;
  }
  }
}

/// VerifySDNode - Sanity check the given SDNode.  Aborts if it is invalid.
static void VerifySDNode(SDNode *N) {
  // The SDNode allocators cannot be used to allocate nodes with fields that are
  // not present in an SDNode!
  assert(!isa<MemSDNode>(N) && "Bad MemSDNode!");
  assert(!isa<ShuffleVectorSDNode>(N) && "Bad ShuffleVectorSDNode!");
  assert(!isa<ConstantSDNode>(N) && "Bad ConstantSDNode!");
  assert(!isa<ConstantFPSDNode>(N) && "Bad ConstantFPSDNode!");
  assert(!isa<GlobalAddressSDNode>(N) && "Bad GlobalAddressSDNode!");
  assert(!isa<FrameIndexSDNode>(N) && "Bad FrameIndexSDNode!");
  assert(!isa<JumpTableSDNode>(N) && "Bad JumpTableSDNode!");
  assert(!isa<ConstantPoolSDNode>(N) && "Bad ConstantPoolSDNode!");
  assert(!isa<BasicBlockSDNode>(N) && "Bad BasicBlockSDNode!");
  assert(!isa<SrcValueSDNode>(N) && "Bad SrcValueSDNode!");
  assert(!isa<MDNodeSDNode>(N) && "Bad MDNodeSDNode!");
  assert(!isa<RegisterSDNode>(N) && "Bad RegisterSDNode!");
  assert(!isa<BlockAddressSDNode>(N) && "Bad BlockAddressSDNode!");
  assert(!isa<EHLabelSDNode>(N) && "Bad EHLabelSDNode!");
  assert(!isa<ExternalSymbolSDNode>(N) && "Bad ExternalSymbolSDNode!");
  assert(!isa<CondCodeSDNode>(N) && "Bad CondCodeSDNode!");
  assert(!isa<CvtRndSatSDNode>(N) && "Bad CvtRndSatSDNode!");
  assert(!isa<VTSDNode>(N) && "Bad VTSDNode!");
  assert(!isa<MachineSDNode>(N) && "Bad MachineSDNode!");

  VerifyNodeCommon(N);
}

/// VerifyMachineNode - Sanity check the given MachineNode.  Aborts if it is
/// invalid.
static void VerifyMachineNode(SDNode *N) {
  // The MachineNode allocators cannot be used to allocate nodes with fields
  // that are not present in a MachineNode!
  // Currently there are no such nodes.

  VerifyNodeCommon(N);
}
#endif // NDEBUG

/// getEVTAlignment - Compute the default alignment value for the
/// given type.
///
unsigned SelectionDAG::getEVTAlignment(EVT VT) const {
  Type *Ty = VT == MVT::iPTR ?
                   PointerType::get(Type::getInt8Ty(*getContext()), 0) :
                   VT.getTypeForEVT(*getContext());

  return TLI.getTargetData()->getABITypeAlignment(Ty);
}

// EntryNode could meaningfully have debug info if we can find it...
SelectionDAG::SelectionDAG(const TargetMachine &tm)
  : TM(tm), TLI(*tm.getTargetLowering()), TSI(*tm.getSelectionDAGInfo()),
    EntryNode(ISD::EntryToken, DebugLoc(), getVTList(MVT::Other)),
    Root(getEntryNode()), Ordering(0) {
  AllNodes.push_back(&EntryNode);
  Ordering = new SDNodeOrdering();
  DbgInfo = new SDDbgInfo();
}

void SelectionDAG::init(MachineFunction &mf) {
  MF = &mf;
  Context = &mf.getFunction()->getContext();
}

SelectionDAG::~SelectionDAG() {
  allnodes_clear();
  delete Ordering;
  delete DbgInfo;
}

void SelectionDAG::allnodes_clear() {
  assert(&*AllNodes.begin() == &EntryNode);
  AllNodes.remove(AllNodes.begin());
  while (!AllNodes.empty())
    DeallocateNode(AllNodes.begin());
}

void SelectionDAG::clear() {
  allnodes_clear();
  OperandAllocator.Reset();
  CSEMap.clear();

  ExtendedValueTypeNodes.clear();
  ExternalSymbols.clear();
  TargetExternalSymbols.clear();
  std::fill(CondCodeNodes.begin(), CondCodeNodes.end(),
            static_cast<CondCodeSDNode*>(0));
  std::fill(ValueTypeNodes.begin(), ValueTypeNodes.end(),
            static_cast<SDNode*>(0));

  EntryNode.UseList = 0;
  AllNodes.push_back(&EntryNode);
  Root = getEntryNode();
  Ordering->clear();
  DbgInfo->clear();
}

SDValue SelectionDAG::getAnyExtOrTrunc(SDValue Op, DebugLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ANY_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getSExtOrTrunc(SDValue Op, DebugLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::SIGN_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getZExtOrTrunc(SDValue Op, DebugLoc DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ZERO_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getZeroExtendInReg(SDValue Op, DebugLoc DL, EVT VT) {
  assert(!VT.isVector() &&
         "getZeroExtendInReg should use the vector element type instead of "
         "the vector type!");
  if (Op.getValueType() == VT) return Op;
  unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
  APInt Imm = APInt::getLowBitsSet(BitWidth,
                                   VT.getSizeInBits());
  return getNode(ISD::AND, DL, Op.getValueType(), Op,
                 getConstant(Imm, Op.getValueType()));
}

/// getNOT - Create a bitwise NOT operation as (XOR Val, -1).
///
SDValue SelectionDAG::getNOT(DebugLoc DL, SDValue Val, EVT VT) {
  EVT EltVT = VT.getScalarType();
  SDValue NegOne =
    getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), VT);
  return getNode(ISD::XOR, DL, VT, Val, NegOne);
}

SDValue SelectionDAG::getConstant(uint64_t Val, EVT VT, bool isT) {
  EVT EltVT = VT.getScalarType();
  assert((EltVT.getSizeInBits() >= 64 ||
         (uint64_t)((int64_t)Val >> EltVT.getSizeInBits()) + 1 < 2) &&
         "getConstant with a uint64_t value that doesn't fit in the type!");
  return getConstant(APInt(EltVT.getSizeInBits(), Val), VT, isT);
}

SDValue SelectionDAG::getConstant(const APInt &Val, EVT VT, bool isT) {
  return getConstant(*ConstantInt::get(*Context, Val), VT, isT);
}

SDValue SelectionDAG::getConstant(const ConstantInt &Val, EVT VT, bool isT) {
  assert(VT.isInteger() && "Cannot create FP integer constant!");

  EVT EltVT = VT.getScalarType();
  const ConstantInt *Elt = &Val;

  // In some cases the vector type is legal but the element type is illegal and
  // needs to be promoted, for example v8i8 on ARM.  In this case, promote the
  // inserted value (the type does not need to match the vector element type).
  // Any extra bits introduced will be truncated away.
  if (VT.isVector() && TLI.getTypeAction(*getContext(), EltVT) ==
      TargetLowering::TypePromoteInteger) {
   EltVT = TLI.getTypeToTransformTo(*getContext(), EltVT);
   APInt NewVal = Elt->getValue().zext(EltVT.getSizeInBits());
   Elt = ConstantInt::get(*getContext(), NewVal);
  }

  assert(Elt->getBitWidth() == EltVT.getSizeInBits() &&
         "APInt size does not match type size!");
  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), 0, 0);
  ID.AddPointer(Elt);
  void *IP = 0;
  SDNode *N = NULL;
  if ((N = CSEMap.FindNodeOrInsertPos(ID, IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = new (NodeAllocator) ConstantSDNode(isT, Elt, EltVT);
    CSEMap.InsertNode(N, IP);
    AllNodes.push_back(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector()) {
    SmallVector<SDValue, 8> Ops;
    Ops.assign(VT.getVectorNumElements(), Result);
    Result = getNode(ISD::BUILD_VECTOR, DebugLoc(), VT, &Ops[0], Ops.size());
  }
  return Result;
}

SDValue SelectionDAG::getIntPtrConstant(uint64_t Val, bool isTarget) {
  return getConstant(Val, TLI.getPointerTy(), isTarget);
}


SDValue SelectionDAG::getConstantFP(const APFloat& V, EVT VT, bool isTarget) {
  return getConstantFP(*ConstantFP::get(*getContext(), V), VT, isTarget);
}

SDValue SelectionDAG::getConstantFP(const ConstantFP& V, EVT VT, bool isTarget){
  assert(VT.isFloatingPoint() && "Cannot create integer FP constant!");

  EVT EltVT = VT.getScalarType();

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), 0, 0);
  ID.AddPointer(&V);
  void *IP = 0;
  SDNode *N = NULL;
  if ((N = CSEMap.FindNodeOrInsertPos(ID, IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = new (NodeAllocator) ConstantFPSDNode(isTarget, &V, EltVT);
    CSEMap.InsertNode(N, IP);
    AllNodes.push_back(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector()) {
    SmallVector<SDValue, 8> Ops;
    Ops.assign(VT.getVectorNumElements(), Result);
    // FIXME DebugLoc info might be appropriate here
    Result = getNode(ISD::BUILD_VECTOR, DebugLoc(), VT, &Ops[0], Ops.size());
  }
  return Result;
}

SDValue SelectionDAG::getConstantFP(double Val, EVT VT, bool isTarget) {
  EVT EltVT = VT.getScalarType();
  if (EltVT==MVT::f32)
    return getConstantFP(APFloat((float)Val), VT, isTarget);
  else if (EltVT==MVT::f64)
    return getConstantFP(APFloat(Val), VT, isTarget);
  else if (EltVT==MVT::f80 || EltVT==MVT::f128) {
    bool ignored;
    APFloat apf = APFloat(Val);
    apf.convert(*EVTToAPFloatSemantics(EltVT), APFloat::rmNearestTiesToEven,
                &ignored);
    return getConstantFP(apf, VT, isTarget);
  } else {
    assert(0 && "Unsupported type in getConstantFP");
    return SDValue();
  }
}

SDValue SelectionDAG::getGlobalAddress(const GlobalValue *GV, DebugLoc DL,
                                       EVT VT, int64_t Offset,
                                       bool isTargetGA,
                                       unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTargetGA) &&
         "Cannot set target flags on target-independent globals");

  // Truncate (with sign-extension) the offset value to the pointer size.
  EVT PTy = TLI.getPointerTy();
  unsigned BitWidth = PTy.getSizeInBits();
  if (BitWidth < 64)
    Offset = (Offset << (64 - BitWidth) >> (64 - BitWidth));

  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar) {
    // If GV is an alias then use the aliasee for determining thread-localness.
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal(false));
  }

  unsigned Opc;
  if (GVar && GVar->isThreadLocal())
    Opc = isTargetGA ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress;
  else
    Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddPointer(GV);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) GlobalAddressSDNode(Opc, DL, GV, VT,
                                                      Offset, TargetFlags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getFrameIndex(int FI, EVT VT, bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetFrameIndex : ISD::FrameIndex;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(FI);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) FrameIndexSDNode(FI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getJumpTable(int JTI, EVT VT, bool isTarget,
                                   unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent jump tables");
  unsigned Opc = isTarget ? ISD::TargetJumpTable : ISD::JumpTable;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(JTI);
  ID.AddInteger(TargetFlags);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) JumpTableSDNode(JTI, VT, isTarget,
                                                  TargetFlags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getConstantPool(const Constant *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = TLI.getTargetData()->getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  ID.AddPointer(C);
  ID.AddInteger(TargetFlags);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) ConstantPoolSDNode(isTarget, C, VT, Offset,
                                                     Alignment, TargetFlags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}


SDValue SelectionDAG::getConstantPool(MachineConstantPoolValue *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = TLI.getTargetData()->getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  C->addSelectionDAGCSEId(ID);
  ID.AddInteger(TargetFlags);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) ConstantPoolSDNode(isTarget, C, VT, Offset,
                                                     Alignment, TargetFlags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BasicBlock, getVTList(MVT::Other), 0, 0);
  ID.AddPointer(MBB);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) BasicBlockSDNode(MBB);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getValueType(EVT VT) {
  if (VT.isSimple() && (unsigned)VT.getSimpleVT().SimpleTy >=
      ValueTypeNodes.size())
    ValueTypeNodes.resize(VT.getSimpleVT().SimpleTy+1);

  SDNode *&N = VT.isExtended() ?
    ExtendedValueTypeNodes[VT] : ValueTypeNodes[VT.getSimpleVT().SimpleTy];

  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) VTSDNode(VT);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getExternalSymbol(const char *Sym, EVT VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) ExternalSymbolSDNode(false, Sym, 0, VT);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTargetExternalSymbol(const char *Sym, EVT VT,
                                              unsigned char TargetFlags) {
  SDNode *&N =
    TargetExternalSymbols[std::pair<std::string,unsigned char>(Sym,
                                                               TargetFlags)];
  if (N) return SDValue(N, 0);
  N = new (NodeAllocator) ExternalSymbolSDNode(true, Sym, TargetFlags, VT);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getCondCode(ISD::CondCode Cond) {
  if ((unsigned)Cond >= CondCodeNodes.size())
    CondCodeNodes.resize(Cond+1);

  if (CondCodeNodes[Cond] == 0) {
    CondCodeSDNode *N = new (NodeAllocator) CondCodeSDNode(Cond);
    CondCodeNodes[Cond] = N;
    AllNodes.push_back(N);
  }

  return SDValue(CondCodeNodes[Cond], 0);
}

// commuteShuffle - swaps the values of N1 and N2, and swaps all indices in
// the shuffle mask M that point at N1 to point at N2, and indices that point
// N2 to point at N1.
static void commuteShuffle(SDValue &N1, SDValue &N2, SmallVectorImpl<int> &M) {
  std::swap(N1, N2);
  int NElts = M.size();
  for (int i = 0; i != NElts; ++i) {
    if (M[i] >= NElts)
      M[i] -= NElts;
    else if (M[i] >= 0)
      M[i] += NElts;
  }
}

SDValue SelectionDAG::getVectorShuffle(EVT VT, DebugLoc dl, SDValue N1,
                                       SDValue N2, const int *Mask) {
  assert(N1.getValueType() == N2.getValueType() && "Invalid VECTOR_SHUFFLE");
  assert(VT.isVector() && N1.getValueType().isVector() &&
         "Vector Shuffle VTs must be a vectors");
  assert(VT.getVectorElementType() == N1.getValueType().getVectorElementType()
         && "Vector Shuffle VTs must have same element type");

  // Canonicalize shuffle undef, undef -> undef
  if (N1.getOpcode() == ISD::UNDEF && N2.getOpcode() == ISD::UNDEF)
    return getUNDEF(VT);

  // Validate that all indices in Mask are within the range of the elements
  // input to the shuffle.
  unsigned NElts = VT.getVectorNumElements();
  SmallVector<int, 8> MaskVec;
  for (unsigned i = 0; i != NElts; ++i) {
    assert(Mask[i] < (int)(NElts * 2) && "Index out of range");
    MaskVec.push_back(Mask[i]);
  }

  // Canonicalize shuffle v, v -> v, undef
  if (N1 == N2) {
    N2 = getUNDEF(VT);
    for (unsigned i = 0; i != NElts; ++i)
      if (MaskVec[i] >= (int)NElts) MaskVec[i] -= NElts;
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N1.getOpcode() == ISD::UNDEF)
    commuteShuffle(N1, N2, MaskVec);

  // Canonicalize all index into lhs, -> shuffle lhs, undef
  // Canonicalize all index into rhs, -> shuffle rhs, undef
  bool AllLHS = true, AllRHS = true;
  bool N2Undef = N2.getOpcode() == ISD::UNDEF;
  for (unsigned i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= (int)NElts) {
      if (N2Undef)
        MaskVec[i] = -1;
      else
        AllLHS = false;
    } else if (MaskVec[i] >= 0) {
      AllRHS = false;
    }
  }
  if (AllLHS && AllRHS)
    return getUNDEF(VT);
  if (AllLHS && !N2Undef)
    N2 = getUNDEF(VT);
  if (AllRHS) {
    N1 = getUNDEF(VT);
    commuteShuffle(N1, N2, MaskVec);
  }

  // If Identity shuffle, or all shuffle in to undef, return that node.
  bool AllUndef = true;
  bool Identity = true;
  for (unsigned i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= 0 && MaskVec[i] != (int)i) Identity = false;
    if (MaskVec[i] >= 0) AllUndef = false;
  }
  if (Identity && NElts == N1.getValueType().getVectorNumElements())
    return N1;
  if (AllUndef)
    return getUNDEF(VT);

  FoldingSetNodeID ID;
  SDValue Ops[2] = { N1, N2 };
  AddNodeIDNode(ID, ISD::VECTOR_SHUFFLE, getVTList(VT), Ops, 2);
  for (unsigned i = 0; i != NElts; ++i)
    ID.AddInteger(MaskVec[i]);

  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  // Allocate the mask array for the node out of the BumpPtrAllocator, since
  // SDNode doesn't have access to it.  This memory will be "leaked" when
  // the node is deallocated, but recovered when the NodeAllocator is released.
  int *MaskAlloc = OperandAllocator.Allocate<int>(NElts);
  memcpy(MaskAlloc, &MaskVec[0], NElts * sizeof(int));

  ShuffleVectorSDNode *N =
    new (NodeAllocator) ShuffleVectorSDNode(VT, dl, N1, N2, MaskAlloc);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getConvertRndSat(EVT VT, DebugLoc dl,
                                       SDValue Val, SDValue DTy,
                                       SDValue STy, SDValue Rnd, SDValue Sat,
                                       ISD::CvtCode Code) {
  // If the src and dest types are the same and the conversion is between
  // integer types of the same sign or two floats, no conversion is necessary.
  if (DTy == STy &&
      (Code == ISD::CVT_UU || Code == ISD::CVT_SS || Code == ISD::CVT_FF))
    return Val;

  FoldingSetNodeID ID;
  SDValue Ops[] = { Val, DTy, STy, Rnd, Sat };
  AddNodeIDNode(ID, ISD::CONVERT_RNDSAT, getVTList(VT), &Ops[0], 5);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  CvtRndSatSDNode *N = new (NodeAllocator) CvtRndSatSDNode(VT, dl, Ops, 5,
                                                           Code);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getRegister(unsigned RegNo, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::Register, getVTList(VT), 0, 0);
  ID.AddInteger(RegNo);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) RegisterSDNode(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getEHLabel(DebugLoc dl, SDValue Root, MCSymbol *Label) {
  FoldingSetNodeID ID;
  SDValue Ops[] = { Root };
  AddNodeIDNode(ID, ISD::EH_LABEL, getVTList(MVT::Other), &Ops[0], 1);
  ID.AddPointer(Label);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) EHLabelSDNode(dl, Root, Label);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}


SDValue SelectionDAG::getBlockAddress(const BlockAddress *BA, EVT VT,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  unsigned Opc = isTarget ? ISD::TargetBlockAddress : ISD::BlockAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddPointer(BA);
  ID.AddInteger(TargetFlags);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) BlockAddressSDNode(Opc, VT, BA, TargetFlags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getSrcValue(const Value *V) {
  assert((!V || V->getType()->isPointerTy()) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::SRCVALUE, getVTList(MVT::Other), 0, 0);
  ID.AddPointer(V);

  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) SrcValueSDNode(V);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

/// getMDNode - Return an MDNodeSDNode which holds an MDNode.
SDValue SelectionDAG::getMDNode(const MDNode *MD) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MDNODE_SDNODE, getVTList(MVT::Other), 0, 0);
  ID.AddPointer(MD);

  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) MDNodeSDNode(MD);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}


/// getShiftAmountOperand - Return the specified value casted to
/// the target's desired shift amount type.
SDValue SelectionDAG::getShiftAmountOperand(EVT LHSTy, SDValue Op) {
  EVT OpTy = Op.getValueType();
  MVT ShTy = TLI.getShiftAmountTy(LHSTy);
  if (OpTy == ShTy || OpTy.isVector()) return Op;

  ISD::NodeType Opcode = OpTy.bitsGT(ShTy) ?  ISD::TRUNCATE : ISD::ZERO_EXTEND;
  return getNode(Opcode, Op.getDebugLoc(), ShTy, Op);
}

/// CreateStackTemporary - Create a stack temporary, suitable for holding the
/// specified value type.
SDValue SelectionDAG::CreateStackTemporary(EVT VT, unsigned minAlign) {
  MachineFrameInfo *FrameInfo = getMachineFunction().getFrameInfo();
  unsigned ByteSize = VT.getStoreSize();
  Type *Ty = VT.getTypeForEVT(*getContext());
  unsigned StackAlign =
  std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty), minAlign);

  int FrameIdx = FrameInfo->CreateStackObject(ByteSize, StackAlign, false);
  return getFrameIndex(FrameIdx, TLI.getPointerTy());
}

/// CreateStackTemporary - Create a stack temporary suitable for holding
/// either of the specified value types.
SDValue SelectionDAG::CreateStackTemporary(EVT VT1, EVT VT2) {
  unsigned Bytes = std::max(VT1.getStoreSizeInBits(),
                            VT2.getStoreSizeInBits())/8;
  Type *Ty1 = VT1.getTypeForEVT(*getContext());
  Type *Ty2 = VT2.getTypeForEVT(*getContext());
  const TargetData *TD = TLI.getTargetData();
  unsigned Align = std::max(TD->getPrefTypeAlignment(Ty1),
                            TD->getPrefTypeAlignment(Ty2));

  MachineFrameInfo *FrameInfo = getMachineFunction().getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(Bytes, Align, false);
  return getFrameIndex(FrameIdx, TLI.getPointerTy());
}

SDValue SelectionDAG::FoldSetCC(EVT VT, SDValue N1,
                                SDValue N2, ISD::CondCode Cond, DebugLoc dl) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return getConstant(0, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return getConstant(1, VT);

  case ISD::SETOEQ:
  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETONE:
  case ISD::SETO:
  case ISD::SETUO:
  case ISD::SETUEQ:
  case ISD::SETUNE:
    assert(!N1.getValueType().isInteger() && "Illegal setcc for integer!");
    break;
  }

  if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode())) {
    const APInt &C2 = N2C->getAPIntValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode())) {
      const APInt &C1 = N1C->getAPIntValue();

      switch (Cond) {
      default: llvm_unreachable("Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, VT);
      case ISD::SETULT: return getConstant(C1.ult(C2), VT);
      case ISD::SETUGT: return getConstant(C1.ugt(C2), VT);
      case ISD::SETULE: return getConstant(C1.ule(C2), VT);
      case ISD::SETUGE: return getConstant(C1.uge(C2), VT);
      case ISD::SETLT:  return getConstant(C1.slt(C2), VT);
      case ISD::SETGT:  return getConstant(C1.sgt(C2), VT);
      case ISD::SETLE:  return getConstant(C1.sle(C2), VT);
      case ISD::SETGE:  return getConstant(C1.sge(C2), VT);
      }
    }
  }
  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.getNode())) {
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2.getNode())) {
      // No compile time operations on this type yet.
      if (N1C->getValueType(0) == MVT::ppcf128)
        return SDValue();

      APFloat::cmpResult R = N1C->getValueAPF().compare(N2C->getValueAPF());
      switch (Cond) {
      default: break;
      case ISD::SETEQ:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOEQ: return getConstant(R==APFloat::cmpEqual, VT);
      case ISD::SETNE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETONE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpLessThan, VT);
      case ISD::SETLT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOLT: return getConstant(R==APFloat::cmpLessThan, VT);
      case ISD::SETGT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOGT: return getConstant(R==APFloat::cmpGreaterThan, VT);
      case ISD::SETLE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOLE: return getConstant(R==APFloat::cmpLessThan ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETGE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        // fall through
      case ISD::SETOGE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETO:   return getConstant(R!=APFloat::cmpUnordered, VT);
      case ISD::SETUO:  return getConstant(R==APFloat::cmpUnordered, VT);
      case ISD::SETUEQ: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETUNE: return getConstant(R!=APFloat::cmpEqual, VT);
      case ISD::SETULT: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpLessThan, VT);
      case ISD::SETUGT: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpUnordered, VT);
      case ISD::SETULE: return getConstant(R!=APFloat::cmpGreaterThan, VT);
      case ISD::SETUGE: return getConstant(R!=APFloat::cmpLessThan, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      return getSetCC(dl, VT, N2, N1, ISD::getSetCCSwappedOperands(Cond));
    }
  }

  // Could not fold it.
  return SDValue();
}

/// SignBitIsZero - Return true if the sign bit of Op is known to be zero.  We
/// use this predicate to simplify operations downstream.
bool SelectionDAG::SignBitIsZero(SDValue Op, unsigned Depth) const {
  // This predicate is not safe for vector operations.
  if (Op.getValueType().isVector())
    return false;

  unsigned BitWidth = Op.getValueType().getScalarType().getSizeInBits();
  return MaskedValueIsZero(Op, APInt::getSignBit(BitWidth), Depth);
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool SelectionDAG::MaskedValueIsZero(SDValue Op, const APInt &Mask,
                                     unsigned Depth) const {
  APInt KnownZero, KnownOne;
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
  return (KnownZero & Mask) == Mask;
}

/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
void SelectionDAG::ComputeMaskedBits(SDValue Op, const APInt &Mask,
                                     APInt &KnownZero, APInt &KnownOne,
                                     unsigned Depth) const {
  unsigned BitWidth = Mask.getBitWidth();
  assert(BitWidth == Op.getValueType().getScalarType().getSizeInBits() &&
         "Mask size mismatches value type size!");

  KnownZero = KnownOne = APInt(BitWidth, 0);   // Don't know anything.
  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.

  APInt KnownZero2, KnownOne2;

  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask & ~KnownZero,
                      KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  case ISD::OR:
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask & ~KnownOne,
                      KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  case ISD::XOR: {
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case ISD::MUL: {
    APInt Mask2 = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(1), Mask2, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conserative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    KnownOne.clearAllBits();
    unsigned TrailZ = KnownZero.countTrailingOnes() +
                      KnownZero2.countTrailingOnes();
    unsigned LeadZ =  std::max(KnownZero.countLeadingOnes() +
                               KnownZero2.countLeadingOnes(),
                               BitWidth) - BitWidth;

    TrailZ = std::min(TrailZ, BitWidth);
    LeadZ = std::min(LeadZ, BitWidth);
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) |
                APInt::getHighBitsSet(BitWidth, LeadZ);
    KnownZero &= Mask;
    return;
  }
  case ISD::UDIV: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(0),
                      AllOnes, KnownZero2, KnownOne2, Depth+1);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clearAllBits();
    KnownZero2.clearAllBits();
    ComputeMaskedBits(Op.getOperand(1),
                      AllOnes, KnownZero2, KnownOne2, Depth+1);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ) & Mask;
    return;
  }
  case ISD::SELECT:
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SELECT_CC:
    ComputeMaskedBits(Op.getOperand(3), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
    if (Op.getResNo() != 1)
      return;
    // The boolean result conforms to getBooleanContents.  Fall through.
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (TLI.getBooleanContents(Op.getValueType().isVector()) ==
        TargetLowering::ZeroOrOneBooleanContent && BitWidth > 1)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    return;
  case ISD::SHL:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      ComputeMaskedBits(Op.getOperand(0), Mask.lshr(ShAmt),
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero <<= ShAmt;
      KnownOne  <<= ShAmt;
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, ShAmt);
    }
    return;
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      ComputeMaskedBits(Op.getOperand(0), (Mask << ShAmt),
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt) & Mask;
      KnownZero |= HighBits;  // High bits known zero.
    }
    return;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      APInt InDemandedMask = (Mask << ShAmt);
      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt) & Mask;
      if (HighBits.getBoolValue())
        InDemandedMask |= APInt::getSignBit(BitWidth);

      ComputeMaskedBits(Op.getOperand(0), InDemandedMask, KnownZero, KnownOne,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      // Handle the sign bits.
      APInt SignBit = APInt::getSignBit(BitWidth);
      SignBit = SignBit.lshr(ShAmt);  // Adjust to where it is now in the mask.

      if (KnownZero.intersects(SignBit)) {
        KnownZero |= HighBits;  // New bits are known zero.
      } else if (KnownOne.intersects(SignBit)) {
        KnownOne  |= HighBits;  // New bits are known one.
      }
    }
    return;
  case ISD::SIGN_EXTEND_INREG: {
    EVT EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    unsigned EBits = EVT.getScalarType().getSizeInBits();

    // Sign extension.  Compute the demanded bits in the result that are not
    // present in the input.
    APInt NewBits = APInt::getHighBitsSet(BitWidth, BitWidth - EBits) & Mask;

    APInt InSignBit = APInt::getSignBit(EBits);
    APInt InputDemandedBits = Mask & APInt::getLowBitsSet(BitWidth, EBits);

    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InSignBit = InSignBit.zext(BitWidth);
    if (NewBits.getBoolValue())
      InputDemandedBits |= InSignBit;

    ComputeMaskedBits(Op.getOperand(0), InputDemandedBits,
                      KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero.intersects(InSignBit)) {         // Input sign bit known clear
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne.intersects(InSignBit)) {   // Input sign bit known set
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {                              // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne  &= ~NewBits;
    }
    return;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP: {
    unsigned LowBits = Log2_32(BitWidth)+1;
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
    KnownOne.clearAllBits();
    return;
  }
  case ISD::LOAD: {
    if (ISD::isZEXTLoad(Op.getNode())) {
      LoadSDNode *LD = cast<LoadSDNode>(Op);
      EVT VT = LD->getMemoryVT();
      unsigned MemBits = VT.getScalarType().getSizeInBits();
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - MemBits) & Mask;
    }
    return;
  }
  case ISD::ZERO_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits) & Mask;
    APInt InMask    = Mask.trunc(InBits);
    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    return;
  }
  case ISD::SIGN_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt InSignBit = APInt::getSignBit(InBits);
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits) & Mask;
    APInt InMask = Mask.trunc(InBits);

    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded. Temporarily set this bit in the mask for our callee.
    if (NewBits.getBoolValue())
      InMask |= InSignBit;

    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);

    // Note if the sign bit is known to be zero or one.
    bool SignBitKnownZero = KnownZero.isNegative();
    bool SignBitKnownOne  = KnownOne.isNegative();
    assert(!(SignBitKnownZero && SignBitKnownOne) &&
           "Sign bit can't be known to be both zero and one!");

    // If the sign bit wasn't actually demanded by our caller, we don't
    // want it set in the KnownZero and KnownOne result values. Reset the
    // mask and reapply it to the result values.
    InMask = Mask.trunc(InBits);
    KnownZero &= InMask;
    KnownOne  &= InMask;

    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit is known zero or one, the top bits match.
    if (SignBitKnownZero)
      KnownZero |= NewBits;
    else if (SignBitKnownOne)
      KnownOne  |= NewBits;
    return;
  }
  case ISD::ANY_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt InMask = Mask.trunc(InBits);
    KnownZero = KnownZero.trunc(InBits);
    KnownOne = KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    return;
  }
  case ISD::TRUNCATE: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt InMask = Mask.zext(InBits);
    KnownZero = KnownZero.zext(InBits);
    KnownOne = KnownOne.zext(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);
    break;
  }
  case ISD::AssertZext: {
    EVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth, VT.getSizeInBits());
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero,
                      KnownOne, Depth+1);
    KnownZero |= (~InMask) & Mask;
    return;
  }
  case ISD::FGETSIGN:
    // All bits are zero except the low bit.
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    return;

  case ISD::SUB: {
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (CLHS->getAPIntValue().isNonNegative()) {
        unsigned NLZ = (CLHS->getAPIntValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        ComputeMaskedBits(Op.getOperand(1), MaskV, KnownZero2, KnownOne2,
                          Depth+1);

        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((KnownZero2 & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getAPIntValue().countLeadingZeros();
          // Top bits known zero.
          KnownZero = APInt::getHighBitsSet(BitWidth, NLZ2) & Mask;
        }
      }
    }
  }
  // fall through
  case ISD::ADD:
  case ISD::ADDE: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    APInt Mask2 = APInt::getLowBitsSet(BitWidth,
                                       BitWidth - Mask.countLeadingZeros());
    ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");
    unsigned KnownZeroOut = KnownZero2.countTrailingOnes();

    ComputeMaskedBits(Op.getOperand(1), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");
    KnownZeroOut = std::min(KnownZeroOut,
                            KnownZero2.countTrailingOnes());

    if (Op.getOpcode() == ISD::ADD) {
      KnownZero |= APInt::getLowBitsSet(BitWidth, KnownZeroOut);
      return;
    }

    // With ADDE, a carry bit may be added in, so we can only use this
    // information if we know (at least) that the low two bits are clear.  We
    // then return to the caller that the low bit is unknown but that other bits
    // are known zero.
    if (KnownZeroOut >= 2) // ADDE
      KnownZero |= APInt::getBitsSet(BitWidth, 1, KnownZeroOut);
    return;
  }
  case ISD::SREM:
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue().abs();
      if (RA.isPowerOf2()) {
        APInt LowBits = RA - 1;
        APInt Mask2 = LowBits | APInt::getSignBit(BitWidth);
        ComputeMaskedBits(Op.getOperand(0), Mask2,KnownZero2,KnownOne2,Depth+1);

        // The low bits of the first operand are unchanged by the srem.
        KnownZero = KnownZero2 & LowBits;
        KnownOne = KnownOne2 & LowBits;

        // If the first operand is non-negative or has all low bits zero, then
        // the upper bits are all zero.
        if (KnownZero2[BitWidth-1] || ((KnownZero2 & LowBits) == LowBits))
          KnownZero |= ~LowBits;

        // If the first operand is negative and not all low bits are zero, then
        // the upper bits are all one.
        if (KnownOne2[BitWidth-1] && ((KnownOne2 & LowBits) != 0))
          KnownOne |= ~LowBits;

        KnownZero &= Mask;
        KnownOne &= Mask;

        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
      }
    }
    return;
  case ISD::UREM: {
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        APInt Mask2 = LowBits & Mask;
        KnownZero |= ~LowBits & Mask;
        ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero, KnownOne,Depth+1);
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(0), AllOnes, KnownZero, KnownOne,
                      Depth+1);
    ComputeMaskedBits(Op.getOperand(1), AllOnes, KnownZero2, KnownOne2,
                      Depth+1);

    uint32_t Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clearAllBits();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders) & Mask;
    return;
  }
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    if (unsigned Align = InferPtrAlignment(Op)) {
      // The low bits are known zero if the pointer is aligned.
      KnownZero = APInt::getLowBitsSet(BitWidth, Log2_32(Align));
      return;
    }
    break;

  default:
    if (Op.getOpcode() < ISD::BUILTIN_OP_END)
      break;
    // Fallthrough
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    // Allow the target to implement this method for its nodes.
    TLI.computeMaskedBitsForTargetNode(Op, Mask, KnownZero, KnownOne, *this,
                                       Depth);
    return;
  }
}

/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "SRA X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
unsigned SelectionDAG::ComputeNumSignBits(SDValue Op, unsigned Depth) const{
  EVT VT = Op.getValueType();
  assert(VT.isInteger() && "Invalid VT!");
  unsigned VTBits = VT.getScalarType().getSizeInBits();
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  if (Depth == 6)
    return 1;  // Limit search depth.

  switch (Op.getOpcode()) {
  default: break;
  case ISD::AssertSext:
    Tmp = cast<VTSDNode>(Op.getOperand(1))->getVT().getSizeInBits();
    return VTBits-Tmp+1;
  case ISD::AssertZext:
    Tmp = cast<VTSDNode>(Op.getOperand(1))->getVT().getSizeInBits();
    return VTBits-Tmp;

  case ISD::Constant: {
    const APInt &Val = cast<ConstantSDNode>(Op)->getAPIntValue();
    return Val.getNumSignBits();
  }

  case ISD::SIGN_EXTEND:
    Tmp = VTBits-Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    return ComputeNumSignBits(Op.getOperand(0), Depth+1) + Tmp;

  case ISD::SIGN_EXTEND_INREG:
    // Max of the input and what this extends.
    Tmp =
      cast<VTSDNode>(Op.getOperand(1))->getVT().getScalarType().getSizeInBits();
    Tmp = VTBits-Tmp+1;

    Tmp2 = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    return std::max(Tmp, Tmp2);

  case ISD::SRA:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    // SRA X, C   -> adds C sign bits.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Tmp += C->getZExtValue();
      if (Tmp > VTBits) Tmp = VTBits;
    }
    return Tmp;
  case ISD::SHL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (C->getZExtValue() >= VTBits ||      // Bad shift.
          C->getZExtValue() >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - C->getZExtValue();
    }
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits at the worst.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // ComputeMaskedBits, and pick whichever answer is better.
    }
    break;

  case ISD::SELECT:
    Tmp = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(2), Depth+1);
    return std::min(Tmp, Tmp2);

  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
    if (Op.getResNo() != 1)
      break;
    // The boolean result conforms to getBooleanContents.  Fall through.
  case ISD::SETCC:
    // If setcc returns 0/-1, all bits are sign bits.
    if (TLI.getBooleanContents(Op.getValueType().isVector()) ==
        TargetLowering::ZeroOrNegativeOneBooleanContent)
      return VTBits;
    break;
  case ISD::ROTL:
  case ISD::ROTR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned RotAmt = C->getZExtValue() & (VTBits-1);

      // Handle rotate right by N like a rotate left by 32-N.
      if (Op.getOpcode() == ISD::ROTR)
        RotAmt = (VTBits-RotAmt) & (VTBits-1);

      // If we aren't rotating out all of the known-in sign bits, return the
      // number that are left.  This handles rotl(sext(x), 1) for example.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (Tmp > RotAmt+1) return Tmp-RotAmt;
    }
    break;
  case ISD::ADD:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.

    // Special case decrementing a value (ADD X, -1):
    if (ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero, KnownOne;
        APInt Mask = APInt::getAllOnesValue(VTBits);
        ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);

        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)) == Mask)
          return VTBits;

        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }

    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;
      return std::min(Tmp, Tmp2)-1;
    break;

  case ISD::SUB:
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;

    // Handle NEG.
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero, KnownOne;
        APInt Mask = APInt::getAllOnesValue(VTBits);
        ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)) == Mask)
          return VTBits;

        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero.isNegative())
          return Tmp2;

        // Otherwise, we treat this like a SUB.
      }

    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      return std::min(Tmp, Tmp2)-1;
    break;
  case ISD::TRUNCATE:
    // FIXME: it's tricky to do anything useful for this, but it is an important
    // case for targets like X86.
    break;
  }

  // Handle LOADX separately here. EXTLOAD case will fallthrough.
  if (Op.getOpcode() == ISD::LOAD) {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    unsigned ExtType = LD->getExtensionType();
    switch (ExtType) {
    default: break;
    case ISD::SEXTLOAD:    // '17' bits known
      Tmp = LD->getMemoryVT().getScalarType().getSizeInBits();
      return VTBits-Tmp+1;
    case ISD::ZEXTLOAD:    // '16' bits known
      Tmp = LD->getMemoryVT().getScalarType().getSizeInBits();
      return VTBits-Tmp;
    }
  }

  // Allow the target to implement this method for its nodes.
  if (Op.getOpcode() >= ISD::BUILTIN_OP_END ||
      Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_VOID) {
    unsigned NumBits = TLI.ComputeNumSignBitsForTargetNode(Op, Depth);
    if (NumBits > 1) FirstAnswer = std::max(FirstAnswer, NumBits);
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  APInt KnownZero, KnownOne;
  APInt Mask = APInt::getAllOnesValue(VTBits);
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);

  if (KnownZero.isNegative()) {        // sign bit is 0
    Mask = KnownZero;
  } else if (KnownOne.isNegative()) {  // sign bit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return FirstAnswer;
  }

  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask = ~Mask;
  Mask <<= Mask.getBitWidth()-VTBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::max(FirstAnswer, std::min(VTBits, Mask.countLeadingZeros()));
}

/// isBaseWithConstantOffset - Return true if the specified operand is an
/// ISD::ADD with a ConstantSDNode on the right-hand side, or if it is an
/// ISD::OR with a ConstantSDNode that is guaranteed to have the same
/// semantics as an ADD.  This handles the equivalence:
///     X|Cst == X+Cst iff X&Cst = 0.
bool SelectionDAG::isBaseWithConstantOffset(SDValue Op) const {
  if ((Op.getOpcode() != ISD::ADD && Op.getOpcode() != ISD::OR) ||
      !isa<ConstantSDNode>(Op.getOperand(1)))
    return false;

  if (Op.getOpcode() == ISD::OR &&
      !MaskedValueIsZero(Op.getOperand(0),
                     cast<ConstantSDNode>(Op.getOperand(1))->getAPIntValue()))
    return false;

  return true;
}


bool SelectionDAG::isKnownNeverNaN(SDValue Op) const {
  // If we're told that NaNs won't happen, assume they won't.
  if (NoNaNsFPMath)
    return true;

  // If the value is a constant, we can obviously see if it is a NaN or not.
  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return !C->getValueAPF().isNaN();

  // TODO: Recognize more cases here.

  return false;
}

bool SelectionDAG::isKnownNeverZero(SDValue Op) const {
  // If the value is a constant, we can obviously see if it is a zero or not.
  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return !C->isZero();

  // TODO: Recognize more cases here.
  switch (Op.getOpcode()) {
  default: break;
  case ISD::OR:
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return !C->isNullValue();
    break;
  }

  return false;
}

bool SelectionDAG::isEqualTo(SDValue A, SDValue B) const {
  // Check the obvious case.
  if (A == B) return true;

  // For for negative and positive zero.
  if (const ConstantFPSDNode *CA = dyn_cast<ConstantFPSDNode>(A))
    if (const ConstantFPSDNode *CB = dyn_cast<ConstantFPSDNode>(B))
      if (CA->isZero() && CB->isZero()) return true;

  // Otherwise they may not be equal.
  return false;
}

/// getNode - Gets or creates the specified node.
///
SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opcode, getVTList(VT), 0, 0);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) SDNode(Opcode, DL, getVTList(VT));
  CSEMap.InsertNode(N, IP);

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL,
                              EVT VT, SDValue Operand) {
  // Constant fold unary operations with an integer constant operand.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand.getNode())) {
    const APInt &Val = C->getAPIntValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND:
      return getConstant(Val.sextOrTrunc(VT.getSizeInBits()), VT);
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::TRUNCATE:
      return getConstant(Val.zextOrTrunc(VT.getSizeInBits()), VT);
    case ISD::UINT_TO_FP:
    case ISD::SINT_TO_FP: {
      // No compile time operations on ppcf128.
      if (VT == MVT::ppcf128) break;
      APFloat apf(APInt::getNullValue(VT.getSizeInBits()));
      (void)apf.convertFromAPInt(Val,
                                 Opcode==ISD::SINT_TO_FP,
                                 APFloat::rmNearestTiesToEven);
      return getConstantFP(apf, VT);
    }
    case ISD::BITCAST:
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(Val.bitsToFloat(), VT);
      else if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(Val.bitsToDouble(), VT);
      break;
    case ISD::BSWAP:
      return getConstant(Val.byteSwap(), VT);
    case ISD::CTPOP:
      return getConstant(Val.countPopulation(), VT);
    case ISD::CTLZ:
      return getConstant(Val.countLeadingZeros(), VT);
    case ISD::CTTZ:
      return getConstant(Val.countTrailingZeros(), VT);
    }
  }

  // Constant fold unary operations with a floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.getNode())) {
    APFloat V = C->getValueAPF();    // make copy
    if (VT != MVT::ppcf128 && Operand.getValueType() != MVT::ppcf128) {
      switch (Opcode) {
      case ISD::FNEG:
        V.changeSign();
        return getConstantFP(V, VT);
      case ISD::FABS:
        V.clearSign();
        return getConstantFP(V, VT);
      case ISD::FP_ROUND:
      case ISD::FP_EXTEND: {
        bool ignored;
        // This can return overflow, underflow, or inexact; we don't care.
        // FIXME need to be more flexible about rounding mode.
        (void)V.convert(*EVTToAPFloatSemantics(VT),
                        APFloat::rmNearestTiesToEven, &ignored);
        return getConstantFP(V, VT);
      }
      case ISD::FP_TO_SINT:
      case ISD::FP_TO_UINT: {
        integerPart x[2];
        bool ignored;
        assert(integerPartWidth >= 64);
        // FIXME need to be more flexible about rounding mode.
        APFloat::opStatus s = V.convertToInteger(x, VT.getSizeInBits(),
                              Opcode==ISD::FP_TO_SINT,
                              APFloat::rmTowardZero, &ignored);
        if (s==APFloat::opInvalidOp)     // inexact is OK, in fact usual
          break;
        APInt api(VT.getSizeInBits(), x);
        return getConstant(api, VT);
      }
      case ISD::BITCAST:
        if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
          return getConstant((uint32_t)V.bitcastToAPInt().getZExtValue(), VT);
        else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
          return getConstant(V.bitcastToAPInt().getZExtValue(), VT);
        break;
      }
    }
  }

  unsigned OpOpcode = Operand.getNode()->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
  case ISD::MERGE_VALUES:
  case ISD::CONCAT_VECTORS:
    return Operand;         // Factor, merge or concat of one node?  No need.
  case ISD::FP_ROUND: llvm_unreachable("Invalid method to make FP_ROUND node");
  case ISD::FP_EXTEND:
    assert(VT.isFloatingPoint() &&
           Operand.getValueType().isFloatingPoint() && "Invalid FP cast!");
    if (Operand.getValueType() == VT) return Operand;  // noop conversion.
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (Operand.getOpcode() == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::SIGN_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid SIGN_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid sext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // sext(undef) = 0, because the top bits will all be the same.
      return getConstant(0, VT);
    break;
  case ISD::ZERO_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ZERO_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid zext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, DL, VT,
                     Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // zext(undef) = 0, because the top bits will be zero.
      return getConstant(0, VT);
    break;
  case ISD::ANY_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ANY_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType().getScalarType().bitsLT(VT.getScalarType()) &&
           "Invalid anyext node, dst < src!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");

    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
        OpOpcode == ISD::ANY_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);

    // (ext (trunx x)) -> x
    if (OpOpcode == ISD::TRUNCATE) {
      SDValue OpOp = Operand.getNode()->getOperand(0);
      if (OpOp.getValueType() == VT)
        return OpOp;
    }
    break;
  case ISD::TRUNCATE:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid TRUNCATE!");
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert(Operand.getValueType().getScalarType().bitsGT(VT.getScalarType()) &&
           "Invalid truncate node, src < dst!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, DL, VT, Operand.getNode()->getOperand(0));
    else if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
             OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.getNode()->getOperand(0).getValueType().getScalarType()
            .bitsLT(VT.getScalarType()))
        return getNode(OpOpcode, DL, VT, Operand.getNode()->getOperand(0));
      else if (Operand.getNode()->getOperand(0).getValueType().bitsGT(VT))
        return getNode(ISD::TRUNCATE, DL, VT, Operand.getNode()->getOperand(0));
      else
        return Operand.getNode()->getOperand(0);
    }
    break;
  case ISD::BITCAST:
    // Basic sanity checking.
    assert(VT.getSizeInBits() == Operand.getValueType().getSizeInBits()
           && "Cannot BITCAST between types of different sizes!");
    if (VT == Operand.getValueType()) return Operand;  // noop conversion.
    if (OpOpcode == ISD::BITCAST)  // bitconv(bitconv(x)) -> bitconv(x)
      return getNode(ISD::BITCAST, DL, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::SCALAR_TO_VECTOR:
    assert(VT.isVector() && !Operand.getValueType().isVector() &&
           (VT.getVectorElementType() == Operand.getValueType() ||
            (VT.getVectorElementType().isInteger() &&
             Operand.getValueType().isInteger() &&
             VT.getVectorElementType().bitsLE(Operand.getValueType()))) &&
           "Illegal SCALAR_TO_VECTOR node!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    // scalar_to_vector(extract_vector_elt V, 0) -> V, top bits are undefined.
    if (OpOpcode == ISD::EXTRACT_VECTOR_ELT &&
        isa<ConstantSDNode>(Operand.getOperand(1)) &&
        Operand.getConstantOperandVal(1) == 0 &&
        Operand.getOperand(0).getValueType() == VT)
      return Operand.getOperand(0);
    break;
  case ISD::FNEG:
    // -(X-Y) -> (Y-X) is unsafe because when X==Y, -0.0 != +0.0
    if (UnsafeFPMath && OpOpcode == ISD::FSUB)
      return getNode(ISD::FSUB, DL, VT, Operand.getNode()->getOperand(1),
                     Operand.getNode()->getOperand(0));
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.getNode()->getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, DL, VT, Operand.getNode()->getOperand(0));
    break;
  }

  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) { // Don't CSE flag producing nodes
    FoldingSetNodeID ID;
    SDValue Ops[1] = { Operand };
    AddNodeIDNode(ID, Opcode, VTs, Ops, 1);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) UnarySDNode(Opcode, DL, VTs, Operand);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) UnarySDNode(Opcode, DL, VTs, Operand);
  }

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::FoldConstantArithmetic(unsigned Opcode,
                                             EVT VT,
                                             ConstantSDNode *Cst1,
                                             ConstantSDNode *Cst2) {
  const APInt &C1 = Cst1->getAPIntValue(), &C2 = Cst2->getAPIntValue();

  switch (Opcode) {
  case ISD::ADD:  return getConstant(C1 + C2, VT);
  case ISD::SUB:  return getConstant(C1 - C2, VT);
  case ISD::MUL:  return getConstant(C1 * C2, VT);
  case ISD::UDIV:
    if (C2.getBoolValue()) return getConstant(C1.udiv(C2), VT);
    break;
  case ISD::UREM:
    if (C2.getBoolValue()) return getConstant(C1.urem(C2), VT);
    break;
  case ISD::SDIV:
    if (C2.getBoolValue()) return getConstant(C1.sdiv(C2), VT);
    break;
  case ISD::SREM:
    if (C2.getBoolValue()) return getConstant(C1.srem(C2), VT);
    break;
  case ISD::AND:  return getConstant(C1 & C2, VT);
  case ISD::OR:   return getConstant(C1 | C2, VT);
  case ISD::XOR:  return getConstant(C1 ^ C2, VT);
  case ISD::SHL:  return getConstant(C1 << C2, VT);
  case ISD::SRL:  return getConstant(C1.lshr(C2), VT);
  case ISD::SRA:  return getConstant(C1.ashr(C2), VT);
  case ISD::ROTL: return getConstant(C1.rotl(C2), VT);
  case ISD::ROTR: return getConstant(C1.rotr(C2), VT);
  default: break;
  }

  return SDValue();
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              SDValue N1, SDValue N2) {
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  switch (Opcode) {
  default: break;
  case ISD::TokenFactor:
    assert(VT == MVT::Other && N1.getValueType() == MVT::Other &&
           N2.getValueType() == MVT::Other && "Invalid token factor!");
    // Fold trivial token factors.
    if (N1.getOpcode() == ISD::EntryToken) return N2;
    if (N2.getOpcode() == ISD::EntryToken) return N1;
    if (N1 == N2) return N1;
    break;
  case ISD::CONCAT_VECTORS:
    // A CONCAT_VECTOR with all operands BUILD_VECTOR can be simplified to
    // one big BUILD_VECTOR.
    if (N1.getOpcode() == ISD::BUILD_VECTOR &&
        N2.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<SDValue, 16> Elts(N1.getNode()->op_begin(),
                                    N1.getNode()->op_end());
      Elts.append(N2.getNode()->op_begin(), N2.getNode()->op_end());
      return getNode(ISD::BUILD_VECTOR, DL, VT, &Elts[0], Elts.size());
    }
    break;
  case ISD::AND:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X & 0) -> 0.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->isNullValue())
      return N2;
    if (N2C && N2C->isAllOnesValue())  // X & -1 -> X
      return N1;
    break;
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X ^|+- 0) -> X.  This commonly occurs when legalizing i64 values, so
    // it's worth handling here.
    if (N2C && N2C->isNullValue())
      return N1;
    break;
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::MULHU:
  case ISD::MULHS:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::SREM:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
    if (UnsafeFPMath) {
      if (Opcode == ISD::FADD) {
        // 0+x --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1))
          if (CFP->getValueAPF().isZero())
            return N2;
        // x+0 --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N2))
          if (CFP->getValueAPF().isZero())
            return N1;
      } else if (Opcode == ISD::FSUB) {
        // x-0 --> x
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N2))
          if (CFP->getValueAPF().isZero())
            return N1;
      }
    }
    assert(VT.isFloatingPoint() && "This operator only applies to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FCOPYSIGN:   // N1 and result must match.  N1/N2 need not match.
    assert(N1.getValueType() == VT &&
           N1.getValueType().isFloatingPoint() &&
           N2.getValueType().isFloatingPoint() &&
           "Invalid FCOPYSIGN!");
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
    assert(VT == N1.getValueType() &&
           "Shift operators return type must be the same as their first arg");
    assert(VT.isInteger() && N2.getValueType().isInteger() &&
           "Shifts only work on integers");
    // Verify that the shift amount VT is bit enough to hold valid shift
    // amounts.  This catches things like trying to shift an i1024 value by an
    // i8, which is easy to fall into in generic code that uses
    // TLI.getShiftAmount().
    assert(N2.getValueType().getSizeInBits() >=
                   Log2_32_Ceil(N1.getValueType().getSizeInBits()) &&
           "Invalid use of small shift amount with oversized value!");

    // Always fold shifts of i1 values so the code generator doesn't need to
    // handle them.  Since we know the size of the shift has to be less than the
    // size of the value, the shift/rotate count is guaranteed to be zero.
    if (VT == MVT::i1)
      return N1;
    if (N2C && N2C->isNullValue())
      return N1;
    break;
  case ISD::FP_ROUND_INREG: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg round!");
    assert(VT.isFloatingPoint() && EVT.isFloatingPoint() &&
           "Cannot FP_ROUND_INREG integer types");
    assert(EVT.isVector() == VT.isVector() &&
           "FP_ROUND_INREG type should be vector iff the operand "
           "type is vector!");
    assert((!EVT.isVector() ||
            EVT.getVectorNumElements() == VT.getVectorNumElements()) &&
           "Vector element counts must match in FP_ROUND_INREG");
    assert(EVT.bitsLE(VT) && "Not rounding down!");
    (void)EVT;
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  }
  case ISD::FP_ROUND:
    assert(VT.isFloatingPoint() &&
           N1.getValueType().isFloatingPoint() &&
           VT.bitsLE(N1.getValueType()) &&
           isa<ConstantSDNode>(N2) && "Invalid FP_ROUND!");
    if (N1.getValueType() == VT) return N1;  // noop conversion.
    break;
  case ISD::AssertSext:
  case ISD::AssertZext: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(VT.isInteger() && EVT.isInteger() &&
           "Cannot *_EXTEND_INREG FP types");
    assert(!EVT.isVector() &&
           "AssertSExt/AssertZExt type should be the vector element type "
           "rather than the vector type!");
    assert(EVT.bitsLE(VT) && "Not extending!");
    if (VT == EVT) return N1; // noop assertion.
    break;
  }
  case ISD::SIGN_EXTEND_INREG: {
    EVT EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(VT.isInteger() && EVT.isInteger() &&
           "Cannot *_EXTEND_INREG FP types");
    assert(EVT.isVector() == VT.isVector() &&
           "SIGN_EXTEND_INREG type should be vector iff the operand "
           "type is vector!");
    assert((!EVT.isVector() ||
            EVT.getVectorNumElements() == VT.getVectorNumElements()) &&
           "Vector element counts must match in SIGN_EXTEND_INREG");
    assert(EVT.bitsLE(VT) && "Not extending!");
    if (EVT == VT) return N1;  // Not actually extending

    if (N1C) {
      APInt Val = N1C->getAPIntValue();
      unsigned FromBits = EVT.getScalarType().getSizeInBits();
      Val <<= Val.getBitWidth()-FromBits;
      Val = Val.ashr(Val.getBitWidth()-FromBits);
      return getConstant(Val, VT);
    }
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    // EXTRACT_VECTOR_ELT of an UNDEF is an UNDEF.
    if (N1.getOpcode() == ISD::UNDEF)
      return getUNDEF(VT);

    // EXTRACT_VECTOR_ELT of CONCAT_VECTORS is often formed while lowering is
    // expanding copies of large vectors from registers.
    if (N2C &&
        N1.getOpcode() == ISD::CONCAT_VECTORS &&
        N1.getNumOperands() > 0) {
      unsigned Factor =
        N1.getOperand(0).getValueType().getVectorNumElements();
      return getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT,
                     N1.getOperand(N2C->getZExtValue() / Factor),
                     getConstant(N2C->getZExtValue() % Factor,
                                 N2.getValueType()));
    }

    // EXTRACT_VECTOR_ELT of BUILD_VECTOR is often formed while lowering is
    // expanding large vector constants.
    if (N2C && N1.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue Elt = N1.getOperand(N2C->getZExtValue());
      EVT VEltTy = N1.getValueType().getVectorElementType();
      if (Elt.getValueType() != VEltTy) {
        // If the vector element type is not legal, the BUILD_VECTOR operands
        // are promoted and implicitly truncated.  Make that explicit here.
        Elt = getNode(ISD::TRUNCATE, DL, VEltTy, Elt);
      }
      if (VT != VEltTy) {
        // If the vector element type is not legal, the EXTRACT_VECTOR_ELT
        // result is implicitly extended.
        Elt = getNode(ISD::ANY_EXTEND, DL, VT, Elt);
      }
      return Elt;
    }

    // EXTRACT_VECTOR_ELT of INSERT_VECTOR_ELT is often formed when vector
    // operations are lowered to scalars.
    if (N1.getOpcode() == ISD::INSERT_VECTOR_ELT) {
      // If the indices are the same, return the inserted element else
      // if the indices are known different, extract the element from
      // the original vector.
      SDValue N1Op2 = N1.getOperand(2);
      ConstantSDNode *N1Op2C = dyn_cast<ConstantSDNode>(N1Op2.getNode());

      if (N1Op2C && N2C) {
        if (N1Op2C->getZExtValue() == N2C->getZExtValue()) {
          if (VT == N1.getOperand(1).getValueType())
            return N1.getOperand(1);
          else
            return getSExtOrTrunc(N1.getOperand(1), DL, VT);
        }

        return getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, N1.getOperand(0), N2);
      }
    }
    break;
  case ISD::EXTRACT_ELEMENT:
    assert(N2C && (unsigned)N2C->getZExtValue() < 2 && "Bad EXTRACT_ELEMENT!");
    assert(!N1.getValueType().isVector() && !VT.isVector() &&
           (N1.getValueType().isInteger() == VT.isInteger()) &&
           N1.getValueType() != VT &&
           "Wrong types for EXTRACT_ELEMENT!");

    // EXTRACT_ELEMENT of BUILD_PAIR is often formed while legalize is expanding
    // 64-bit integers into 32-bit parts.  Instead of building the extract of
    // the BUILD_PAIR, only to have legalize rip it apart, just do it now.
    if (N1.getOpcode() == ISD::BUILD_PAIR)
      return N1.getOperand(N2C->getZExtValue());

    // EXTRACT_ELEMENT of a constant int is also very common.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      unsigned ElementSize = VT.getSizeInBits();
      unsigned Shift = ElementSize * N2C->getZExtValue();
      APInt ShiftedVal = C->getAPIntValue().lshr(Shift);
      return getConstant(ShiftedVal.trunc(ElementSize), VT);
    }
    break;
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue Index = N2;
    if (VT.isSimple() && N1.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             "Extract subvector VTs must be a vectors!");
      assert(VT.getVectorElementType() == N1.getValueType().getVectorElementType() &&
             "Extract subvector VTs must have the same element type!");
      assert(VT.getSimpleVT() <= N1.getValueType().getSimpleVT() &&
             "Extract subvector must be from larger vector to smaller vector!");

      if (isa<ConstantSDNode>(Index.getNode())) {
        assert((VT.getVectorNumElements() +
                cast<ConstantSDNode>(Index.getNode())->getZExtValue()
                <= N1.getValueType().getVectorNumElements())
               && "Extract subvector overflow!");
      }

      // Trivial extraction.
      if (VT.getSimpleVT() == N1.getValueType().getSimpleVT())
        return N1;
    }
    break;
  }
  }

  if (N1C) {
    if (N2C) {
      SDValue SV = FoldConstantArithmetic(Opcode, VT, N1C, N2C);
      if (SV.getNode()) return SV;
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }
  }

  // Constant fold FP operations.
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.getNode());
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2.getNode());
  if (N1CFP) {
    if (!N2CFP && isCommutativeBinOp(Opcode)) {
      // Cannonicalize constant to RHS if commutative
      std::swap(N1CFP, N2CFP);
      std::swap(N1, N2);
    } else if (N2CFP && VT != MVT::ppcf128) {
      APFloat V1 = N1CFP->getValueAPF(), V2 = N2CFP->getValueAPF();
      APFloat::opStatus s;
      switch (Opcode) {
      case ISD::FADD:
        s = V1.add(V2, APFloat::rmNearestTiesToEven);
        if (s != APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FSUB:
        s = V1.subtract(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FMUL:
        s = V1.multiply(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FDIV:
        s = V1.divide(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp && s!=APFloat::opDivByZero)
          return getConstantFP(V1, VT);
        break;
      case ISD::FREM :
        s = V1.mod(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp && s!=APFloat::opDivByZero)
          return getConstantFP(V1, VT);
        break;
      case ISD::FCOPYSIGN:
        V1.copySign(V2);
        return getConstantFP(V1, VT);
      default: break;
      }
    }
  }

  // Canonicalize an UNDEF to the RHS, even over a constant.
  if (N1.getOpcode() == ISD::UNDEF) {
    if (isCommutativeBinOp(Opcode)) {
      std::swap(N1, N2);
    } else {
      switch (Opcode) {
      case ISD::FP_ROUND_INREG:
      case ISD::SIGN_EXTEND_INREG:
      case ISD::SUB:
      case ISD::FSUB:
      case ISD::FDIV:
      case ISD::FREM:
      case ISD::SRA:
        return N1;     // fold op(undef, arg2) -> undef
      case ISD::UDIV:
      case ISD::SDIV:
      case ISD::UREM:
      case ISD::SREM:
      case ISD::SRL:
      case ISD::SHL:
        if (!VT.isVector())
          return getConstant(0, VT);    // fold op(undef, arg2) -> 0
        // For vectors, we can't easily build an all zero vector, just return
        // the LHS.
        return N2;
      }
    }
  }

  // Fold a bunch of operators when the RHS is undef.
  if (N2.getOpcode() == ISD::UNDEF) {
    switch (Opcode) {
    case ISD::XOR:
      if (N1.getOpcode() == ISD::UNDEF)
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return getConstant(0, VT);
      // fallthrough
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::SUB:
    case ISD::UDIV:
    case ISD::SDIV:
    case ISD::UREM:
    case ISD::SREM:
      return N2;       // fold op(arg1, undef) -> undef
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::FDIV:
    case ISD::FREM:
      if (UnsafeFPMath)
        return N2;
      break;
    case ISD::MUL:
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      if (!VT.isVector())
        return getConstant(0, VT);  // fold op(arg1, undef) -> 0
      // For vectors, we can't easily build an all zero vector, just return
      // the LHS.
      return N1;
    case ISD::OR:
      if (!VT.isVector())
        return getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), VT);
      // For vectors, we can't easily build an all one vector, just return
      // the LHS.
      return N1;
    case ISD::SRA:
      return N1;
    }
  }

  // Memoize this node if possible.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) {
    SDValue Ops[] = { N1, N2 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 2);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) BinarySDNode(Opcode, DL, VTs, N1, N2);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) BinarySDNode(Opcode, DL, VTs, N1, N2);
  }

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  switch (Opcode) {
  case ISD::CONCAT_VECTORS:
    // A CONCAT_VECTOR with all operands BUILD_VECTOR can be simplified to
    // one big BUILD_VECTOR.
    if (N1.getOpcode() == ISD::BUILD_VECTOR &&
        N2.getOpcode() == ISD::BUILD_VECTOR &&
        N3.getOpcode() == ISD::BUILD_VECTOR) {
      SmallVector<SDValue, 16> Elts(N1.getNode()->op_begin(),
                                    N1.getNode()->op_end());
      Elts.append(N2.getNode()->op_begin(), N2.getNode()->op_end());
      Elts.append(N3.getNode()->op_begin(), N3.getNode()->op_end());
      return getNode(ISD::BUILD_VECTOR, DL, VT, &Elts[0], Elts.size());
    }
    break;
  case ISD::SETCC: {
    // Use FoldSetCC to simplify SETCC's.
    SDValue Simp = FoldSetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get(), DL);
    if (Simp.getNode()) return Simp;
    break;
  }
  case ISD::SELECT:
    if (N1C) {
     if (N1C->getZExtValue())
        return N2;             // select true, X, Y -> X
      else
        return N3;             // select false, X, Y -> Y
    }

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::VECTOR_SHUFFLE:
    llvm_unreachable("should use getVectorShuffle constructor!");
    break;
  case ISD::INSERT_SUBVECTOR: {
    SDValue Index = N3;
    if (VT.isSimple() && N1.getValueType().isSimple()
        && N2.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             N2.getValueType().isVector() &&
             "Insert subvector VTs must be a vectors");
      assert(VT == N1.getValueType() &&
             "Dest and insert subvector source types must match!");
      assert(N2.getValueType().getSimpleVT() <= N1.getValueType().getSimpleVT() &&
             "Insert subvector must be from smaller vector to larger vector!");
      if (isa<ConstantSDNode>(Index.getNode())) {
        assert((N2.getValueType().getVectorNumElements() +
                cast<ConstantSDNode>(Index.getNode())->getZExtValue()
                <= VT.getVectorNumElements())
               && "Insert subvector overflow!");
      }

      // Trivial insertion.
      if (VT.getSimpleVT() == N2.getValueType().getSimpleVT())
        return N2;
    }
    break;
  }
  case ISD::BITCAST:
    // Fold bit_convert nodes from a type to themselves.
    if (N1.getValueType() == VT)
      return N1;
    break;
  }

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Glue) {
    SDValue Ops[] = { N1, N2, N3 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 3);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) TernarySDNode(Opcode, DL, VTs, N1, N2, N3);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) TernarySDNode(Opcode, DL, VTs, N1, N2, N3);
  }

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VT, Ops, 4);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4, SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VT, Ops, 5);
}

/// getStackArgumentTokenFactor - Compute a TokenFactor to force all
/// the incoming stack arguments to be loaded from the stack.
SDValue SelectionDAG::getStackArgumentTokenFactor(SDValue Chain) {
  SmallVector<SDValue, 8> ArgChains;

  // Include the original chain at the beginning of the list. When this is
  // used by target LowerCall hooks, this helps legalize find the
  // CALLSEQ_BEGIN node.
  ArgChains.push_back(Chain);

  // Add a chain value for each stack argument.
  for (SDNode::use_iterator U = getEntryNode().getNode()->use_begin(),
       UE = getEntryNode().getNode()->use_end(); U != UE; ++U)
    if (LoadSDNode *L = dyn_cast<LoadSDNode>(*U))
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(L->getBasePtr()))
        if (FI->getIndex() < 0)
          ArgChains.push_back(SDValue(L, 1));

  // Build a tokenfactor for all the chains.
  return getNode(ISD::TokenFactor, Chain.getDebugLoc(), MVT::Other,
                 &ArgChains[0], ArgChains.size());
}

/// SplatByte - Distribute ByteVal over NumBits bits.
static APInt SplatByte(unsigned NumBits, uint8_t ByteVal) {
  APInt Val = APInt(NumBits, ByteVal);
  unsigned Shift = 8;
  for (unsigned i = NumBits; i > 8; i >>= 1) {
    Val = (Val << Shift) | Val;
    Shift <<= 1;
  }
  return Val;
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDValue getMemsetValue(SDValue Value, EVT VT, SelectionDAG &DAG,
                              DebugLoc dl) {
  assert(Value.getOpcode() != ISD::UNDEF);

  unsigned NumBits = VT.getScalarType().getSizeInBits();
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    APInt Val = SplatByte(NumBits, C->getZExtValue() & 255);
    if (VT.isInteger())
      return DAG.getConstant(Val, VT);
    return DAG.getConstantFP(APFloat(Val), VT);
  }

  Value = DAG.getNode(ISD::ZERO_EXTEND, dl, VT, Value);
  if (NumBits > 8) {
    // Use a multiplication with 0x010101... to extend the input to the
    // required length.
    APInt Magic = SplatByte(NumBits, 0x01);
    Value = DAG.getNode(ISD::MUL, dl, VT, Value, DAG.getConstant(Magic, VT));
  }

  return Value;
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDValue getMemsetStringVal(EVT VT, DebugLoc dl, SelectionDAG &DAG,
                                  const TargetLowering &TLI,
                                  std::string &Str, unsigned Offset) {
  // Handle vector with all elements zero.
  if (Str.empty()) {
    if (VT.isInteger())
      return DAG.getConstant(0, VT);
    else if (VT == MVT::f32 || VT == MVT::f64)
      return DAG.getConstantFP(0.0, VT);
    else if (VT.isVector()) {
      unsigned NumElts = VT.getVectorNumElements();
      MVT EltVT = (VT.getVectorElementType() == MVT::f32) ? MVT::i32 : MVT::i64;
      return DAG.getNode(ISD::BITCAST, dl, VT,
                         DAG.getConstant(0, EVT::getVectorVT(*DAG.getContext(),
                                                             EltVT, NumElts)));
    } else
      llvm_unreachable("Expected type!");
  }

  assert(!VT.isVector() && "Can't handle vector type here!");
  unsigned NumBits = VT.getSizeInBits();
  unsigned MSB = NumBits / 8;
  uint64_t Val = 0;
  if (TLI.isLittleEndian())
    Offset = Offset + MSB - 1;
  for (unsigned i = 0; i != MSB; ++i) {
    Val = (Val << 8) | (unsigned char)Str[Offset];
    Offset += TLI.isLittleEndian() ? -1 : 1;
  }
  return DAG.getConstant(Val, VT);
}

/// getMemBasePlusOffset - Returns base and offset node for the
///
static SDValue getMemBasePlusOffset(SDValue Base, unsigned Offset,
                                      SelectionDAG &DAG) {
  EVT VT = Base.getValueType();
  return DAG.getNode(ISD::ADD, Base.getDebugLoc(),
                     VT, Base, DAG.getConstant(Offset, VT));
}

/// isMemSrcFromString - Returns true if memcpy source is a string constant.
///
static bool isMemSrcFromString(SDValue Src, std::string &Str) {
  unsigned SrcDelta = 0;
  GlobalAddressSDNode *G = NULL;
  if (Src.getOpcode() == ISD::GlobalAddress)
    G = cast<GlobalAddressSDNode>(Src);
  else if (Src.getOpcode() == ISD::ADD &&
           Src.getOperand(0).getOpcode() == ISD::GlobalAddress &&
           Src.getOperand(1).getOpcode() == ISD::Constant) {
    G = cast<GlobalAddressSDNode>(Src.getOperand(0));
    SrcDelta = cast<ConstantSDNode>(Src.getOperand(1))->getZExtValue();
  }
  if (!G)
    return false;

  const GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getGlobal());
  if (GV && GetConstantStringInfo(GV, Str, SrcDelta, false))
    return true;

  return false;
}

/// FindOptimalMemOpLowering - Determines the optimial series memory ops
/// to replace the memset / memcpy. Return true if the number of memory ops
/// is below the threshold. It returns the types of the sequence of
/// memory ops to perform memset / memcpy by reference.
static bool FindOptimalMemOpLowering(std::vector<EVT> &MemOps,
                                     unsigned Limit, uint64_t Size,
                                     unsigned DstAlign, unsigned SrcAlign,
                                     bool NonScalarIntSafe,
                                     bool MemcpyStrSrc,
                                     SelectionDAG &DAG,
                                     const TargetLowering &TLI) {
  assert((SrcAlign == 0 || SrcAlign >= DstAlign) &&
         "Expecting memcpy / memset source to meet alignment requirement!");
  // If 'SrcAlign' is zero, that means the memory operation does not need to
  // load the value, i.e. memset or memcpy from constant string. Otherwise,
  // it's the inferred alignment of the source. 'DstAlign', on the other hand,
  // is the specified alignment of the memory operation. If it is zero, that
  // means it's possible to change the alignment of the destination.
  // 'MemcpyStrSrc' indicates whether the memcpy source is constant so it does
  // not need to be loaded.
  EVT VT = TLI.getOptimalMemOpType(Size, DstAlign, SrcAlign,
                                   NonScalarIntSafe, MemcpyStrSrc,
                                   DAG.getMachineFunction());

  if (VT == MVT::Other) {
    if (DstAlign >= TLI.getTargetData()->getPointerPrefAlignment() ||
        TLI.allowsUnalignedMemoryAccesses(VT)) {
      VT = TLI.getPointerTy();
    } else {
      switch (DstAlign & 7) {
      case 0:  VT = MVT::i64; break;
      case 4:  VT = MVT::i32; break;
      case 2:  VT = MVT::i16; break;
      default: VT = MVT::i8;  break;
      }
    }

    MVT LVT = MVT::i64;
    while (!TLI.isTypeLegal(LVT))
      LVT = (MVT::SimpleValueType)(LVT.SimpleTy - 1);
    assert(LVT.isInteger());

    if (VT.bitsGT(LVT))
      VT = LVT;
  }

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned VTSize = VT.getSizeInBits() / 8;
    while (VTSize > Size) {
      // For now, only use non-vector load / store's for the left-over pieces.
      if (VT.isVector() || VT.isFloatingPoint()) {
        VT = MVT::i64;
        while (!TLI.isTypeLegal(VT))
          VT = (MVT::SimpleValueType)(VT.getSimpleVT().SimpleTy - 1);
        VTSize = VT.getSizeInBits() / 8;
      } else {
        // This can result in a type that is not legal on the target, e.g.
        // 1 or 2 bytes on PPC.
        VT = (MVT::SimpleValueType)(VT.getSimpleVT().SimpleTy - 1);
        VTSize >>= 1;
      }
    }

    if (++NumMemOps > Limit)
      return false;
    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

static SDValue getMemcpyLoadsAndStores(SelectionDAG &DAG, DebugLoc dl,
                                       SDValue Chain, SDValue Dst,
                                       SDValue Src, uint64_t Size,
                                       unsigned Align, bool isVol,
                                       bool AlwaysInline,
                                       MachinePointerInfo DstPtrInfo,
                                       MachinePointerInfo SrcPtrInfo) {
  // Turn a memcpy of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memcpy to a series of load and store ops if the size operand falls
  // below a certain threshold.
  // TODO: In the AlwaysInline case, if the size is big then generate a loop
  // rather than maybe a humongous number of loads and stores.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  std::string Str;
  bool CopyFromStr = isMemSrcFromString(Src, Str);
  bool isZeroStr = CopyFromStr && Str.empty();
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemcpy(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align),
                                (isZeroStr ? 0 : SrcAlign),
                                true, CopyFromStr, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned) TLI.getTargetData()->getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  uint64_t SrcOff = 0, DstOff = 0;
  for (unsigned i = 0; i != NumMemOps; ++i) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value, Store;

    if (CopyFromStr &&
        (isZeroStr || (VT.isInteger() && !VT.isVector()))) {
      // It's unlikely a store of a vector immediate can be done in a single
      // instruction. It would require a load from a constantpool first.
      // We only handle zero vectors here.
      // FIXME: Handle other cases where store of vector immediate is done in
      // a single instruction.
      Value = getMemsetStringVal(VT, dl, DAG, TLI, Str, SrcOff);
      Store = DAG.getStore(Chain, dl, Value,
                           getMemBasePlusOffset(Dst, DstOff, DAG),
                           DstPtrInfo.getWithOffset(DstOff), isVol,
                           false, Align);
    } else {
      // The type might not be legal for the target.  This should only happen
      // if the type is smaller than a legal type, as on PPC, so the right
      // thing to do is generate a LoadExt/StoreTrunc pair.  These simplify
      // to Load/Store if NVT==VT.
      // FIXME does the case above also need this?
      EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
      assert(NVT.bitsGE(VT));
      Value = DAG.getExtLoad(ISD::EXTLOAD, dl, NVT, Chain,
                             getMemBasePlusOffset(Src, SrcOff, DAG),
                             SrcPtrInfo.getWithOffset(SrcOff), VT, isVol, false,
                             MinAlign(SrcAlign, SrcOff));
      Store = DAG.getTruncStore(Chain, dl, Value,
                                getMemBasePlusOffset(Dst, DstOff, DAG),
                                DstPtrInfo.getWithOffset(DstOff), VT, isVol,
                                false, Align);
    }
    OutChains.push_back(Store);
    SrcOff += VTSize;
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                     &OutChains[0], OutChains.size());
}

static SDValue getMemmoveLoadsAndStores(SelectionDAG &DAG, DebugLoc dl,
                                        SDValue Chain, SDValue Dst,
                                        SDValue Src, uint64_t Size,
                                        unsigned Align,  bool isVol,
                                        bool AlwaysInline,
                                        MachinePointerInfo DstPtrInfo,
                                        MachinePointerInfo SrcPtrInfo) {
  // Turn a memmove of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memmove to a series of load and store ops if the size operand falls
  // below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemmove(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align),
                                SrcAlign, true, false, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned) TLI.getTargetData()->getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  uint64_t SrcOff = 0, DstOff = 0;
  SmallVector<SDValue, 8> LoadValues;
  SmallVector<SDValue, 8> LoadChains;
  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value, Store;

    Value = DAG.getLoad(VT, dl, Chain,
                        getMemBasePlusOffset(Src, SrcOff, DAG),
                        SrcPtrInfo.getWithOffset(SrcOff), isVol,
                        false, SrcAlign);
    LoadValues.push_back(Value);
    LoadChains.push_back(Value.getValue(1));
    SrcOff += VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                      &LoadChains[0], LoadChains.size());
  OutChains.clear();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value, Store;

    Store = DAG.getStore(Chain, dl, LoadValues[i],
                         getMemBasePlusOffset(Dst, DstOff, DAG),
                         DstPtrInfo.getWithOffset(DstOff), isVol, false, Align);
    OutChains.push_back(Store);
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                     &OutChains[0], OutChains.size());
}

static SDValue getMemsetStores(SelectionDAG &DAG, DebugLoc dl,
                               SDValue Chain, SDValue Dst,
                               SDValue Src, uint64_t Size,
                               unsigned Align, bool isVol,
                               MachinePointerInfo DstPtrInfo) {
  // Turn a memset of undef to nop.
  if (Src.getOpcode() == ISD::UNDEF)
    return Chain;

  // Expand memset to a series of load/store ops if the size operand
  // falls below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool OptSize = MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI->isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  bool NonScalarIntSafe =
    isa<ConstantSDNode>(Src) && cast<ConstantSDNode>(Src)->isNullValue();
  if (!FindOptimalMemOpLowering(MemOps, TLI.getMaxStoresPerMemset(OptSize),
                                Size, (DstAlignCanChange ? 0 : Align), 0,
                                NonScalarIntSafe, false, DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned) TLI.getTargetData()->getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI->getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI->setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  SmallVector<SDValue, 8> OutChains;
  uint64_t DstOff = 0;
  unsigned NumMemOps = MemOps.size();

  // Find the largest store and generate the bit pattern for it.
  EVT LargestVT = MemOps[0];
  for (unsigned i = 1; i < NumMemOps; i++)
    if (MemOps[i].bitsGT(LargestVT))
      LargestVT = MemOps[i];
  SDValue MemSetValue = getMemsetValue(Src, LargestVT, DAG, dl);

  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];

    // If this store is smaller than the largest store see whether we can get
    // the smaller value for free with a truncate.
    SDValue Value = MemSetValue;
    if (VT.bitsLT(LargestVT)) {
      if (!LargestVT.isVector() && !VT.isVector() &&
          TLI.isTruncateFree(LargestVT, VT))
        Value = DAG.getNode(ISD::TRUNCATE, dl, VT, MemSetValue);
      else
        Value = getMemsetValue(Src, VT, DAG, dl);
    }
    assert(Value.getValueType() == VT && "Value with wrong type.");
    SDValue Store = DAG.getStore(Chain, dl, Value,
                                 getMemBasePlusOffset(Dst, DstOff, DAG),
                                 DstPtrInfo.getWithOffset(DstOff),
                                 isVol, false, Align);
    OutChains.push_back(Store);
    DstOff += VT.getSizeInBits() / 8;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                     &OutChains[0], OutChains.size());
}

SDValue SelectionDAG::getMemcpy(SDValue Chain, DebugLoc dl, SDValue Dst,
                                SDValue Src, SDValue Size,
                                unsigned Align, bool isVol, bool AlwaysInline,
                                MachinePointerInfo DstPtrInfo,
                                MachinePointerInfo SrcPtrInfo) {

  // Check to see if we should lower the memcpy to loads and stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memcpy with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result = getMemcpyLoadsAndStores(*this, dl, Chain, Dst, Src,
                                             ConstantSize->getZExtValue(),Align,
                                isVol, false, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memcpy with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDValue Result =
    TSI.EmitTargetCodeForMemcpy(*this, dl, Chain, Dst, Src, Size, Align,
                                isVol, AlwaysInline,
                                DstPtrInfo, SrcPtrInfo);
  if (Result.getNode())
    return Result;

  // If we really need inline code and the target declined to provide it,
  // use a (potentially long) sequence of loads and stores.
  if (AlwaysInline) {
    assert(ConstantSize && "AlwaysInline requires a constant size!");
    return getMemcpyLoadsAndStores(*this, dl, Chain, Dst, Src,
                                   ConstantSize->getZExtValue(), Align, isVol,
                                   true, DstPtrInfo, SrcPtrInfo);
  }

  // FIXME: If the memcpy is volatile (isVol), lowering it to a plain libc
  // memcpy is not guaranteed to be safe. libc memcpys aren't required to
  // respect volatile, so they may do things like read or write memory
  // beyond the given memory regions. But fixing this isn't easy, and most
  // people don't care.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = TLI.getTargetData()->getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME: pass in DebugLoc
  std::pair<SDValue,SDValue> CallResult =
    TLI.LowerCallTo(Chain, Type::getVoidTy(*getContext()),
                    false, false, false, false, 0,
                    TLI.getLibcallCallingConv(RTLIB::MEMCPY), false,
                    /*isReturnValueUsed=*/false,
                    getExternalSymbol(TLI.getLibcallName(RTLIB::MEMCPY),
                                      TLI.getPointerTy()),
                    Args, *this, dl);
  return CallResult.second;
}

SDValue SelectionDAG::getMemmove(SDValue Chain, DebugLoc dl, SDValue Dst,
                                 SDValue Src, SDValue Size,
                                 unsigned Align, bool isVol,
                                 MachinePointerInfo DstPtrInfo,
                                 MachinePointerInfo SrcPtrInfo) {

  // Check to see if we should lower the memmove to loads and stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memmove with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result =
      getMemmoveLoadsAndStores(*this, dl, Chain, Dst, Src,
                               ConstantSize->getZExtValue(), Align, isVol,
                               false, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memmove with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDValue Result =
    TSI.EmitTargetCodeForMemmove(*this, dl, Chain, Dst, Src, Size, Align, isVol,
                                 DstPtrInfo, SrcPtrInfo);
  if (Result.getNode())
    return Result;

  // FIXME: If the memmove is volatile, lowering it to plain libc memmove may
  // not be safe.  See memcpy above for more details.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = TLI.getTargetData()->getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME:  pass in DebugLoc
  std::pair<SDValue,SDValue> CallResult =
    TLI.LowerCallTo(Chain, Type::getVoidTy(*getContext()),
                    false, false, false, false, 0,
                    TLI.getLibcallCallingConv(RTLIB::MEMMOVE), false,
                    /*isReturnValueUsed=*/false,
                    getExternalSymbol(TLI.getLibcallName(RTLIB::MEMMOVE),
                                      TLI.getPointerTy()),
                    Args, *this, dl);
  return CallResult.second;
}

SDValue SelectionDAG::getMemset(SDValue Chain, DebugLoc dl, SDValue Dst,
                                SDValue Src, SDValue Size,
                                unsigned Align, bool isVol,
                                MachinePointerInfo DstPtrInfo) {

  // Check to see if we should lower the memset to stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memset with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDValue Result =
      getMemsetStores(*this, dl, Chain, Dst, Src, ConstantSize->getZExtValue(),
                      Align, isVol, DstPtrInfo);

    if (Result.getNode())
      return Result;
  }

  // Then check to see if we should lower the memset with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDValue Result =
    TSI.EmitTargetCodeForMemset(*this, dl, Chain, Dst, Src, Size, Align, isVol,
                                DstPtrInfo);
  if (Result.getNode())
    return Result;

  // Emit a library call.
  Type *IntPtrTy = TLI.getTargetData()->getIntPtrType(*getContext());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Dst; Entry.Ty = IntPtrTy;
  Args.push_back(Entry);
  // Extend or truncate the argument to be an i32 value for the call.
  if (Src.getValueType().bitsGT(MVT::i32))
    Src = getNode(ISD::TRUNCATE, dl, MVT::i32, Src);
  else
    Src = getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Src);
  Entry.Node = Src;
  Entry.Ty = Type::getInt32Ty(*getContext());
  Entry.isSExt = true;
  Args.push_back(Entry);
  Entry.Node = Size;
  Entry.Ty = IntPtrTy;
  Entry.isSExt = false;
  Args.push_back(Entry);
  // FIXME: pass in DebugLoc
  std::pair<SDValue,SDValue> CallResult =
    TLI.LowerCallTo(Chain, Type::getVoidTy(*getContext()),
                    false, false, false, false, 0,
                    TLI.getLibcallCallingConv(RTLIB::MEMSET), false,
                    /*isReturnValueUsed=*/false,
                    getExternalSymbol(TLI.getLibcallName(RTLIB::MEMSET),
                                      TLI.getPointerTy()),
                    Args, *this, dl);
  return CallResult.second;
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                SDValue Chain, SDValue Ptr, SDValue Cmp,
                                SDValue Swp, MachinePointerInfo PtrInfo,
                                unsigned Alignment,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {                                
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  unsigned Flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;

  // For now, atomics are considered to be volatile always.
  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  Flags |= MachineMemOperand::MOVolatile;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Alignment);

  return getAtomic(Opcode, dl, MemVT, Chain, Ptr, Cmp, Swp, MMO,
                   Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                SDValue Chain,
                                SDValue Ptr, SDValue Cmp,
                                SDValue Swp, MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_CMP_SWAP && "Invalid Atomic Op");
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");

  EVT VT = Cmp.getValueType();

  SDVTList VTs = getVTList(VT, MVT::Other);
  FoldingSetNodeID ID;
  ID.AddInteger(MemVT.getRawBits());
  SDValue Ops[] = {Chain, Ptr, Cmp, Swp};
  AddNodeIDNode(ID, Opcode, VTs, Ops, 4);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<AtomicSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) AtomicSDNode(Opcode, dl, VTs, MemVT, Chain,
                                               Ptr, Cmp, Swp, MMO, Ordering,
                                               SynchScope);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                SDValue Chain,
                                SDValue Ptr, SDValue Val,
                                const Value* PtrVal,
                                unsigned Alignment,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  // A monotonic store does not load; a release store "loads" in the sense
  // that other stores cannot be sunk past it.
  // (An atomicrmw obviously both loads and stores.)
  unsigned Flags = MachineMemOperand::MOStore;
  if (Opcode != ISD::ATOMIC_STORE || Ordering > Monotonic)
    Flags |= MachineMemOperand::MOLoad;

  // For now, atomics are considered to be volatile always.
  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  Flags |= MachineMemOperand::MOVolatile;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(PtrVal), Flags,
                            MemVT.getStoreSize(), Alignment);

  return getAtomic(Opcode, dl, MemVT, Chain, Ptr, Val, MMO,
                   Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                SDValue Chain,
                                SDValue Ptr, SDValue Val,
                                MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  assert((Opcode == ISD::ATOMIC_LOAD_ADD ||
          Opcode == ISD::ATOMIC_LOAD_SUB ||
          Opcode == ISD::ATOMIC_LOAD_AND ||
          Opcode == ISD::ATOMIC_LOAD_OR ||
          Opcode == ISD::ATOMIC_LOAD_XOR ||
          Opcode == ISD::ATOMIC_LOAD_NAND ||
          Opcode == ISD::ATOMIC_LOAD_MIN ||
          Opcode == ISD::ATOMIC_LOAD_MAX ||
          Opcode == ISD::ATOMIC_LOAD_UMIN ||
          Opcode == ISD::ATOMIC_LOAD_UMAX ||
          Opcode == ISD::ATOMIC_SWAP ||
          Opcode == ISD::ATOMIC_STORE) &&
         "Invalid Atomic Op");

  EVT VT = Val.getValueType();

  SDVTList VTs = Opcode == ISD::ATOMIC_STORE ? getVTList(MVT::Other) :
                                               getVTList(VT, MVT::Other);
  FoldingSetNodeID ID;
  ID.AddInteger(MemVT.getRawBits());
  SDValue Ops[] = {Chain, Ptr, Val};
  AddNodeIDNode(ID, Opcode, VTs, Ops, 3);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<AtomicSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) AtomicSDNode(Opcode, dl, VTs, MemVT, Chain,
                                               Ptr, Val, MMO,
                                               Ordering, SynchScope);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                EVT VT, SDValue Chain,
                                SDValue Ptr,
                                const Value* PtrVal,
                                unsigned Alignment,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  // A monotonic load does not store; an acquire load "stores" in the sense
  // that other loads cannot be hoisted past it.
  unsigned Flags = MachineMemOperand::MOLoad;
  if (Ordering > Monotonic)
    Flags |= MachineMemOperand::MOStore;

  // For now, atomics are considered to be volatile always.
  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  Flags |= MachineMemOperand::MOVolatile;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(PtrVal), Flags,
                            MemVT.getStoreSize(), Alignment);

  return getAtomic(Opcode, dl, MemVT, VT, Chain, Ptr, MMO,
                   Ordering, SynchScope);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, DebugLoc dl, EVT MemVT,
                                EVT VT, SDValue Chain,
                                SDValue Ptr,
                                MachineMemOperand *MMO,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_LOAD && "Invalid Atomic Op");

  SDVTList VTs = getVTList(VT, MVT::Other);
  FoldingSetNodeID ID;
  ID.AddInteger(MemVT.getRawBits());
  SDValue Ops[] = {Chain, Ptr};
  AddNodeIDNode(ID, Opcode, VTs, Ops, 2);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<AtomicSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) AtomicSDNode(Opcode, dl, VTs, MemVT, Chain,
                                               Ptr, MMO, Ordering, SynchScope);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

/// getMergeValues - Create a MERGE_VALUES node from the given operands.
SDValue SelectionDAG::getMergeValues(const SDValue *Ops, unsigned NumOps,
                                     DebugLoc dl) {
  if (NumOps == 1)
    return Ops[0];

  SmallVector<EVT, 4> VTs;
  VTs.reserve(NumOps);
  for (unsigned i = 0; i < NumOps; ++i)
    VTs.push_back(Ops[i].getValueType());
  return getNode(ISD::MERGE_VALUES, dl, getVTList(&VTs[0], NumOps),
                 Ops, NumOps);
}

SDValue
SelectionDAG::getMemIntrinsicNode(unsigned Opcode, DebugLoc dl,
                                  const EVT *VTs, unsigned NumVTs,
                                  const SDValue *Ops, unsigned NumOps,
                                  EVT MemVT, MachinePointerInfo PtrInfo,
                                  unsigned Align, bool Vol,
                                  bool ReadMem, bool WriteMem) {
  return getMemIntrinsicNode(Opcode, dl, makeVTList(VTs, NumVTs), Ops, NumOps,
                             MemVT, PtrInfo, Align, Vol,
                             ReadMem, WriteMem);
}

SDValue
SelectionDAG::getMemIntrinsicNode(unsigned Opcode, DebugLoc dl, SDVTList VTList,
                                  const SDValue *Ops, unsigned NumOps,
                                  EVT MemVT, MachinePointerInfo PtrInfo,
                                  unsigned Align, bool Vol,
                                  bool ReadMem, bool WriteMem) {
  if (Align == 0)  // Ensure that codegen never sees alignment 0
    Align = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  unsigned Flags = 0;
  if (WriteMem)
    Flags |= MachineMemOperand::MOStore;
  if (ReadMem)
    Flags |= MachineMemOperand::MOLoad;
  if (Vol)
    Flags |= MachineMemOperand::MOVolatile;
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Align);

  return getMemIntrinsicNode(Opcode, dl, VTList, Ops, NumOps, MemVT, MMO);
}

SDValue
SelectionDAG::getMemIntrinsicNode(unsigned Opcode, DebugLoc dl, SDVTList VTList,
                                  const SDValue *Ops, unsigned NumOps,
                                  EVT MemVT, MachineMemOperand *MMO) {
  assert((Opcode == ISD::INTRINSIC_VOID ||
          Opcode == ISD::INTRINSIC_W_CHAIN ||
          Opcode == ISD::PREFETCH ||
          (Opcode <= INT_MAX &&
           (int)Opcode >= ISD::FIRST_TARGET_MEMORY_OPCODE)) &&
         "Opcode is not a memory-accessing opcode!");

  // Memoize the node unless it returns a flag.
  MemIntrinsicSDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
      cast<MemIntrinsicSDNode>(E)->refineAlignment(MMO);
      return SDValue(E, 0);
    }

    N = new (NodeAllocator) MemIntrinsicSDNode(Opcode, dl, VTList, Ops, NumOps,
                                               MemVT, MMO);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) MemIntrinsicSDNode(Opcode, dl, VTList, Ops, NumOps,
                                               MemVT, MMO);
  }
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SDValue Ptr, int64_t Offset = 0) {
  // If this is FI+Offset, we can model it.
  if (const FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr))
    return MachinePointerInfo::getFixedStack(FI->getIndex(), Offset);

  // If this is (FI+Offset1)+Offset2, we can model it.
  if (Ptr.getOpcode() != ISD::ADD ||
      !isa<ConstantSDNode>(Ptr.getOperand(1)) ||
      !isa<FrameIndexSDNode>(Ptr.getOperand(0)))
    return MachinePointerInfo();

  int FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
  return MachinePointerInfo::getFixedStack(FI, Offset+
                       cast<ConstantSDNode>(Ptr.getOperand(1))->getSExtValue());
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SDValue Ptr, SDValue OffsetOp) {
  // If the 'Offset' value isn't a constant, we can't handle this.
  if (ConstantSDNode *OffsetNode = dyn_cast<ConstantSDNode>(OffsetOp))
    return InferPointerInfo(Ptr, OffsetNode->getSExtValue());
  if (OffsetOp.getOpcode() == ISD::UNDEF)
    return InferPointerInfo(Ptr);
  return MachinePointerInfo();
}


SDValue
SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                      EVT VT, DebugLoc dl, SDValue Chain,
                      SDValue Ptr, SDValue Offset,
                      MachinePointerInfo PtrInfo, EVT MemVT,
                      bool isVolatile, bool isNonTemporal,
                      unsigned Alignment, const MDNode *TBAAInfo) {
  assert(Chain.getValueType() == MVT::Other && 
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(VT);

  unsigned Flags = MachineMemOperand::MOLoad;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;

  // If we don't have a PtrInfo, infer the trivial frame index case to simplify
  // clients.
  if (PtrInfo.V == 0)
    PtrInfo = InferPointerInfo(Ptr, Offset);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Alignment,
                            TBAAInfo);
  return getLoad(AM, ExtType, VT, dl, Chain, Ptr, Offset, MemVT, MMO);
}

SDValue
SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                      EVT VT, DebugLoc dl, SDValue Chain,
                      SDValue Ptr, SDValue Offset, EVT MemVT,
                      MachineMemOperand *MMO) {
  if (VT == MemVT) {
    ExtType = ISD::NON_EXTLOAD;
  } else if (ExtType == ISD::NON_EXTLOAD) {
    assert(VT == MemVT && "Non-extending load from different memory type!");
  } else {
    // Extending load.
    assert(MemVT.getScalarType().bitsLT(VT.getScalarType()) &&
           "Should only be an extending load, not truncating!");
    assert(VT.isInteger() == MemVT.isInteger() &&
           "Cannot convert from FP to Int or Int -> FP!");
    assert(VT.isVector() == MemVT.isVector() &&
           "Cannot use trunc store to convert to or from a vector!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() == MemVT.getVectorNumElements()) &&
           "Cannot use trunc store to change the number of vector elements!");
  }

  bool Indexed = AM != ISD::UNINDEXED;
  assert((Indexed || Offset.getOpcode() == ISD::UNDEF) &&
         "Unindexed load with an offset!");

  SDVTList VTs = Indexed ?
    getVTList(VT, Ptr.getValueType(), MVT::Other) : getVTList(VT, MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops, 3);
  ID.AddInteger(MemVT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(ExtType, AM, MMO->isVolatile(),
                                     MMO->isNonTemporal()));
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<LoadSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) LoadSDNode(Ops, dl, VTs, AM, ExtType,
                                             MemVT, MMO);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getLoad(EVT VT, DebugLoc dl,
                              SDValue Chain, SDValue Ptr,
                              MachinePointerInfo PtrInfo,
                              bool isVolatile, bool isNonTemporal,
                              unsigned Alignment, const MDNode *TBAAInfo) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, dl, Chain, Ptr, Undef,
                 PtrInfo, VT, isVolatile, isNonTemporal, Alignment, TBAAInfo);
}

SDValue SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, DebugLoc dl, EVT VT,
                                 SDValue Chain, SDValue Ptr,
                                 MachinePointerInfo PtrInfo, EVT MemVT,
                                 bool isVolatile, bool isNonTemporal,
                                 unsigned Alignment, const MDNode *TBAAInfo) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, dl, Chain, Ptr, Undef,
                 PtrInfo, MemVT, isVolatile, isNonTemporal, Alignment,
                 TBAAInfo);
}


SDValue
SelectionDAG::getIndexedLoad(SDValue OrigLoad, DebugLoc dl, SDValue Base,
                             SDValue Offset, ISD::MemIndexedMode AM) {
  LoadSDNode *LD = cast<LoadSDNode>(OrigLoad);
  assert(LD->getOffset().getOpcode() == ISD::UNDEF &&
         "Load is already a indexed load!");
  return getLoad(AM, LD->getExtensionType(), OrigLoad.getValueType(), dl,
                 LD->getChain(), Base, Offset, LD->getPointerInfo(),
                 LD->getMemoryVT(),
                 LD->isVolatile(), LD->isNonTemporal(), LD->getAlignment());
}

SDValue SelectionDAG::getStore(SDValue Chain, DebugLoc dl, SDValue Val,
                               SDValue Ptr, MachinePointerInfo PtrInfo,
                               bool isVolatile, bool isNonTemporal,
                               unsigned Alignment, const MDNode *TBAAInfo) {
  assert(Chain.getValueType() == MVT::Other && 
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(Val.getValueType());

  unsigned Flags = MachineMemOperand::MOStore;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;

  if (PtrInfo.V == 0)
    PtrInfo = InferPointerInfo(Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags,
                            Val.getValueType().getStoreSize(), Alignment,
                            TBAAInfo);

  return getStore(Chain, dl, Val, Ptr, MMO);
}

SDValue SelectionDAG::getStore(SDValue Chain, DebugLoc dl, SDValue Val,
                               SDValue Ptr, MachineMemOperand *MMO) {
  assert(Chain.getValueType() == MVT::Other && 
        "Invalid chain type");
  EVT VT = Val.getValueType();
  SDVTList VTs = getVTList(MVT::Other);
  SDValue Undef = getUNDEF(Ptr.getValueType());
  SDValue Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(false, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal()));
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl, VTs, ISD::UNINDEXED,
                                              false, VT, MMO);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, DebugLoc dl, SDValue Val,
                                    SDValue Ptr, MachinePointerInfo PtrInfo,
                                    EVT SVT,bool isVolatile, bool isNonTemporal,
                                    unsigned Alignment,
                                    const MDNode *TBAAInfo) {
  assert(Chain.getValueType() == MVT::Other && 
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(SVT);

  unsigned Flags = MachineMemOperand::MOStore;
  if (isVolatile)
    Flags |= MachineMemOperand::MOVolatile;
  if (isNonTemporal)
    Flags |= MachineMemOperand::MONonTemporal;

  if (PtrInfo.V == 0)
    PtrInfo = InferPointerInfo(Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, SVT.getStoreSize(), Alignment,
                            TBAAInfo);

  return getTruncStore(Chain, dl, Val, Ptr, SVT, MMO);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, DebugLoc dl, SDValue Val,
                                    SDValue Ptr, EVT SVT,
                                    MachineMemOperand *MMO) {
  EVT VT = Val.getValueType();

  assert(Chain.getValueType() == MVT::Other && 
        "Invalid chain type");
  if (VT == SVT)
    return getStore(Chain, dl, Val, Ptr, MMO);

  assert(SVT.getScalarType().bitsLT(VT.getScalarType()) &&
         "Should only be a truncating store, not extending!");
  assert(VT.isInteger() == SVT.isInteger() &&
         "Can't do FP-INT conversion!");
  assert(VT.isVector() == SVT.isVector() &&
         "Cannot use trunc store to convert to or from a vector!");
  assert((!VT.isVector() ||
          VT.getVectorNumElements() == SVT.getVectorNumElements()) &&
         "Cannot use trunc store to change the number of vector elements!");

  SDVTList VTs = getVTList(MVT::Other);
  SDValue Undef = getUNDEF(Ptr.getValueType());
  SDValue Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(SVT.getRawBits());
  ID.AddInteger(encodeMemSDNodeFlags(true, ISD::UNINDEXED, MMO->isVolatile(),
                                     MMO->isNonTemporal()));
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl, VTs, ISD::UNINDEXED,
                                              true, SVT, MMO);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue
SelectionDAG::getIndexedStore(SDValue OrigStore, DebugLoc dl, SDValue Base,
                              SDValue Offset, ISD::MemIndexedMode AM) {
  StoreSDNode *ST = cast<StoreSDNode>(OrigStore);
  assert(ST->getOffset().getOpcode() == ISD::UNDEF &&
         "Store is already a indexed store!");
  SDVTList VTs = getVTList(Base.getValueType(), MVT::Other);
  SDValue Ops[] = { ST->getChain(), ST->getValue(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(ST->getMemoryVT().getRawBits());
  ID.AddInteger(ST->getRawSubclassData());
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  SDNode *N = new (NodeAllocator) StoreSDNode(Ops, dl, VTs, AM,
                                              ST->isTruncatingStore(),
                                              ST->getMemoryVT(),
                                              ST->getMemOperand());
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getVAArg(EVT VT, DebugLoc dl,
                               SDValue Chain, SDValue Ptr,
                               SDValue SV,
                               unsigned Align) {
  SDValue Ops[] = { Chain, Ptr, SV, getTargetConstant(Align, MVT::i32) };
  return getNode(ISD::VAARG, dl, getVTList(VT, MVT::Other), Ops, 4);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              const SDUse *Ops, unsigned NumOps) {
  switch (NumOps) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, Ops[0]);
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  // Copy from an SDUse array into an SDValue array for use with
  // the regular getNode logic.
  SmallVector<SDValue, 8> NewOps(Ops, Ops + NumOps);
  return getNode(Opcode, DL, VT, &NewOps[0], NumOps);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, EVT VT,
                              const SDValue *Ops, unsigned NumOps) {
  switch (NumOps) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, Ops[0]);
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  switch (Opcode) {
  default: break;
  case ISD::SELECT_CC: {
    assert(NumOps == 5 && "SELECT_CC takes 5 operands!");
    assert(Ops[0].getValueType() == Ops[1].getValueType() &&
           "LHS and RHS of condition must have same type!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "True and False arms of SelectCC must have same type!");
    assert(Ops[2].getValueType() == VT &&
           "select_cc node must be of same type as true and false value!");
    break;
  }
  case ISD::BR_CC: {
    assert(NumOps == 5 && "BR_CC takes 5 operands!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "LHS/RHS of comparison should match types!");
    break;
  }
  }

  // Memoize nodes.
  SDNode *N;
  SDVTList VTs = getVTList(VT);

  if (VT != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, NumOps);
    void *IP = 0;

    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDValue(E, 0);

    N = new (NodeAllocator) SDNode(Opcode, DL, VTs, Ops, NumOps);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new (NodeAllocator) SDNode(Opcode, DL, VTs, Ops, NumOps);
  }

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL,
                              const std::vector<EVT> &ResultTys,
                              const SDValue *Ops, unsigned NumOps) {
  return getNode(Opcode, DL, getVTList(&ResultTys[0], ResultTys.size()),
                 Ops, NumOps);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL,
                              const EVT *VTs, unsigned NumVTs,
                              const SDValue *Ops, unsigned NumOps) {
  if (NumVTs == 1)
    return getNode(Opcode, DL, VTs[0], Ops, NumOps);
  return getNode(Opcode, DL, makeVTList(VTs, NumVTs), Ops, NumOps);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              const SDValue *Ops, unsigned NumOps) {
  if (VTList.NumVTs == 1)
    return getNode(Opcode, DL, VTList.VTs[0], Ops, NumOps);

#if 0
  switch (Opcode) {
  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SHL_PARTS:
    if (N3.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N3.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, DL, VT, N1, N2, N3.getOperand(0));
    else if (N3.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N3.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = VT.getScalarType().getSizeInBits()*2;
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, DL, VT, N1, N2, N3.getOperand(0));
      }
    break;
  }
#endif

  // Memoize the node unless it returns a flag.
  SDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDValue(E, 0);

    if (NumOps == 1) {
      N = new (NodeAllocator) UnarySDNode(Opcode, DL, VTList, Ops[0]);
    } else if (NumOps == 2) {
      N = new (NodeAllocator) BinarySDNode(Opcode, DL, VTList, Ops[0], Ops[1]);
    } else if (NumOps == 3) {
      N = new (NodeAllocator) TernarySDNode(Opcode, DL, VTList, Ops[0], Ops[1],
                                            Ops[2]);
    } else {
      N = new (NodeAllocator) SDNode(Opcode, DL, VTList, Ops, NumOps);
    }
    CSEMap.InsertNode(N, IP);
  } else {
    if (NumOps == 1) {
      N = new (NodeAllocator) UnarySDNode(Opcode, DL, VTList, Ops[0]);
    } else if (NumOps == 2) {
      N = new (NodeAllocator) BinarySDNode(Opcode, DL, VTList, Ops[0], Ops[1]);
    } else if (NumOps == 3) {
      N = new (NodeAllocator) TernarySDNode(Opcode, DL, VTList, Ops[0], Ops[1],
                                            Ops[2]);
    } else {
      N = new (NodeAllocator) SDNode(Opcode, DL, VTList, Ops, NumOps);
    }
  }
  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifySDNode(N);
#endif
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList) {
  return getNode(Opcode, DL, VTList, 0, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              SDValue N1) {
  SDValue Ops[] = { N1 };
  return getNode(Opcode, DL, VTList, Ops, 1);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2) {
  SDValue Ops[] = { N1, N2 };
  return getNode(Opcode, DL, VTList, Ops, 2);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3) {
  SDValue Ops[] = { N1, N2, N3 };
  return getNode(Opcode, DL, VTList, Ops, 3);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VTList, Ops, 4);
}

SDValue SelectionDAG::getNode(unsigned Opcode, DebugLoc DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3,
                              SDValue N4, SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VTList, Ops, 5);
}

SDVTList SelectionDAG::getVTList(EVT VT) {
  return makeVTList(SDNode::getValueTypeList(VT), 1);
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2) {
  for (std::vector<SDVTList>::reverse_iterator I = VTList.rbegin(),
       E = VTList.rend(); I != E; ++I)
    if (I->NumVTs == 2 && I->VTs[0] == VT1 && I->VTs[1] == VT2)
      return *I;

  EVT *Array = Allocator.Allocate<EVT>(2);
  Array[0] = VT1;
  Array[1] = VT2;
  SDVTList Result = makeVTList(Array, 2);
  VTList.push_back(Result);
  return Result;
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3) {
  for (std::vector<SDVTList>::reverse_iterator I = VTList.rbegin(),
       E = VTList.rend(); I != E; ++I)
    if (I->NumVTs == 3 && I->VTs[0] == VT1 && I->VTs[1] == VT2 &&
                          I->VTs[2] == VT3)
      return *I;

  EVT *Array = Allocator.Allocate<EVT>(3);
  Array[0] = VT1;
  Array[1] = VT2;
  Array[2] = VT3;
  SDVTList Result = makeVTList(Array, 3);
  VTList.push_back(Result);
  return Result;
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3, EVT VT4) {
  for (std::vector<SDVTList>::reverse_iterator I = VTList.rbegin(),
       E = VTList.rend(); I != E; ++I)
    if (I->NumVTs == 4 && I->VTs[0] == VT1 && I->VTs[1] == VT2 &&
                          I->VTs[2] == VT3 && I->VTs[3] == VT4)
      return *I;

  EVT *Array = Allocator.Allocate<EVT>(4);
  Array[0] = VT1;
  Array[1] = VT2;
  Array[2] = VT3;
  Array[3] = VT4;
  SDVTList Result = makeVTList(Array, 4);
  VTList.push_back(Result);
  return Result;
}

SDVTList SelectionDAG::getVTList(const EVT *VTs, unsigned NumVTs) {
  switch (NumVTs) {
    case 0: llvm_unreachable("Cannot have nodes without results!");
    case 1: return getVTList(VTs[0]);
    case 2: return getVTList(VTs[0], VTs[1]);
    case 3: return getVTList(VTs[0], VTs[1], VTs[2]);
    case 4: return getVTList(VTs[0], VTs[1], VTs[2], VTs[3]);
    default: break;
  }

  for (std::vector<SDVTList>::reverse_iterator I = VTList.rbegin(),
       E = VTList.rend(); I != E; ++I) {
    if (I->NumVTs != NumVTs || VTs[0] != I->VTs[0] || VTs[1] != I->VTs[1])
      continue;

    bool NoMatch = false;
    for (unsigned i = 2; i != NumVTs; ++i)
      if (VTs[i] != I->VTs[i]) {
        NoMatch = true;
        break;
      }
    if (!NoMatch)
      return *I;
  }

  EVT *Array = Allocator.Allocate<EVT>(NumVTs);
  std::copy(VTs, VTs+NumVTs, Array);
  SDVTList Result = makeVTList(Array, NumVTs);
  VTList.push_back(Result);
  return Result;
}


/// UpdateNodeOperands - *Mutate* the specified node in-place to have the
/// specified operands.  If the resultant node already exists in the DAG,
/// this does not modify the specified node, instead it returns the node that
/// already exists.  If the resultant node does not exist in the DAG, the
/// input node is returned.  As a degenerate case, if you specify the same
/// input operands as the node already has, the input node is returned.
SDNode *SelectionDAG::UpdateNodeOperands(SDNode *N, SDValue Op) {
  assert(N->getNumOperands() == 1 && "Update with wrong number of operands");

  // Check to see if there is no change.
  if (Op == N->getOperand(0)) return N;

  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = 0;

  // Now we update the operands.
  N->OperandList[0].set(Op);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

SDNode *SelectionDAG::UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2) {
  assert(N->getNumOperands() == 2 && "Update with wrong number of operands");

  // Check to see if there is no change.
  if (Op1 == N->getOperand(0) && Op2 == N->getOperand(1))
    return N;   // No operands changed, just return the input node.

  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op1, Op2, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = 0;

  // Now we update the operands.
  if (N->OperandList[0] != Op1)
    N->OperandList[0].set(Op1);
  if (N->OperandList[1] != Op2)
    N->OperandList[1].set(Op2);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2, SDValue Op3) {
  SDValue Ops[] = { Op1, Op2, Op3 };
  return UpdateNodeOperands(N, Ops, 3);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4 };
  return UpdateNodeOperands(N, Ops, 4);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4, SDValue Op5) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return UpdateNodeOperands(N, Ops, 5);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, const SDValue *Ops, unsigned NumOps) {
  assert(N->getNumOperands() == NumOps &&
         "Update with wrong number of operands");

  // Check to see if there is no change.
  bool AnyChange = false;
  for (unsigned i = 0; i != NumOps; ++i) {
    if (Ops[i] != N->getOperand(i)) {
      AnyChange = true;
      break;
    }
  }

  // No operands changed, just return the input node.
  if (!AnyChange) return N;

  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Ops, NumOps, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = 0;

  // Now we update the operands.
  for (unsigned i = 0; i != NumOps; ++i)
    if (N->OperandList[i] != Ops[i])
      N->OperandList[i].set(Ops[i]);

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return N;
}

/// DropOperands - Release the operands and set this node to have
/// zero operands.
void SDNode::DropOperands() {
  // Unlike the code in MorphNodeTo that does this, we don't need to
  // watch for dead nodes here.
  for (op_iterator I = op_begin(), E = op_end(); I != E; ) {
    SDUse &Use = *I++;
    Use.set(SDValue());
  }
}

/// SelectNodeTo - These are wrappers around MorphNodeTo that accept a
/// machine opcode.
///
SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT) {
  SDVTList VTs = getVTList(VT);
  return SelectNodeTo(N, MachineOpc, VTs, 0, 0);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 1);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 2);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 3);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, const SDValue *Ops,
                                   unsigned NumOps) {
  SDVTList VTs = getVTList(VT);
  return SelectNodeTo(N, MachineOpc, VTs, Ops, NumOps);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, const SDValue *Ops,
                                   unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, Ops, NumOps);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, (SDValue *)0, 0);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3,
                                   const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return SelectNodeTo(N, MachineOpc, VTs, Ops, NumOps);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3, EVT VT4,
                                   const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2, VT3, VT4);
  return SelectNodeTo(N, MachineOpc, VTs, Ops, NumOps);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 1);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 2);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1, SDValue Op2,
                                   SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 3);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3,
                                   SDValue Op1, SDValue Op2,
                                   SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops, 3);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   SDVTList VTs, const SDValue *Ops,
                                   unsigned NumOps) {
  N = MorphNodeTo(N, ~MachineOpc, VTs, Ops, NumOps);
  // Reset the NodeID to -1.
  N->setNodeId(-1);
  return N;
}

/// MorphNodeTo - This *mutates* the specified node to have the specified
/// return type, opcode, and operands.
///
/// Note that MorphNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.  Note that the DebugLoc need not be the same.
///
/// Using MorphNodeTo is faster than creating a new node and swapping it in
/// with ReplaceAllUsesWith both because it often avoids allocating a new
/// node, and because it doesn't require CSE recalculation for any of
/// the node's users.
///
SDNode *SelectionDAG::MorphNodeTo(SDNode *N, unsigned Opc,
                                  SDVTList VTs, const SDValue *Ops,
                                  unsigned NumOps) {
  // If an identical node already exists, use it.
  void *IP = 0;
  if (VTs.VTs[VTs.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opc, VTs, Ops, NumOps);
    if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
      return ON;
  }

  if (!RemoveNodeFromCSEMaps(N))
    IP = 0;

  // Start the morphing.
  N->NodeType = Opc;
  N->ValueList = VTs.VTs;
  N->NumValues = VTs.NumVTs;

  // Clear the operands list, updating used nodes to remove this from their
  // use list.  Keep track of any operands that become dead as a result.
  SmallPtrSet<SDNode*, 16> DeadNodeSet;
  for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ) {
    SDUse &Use = *I++;
    SDNode *Used = Use.getNode();
    Use.set(SDValue());
    if (Used->use_empty())
      DeadNodeSet.insert(Used);
  }

  if (MachineSDNode *MN = dyn_cast<MachineSDNode>(N)) {
    // Initialize the memory references information.
    MN->setMemRefs(0, 0);
    // If NumOps is larger than the # of operands we can have in a
    // MachineSDNode, reallocate the operand list.
    if (NumOps > MN->NumOperands || !MN->OperandsNeedDelete) {
      if (MN->OperandsNeedDelete)
        delete[] MN->OperandList;
      if (NumOps > array_lengthof(MN->LocalOperands))
        // We're creating a final node that will live unmorphed for the
        // remainder of the current SelectionDAG iteration, so we can allocate
        // the operands directly out of a pool with no recycling metadata.
        MN->InitOperands(OperandAllocator.Allocate<SDUse>(NumOps),
                         Ops, NumOps);
      else
        MN->InitOperands(MN->LocalOperands, Ops, NumOps);
      MN->OperandsNeedDelete = false;
    } else
      MN->InitOperands(MN->OperandList, Ops, NumOps);
  } else {
    // If NumOps is larger than the # of operands we currently have, reallocate
    // the operand list.
    if (NumOps > N->NumOperands) {
      if (N->OperandsNeedDelete)
        delete[] N->OperandList;
      N->InitOperands(new SDUse[NumOps], Ops, NumOps);
      N->OperandsNeedDelete = true;
    } else
      N->InitOperands(N->OperandList, Ops, NumOps);
  }

  // Delete any nodes that are still dead after adding the uses for the
  // new operands.
  if (!DeadNodeSet.empty()) {
    SmallVector<SDNode *, 16> DeadNodes;
    for (SmallPtrSet<SDNode *, 16>::iterator I = DeadNodeSet.begin(),
         E = DeadNodeSet.end(); I != E; ++I)
      if ((*I)->use_empty())
        DeadNodes.push_back(*I);
    RemoveDeadNodes(DeadNodes);
  }

  if (IP)
    CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}


/// getMachineNode - These are used for target selectors to create a new node
/// with specified return type(s), MachineInstr opcode, and operands.
///
/// Note that getMachineNode returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, 0, 0);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT,
                             SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT,
                             SDValue Op1, SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT,
                             const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, Ops, NumOps);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT1, EVT VT2) {
  SDVTList VTs = getVTList(VT1, VT2);
  return getMachineNode(Opcode, dl, VTs, 0, 0);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, SDValue Op1,
                             SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2,
                             const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2);
  return getMachineNode(Opcode, dl, VTs, Ops, NumOps);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             SDValue Op1, SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops, array_lengthof(Ops));
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             EVT VT1, EVT VT2, EVT VT3,
                             const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return getMachineNode(Opcode, dl, VTs, Ops, NumOps);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl, EVT VT1,
                             EVT VT2, EVT VT3, EVT VT4,
                             const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(VT1, VT2, VT3, VT4);
  return getMachineNode(Opcode, dl, VTs, Ops, NumOps);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc dl,
                             const std::vector<EVT> &ResultTys,
                             const SDValue *Ops, unsigned NumOps) {
  SDVTList VTs = getVTList(&ResultTys[0], ResultTys.size());
  return getMachineNode(Opcode, dl, VTs, Ops, NumOps);
}

MachineSDNode *
SelectionDAG::getMachineNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                             const SDValue *Ops, unsigned NumOps) {
  bool DoCSE = VTs.VTs[VTs.NumVTs-1] != MVT::Glue;
  MachineSDNode *N;
  void *IP = 0;

  if (DoCSE) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, ~Opcode, VTs, Ops, NumOps);
    IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return cast<MachineSDNode>(E);
  }

  // Allocate a new MachineSDNode.
  N = new (NodeAllocator) MachineSDNode(~Opcode, DL, VTs);

  // Initialize the operands list.
  if (NumOps > array_lengthof(N->LocalOperands))
    // We're creating a final node that will live unmorphed for the
    // remainder of the current SelectionDAG iteration, so we can allocate
    // the operands directly out of a pool with no recycling metadata.
    N->InitOperands(OperandAllocator.Allocate<SDUse>(NumOps),
                    Ops, NumOps);
  else
    N->InitOperands(N->LocalOperands, Ops, NumOps);
  N->OperandsNeedDelete = false;

  if (DoCSE)
    CSEMap.InsertNode(N, IP);

  AllNodes.push_back(N);
#ifndef NDEBUG
  VerifyMachineNode(N);
#endif
  return N;
}

/// getTargetExtractSubreg - A convenience function for creating
/// TargetOpcode::EXTRACT_SUBREG nodes.
SDValue
SelectionDAG::getTargetExtractSubreg(int SRIdx, DebugLoc DL, EVT VT,
                                     SDValue Operand) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, MVT::i32);
  SDNode *Subreg = getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL,
                                  VT, Operand, SRIdxVal);
  return SDValue(Subreg, 0);
}

/// getTargetInsertSubreg - A convenience function for creating
/// TargetOpcode::INSERT_SUBREG nodes.
SDValue
SelectionDAG::getTargetInsertSubreg(int SRIdx, DebugLoc DL, EVT VT,
                                    SDValue Operand, SDValue Subreg) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, MVT::i32);
  SDNode *Result = getMachineNode(TargetOpcode::INSERT_SUBREG, DL,
                                  VT, Operand, Subreg, SRIdxVal);
  return SDValue(Result, 0);
}

/// getNodeIfExists - Get the specified node if it's already available, or
/// else return NULL.
SDNode *SelectionDAG::getNodeIfExists(unsigned Opcode, SDVTList VTList,
                                      const SDValue *Ops, unsigned NumOps) {
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return E;
  }
  return NULL;
}

/// getDbgValue - Creates a SDDbgValue node.
///
SDDbgValue *
SelectionDAG::getDbgValue(MDNode *MDPtr, SDNode *N, unsigned R, uint64_t Off,
                          DebugLoc DL, unsigned O) {
  return new (Allocator) SDDbgValue(MDPtr, N, R, Off, DL, O);
}

SDDbgValue *
SelectionDAG::getDbgValue(MDNode *MDPtr, const Value *C, uint64_t Off,
                          DebugLoc DL, unsigned O) {
  return new (Allocator) SDDbgValue(MDPtr, C, Off, DL, O);
}

SDDbgValue *
SelectionDAG::getDbgValue(MDNode *MDPtr, unsigned FI, uint64_t Off,
                          DebugLoc DL, unsigned O) {
  return new (Allocator) SDDbgValue(MDPtr, FI, Off, DL, O);
}

namespace {

/// RAUWUpdateListener - Helper for ReplaceAllUsesWith - When the node
/// pointed to by a use iterator is deleted, increment the use iterator
/// so that it doesn't dangle.
///
/// This class also manages a "downlink" DAGUpdateListener, to forward
/// messages to ReplaceAllUsesWith's callers.
///
class RAUWUpdateListener : public SelectionDAG::DAGUpdateListener {
  SelectionDAG::DAGUpdateListener *DownLink;
  SDNode::use_iterator &UI;
  SDNode::use_iterator &UE;

  virtual void NodeDeleted(SDNode *N, SDNode *E) {
    // Increment the iterator as needed.
    while (UI != UE && N == *UI)
      ++UI;

    // Then forward the message.
    if (DownLink) DownLink->NodeDeleted(N, E);
  }

  virtual void NodeUpdated(SDNode *N) {
    // Just forward the message.
    if (DownLink) DownLink->NodeUpdated(N);
  }

public:
  RAUWUpdateListener(SelectionDAG::DAGUpdateListener *dl,
                     SDNode::use_iterator &ui,
                     SDNode::use_iterator &ue)
    : DownLink(dl), UI(ui), UE(ue) {}
};

}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From has a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDValue FromN, SDValue To,
                                      DAGUpdateListener *UpdateListener) {
  SDNode *From = FromN.getNode();
  assert(From->getNumValues() == 1 && FromN.getResNo() == 0 &&
         "Cannot replace with this method!");
  assert(From != To.getNode() && "Cannot replace uses of with self");

  // Iterate over all the existing uses of From. New uses will be added
  // to the beginning of the use list, which we avoid visiting.
  // This specifically avoids visiting uses of From that arise while the
  // replacement is happening, because any such uses would be the result
  // of CSE: If an existing node looks like From after one of its operands
  // is replaced by To, we don't want to replace of all its users with To
  // too. See PR3018 for more info.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(UpdateListener, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      ++UI;
      Use.set(To);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User, &Listener);
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes that for each value of From, there is a
/// corresponding value in To in the same position with the same type.
///
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, SDNode *To,
                                      DAGUpdateListener *UpdateListener) {
#ifndef NDEBUG
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i)
    assert((!From->hasAnyUseOfValue(i) ||
            From->getValueType(i) == To->getValueType(i)) &&
           "Cannot use this version of ReplaceAllUsesWith!");
#endif

  // Handle the trivial case.
  if (From == To)
    return;

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(UpdateListener, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      ++UI;
      Use.setNode(To);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User, &Listener);
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From,
                                      const SDValue *To,
                                      DAGUpdateListener *UpdateListener) {
  if (From->getNumValues() == 1)  // Handle the simple case efficiently.
    return ReplaceAllUsesWith(SDValue(From, 0), To[0], UpdateListener);

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(UpdateListener, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();
      const SDValue &ToOp = To[Use.getResNo()];
      ++UI;
      Use.set(ToOp);
    } while (UI != UE && *UI == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User, &Listener);
  }
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.getNode() alone.  The Deleted
/// vector is handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDValue From, SDValue To,
                                             DAGUpdateListener *UpdateListener){
  // Handle the really simple, really trivial case efficiently.
  if (From == To) return;

  // Handle the simple, trivial, case efficiently.
  if (From.getNode()->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To, UpdateListener);
    return;
  }

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From.getNode()->use_begin(),
                       UE = From.getNode()->use_end();
  RAUWUpdateListener Listener(UpdateListener, UI, UE);
  while (UI != UE) {
    SDNode *User = *UI;
    bool UserRemovedFromCSEMaps = false;

    // A user can appear in a use list multiple times, and when this
    // happens the uses are usually next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      SDUse &Use = UI.getUse();

      // Skip uses of different values from the same node.
      if (Use.getResNo() != From.getResNo()) {
        ++UI;
        continue;
      }

      // If this node hasn't been modified yet, it's still in the CSE maps,
      // so remove its old self from the CSE maps.
      if (!UserRemovedFromCSEMaps) {
        RemoveNodeFromCSEMaps(User);
        UserRemovedFromCSEMaps = true;
      }

      ++UI;
      Use.set(To);
    } while (UI != UE && *UI == User);

    // We are iterating over all uses of the From node, so if a use
    // doesn't use the specific value, no changes are made.
    if (!UserRemovedFromCSEMaps)
      continue;

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User, &Listener);
  }
}

namespace {
  /// UseMemo - This class is used by SelectionDAG::ReplaceAllUsesOfValuesWith
  /// to record information about a use.
  struct UseMemo {
    SDNode *User;
    unsigned Index;
    SDUse *Use;
  };

  /// operator< - Sort Memos by User.
  bool operator<(const UseMemo &L, const UseMemo &R) {
    return (intptr_t)L.User < (intptr_t)R.User;
  }
}

/// ReplaceAllUsesOfValuesWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.getNode() alone.  The same value
/// may appear in both the From and To list.  The Deleted vector is
/// handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValuesWith(const SDValue *From,
                                              const SDValue *To,
                                              unsigned Num,
                                              DAGUpdateListener *UpdateListener){
  // Handle the simple, trivial case efficiently.
  if (Num == 1)
    return ReplaceAllUsesOfValueWith(*From, *To, UpdateListener);

  // Read up all the uses and make records of them. This helps
  // processing new uses that are introduced during the
  // replacement process.
  SmallVector<UseMemo, 4> Uses;
  for (unsigned i = 0; i != Num; ++i) {
    unsigned FromResNo = From[i].getResNo();
    SDNode *FromNode = From[i].getNode();
    for (SDNode::use_iterator UI = FromNode->use_begin(),
         E = FromNode->use_end(); UI != E; ++UI) {
      SDUse &Use = UI.getUse();
      if (Use.getResNo() == FromResNo) {
        UseMemo Memo = { *UI, i, &Use };
        Uses.push_back(Memo);
      }
    }
  }

  // Sort the uses, so that all the uses from a given User are together.
  std::sort(Uses.begin(), Uses.end());

  for (unsigned UseIndex = 0, UseIndexEnd = Uses.size();
       UseIndex != UseIndexEnd; ) {
    // We know that this user uses some value of From.  If it is the right
    // value, update it.
    SDNode *User = Uses[UseIndex].User;

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(User);

    // The Uses array is sorted, so all the uses for a given User
    // are next to each other in the list.
    // To help reduce the number of CSE recomputations, process all
    // the uses of this user that we can find this way.
    do {
      unsigned i = Uses[UseIndex].Index;
      SDUse &Use = *Uses[UseIndex].Use;
      ++UseIndex;

      Use.set(To[i]);
    } while (UseIndex != UseIndexEnd && Uses[UseIndex].User == User);

    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    AddModifiedNodeToCSEMaps(User, UpdateListener);
  }
}

/// AssignTopologicalOrder - Assign a unique node id for each node in the DAG
/// based on their topological order. It returns the maximum id and a vector
/// of the SDNodes* in assigned order by reference.
unsigned SelectionDAG::AssignTopologicalOrder() {

  unsigned DAGSize = 0;

  // SortedPos tracks the progress of the algorithm. Nodes before it are
  // sorted, nodes after it are unsorted. When the algorithm completes
  // it is at the end of the list.
  allnodes_iterator SortedPos = allnodes_begin();

  // Visit all the nodes. Move nodes with no operands to the front of
  // the list immediately. Annotate nodes that do have operands with their
  // operand count. Before we do this, the Node Id fields of the nodes
  // may contain arbitrary values. After, the Node Id fields for nodes
  // before SortedPos will contain the topological sort index, and the
  // Node Id fields for nodes At SortedPos and after will contain the
  // count of outstanding operands.
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ) {
    SDNode *N = I++;
    checkForCycles(N);
    unsigned Degree = N->getNumOperands();
    if (Degree == 0) {
      // A node with no uses, add it to the result array immediately.
      N->setNodeId(DAGSize++);
      allnodes_iterator Q = N;
      if (Q != SortedPos)
        SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(Q));
      assert(SortedPos != AllNodes.end() && "Overran node list");
      ++SortedPos;
    } else {
      // Temporarily use the Node Id as scratch space for the degree count.
      N->setNodeId(Degree);
    }
  }

  // Visit all the nodes. As we iterate, moves nodes into sorted order,
  // such that by the time the end is reached all nodes will be sorted.
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ++I) {
    SDNode *N = I;
    checkForCycles(N);
    // N is in sorted position, so all its uses have one less operand
    // that needs to be sorted.
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDNode *P = *UI;
      unsigned Degree = P->getNodeId();
      assert(Degree != 0 && "Invalid node degree");
      --Degree;
      if (Degree == 0) {
        // All of P's operands are sorted, so P may sorted now.
        P->setNodeId(DAGSize++);
        if (P != SortedPos)
          SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(P));
        assert(SortedPos != AllNodes.end() && "Overran node list");
        ++SortedPos;
      } else {
        // Update P's outstanding operand count.
        P->setNodeId(Degree);
      }
    }
    if (I == SortedPos) {
#ifndef NDEBUG
      SDNode *S = ++I;
      dbgs() << "Overran sorted position:\n";
      S->dumprFull();
#endif
      llvm_unreachable(0);
    }
  }

  assert(SortedPos == AllNodes.end() &&
         "Topological sort incomplete!");
  assert(AllNodes.front().getOpcode() == ISD::EntryToken &&
         "First node in topological sort is not the entry token!");
  assert(AllNodes.front().getNodeId() == 0 &&
         "First node in topological sort has non-zero id!");
  assert(AllNodes.front().getNumOperands() == 0 &&
         "First node in topological sort has operands!");
  assert(AllNodes.back().getNodeId() == (int)DAGSize-1 &&
         "Last node in topologic sort has unexpected id!");
  assert(AllNodes.back().use_empty() &&
         "Last node in topologic sort has users!");
  assert(DAGSize == allnodes_size() && "Node count mismatch!");
  return DAGSize;
}

/// AssignOrdering - Assign an order to the SDNode.
void SelectionDAG::AssignOrdering(const SDNode *SD, unsigned Order) {
  assert(SD && "Trying to assign an order to a null node!");
  Ordering->add(SD, Order);
}

/// GetOrdering - Get the order for the SDNode.
unsigned SelectionDAG::GetOrdering(const SDNode *SD) const {
  assert(SD && "Trying to get the order of a null node!");
  return Ordering->getOrder(SD);
}

/// AddDbgValue - Add a dbg_value SDNode. If SD is non-null that means the
/// value is produced by SD.
void SelectionDAG::AddDbgValue(SDDbgValue *DB, SDNode *SD, bool isParameter) {
  DbgInfo->add(DB, SD, isParameter);
  if (SD)
    SD->setHasDebugValue(true);
}

/// TransferDbgValues - Transfer SDDbgValues.
void SelectionDAG::TransferDbgValues(SDValue From, SDValue To) {
  if (From == To || !From.getNode()->getHasDebugValue())
    return;
  SDNode *FromNode = From.getNode();
  SDNode *ToNode = To.getNode();
  ArrayRef<SDDbgValue *> DVs = GetDbgValues(FromNode);
  SmallVector<SDDbgValue *, 2> ClonedDVs;
  for (ArrayRef<SDDbgValue *>::iterator I = DVs.begin(), E = DVs.end();
       I != E; ++I) {
    SDDbgValue *Dbg = *I;
    if (Dbg->getKind() == SDDbgValue::SDNODE) {
      SDDbgValue *Clone = getDbgValue(Dbg->getMDPtr(), ToNode, To.getResNo(),
                                      Dbg->getOffset(), Dbg->getDebugLoc(),
                                      Dbg->getOrder());
      ClonedDVs.push_back(Clone);
    }
  }
  for (SmallVector<SDDbgValue *, 2>::iterator I = ClonedDVs.begin(),
         E = ClonedDVs.end(); I != E; ++I)
    AddDbgValue(*I, ToNode, false);
}

//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//

HandleSDNode::~HandleSDNode() {
  DropOperands();
}

GlobalAddressSDNode::GlobalAddressSDNode(unsigned Opc, DebugLoc DL,
                                         const GlobalValue *GA,
                                         EVT VT, int64_t o, unsigned char TF)
  : SDNode(Opc, DL, getSDVTList(VT)), Offset(o), TargetFlags(TF) {
  TheGlobal = GA;
}

MemSDNode::MemSDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, EVT memvt,
                     MachineMemOperand *mmo)
 : SDNode(Opc, dl, VTs), MemoryVT(memvt), MMO(mmo) {
  SubclassData = encodeMemSDNodeFlags(0, ISD::UNINDEXED, MMO->isVolatile(),
                                      MMO->isNonTemporal());
  assert(isVolatile() == MMO->isVolatile() && "Volatile encoding error!");
  assert(isNonTemporal() == MMO->isNonTemporal() &&
         "Non-temporal encoding error!");
  assert(memvt.getStoreSize() == MMO->getSize() && "Size mismatch!");
}

MemSDNode::MemSDNode(unsigned Opc, DebugLoc dl, SDVTList VTs,
                     const SDValue *Ops, unsigned NumOps, EVT memvt,
                     MachineMemOperand *mmo)
   : SDNode(Opc, dl, VTs, Ops, NumOps),
     MemoryVT(memvt), MMO(mmo) {
  SubclassData = encodeMemSDNodeFlags(0, ISD::UNINDEXED, MMO->isVolatile(),
                                      MMO->isNonTemporal());
  assert(isVolatile() == MMO->isVolatile() && "Volatile encoding error!");
  assert(memvt.getStoreSize() == MMO->getSize() && "Size mismatch!");
}

/// Profile - Gather unique data for the node.
///
void SDNode::Profile(FoldingSetNodeID &ID) const {
  AddNodeIDNode(ID, this);
}

namespace {
  struct EVTArray {
    std::vector<EVT> VTs;

    EVTArray() {
      VTs.reserve(MVT::LAST_VALUETYPE);
      for (unsigned i = 0; i < MVT::LAST_VALUETYPE; ++i)
        VTs.push_back(MVT((MVT::SimpleValueType)i));
    }
  };
}

static ManagedStatic<std::set<EVT, EVT::compareRawBits> > EVTs;
static ManagedStatic<EVTArray> SimpleVTArray;
static ManagedStatic<sys::SmartMutex<true> > VTMutex;

/// getValueTypeList - Return a pointer to the specified value type.
///
const EVT *SDNode::getValueTypeList(EVT VT) {
  if (VT.isExtended()) {
    sys::SmartScopedLock<true> Lock(*VTMutex);
    return &(*EVTs->insert(VT).first);
  } else {
    assert(VT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Value type out of range!");
    return &SimpleVTArray->VTs[VT.getSimpleVT().SimpleTy];
  }
}

/// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
/// indicated value.  This method ignores uses of other values defined by this
/// operation.
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  // TODO: Only iterate over uses of a given value of the node
  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    if (UI.getUse().getResNo() == Value) {
      if (NUses == 0)
        return false;
      --NUses;
    }
  }

  // Found exactly the right number of uses?
  return NUses == 0;
}


/// hasAnyUseOfValue - Return true if there are any use of the indicated
/// value. This method ignores uses of other values defined by this operation.
bool SDNode::hasAnyUseOfValue(unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI)
    if (UI.getUse().getResNo() == Value)
      return true;

  return false;
}


/// isOnlyUserOf - Return true if this node is the only use of N.
///
bool SDNode::isOnlyUserOf(SDNode *N) const {
  bool Seen = false;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDNode *User = *I;
    if (User == this)
      Seen = true;
    else
      return false;
  }

  return Seen;
}

/// isOperand - Return true if this node is an operand of N.
///
bool SDValue::isOperandOf(SDNode *N) const {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (*this == N->getOperand(i))
      return true;
  return false;
}

bool SDNode::isOperandOf(SDNode *N) const {
  for (unsigned i = 0, e = N->NumOperands; i != e; ++i)
    if (this == N->OperandList[i].getNode())
      return true;
  return false;
}

/// reachesChainWithoutSideEffects - Return true if this operand (which must
/// be a chain) reaches the specified operand without crossing any
/// side-effecting instructions on any chain path.  In practice, this looks
/// through token factors and non-volatile loads.  In order to remain efficient,
/// this only looks a couple of nodes in, it does not do an exhaustive search.
bool SDValue::reachesChainWithoutSideEffects(SDValue Dest,
                                               unsigned Depth) const {
  if (*this == Dest) return true;

  // Don't search too deeply, we just want to be able to see through
  // TokenFactor's etc.
  if (Depth == 0) return false;

  // If this is a token factor, all inputs to the TF happen in parallel.  If any
  // of the operands of the TF does not reach dest, then we cannot do the xform.
  if (getOpcode() == ISD::TokenFactor) {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!getOperand(i).reachesChainWithoutSideEffects(Dest, Depth-1))
        return false;
    return true;
  }

  // Loads don't have side effects, look through them.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(*this)) {
    if (!Ld->isVolatile())
      return Ld->getChain().reachesChainWithoutSideEffects(Dest, Depth-1);
  }
  return false;
}

/// hasPredecessor - Return true if N is a predecessor of this node.
/// N is either an operand of this node, or can be reached by recursively
/// traversing up the operands.
/// NOTE: This is an expensive method. Use it carefully.
bool SDNode::hasPredecessor(const SDNode *N) const {
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;
  return hasPredecessorHelper(N, Visited, Worklist);
}

bool SDNode::hasPredecessorHelper(const SDNode *N,
                                  SmallPtrSet<const SDNode *, 32> &Visited,
                                  SmallVector<const SDNode *, 16> &Worklist) const {
  if (Visited.empty()) {
    Worklist.push_back(this);
  } else {
    // Take a look in the visited set. If we've already encountered this node
    // we needn't search further.
    if (Visited.count(N))
      return true;
  }

  // Haven't visited N yet. Continue the search.
  while (!Worklist.empty()) {
    const SDNode *M = Worklist.pop_back_val();
    for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i) {
      SDNode *Op = M->getOperand(i).getNode();
      if (Visited.insert(Op))
        Worklist.push_back(Op);
      if (Op == N)
        return true;
    }
  }

  return false;
}

uint64_t SDNode::getConstantOperandVal(unsigned Num) const {
  assert(Num < NumOperands && "Invalid child # of SDNode!");
  return cast<ConstantSDNode>(OperandList[Num])->getZExtValue();
}

std::string SDNode::getOperationName(const SelectionDAG *G) const {
  switch (getOpcode()) {
  default:
    if (getOpcode() < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    if (isMachineOpcode()) {
      if (G)
        if (const TargetInstrInfo *TII = G->getTarget().getInstrInfo())
          if (getMachineOpcode() < TII->getNumOpcodes())
            return TII->get(getMachineOpcode()).getName();
      return "<<Unknown Machine Node #" + utostr(getOpcode()) + ">>";
    }
    if (G) {
      const TargetLowering &TLI = G->getTargetLoweringInfo();
      const char *Name = TLI.getTargetNodeName(getOpcode());
      if (Name) return Name;
      return "<<Unknown Target Node #" + utostr(getOpcode()) + ">>";
    }
    return "<<Unknown Node #" + utostr(getOpcode()) + ">>";

#ifndef NDEBUG
  case ISD::DELETED_NODE:
    return "<<Deleted Node!>>";
#endif
  case ISD::PREFETCH:      return "Prefetch";
  case ISD::MEMBARRIER:    return "MemBarrier";
  case ISD::ATOMIC_FENCE:    return "AtomicFence";
  case ISD::ATOMIC_CMP_SWAP:    return "AtomicCmpSwap";
  case ISD::ATOMIC_SWAP:        return "AtomicSwap";
  case ISD::ATOMIC_LOAD_ADD:    return "AtomicLoadAdd";
  case ISD::ATOMIC_LOAD_SUB:    return "AtomicLoadSub";
  case ISD::ATOMIC_LOAD_AND:    return "AtomicLoadAnd";
  case ISD::ATOMIC_LOAD_OR:     return "AtomicLoadOr";
  case ISD::ATOMIC_LOAD_XOR:    return "AtomicLoadXor";
  case ISD::ATOMIC_LOAD_NAND:   return "AtomicLoadNand";
  case ISD::ATOMIC_LOAD_MIN:    return "AtomicLoadMin";
  case ISD::ATOMIC_LOAD_MAX:    return "AtomicLoadMax";
  case ISD::ATOMIC_LOAD_UMIN:   return "AtomicLoadUMin";
  case ISD::ATOMIC_LOAD_UMAX:   return "AtomicLoadUMax";
  case ISD::ATOMIC_LOAD:        return "AtomicLoad";
  case ISD::ATOMIC_STORE:       return "AtomicStore";
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::READCYCLECOUNTER: return "ReadCycleCounter";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::MDNODE_SDNODE: return "MDNode";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::AssertSext:    return "AssertSext";
  case ISD::AssertZext:    return "AssertZext";

  case ISD::BasicBlock:    return "BasicBlock";
  case ISD::VALUETYPE:     return "ValueType";
  case ISD::Register:      return "Register";

  case ISD::Constant:      return "Constant";
  case ISD::ConstantFP:    return "ConstantFP";
  case ISD::GlobalAddress: return "GlobalAddress";
  case ISD::GlobalTLSAddress: return "GlobalTLSAddress";
  case ISD::FrameIndex:    return "FrameIndex";
  case ISD::JumpTable:     return "JumpTable";
  case ISD::GLOBAL_OFFSET_TABLE: return "GLOBAL_OFFSET_TABLE";
  case ISD::RETURNADDR: return "RETURNADDR";
  case ISD::FRAMEADDR: return "FRAMEADDR";
  case ISD::FRAME_TO_ARGS_OFFSET: return "FRAME_TO_ARGS_OFFSET";
  case ISD::EXCEPTIONADDR: return "EXCEPTIONADDR";
  case ISD::LSDAADDR: return "LSDAADDR";
  case ISD::EHSELECTION: return "EHSELECTION";
  case ISD::EH_RETURN: return "EH_RETURN";
  case ISD::EH_SJLJ_SETJMP: return "EH_SJLJ_SETJMP";
  case ISD::EH_SJLJ_LONGJMP: return "EH_SJLJ_LONGJMP";
  case ISD::EH_SJLJ_DISPATCHSETUP: return "EH_SJLJ_DISPATCHSETUP";
  case ISD::ConstantPool:  return "ConstantPool";
  case ISD::ExternalSymbol: return "ExternalSymbol";
  case ISD::BlockAddress:  return "BlockAddress";
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned OpNo = getOpcode() == ISD::INTRINSIC_WO_CHAIN ? 0 : 1;
    unsigned IID = cast<ConstantSDNode>(getOperand(OpNo))->getZExtValue();
    if (IID < Intrinsic::num_intrinsics)
      return Intrinsic::getName((Intrinsic::ID)IID);
    else if (const TargetIntrinsicInfo *TII = G->getTarget().getIntrinsicInfo())
      return TII->getName(IID);
    llvm_unreachable("Invalid intrinsic ID");
  }

  case ISD::BUILD_VECTOR:   return "BUILD_VECTOR";
  case ISD::TargetConstant: return "TargetConstant";
  case ISD::TargetConstantFP:return "TargetConstantFP";
  case ISD::TargetGlobalAddress: return "TargetGlobalAddress";
  case ISD::TargetGlobalTLSAddress: return "TargetGlobalTLSAddress";
  case ISD::TargetFrameIndex: return "TargetFrameIndex";
  case ISD::TargetJumpTable:  return "TargetJumpTable";
  case ISD::TargetConstantPool:  return "TargetConstantPool";
  case ISD::TargetExternalSymbol: return "TargetExternalSymbol";
  case ISD::TargetBlockAddress: return "TargetBlockAddress";

  case ISD::CopyToReg:     return "CopyToReg";
  case ISD::CopyFromReg:   return "CopyFromReg";
  case ISD::UNDEF:         return "undef";
  case ISD::MERGE_VALUES:  return "merge_values";
  case ISD::INLINEASM:     return "inlineasm";
  case ISD::EH_LABEL:      return "eh_label";
  case ISD::HANDLENODE:    return "handlenode";

  // Unary operators
  case ISD::FABS:   return "fabs";
  case ISD::FNEG:   return "fneg";
  case ISD::FSQRT:  return "fsqrt";
  case ISD::FSIN:   return "fsin";
  case ISD::FCOS:   return "fcos";
  case ISD::FTRUNC: return "ftrunc";
  case ISD::FFLOOR: return "ffloor";
  case ISD::FCEIL:  return "fceil";
  case ISD::FRINT:  return "frint";
  case ISD::FNEARBYINT: return "fnearbyint";
  case ISD::FEXP:   return "fexp";
  case ISD::FEXP2:  return "fexp2";
  case ISD::FLOG:   return "flog";
  case ISD::FLOG2:  return "flog2";
  case ISD::FLOG10: return "flog10";

  // Binary operators
  case ISD::ADD:    return "add";
  case ISD::SUB:    return "sub";
  case ISD::MUL:    return "mul";
  case ISD::MULHU:  return "mulhu";
  case ISD::MULHS:  return "mulhs";
  case ISD::SDIV:   return "sdiv";
  case ISD::UDIV:   return "udiv";
  case ISD::SREM:   return "srem";
  case ISD::UREM:   return "urem";
  case ISD::SMUL_LOHI:  return "smul_lohi";
  case ISD::UMUL_LOHI:  return "umul_lohi";
  case ISD::SDIVREM:    return "sdivrem";
  case ISD::UDIVREM:    return "udivrem";
  case ISD::AND:    return "and";
  case ISD::OR:     return "or";
  case ISD::XOR:    return "xor";
  case ISD::SHL:    return "shl";
  case ISD::SRA:    return "sra";
  case ISD::SRL:    return "srl";
  case ISD::ROTL:   return "rotl";
  case ISD::ROTR:   return "rotr";
  case ISD::FADD:   return "fadd";
  case ISD::FSUB:   return "fsub";
  case ISD::FMUL:   return "fmul";
  case ISD::FDIV:   return "fdiv";
  case ISD::FMA:    return "fma";
  case ISD::FREM:   return "frem";
  case ISD::FCOPYSIGN: return "fcopysign";
  case ISD::FGETSIGN:  return "fgetsign";
  case ISD::FPOW:   return "fpow";

  case ISD::FPOWI:  return "fpowi";
  case ISD::SETCC:       return "setcc";
  case ISD::SELECT:      return "select";
  case ISD::VSELECT:     return "vselect";
  case ISD::SELECT_CC:   return "select_cc";
  case ISD::INSERT_VECTOR_ELT:   return "insert_vector_elt";
  case ISD::EXTRACT_VECTOR_ELT:  return "extract_vector_elt";
  case ISD::CONCAT_VECTORS:      return "concat_vectors";
  case ISD::INSERT_SUBVECTOR:    return "insert_subvector";
  case ISD::EXTRACT_SUBVECTOR:   return "extract_subvector";
  case ISD::SCALAR_TO_VECTOR:    return "scalar_to_vector";
  case ISD::VECTOR_SHUFFLE:      return "vector_shuffle";
  case ISD::CARRY_FALSE:         return "carry_false";
  case ISD::ADDC:        return "addc";
  case ISD::ADDE:        return "adde";
  case ISD::SADDO:       return "saddo";
  case ISD::UADDO:       return "uaddo";
  case ISD::SSUBO:       return "ssubo";
  case ISD::USUBO:       return "usubo";
  case ISD::SMULO:       return "smulo";
  case ISD::UMULO:       return "umulo";
  case ISD::SUBC:        return "subc";
  case ISD::SUBE:        return "sube";
  case ISD::SHL_PARTS:   return "shl_parts";
  case ISD::SRA_PARTS:   return "sra_parts";
  case ISD::SRL_PARTS:   return "srl_parts";

  // Conversion operators.
  case ISD::SIGN_EXTEND: return "sign_extend";
  case ISD::ZERO_EXTEND: return "zero_extend";
  case ISD::ANY_EXTEND:  return "any_extend";
  case ISD::SIGN_EXTEND_INREG: return "sign_extend_inreg";
  case ISD::TRUNCATE:    return "truncate";
  case ISD::FP_ROUND:    return "fp_round";
  case ISD::FLT_ROUNDS_: return "flt_rounds";
  case ISD::FP_ROUND_INREG: return "fp_round_inreg";
  case ISD::FP_EXTEND:   return "fp_extend";

  case ISD::SINT_TO_FP:  return "sint_to_fp";
  case ISD::UINT_TO_FP:  return "uint_to_fp";
  case ISD::FP_TO_SINT:  return "fp_to_sint";
  case ISD::FP_TO_UINT:  return "fp_to_uint";
  case ISD::BITCAST:     return "bitcast";
  case ISD::FP16_TO_FP32: return "fp16_to_fp32";
  case ISD::FP32_TO_FP16: return "fp32_to_fp16";

  case ISD::CONVERT_RNDSAT: {
    switch (cast<CvtRndSatSDNode>(this)->getCvtCode()) {
    default: llvm_unreachable("Unknown cvt code!");
    case ISD::CVT_FF:  return "cvt_ff";
    case ISD::CVT_FS:  return "cvt_fs";
    case ISD::CVT_FU:  return "cvt_fu";
    case ISD::CVT_SF:  return "cvt_sf";
    case ISD::CVT_UF:  return "cvt_uf";
    case ISD::CVT_SS:  return "cvt_ss";
    case ISD::CVT_SU:  return "cvt_su";
    case ISD::CVT_US:  return "cvt_us";
    case ISD::CVT_UU:  return "cvt_uu";
    }
  }

    // Control flow instructions
  case ISD::BR:      return "br";
  case ISD::BRIND:   return "brind";
  case ISD::BR_JT:   return "br_jt";
  case ISD::BRCOND:  return "brcond";
  case ISD::BR_CC:   return "br_cc";
  case ISD::CALLSEQ_START:  return "callseq_start";
  case ISD::CALLSEQ_END:    return "callseq_end";

    // Other operators
  case ISD::LOAD:               return "load";
  case ISD::STORE:              return "store";
  case ISD::VAARG:              return "vaarg";
  case ISD::VACOPY:             return "vacopy";
  case ISD::VAEND:              return "vaend";
  case ISD::VASTART:            return "vastart";
  case ISD::DYNAMIC_STACKALLOC: return "dynamic_stackalloc";
  case ISD::EXTRACT_ELEMENT:    return "extract_element";
  case ISD::BUILD_PAIR:         return "build_pair";
  case ISD::STACKSAVE:          return "stacksave";
  case ISD::STACKRESTORE:       return "stackrestore";
  case ISD::TRAP:               return "trap";

  // Bit manipulation
  case ISD::BSWAP:   return "bswap";
  case ISD::CTPOP:   return "ctpop";
  case ISD::CTTZ:    return "cttz";
  case ISD::CTLZ:    return "ctlz";

  // Trampolines
  case ISD::INIT_TRAMPOLINE: return "init_trampoline";
  case ISD::ADJUST_TRAMPOLINE: return "adjust_trampoline";

  case ISD::CONDCODE:
    switch (cast<CondCodeSDNode>(this)->get()) {
    default: llvm_unreachable("Unknown setcc condition!");
    case ISD::SETOEQ:  return "setoeq";
    case ISD::SETOGT:  return "setogt";
    case ISD::SETOGE:  return "setoge";
    case ISD::SETOLT:  return "setolt";
    case ISD::SETOLE:  return "setole";
    case ISD::SETONE:  return "setone";

    case ISD::SETO:    return "seto";
    case ISD::SETUO:   return "setuo";
    case ISD::SETUEQ:  return "setue";
    case ISD::SETUGT:  return "setugt";
    case ISD::SETUGE:  return "setuge";
    case ISD::SETULT:  return "setult";
    case ISD::SETULE:  return "setule";
    case ISD::SETUNE:  return "setune";

    case ISD::SETEQ:   return "seteq";
    case ISD::SETGT:   return "setgt";
    case ISD::SETGE:   return "setge";
    case ISD::SETLT:   return "setlt";
    case ISD::SETLE:   return "setle";
    case ISD::SETNE:   return "setne";
    }
  }
}

const char *SDNode::getIndexedModeName(ISD::MemIndexedMode AM) {
  switch (AM) {
  default:
    return "";
  case ISD::PRE_INC:
    return "<pre-inc>";
  case ISD::PRE_DEC:
    return "<pre-dec>";
  case ISD::POST_INC:
    return "<post-inc>";
  case ISD::POST_DEC:
    return "<post-dec>";
  }
}

std::string ISD::ArgFlagsTy::getArgFlagsString() {
  std::string S = "< ";

  if (isZExt())
    S += "zext ";
  if (isSExt())
    S += "sext ";
  if (isInReg())
    S += "inreg ";
  if (isSRet())
    S += "sret ";
  if (isByVal())
    S += "byval ";
  if (isNest())
    S += "nest ";
  if (getByValAlign())
    S += "byval-align:" + utostr(getByValAlign()) + " ";
  if (getOrigAlign())
    S += "orig-align:" + utostr(getOrigAlign()) + " ";
  if (getByValSize())
    S += "byval-size:" + utostr(getByValSize()) + " ";
  return S + ">";
}

void SDNode::dump() const { dump(0); }
void SDNode::dump(const SelectionDAG *G) const {
  print(dbgs(), G);
  dbgs() << '\n';
}

void SDNode::print_types(raw_ostream &OS, const SelectionDAG *G) const {
  OS << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) OS << ",";
    if (getValueType(i) == MVT::Other)
      OS << "ch";
    else
      OS << getValueType(i).getEVTString();
  }
  OS << " = " << getOperationName(G);
}

void SDNode::print_details(raw_ostream &OS, const SelectionDAG *G) const {
  if (const MachineSDNode *MN = dyn_cast<MachineSDNode>(this)) {
    if (!MN->memoperands_empty()) {
      OS << "<";
      OS << "Mem:";
      for (MachineSDNode::mmo_iterator i = MN->memoperands_begin(),
           e = MN->memoperands_end(); i != e; ++i) {
        OS << **i;
        if (llvm::next(i) != e)
          OS << " ";
      }
      OS << ">";
    }
  } else if (const ShuffleVectorSDNode *SVN =
               dyn_cast<ShuffleVectorSDNode>(this)) {
    OS << "<";
    for (unsigned i = 0, e = ValueList[0].getVectorNumElements(); i != e; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (i) OS << ",";
      if (Idx < 0)
        OS << "u";
      else
        OS << Idx;
    }
    OS << ">";
  } else if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(this)) {
    OS << '<' << CSDN->getAPIntValue() << '>';
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(this)) {
    if (&CSDN->getValueAPF().getSemantics()==&APFloat::IEEEsingle)
      OS << '<' << CSDN->getValueAPF().convertToFloat() << '>';
    else if (&CSDN->getValueAPF().getSemantics()==&APFloat::IEEEdouble)
      OS << '<' << CSDN->getValueAPF().convertToDouble() << '>';
    else {
      OS << "<APFloat(";
      CSDN->getValueAPF().bitcastToAPInt().dump();
      OS << ")>";
    }
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(this)) {
    int64_t offset = GADN->getOffset();
    OS << '<';
    WriteAsOperand(OS, GADN->getGlobal());
    OS << '>';
    if (offset > 0)
      OS << " + " << offset;
    else
      OS << " " << offset;
    if (unsigned int TF = GADN->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    OS << "<" << FIDN->getIndex() << ">";
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(this)) {
    OS << "<" << JTDN->getIndex() << ">";
    if (unsigned int TF = JTDN->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    int offset = CP->getOffset();
    if (CP->isMachineConstantPoolEntry())
      OS << "<" << *CP->getMachineCPVal() << ">";
    else
      OS << "<" << *CP->getConstVal() << ">";
    if (offset > 0)
      OS << " + " << offset;
    else
      OS << " " << offset;
    if (unsigned int TF = CP->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    OS << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      OS << LBB->getName() << " ";
    OS << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    OS << ' ' << PrintReg(R->getReg(), G ? G->getTarget().getRegisterInfo() :0);
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    OS << "'" << ES->getSymbol() << "'";
    if (unsigned int TF = ES->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      OS << "<" << M->getValue() << ">";
    else
      OS << "<null>";
  } else if (const MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(this)) {
    if (MD->getMD())
      OS << "<" << MD->getMD() << ">";
    else
      OS << "<null>";
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(this)) {
    OS << ":" << N->getVT().getEVTString();
  }
  else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(this)) {
    OS << "<" << *LD->getMemOperand();

    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD: OS << ", anyext"; break;
    case ISD::SEXTLOAD: OS << ", sext"; break;
    case ISD::ZEXTLOAD: OS << ", zext"; break;
    }
    if (doExt)
      OS << " from " << LD->getMemoryVT().getEVTString();

    const char *AM = getIndexedModeName(LD->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    OS << ">";
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(this)) {
    OS << "<" << *ST->getMemOperand();

    if (ST->isTruncatingStore())
      OS << ", trunc to " << ST->getMemoryVT().getEVTString();

    const char *AM = getIndexedModeName(ST->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    OS << ">";
  } else if (const MemSDNode* M = dyn_cast<MemSDNode>(this)) {
    OS << "<" << *M->getMemOperand() << ">";
  } else if (const BlockAddressSDNode *BA =
               dyn_cast<BlockAddressSDNode>(this)) {
    OS << "<";
    WriteAsOperand(OS, BA->getBlockAddress()->getFunction(), false);
    OS << ", ";
    WriteAsOperand(OS, BA->getBlockAddress()->getBasicBlock(), false);
    OS << ">";
    if (unsigned int TF = BA->getTargetFlags())
      OS << " [TF=" << TF << ']';
  }

  if (G)
    if (unsigned Order = G->GetOrdering(this))
      OS << " [ORD=" << Order << ']';

  if (getNodeId() != -1)
    OS << " [ID=" << getNodeId() << ']';

  DebugLoc dl = getDebugLoc();
  if (G && !dl.isUnknown()) {
    DIScope
      Scope(dl.getScope(G->getMachineFunction().getFunction()->getContext()));
    OS << " dbg:";
    // Omit the directory, since it's usually long and uninteresting.
    if (Scope.Verify())
      OS << Scope.getFilename();
    else
      OS << "<unknown>";
    OS << ':' << dl.getLine();
    if (dl.getCol() != 0)
      OS << ':' << dl.getCol();
  }
}

void SDNode::print(raw_ostream &OS, const SelectionDAG *G) const {
  print_types(OS, G);
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i) OS << ", "; else OS << " ";
    OS << (void*)getOperand(i).getNode();
    if (unsigned RN = getOperand(i).getResNo())
      OS << ":" << RN;
  }
  print_details(OS, G);
}

static void printrWithDepthHelper(raw_ostream &OS, const SDNode *N,
                                  const SelectionDAG *G, unsigned depth,
                                  unsigned indent) {
  if (depth == 0)
    return;

  OS.indent(indent);

  N->print(OS, G);

  if (depth < 1)
    return;

  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    // Don't follow chain operands.
    if (N->getOperand(i).getValueType() == MVT::Other)
      continue;
    OS << '\n';
    printrWithDepthHelper(OS, N->getOperand(i).getNode(), G, depth-1, indent+2);
  }
}

void SDNode::printrWithDepth(raw_ostream &OS, const SelectionDAG *G,
                            unsigned depth) const {
  printrWithDepthHelper(OS, this, G, depth, 0);
}

void SDNode::printrFull(raw_ostream &OS, const SelectionDAG *G) const {
  // Don't print impossibly deep things.
  printrWithDepth(OS, G, 10);
}

void SDNode::dumprWithDepth(const SelectionDAG *G, unsigned depth) const {
  printrWithDepth(dbgs(), G, depth);
}

void SDNode::dumprFull(const SelectionDAG *G) const {
  // Don't print impossibly deep things.
  dumprWithDepth(G, 10);
}

static void DumpNodes(const SDNode *N, unsigned indent, const SelectionDAG *G) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i).getNode()->hasOneUse())
      DumpNodes(N->getOperand(i).getNode(), indent+2, G);
    else
      dbgs() << "\n" << std::string(indent+2, ' ')
           << (void*)N->getOperand(i).getNode() << ": <multiple use>";


  dbgs() << "\n";
  dbgs().indent(indent);
  N->dump(G);
}

SDValue SelectionDAG::UnrollVectorOp(SDNode *N, unsigned ResNE) {
  assert(N->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");

  EVT VT = N->getValueType(0);
  unsigned NE = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  DebugLoc dl = N->getDebugLoc();

  SmallVector<SDValue, 8> Scalars;
  SmallVector<SDValue, 4> Operands(N->getNumOperands());

  // If ResNE is 0, fully unroll the vector op.
  if (ResNE == 0)
    ResNE = NE;
  else if (NE > ResNE)
    NE = ResNE;

  unsigned i;
  for (i= 0; i != NE; ++i) {
    for (unsigned j = 0, e = N->getNumOperands(); j != e; ++j) {
      SDValue Operand = N->getOperand(j);
      EVT OperandVT = Operand.getValueType();
      if (OperandVT.isVector()) {
        // A vector operand; extract a single element.
        EVT OperandEltVT = OperandVT.getVectorElementType();
        Operands[j] = getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                              OperandEltVT,
                              Operand,
                              getConstant(i, TLI.getPointerTy()));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }

    switch (N->getOpcode()) {
    default:
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT,
                                &Operands[0], Operands.size()));
      break;
    case ISD::VSELECT:
      Scalars.push_back(getNode(ISD::SELECT, dl, EltVT,
                                &Operands[0], Operands.size()));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
    case ISD::ROTL:
    case ISD::ROTR:
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT, Operands[0],
                                getShiftAmountOperand(Operands[0].getValueType(),
                                                      Operands[1])));
      break;
    case ISD::SIGN_EXTEND_INREG:
    case ISD::FP_ROUND_INREG: {
      EVT ExtVT = cast<VTSDNode>(Operands[1])->getVT().getVectorElementType();
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT,
                                Operands[0],
                                getValueType(ExtVT)));
    }
    }
  }

  for (; i < ResNE; ++i)
    Scalars.push_back(getUNDEF(EltVT));

  return getNode(ISD::BUILD_VECTOR, dl,
                 EVT::getVectorVT(*getContext(), EltVT, ResNE),
                 &Scalars[0], Scalars.size());
}


/// isConsecutiveLoad - Return true if LD is loading 'Bytes' bytes from a
/// location that is 'Dist' units away from the location that the 'Base' load
/// is loading from.
bool SelectionDAG::isConsecutiveLoad(LoadSDNode *LD, LoadSDNode *Base,
                                     unsigned Bytes, int Dist) const {
  if (LD->getChain() != Base->getChain())
    return false;
  EVT VT = LD->getValueType(0);
  if (VT.getSizeInBits() / 8 != Bytes)
    return false;

  SDValue Loc = LD->getOperand(1);
  SDValue BaseLoc = Base->getOperand(1);
  if (Loc.getOpcode() == ISD::FrameIndex) {
    if (BaseLoc.getOpcode() != ISD::FrameIndex)
      return false;
    const MachineFrameInfo *MFI = getMachineFunction().getFrameInfo();
    int FI  = cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = cast<FrameIndexSDNode>(BaseLoc)->getIndex();
    int FS  = MFI->getObjectSize(FI);
    int BFS = MFI->getObjectSize(BFI);
    if (FS != BFS || FS != (int)Bytes) return false;
    return MFI->getObjectOffset(FI) == (MFI->getObjectOffset(BFI) + Dist*Bytes);
  }

  // Handle X+C
  if (isBaseWithConstantOffset(Loc) && Loc.getOperand(0) == BaseLoc &&
      cast<ConstantSDNode>(Loc.getOperand(1))->getSExtValue() == Dist*Bytes)
    return true;

  const GlobalValue *GV1 = NULL;
  const GlobalValue *GV2 = NULL;
  int64_t Offset1 = 0;
  int64_t Offset2 = 0;
  bool isGA1 = TLI.isGAPlusOffset(Loc.getNode(), GV1, Offset1);
  bool isGA2 = TLI.isGAPlusOffset(BaseLoc.getNode(), GV2, Offset2);
  if (isGA1 && isGA2 && GV1 == GV2)
    return Offset1 == (Offset2 + Dist*Bytes);
  return false;
}


/// InferPtrAlignment - Infer alignment of a load / store address. Return 0 if
/// it cannot be inferred.
unsigned SelectionDAG::InferPtrAlignment(SDValue Ptr) const {
  // If this is a GlobalAddress + cst, return the alignment.
  const GlobalValue *GV;
  int64_t GVOffset = 0;
  if (TLI.isGAPlusOffset(Ptr.getNode(), GV, GVOffset)) {
    // If GV has specified alignment, then use it. Otherwise, use the preferred
    // alignment.
    unsigned Align = GV->getAlignment();
    if (!Align) {
      if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
        if (GVar->hasInitializer()) {
          const TargetData *TD = TLI.getTargetData();
          Align = TD->getPreferredAlignment(GVar);
        }
      }
      if (!Align)
        Align = TLI.getTargetData()->getABITypeAlignment(GV->getType());
    }
    return MinAlign(Align, GVOffset);
  }

  // If this is a direct reference to a stack slot, use information about the
  // stack slot's alignment.
  int FrameIdx = 1 << 31;
  int64_t FrameOffset = 0;
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr)) {
    FrameIdx = FI->getIndex();
  } else if (isBaseWithConstantOffset(Ptr) &&
             isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
    // Handle FI+Cst
    FrameIdx = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
    FrameOffset = Ptr.getConstantOperandVal(1);
  }

  if (FrameIdx != (1 << 31)) {
    const MachineFrameInfo &MFI = *getMachineFunction().getFrameInfo();
    unsigned FIInfoAlign = MinAlign(MFI.getObjectAlignment(FrameIdx),
                                    FrameOffset);
    return FIInfoAlign;
  }

  return 0;
}

void SelectionDAG::dump() const {
  dbgs() << "SelectionDAG has " << AllNodes.size() << " nodes:";

  for (allnodes_const_iterator I = allnodes_begin(), E = allnodes_end();
       I != E; ++I) {
    const SDNode *N = I;
    if (!N->hasOneUse() && N != getRoot().getNode())
      DumpNodes(N, 2, this);
  }

  if (getRoot().getNode()) DumpNodes(getRoot().getNode(), 2, this);

  dbgs() << "\n\n";
}

void SDNode::printr(raw_ostream &OS, const SelectionDAG *G) const {
  print_types(OS, G);
  print_details(OS, G);
}

typedef SmallPtrSet<const SDNode *, 128> VisitedSDNodeSet;
static void DumpNodesr(raw_ostream &OS, const SDNode *N, unsigned indent,
                       const SelectionDAG *G, VisitedSDNodeSet &once) {
  if (!once.insert(N))          // If we've been here before, return now.
    return;

  // Dump the current SDNode, but don't end the line yet.
  OS << std::string(indent, ' ');
  N->printr(OS, G);

  // Having printed this SDNode, walk the children:
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDNode *child = N->getOperand(i).getNode();

    if (i) OS << ",";
    OS << " ";

    if (child->getNumOperands() == 0) {
      // This child has no grandchildren; print it inline right here.
      child->printr(OS, G);
      once.insert(child);
    } else {         // Just the address. FIXME: also print the child's opcode.
      OS << (void*)child;
      if (unsigned RN = N->getOperand(i).getResNo())
        OS << ":" << RN;
    }
  }

  OS << "\n";

  // Dump children that have grandchildren on their own line(s).
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDNode *child = N->getOperand(i).getNode();
    DumpNodesr(OS, child, indent+2, G, once);
  }
}

void SDNode::dumpr() const {
  VisitedSDNodeSet once;
  DumpNodesr(dbgs(), this, 0, 0, once);
}

void SDNode::dumpr(const SelectionDAG *G) const {
  VisitedSDNodeSet once;
  DumpNodesr(dbgs(), this, 0, G, once);
}


// getAddressSpace - Return the address space this GlobalAddress belongs to.
unsigned GlobalAddressSDNode::getAddressSpace() const {
  return getGlobal()->getType()->getAddressSpace();
}


Type *ConstantPoolSDNode::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}

bool BuildVectorSDNode::isConstantSplat(APInt &SplatValue,
                                        APInt &SplatUndef,
                                        unsigned &SplatBitSize,
                                        bool &HasAnyUndefs,
                                        unsigned MinSplatBits,
                                        bool isBigEndian) {
  EVT VT = getValueType(0);
  assert(VT.isVector() && "Expected a vector type");
  unsigned sz = VT.getSizeInBits();
  if (MinSplatBits > sz)
    return false;

  SplatValue = APInt(sz, 0);
  SplatUndef = APInt(sz, 0);

  // Get the bits.  Bits with undefined values (when the corresponding element
  // of the vector is an ISD::UNDEF value) are set in SplatUndef and cleared
  // in SplatValue.  If any of the values are not constant, give up and return
  // false.
  unsigned int nOps = getNumOperands();
  assert(nOps > 0 && "isConstantSplat has 0-size build vector");
  unsigned EltBitSize = VT.getVectorElementType().getSizeInBits();

  for (unsigned j = 0; j < nOps; ++j) {
    unsigned i = isBigEndian ? nOps-1-j : j;
    SDValue OpVal = getOperand(i);
    unsigned BitPos = j * EltBitSize;

    if (OpVal.getOpcode() == ISD::UNDEF)
      SplatUndef |= APInt::getBitsSet(sz, BitPos, BitPos + EltBitSize);
    else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal))
      SplatValue |= CN->getAPIntValue().zextOrTrunc(EltBitSize).
                    zextOrTrunc(sz) << BitPos;
    else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal))
      SplatValue |= CN->getValueAPF().bitcastToAPInt().zextOrTrunc(sz) <<BitPos;
     else
      return false;
  }

  // The build_vector is all constants or undefs.  Find the smallest element
  // size that splats the vector.

  HasAnyUndefs = (SplatUndef != 0);
  while (sz > 8) {

    unsigned HalfSize = sz / 2;
    APInt HighValue = SplatValue.lshr(HalfSize).trunc(HalfSize);
    APInt LowValue = SplatValue.trunc(HalfSize);
    APInt HighUndef = SplatUndef.lshr(HalfSize).trunc(HalfSize);
    APInt LowUndef = SplatUndef.trunc(HalfSize);

    // If the two halves do not match (ignoring undef bits), stop here.
    if ((HighValue & ~LowUndef) != (LowValue & ~HighUndef) ||
        MinSplatBits > HalfSize)
      break;

    SplatValue = HighValue | LowValue;
    SplatUndef = HighUndef & LowUndef;

    sz = HalfSize;
  }

  SplatBitSize = sz;
  return true;
}

bool ShuffleVectorSDNode::isSplatMask(const int *Mask, EVT VT) {
  // Find the first non-undef value in the shuffle mask.
  unsigned i, e;
  for (i = 0, e = VT.getVectorNumElements(); i != e && Mask[i] < 0; ++i)
    /* search */;

  assert(i != e && "VECTOR_SHUFFLE node with all undef indices!");

  // Make sure all remaining elements are either undef or the same as the first
  // non-undef value.
  for (int Idx = Mask[i]; i != e; ++i)
    if (Mask[i] >= 0 && Mask[i] != Idx)
      return false;
  return true;
}

#ifdef XDEBUG
static void checkForCyclesHelper(const SDNode *N,
                                 SmallPtrSet<const SDNode*, 32> &Visited,
                                 SmallPtrSet<const SDNode*, 32> &Checked) {
  // If this node has already been checked, don't check it again.
  if (Checked.count(N))
    return;

  // If a node has already been visited on this depth-first walk, reject it as
  // a cycle.
  if (!Visited.insert(N)) {
    dbgs() << "Offending node:\n";
    N->dumprFull();
    errs() << "Detected cycle in SelectionDAG\n";
    abort();
  }

  for(unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    checkForCyclesHelper(N->getOperand(i).getNode(), Visited, Checked);

  Checked.insert(N);
  Visited.erase(N);
}
#endif

void llvm::checkForCycles(const llvm::SDNode *N) {
#ifdef XDEBUG
  assert(N && "Checking nonexistant SDNode");
  SmallPtrSet<const SDNode*, 32> visited;
  SmallPtrSet<const SDNode*, 32> checked;
  checkForCyclesHelper(N, visited, checked);
#endif
}

void llvm::checkForCycles(const llvm::SelectionDAG *DAG) {
  checkForCycles(DAG->getRoot().getNode());
}
