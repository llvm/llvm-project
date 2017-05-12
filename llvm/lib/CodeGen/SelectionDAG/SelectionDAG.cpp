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
#include "SDNodeDbgValue.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace llvm;

/// makeVTList - Return an instance of the SDVTList struct initialized with the
/// specified members.
static SDVTList makeVTList(const EVT *VTs, unsigned NumVTs) {
  SDVTList Res = {VTs, NumVTs};
  return Res;
}

// Default null implementations of the callbacks.
void SelectionDAG::DAGUpdateListener::NodeDeleted(SDNode*, SDNode*) {}
void SelectionDAG::DAGUpdateListener::NodeUpdated(SDNode*) {}

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

  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  (void) Val2.convert(SelectionDAG::EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven,
                      &losesInfo);
  return !losesInfo;
}

//===----------------------------------------------------------------------===//
//                              ISD Namespace
//===----------------------------------------------------------------------===//

bool ISD::isConstantSplatVector(const SDNode *N, APInt &SplatVal) {
  auto *BV = dyn_cast<BuildVectorSDNode>(N);
  if (!BV)
    return false;

  APInt SplatUndef;
  unsigned SplatBitSize;
  bool HasUndefs;
  EVT EltVT = N->getValueType(0).getVectorElementType();
  return BV->isConstantSplat(SplatVal, SplatUndef, SplatBitSize, HasUndefs) &&
         EltVT.getSizeInBits() >= SplatBitSize;
}

// FIXME: AllOnes and AllZeros duplicate a lot of code. Could these be
// specializations of the more general isConstantSplatVector()?

bool ISD::isBuildVectorAllOnes(const SDNode *N) {
  // Look through a bit convert.
  while (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  unsigned i = 0, e = N->getNumOperands();

  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).isUndef())
    ++i;

  // Do not accept an all-undef vector.
  if (i == e) return false;

  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements. We have to be a bit careful here, as the type of the constant
  // may not be the same as the type of the vector elements due to type
  // legalization (the elements are promoted to a legal type for the target and
  // a vector of a type may be legal when the base element type is not).
  // We only want to check enough bits to cover the vector elements, because
  // we care if the resultant vector is all ones, not whether the individual
  // constants are.
  SDValue NotZero = N->getOperand(i);
  unsigned EltSize = N->getValueType(0).getScalarSizeInBits();
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(NotZero)) {
    if (CN->getAPIntValue().countTrailingOnes() < EltSize)
      return false;
  } else if (ConstantFPSDNode *CFPN = dyn_cast<ConstantFPSDNode>(NotZero)) {
    if (CFPN->getValueAPF().bitcastToAPInt().countTrailingOnes() < EltSize)
      return false;
  } else
    return false;

  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs. Even with the above element type twiddling, this should be OK, as
  // the same type legalization should have applied to all the elements.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != NotZero && !N->getOperand(i).isUndef())
      return false;
  return true;
}

bool ISD::isBuildVectorAllZeros(const SDNode *N) {
  // Look through a bit convert.
  while (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;

  bool IsAllUndef = true;
  for (const SDValue &Op : N->op_values()) {
    if (Op.isUndef())
      continue;
    IsAllUndef = false;
    // Do not accept build_vectors that aren't all constants or which have non-0
    // elements. We have to be a bit careful here, as the type of the constant
    // may not be the same as the type of the vector elements due to type
    // legalization (the elements are promoted to a legal type for the target
    // and a vector of a type may be legal when the base element type is not).
    // We only want to check enough bits to cover the vector elements, because
    // we care if the resultant vector is all zeros, not whether the individual
    // constants are.
    unsigned EltSize = N->getValueType(0).getScalarSizeInBits();
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op)) {
      if (CN->getAPIntValue().countTrailingZeros() < EltSize)
        return false;
    } else if (ConstantFPSDNode *CFPN = dyn_cast<ConstantFPSDNode>(Op)) {
      if (CFPN->getValueAPF().bitcastToAPInt().countTrailingZeros() < EltSize)
        return false;
    } else
      return false;
  }

  // Do not accept an all-undef vector.
  if (IsAllUndef)
    return false;
  return true;
}

bool ISD::isBuildVectorOfConstantSDNodes(const SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (const SDValue &Op : N->op_values()) {
    if (Op.isUndef())
      continue;
    if (!isa<ConstantSDNode>(Op))
      return false;
  }
  return true;
}

bool ISD::isBuildVectorOfConstantFPSDNodes(const SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (const SDValue &Op : N->op_values()) {
    if (Op.isUndef())
      continue;
    if (!isa<ConstantFPSDNode>(Op))
      return false;
  }
  return true;
}

bool ISD::allOperandsUndef(const SDNode *N) {
  // Return false if the node has no operands.
  // This is "logically inconsistent" with the definition of "all" but
  // is probably the desired behavior.
  if (N->getNumOperands() == 0)
    return false;

  for (const SDValue &Op : N->op_values())
    if (!Op.isUndef())
      return false;

  return true;
}

ISD::NodeType ISD::getExtForLoadExtType(bool IsFP, ISD::LoadExtType ExtType) {
  switch (ExtType) {
  case ISD::EXTLOAD:
    return IsFP ? ISD::FP_EXTEND : ISD::ANY_EXTEND;
  case ISD::SEXTLOAD:
    return ISD::SIGN_EXTEND;
  case ISD::ZEXTLOAD:
    return ISD::ZERO_EXTEND;
  default:
    break;
  }

  llvm_unreachable("Invalid LoadExtType");
}

ISD::CondCode ISD::getSetCCSwappedOperands(ISD::CondCode Operation) {
  // To perform this operation, we just need to swap the L and G bits of the
  // operation.
  unsigned OldL = (Operation >> 2) & 1;
  unsigned OldG = (Operation >> 1) & 1;
  return ISD::CondCode((Operation & ~6) |  // Keep the N, U, E bits
                       (OldL << 1) |       // New G bit
                       (OldG << 2));       // New L bit.
}

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


/// For an integer comparison, return 1 if the comparison is a signed operation
/// and 2 if the result is an unsigned comparison. Return zero if the operation
/// does not depend on the sign of the input (setne and seteq).
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

ISD::CondCode ISD::getSetCCOrOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                       bool IsInteger) {
  if (IsInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed integer setcc with an unsigned integer setcc.
    return ISD::SETCC_INVALID;

  unsigned Op = Op1 | Op2;  // Combine all of the condition bits.

  // If the N and U bits get set, then the resultant comparison DOES suddenly
  // care about orderedness, and it is true when ordered.
  if (Op > ISD::SETTRUE2)
    Op &= ~16;     // Clear the U bit if the N bit is set.

  // Canonicalize illegal integer setcc's.
  if (IsInteger && Op == ISD::SETUNE)  // e.g. SETUGT | SETULT
    Op = ISD::SETNE;

  return ISD::CondCode(Op);
}

ISD::CondCode ISD::getSetCCAndOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                        bool IsInteger) {
  if (IsInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed setcc with an unsigned setcc.
    return ISD::SETCC_INVALID;

  // Combine all of the condition bits.
  ISD::CondCode Result = ISD::CondCode(Op1 & Op2);

  // Canonicalize illegal integer setcc's.
  if (IsInteger) {
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
                              ArrayRef<SDValue> Ops) {
  for (auto& Op : Ops) {
    ID.AddPointer(Op.getNode());
    ID.AddInteger(Op.getResNo());
  }
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              ArrayRef<SDUse> Ops) {
  for (auto& Op : Ops) {
    ID.AddPointer(Op.getNode());
    ID.AddInteger(Op.getResNo());
  }
}

static void AddNodeIDNode(FoldingSetNodeID &ID, unsigned short OpC,
                          SDVTList VTList, ArrayRef<SDValue> OpList) {
  AddNodeIDOpcode(ID, OpC);
  AddNodeIDValueTypes(ID, VTList);
  AddNodeIDOperands(ID, OpList);
}

/// If this is an SDNode with special info, add this info to the NodeID data.
static void AddNodeIDCustom(FoldingSetNodeID &ID, const SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::TargetExternalSymbol:
  case ISD::ExternalSymbol:
  case ISD::MCSymbol:
    llvm_unreachable("Should only be used on nodes with operands");
  default: break;  // Normal nodes don't need extra info.
  case ISD::TargetConstant:
  case ISD::Constant: {
    const ConstantSDNode *C = cast<ConstantSDNode>(N);
    ID.AddPointer(C->getConstantIntValue());
    ID.AddBoolean(C->isOpaque());
    break;
  }
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
  case ISD::RegisterMask:
    ID.AddPointer(cast<RegisterMaskSDNode>(N)->getRegMask());
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
  case ISD::TargetIndex: {
    const TargetIndexSDNode *TI = cast<TargetIndexSDNode>(N);
    ID.AddInteger(TI->getIndex());
    ID.AddInteger(TI->getOffset());
    ID.AddInteger(TI->getTargetFlags());
    break;
  }
  case ISD::LOAD: {
    const LoadSDNode *LD = cast<LoadSDNode>(N);
    ID.AddInteger(LD->getMemoryVT().getRawBits());
    ID.AddInteger(LD->getRawSubclassData());
    ID.AddInteger(LD->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::STORE: {
    const StoreSDNode *ST = cast<StoreSDNode>(N);
    ID.AddInteger(ST->getMemoryVT().getRawBits());
    ID.AddInteger(ST->getRawSubclassData());
    ID.AddInteger(ST->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
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
    ID.AddInteger(AT->getPointerInfo().getAddrSpace());
    break;
  }
  case ISD::PREFETCH: {
    const MemSDNode *PF = cast<MemSDNode>(N);
    ID.AddInteger(PF->getPointerInfo().getAddrSpace());
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
    const BlockAddressSDNode *BA = cast<BlockAddressSDNode>(N);
    ID.AddPointer(BA->getBlockAddress());
    ID.AddInteger(BA->getOffset());
    ID.AddInteger(BA->getTargetFlags());
    break;
  }
  } // end switch (N->getOpcode())

  // Target specific memory nodes could also have address spaces to check.
  if (N->isTargetMemoryOpcode())
    ID.AddInteger(cast<MemSDNode>(N)->getPointerInfo().getAddrSpace());
}

/// AddNodeIDNode - Generic routine for adding a nodes info to the NodeID
/// data.
static void AddNodeIDNode(FoldingSetNodeID &ID, const SDNode *N) {
  AddNodeIDOpcode(ID, N->getOpcode());
  // Add the return value info.
  AddNodeIDValueTypes(ID, N->getVTList());
  // Add the operand info.
  AddNodeIDOperands(ID, N->ops());

  // Handle SDNode leafs with special info.
  AddNodeIDCustom(ID, N);
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
  for (SDNode &Node : allnodes())
    if (Node.use_empty())
      DeadNodes.push_back(&Node);

  RemoveDeadNodes(DeadNodes);

  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

/// RemoveDeadNodes - This method deletes the unreachable nodes in the
/// given list, and any nodes that become unreachable as a result.
void SelectionDAG::RemoveDeadNodes(SmallVectorImpl<SDNode *> &DeadNodes) {

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.pop_back_val();

    for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
      DUL->NodeDeleted(N, nullptr);

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

void SelectionDAG::RemoveDeadNode(SDNode *N){
  SmallVector<SDNode*, 16> DeadNodes(1, N);

  // Create a dummy node that adds a reference to the root node, preventing
  // it from being deleted.  (This matters if the root is an operand of the
  // dead node.)
  HandleSDNode Dummy(getRoot());

  RemoveDeadNodes(DeadNodes);
}

void SelectionDAG::DeleteNode(SDNode *N) {
  // First take this out of the appropriate CSE map.
  RemoveNodeFromCSEMaps(N);

  // Finally, remove uses due to operands of this node, remove from the
  // AllNodes list, and delete the node.
  DeleteNodeNotInCSEMaps(N);
}

void SelectionDAG::DeleteNodeNotInCSEMaps(SDNode *N) {
  assert(N->getIterator() != AllNodes.begin() &&
         "Cannot delete the entry node!");
  assert(N->use_empty() && "Cannot delete a node that is not dead!");

  // Drop all of the operands and decrement used node's use counts.
  N->DropOperands();

  DeallocateNode(N);
}

void SDDbgInfo::erase(const SDNode *Node) {
  DbgValMapType::iterator I = DbgValMap.find(Node);
  if (I == DbgValMap.end())
    return;
  for (auto &Val: I->second)
    Val->setIsInvalidated();
  DbgValMap.erase(I);
}

void SelectionDAG::DeallocateNode(SDNode *N) {
  // If we have operands, deallocate them.
  removeOperands(N);

  NodeAllocator.Deallocate(AllNodes.remove(N));

  // Set the opcode to DELETED_NODE to help catch bugs when node
  // memory is reallocated.
  // FIXME: There are places in SDag that have grown a dependency on the opcode
  // value in the released node.
  __asan_unpoison_memory_region(&N->NodeType, sizeof(N->NodeType));
  N->NodeType = ISD::DELETED_NODE;

  // If any of the SDDbgValue nodes refer to this SDNode, invalidate
  // them and forget about that node.
  DbgInfo->erase(N);
}

#ifndef NDEBUG
/// VerifySDNode - Sanity check the given SDNode.  Aborts if it is invalid.
static void VerifySDNode(SDNode *N) {
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
#endif // NDEBUG

/// \brief Insert a newly allocated node into the DAG.
///
/// Handles insertion into the all nodes list and CSE map, as well as
/// verification and other common operations when a new node is allocated.
void SelectionDAG::InsertNode(SDNode *N) {
  AllNodes.push_back(N);
#ifndef NDEBUG
  N->PersistentId = NextPersistentId++;
  VerifySDNode(N);
#endif
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
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != nullptr;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = nullptr;
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
  case ISD::MCSymbol: {
    auto *MCSN = cast<MCSymbolSDNode>(N);
    Erased = MCSymbols.erase(MCSN->getMCSymbol());
    break;
  }
  case ISD::VALUETYPE: {
    EVT VT = cast<VTSDNode>(N)->getVT();
    if (VT.isExtended()) {
      Erased = ExtendedValueTypeNodes.erase(VT);
    } else {
      Erased = ValueTypeNodes[VT.getSimpleVT().SimpleTy] != nullptr;
      ValueTypeNodes[VT.getSimpleVT().SimpleTy] = nullptr;
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
SelectionDAG::AddModifiedNodeToCSEMaps(SDNode *N) {
  // For node types that aren't CSE'd, just act as if no identical node
  // already exists.
  if (!doNotCSE(N)) {
    SDNode *Existing = CSEMap.GetOrInsertNode(N);
    if (Existing != N) {
      // If there was already an existing matching node, use ReplaceAllUsesWith
      // to replace the dead one with the existing one.  This can cause
      // recursive merging of other unrelated nodes down the line.
      ReplaceAllUsesWith(N, Existing);

      // N is now dead. Inform the listeners and delete it.
      for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
        DUL->NodeDeleted(N, Existing);
      DeleteNodeNotInCSEMaps(N);
      return;
    }
  }

  // If the node doesn't already exist, we updated it.  Inform listeners.
  for (DAGUpdateListener *DUL = UpdateListeners; DUL; DUL = DUL->Next)
    DUL->NodeUpdated(N);
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, SDValue Op,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return nullptr;

  SDValue Ops[] = { Op };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, SDLoc(N), InsertPos);
  if (Node)
    Node->intersectFlagsWith(N->getFlags());
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
    return nullptr;

  SDValue Ops[] = { Op1, Op2 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, SDLoc(N), InsertPos);
  if (Node)
    Node->intersectFlagsWith(N->getFlags());
  return Node;
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized,
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, ArrayRef<SDValue> Ops,
                                           void *&InsertPos) {
  if (doNotCSE(N))
    return nullptr;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops);
  AddNodeIDCustom(ID, N);
  SDNode *Node = FindNodeOrInsertPos(ID, SDLoc(N), InsertPos);
  if (Node)
    Node->intersectFlagsWith(N->getFlags());
  return Node;
}

unsigned SelectionDAG::getEVTAlignment(EVT VT) const {
  Type *Ty = VT == MVT::iPTR ?
                   PointerType::get(Type::getInt8Ty(*getContext()), 0) :
                   VT.getTypeForEVT(*getContext());

  return getDataLayout().getABITypeAlignment(Ty);
}

// EntryNode could meaningfully have debug info if we can find it...
SelectionDAG::SelectionDAG(const TargetMachine &tm, CodeGenOpt::Level OL)
    : TM(tm), TSI(nullptr), TLI(nullptr), OptLevel(OL),
      EntryNode(ISD::EntryToken, 0, DebugLoc(), getVTList(MVT::Other)),
      Root(getEntryNode()), NewNodesMustHaveLegalTypes(false),
      UpdateListeners(nullptr) {
  InsertNode(&EntryNode);
  DbgInfo = new SDDbgInfo();
}

void SelectionDAG::init(MachineFunction &NewMF,
                        OptimizationRemarkEmitter &NewORE) {
  MF = &NewMF;
  ORE = &NewORE;
  TLI = getSubtarget().getTargetLowering();
  TSI = getSubtarget().getSelectionDAGInfo();
  Context = &MF->getFunction()->getContext();
}

SelectionDAG::~SelectionDAG() {
  assert(!UpdateListeners && "Dangling registered DAGUpdateListeners");
  allnodes_clear();
  OperandRecycler.clear(OperandAllocator);
  delete DbgInfo;
}

void SelectionDAG::allnodes_clear() {
  assert(&*AllNodes.begin() == &EntryNode);
  AllNodes.remove(AllNodes.begin());
  while (!AllNodes.empty())
    DeallocateNode(&AllNodes.front());
#ifndef NDEBUG
  NextPersistentId = 0;
#endif
}

SDNode *SelectionDAG::FindNodeOrInsertPos(const FoldingSetNodeID &ID,
                                          void *&InsertPos) {
  SDNode *N = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  if (N) {
    switch (N->getOpcode()) {
    default: break;
    case ISD::Constant:
    case ISD::ConstantFP:
      llvm_unreachable("Querying for Constant and ConstantFP nodes requires "
                       "debug location.  Use another overload.");
    }
  }
  return N;
}

SDNode *SelectionDAG::FindNodeOrInsertPos(const FoldingSetNodeID &ID,
                                          const SDLoc &DL, void *&InsertPos) {
  SDNode *N = CSEMap.FindNodeOrInsertPos(ID, InsertPos);
  if (N) {
    switch (N->getOpcode()) {
    case ISD::Constant:
    case ISD::ConstantFP:
      // Erase debug location from the node if the node is used at several
      // different places. Do not propagate one location to all uses as it
      // will cause a worse single stepping debugging experience.
      if (N->getDebugLoc() != DL.getDebugLoc())
        N->setDebugLoc(DebugLoc());
      break;
    default:
      // When the node's point of use is located earlier in the instruction
      // sequence than its prior point of use, update its debug info to the
      // earlier location.
      if (DL.getIROrder() && DL.getIROrder() < N->getIROrder())
        N->setDebugLoc(DL.getDebugLoc());
      break;
    }
  }
  return N;
}

void SelectionDAG::clear() {
  allnodes_clear();
  OperandRecycler.clear(OperandAllocator);
  OperandAllocator.Reset();
  CSEMap.clear();

  ExtendedValueTypeNodes.clear();
  ExternalSymbols.clear();
  TargetExternalSymbols.clear();
  MCSymbols.clear();
  std::fill(CondCodeNodes.begin(), CondCodeNodes.end(),
            static_cast<CondCodeSDNode*>(nullptr));
  std::fill(ValueTypeNodes.begin(), ValueTypeNodes.end(),
            static_cast<SDNode*>(nullptr));

  EntryNode.UseList = nullptr;
  InsertNode(&EntryNode);
  Root = getEntryNode();
  DbgInfo->clear();
}

SDValue SelectionDAG::getFPExtendOrRound(SDValue Op, const SDLoc &DL, EVT VT) {
  return VT.bitsGT(Op.getValueType())
             ? getNode(ISD::FP_EXTEND, DL, VT, Op)
             : getNode(ISD::FP_ROUND, DL, VT, Op, getIntPtrConstant(0, DL));
}

SDValue SelectionDAG::getAnyExtOrTrunc(SDValue Op, const SDLoc &DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ANY_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getSExtOrTrunc(SDValue Op, const SDLoc &DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::SIGN_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getZExtOrTrunc(SDValue Op, const SDLoc &DL, EVT VT) {
  return VT.bitsGT(Op.getValueType()) ?
    getNode(ISD::ZERO_EXTEND, DL, VT, Op) :
    getNode(ISD::TRUNCATE, DL, VT, Op);
}

SDValue SelectionDAG::getBoolExtOrTrunc(SDValue Op, const SDLoc &SL, EVT VT,
                                        EVT OpVT) {
  if (VT.bitsLE(Op.getValueType()))
    return getNode(ISD::TRUNCATE, SL, VT, Op);

  TargetLowering::BooleanContent BType = TLI->getBooleanContents(OpVT);
  return getNode(TLI->getExtendForContent(BType), SL, VT, Op);
}

SDValue SelectionDAG::getZeroExtendInReg(SDValue Op, const SDLoc &DL, EVT VT) {
  assert(!VT.isVector() &&
         "getZeroExtendInReg should use the vector element type instead of "
         "the vector type!");
  if (Op.getValueType() == VT) return Op;
  unsigned BitWidth = Op.getScalarValueSizeInBits();
  APInt Imm = APInt::getLowBitsSet(BitWidth,
                                   VT.getSizeInBits());
  return getNode(ISD::AND, DL, Op.getValueType(), Op,
                 getConstant(Imm, DL, Op.getValueType()));
}

SDValue SelectionDAG::getAnyExtendVectorInReg(SDValue Op, const SDLoc &DL,
                                              EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::ANY_EXTEND_VECTOR_INREG, DL, VT, Op);
}

SDValue SelectionDAG::getSignExtendVectorInReg(SDValue Op, const SDLoc &DL,
                                               EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::SIGN_EXTEND_VECTOR_INREG, DL, VT, Op);
}

SDValue SelectionDAG::getZeroExtendVectorInReg(SDValue Op, const SDLoc &DL,
                                               EVT VT) {
  assert(VT.isVector() && "This DAG node is restricted to vector types.");
  assert(VT.getSizeInBits() == Op.getValueSizeInBits() &&
         "The sizes of the input and result must match in order to perform the "
         "extend in-register.");
  assert(VT.getVectorNumElements() < Op.getValueType().getVectorNumElements() &&
         "The destination vector type must have fewer lanes than the input.");
  return getNode(ISD::ZERO_EXTEND_VECTOR_INREG, DL, VT, Op);
}

/// getNOT - Create a bitwise NOT operation as (XOR Val, -1).
///
SDValue SelectionDAG::getNOT(const SDLoc &DL, SDValue Val, EVT VT) {
  EVT EltVT = VT.getScalarType();
  SDValue NegOne =
    getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), DL, VT);
  return getNode(ISD::XOR, DL, VT, Val, NegOne);
}

SDValue SelectionDAG::getLogicalNOT(const SDLoc &DL, SDValue Val, EVT VT) {
  EVT EltVT = VT.getScalarType();
  SDValue TrueValue;
  switch (TLI->getBooleanContents(VT)) {
    case TargetLowering::ZeroOrOneBooleanContent:
    case TargetLowering::UndefinedBooleanContent:
      TrueValue = getConstant(1, DL, VT);
      break;
    case TargetLowering::ZeroOrNegativeOneBooleanContent:
      TrueValue = getConstant(APInt::getAllOnesValue(EltVT.getSizeInBits()), DL,
                              VT);
      break;
  }
  return getNode(ISD::XOR, DL, VT, Val, TrueValue);
}

SDValue SelectionDAG::getConstant(uint64_t Val, const SDLoc &DL, EVT VT,
                                  bool isT, bool isO) {
  EVT EltVT = VT.getScalarType();
  assert((EltVT.getSizeInBits() >= 64 ||
         (uint64_t)((int64_t)Val >> EltVT.getSizeInBits()) + 1 < 2) &&
         "getConstant with a uint64_t value that doesn't fit in the type!");
  return getConstant(APInt(EltVT.getSizeInBits(), Val), DL, VT, isT, isO);
}

SDValue SelectionDAG::getConstant(const APInt &Val, const SDLoc &DL, EVT VT,
                                  bool isT, bool isO) {
  return getConstant(*ConstantInt::get(*Context, Val), DL, VT, isT, isO);
}

SDValue SelectionDAG::getConstant(const ConstantInt &Val, const SDLoc &DL,
                                  EVT VT, bool isT, bool isO) {
  assert(VT.isInteger() && "Cannot create FP integer constant!");

  EVT EltVT = VT.getScalarType();
  const ConstantInt *Elt = &Val;

  // In some cases the vector type is legal but the element type is illegal and
  // needs to be promoted, for example v8i8 on ARM.  In this case, promote the
  // inserted value (the type does not need to match the vector element type).
  // Any extra bits introduced will be truncated away.
  if (VT.isVector() && TLI->getTypeAction(*getContext(), EltVT) ==
      TargetLowering::TypePromoteInteger) {
   EltVT = TLI->getTypeToTransformTo(*getContext(), EltVT);
   APInt NewVal = Elt->getValue().zextOrTrunc(EltVT.getSizeInBits());
   Elt = ConstantInt::get(*getContext(), NewVal);
  }
  // In other cases the element type is illegal and needs to be expanded, for
  // example v2i64 on MIPS32. In this case, find the nearest legal type, split
  // the value into n parts and use a vector type with n-times the elements.
  // Then bitcast to the type requested.
  // Legalizing constants too early makes the DAGCombiner's job harder so we
  // only legalize if the DAG tells us we must produce legal types.
  else if (NewNodesMustHaveLegalTypes && VT.isVector() &&
           TLI->getTypeAction(*getContext(), EltVT) ==
           TargetLowering::TypeExpandInteger) {
    const APInt &NewVal = Elt->getValue();
    EVT ViaEltVT = TLI->getTypeToTransformTo(*getContext(), EltVT);
    unsigned ViaEltSizeInBits = ViaEltVT.getSizeInBits();
    unsigned ViaVecNumElts = VT.getSizeInBits() / ViaEltSizeInBits;
    EVT ViaVecVT = EVT::getVectorVT(*getContext(), ViaEltVT, ViaVecNumElts);

    // Check the temporary vector is the correct size. If this fails then
    // getTypeToTransformTo() probably returned a type whose size (in bits)
    // isn't a power-of-2 factor of the requested type size.
    assert(ViaVecVT.getSizeInBits() == VT.getSizeInBits());

    SmallVector<SDValue, 2> EltParts;
    for (unsigned i = 0; i < ViaVecNumElts / VT.getVectorNumElements(); ++i) {
      EltParts.push_back(getConstant(NewVal.lshr(i * ViaEltSizeInBits)
                                           .zextOrTrunc(ViaEltSizeInBits), DL,
                                     ViaEltVT, isT, isO));
    }

    // EltParts is currently in little endian order. If we actually want
    // big-endian order then reverse it now.
    if (getDataLayout().isBigEndian())
      std::reverse(EltParts.begin(), EltParts.end());

    // The elements must be reversed when the element order is different
    // to the endianness of the elements (because the BITCAST is itself a
    // vector shuffle in this situation). However, we do not need any code to
    // perform this reversal because getConstant() is producing a vector
    // splat.
    // This situation occurs in MIPS MSA.

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i)
      Ops.insert(Ops.end(), EltParts.begin(), EltParts.end());
    return getNode(ISD::BITCAST, DL, VT, getBuildVector(ViaVecVT, DL, Ops));
  }

  assert(Elt->getBitWidth() == EltVT.getSizeInBits() &&
         "APInt size does not match type size!");
  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), None);
  ID.AddPointer(Elt);
  ID.AddBoolean(isO);
  void *IP = nullptr;
  SDNode *N = nullptr;
  if ((N = FindNodeOrInsertPos(ID, DL, IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = newSDNode<ConstantSDNode>(isT, isO, Elt, DL.getDebugLoc(), EltVT);
    CSEMap.InsertNode(N, IP);
    InsertNode(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector())
    Result = getSplatBuildVector(VT, DL, Result);
  return Result;
}

SDValue SelectionDAG::getIntPtrConstant(uint64_t Val, const SDLoc &DL,
                                        bool isTarget) {
  return getConstant(Val, DL, TLI->getPointerTy(getDataLayout()), isTarget);
}

SDValue SelectionDAG::getConstantFP(const APFloat &V, const SDLoc &DL, EVT VT,
                                    bool isTarget) {
  return getConstantFP(*ConstantFP::get(*getContext(), V), DL, VT, isTarget);
}

SDValue SelectionDAG::getConstantFP(const ConstantFP &V, const SDLoc &DL,
                                    EVT VT, bool isTarget) {
  assert(VT.isFloatingPoint() && "Cannot create integer FP constant!");

  EVT EltVT = VT.getScalarType();

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), None);
  ID.AddPointer(&V);
  void *IP = nullptr;
  SDNode *N = nullptr;
  if ((N = FindNodeOrInsertPos(ID, DL, IP)))
    if (!VT.isVector())
      return SDValue(N, 0);

  if (!N) {
    N = newSDNode<ConstantFPSDNode>(isTarget, &V, DL.getDebugLoc(), EltVT);
    CSEMap.InsertNode(N, IP);
    InsertNode(N);
  }

  SDValue Result(N, 0);
  if (VT.isVector())
    Result = getSplatBuildVector(VT, DL, Result);
  return Result;
}

SDValue SelectionDAG::getConstantFP(double Val, const SDLoc &DL, EVT VT,
                                    bool isTarget) {
  EVT EltVT = VT.getScalarType();
  if (EltVT == MVT::f32)
    return getConstantFP(APFloat((float)Val), DL, VT, isTarget);
  else if (EltVT == MVT::f64)
    return getConstantFP(APFloat(Val), DL, VT, isTarget);
  else if (EltVT == MVT::f80 || EltVT == MVT::f128 || EltVT == MVT::ppcf128 ||
           EltVT == MVT::f16) {
    bool Ignored;
    APFloat APF = APFloat(Val);
    APF.convert(EVTToAPFloatSemantics(EltVT), APFloat::rmNearestTiesToEven,
                &Ignored);
    return getConstantFP(APF, DL, VT, isTarget);
  } else
    llvm_unreachable("Unsupported type in getConstantFP");
}

SDValue SelectionDAG::getGlobalAddress(const GlobalValue *GV, const SDLoc &DL,
                                       EVT VT, int64_t Offset, bool isTargetGA,
                                       unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTargetGA) &&
         "Cannot set target flags on target-independent globals");

  // Truncate (with sign-extension) the offset value to the pointer size.
  unsigned BitWidth = getDataLayout().getPointerTypeSizeInBits(GV->getType());
  if (BitWidth < 64)
    Offset = SignExtend64(Offset, BitWidth);

  unsigned Opc;
  if (GV->isThreadLocal())
    Opc = isTargetGA ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress;
  else
    Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddPointer(GV);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<GlobalAddressSDNode>(
      Opc, DL.getIROrder(), DL.getDebugLoc(), GV, VT, Offset, TargetFlags);
  CSEMap.InsertNode(N, IP);
    InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getFrameIndex(int FI, EVT VT, bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetFrameIndex : ISD::FrameIndex;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(FI);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<FrameIndexSDNode>(FI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getJumpTable(int JTI, EVT VT, bool isTarget,
                                   unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent jump tables");
  unsigned Opc = isTarget ? ISD::TargetJumpTable : ISD::JumpTable;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(JTI);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<JumpTableSDNode>(JTI, VT, isTarget, TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getConstantPool(const Constant *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = MF->getFunction()->optForSize()
                    ? getDataLayout().getABITypeAlignment(C->getType())
                    : getDataLayout().getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  ID.AddPointer(C);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<ConstantPoolSDNode>(isTarget, C, VT, Offset, Alignment,
                                          TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}


SDValue SelectionDAG::getConstantPool(MachineConstantPoolValue *C, EVT VT,
                                      unsigned Alignment, int Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  assert((TargetFlags == 0 || isTarget) &&
         "Cannot set target flags on target-independent globals");
  if (Alignment == 0)
    Alignment = getDataLayout().getPrefTypeAlignment(C->getType());
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  C->addSelectionDAGCSEId(ID);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<ConstantPoolSDNode>(isTarget, C, VT, Offset, Alignment,
                                          TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTargetIndex(int Index, EVT VT, int64_t Offset,
                                     unsigned char TargetFlags) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::TargetIndex, getVTList(VT), None);
  ID.AddInteger(Index);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<TargetIndexSDNode>(Index, VT, Offset, TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BasicBlock, getVTList(MVT::Other), None);
  ID.AddPointer(MBB);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<BasicBlockSDNode>(MBB);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getValueType(EVT VT) {
  if (VT.isSimple() && (unsigned)VT.getSimpleVT().SimpleTy >=
      ValueTypeNodes.size())
    ValueTypeNodes.resize(VT.getSimpleVT().SimpleTy+1);

  SDNode *&N = VT.isExtended() ?
    ExtendedValueTypeNodes[VT] : ValueTypeNodes[VT.getSimpleVT().SimpleTy];

  if (N) return SDValue(N, 0);
  N = newSDNode<VTSDNode>(VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getExternalSymbol(const char *Sym, EVT VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDValue(N, 0);
  N = newSDNode<ExternalSymbolSDNode>(false, Sym, 0, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMCSymbol(MCSymbol *Sym, EVT VT) {
  SDNode *&N = MCSymbols[Sym];
  if (N)
    return SDValue(N, 0);
  N = newSDNode<MCSymbolSDNode>(Sym, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTargetExternalSymbol(const char *Sym, EVT VT,
                                              unsigned char TargetFlags) {
  SDNode *&N =
    TargetExternalSymbols[std::pair<std::string,unsigned char>(Sym,
                                                               TargetFlags)];
  if (N) return SDValue(N, 0);
  N = newSDNode<ExternalSymbolSDNode>(true, Sym, TargetFlags, VT);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getCondCode(ISD::CondCode Cond) {
  if ((unsigned)Cond >= CondCodeNodes.size())
    CondCodeNodes.resize(Cond+1);

  if (!CondCodeNodes[Cond]) {
    auto *N = newSDNode<CondCodeSDNode>(Cond);
    CondCodeNodes[Cond] = N;
    InsertNode(N);
  }

  return SDValue(CondCodeNodes[Cond], 0);
}

/// Swaps the values of N1 and N2. Swaps all indices in the shuffle mask M that
/// point at N1 to point at N2 and indices that point at N2 to point at N1.
static void commuteShuffle(SDValue &N1, SDValue &N2, MutableArrayRef<int> M) {
  std::swap(N1, N2);
  ShuffleVectorSDNode::commuteMask(M);
}

SDValue SelectionDAG::getVectorShuffle(EVT VT, const SDLoc &dl, SDValue N1,
                                       SDValue N2, ArrayRef<int> Mask) {
  assert(VT.getVectorNumElements() == Mask.size() &&
           "Must have the same number of vector elements as mask elements!");
  assert(VT == N1.getValueType() && VT == N2.getValueType() &&
         "Invalid VECTOR_SHUFFLE");

  // Canonicalize shuffle undef, undef -> undef
  if (N1.isUndef() && N2.isUndef())
    return getUNDEF(VT);

  // Validate that all indices in Mask are within the range of the elements
  // input to the shuffle.
  int NElts = Mask.size();
  assert(all_of(Mask, [&](int M) { return M < (NElts * 2); }) &&
         "Index out of range");

  // Copy the mask so we can do any needed cleanup.
  SmallVector<int, 8> MaskVec(Mask.begin(), Mask.end());

  // Canonicalize shuffle v, v -> v, undef
  if (N1 == N2) {
    N2 = getUNDEF(VT);
    for (int i = 0; i != NElts; ++i)
      if (MaskVec[i] >= NElts) MaskVec[i] -= NElts;
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N1.isUndef())
    commuteShuffle(N1, N2, MaskVec);

  // If shuffling a splat, try to blend the splat instead. We do this here so
  // that even when this arises during lowering we don't have to re-handle it.
  auto BlendSplat = [&](BuildVectorSDNode *BV, int Offset) {
    BitVector UndefElements;
    SDValue Splat = BV->getSplatValue(&UndefElements);
    if (!Splat)
      return;

    for (int i = 0; i < NElts; ++i) {
      if (MaskVec[i] < Offset || MaskVec[i] >= (Offset + NElts))
        continue;

      // If this input comes from undef, mark it as such.
      if (UndefElements[MaskVec[i] - Offset]) {
        MaskVec[i] = -1;
        continue;
      }

      // If we can blend a non-undef lane, use that instead.
      if (!UndefElements[i])
        MaskVec[i] = i + Offset;
    }
  };
  if (auto *N1BV = dyn_cast<BuildVectorSDNode>(N1))
    BlendSplat(N1BV, 0);
  if (auto *N2BV = dyn_cast<BuildVectorSDNode>(N2))
    BlendSplat(N2BV, NElts);

  // Canonicalize all index into lhs, -> shuffle lhs, undef
  // Canonicalize all index into rhs, -> shuffle rhs, undef
  bool AllLHS = true, AllRHS = true;
  bool N2Undef = N2.isUndef();
  for (int i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= NElts) {
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
  // Reset our undef status after accounting for the mask.
  N2Undef = N2.isUndef();
  // Re-check whether both sides ended up undef.
  if (N1.isUndef() && N2Undef)
    return getUNDEF(VT);

  // If Identity shuffle return that node.
  bool Identity = true, AllSame = true;
  for (int i = 0; i != NElts; ++i) {
    if (MaskVec[i] >= 0 && MaskVec[i] != i) Identity = false;
    if (MaskVec[i] != MaskVec[0]) AllSame = false;
  }
  if (Identity && NElts)
    return N1;

  // Shuffling a constant splat doesn't change the result.
  if (N2Undef) {
    SDValue V = N1;

    // Look through any bitcasts. We check that these don't change the number
    // (and size) of elements and just changes their types.
    while (V.getOpcode() == ISD::BITCAST)
      V = V->getOperand(0);

    // A splat should always show up as a build vector node.
    if (auto *BV = dyn_cast<BuildVectorSDNode>(V)) {
      BitVector UndefElements;
      SDValue Splat = BV->getSplatValue(&UndefElements);
      // If this is a splat of an undef, shuffling it is also undef.
      if (Splat && Splat.isUndef())
        return getUNDEF(VT);

      bool SameNumElts =
          V.getValueType().getVectorNumElements() == VT.getVectorNumElements();

      // We only have a splat which can skip shuffles if there is a splatted
      // value and no undef lanes rearranged by the shuffle.
      if (Splat && UndefElements.none()) {
        // Splat of <x, x, ..., x>, return <x, x, ..., x>, provided that the
        // number of elements match or the value splatted is a zero constant.
        if (SameNumElts)
          return N1;
        if (auto *C = dyn_cast<ConstantSDNode>(Splat))
          if (C->isNullValue())
            return N1;
      }

      // If the shuffle itself creates a splat, build the vector directly.
      if (AllSame && SameNumElts) {
        EVT BuildVT = BV->getValueType(0);
        const SDValue &Splatted = BV->getOperand(MaskVec[0]);
        SDValue NewBV = getSplatBuildVector(BuildVT, dl, Splatted);

        // We may have jumped through bitcasts, so the type of the
        // BUILD_VECTOR may not match the type of the shuffle.
        if (BuildVT != VT)
          NewBV = getNode(ISD::BITCAST, dl, VT, NewBV);
        return NewBV;
      }
    }
  }

  FoldingSetNodeID ID;
  SDValue Ops[2] = { N1, N2 };
  AddNodeIDNode(ID, ISD::VECTOR_SHUFFLE, getVTList(VT), Ops);
  for (int i = 0; i != NElts; ++i)
    ID.AddInteger(MaskVec[i]);

  void* IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP))
    return SDValue(E, 0);

  // Allocate the mask array for the node out of the BumpPtrAllocator, since
  // SDNode doesn't have access to it.  This memory will be "leaked" when
  // the node is deallocated, but recovered when the NodeAllocator is released.
  int *MaskAlloc = OperandAllocator.Allocate<int>(NElts);
  std::copy(MaskVec.begin(), MaskVec.end(), MaskAlloc);

  auto *N = newSDNode<ShuffleVectorSDNode>(VT, dl.getIROrder(),
                                           dl.getDebugLoc(), MaskAlloc);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getCommutedVectorShuffle(const ShuffleVectorSDNode &SV) {
  MVT VT = SV.getSimpleValueType(0);
  SmallVector<int, 8> MaskVec(SV.getMask().begin(), SV.getMask().end());
  ShuffleVectorSDNode::commuteMask(MaskVec);

  SDValue Op0 = SV.getOperand(0);
  SDValue Op1 = SV.getOperand(1);
  return getVectorShuffle(VT, SDLoc(&SV), Op1, Op0, MaskVec);
}

SDValue SelectionDAG::getRegister(unsigned RegNo, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::Register, getVTList(VT), None);
  ID.AddInteger(RegNo);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<RegisterSDNode>(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getRegisterMask(const uint32_t *RegMask) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::RegisterMask, getVTList(MVT::Untyped), None);
  ID.AddPointer(RegMask);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<RegisterMaskSDNode>(RegMask);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getEHLabel(const SDLoc &dl, SDValue Root,
                                 MCSymbol *Label) {
  FoldingSetNodeID ID;
  SDValue Ops[] = { Root };
  AddNodeIDNode(ID, ISD::EH_LABEL, getVTList(MVT::Other), Ops);
  ID.AddPointer(Label);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<EHLabelSDNode>(dl.getIROrder(), dl.getDebugLoc(), Label);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBlockAddress(const BlockAddress *BA, EVT VT,
                                      int64_t Offset,
                                      bool isTarget,
                                      unsigned char TargetFlags) {
  unsigned Opc = isTarget ? ISD::TargetBlockAddress : ISD::BlockAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), None);
  ID.AddPointer(BA);
  ID.AddInteger(Offset);
  ID.AddInteger(TargetFlags);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<BlockAddressSDNode>(Opc, VT, BA, Offset, TargetFlags);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getSrcValue(const Value *V) {
  assert((!V || V->getType()->isPointerTy()) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::SRCVALUE, getVTList(MVT::Other), None);
  ID.AddPointer(V);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<SrcValueSDNode>(V);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMDNode(const MDNode *MD) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MDNODE_SDNODE, getVTList(MVT::Other), None);
  ID.AddPointer(MD);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<MDNodeSDNode>(MD);
  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getBitcast(EVT VT, SDValue V) {
  if (VT == V.getValueType())
    return V;

  return getNode(ISD::BITCAST, SDLoc(V), VT, V);
}

SDValue SelectionDAG::getAddrSpaceCast(const SDLoc &dl, EVT VT, SDValue Ptr,
                                       unsigned SrcAS, unsigned DestAS) {
  SDValue Ops[] = {Ptr};
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::ADDRSPACECAST, getVTList(VT), Ops);
  ID.AddInteger(SrcAS);
  ID.AddInteger(DestAS);

  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<AddrSpaceCastSDNode>(dl.getIROrder(), dl.getDebugLoc(),
                                           VT, SrcAS, DestAS);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

/// getShiftAmountOperand - Return the specified value casted to
/// the target's desired shift amount type.
SDValue SelectionDAG::getShiftAmountOperand(EVT LHSTy, SDValue Op) {
  EVT OpTy = Op.getValueType();
  EVT ShTy = TLI->getShiftAmountTy(LHSTy, getDataLayout());
  if (OpTy == ShTy || OpTy.isVector()) return Op;

  return getZExtOrTrunc(Op, SDLoc(Op), ShTy);
}

SDValue SelectionDAG::expandVAArg(SDNode *Node) {
  SDLoc dl(Node);
  const TargetLowering &TLI = getTargetLoweringInfo();
  const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  EVT VT = Node->getValueType(0);
  SDValue Tmp1 = Node->getOperand(0);
  SDValue Tmp2 = Node->getOperand(1);
  unsigned Align = Node->getConstantOperandVal(3);

  SDValue VAListLoad = getLoad(TLI.getPointerTy(getDataLayout()), dl, Tmp1,
                               Tmp2, MachinePointerInfo(V));
  SDValue VAList = VAListLoad;

  if (Align > TLI.getMinStackArgumentAlignment()) {
    assert(((Align & (Align-1)) == 0) && "Expected Align to be a power of 2");

    VAList = getNode(ISD::ADD, dl, VAList.getValueType(), VAList,
                     getConstant(Align - 1, dl, VAList.getValueType()));

    VAList = getNode(ISD::AND, dl, VAList.getValueType(), VAList,
                     getConstant(-(int64_t)Align, dl, VAList.getValueType()));
  }

  // Increment the pointer, VAList, to the next vaarg
  Tmp1 = getNode(ISD::ADD, dl, VAList.getValueType(), VAList,
                 getConstant(getDataLayout().getTypeAllocSize(
                                               VT.getTypeForEVT(*getContext())),
                             dl, VAList.getValueType()));
  // Store the incremented VAList to the legalized pointer
  Tmp1 =
      getStore(VAListLoad.getValue(1), dl, Tmp1, Tmp2, MachinePointerInfo(V));
  // Load the actual argument out of the pointer VAList
  return getLoad(VT, dl, Tmp1, VAList, MachinePointerInfo());
}

SDValue SelectionDAG::expandVACopy(SDNode *Node) {
  SDLoc dl(Node);
  const TargetLowering &TLI = getTargetLoweringInfo();
  // This defaults to loading a pointer from the input and storing it to the
  // output, returning the chain.
  const Value *VD = cast<SrcValueSDNode>(Node->getOperand(3))->getValue();
  const Value *VS = cast<SrcValueSDNode>(Node->getOperand(4))->getValue();
  SDValue Tmp1 =
      getLoad(TLI.getPointerTy(getDataLayout()), dl, Node->getOperand(0),
              Node->getOperand(2), MachinePointerInfo(VS));
  return getStore(Tmp1.getValue(1), dl, Tmp1, Node->getOperand(1),
                  MachinePointerInfo(VD));
}

SDValue SelectionDAG::CreateStackTemporary(EVT VT, unsigned minAlign) {
  MachineFrameInfo &MFI = getMachineFunction().getFrameInfo();
  unsigned ByteSize = VT.getStoreSize();
  Type *Ty = VT.getTypeForEVT(*getContext());
  unsigned StackAlign =
      std::max((unsigned)getDataLayout().getPrefTypeAlignment(Ty), minAlign);

  int FrameIdx = MFI.CreateStackObject(ByteSize, StackAlign, false);
  return getFrameIndex(FrameIdx, TLI->getFrameIndexTy(getDataLayout()));
}

SDValue SelectionDAG::CreateStackTemporary(EVT VT1, EVT VT2) {
  unsigned Bytes = std::max(VT1.getStoreSize(), VT2.getStoreSize());
  Type *Ty1 = VT1.getTypeForEVT(*getContext());
  Type *Ty2 = VT2.getTypeForEVT(*getContext());
  const DataLayout &DL = getDataLayout();
  unsigned Align =
      std::max(DL.getPrefTypeAlignment(Ty1), DL.getPrefTypeAlignment(Ty2));

  MachineFrameInfo &MFI = getMachineFunction().getFrameInfo();
  int FrameIdx = MFI.CreateStackObject(Bytes, Align, false);
  return getFrameIndex(FrameIdx, TLI->getFrameIndexTy(getDataLayout()));
}

SDValue SelectionDAG::FoldSetCC(EVT VT, SDValue N1, SDValue N2,
                                ISD::CondCode Cond, const SDLoc &dl) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return getConstant(0, dl, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2: {
    TargetLowering::BooleanContent Cnt =
        TLI->getBooleanContents(N1->getValueType(0));
    return getConstant(
        Cnt == TargetLowering::ZeroOrNegativeOneBooleanContent ? -1ULL : 1, dl,
        VT);
  }

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

  if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2)) {
    const APInt &C2 = N2C->getAPIntValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1)) {
      const APInt &C1 = N1C->getAPIntValue();

      switch (Cond) {
      default: llvm_unreachable("Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, dl, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, dl, VT);
      case ISD::SETULT: return getConstant(C1.ult(C2), dl, VT);
      case ISD::SETUGT: return getConstant(C1.ugt(C2), dl, VT);
      case ISD::SETULE: return getConstant(C1.ule(C2), dl, VT);
      case ISD::SETUGE: return getConstant(C1.uge(C2), dl, VT);
      case ISD::SETLT:  return getConstant(C1.slt(C2), dl, VT);
      case ISD::SETGT:  return getConstant(C1.sgt(C2), dl, VT);
      case ISD::SETLE:  return getConstant(C1.sle(C2), dl, VT);
      case ISD::SETGE:  return getConstant(C1.sge(C2), dl, VT);
      }
    }
  }
  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1)) {
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2)) {
      APFloat::cmpResult R = N1C->getValueAPF().compare(N2C->getValueAPF());
      switch (Cond) {
      default: break;
      case ISD::SETEQ:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETOEQ: return getConstant(R==APFloat::cmpEqual, dl, VT);
      case ISD::SETNE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETONE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETLT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETOLT: return getConstant(R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETGT:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETOGT: return getConstant(R==APFloat::cmpGreaterThan, dl, VT);
      case ISD::SETLE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETOLE: return getConstant(R==APFloat::cmpLessThan ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETGE:  if (R==APFloat::cmpUnordered)
                          return getUNDEF(VT);
                        LLVM_FALLTHROUGH;
      case ISD::SETOGE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETO:   return getConstant(R!=APFloat::cmpUnordered, dl, VT);
      case ISD::SETUO:  return getConstant(R==APFloat::cmpUnordered, dl, VT);
      case ISD::SETUEQ: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpEqual, dl, VT);
      case ISD::SETUNE: return getConstant(R!=APFloat::cmpEqual, dl, VT);
      case ISD::SETULT: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpLessThan, dl, VT);
      case ISD::SETUGT: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpUnordered, dl, VT);
      case ISD::SETULE: return getConstant(R!=APFloat::cmpGreaterThan, dl, VT);
      case ISD::SETUGE: return getConstant(R!=APFloat::cmpLessThan, dl, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      ISD::CondCode SwappedCond = ISD::getSetCCSwappedOperands(Cond);
      MVT CompVT = N1.getValueType().getSimpleVT();
      if (!TLI->isCondCodeLegal(SwappedCond, CompVT))
        return SDValue();

      return getSetCC(dl, VT, N2, N1, SwappedCond);
    }
  }

  // Could not fold it.
  return SDValue();
}

/// SignBitIsZero - Return true if the sign bit of Op is known to be zero.  We
/// use this predicate to simplify operations downstream.
bool SelectionDAG::SignBitIsZero(SDValue Op, unsigned Depth) const {
  unsigned BitWidth = Op.getScalarValueSizeInBits();
  return MaskedValueIsZero(Op, APInt::getSignMask(BitWidth), Depth);
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool SelectionDAG::MaskedValueIsZero(SDValue Op, const APInt &Mask,
                                     unsigned Depth) const {
  KnownBits Known;
  computeKnownBits(Op, Known, Depth);
  return Mask.isSubsetOf(Known.Zero);
}

/// If a SHL/SRA/SRL node has a constant or splat constant shift amount that
/// is less than the element bit-width of the shift node, return it.
static const APInt *getValidShiftAmountConstant(SDValue V) {
  if (ConstantSDNode *SA = isConstOrConstSplat(V.getOperand(1))) {
    // Shifting more than the bitwidth is not valid.
    const APInt &ShAmt = SA->getAPIntValue();
    if (ShAmt.ult(V.getScalarValueSizeInBits()))
      return &ShAmt;
  }
  return nullptr;
}

/// Determine which bits of Op are known to be either zero or one and return
/// them in Known. For vectors, the known bits are those that are shared by
/// every vector element.
void SelectionDAG::computeKnownBits(SDValue Op, KnownBits &Known,
                                    unsigned Depth) const {
  EVT VT = Op.getValueType();
  APInt DemandedElts = VT.isVector()
                           ? APInt::getAllOnesValue(VT.getVectorNumElements())
                           : APInt(1, 1);
  computeKnownBits(Op, Known, DemandedElts, Depth);
}

/// Determine which bits of Op are known to be either zero or one and return
/// them in Known. The DemandedElts argument allows us to only collect the known
/// bits that are shared by the requested vector elements.
void SelectionDAG::computeKnownBits(SDValue Op, KnownBits &Known,
                                    const APInt &DemandedElts,
                                    unsigned Depth) const {
  unsigned BitWidth = Op.getScalarValueSizeInBits();

  Known = KnownBits(BitWidth);   // Don't know anything.
  if (Depth == 6)
    return;  // Limit search depth.

  KnownBits Known2;
  unsigned NumElts = DemandedElts.getBitWidth();

  if (!DemandedElts)
    return;  // No demanded elts, better to assume we don't know anything.

  unsigned Opcode = Op.getOpcode();
  switch (Opcode) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    Known.One = cast<ConstantSDNode>(Op)->getAPIntValue();
    Known.Zero = ~Known.One;
    break;
  case ISD::BUILD_VECTOR:
    // Collect the known bits that are shared by every demanded vector element.
    assert(NumElts == Op.getValueType().getVectorNumElements() &&
           "Unexpected vector size");
    Known.Zero.setAllBits(); Known.One.setAllBits();
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;

      SDValue SrcOp = Op.getOperand(i);
      computeKnownBits(SrcOp, Known2, Depth + 1);

      // BUILD_VECTOR can implicitly truncate sources, we must handle this.
      if (SrcOp.getValueSizeInBits() != BitWidth) {
        assert(SrcOp.getValueSizeInBits() > BitWidth &&
               "Expected BUILD_VECTOR implicit truncation");
        Known2 = Known2.trunc(BitWidth);
      }

      // Known bits are the values that are shared by every demanded element.
      Known.One &= Known2.One;
      Known.Zero &= Known2.Zero;

      // If we don't know any bits, early out.
      if (!Known.One && !Known.Zero)
        break;
    }
    break;
  case ISD::VECTOR_SHUFFLE: {
    // Collect the known bits that are shared by every vector element referenced
    // by the shuffle.
    APInt DemandedLHS(NumElts, 0), DemandedRHS(NumElts, 0);
    Known.Zero.setAllBits(); Known.One.setAllBits();
    const ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op);
    assert(NumElts == SVN->getMask().size() && "Unexpected vector size");
    for (unsigned i = 0; i != NumElts; ++i) {
      if (!DemandedElts[i])
        continue;

      int M = SVN->getMaskElt(i);
      if (M < 0) {
        // For UNDEF elements, we don't know anything about the common state of
        // the shuffle result.
        Known.resetAll();
        DemandedLHS.clearAllBits();
        DemandedRHS.clearAllBits();
        break;
      }

      if ((unsigned)M < NumElts)
        DemandedLHS.setBit((unsigned)M % NumElts);
      else
        DemandedRHS.setBit((unsigned)M % NumElts);
    }
    // Known bits are the values that are shared by every demanded element.
    if (!!DemandedLHS) {
      SDValue LHS = Op.getOperand(0);
      computeKnownBits(LHS, Known2, DemandedLHS, Depth + 1);
      Known.One &= Known2.One;
      Known.Zero &= Known2.Zero;
    }
    // If we don't know any bits, early out.
    if (!Known.One && !Known.Zero)
      break;
    if (!!DemandedRHS) {
      SDValue RHS = Op.getOperand(1);
      computeKnownBits(RHS, Known2, DemandedRHS, Depth + 1);
      Known.One &= Known2.One;
      Known.Zero &= Known2.Zero;
    }
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // Split DemandedElts and test each of the demanded subvectors.
    Known.Zero.setAllBits(); Known.One.setAllBits();
    EVT SubVectorVT = Op.getOperand(0).getValueType();
    unsigned NumSubVectorElts = SubVectorVT.getVectorNumElements();
    unsigned NumSubVectors = Op.getNumOperands();
    for (unsigned i = 0; i != NumSubVectors; ++i) {
      APInt DemandedSub = DemandedElts.lshr(i * NumSubVectorElts);
      DemandedSub = DemandedSub.trunc(NumSubVectorElts);
      if (!!DemandedSub) {
        SDValue Sub = Op.getOperand(i);
        computeKnownBits(Sub, Known2, DemandedSub, Depth + 1);
        Known.One &= Known2.One;
        Known.Zero &= Known2.Zero;
      }
      // If we don't know any bits, early out.
      if (!Known.One && !Known.Zero)
        break;
    }
    break;
  }
  case ISD::EXTRACT_SUBVECTOR: {
    // If we know the element index, just demand that subvector elements,
    // otherwise demand them all.
    SDValue Src = Op.getOperand(0);
    ConstantSDNode *SubIdx = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    unsigned NumSrcElts = Src.getValueType().getVectorNumElements();
    if (SubIdx && SubIdx->getAPIntValue().ule(NumSrcElts - NumElts)) {
      // Offset the demanded elts by the subvector index.
      uint64_t Idx = SubIdx->getZExtValue();
      APInt DemandedSrc = DemandedElts.zext(NumSrcElts).shl(Idx);
      computeKnownBits(Src, Known, DemandedSrc, Depth + 1);
    } else {
      computeKnownBits(Src, Known, Depth + 1);
    }
    break;
  }
  case ISD::BITCAST: {
    SDValue N0 = Op.getOperand(0);
    unsigned SubBitWidth = N0.getScalarValueSizeInBits();

    // Ignore bitcasts from floating point.
    if (!N0.getValueType().isInteger())
      break;

    // Fast handling of 'identity' bitcasts.
    if (BitWidth == SubBitWidth) {
      computeKnownBits(N0, Known, DemandedElts, Depth + 1);
      break;
    }

    // Support big-endian targets when it becomes useful.
    bool IsLE = getDataLayout().isLittleEndian();
    if (!IsLE)
      break;

    // Bitcast 'small element' vector to 'large element' scalar/vector.
    if ((BitWidth % SubBitWidth) == 0) {
      assert(N0.getValueType().isVector() && "Expected bitcast from vector");

      // Collect known bits for the (larger) output by collecting the known
      // bits from each set of sub elements and shift these into place.
      // We need to separately call computeKnownBits for each set of
      // sub elements as the knownbits for each is likely to be different.
      unsigned SubScale = BitWidth / SubBitWidth;
      APInt SubDemandedElts(NumElts * SubScale, 0);
      for (unsigned i = 0; i != NumElts; ++i)
        if (DemandedElts[i])
          SubDemandedElts.setBit(i * SubScale);

      for (unsigned i = 0; i != SubScale; ++i) {
        computeKnownBits(N0, Known2, SubDemandedElts.shl(i),
                         Depth + 1);
        Known.One |= Known2.One.zext(BitWidth).shl(SubBitWidth * i);
        Known.Zero |= Known2.Zero.zext(BitWidth).shl(SubBitWidth * i);
      }
    }

    // Bitcast 'large element' scalar/vector to 'small element' vector.
    if ((SubBitWidth % BitWidth) == 0) {
      assert(Op.getValueType().isVector() && "Expected bitcast to vector");

      // Collect known bits for the (smaller) output by collecting the known
      // bits from the overlapping larger input elements and extracting the
      // sub sections we actually care about.
      unsigned SubScale = SubBitWidth / BitWidth;
      APInt SubDemandedElts(NumElts / SubScale, 0);
      for (unsigned i = 0; i != NumElts; ++i)
        if (DemandedElts[i])
          SubDemandedElts.setBit(i / SubScale);

      computeKnownBits(N0, Known2, SubDemandedElts, Depth + 1);

      Known.Zero.setAllBits(); Known.One.setAllBits();
      for (unsigned i = 0; i != NumElts; ++i)
        if (DemandedElts[i]) {
          unsigned Offset = (i % SubScale) * BitWidth;
          Known.One &= Known2.One.lshr(Offset).trunc(BitWidth);
          Known.Zero &= Known2.Zero.lshr(Offset).trunc(BitWidth);
          // If we don't know any bits, early out.
          if (!Known.One && !Known.Zero)
            break;
        }
    }
    break;
  }
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBits(Op.getOperand(1), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

    // Output known-1 bits are only known if set in both the LHS & RHS.
    Known.One &= Known2.One;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    Known.Zero |= Known2.Zero;
    break;
  case ISD::OR:
    computeKnownBits(Op.getOperand(1), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    Known.Zero &= Known2.Zero;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    Known.One |= Known2.One;
    break;
  case ISD::XOR: {
    computeKnownBits(Op.getOperand(1), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (Known.Zero & Known2.Zero) | (Known.One & Known2.One);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    Known.One = (Known.Zero & Known2.One) | (Known.One & Known2.Zero);
    Known.Zero = KnownZeroOut;
    break;
  }
  case ISD::MUL: {
    computeKnownBits(Op.getOperand(1), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conservative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    unsigned TrailZ = Known.Zero.countTrailingOnes() +
                      Known2.Zero.countTrailingOnes();
    unsigned LeadZ =  std::max(Known.Zero.countLeadingOnes() +
                               Known2.Zero.countLeadingOnes(),
                               BitWidth) - BitWidth;

    Known.resetAll();
    Known.Zero.setLowBits(std::min(TrailZ, BitWidth));
    Known.Zero.setHighBits(std::min(LeadZ, BitWidth));
    break;
  }
  case ISD::UDIV: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    unsigned LeadZ = Known2.Zero.countLeadingOnes();

    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);
    unsigned RHSUnknownLeadingOnes = Known2.One.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    Known.Zero.setHighBits(LeadZ);
    break;
  }
  case ISD::SELECT:
    computeKnownBits(Op.getOperand(2), Known, Depth+1);
    // If we don't know any bits, early out.
    if (!Known.One && !Known.Zero)
      break;
    computeKnownBits(Op.getOperand(1), Known2, Depth+1);

    // Only known if known in both the LHS and RHS.
    Known.One &= Known2.One;
    Known.Zero &= Known2.Zero;
    break;
  case ISD::SELECT_CC:
    computeKnownBits(Op.getOperand(3), Known, Depth+1);
    // If we don't know any bits, early out.
    if (!Known.One && !Known.Zero)
      break;
    computeKnownBits(Op.getOperand(2), Known2, Depth+1);

    // Only known if known in both the LHS and RHS.
    Known.One &= Known2.One;
    Known.Zero &= Known2.Zero;
    break;
  case ISD::SMULO:
  case ISD::UMULO:
    if (Op.getResNo() != 1)
      break;
    // The boolean result conforms to getBooleanContents.
    // If we know the result of a setcc has the top bits zero, use this info.
    // We know that we have an integer-based boolean since these operations
    // are only available for integer.
    if (TLI->getBooleanContents(Op.getValueType().isVector(), false) ==
            TargetLowering::ZeroOrOneBooleanContent &&
        BitWidth > 1)
      Known.Zero.setBitsFrom(1);
    break;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
            TargetLowering::ZeroOrOneBooleanContent &&
        BitWidth > 1)
      Known.Zero.setBitsFrom(1);
    break;
  case ISD::SHL:
    if (const APInt *ShAmt = getValidShiftAmountConstant(Op)) {
      computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
      Known.Zero <<= *ShAmt;
      Known.One <<= *ShAmt;
      // Low bits are known zero.
      Known.Zero.setLowBits(ShAmt->getZExtValue());
    }
    break;
  case ISD::SRL:
    if (const APInt *ShAmt = getValidShiftAmountConstant(Op)) {
      computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
      Known.Zero.lshrInPlace(*ShAmt);
      Known.One.lshrInPlace(*ShAmt);
      // High bits are known zero.
      Known.Zero.setHighBits(ShAmt->getZExtValue());
    }
    break;
  case ISD::SRA:
    if (const APInt *ShAmt = getValidShiftAmountConstant(Op)) {
      computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
      Known.Zero.lshrInPlace(*ShAmt);
      Known.One.lshrInPlace(*ShAmt);
      // If we know the value of the sign bit, then we know it is copied across
      // the high bits by the shift amount.
      APInt SignMask = APInt::getSignMask(BitWidth);
      SignMask.lshrInPlace(*ShAmt);  // Adjust to where it is now in the mask.
      if (Known.Zero.intersects(SignMask)) {
        Known.Zero.setHighBits(ShAmt->getZExtValue());// New bits are known zero.
      } else if (Known.One.intersects(SignMask)) {
        Known.One.setHighBits(ShAmt->getZExtValue()); // New bits are known one.
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    EVT EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    unsigned EBits = EVT.getScalarSizeInBits();

    // Sign extension.  Compute the demanded bits in the result that are not
    // present in the input.
    APInt NewBits = APInt::getHighBitsSet(BitWidth, BitWidth - EBits);

    APInt InSignMask = APInt::getSignMask(EBits);
    APInt InputDemandedBits = APInt::getLowBitsSet(BitWidth, EBits);

    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InSignMask = InSignMask.zext(BitWidth);
    if (NewBits.getBoolValue())
      InputDemandedBits |= InSignMask;

    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
    Known.One &= InputDemandedBits;
    Known.Zero &= InputDemandedBits;

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (Known.Zero.intersects(InSignMask)) {        // Input sign bit known clear
      Known.Zero |= NewBits;
      Known.One  &= ~NewBits;
    } else if (Known.One.intersects(InSignMask)) {  // Input sign bit known set
      Known.One  |= NewBits;
      Known.Zero &= ~NewBits;
    } else {                              // Input sign bit unknown
      Known.Zero &= ~NewBits;
      Known.One  &= ~NewBits;
    }
    break;
  }
  case ISD::CTTZ:
  case ISD::CTTZ_ZERO_UNDEF: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    // If we have a known 1, its position is our upper bound.
    unsigned PossibleTZ = Known2.One.countTrailingZeros();
    unsigned LowBits = Log2_32(PossibleTZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    // If we have a known 1, its position is our upper bound.
    unsigned PossibleLZ = Known2.One.countLeadingZeros();
    unsigned LowBits = Log2_32(PossibleLZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case ISD::CTPOP: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    // If we know some of the bits are zero, they can't be one.
    unsigned PossibleOnes = BitWidth - Known2.Zero.countPopulation();
    Known.Zero.setBitsFrom(Log2_32(PossibleOnes) + 1);
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    // If this is a ZEXTLoad and we are looking at the loaded value.
    if (ISD::isZEXTLoad(Op.getNode()) && Op.getResNo() == 0) {
      EVT VT = LD->getMemoryVT();
      unsigned MemBits = VT.getScalarSizeInBits();
      Known.Zero.setBitsFrom(MemBits);
    } else if (const MDNode *Ranges = LD->getRanges()) {
      if (LD->getExtensionType() == ISD::NON_EXTLOAD)
        computeKnownBitsFromRangeMetadata(*Ranges, Known);
    }
    break;
  }
  case ISD::ZERO_EXTEND_VECTOR_INREG: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarSizeInBits();
    Known = Known.trunc(InBits);
    computeKnownBits(Op.getOperand(0), Known,
                     DemandedElts.zext(InVT.getVectorNumElements()),
                     Depth + 1);
    Known = Known.zext(BitWidth);
    Known.Zero.setBitsFrom(InBits);
    break;
  }
  case ISD::ZERO_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarSizeInBits();
    Known = Known.trunc(InBits);
    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
    Known = Known.zext(BitWidth);
    Known.Zero.setBitsFrom(InBits);
    break;
  }
  // TODO ISD::SIGN_EXTEND_VECTOR_INREG
  case ISD::SIGN_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarSizeInBits();

    Known = Known.trunc(InBits);
    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);

    // If the sign bit is known to be zero or one, then sext will extend
    // it to the top bits, else it will just zext.
    Known = Known.sext(BitWidth);
    break;
  }
  case ISD::ANY_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarSizeInBits();
    Known = Known.trunc(InBits);
    computeKnownBits(Op.getOperand(0), Known, Depth+1);
    Known = Known.zext(BitWidth);
    break;
  }
  case ISD::TRUNCATE: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarSizeInBits();
    Known = Known.zext(InBits);
    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
    Known = Known.trunc(BitWidth);
    break;
  }
  case ISD::AssertZext: {
    EVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth, VT.getSizeInBits());
    computeKnownBits(Op.getOperand(0), Known, Depth+1);
    Known.Zero |= (~InMask);
    Known.One  &= (~Known.Zero);
    break;
  }
  case ISD::FGETSIGN:
    // All bits are zero except the low bit.
    Known.Zero.setBitsFrom(1);
    break;
  case ISD::USUBO:
  case ISD::SSUBO:
    if (Op.getResNo() == 1) {
      // If we know the result of a setcc has the top bits zero, use this info.
      if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
              TargetLowering::ZeroOrOneBooleanContent &&
          BitWidth > 1)
        Known.Zero.setBitsFrom(1);
      break;
    }
    LLVM_FALLTHROUGH;
  case ISD::SUB:
  case ISD::SUBC: {
    if (ConstantSDNode *CLHS = isConstOrConstSplat(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (CLHS->getAPIntValue().isNonNegative()) {
        unsigned NLZ = (CLHS->getAPIntValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        computeKnownBits(Op.getOperand(1), Known2, DemandedElts,
                         Depth + 1);

        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((Known2.Zero & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getAPIntValue().countLeadingZeros();
          // Top bits known zero.
          Known.Zero.setHighBits(NLZ2);
        }
      }
    }

    // If low bits are know to be zero in both operands, then we know they are
    // going to be 0 in the result. Both addition and complement operations
    // preserve the low zero bits.
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    unsigned KnownZeroLow = Known2.Zero.countTrailingOnes();
    if (KnownZeroLow == 0)
      break;

    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);
    KnownZeroLow = std::min(KnownZeroLow,
                            Known2.Zero.countTrailingOnes());
    Known.Zero.setLowBits(KnownZeroLow);
    break;
  }
  case ISD::UADDO:
  case ISD::SADDO:
  case ISD::ADDCARRY:
    if (Op.getResNo() == 1) {
      // If we know the result of a setcc has the top bits zero, use this info.
      if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
              TargetLowering::ZeroOrOneBooleanContent &&
          BitWidth > 1)
        Known.Zero.setBitsFrom(1);
      break;
    }
    LLVM_FALLTHROUGH;
  case ISD::ADD:
  case ISD::ADDC:
  case ISD::ADDE: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    // Output known-0 bits are also known if the top bits of each input are
    // known to be clear. For example, if one input has the top 10 bits clear
    // and the other has the top 8 bits clear, we know the top 7 bits of the
    // output must be clear.
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    unsigned KnownZeroHigh = Known2.Zero.countLeadingOnes();
    unsigned KnownZeroLow = Known2.Zero.countTrailingOnes();

    computeKnownBits(Op.getOperand(1), Known2, DemandedElts,
                     Depth + 1);
    KnownZeroHigh = std::min(KnownZeroHigh,
                             Known2.Zero.countLeadingOnes());
    KnownZeroLow = std::min(KnownZeroLow,
                            Known2.Zero.countTrailingOnes());

    if (Opcode == ISD::ADDE || Opcode == ISD::ADDCARRY) {
      // With ADDE and ADDCARRY, a carry bit may be added in, so we can only
      // use this information if we know (at least) that the low two bits are
      // clear. We then return to the caller that the low bit is unknown but
      // that other bits are known zero.
      if (KnownZeroLow >= 2)
        Known.Zero.setBits(1, KnownZeroLow);
      break;
    }

    Known.Zero.setLowBits(KnownZeroLow);
    if (KnownZeroHigh > 1)
      Known.Zero.setHighBits(KnownZeroHigh - 1);
    break;
  }
  case ISD::SREM:
    if (ConstantSDNode *Rem = isConstOrConstSplat(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue().abs();
      if (RA.isPowerOf2()) {
        APInt LowBits = RA - 1;
        computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

        // The low bits of the first operand are unchanged by the srem.
        Known.Zero = Known2.Zero & LowBits;
        Known.One = Known2.One & LowBits;

        // If the first operand is non-negative or has all low bits zero, then
        // the upper bits are all zero.
        if (Known2.Zero[BitWidth-1] || ((Known2.Zero & LowBits) == LowBits))
          Known.Zero |= ~LowBits;

        // If the first operand is negative and not all low bits are zero, then
        // the upper bits are all one.
        if (Known2.One[BitWidth-1] && ((Known2.One & LowBits) != 0))
          Known.One |= ~LowBits;
        assert((Known.Zero & Known.One) == 0&&"Bits known to be one AND zero?");
      }
    }
    break;
  case ISD::UREM: {
    if (ConstantSDNode *Rem = isConstOrConstSplat(Op.getOperand(1))) {
      const APInt &RA = Rem->getAPIntValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

        // The upper bits are all zero, the lower ones are unchanged.
        Known.Zero = Known2.Zero | ~LowBits;
        Known.One = Known2.One & LowBits;
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);

    uint32_t Leaders = std::max(Known.Zero.countLeadingOnes(),
                                Known2.Zero.countLeadingOnes());
    Known.resetAll();
    Known.Zero.setHighBits(Leaders);
    break;
  }
  case ISD::EXTRACT_ELEMENT: {
    computeKnownBits(Op.getOperand(0), Known, Depth+1);
    const unsigned Index = Op.getConstantOperandVal(1);
    const unsigned BitWidth = Op.getValueSizeInBits();

    // Remove low part of known bits mask
    Known.Zero = Known.Zero.getHiBits(Known.Zero.getBitWidth() - Index * BitWidth);
    Known.One = Known.One.getHiBits(Known.One.getBitWidth() - Index * BitWidth);

    // Remove high part of known bit mask
    Known = Known.trunc(BitWidth);
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    SDValue InVec = Op.getOperand(0);
    SDValue EltNo = Op.getOperand(1);
    EVT VecVT = InVec.getValueType();
    const unsigned BitWidth = Op.getValueSizeInBits();
    const unsigned EltBitWidth = VecVT.getScalarSizeInBits();
    const unsigned NumSrcElts = VecVT.getVectorNumElements();
    // If BitWidth > EltBitWidth the value is anyext:ed. So we do not know
    // anything about the extended bits.
    if (BitWidth > EltBitWidth)
      Known = Known.trunc(EltBitWidth);
    ConstantSDNode *ConstEltNo = dyn_cast<ConstantSDNode>(EltNo);
    if (ConstEltNo && ConstEltNo->getAPIntValue().ult(NumSrcElts)) {
      // If we know the element index, just demand that vector element.
      unsigned Idx = ConstEltNo->getZExtValue();
      APInt DemandedElt = APInt::getOneBitSet(NumSrcElts, Idx);
      computeKnownBits(InVec, Known, DemandedElt, Depth + 1);
    } else {
      // Unknown element index, so ignore DemandedElts and demand them all.
      computeKnownBits(InVec, Known, Depth + 1);
    }
    if (BitWidth > EltBitWidth)
      Known = Known.zext(BitWidth);
    break;
  }
  case ISD::INSERT_VECTOR_ELT: {
    SDValue InVec = Op.getOperand(0);
    SDValue InVal = Op.getOperand(1);
    SDValue EltNo = Op.getOperand(2);

    ConstantSDNode *CEltNo = dyn_cast<ConstantSDNode>(EltNo);
    if (CEltNo && CEltNo->getAPIntValue().ult(NumElts)) {
      // If we know the element index, split the demand between the
      // source vector and the inserted element.
      Known.Zero = Known.One = APInt::getAllOnesValue(BitWidth);
      unsigned EltIdx = CEltNo->getZExtValue();

      // If we demand the inserted element then add its common known bits.
      if (DemandedElts[EltIdx]) {
        computeKnownBits(InVal, Known2, Depth + 1);
        Known.One &= Known2.One.zextOrTrunc(Known.One.getBitWidth());
        Known.Zero &= Known2.Zero.zextOrTrunc(Known.Zero.getBitWidth());;
      }

      // If we demand the source vector then add its common known bits, ensuring
      // that we don't demand the inserted element.
      APInt VectorElts = DemandedElts & ~(APInt::getOneBitSet(NumElts, EltIdx));
      if (!!VectorElts) {
        computeKnownBits(InVec, Known2, VectorElts, Depth + 1);
        Known.One &= Known2.One;
        Known.Zero &= Known2.Zero;
      }
    } else {
      // Unknown element index, so ignore DemandedElts and demand them all.
      computeKnownBits(InVec, Known, Depth + 1);
      computeKnownBits(InVal, Known2, Depth + 1);
      Known.One &= Known2.One.zextOrTrunc(Known.One.getBitWidth());
      Known.Zero &= Known2.Zero.zextOrTrunc(Known.Zero.getBitWidth());;
    }
    break;
  }
  case ISD::BITREVERSE: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    Known.Zero = Known2.Zero.reverseBits();
    Known.One = Known2.One.reverseBits();
    break;
  }
  case ISD::BSWAP: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);
    Known.Zero = Known2.Zero.byteSwap();
    Known.One = Known2.One.byteSwap();
    break;
  }
  case ISD::ABS: {
    computeKnownBits(Op.getOperand(0), Known2, DemandedElts, Depth + 1);

    // If the source's MSB is zero then we know the rest of the bits already.
    if (Known2.isNonNegative()) {
      Known.Zero = Known2.Zero;
      Known.One = Known2.One;
      break;
    }

    // We only know that the absolute values's MSB will be zero iff there is
    // a set bit that isn't the sign bit (otherwise it could be INT_MIN).
    Known2.One.clearSignBit();
    if (Known2.One.getBoolValue()) {
      Known.Zero = APInt::getSignMask(BitWidth);
      break;
    }
    break;
  }
  case ISD::UMIN: {
    computeKnownBits(Op.getOperand(0), Known, DemandedElts, Depth + 1);
    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);

    // UMIN - we know that the result will have the maximum of the
    // known zero leading bits of the inputs.
    unsigned LeadZero = Known.Zero.countLeadingOnes();
    LeadZero = std::max(LeadZero, Known2.Zero.countLeadingOnes());

    Known.Zero &= Known2.Zero;
    Known.One &= Known2.One;
    Known.Zero.setHighBits(LeadZero);
    break;
  }
  case ISD::UMAX: {
    computeKnownBits(Op.getOperand(0), Known, DemandedElts,
                     Depth + 1);
    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);

    // UMAX - we know that the result will have the maximum of the
    // known one leading bits of the inputs.
    unsigned LeadOne = Known.One.countLeadingOnes();
    LeadOne = std::max(LeadOne, Known2.One.countLeadingOnes());

    Known.Zero &= Known2.Zero;
    Known.One &= Known2.One;
    Known.One.setHighBits(LeadOne);
    break;
  }
  case ISD::SMIN:
  case ISD::SMAX: {
    computeKnownBits(Op.getOperand(0), Known, DemandedElts,
                     Depth + 1);
    // If we don't know any bits, early out.
    if (!Known.One && !Known.Zero)
      break;
    computeKnownBits(Op.getOperand(1), Known2, DemandedElts, Depth + 1);
    Known.Zero &= Known2.Zero;
    Known.One &= Known2.One;
    break;
  }
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    if (unsigned Align = InferPtrAlignment(Op)) {
      // The low bits are known zero if the pointer is aligned.
      Known.Zero.setLowBits(Log2_32(Align));
      break;
    }
    break;

  default:
    if (Opcode < ISD::BUILTIN_OP_END)
      break;
    LLVM_FALLTHROUGH;
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    // Allow the target to implement this method for its nodes.
    TLI->computeKnownBitsForTargetNode(Op, Known, DemandedElts, *this, Depth);
    break;
  }

  assert((Known.Zero & Known.One) == 0 && "Bits known to be one AND zero?");
}

SelectionDAG::OverflowKind SelectionDAG::computeOverflowKind(SDValue N0,
                                                             SDValue N1) const {
  // X + 0 never overflow
  if (isNullConstant(N1))
    return OFK_Never;

  KnownBits N1Known;
  computeKnownBits(N1, N1Known);
  if (N1Known.Zero.getBoolValue()) {
    KnownBits N0Known;
    computeKnownBits(N0, N0Known);

    bool overflow;
    (void)(~N0Known.Zero).uadd_ov(~N1Known.Zero, overflow);
    if (!overflow)
      return OFK_Never;
  }

  // mulhi + 1 never overflow
  if (N0.getOpcode() == ISD::UMUL_LOHI && N0.getResNo() == 1 &&
      (~N1Known.Zero & 0x01) == ~N1Known.Zero)
    return OFK_Never;

  if (N1.getOpcode() == ISD::UMUL_LOHI && N1.getResNo() == 1) {
    KnownBits N0Known;
    computeKnownBits(N0, N0Known);

    if ((~N0Known.Zero & 0x01) == ~N0Known.Zero)
      return OFK_Never;
  }

  return OFK_Sometime;
}

bool SelectionDAG::isKnownToBeAPowerOfTwo(SDValue Val) const {
  EVT OpVT = Val.getValueType();
  unsigned BitWidth = OpVT.getScalarSizeInBits();

  // Is the constant a known power of 2?
  if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(Val))
    return Const->getAPIntValue().zextOrTrunc(BitWidth).isPowerOf2();

  // A left-shift of a constant one will have exactly one bit set because
  // shifting the bit off the end is undefined.
  if (Val.getOpcode() == ISD::SHL) {
    auto *C = isConstOrConstSplat(Val.getOperand(0));
    if (C && C->getAPIntValue() == 1)
      return true;
  }

  // Similarly, a logical right-shift of a constant sign-bit will have exactly
  // one bit set.
  if (Val.getOpcode() == ISD::SRL) {
    auto *C = isConstOrConstSplat(Val.getOperand(0));
    if (C && C->getAPIntValue().isSignMask())
      return true;
  }

  // Are all operands of a build vector constant powers of two?
  if (Val.getOpcode() == ISD::BUILD_VECTOR)
    if (llvm::all_of(Val->ops(), [BitWidth](SDValue E) {
          if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(E))
            return C->getAPIntValue().zextOrTrunc(BitWidth).isPowerOf2();
          return false;
        }))
      return true;

  // More could be done here, though the above checks are enough
  // to handle some common cases.

  // Fall back to computeKnownBits to catch other known cases.
  KnownBits Known;
  computeKnownBits(Val, Known);
  return (Known.Zero.countPopulation() == BitWidth - 1) &&
         (Known.One.countPopulation() == 1);
}

unsigned SelectionDAG::ComputeNumSignBits(SDValue Op, unsigned Depth) const {
  EVT VT = Op.getValueType();
  APInt DemandedElts = VT.isVector()
                           ? APInt::getAllOnesValue(VT.getVectorNumElements())
                           : APInt(1, 1);
  return ComputeNumSignBits(Op, DemandedElts, Depth);
}

unsigned SelectionDAG::ComputeNumSignBits(SDValue Op, const APInt &DemandedElts,
                                          unsigned Depth) const {
  EVT VT = Op.getValueType();
  assert(VT.isInteger() && "Invalid VT!");
  unsigned VTBits = VT.getScalarSizeInBits();
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  if (Depth == 6)
    return 1;  // Limit search depth.

  if (!DemandedElts)
    return 1;  // No demanded elts, better to assume we don't know anything.

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

  case ISD::BUILD_VECTOR:
    Tmp = VTBits;
    for (unsigned i = 0, e = Op.getNumOperands(); (i < e) && (Tmp > 1); ++i) {
      if (!DemandedElts[i])
        continue;

      SDValue SrcOp = Op.getOperand(i);
      Tmp2 = ComputeNumSignBits(Op.getOperand(i), Depth + 1);

      // BUILD_VECTOR can implicitly truncate sources, we must handle this.
      if (SrcOp.getValueSizeInBits() != VTBits) {
        assert(SrcOp.getValueSizeInBits() > VTBits &&
               "Expected BUILD_VECTOR implicit truncation");
        unsigned ExtraBits = SrcOp.getValueSizeInBits() - VTBits;
        Tmp2 = (Tmp2 > ExtraBits ? Tmp2 - ExtraBits : 1);
      }
      Tmp = std::min(Tmp, Tmp2);
    }
    return Tmp;

  case ISD::SIGN_EXTEND:
  case ISD::SIGN_EXTEND_VECTOR_INREG:
    Tmp = VTBits - Op.getOperand(0).getScalarValueSizeInBits();
    return ComputeNumSignBits(Op.getOperand(0), Depth+1) + Tmp;

  case ISD::SIGN_EXTEND_INREG:
    // Max of the input and what this extends.
    Tmp = cast<VTSDNode>(Op.getOperand(1))->getVT().getScalarSizeInBits();
    Tmp = VTBits-Tmp+1;

    Tmp2 = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    return std::max(Tmp, Tmp2);

  case ISD::SRA:
    Tmp = ComputeNumSignBits(Op.getOperand(0), DemandedElts, Depth+1);
    // SRA X, C   -> adds C sign bits.
    if (ConstantSDNode *C = isConstOrConstSplat(Op.getOperand(1))) {
      APInt ShiftVal = C->getAPIntValue();
      ShiftVal += Tmp;
      Tmp = ShiftVal.uge(VTBits) ? VTBits : ShiftVal.getZExtValue();
    }
    return Tmp;
  case ISD::SHL:
    if (ConstantSDNode *C = isConstOrConstSplat(Op.getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (C->getAPIntValue().uge(VTBits) ||      // Bad shift.
          C->getAPIntValue().uge(Tmp)) break;    // Shifted all sign bits out.
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
      // computeKnownBits, and pick whichever answer is better.
    }
    break;

  case ISD::SELECT:
    Tmp = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(2), Depth+1);
    return std::min(Tmp, Tmp2);
  case ISD::SELECT_CC:
    Tmp = ComputeNumSignBits(Op.getOperand(2), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(3), Depth+1);
    return std::min(Tmp, Tmp2);
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    if (Tmp == 1)
      return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth + 1);
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
    // If setcc returns 0/-1, all bits are sign bits.
    // We know that we have an integer-based boolean since these operations
    // are only available for integer.
    if (TLI->getBooleanContents(Op.getValueType().isVector(), false) ==
        TargetLowering::ZeroOrNegativeOneBooleanContent)
      return VTBits;
    break;
  case ISD::SETCC:
    // If setcc returns 0/-1, all bits are sign bits.
    if (TLI->getBooleanContents(Op.getOperand(0).getValueType()) ==
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
  case ISD::ADDC:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.

    // Special case decrementing a value (ADD X, -1):
    if (ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if (CRHS->isAllOnesValue()) {
        KnownBits Known;
        computeKnownBits(Op.getOperand(0), Known, Depth+1);

        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((Known.Zero | 1).isAllOnesValue())
          return VTBits;

        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (Known.isNonNegative())
          return Tmp;
      }

    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;
    return std::min(Tmp, Tmp2)-1;

  case ISD::SUB:
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;

    // Handle NEG.
    if (ConstantSDNode *CLHS = isConstOrConstSplat(Op.getOperand(0)))
      if (CLHS->isNullValue()) {
        KnownBits Known;
        computeKnownBits(Op.getOperand(1), Known, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((Known.Zero | 1).isAllOnesValue())
          return VTBits;

        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (Known.isNonNegative())
          return Tmp2;

        // Otherwise, we treat this like a SUB.
      }

    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    return std::min(Tmp, Tmp2)-1;
  case ISD::TRUNCATE: {
    // Check if the sign bits of source go down as far as the truncated value.
    unsigned NumSrcBits = Op.getOperand(0).getScalarValueSizeInBits();
    unsigned NumSrcSignBits = ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    if (NumSrcSignBits > (NumSrcBits - VTBits))
      return NumSrcSignBits - (NumSrcBits - VTBits);
    break;
  }
  case ISD::EXTRACT_ELEMENT: {
    const int KnownSign = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    const int BitWidth = Op.getValueSizeInBits();
    const int Items = Op.getOperand(0).getValueSizeInBits() / BitWidth;

    // Get reverse index (starting from 1), Op1 value indexes elements from
    // little end. Sign starts at big end.
    const int rIndex = Items - 1 - Op.getConstantOperandVal(1);

    // If the sign portion ends in our element the subtraction gives correct
    // result. Otherwise it gives either negative or > bitwidth result
    return std::max(std::min(KnownSign - rIndex * BitWidth, BitWidth), 0);
  }
  case ISD::INSERT_VECTOR_ELT: {
    SDValue InVec = Op.getOperand(0);
    SDValue InVal = Op.getOperand(1);
    SDValue EltNo = Op.getOperand(2);
    unsigned NumElts = InVec.getValueType().getVectorNumElements();

    ConstantSDNode *CEltNo = dyn_cast<ConstantSDNode>(EltNo);
    if (CEltNo && CEltNo->getAPIntValue().ult(NumElts)) {
      // If we know the element index, split the demand between the
      // source vector and the inserted element.
      unsigned EltIdx = CEltNo->getZExtValue();

      // If we demand the inserted element then get its sign bits.
      Tmp = UINT_MAX;
      if (DemandedElts[EltIdx]) {
        // TODO - handle implicit truncation of inserted elements.
        if (InVal.getScalarValueSizeInBits() != VTBits)
          break;
        Tmp = ComputeNumSignBits(InVal, Depth + 1);
      }

      // If we demand the source vector then get its sign bits, and determine
      // the minimum.
      APInt VectorElts = DemandedElts;
      VectorElts.clearBit(EltIdx);
      if (!!VectorElts) {
        Tmp2 = ComputeNumSignBits(InVec, VectorElts, Depth + 1);
        Tmp = std::min(Tmp, Tmp2);
      }
    } else {
      // Unknown element index, so ignore DemandedElts and demand them all.
      Tmp = ComputeNumSignBits(InVec, Depth + 1);
      Tmp2 = ComputeNumSignBits(InVal, Depth + 1);
      Tmp = std::min(Tmp, Tmp2);
    }
    assert(Tmp <= VTBits && "Failed to determine minimum sign bits");
    return Tmp;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    SDValue InVec = Op.getOperand(0);
    SDValue EltNo = Op.getOperand(1);
    EVT VecVT = InVec.getValueType();
    const unsigned BitWidth = Op.getValueSizeInBits();
    const unsigned EltBitWidth = Op.getOperand(0).getScalarValueSizeInBits();
    const unsigned NumSrcElts = VecVT.getVectorNumElements();

    // If BitWidth > EltBitWidth the value is anyext:ed, and we do not know
    // anything about sign bits. But if the sizes match we can derive knowledge
    // about sign bits from the vector operand.
    if (BitWidth != EltBitWidth)
      break;

    // If we know the element index, just demand that vector element, else for
    // an unknown element index, ignore DemandedElts and demand them all.
    APInt DemandedSrcElts = APInt::getAllOnesValue(NumSrcElts);
    ConstantSDNode *ConstEltNo = dyn_cast<ConstantSDNode>(EltNo);
    if (ConstEltNo && ConstEltNo->getAPIntValue().ult(NumSrcElts))
      DemandedSrcElts =
          APInt::getOneBitSet(NumSrcElts, ConstEltNo->getZExtValue());

    return ComputeNumSignBits(InVec, DemandedSrcElts, Depth + 1);
  }
  case ISD::EXTRACT_SUBVECTOR:
    return ComputeNumSignBits(Op.getOperand(0), Depth + 1);
  case ISD::CONCAT_VECTORS:
    // Determine the minimum number of sign bits across all input vectors.
    // Early out if the result is already 1.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    for (unsigned i = 1, e = Op.getNumOperands(); (i < e) && (Tmp > 1); ++i)
      Tmp = std::min(Tmp, ComputeNumSignBits(Op.getOperand(i), Depth + 1));
    return Tmp;
  }

  // If we are looking at the loaded value of the SDNode.
  if (Op.getResNo() == 0) {
    // Handle LOADX separately here. EXTLOAD case will fallthrough.
    if (LoadSDNode *LD = dyn_cast<LoadSDNode>(Op)) {
      unsigned ExtType = LD->getExtensionType();
      switch (ExtType) {
        default: break;
        case ISD::SEXTLOAD:    // '17' bits known
          Tmp = LD->getMemoryVT().getScalarSizeInBits();
          return VTBits-Tmp+1;
        case ISD::ZEXTLOAD:    // '16' bits known
          Tmp = LD->getMemoryVT().getScalarSizeInBits();
          return VTBits-Tmp;
      }
    }
  }

  // Allow the target to implement this method for its nodes.
  if (Op.getOpcode() >= ISD::BUILTIN_OP_END ||
      Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_VOID) {
    unsigned NumBits =
        TLI->ComputeNumSignBitsForTargetNode(Op, DemandedElts, *this, Depth);
    if (NumBits > 1)
      FirstAnswer = std::max(FirstAnswer, NumBits);
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  KnownBits Known;
  computeKnownBits(Op, Known, DemandedElts, Depth);

  APInt Mask;
  if (Known.isNonNegative()) {        // sign bit is 0
    Mask = Known.Zero;
  } else if (Known.isNegative()) {  // sign bit is 1;
    Mask = Known.One;
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
  if (getTarget().Options.NoNaNsFPMath)
    return true;

  if (Op->getFlags().hasNoNaNs())
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

bool SelectionDAG::haveNoCommonBitsSet(SDValue A, SDValue B) const {
  assert(A.getValueType() == B.getValueType() &&
         "Values must have the same type");
  KnownBits AKnown, BKnown;
  computeKnownBits(A, AKnown);
  computeKnownBits(B, BKnown);
  return (AKnown.Zero | BKnown.Zero).isAllOnesValue();
}

static SDValue FoldCONCAT_VECTORS(const SDLoc &DL, EVT VT,
                                  ArrayRef<SDValue> Ops,
                                  llvm::SelectionDAG &DAG) {
  assert(!Ops.empty() && "Can't concatenate an empty list of vectors!");
  assert(llvm::all_of(Ops,
                      [Ops](SDValue Op) {
                        return Ops[0].getValueType() == Op.getValueType();
                      }) &&
         "Concatenation of vectors with inconsistent value types!");
  assert((Ops.size() * Ops[0].getValueType().getVectorNumElements()) ==
             VT.getVectorNumElements() &&
         "Incorrect element count in vector concatenation!");

  if (Ops.size() == 1)
    return Ops[0];

  // Concat of UNDEFs is UNDEF.
  if (llvm::all_of(Ops, [](SDValue Op) { return Op.isUndef(); }))
    return DAG.getUNDEF(VT);

  // A CONCAT_VECTOR with all UNDEF/BUILD_VECTOR operands can be
  // simplified to one big BUILD_VECTOR.
  // FIXME: Add support for SCALAR_TO_VECTOR as well.
  EVT SVT = VT.getScalarType();
  SmallVector<SDValue, 16> Elts;
  for (SDValue Op : Ops) {
    EVT OpVT = Op.getValueType();
    if (Op.isUndef())
      Elts.append(OpVT.getVectorNumElements(), DAG.getUNDEF(SVT));
    else if (Op.getOpcode() == ISD::BUILD_VECTOR)
      Elts.append(Op->op_begin(), Op->op_end());
    else
      return SDValue();
  }

  // BUILD_VECTOR requires all inputs to be of the same type, find the
  // maximum type and extend them all.
  for (SDValue Op : Elts)
    SVT = (SVT.bitsLT(Op.getValueType()) ? Op.getValueType() : SVT);

  if (SVT.bitsGT(VT.getScalarType()))
    for (SDValue &Op : Elts)
      Op = DAG.getTargetLoweringInfo().isZExtFree(Op.getValueType(), SVT)
               ? DAG.getZExtOrTrunc(Op, DL, SVT)
               : DAG.getSExtOrTrunc(Op, DL, SVT);

  return DAG.getBuildVector(VT, DL, Elts);
}

/// Gets or creates the specified node.
SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opcode, getVTList(VT), None);
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(),
                              getVTList(VT));
  CSEMap.InsertNode(N, IP);

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              SDValue Operand, const SDNodeFlags Flags) {
  // Constant fold unary operations with an integer constant operand. Even
  // opaque constant will be folded, because the folding of unary operations
  // doesn't create new constants with different values. Nevertheless, the
  // opaque flag is preserved during folding to prevent future folding with
  // other constants.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand)) {
    const APInt &Val = C->getAPIntValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND:
      return getConstant(Val.sextOrTrunc(VT.getSizeInBits()), DL, VT,
                         C->isTargetOpcode(), C->isOpaque());
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::TRUNCATE:
      return getConstant(Val.zextOrTrunc(VT.getSizeInBits()), DL, VT,
                         C->isTargetOpcode(), C->isOpaque());
    case ISD::UINT_TO_FP:
    case ISD::SINT_TO_FP: {
      APFloat apf(EVTToAPFloatSemantics(VT),
                  APInt::getNullValue(VT.getSizeInBits()));
      (void)apf.convertFromAPInt(Val,
                                 Opcode==ISD::SINT_TO_FP,
                                 APFloat::rmNearestTiesToEven);
      return getConstantFP(apf, DL, VT);
    }
    case ISD::BITCAST:
      if (VT == MVT::f16 && C->getValueType(0) == MVT::i16)
        return getConstantFP(APFloat(APFloat::IEEEhalf(), Val), DL, VT);
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(APFloat(APFloat::IEEEsingle(), Val), DL, VT);
      if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(APFloat(APFloat::IEEEdouble(), Val), DL, VT);
      if (VT == MVT::f128 && C->getValueType(0) == MVT::i128)
        return getConstantFP(APFloat(APFloat::IEEEquad(), Val), DL, VT);
      break;
    case ISD::ABS:
      return getConstant(Val.abs(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::BITREVERSE:
      return getConstant(Val.reverseBits(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::BSWAP:
      return getConstant(Val.byteSwap(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTPOP:
      return getConstant(Val.countPopulation(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTLZ:
    case ISD::CTLZ_ZERO_UNDEF:
      return getConstant(Val.countLeadingZeros(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::CTTZ:
    case ISD::CTTZ_ZERO_UNDEF:
      return getConstant(Val.countTrailingZeros(), DL, VT, C->isTargetOpcode(),
                         C->isOpaque());
    case ISD::FP16_TO_FP: {
      bool Ignored;
      APFloat FPV(APFloat::IEEEhalf(),
                  (Val.getBitWidth() == 16) ? Val : Val.trunc(16));

      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)FPV.convert(EVTToAPFloatSemantics(VT),
                        APFloat::rmNearestTiesToEven, &Ignored);
      return getConstantFP(FPV, DL, VT);
    }
    }
  }

  // Constant fold unary operations with a floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand)) {
    APFloat V = C->getValueAPF();    // make copy
    switch (Opcode) {
    case ISD::FNEG:
      V.changeSign();
      return getConstantFP(V, DL, VT);
    case ISD::FABS:
      V.clearSign();
      return getConstantFP(V, DL, VT);
    case ISD::FCEIL: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardPositive);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FTRUNC: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardZero);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FFLOOR: {
      APFloat::opStatus fs = V.roundToIntegral(APFloat::rmTowardNegative);
      if (fs == APFloat::opOK || fs == APFloat::opInexact)
        return getConstantFP(V, DL, VT);
      break;
    }
    case ISD::FP_EXTEND: {
      bool ignored;
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)V.convert(EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven, &ignored);
      return getConstantFP(V, DL, VT);
    }
    case ISD::FP_TO_SINT:
    case ISD::FP_TO_UINT: {
      bool ignored;
      APSInt IntVal(VT.getSizeInBits(), Opcode == ISD::FP_TO_UINT);
      // FIXME need to be more flexible about rounding mode.
      APFloat::opStatus s =
          V.convertToInteger(IntVal, APFloat::rmTowardZero, &ignored);
      if (s == APFloat::opInvalidOp) // inexact is OK, in fact usual
        break;
      return getConstant(IntVal, DL, VT);
    }
    case ISD::BITCAST:
      if (VT == MVT::i16 && C->getValueType(0) == MVT::f16)
        return getConstant((uint16_t)V.bitcastToAPInt().getZExtValue(), DL, VT);
      else if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
        return getConstant((uint32_t)V.bitcastToAPInt().getZExtValue(), DL, VT);
      else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
        return getConstant(V.bitcastToAPInt().getZExtValue(), DL, VT);
      break;
    case ISD::FP_TO_FP16: {
      bool Ignored;
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)V.convert(APFloat::IEEEhalf(),
                      APFloat::rmNearestTiesToEven, &Ignored);
      return getConstant(V.bitcastToAPInt(), DL, VT);
    }
    }
  }

  // Constant fold unary operations with a vector integer or float operand.
  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(Operand)) {
    if (BV->isConstant()) {
      switch (Opcode) {
      default:
        // FIXME: Entirely reasonable to perform folding of other unary
        // operations here as the need arises.
        break;
      case ISD::FNEG:
      case ISD::FABS:
      case ISD::FCEIL:
      case ISD::FTRUNC:
      case ISD::FFLOOR:
      case ISD::FP_EXTEND:
      case ISD::FP_TO_SINT:
      case ISD::FP_TO_UINT:
      case ISD::TRUNCATE:
      case ISD::UINT_TO_FP:
      case ISD::SINT_TO_FP:
      case ISD::ABS:
      case ISD::BITREVERSE:
      case ISD::BSWAP:
      case ISD::CTLZ:
      case ISD::CTLZ_ZERO_UNDEF:
      case ISD::CTTZ:
      case ISD::CTTZ_ZERO_UNDEF:
      case ISD::CTPOP: {
        SDValue Ops = { Operand };
        if (SDValue Fold = FoldConstantVectorArithmetic(Opcode, DL, VT, Ops))
          return Fold;
      }
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
    assert(Operand.getValueType().bitsLT(VT) &&
           "Invalid fpext node, dst < src!");
    if (Operand.isUndef())
      return getUNDEF(VT);
    break;
  case ISD::SIGN_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid SIGN_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    assert(Operand.getValueType().bitsLT(VT) &&
           "Invalid sext node, dst < src!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, DL, VT, Operand.getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // sext(undef) = 0, because the top bits will all be the same.
      return getConstant(0, DL, VT);
    break;
  case ISD::ZERO_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ZERO_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    assert(Operand.getValueType().bitsLT(VT) &&
           "Invalid zext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, DL, VT, Operand.getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      // zext(undef) = 0, because the top bits will be zero.
      return getConstant(0, DL, VT);
    break;
  case ISD::ANY_EXTEND:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid ANY_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    assert(Operand.getValueType().bitsLT(VT) &&
           "Invalid anyext node, dst < src!");

    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
        OpOpcode == ISD::ANY_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, DL, VT, Operand.getOperand(0));
    else if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);

    // (ext (trunx x)) -> x
    if (OpOpcode == ISD::TRUNCATE) {
      SDValue OpOp = Operand.getOperand(0);
      if (OpOp.getValueType() == VT)
        return OpOp;
    }
    break;
  case ISD::TRUNCATE:
    assert(VT.isInteger() && Operand.getValueType().isInteger() &&
           "Invalid TRUNCATE!");
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert((!VT.isVector() ||
            VT.getVectorNumElements() ==
            Operand.getValueType().getVectorNumElements()) &&
           "Vector element count mismatch!");
    assert(Operand.getValueType().bitsGT(VT) &&
           "Invalid truncate node, src < dst!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, DL, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
        OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.getOperand(0).getValueType().getScalarType()
            .bitsLT(VT.getScalarType()))
        return getNode(OpOpcode, DL, VT, Operand.getOperand(0));
      if (Operand.getOperand(0).getValueType().bitsGT(VT))
        return getNode(ISD::TRUNCATE, DL, VT, Operand.getOperand(0));
      return Operand.getOperand(0);
    }
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::ABS:
    assert(VT.isInteger() && VT == Operand.getValueType() &&
           "Invalid ABS!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::BSWAP:
    assert(VT.isInteger() && VT == Operand.getValueType() &&
           "Invalid BSWAP!");
    assert((VT.getScalarSizeInBits() % 16 == 0) &&
           "BSWAP types must be a multiple of 16 bits!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::BITREVERSE:
    assert(VT.isInteger() && VT == Operand.getValueType() &&
           "Invalid BITREVERSE!");
    if (OpOpcode == ISD::UNDEF)
      return getUNDEF(VT);
    break;
  case ISD::BITCAST:
    // Basic sanity checking.
    assert(VT.getSizeInBits() == Operand.getValueSizeInBits() &&
           "Cannot BITCAST between types of different sizes!");
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
    if (getTarget().Options.UnsafeFPMath && OpOpcode == ISD::FSUB)
      // FIXME: FNEG has no fast-math-flags to propagate; use the FSUB's flags?
      return getNode(ISD::FSUB, DL, VT, Operand.getOperand(1),
                     Operand.getOperand(0), Operand.getNode()->getFlags());
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, DL, VT, Operand.getOperand(0));
    break;
  }

  SDNode *N;
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = {Operand};
  if (VT != MVT::Glue) { // Don't CSE flag producing nodes
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP)) {
      E->intersectFlagsWith(Flags);
      return SDValue(E, 0);
    }

    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    N->setFlags(Flags);
    createOperands(N, Ops);
    CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

static std::pair<APInt, bool> FoldValue(unsigned Opcode, const APInt &C1,
                                        const APInt &C2) {
  switch (Opcode) {
  case ISD::ADD:  return std::make_pair(C1 + C2, true);
  case ISD::SUB:  return std::make_pair(C1 - C2, true);
  case ISD::MUL:  return std::make_pair(C1 * C2, true);
  case ISD::AND:  return std::make_pair(C1 & C2, true);
  case ISD::OR:   return std::make_pair(C1 | C2, true);
  case ISD::XOR:  return std::make_pair(C1 ^ C2, true);
  case ISD::SHL:  return std::make_pair(C1 << C2, true);
  case ISD::SRL:  return std::make_pair(C1.lshr(C2), true);
  case ISD::SRA:  return std::make_pair(C1.ashr(C2), true);
  case ISD::ROTL: return std::make_pair(C1.rotl(C2), true);
  case ISD::ROTR: return std::make_pair(C1.rotr(C2), true);
  case ISD::SMIN: return std::make_pair(C1.sle(C2) ? C1 : C2, true);
  case ISD::SMAX: return std::make_pair(C1.sge(C2) ? C1 : C2, true);
  case ISD::UMIN: return std::make_pair(C1.ule(C2) ? C1 : C2, true);
  case ISD::UMAX: return std::make_pair(C1.uge(C2) ? C1 : C2, true);
  case ISD::UDIV:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.udiv(C2), true);
  case ISD::UREM:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.urem(C2), true);
  case ISD::SDIV:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.sdiv(C2), true);
  case ISD::SREM:
    if (!C2.getBoolValue())
      break;
    return std::make_pair(C1.srem(C2), true);
  }
  return std::make_pair(APInt(1, 0), false);
}

SDValue SelectionDAG::FoldConstantArithmetic(unsigned Opcode, const SDLoc &DL,
                                             EVT VT, const ConstantSDNode *Cst1,
                                             const ConstantSDNode *Cst2) {
  if (Cst1->isOpaque() || Cst2->isOpaque())
    return SDValue();

  std::pair<APInt, bool> Folded = FoldValue(Opcode, Cst1->getAPIntValue(),
                                            Cst2->getAPIntValue());
  if (!Folded.second)
    return SDValue();
  return getConstant(Folded.first, DL, VT);
}

SDValue SelectionDAG::FoldSymbolOffset(unsigned Opcode, EVT VT,
                                       const GlobalAddressSDNode *GA,
                                       const SDNode *N2) {
  if (GA->getOpcode() != ISD::GlobalAddress)
    return SDValue();
  if (!TLI->isOffsetFoldingLegal(GA))
    return SDValue();
  const ConstantSDNode *Cst2 = dyn_cast<ConstantSDNode>(N2);
  if (!Cst2)
    return SDValue();
  int64_t Offset = Cst2->getSExtValue();
  switch (Opcode) {
  case ISD::ADD: break;
  case ISD::SUB: Offset = -uint64_t(Offset); break;
  default: return SDValue();
  }
  return getGlobalAddress(GA->getGlobal(), SDLoc(Cst2), VT,
                          GA->getOffset() + uint64_t(Offset));
}

bool SelectionDAG::isUndef(unsigned Opcode, ArrayRef<SDValue> Ops) {
  switch (Opcode) {
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM: {
    // If a divisor is zero/undef or any element of a divisor vector is
    // zero/undef, the whole op is undef.
    assert(Ops.size() == 2 && "Div/rem should have 2 operands");
    SDValue Divisor = Ops[1];
    if (Divisor.isUndef() || isNullConstant(Divisor))
      return true;

    return ISD::isBuildVectorOfConstantSDNodes(Divisor.getNode()) &&
           any_of(Divisor->op_values(),
                  [](SDValue V) { return V.isUndef() || isNullConstant(V); });
    // TODO: Handle signed overflow.
  }
  // TODO: Handle oversized shifts.
  default:
    return false;
  }
}

SDValue SelectionDAG::FoldConstantArithmetic(unsigned Opcode, const SDLoc &DL,
                                             EVT VT, SDNode *Cst1,
                                             SDNode *Cst2) {
  // If the opcode is a target-specific ISD node, there's nothing we can
  // do here and the operand rules may not line up with the below, so
  // bail early.
  if (Opcode >= ISD::BUILTIN_OP_END)
    return SDValue();

  if (isUndef(Opcode, {SDValue(Cst1, 0), SDValue(Cst2, 0)}))
    return getUNDEF(VT);

  // Handle the case of two scalars.
  if (const ConstantSDNode *Scalar1 = dyn_cast<ConstantSDNode>(Cst1)) {
    if (const ConstantSDNode *Scalar2 = dyn_cast<ConstantSDNode>(Cst2)) {
      SDValue Folded = FoldConstantArithmetic(Opcode, DL, VT, Scalar1, Scalar2);
      assert((!Folded || !VT.isVector()) &&
             "Can't fold vectors ops with scalar operands");
      return Folded;
    }
  }

  // fold (add Sym, c) -> Sym+c
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Cst1))
    return FoldSymbolOffset(Opcode, VT, GA, Cst2);
  if (isCommutativeBinOp(Opcode))
    if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Cst2))
      return FoldSymbolOffset(Opcode, VT, GA, Cst1);

  // For vectors extract each constant element into Inputs so we can constant
  // fold them individually.
  BuildVectorSDNode *BV1 = dyn_cast<BuildVectorSDNode>(Cst1);
  BuildVectorSDNode *BV2 = dyn_cast<BuildVectorSDNode>(Cst2);
  if (!BV1 || !BV2)
    return SDValue();

  assert(BV1->getNumOperands() == BV2->getNumOperands() && "Out of sync!");

  EVT SVT = VT.getScalarType();
  SmallVector<SDValue, 4> Outputs;
  for (unsigned I = 0, E = BV1->getNumOperands(); I != E; ++I) {
    SDValue V1 = BV1->getOperand(I);
    SDValue V2 = BV2->getOperand(I);

    // Avoid BUILD_VECTOR nodes that perform implicit truncation.
    // FIXME: This is valid and could be handled by truncation.
    if (V1->getValueType(0) != SVT || V2->getValueType(0) != SVT)
      return SDValue();

    // Fold one vector element.
    SDValue ScalarResult = getNode(Opcode, DL, SVT, V1, V2);

    // Scalar folding only succeeded if the result is a constant or UNDEF.
    if (!ScalarResult.isUndef() && ScalarResult.getOpcode() != ISD::Constant &&
        ScalarResult.getOpcode() != ISD::ConstantFP)
      return SDValue();
    Outputs.push_back(ScalarResult);
  }

  assert(VT.getVectorNumElements() == Outputs.size() &&
         "Vector size mismatch!");

  // We may have a vector type but a scalar result. Create a splat.
  Outputs.resize(VT.getVectorNumElements(), Outputs.back());

  // Build a big vector out of the scalar elements we generated.
  return getBuildVector(VT, SDLoc(), Outputs);
}

SDValue SelectionDAG::FoldConstantVectorArithmetic(unsigned Opcode,
                                                   const SDLoc &DL, EVT VT,
                                                   ArrayRef<SDValue> Ops,
                                                   const SDNodeFlags Flags) {
  // If the opcode is a target-specific ISD node, there's nothing we can
  // do here and the operand rules may not line up with the below, so
  // bail early.
  if (Opcode >= ISD::BUILTIN_OP_END)
    return SDValue();

  if (isUndef(Opcode, Ops))
    return getUNDEF(VT);

  // We can only fold vectors - maybe merge with FoldConstantArithmetic someday?
  if (!VT.isVector())
    return SDValue();

  unsigned NumElts = VT.getVectorNumElements();

  auto IsScalarOrSameVectorSize = [&](const SDValue &Op) {
    return !Op.getValueType().isVector() ||
           Op.getValueType().getVectorNumElements() == NumElts;
  };

  auto IsConstantBuildVectorOrUndef = [&](const SDValue &Op) {
    BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(Op);
    return (Op.isUndef()) || (Op.getOpcode() == ISD::CONDCODE) ||
           (BV && BV->isConstant());
  };

  // All operands must be vector types with the same number of elements as
  // the result type and must be either UNDEF or a build vector of constant
  // or UNDEF scalars.
  if (!all_of(Ops, IsConstantBuildVectorOrUndef) ||
      !all_of(Ops, IsScalarOrSameVectorSize))
    return SDValue();

  // If we are comparing vectors, then the result needs to be a i1 boolean
  // that is then sign-extended back to the legal result type.
  EVT SVT = (Opcode == ISD::SETCC ? MVT::i1 : VT.getScalarType());

  // Find legal integer scalar type for constant promotion and
  // ensure that its scalar size is at least as large as source.
  EVT LegalSVT = VT.getScalarType();
  if (NewNodesMustHaveLegalTypes && LegalSVT.isInteger()) {
    LegalSVT = TLI->getTypeToTransformTo(*getContext(), LegalSVT);
    if (LegalSVT.bitsLT(VT.getScalarType()))
      return SDValue();
  }

  // Constant fold each scalar lane separately.
  SmallVector<SDValue, 4> ScalarResults;
  for (unsigned i = 0; i != NumElts; i++) {
    SmallVector<SDValue, 4> ScalarOps;
    for (SDValue Op : Ops) {
      EVT InSVT = Op.getValueType().getScalarType();
      BuildVectorSDNode *InBV = dyn_cast<BuildVectorSDNode>(Op);
      if (!InBV) {
        // We've checked that this is UNDEF or a constant of some kind.
        if (Op.isUndef())
          ScalarOps.push_back(getUNDEF(InSVT));
        else
          ScalarOps.push_back(Op);
        continue;
      }

      SDValue ScalarOp = InBV->getOperand(i);
      EVT ScalarVT = ScalarOp.getValueType();

      // Build vector (integer) scalar operands may need implicit
      // truncation - do this before constant folding.
      if (ScalarVT.isInteger() && ScalarVT.bitsGT(InSVT))
        ScalarOp = getNode(ISD::TRUNCATE, DL, InSVT, ScalarOp);

      ScalarOps.push_back(ScalarOp);
    }

    // Constant fold the scalar operands.
    SDValue ScalarResult = getNode(Opcode, DL, SVT, ScalarOps, Flags);

    // Legalize the (integer) scalar constant if necessary.
    if (LegalSVT != SVT)
      ScalarResult = getNode(ISD::SIGN_EXTEND, DL, LegalSVT, ScalarResult);

    // Scalar folding only succeeded if the result is a constant or UNDEF.
    if (!ScalarResult.isUndef() && ScalarResult.getOpcode() != ISD::Constant &&
        ScalarResult.getOpcode() != ISD::ConstantFP)
      return SDValue();
    ScalarResults.push_back(ScalarResult);
  }

  return getBuildVector(VT, DL, ScalarResults);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              SDValue N1, SDValue N2, const SDNodeFlags Flags) {
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2);

  // Canonicalize constant to RHS if commutative.
  if (isCommutativeBinOp(Opcode)) {
    if (N1C && !N2C) {
      std::swap(N1C, N2C);
      std::swap(N1, N2);
    } else if (N1CFP && !N2CFP) {
      std::swap(N1CFP, N2CFP);
      std::swap(N1, N2);
    }
  }

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
  case ISD::CONCAT_VECTORS: {
    // Attempt to fold CONCAT_VECTORS into BUILD_VECTOR or UNDEF.
    SDValue Ops[] = {N1, N2};
    if (SDValue V = FoldCONCAT_VECTORS(DL, VT, Ops, *this))
      return V;
    break;
  }
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
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
    assert(VT.isInteger() && "This operator does not apply to FP types!");
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
    if (getTarget().Options.UnsafeFPMath) {
      if (Opcode == ISD::FADD) {
        // x+0 --> x
        if (N2CFP && N2CFP->getValueAPF().isZero())
          return N1;
      } else if (Opcode == ISD::FSUB) {
        // x-0 --> x
        if (N2CFP && N2CFP->getValueAPF().isZero())
          return N1;
      } else if (Opcode == ISD::FMUL) {
        // x*0 --> 0
        if (N2CFP && N2CFP->isZero())
          return N2;
        // x*1 --> x
        if (N2CFP && N2CFP->isExactlyValue(1.0))
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
    assert((!VT.isVector() || VT == N2.getValueType()) &&
           "Vector shift amounts must be in the same as their first arg");
    // Verify that the shift amount VT is bit enough to hold valid shift
    // amounts.  This catches things like trying to shift an i1024 value by an
    // i8, which is easy to fall into in generic code that uses
    // TLI.getShiftAmount().
    assert(N2.getValueSizeInBits() >= Log2_32_Ceil(N1.getValueSizeInBits()) &&
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
           N2C && (N2C->getZExtValue() == 0 || N2C->getZExtValue() == 1) &&
           "Invalid FP_ROUND!");
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

    auto SignExtendInReg = [&](APInt Val, llvm::EVT ConstantVT) {
      unsigned FromBits = EVT.getScalarSizeInBits();
      Val <<= Val.getBitWidth() - FromBits;
      Val.ashrInPlace(Val.getBitWidth() - FromBits);
      return getConstant(Val, DL, ConstantVT);
    };

    if (N1C) {
      const APInt &Val = N1C->getAPIntValue();
      return SignExtendInReg(Val, VT);
    }
    if (ISD::isBuildVectorOfConstantSDNodes(N1.getNode())) {
      SmallVector<SDValue, 8> Ops;
      llvm::EVT OpVT = N1.getOperand(0).getValueType();
      for (int i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
        SDValue Op = N1.getOperand(i);
        if (Op.isUndef()) {
          Ops.push_back(getUNDEF(OpVT));
          continue;
        }
        ConstantSDNode *C = cast<ConstantSDNode>(Op);
        APInt Val = C->getAPIntValue();
        Ops.push_back(SignExtendInReg(Val, OpVT));
      }
      return getBuildVector(VT, DL, Ops);
    }
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    // EXTRACT_VECTOR_ELT of an UNDEF is an UNDEF.
    if (N1.isUndef())
      return getUNDEF(VT);

    // EXTRACT_VECTOR_ELT of out-of-bounds element is an UNDEF
    if (N2C && N2C->getZExtValue() >= N1.getValueType().getVectorNumElements())
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
                     getConstant(N2C->getZExtValue() % Factor, DL,
                                 N2.getValueType()));
    }

    // EXTRACT_VECTOR_ELT of BUILD_VECTOR is often formed while lowering is
    // expanding large vector constants.
    if (N2C && N1.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue Elt = N1.getOperand(N2C->getZExtValue());

      if (VT != Elt.getValueType())
        // If the vector element type is not legal, the BUILD_VECTOR operands
        // are promoted and implicitly truncated, and the result implicitly
        // extended. Make that explicit here.
        Elt = getAnyExtOrTrunc(Elt, DL, VT);

      return Elt;
    }

    // EXTRACT_VECTOR_ELT of INSERT_VECTOR_ELT is often formed when vector
    // operations are lowered to scalars.
    if (N1.getOpcode() == ISD::INSERT_VECTOR_ELT) {
      // If the indices are the same, return the inserted element else
      // if the indices are known different, extract the element from
      // the original vector.
      SDValue N1Op2 = N1.getOperand(2);
      ConstantSDNode *N1Op2C = dyn_cast<ConstantSDNode>(N1Op2);

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
    if (N1C) {
      unsigned ElementSize = VT.getSizeInBits();
      unsigned Shift = ElementSize * N2C->getZExtValue();
      APInt ShiftedVal = N1C->getAPIntValue().lshr(Shift);
      return getConstant(ShiftedVal.trunc(ElementSize), DL, VT);
    }
    break;
  case ISD::EXTRACT_SUBVECTOR:
    if (VT.isSimple() && N1.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             "Extract subvector VTs must be a vectors!");
      assert(VT.getVectorElementType() ==
             N1.getValueType().getVectorElementType() &&
             "Extract subvector VTs must have the same element type!");
      assert(VT.getSimpleVT() <= N1.getSimpleValueType() &&
             "Extract subvector must be from larger vector to smaller vector!");

      if (N2C) {
        assert((VT.getVectorNumElements() + N2C->getZExtValue()
                <= N1.getValueType().getVectorNumElements())
               && "Extract subvector overflow!");
      }

      // Trivial extraction.
      if (VT.getSimpleVT() == N1.getSimpleValueType())
        return N1;

      // EXTRACT_SUBVECTOR of an UNDEF is an UNDEF.
      if (N1.isUndef())
        return getUNDEF(VT);

      // EXTRACT_SUBVECTOR of CONCAT_VECTOR can be simplified if the pieces of
      // the concat have the same type as the extract.
      if (N2C && N1.getOpcode() == ISD::CONCAT_VECTORS &&
          N1.getNumOperands() > 0 &&
          VT == N1.getOperand(0).getValueType()) {
        unsigned Factor = VT.getVectorNumElements();
        return N1.getOperand(N2C->getZExtValue() / Factor);
      }

      // EXTRACT_SUBVECTOR of INSERT_SUBVECTOR is often created
      // during shuffle legalization.
      if (N1.getOpcode() == ISD::INSERT_SUBVECTOR && N2 == N1.getOperand(2) &&
          VT == N1.getOperand(1).getValueType())
        return N1.getOperand(1);
    }
    break;
  }

  // Perform trivial constant folding.
  if (SDValue SV =
          FoldConstantArithmetic(Opcode, DL, VT, N1.getNode(), N2.getNode()))
    return SV;

  // Constant fold FP operations.
  bool HasFPExceptions = TLI->hasFloatingPointExceptions();
  if (N1CFP) {
    if (N2CFP) {
      APFloat V1 = N1CFP->getValueAPF(), V2 = N2CFP->getValueAPF();
      APFloat::opStatus s;
      switch (Opcode) {
      case ISD::FADD:
        s = V1.add(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s != APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FSUB:
        s = V1.subtract(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s!=APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FMUL:
        s = V1.multiply(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || s!=APFloat::opInvalidOp)
          return getConstantFP(V1, DL, VT);
        break;
      case ISD::FDIV:
        s = V1.divide(V2, APFloat::rmNearestTiesToEven);
        if (!HasFPExceptions || (s!=APFloat::opInvalidOp &&
                                 s!=APFloat::opDivByZero)) {
          return getConstantFP(V1, DL, VT);
        }
        break;
      case ISD::FREM :
        s = V1.mod(V2);
        if (!HasFPExceptions || (s!=APFloat::opInvalidOp &&
                                 s!=APFloat::opDivByZero)) {
          return getConstantFP(V1, DL, VT);
        }
        break;
      case ISD::FCOPYSIGN:
        V1.copySign(V2);
        return getConstantFP(V1, DL, VT);
      default: break;
      }
    }

    if (Opcode == ISD::FP_ROUND) {
      APFloat V = N1CFP->getValueAPF();    // make copy
      bool ignored;
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void)V.convert(EVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven, &ignored);
      return getConstantFP(V, DL, VT);
    }
  }

  // Canonicalize an UNDEF to the RHS, even over a constant.
  if (N1.isUndef()) {
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
          return getConstant(0, DL, VT);    // fold op(undef, arg2) -> 0
        // For vectors, we can't easily build an all zero vector, just return
        // the LHS.
        return N2;
      }
    }
  }

  // Fold a bunch of operators when the RHS is undef.
  if (N2.isUndef()) {
    switch (Opcode) {
    case ISD::XOR:
      if (N1.isUndef())
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return getConstant(0, DL, VT);
      LLVM_FALLTHROUGH;
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
      if (getTarget().Options.UnsafeFPMath)
        return N2;
      break;
    case ISD::MUL:
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      if (!VT.isVector())
        return getConstant(0, DL, VT);  // fold op(arg1, undef) -> 0
      // For vectors, we can't easily build an all zero vector, just return
      // the LHS.
      return N1;
    case ISD::OR:
      if (!VT.isVector())
        return getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), DL, VT);
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
  SDValue Ops[] = {N1, N2};
  if (VT != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP)) {
      E->intersectFlagsWith(Flags);
      return SDValue(E, 0);
    }

    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    N->setFlags(Flags);
    createOperands(N, Ops);
    CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3) {
  // Perform various simplifications.
  switch (Opcode) {
  case ISD::FMA: {
    ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
    ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2);
    ConstantFPSDNode *N3CFP = dyn_cast<ConstantFPSDNode>(N3);
    if (N1CFP && N2CFP && N3CFP) {
      APFloat  V1 = N1CFP->getValueAPF();
      const APFloat &V2 = N2CFP->getValueAPF();
      const APFloat &V3 = N3CFP->getValueAPF();
      APFloat::opStatus s =
        V1.fusedMultiplyAdd(V2, V3, APFloat::rmNearestTiesToEven);
      if (!TLI->hasFloatingPointExceptions() || s != APFloat::opInvalidOp)
        return getConstantFP(V1, DL, VT);
    }
    break;
  }
  case ISD::CONCAT_VECTORS: {
    // Attempt to fold CONCAT_VECTORS into BUILD_VECTOR or UNDEF.
    SDValue Ops[] = {N1, N2, N3};
    if (SDValue V = FoldCONCAT_VECTORS(DL, VT, Ops, *this))
      return V;
    break;
  }
  case ISD::SETCC: {
    // Use FoldSetCC to simplify SETCC's.
    if (SDValue V = FoldSetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get(), DL))
      return V;
    // Vector constant folding.
    SDValue Ops[] = {N1, N2, N3};
    if (SDValue V = FoldConstantVectorArithmetic(Opcode, DL, VT, Ops))
      return V;
    break;
  }
  case ISD::SELECT:
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1)) {
     if (N1C->getZExtValue())
       return N2;             // select true, X, Y -> X
     return N3;             // select false, X, Y -> Y
    }

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::VECTOR_SHUFFLE:
    llvm_unreachable("should use getVectorShuffle constructor!");
  case ISD::INSERT_VECTOR_ELT: {
    ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3);
    // INSERT_VECTOR_ELT into out-of-bounds element is an UNDEF
    if (N3C && N3C->getZExtValue() >= N1.getValueType().getVectorNumElements())
      return getUNDEF(VT);
    break;
  }
  case ISD::INSERT_SUBVECTOR: {
    SDValue Index = N3;
    if (VT.isSimple() && N1.getValueType().isSimple()
        && N2.getValueType().isSimple()) {
      assert(VT.isVector() && N1.getValueType().isVector() &&
             N2.getValueType().isVector() &&
             "Insert subvector VTs must be a vectors");
      assert(VT == N1.getValueType() &&
             "Dest and insert subvector source types must match!");
      assert(N2.getSimpleValueType() <= N1.getSimpleValueType() &&
             "Insert subvector must be from smaller vector to larger vector!");
      if (isa<ConstantSDNode>(Index)) {
        assert((N2.getValueType().getVectorNumElements() +
                cast<ConstantSDNode>(Index)->getZExtValue()
                <= VT.getVectorNumElements())
               && "Insert subvector overflow!");
      }

      // Trivial insertion.
      if (VT.getSimpleVT() == N2.getSimpleValueType())
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
  SDValue Ops[] = {N1, N2, N3};
  if (VT != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP))
      return SDValue(E, 0);

    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);
    CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3, SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VT, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                              SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VT, Ops);
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
  return getNode(ISD::TokenFactor, SDLoc(Chain), MVT::Other, ArgChains);
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDValue getMemsetValue(SDValue Value, EVT VT, SelectionDAG &DAG,
                              const SDLoc &dl) {
  assert(!Value.isUndef());

  unsigned NumBits = VT.getScalarSizeInBits();
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    assert(C->getAPIntValue().getBitWidth() == 8);
    APInt Val = APInt::getSplat(NumBits, C->getAPIntValue());
    if (VT.isInteger())
      return DAG.getConstant(Val, dl, VT);
    return DAG.getConstantFP(APFloat(DAG.EVTToAPFloatSemantics(VT), Val), dl,
                             VT);
  }

  assert(Value.getValueType() == MVT::i8 && "memset with non-byte fill value?");
  EVT IntVT = VT.getScalarType();
  if (!IntVT.isInteger())
    IntVT = EVT::getIntegerVT(*DAG.getContext(), IntVT.getSizeInBits());

  Value = DAG.getNode(ISD::ZERO_EXTEND, dl, IntVT, Value);
  if (NumBits > 8) {
    // Use a multiplication with 0x010101... to extend the input to the
    // required length.
    APInt Magic = APInt::getSplat(NumBits, APInt(8, 0x01));
    Value = DAG.getNode(ISD::MUL, dl, IntVT, Value,
                        DAG.getConstant(Magic, dl, IntVT));
  }

  if (VT != Value.getValueType() && !VT.isInteger())
    Value = DAG.getBitcast(VT.getScalarType(), Value);
  if (VT != Value.getValueType())
    Value = DAG.getSplatBuildVector(VT, dl, Value);

  return Value;
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDValue getMemsetStringVal(EVT VT, const SDLoc &dl, SelectionDAG &DAG,
                                  const TargetLowering &TLI, StringRef Str) {
  // Handle vector with all elements zero.
  if (Str.empty()) {
    if (VT.isInteger())
      return DAG.getConstant(0, dl, VT);
    else if (VT == MVT::f32 || VT == MVT::f64 || VT == MVT::f128)
      return DAG.getConstantFP(0.0, dl, VT);
    else if (VT.isVector()) {
      unsigned NumElts = VT.getVectorNumElements();
      MVT EltVT = (VT.getVectorElementType() == MVT::f32) ? MVT::i32 : MVT::i64;
      return DAG.getNode(ISD::BITCAST, dl, VT,
                         DAG.getConstant(0, dl,
                                         EVT::getVectorVT(*DAG.getContext(),
                                                          EltVT, NumElts)));
    } else
      llvm_unreachable("Expected type!");
  }

  assert(!VT.isVector() && "Can't handle vector type here!");
  unsigned NumVTBits = VT.getSizeInBits();
  unsigned NumVTBytes = NumVTBits / 8;
  unsigned NumBytes = std::min(NumVTBytes, unsigned(Str.size()));

  APInt Val(NumVTBits, 0);
  if (DAG.getDataLayout().isLittleEndian()) {
    for (unsigned i = 0; i != NumBytes; ++i)
      Val |= (uint64_t)(unsigned char)Str[i] << i*8;
  } else {
    for (unsigned i = 0; i != NumBytes; ++i)
      Val |= (uint64_t)(unsigned char)Str[i] << (NumVTBytes-i-1)*8;
  }

  // If the "cost" of materializing the integer immediate is less than the cost
  // of a load, then it is cost effective to turn the load into the immediate.
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());
  if (TLI.shouldConvertConstantLoadToIntImm(Val, Ty))
    return DAG.getConstant(Val, dl, VT);
  return SDValue(nullptr, 0);
}

SDValue SelectionDAG::getMemBasePlusOffset(SDValue Base, unsigned Offset,
                                           const SDLoc &DL) {
  EVT VT = Base.getValueType();
  return getNode(ISD::ADD, DL, VT, Base, getConstant(Offset, DL, VT));
}

/// isMemSrcFromString - Returns true if memcpy source is a string constant.
///
static bool isMemSrcFromString(SDValue Src, StringRef &Str) {
  uint64_t SrcDelta = 0;
  GlobalAddressSDNode *G = nullptr;
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

  return getConstantStringInfo(G->getGlobal(), Str,
                               SrcDelta + G->getOffset(), false);
}

/// Determines the optimal series of memory ops to replace the memset / memcpy.
/// Return true if the number of memory ops is below the threshold (Limit).
/// It returns the types of the sequence of memory ops to perform
/// memset / memcpy by reference.
static bool FindOptimalMemOpLowering(std::vector<EVT> &MemOps,
                                     unsigned Limit, uint64_t Size,
                                     unsigned DstAlign, unsigned SrcAlign,
                                     bool IsMemset,
                                     bool ZeroMemset,
                                     bool MemcpyStrSrc,
                                     bool AllowOverlap,
                                     unsigned DstAS, unsigned SrcAS,
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
                                   IsMemset, ZeroMemset, MemcpyStrSrc,
                                   DAG.getMachineFunction());

  if (VT == MVT::Other) {
    if (DstAlign >= DAG.getDataLayout().getPointerPrefAlignment(DstAS) ||
        TLI.allowsMisalignedMemoryAccesses(VT, DstAS, DstAlign)) {
      VT = TLI.getPointerTy(DAG.getDataLayout(), DstAS);
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
      EVT NewVT = VT;
      unsigned NewVTSize;

      bool Found = false;
      if (VT.isVector() || VT.isFloatingPoint()) {
        NewVT = (VT.getSizeInBits() > 64) ? MVT::i64 : MVT::i32;
        if (TLI.isOperationLegalOrCustom(ISD::STORE, NewVT) &&
            TLI.isSafeMemOpType(NewVT.getSimpleVT()))
          Found = true;
        else if (NewVT == MVT::i64 &&
                 TLI.isOperationLegalOrCustom(ISD::STORE, MVT::f64) &&
                 TLI.isSafeMemOpType(MVT::f64)) {
          // i64 is usually not legal on 32-bit targets, but f64 may be.
          NewVT = MVT::f64;
          Found = true;
        }
      }

      if (!Found) {
        do {
          NewVT = (MVT::SimpleValueType)(NewVT.getSimpleVT().SimpleTy - 1);
          if (NewVT == MVT::i8)
            break;
        } while (!TLI.isSafeMemOpType(NewVT.getSimpleVT()));
      }
      NewVTSize = NewVT.getSizeInBits() / 8;

      // If the new VT cannot cover all of the remaining bits, then consider
      // issuing a (or a pair of) unaligned and overlapping load / store.
      // FIXME: Only does this for 64-bit or more since we don't have proper
      // cost model for unaligned load / store.
      bool Fast;
      if (NumMemOps && AllowOverlap &&
          VTSize >= 8 && NewVTSize < Size &&
          TLI.allowsMisalignedMemoryAccesses(VT, DstAS, DstAlign, &Fast) && Fast)
        VTSize = Size;
      else {
        VT = NewVT;
        VTSize = NewVTSize;
      }
    }

    if (++NumMemOps > Limit)
      return false;

    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

static bool shouldLowerMemFuncForSize(const MachineFunction &MF) {
  // On Darwin, -Os means optimize for size without hurting performance, so
  // only really optimize for size when -Oz (MinSize) is used.
  if (MF.getTarget().getTargetTriple().isOSDarwin())
    return MF.getFunction()->optForMinSize();
  return MF.getFunction()->optForSize();
}

static SDValue getMemcpyLoadsAndStores(SelectionDAG &DAG, const SDLoc &dl,
                                       SDValue Chain, SDValue Dst, SDValue Src,
                                       uint64_t Size, unsigned Align,
                                       bool isVol, bool AlwaysInline,
                                       MachinePointerInfo DstPtrInfo,
                                       MachinePointerInfo SrcPtrInfo) {
  // Turn a memcpy of undef to nop.
  if (Src.isUndef())
    return Chain;

  // Expand memcpy to a series of load and store ops if the size operand falls
  // below a certain threshold.
  // TODO: In the AlwaysInline case, if the size is big then generate a loop
  // rather than maybe a humongous number of loads and stores.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI.isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  StringRef Str;
  bool CopyFromStr = isMemSrcFromString(Src, Str);
  bool isZeroStr = CopyFromStr && Str.empty();
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemcpy(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align),
                                (isZeroStr ? 0 : SrcAlign),
                                false, false, CopyFromStr, true,
                                DstPtrInfo.getAddrSpace(),
                                SrcPtrInfo.getAddrSpace(),
                                DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->needsStackRealignment(MF))
      while (NewAlign > Align &&
             DAG.getDataLayout().exceedsNaturalStackAlignment(NewAlign))
          NewAlign /= 2;

    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI.setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  MachineMemOperand::Flags MMOFlags =
      isVol ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;
  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  uint64_t SrcOff = 0, DstOff = 0;
  for (unsigned i = 0; i != NumMemOps; ++i) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value, Store;

    if (VTSize > Size) {
      // Issuing an unaligned load / store pair  that overlaps with the previous
      // pair. Adjust the offset accordingly.
      assert(i == NumMemOps-1 && i != 0);
      SrcOff -= VTSize - Size;
      DstOff -= VTSize - Size;
    }

    if (CopyFromStr &&
        (isZeroStr || (VT.isInteger() && !VT.isVector()))) {
      // It's unlikely a store of a vector immediate can be done in a single
      // instruction. It would require a load from a constantpool first.
      // We only handle zero vectors here.
      // FIXME: Handle other cases where store of vector immediate is done in
      // a single instruction.
      Value = getMemsetStringVal(VT, dl, DAG, TLI, Str.substr(SrcOff));
      if (Value.getNode())
        Store = DAG.getStore(Chain, dl, Value,
                             DAG.getMemBasePlusOffset(Dst, DstOff, dl),
                             DstPtrInfo.getWithOffset(DstOff), Align, MMOFlags);
    }

    if (!Store.getNode()) {
      // The type might not be legal for the target.  This should only happen
      // if the type is smaller than a legal type, as on PPC, so the right
      // thing to do is generate a LoadExt/StoreTrunc pair.  These simplify
      // to Load/Store if NVT==VT.
      // FIXME does the case above also need this?
      EVT NVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
      assert(NVT.bitsGE(VT));
      Value = DAG.getExtLoad(ISD::EXTLOAD, dl, NVT, Chain,
                             DAG.getMemBasePlusOffset(Src, SrcOff, dl),
                             SrcPtrInfo.getWithOffset(SrcOff), VT,
                             MinAlign(SrcAlign, SrcOff), MMOFlags);
      OutChains.push_back(Value.getValue(1));
      Store = DAG.getTruncStore(
          Chain, dl, Value, DAG.getMemBasePlusOffset(Dst, DstOff, dl),
          DstPtrInfo.getWithOffset(DstOff), VT, Align, MMOFlags);
    }
    OutChains.push_back(Store);
    SrcOff += VTSize;
    DstOff += VTSize;
    Size -= VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

static SDValue getMemmoveLoadsAndStores(SelectionDAG &DAG, const SDLoc &dl,
                                        SDValue Chain, SDValue Dst, SDValue Src,
                                        uint64_t Size, unsigned Align,
                                        bool isVol, bool AlwaysInline,
                                        MachinePointerInfo DstPtrInfo,
                                        MachinePointerInfo SrcPtrInfo) {
  // Turn a memmove of undef to nop.
  if (Src.isUndef())
    return Chain;

  // Expand memmove to a series of load and store ops if the size operand falls
  // below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI.isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  unsigned SrcAlign = DAG.InferPtrAlignment(Src);
  if (Align > SrcAlign)
    SrcAlign = Align;
  unsigned Limit = AlwaysInline ? ~0U : TLI.getMaxStoresPerMemmove(OptSize);

  if (!FindOptimalMemOpLowering(MemOps, Limit, Size,
                                (DstAlignCanChange ? 0 : Align), SrcAlign,
                                false, false, false, false,
                                DstPtrInfo.getAddrSpace(),
                                SrcPtrInfo.getAddrSpace(),
                                DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI.setObjectAlignment(FI->getIndex(), NewAlign);
      Align = NewAlign;
    }
  }

  MachineMemOperand::Flags MMOFlags =
      isVol ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;
  uint64_t SrcOff = 0, DstOff = 0;
  SmallVector<SDValue, 8> LoadValues;
  SmallVector<SDValue, 8> LoadChains;
  SmallVector<SDValue, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Value;

    Value =
        DAG.getLoad(VT, dl, Chain, DAG.getMemBasePlusOffset(Src, SrcOff, dl),
                    SrcPtrInfo.getWithOffset(SrcOff), SrcAlign, MMOFlags);
    LoadValues.push_back(Value);
    LoadChains.push_back(Value.getValue(1));
    SrcOff += VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, LoadChains);
  OutChains.clear();
  for (unsigned i = 0; i < NumMemOps; i++) {
    EVT VT = MemOps[i];
    unsigned VTSize = VT.getSizeInBits() / 8;
    SDValue Store;

    Store = DAG.getStore(Chain, dl, LoadValues[i],
                         DAG.getMemBasePlusOffset(Dst, DstOff, dl),
                         DstPtrInfo.getWithOffset(DstOff), Align, MMOFlags);
    OutChains.push_back(Store);
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

/// \brief Lower the call to 'memset' intrinsic function into a series of store
/// operations.
///
/// \param DAG Selection DAG where lowered code is placed.
/// \param dl Link to corresponding IR location.
/// \param Chain Control flow dependency.
/// \param Dst Pointer to destination memory location.
/// \param Src Value of byte to write into the memory.
/// \param Size Number of bytes to write.
/// \param Align Alignment of the destination in bytes.
/// \param isVol True if destination is volatile.
/// \param DstPtrInfo IR information on the memory pointer.
/// \returns New head in the control flow, if lowering was successful, empty
/// SDValue otherwise.
///
/// The function tries to replace 'llvm.memset' intrinsic with several store
/// operations and value calculation code. This is usually profitable for small
/// memory size.
static SDValue getMemsetStores(SelectionDAG &DAG, const SDLoc &dl,
                               SDValue Chain, SDValue Dst, SDValue Src,
                               uint64_t Size, unsigned Align, bool isVol,
                               MachinePointerInfo DstPtrInfo) {
  // Turn a memset of undef to nop.
  if (Src.isUndef())
    return Chain;

  // Expand memset to a series of load/store ops if the size operand
  // falls below a certain threshold.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  std::vector<EVT> MemOps;
  bool DstAlignCanChange = false;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Dst);
  if (FI && !MFI.isFixedObjectIndex(FI->getIndex()))
    DstAlignCanChange = true;
  bool IsZeroVal =
    isa<ConstantSDNode>(Src) && cast<ConstantSDNode>(Src)->isNullValue();
  if (!FindOptimalMemOpLowering(MemOps, TLI.getMaxStoresPerMemset(OptSize),
                                Size, (DstAlignCanChange ? 0 : Align), 0,
                                true, IsZeroVal, false, true,
                                DstPtrInfo.getAddrSpace(), ~0u,
                                DAG, TLI))
    return SDValue();

  if (DstAlignCanChange) {
    Type *Ty = MemOps[0].getTypeForEVT(*DAG.getContext());
    unsigned NewAlign = (unsigned)DAG.getDataLayout().getABITypeAlignment(Ty);
    if (NewAlign > Align) {
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI->getIndex()) < NewAlign)
        MFI.setObjectAlignment(FI->getIndex(), NewAlign);
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
    unsigned VTSize = VT.getSizeInBits() / 8;
    if (VTSize > Size) {
      // Issuing an unaligned load / store pair  that overlaps with the previous
      // pair. Adjust the offset accordingly.
      assert(i == NumMemOps-1 && i != 0);
      DstOff -= VTSize - Size;
    }

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
    SDValue Store = DAG.getStore(
        Chain, dl, Value, DAG.getMemBasePlusOffset(Dst, DstOff, dl),
        DstPtrInfo.getWithOffset(DstOff), Align,
        isVol ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone);
    OutChains.push_back(Store);
    DstOff += VT.getSizeInBits() / 8;
    Size -= VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
}

static void checkAddrSpaceIsValidForLibcall(const TargetLowering *TLI,
                                            unsigned AS) {
  // Lowering memcpy / memset / memmove intrinsics to calls is only valid if all
  // pointer operands can be losslessly bitcasted to pointers of address space 0
  if (AS != 0 && !TLI->isNoopAddrSpaceCast(AS, 0)) {
    report_fatal_error("cannot lower memory intrinsic in address space " +
                       Twine(AS));
  }
}

SDValue SelectionDAG::getMemcpy(SDValue Chain, const SDLoc &dl, SDValue Dst,
                                SDValue Src, SDValue Size, unsigned Align,
                                bool isVol, bool AlwaysInline, bool isTailCall,
                                MachinePointerInfo DstPtrInfo,
                                MachinePointerInfo SrcPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

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
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemcpy(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, AlwaysInline,
        DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  // If we really need inline code and the target declined to provide it,
  // use a (potentially long) sequence of loads and stores.
  if (AlwaysInline) {
    assert(ConstantSize && "AlwaysInline requires a constant size!");
    return getMemcpyLoadsAndStores(*this, dl, Chain, Dst, Src,
                                   ConstantSize->getZExtValue(), Align, isVol,
                                   true, DstPtrInfo, SrcPtrInfo);
  }

  checkAddrSpaceIsValidForLibcall(TLI, DstPtrInfo.getAddrSpace());
  checkAddrSpaceIsValidForLibcall(TLI, SrcPtrInfo.getAddrSpace());

  // FIXME: If the memcpy is volatile (isVol), lowering it to a plain libc
  // memcpy is not guaranteed to be safe. libc memcpys aren't required to
  // respect volatile, so they may do things like read or write memory
  // beyond the given memory regions. But fixing this isn't easy, and most
  // people don't care.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = getDataLayout().getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME: pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(TLI->getLibcallCallingConv(RTLIB::MEMCPY),
                    Dst.getValueType().getTypeForEVT(*getContext()),
                    getExternalSymbol(TLI->getLibcallName(RTLIB::MEMCPY),
                                      TLI->getPointerTy(getDataLayout())),
                    std::move(Args))
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getMemmove(SDValue Chain, const SDLoc &dl, SDValue Dst,
                                 SDValue Src, SDValue Size, unsigned Align,
                                 bool isVol, bool isTailCall,
                                 MachinePointerInfo DstPtrInfo,
                                 MachinePointerInfo SrcPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

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
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemmove(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, DstPtrInfo, SrcPtrInfo);
    if (Result.getNode())
      return Result;
  }

  checkAddrSpaceIsValidForLibcall(TLI, DstPtrInfo.getAddrSpace());
  checkAddrSpaceIsValidForLibcall(TLI, SrcPtrInfo.getAddrSpace());

  // FIXME: If the memmove is volatile, lowering it to plain libc memmove may
  // not be safe.  See memcpy above for more details.

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = getDataLayout().getIntPtrType(*getContext());
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  // FIXME:  pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(TLI->getLibcallCallingConv(RTLIB::MEMMOVE),
                    Dst.getValueType().getTypeForEVT(*getContext()),
                    getExternalSymbol(TLI->getLibcallName(RTLIB::MEMMOVE),
                                      TLI->getPointerTy(getDataLayout())),
                    std::move(Args))
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getMemset(SDValue Chain, const SDLoc &dl, SDValue Dst,
                                SDValue Src, SDValue Size, unsigned Align,
                                bool isVol, bool isTailCall,
                                MachinePointerInfo DstPtrInfo) {
  assert(Align && "The SDAG layer expects explicit alignment and reserves 0");

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
  if (TSI) {
    SDValue Result = TSI->EmitTargetCodeForMemset(
        *this, dl, Chain, Dst, Src, Size, Align, isVol, DstPtrInfo);
    if (Result.getNode())
      return Result;
  }

  checkAddrSpaceIsValidForLibcall(TLI, DstPtrInfo.getAddrSpace());

  // Emit a library call.
  Type *IntPtrTy = getDataLayout().getIntPtrType(*getContext());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Dst; Entry.Ty = IntPtrTy;
  Args.push_back(Entry);
  Entry.Node = Src;
  Entry.Ty = Src.getValueType().getTypeForEVT(*getContext());
  Args.push_back(Entry);
  Entry.Node = Size;
  Entry.Ty = IntPtrTy;
  Args.push_back(Entry);

  // FIXME: pass in SDLoc
  TargetLowering::CallLoweringInfo CLI(*this);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(TLI->getLibcallCallingConv(RTLIB::MEMSET),
                    Dst.getValueType().getTypeForEVT(*getContext()),
                    getExternalSymbol(TLI->getLibcallName(RTLIB::MEMSET),
                                      TLI->getPointerTy(getDataLayout())),
                    std::move(Args))
      .setDiscardResult()
      .setTailCall(isTailCall);

  std::pair<SDValue,SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, const SDLoc &dl, EVT MemVT,
                                SDVTList VTList, ArrayRef<SDValue> Ops,
                                MachineMemOperand *MMO) {
  FoldingSetNodeID ID;
  ID.AddInteger(MemVT.getRawBits());
  AddNodeIDNode(ID, Opcode, VTList, Ops);
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void* IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<AtomicSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }

  auto *N = newSDNode<AtomicSDNode>(Opcode, dl.getIROrder(), dl.getDebugLoc(),
                                    VTList, MemVT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getAtomicCmpSwap(
    unsigned Opcode, const SDLoc &dl, EVT MemVT, SDVTList VTs, SDValue Chain,
    SDValue Ptr, SDValue Cmp, SDValue Swp, MachinePointerInfo PtrInfo,
    unsigned Alignment, AtomicOrdering SuccessOrdering,
    AtomicOrdering FailureOrdering, SynchronizationScope SynchScope) {
  assert(Opcode == ISD::ATOMIC_CMP_SWAP ||
         Opcode == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");

  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();

  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  auto Flags = MachineMemOperand::MOVolatile | MachineMemOperand::MOLoad |
               MachineMemOperand::MOStore;
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, MemVT.getStoreSize(), Alignment,
                            AAMDNodes(), nullptr, SynchScope, SuccessOrdering,
                            FailureOrdering);

  return getAtomicCmpSwap(Opcode, dl, MemVT, VTs, Chain, Ptr, Cmp, Swp, MMO);
}

SDValue SelectionDAG::getAtomicCmpSwap(unsigned Opcode, const SDLoc &dl,
                                       EVT MemVT, SDVTList VTs, SDValue Chain,
                                       SDValue Ptr, SDValue Cmp, SDValue Swp,
                                       MachineMemOperand *MMO) {
  assert(Opcode == ISD::ATOMIC_CMP_SWAP ||
         Opcode == ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");

  SDValue Ops[] = {Chain, Ptr, Cmp, Swp};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, const SDLoc &dl, EVT MemVT,
                                SDValue Chain, SDValue Ptr, SDValue Val,
                                const Value *PtrVal, unsigned Alignment,
                                AtomicOrdering Ordering,
                                SynchronizationScope SynchScope) {
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  // An atomic store does not load. An atomic load does not store.
  // (An atomicrmw obviously both loads and stores.)
  // For now, atomics are considered to be volatile always, and they are
  // chained as such.
  // FIXME: Volatile isn't really correct; we should keep track of atomic
  // orderings in the memoperand.
  auto Flags = MachineMemOperand::MOVolatile;
  if (Opcode != ISD::ATOMIC_STORE)
    Flags |= MachineMemOperand::MOLoad;
  if (Opcode != ISD::ATOMIC_LOAD)
    Flags |= MachineMemOperand::MOStore;

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(PtrVal), Flags,
                            MemVT.getStoreSize(), Alignment, AAMDNodes(),
                            nullptr, SynchScope, Ordering);

  return getAtomic(Opcode, dl, MemVT, Chain, Ptr, Val, MMO);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, const SDLoc &dl, EVT MemVT,
                                SDValue Chain, SDValue Ptr, SDValue Val,
                                MachineMemOperand *MMO) {
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
  SDValue Ops[] = {Chain, Ptr, Val};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO);
}

SDValue SelectionDAG::getAtomic(unsigned Opcode, const SDLoc &dl, EVT MemVT,
                                EVT VT, SDValue Chain, SDValue Ptr,
                                MachineMemOperand *MMO) {
  assert(Opcode == ISD::ATOMIC_LOAD && "Invalid Atomic Op");

  SDVTList VTs = getVTList(VT, MVT::Other);
  SDValue Ops[] = {Chain, Ptr};
  return getAtomic(Opcode, dl, MemVT, VTs, Ops, MMO);
}

/// getMergeValues - Create a MERGE_VALUES node from the given operands.
SDValue SelectionDAG::getMergeValues(ArrayRef<SDValue> Ops, const SDLoc &dl) {
  if (Ops.size() == 1)
    return Ops[0];

  SmallVector<EVT, 4> VTs;
  VTs.reserve(Ops.size());
  for (unsigned i = 0; i < Ops.size(); ++i)
    VTs.push_back(Ops[i].getValueType());
  return getNode(ISD::MERGE_VALUES, dl, getVTList(VTs), Ops);
}

SDValue SelectionDAG::getMemIntrinsicNode(
    unsigned Opcode, const SDLoc &dl, SDVTList VTList, ArrayRef<SDValue> Ops,
    EVT MemVT, MachinePointerInfo PtrInfo, unsigned Align, bool Vol,
    bool ReadMem, bool WriteMem, unsigned Size) {
  if (Align == 0)  // Ensure that codegen never sees alignment 0
    Align = getEVTAlignment(MemVT);

  MachineFunction &MF = getMachineFunction();
  auto Flags = MachineMemOperand::MONone;
  if (WriteMem)
    Flags |= MachineMemOperand::MOStore;
  if (ReadMem)
    Flags |= MachineMemOperand::MOLoad;
  if (Vol)
    Flags |= MachineMemOperand::MOVolatile;
  if (!Size)
    Size = MemVT.getStoreSize();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PtrInfo, Flags, Size, Align);

  return getMemIntrinsicNode(Opcode, dl, VTList, Ops, MemVT, MMO);
}

SDValue SelectionDAG::getMemIntrinsicNode(unsigned Opcode, const SDLoc &dl,
                                          SDVTList VTList,
                                          ArrayRef<SDValue> Ops, EVT MemVT,
                                          MachineMemOperand *MMO) {
  assert((Opcode == ISD::INTRINSIC_VOID ||
          Opcode == ISD::INTRINSIC_W_CHAIN ||
          Opcode == ISD::PREFETCH ||
          Opcode == ISD::LIFETIME_START ||
          Opcode == ISD::LIFETIME_END ||
          (Opcode <= INT_MAX &&
           (int)Opcode >= ISD::FIRST_TARGET_MEMORY_OPCODE)) &&
         "Opcode is not a memory-accessing opcode!");

  // Memoize the node unless it returns a flag.
  MemIntrinsicSDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
      cast<MemIntrinsicSDNode>(E)->refineAlignment(MMO);
      return SDValue(E, 0);
    }

    N = newSDNode<MemIntrinsicSDNode>(Opcode, dl.getIROrder(), dl.getDebugLoc(),
                                      VTList, MemVT, MMO);
    createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<MemIntrinsicSDNode>(Opcode, dl.getIROrder(), dl.getDebugLoc(),
                                      VTList, MemVT, MMO);
    createOperands(N, Ops);
  }
  InsertNode(N);
  return SDValue(N, 0);
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SelectionDAG &DAG, SDValue Ptr,
                                           int64_t Offset = 0) {
  // If this is FI+Offset, we can model it.
  if (const FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Ptr))
    return MachinePointerInfo::getFixedStack(DAG.getMachineFunction(),
                                             FI->getIndex(), Offset);

  // If this is (FI+Offset1)+Offset2, we can model it.
  if (Ptr.getOpcode() != ISD::ADD ||
      !isa<ConstantSDNode>(Ptr.getOperand(1)) ||
      !isa<FrameIndexSDNode>(Ptr.getOperand(0)))
    return MachinePointerInfo();

  int FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
  return MachinePointerInfo::getFixedStack(
      DAG.getMachineFunction(), FI,
      Offset + cast<ConstantSDNode>(Ptr.getOperand(1))->getSExtValue());
}

/// InferPointerInfo - If the specified ptr/offset is a frame index, infer a
/// MachinePointerInfo record from it.  This is particularly useful because the
/// code generator has many cases where it doesn't bother passing in a
/// MachinePointerInfo to getLoad or getStore when it has "FI+Cst".
static MachinePointerInfo InferPointerInfo(SelectionDAG &DAG, SDValue Ptr,
                                           SDValue OffsetOp) {
  // If the 'Offset' value isn't a constant, we can't handle this.
  if (ConstantSDNode *OffsetNode = dyn_cast<ConstantSDNode>(OffsetOp))
    return InferPointerInfo(DAG, Ptr, OffsetNode->getSExtValue());
  if (OffsetOp.isUndef())
    return InferPointerInfo(DAG, Ptr);
  return MachinePointerInfo();
}

SDValue SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                              EVT VT, const SDLoc &dl, SDValue Chain,
                              SDValue Ptr, SDValue Offset,
                              MachinePointerInfo PtrInfo, EVT MemVT,
                              unsigned Alignment,
                              MachineMemOperand::Flags MMOFlags,
                              const AAMDNodes &AAInfo, const MDNode *Ranges) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(MemVT);

  MMOFlags |= MachineMemOperand::MOLoad;
  assert((MMOFlags & MachineMemOperand::MOStore) == 0);
  // If we don't have a PtrInfo, infer the trivial frame index case to simplify
  // clients.
  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(*this, Ptr, Offset);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MMOFlags, MemVT.getStoreSize(), Alignment, AAInfo, Ranges);
  return getLoad(AM, ExtType, VT, dl, Chain, Ptr, Offset, MemVT, MMO);
}

SDValue SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                              EVT VT, const SDLoc &dl, SDValue Chain,
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
           "Cannot use an ext load to convert to or from a vector!");
    assert((!VT.isVector() ||
            VT.getVectorNumElements() == MemVT.getVectorNumElements()) &&
           "Cannot use an ext load to change the number of vector elements!");
  }

  bool Indexed = AM != ISD::UNINDEXED;
  assert((Indexed || Offset.isUndef()) && "Unindexed load with an offset!");

  SDVTList VTs = Indexed ?
    getVTList(VT, Ptr.getValueType(), MVT::Other) : getVTList(VT, MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops);
  ID.AddInteger(MemVT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<LoadSDNode>(
      dl.getIROrder(), VTs, AM, ExtType, MemVT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<LoadSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<LoadSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs, AM,
                                  ExtType, MemVT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getLoad(EVT VT, const SDLoc &dl, SDValue Chain,
                              SDValue Ptr, MachinePointerInfo PtrInfo,
                              unsigned Alignment,
                              MachineMemOperand::Flags MMOFlags,
                              const AAMDNodes &AAInfo, const MDNode *Ranges) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, dl, Chain, Ptr, Undef,
                 PtrInfo, VT, Alignment, MMOFlags, AAInfo, Ranges);
}

SDValue SelectionDAG::getLoad(EVT VT, const SDLoc &dl, SDValue Chain,
                              SDValue Ptr, MachineMemOperand *MMO) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, dl, Chain, Ptr, Undef,
                 VT, MMO);
}

SDValue SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, const SDLoc &dl,
                                 EVT VT, SDValue Chain, SDValue Ptr,
                                 MachinePointerInfo PtrInfo, EVT MemVT,
                                 unsigned Alignment,
                                 MachineMemOperand::Flags MMOFlags,
                                 const AAMDNodes &AAInfo) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, dl, Chain, Ptr, Undef, PtrInfo,
                 MemVT, Alignment, MMOFlags, AAInfo);
}

SDValue SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, const SDLoc &dl,
                                 EVT VT, SDValue Chain, SDValue Ptr, EVT MemVT,
                                 MachineMemOperand *MMO) {
  SDValue Undef = getUNDEF(Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, dl, Chain, Ptr, Undef,
                 MemVT, MMO);
}

SDValue SelectionDAG::getIndexedLoad(SDValue OrigLoad, const SDLoc &dl,
                                     SDValue Base, SDValue Offset,
                                     ISD::MemIndexedMode AM) {
  LoadSDNode *LD = cast<LoadSDNode>(OrigLoad);
  assert(LD->getOffset().isUndef() && "Load is already a indexed load!");
  // Don't propagate the invariant or dereferenceable flags.
  auto MMOFlags =
      LD->getMemOperand()->getFlags() &
      ~(MachineMemOperand::MOInvariant | MachineMemOperand::MODereferenceable);
  return getLoad(AM, LD->getExtensionType(), OrigLoad.getValueType(), dl,
                 LD->getChain(), Base, Offset, LD->getPointerInfo(),
                 LD->getMemoryVT(), LD->getAlignment(), MMOFlags,
                 LD->getAAInfo());
}

SDValue SelectionDAG::getStore(SDValue Chain, const SDLoc &dl, SDValue Val,
                               SDValue Ptr, MachinePointerInfo PtrInfo,
                               unsigned Alignment,
                               MachineMemOperand::Flags MMOFlags,
                               const AAMDNodes &AAInfo) {
  assert(Chain.getValueType() == MVT::Other && "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(Val.getValueType());

  MMOFlags |= MachineMemOperand::MOStore;
  assert((MMOFlags & MachineMemOperand::MOLoad) == 0);

  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(*this, Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MMOFlags, Val.getValueType().getStoreSize(), Alignment, AAInfo);
  return getStore(Chain, dl, Val, Ptr, MMO);
}

SDValue SelectionDAG::getStore(SDValue Chain, const SDLoc &dl, SDValue Val,
                               SDValue Ptr, MachineMemOperand *MMO) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  EVT VT = Val.getValueType();
  SDVTList VTs = getVTList(MVT::Other);
  SDValue Undef = getUNDEF(Ptr.getValueType());
  SDValue Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<StoreSDNode>(
      dl.getIROrder(), VTs, ISD::UNINDEXED, false, VT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<StoreSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs,
                                   ISD::UNINDEXED, false, VT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, const SDLoc &dl, SDValue Val,
                                    SDValue Ptr, MachinePointerInfo PtrInfo,
                                    EVT SVT, unsigned Alignment,
                                    MachineMemOperand::Flags MMOFlags,
                                    const AAMDNodes &AAInfo) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  if (Alignment == 0)  // Ensure that codegen never sees alignment 0
    Alignment = getEVTAlignment(SVT);

  MMOFlags |= MachineMemOperand::MOStore;
  assert((MMOFlags & MachineMemOperand::MOLoad) == 0);

  if (PtrInfo.V.isNull())
    PtrInfo = InferPointerInfo(*this, Ptr);

  MachineFunction &MF = getMachineFunction();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MMOFlags, SVT.getStoreSize(), Alignment, AAInfo);
  return getTruncStore(Chain, dl, Val, Ptr, SVT, MMO);
}

SDValue SelectionDAG::getTruncStore(SDValue Chain, const SDLoc &dl, SDValue Val,
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
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(SVT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<StoreSDNode>(
      dl.getIROrder(), VTs, ISD::UNINDEXED, true, SVT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<StoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<StoreSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs,
                                   ISD::UNINDEXED, true, SVT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getIndexedStore(SDValue OrigStore, const SDLoc &dl,
                                      SDValue Base, SDValue Offset,
                                      ISD::MemIndexedMode AM) {
  StoreSDNode *ST = cast<StoreSDNode>(OrigStore);
  assert(ST->getOffset().isUndef() && "Store is already a indexed store!");
  SDVTList VTs = getVTList(Base.getValueType(), MVT::Other);
  SDValue Ops[] = { ST->getChain(), ST->getValue(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops);
  ID.AddInteger(ST->getMemoryVT().getRawBits());
  ID.AddInteger(ST->getRawSubclassData());
  ID.AddInteger(ST->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP))
    return SDValue(E, 0);

  auto *N = newSDNode<StoreSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs, AM,
                                   ST->isTruncatingStore(), ST->getMemoryVT(),
                                   ST->getMemOperand());
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedLoad(EVT VT, const SDLoc &dl, SDValue Chain,
                                    SDValue Ptr, SDValue Mask, SDValue Src0,
                                    EVT MemVT, MachineMemOperand *MMO,
                                    ISD::LoadExtType ExtTy, bool isExpanding) {

  SDVTList VTs = getVTList(VT, MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Mask, Src0 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MLOAD, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<MaskedLoadSDNode>(
      dl.getIROrder(), VTs, ExtTy, isExpanding, MemVT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<MaskedLoadSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<MaskedLoadSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs,
                                        ExtTy, isExpanding, MemVT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedStore(SDValue Chain, const SDLoc &dl,
                                     SDValue Val, SDValue Ptr, SDValue Mask,
                                     EVT MemVT, MachineMemOperand *MMO,
                                     bool IsTruncating, bool IsCompressing) {
  assert(Chain.getValueType() == MVT::Other &&
        "Invalid chain type");
  EVT VT = Val.getValueType();
  SDVTList VTs = getVTList(MVT::Other);
  SDValue Ops[] = { Chain, Ptr, Mask, Val };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MSTORE, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<MaskedStoreSDNode>(
      dl.getIROrder(), VTs, IsTruncating, IsCompressing, MemVT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<MaskedStoreSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<MaskedStoreSDNode>(dl.getIROrder(), dl.getDebugLoc(), VTs,
                                         IsTruncating, IsCompressing, MemVT, MMO);
  createOperands(N, Ops);

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedGather(SDVTList VTs, EVT VT, const SDLoc &dl,
                                      ArrayRef<SDValue> Ops,
                                      MachineMemOperand *MMO) {
  assert(Ops.size() == 5 && "Incompatible number of operands");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MGATHER, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<MaskedGatherSDNode>(
      dl.getIROrder(), VTs, VT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<MaskedGatherSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }

  auto *N = newSDNode<MaskedGatherSDNode>(dl.getIROrder(), dl.getDebugLoc(),
                                          VTs, VT, MMO);
  createOperands(N, Ops);

  assert(N->getValue().getValueType() == N->getValueType(0) &&
         "Incompatible type of the PassThru value in MaskedGatherSDNode");
  assert(N->getMask().getValueType().getVectorNumElements() ==
             N->getValueType(0).getVectorNumElements() &&
         "Vector width mismatch between mask and data");
  assert(N->getIndex().getValueType().getVectorNumElements() ==
             N->getValueType(0).getVectorNumElements() &&
         "Vector width mismatch between index and data");

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getMaskedScatter(SDVTList VTs, EVT VT, const SDLoc &dl,
                                       ArrayRef<SDValue> Ops,
                                       MachineMemOperand *MMO) {
  assert(Ops.size() == 5 && "Incompatible number of operands");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MSCATTER, VTs, Ops);
  ID.AddInteger(VT.getRawBits());
  ID.AddInteger(getSyntheticNodeSubclassData<MaskedScatterSDNode>(
      dl.getIROrder(), VTs, VT, MMO));
  ID.AddInteger(MMO->getPointerInfo().getAddrSpace());
  void *IP = nullptr;
  if (SDNode *E = FindNodeOrInsertPos(ID, dl, IP)) {
    cast<MaskedScatterSDNode>(E)->refineAlignment(MMO);
    return SDValue(E, 0);
  }
  auto *N = newSDNode<MaskedScatterSDNode>(dl.getIROrder(), dl.getDebugLoc(),
                                           VTs, VT, MMO);
  createOperands(N, Ops);

  assert(N->getMask().getValueType().getVectorNumElements() ==
             N->getValue().getValueType().getVectorNumElements() &&
         "Vector width mismatch between mask and data");
  assert(N->getIndex().getValueType().getVectorNumElements() ==
             N->getValue().getValueType().getVectorNumElements() &&
         "Vector width mismatch between index and data");

  CSEMap.InsertNode(N, IP);
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getVAArg(EVT VT, const SDLoc &dl, SDValue Chain,
                               SDValue Ptr, SDValue SV, unsigned Align) {
  SDValue Ops[] = { Chain, Ptr, SV, getTargetConstant(Align, dl, MVT::i32) };
  return getNode(ISD::VAARG, dl, getVTList(VT, MVT::Other), Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              ArrayRef<SDUse> Ops) {
  switch (Ops.size()) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, static_cast<const SDValue>(Ops[0]));
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  // Copy from an SDUse array into an SDValue array for use with
  // the regular getNode logic.
  SmallVector<SDValue, 8> NewOps(Ops.begin(), Ops.end());
  return getNode(Opcode, DL, VT, NewOps);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, EVT VT,
                              ArrayRef<SDValue> Ops, const SDNodeFlags Flags) {
  unsigned NumOps = Ops.size();
  switch (NumOps) {
  case 0: return getNode(Opcode, DL, VT);
  case 1: return getNode(Opcode, DL, VT, Ops[0], Flags);
  case 2: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Flags);
  case 3: return getNode(Opcode, DL, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }

  switch (Opcode) {
  default: break;
  case ISD::CONCAT_VECTORS: {
    // Attempt to fold CONCAT_VECTORS into BUILD_VECTOR or UNDEF.
    if (SDValue V = FoldCONCAT_VECTORS(DL, VT, Ops, *this))
      return V;
    break;
  }
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
    AddNodeIDNode(ID, Opcode, VTs, Ops);
    void *IP = nullptr;

    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP))
      return SDValue(E, 0);

    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);

    CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
    createOperands(N, Ops);
  }

  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL,
                              ArrayRef<EVT> ResultTys, ArrayRef<SDValue> Ops) {
  return getNode(Opcode, DL, getVTList(ResultTys), Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              ArrayRef<SDValue> Ops) {
  if (VTList.NumVTs == 1)
    return getNode(Opcode, DL, VTList.VTs[0], Ops);

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
        unsigned NumBits = VT.getScalarSizeInBits()*2;
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
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP))
      return SDValue(E, 0);

    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTList);
    createOperands(N, Ops);
    CSEMap.InsertNode(N, IP);
  } else {
    N = newSDNode<SDNode>(Opcode, DL.getIROrder(), DL.getDebugLoc(), VTList);
    createOperands(N, Ops);
  }
  InsertNode(N);
  return SDValue(N, 0);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL,
                              SDVTList VTList) {
  return getNode(Opcode, DL, VTList, None);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              SDValue N1) {
  SDValue Ops[] = { N1 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              SDValue N1, SDValue N2) {
  SDValue Ops[] = { N1, N2 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3) {
  SDValue Ops[] = { N1, N2, N3 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3, SDValue N4) {
  SDValue Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDValue SelectionDAG::getNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                              SDValue N5) {
  SDValue Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, DL, VTList, Ops);
}

SDVTList SelectionDAG::getVTList(EVT VT) {
  return makeVTList(SDNode::getValueTypeList(VT), 1);
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2) {
  FoldingSetNodeID ID;
  ID.AddInteger(2U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(2);
    Array[0] = VT1;
    Array[1] = VT2;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 2);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3) {
  FoldingSetNodeID ID;
  ID.AddInteger(3U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());
  ID.AddInteger(VT3.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(3);
    Array[0] = VT1;
    Array[1] = VT2;
    Array[2] = VT3;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 3);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(EVT VT1, EVT VT2, EVT VT3, EVT VT4) {
  FoldingSetNodeID ID;
  ID.AddInteger(4U);
  ID.AddInteger(VT1.getRawBits());
  ID.AddInteger(VT2.getRawBits());
  ID.AddInteger(VT3.getRawBits());
  ID.AddInteger(VT4.getRawBits());

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(4);
    Array[0] = VT1;
    Array[1] = VT2;
    Array[2] = VT3;
    Array[3] = VT4;
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, 4);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
}

SDVTList SelectionDAG::getVTList(ArrayRef<EVT> VTs) {
  unsigned NumVTs = VTs.size();
  FoldingSetNodeID ID;
  ID.AddInteger(NumVTs);
  for (unsigned index = 0; index < NumVTs; index++) {
    ID.AddInteger(VTs[index].getRawBits());
  }

  void *IP = nullptr;
  SDVTListNode *Result = VTListMap.FindNodeOrInsertPos(ID, IP);
  if (!Result) {
    EVT *Array = Allocator.Allocate<EVT>(NumVTs);
    std::copy(VTs.begin(), VTs.end(), Array);
    Result = new (Allocator) SDVTListNode(ID.Intern(Allocator), Array, NumVTs);
    VTListMap.InsertNode(Result, IP);
  }
  return Result->getSDVTList();
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
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

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
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op1, Op2, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

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
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4 };
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                   SDValue Op3, SDValue Op4, SDValue Op5) {
  SDValue Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return UpdateNodeOperands(N, Ops);
}

SDNode *SelectionDAG::
UpdateNodeOperands(SDNode *N, ArrayRef<SDValue> Ops) {
  unsigned NumOps = Ops.size();
  assert(N->getNumOperands() == NumOps &&
         "Update with wrong number of operands");

  // If no operands changed just return the input node.
  if (std::equal(Ops.begin(), Ops.end(), N->op_begin()))
    return N;

  // See if the modified node already exists.
  void *InsertPos = nullptr;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Ops, InsertPos))
    return Existing;

  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    if (!RemoveNodeFromCSEMaps(N))
      InsertPos = nullptr;

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
  return SelectNodeTo(N, MachineOpc, VTs, None);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, SDValue Op1,
                                   SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT, ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2) {
  SDVTList VTs = getVTList(VT1, VT2);
  return SelectNodeTo(N, MachineOpc, VTs, None);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2, EVT VT3,
                                   ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   EVT VT1, EVT VT2,
                                   SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return SelectNodeTo(N, MachineOpc, VTs, Ops);
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned MachineOpc,
                                   SDVTList VTs,ArrayRef<SDValue> Ops) {
  SDNode *New = MorphNodeTo(N, ~MachineOpc, VTs, Ops);
  // Reset the NodeID to -1.
  New->setNodeId(-1);
  if (New != N) {
    ReplaceAllUsesWith(N, New);
    RemoveDeadNode(N);
  }
  return New;
}

/// UpdateSDLocOnMergeSDNode - If the opt level is -O0 then it throws away
/// the line number information on the merged node since it is not possible to
/// preserve the information that operation is associated with multiple lines.
/// This will make the debugger working better at -O0, were there is a higher
/// probability having other instructions associated with that line.
///
/// For IROrder, we keep the smaller of the two
SDNode *SelectionDAG::UpdateSDLocOnMergeSDNode(SDNode *N, const SDLoc &OLoc) {
  DebugLoc NLoc = N->getDebugLoc();
  if (NLoc && OptLevel == CodeGenOpt::None && OLoc.getDebugLoc() != NLoc) {
    N->setDebugLoc(DebugLoc());
  }
  unsigned Order = std::min(N->getIROrder(), OLoc.getIROrder());
  N->setIROrder(Order);
  return N;
}

/// MorphNodeTo - This *mutates* the specified node to have the specified
/// return type, opcode, and operands.
///
/// Note that MorphNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.  Note that the SDLoc need not be the same.
///
/// Using MorphNodeTo is faster than creating a new node and swapping it in
/// with ReplaceAllUsesWith both because it often avoids allocating a new
/// node, and because it doesn't require CSE recalculation for any of
/// the node's users.
///
/// However, note that MorphNodeTo recursively deletes dead nodes from the DAG.
/// As a consequence it isn't appropriate to use from within the DAG combiner or
/// the legalizer which maintain worklists that would need to be updated when
/// deleting things.
SDNode *SelectionDAG::MorphNodeTo(SDNode *N, unsigned Opc,
                                  SDVTList VTs, ArrayRef<SDValue> Ops) {
  // If an identical node already exists, use it.
  void *IP = nullptr;
  if (VTs.VTs[VTs.NumVTs-1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opc, VTs, Ops);
    if (SDNode *ON = FindNodeOrInsertPos(ID, SDLoc(N), IP))
      return UpdateSDLocOnMergeSDNode(ON, SDLoc(N));
  }

  if (!RemoveNodeFromCSEMaps(N))
    IP = nullptr;

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

  // For MachineNode, initialize the memory references information.
  if (MachineSDNode *MN = dyn_cast<MachineSDNode>(N))
    MN->setMemRefs(nullptr, nullptr);

  // Swap for an appropriately sized array from the recycler.
  removeOperands(N);
  createOperands(N, Ops);

  // Delete any nodes that are still dead after adding the uses for the
  // new operands.
  if (!DeadNodeSet.empty()) {
    SmallVector<SDNode *, 16> DeadNodes;
    for (SDNode *N : DeadNodeSet)
      if (N->use_empty())
        DeadNodes.push_back(N);
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
MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, None);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT, SDValue Op1) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT, SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT, SDValue Op1, SDValue Op2,
                                            SDValue Op3) {
  SDVTList VTs = getVTList(VT);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT, ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2, SDValue Op1,
                                            SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2, SDValue Op1,
                                            SDValue Op2, SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2,
                                            ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2, EVT VT3,
                                            SDValue Op1, SDValue Op2) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2, EVT VT3,
                                            SDValue Op1, SDValue Op2,
                                            SDValue Op3) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  SDValue Ops[] = { Op1, Op2, Op3 };
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            EVT VT1, EVT VT2, EVT VT3,
                                            ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(VT1, VT2, VT3);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &dl,
                                            ArrayRef<EVT> ResultTys,
                                            ArrayRef<SDValue> Ops) {
  SDVTList VTs = getVTList(ResultTys);
  return getMachineNode(Opcode, dl, VTs, Ops);
}

MachineSDNode *SelectionDAG::getMachineNode(unsigned Opcode, const SDLoc &DL,
                                            SDVTList VTs,
                                            ArrayRef<SDValue> Ops) {
  bool DoCSE = VTs.VTs[VTs.NumVTs-1] != MVT::Glue;
  MachineSDNode *N;
  void *IP = nullptr;

  if (DoCSE) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, ~Opcode, VTs, Ops);
    IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, DL, IP)) {
      return cast<MachineSDNode>(UpdateSDLocOnMergeSDNode(E, DL));
    }
  }

  // Allocate a new MachineSDNode.
  N = newSDNode<MachineSDNode>(~Opcode, DL.getIROrder(), DL.getDebugLoc(), VTs);
  createOperands(N, Ops);

  if (DoCSE)
    CSEMap.InsertNode(N, IP);

  InsertNode(N);
  return N;
}

/// getTargetExtractSubreg - A convenience function for creating
/// TargetOpcode::EXTRACT_SUBREG nodes.
SDValue SelectionDAG::getTargetExtractSubreg(int SRIdx, const SDLoc &DL, EVT VT,
                                             SDValue Operand) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, DL, MVT::i32);
  SDNode *Subreg = getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL,
                                  VT, Operand, SRIdxVal);
  return SDValue(Subreg, 0);
}

/// getTargetInsertSubreg - A convenience function for creating
/// TargetOpcode::INSERT_SUBREG nodes.
SDValue SelectionDAG::getTargetInsertSubreg(int SRIdx, const SDLoc &DL, EVT VT,
                                            SDValue Operand, SDValue Subreg) {
  SDValue SRIdxVal = getTargetConstant(SRIdx, DL, MVT::i32);
  SDNode *Result = getMachineNode(TargetOpcode::INSERT_SUBREG, DL,
                                  VT, Operand, Subreg, SRIdxVal);
  return SDValue(Result, 0);
}

/// getNodeIfExists - Get the specified node if it's already available, or
/// else return NULL.
SDNode *SelectionDAG::getNodeIfExists(unsigned Opcode, SDVTList VTList,
                                      ArrayRef<SDValue> Ops,
                                      const SDNodeFlags Flags) {
  if (VTList.VTs[VTList.NumVTs - 1] != MVT::Glue) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops);
    void *IP = nullptr;
    if (SDNode *E = FindNodeOrInsertPos(ID, SDLoc(), IP)) {
      E->intersectFlagsWith(Flags);
      return E;
    }
  }
  return nullptr;
}

/// getDbgValue - Creates a SDDbgValue node.
///
/// SDNode
SDDbgValue *SelectionDAG::getDbgValue(MDNode *Var, MDNode *Expr, SDNode *N,
                                      unsigned R, bool IsIndirect, uint64_t Off,
                                      const DebugLoc &DL, unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc())
      SDDbgValue(Var, Expr, N, R, IsIndirect, Off, DL, O);
}

/// Constant
SDDbgValue *SelectionDAG::getConstantDbgValue(MDNode *Var, MDNode *Expr,
                                              const Value *C, uint64_t Off,
                                              const DebugLoc &DL, unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc()) SDDbgValue(Var, Expr, C, Off, DL, O);
}

/// FrameIndex
SDDbgValue *SelectionDAG::getFrameIndexDbgValue(MDNode *Var, MDNode *Expr,
                                                unsigned FI, uint64_t Off,
                                                const DebugLoc &DL,
                                                unsigned O) {
  assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return new (DbgInfo->getAlloc()) SDDbgValue(Var, Expr, FI, Off, DL, O);
}

namespace {

/// RAUWUpdateListener - Helper for ReplaceAllUsesWith - When the node
/// pointed to by a use iterator is deleted, increment the use iterator
/// so that it doesn't dangle.
///
class RAUWUpdateListener : public SelectionDAG::DAGUpdateListener {
  SDNode::use_iterator &UI;
  SDNode::use_iterator &UE;

  void NodeDeleted(SDNode *N, SDNode *E) override {
    // Increment the iterator as needed.
    while (UI != UE && N == *UI)
      ++UI;
  }

public:
  RAUWUpdateListener(SelectionDAG &d,
                     SDNode::use_iterator &ui,
                     SDNode::use_iterator &ue)
    : SelectionDAG::DAGUpdateListener(d), UI(ui), UE(ue) {}
};

}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From has a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDValue FromN, SDValue To) {
  SDNode *From = FromN.getNode();
  assert(From->getNumValues() == 1 && FromN.getResNo() == 0 &&
         "Cannot replace with this method!");
  assert(From != To.getNode() && "Cannot replace uses of with self");

  // Preserve Debug Values
  TransferDbgValues(FromN, To);

  // Iterate over all the existing uses of From. New uses will be added
  // to the beginning of the use list, which we avoid visiting.
  // This specifically avoids visiting uses of From that arise while the
  // replacement is happening, because any such uses would be the result
  // of CSE: If an existing node looks like From after one of its operands
  // is replaced by To, we don't want to replace of all its users with To
  // too. See PR3018 for more info.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
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
    AddModifiedNodeToCSEMaps(User);
  }


  // If we just RAUW'd the root, take note.
  if (FromN == getRoot())
    setRoot(To);
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes that for each value of From, there is a
/// corresponding value in To in the same position with the same type.
///
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, SDNode *To) {
#ifndef NDEBUG
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i)
    assert((!From->hasAnyUseOfValue(i) ||
            From->getValueType(i) == To->getValueType(i)) &&
           "Cannot use this version of ReplaceAllUsesWith!");
#endif

  // Handle the trivial case.
  if (From == To)
    return;

  // Preserve Debug Info. Only do this if there's a use.
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i)
    if (From->hasAnyUseOfValue(i)) {
      assert((i < To->getNumValues()) && "Invalid To location");
      TransferDbgValues(SDValue(From, i), SDValue(To, i));
    }

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
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
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot().getNode())
    setRoot(SDValue(To, getRoot().getResNo()));
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, const SDValue *To) {
  if (From->getNumValues() == 1)  // Handle the simple case efficiently.
    return ReplaceAllUsesWith(SDValue(From, 0), To[0]);

  // Preserve Debug Info.
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i)
    TransferDbgValues(SDValue(From, i), *To);

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From->use_begin(), UE = From->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
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
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot().getNode())
    setRoot(SDValue(To[getRoot().getResNo()]));
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.getNode() alone.  The Deleted
/// vector is handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDValue From, SDValue To){
  // Handle the really simple, really trivial case efficiently.
  if (From == To) return;

  // Handle the simple, trivial, case efficiently.
  if (From.getNode()->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To);
    return;
  }

  // Preserve Debug Info.
  TransferDbgValues(From, To);

  // Iterate over just the existing users of From. See the comments in
  // the ReplaceAllUsesWith above.
  SDNode::use_iterator UI = From.getNode()->use_begin(),
                       UE = From.getNode()->use_end();
  RAUWUpdateListener Listener(*this, UI, UE);
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
    AddModifiedNodeToCSEMaps(User);
  }

  // If we just RAUW'd the root, take note.
  if (From == getRoot())
    setRoot(To);
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
                                              unsigned Num){
  // Handle the simple, trivial case efficiently.
  if (Num == 1)
    return ReplaceAllUsesOfValueWith(*From, *To);

  TransferDbgValues(*From, *To);

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
    AddModifiedNodeToCSEMaps(User);
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
    SDNode *N = &*I++;
    checkForCycles(N, this);
    unsigned Degree = N->getNumOperands();
    if (Degree == 0) {
      // A node with no uses, add it to the result array immediately.
      N->setNodeId(DAGSize++);
      allnodes_iterator Q(N);
      if (Q != SortedPos)
        SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(Q));
      assert(SortedPos != AllNodes.end() && "Overran node list");
      ++SortedPos;
    } else {
      // Temporarily use the Node Id as scratch space for the degree count.
      N->setNodeId(Degree);
    }
  }

  // Visit all the nodes. As we iterate, move nodes into sorted order,
  // such that by the time the end is reached all nodes will be sorted.
  for (SDNode &Node : allnodes()) {
    SDNode *N = &Node;
    checkForCycles(N, this);
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
        if (P->getIterator() != SortedPos)
          SortedPos = AllNodes.insert(SortedPos, AllNodes.remove(P));
        assert(SortedPos != AllNodes.end() && "Overran node list");
        ++SortedPos;
      } else {
        // Update P's outstanding operand count.
        P->setNodeId(Degree);
      }
    }
    if (Node.getIterator() == SortedPos) {
#ifndef NDEBUG
      allnodes_iterator I(N);
      SDNode *S = &*++I;
      dbgs() << "Overran sorted position:\n";
      S->dumprFull(this); dbgs() << "\n";
      dbgs() << "Checking if this is due to cycles\n";
      checkForCycles(this, true);
#endif
      llvm_unreachable(nullptr);
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

/// AddDbgValue - Add a dbg_value SDNode. If SD is non-null that means the
/// value is produced by SD.
void SelectionDAG::AddDbgValue(SDDbgValue *DB, SDNode *SD, bool isParameter) {
  if (SD) {
    assert(DbgInfo->getSDDbgValues(SD).empty() || SD->getHasDebugValue());
    SD->setHasDebugValue(true);
  }
  DbgInfo->add(DB, SD, isParameter);
}

/// TransferDbgValues - Transfer SDDbgValues. Called in replace nodes.
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
    // Only add Dbgvalues attached to same ResNo.
    if (Dbg->getKind() == SDDbgValue::SDNODE &&
        Dbg->getSDNode() == From.getNode() &&
        Dbg->getResNo() == From.getResNo() && !Dbg->isInvalidated()) {
      assert(FromNode != ToNode &&
             "Should not transfer Debug Values intranode");
      SDDbgValue *Clone =
          getDbgValue(Dbg->getVariable(), Dbg->getExpression(), ToNode,
                      To.getResNo(), Dbg->isIndirect(), Dbg->getOffset(),
                      Dbg->getDebugLoc(), Dbg->getOrder());
      ClonedDVs.push_back(Clone);
      Dbg->setIsInvalidated();
    }
  }
  for (SDDbgValue *I : ClonedDVs)
    AddDbgValue(I, ToNode, false);
}

//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//

bool llvm::isNullConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isNullValue();
}

bool llvm::isNullFPConstant(SDValue V) {
  ConstantFPSDNode *Const = dyn_cast<ConstantFPSDNode>(V);
  return Const != nullptr && Const->isZero() && !Const->isNegative();
}

bool llvm::isAllOnesConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isAllOnesValue();
}

bool llvm::isOneConstant(SDValue V) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isOne();
}

bool llvm::isBitwiseNot(SDValue V) {
  return V.getOpcode() == ISD::XOR && isAllOnesConstant(V.getOperand(1));
}

ConstantSDNode *llvm::isConstOrConstSplat(SDValue N) {
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N))
    return CN;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N)) {
    BitVector UndefElements;
    ConstantSDNode *CN = BV->getConstantSplatNode(&UndefElements);

    // BuildVectors can truncate their operands. Ignore that case here.
    // FIXME: We blindly ignore splats which include undef which is overly
    // pessimistic.
    if (CN && UndefElements.none() &&
        CN->getValueType(0) == N.getValueType().getScalarType())
      return CN;
  }

  return nullptr;
}

ConstantFPSDNode *llvm::isConstOrConstSplatFP(SDValue N) {
  if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N))
    return CN;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(N)) {
    BitVector UndefElements;
    ConstantFPSDNode *CN = BV->getConstantFPSplatNode(&UndefElements);

    if (CN && UndefElements.none())
      return CN;
  }

  return nullptr;
}

HandleSDNode::~HandleSDNode() {
  DropOperands();
}

GlobalAddressSDNode::GlobalAddressSDNode(unsigned Opc, unsigned Order,
                                         const DebugLoc &DL,
                                         const GlobalValue *GA, EVT VT,
                                         int64_t o, unsigned char TF)
    : SDNode(Opc, Order, DL, getSDVTList(VT)), Offset(o), TargetFlags(TF) {
  TheGlobal = GA;
}

AddrSpaceCastSDNode::AddrSpaceCastSDNode(unsigned Order, const DebugLoc &dl,
                                         EVT VT, unsigned SrcAS,
                                         unsigned DestAS)
    : SDNode(ISD::ADDRSPACECAST, Order, dl, getSDVTList(VT)),
      SrcAddrSpace(SrcAS), DestAddrSpace(DestAS) {}

MemSDNode::MemSDNode(unsigned Opc, unsigned Order, const DebugLoc &dl,
                     SDVTList VTs, EVT memvt, MachineMemOperand *mmo)
    : SDNode(Opc, Order, dl, VTs), MemoryVT(memvt), MMO(mmo) {
  MemSDNodeBits.IsVolatile = MMO->isVolatile();
  MemSDNodeBits.IsNonTemporal = MMO->isNonTemporal();
  MemSDNodeBits.IsDereferenceable = MMO->isDereferenceable();
  MemSDNodeBits.IsInvariant = MMO->isInvariant();

  // We check here that the size of the memory operand fits within the size of
  // the MMO. This is because the MMO might indicate only a possible address
  // range instead of specifying the affected memory addresses precisely.
  assert(memvt.getStoreSize() <= MMO->getSize() && "Size mismatch!");
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
bool SDNode::isOnlyUserOf(const SDNode *N) const {
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

/// Return true if the only users of N are contained in Nodes.
bool SDNode::areOnlyUsersOf(ArrayRef<const SDNode *> Nodes, const SDNode *N) {
  bool Seen = false;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDNode *User = *I;
    if (llvm::any_of(Nodes,
                     [&User](const SDNode *Node) { return User == Node; }))
      Seen = true;
    else
      return false;
  }

  return Seen;
}

/// isOperand - Return true if this node is an operand of N.
///
bool SDValue::isOperandOf(const SDNode *N) const {
  for (const SDValue &Op : N->op_values())
    if (*this == Op)
      return true;
  return false;
}

bool SDNode::isOperandOf(const SDNode *N) const {
  for (const SDValue &Op : N->op_values())
    if (this == Op.getNode())
      return true;
  return false;
}

/// reachesChainWithoutSideEffects - Return true if this operand (which must
/// be a chain) reaches the specified operand without crossing any
/// side-effecting instructions on any chain path.  In practice, this looks
/// through token factors and non-volatile loads.  In order to remain efficient,
/// this only looks a couple of nodes in, it does not do an exhaustive search.
///
/// Note that we only need to examine chains when we're searching for
/// side-effects; SelectionDAG requires that all side-effects are represented
/// by chains, even if another operand would force a specific ordering. This
/// constraint is necessary to allow transformations like splitting loads.
bool SDValue::reachesChainWithoutSideEffects(SDValue Dest,
                                             unsigned Depth) const {
  if (*this == Dest) return true;

  // Don't search too deeply, we just want to be able to see through
  // TokenFactor's etc.
  if (Depth == 0) return false;

  // If this is a token factor, all inputs to the TF happen in parallel.
  if (getOpcode() == ISD::TokenFactor) {
    // First, try a shallow search.
    if (is_contained((*this)->ops(), Dest)) {
      // We found the chain we want as an operand of this TokenFactor.
      // Essentially, we reach the chain without side-effects if we could
      // serialize the TokenFactor into a simple chain of operations with
      // Dest as the last operation. This is automatically true if the
      // chain has one use: there are no other ordering constraints.
      // If the chain has more than one use, we give up: some other
      // use of Dest might force a side-effect between Dest and the current
      // node.
      if (Dest.hasOneUse())
        return true;
    }
    // Next, try a deep search: check whether every operand of the TokenFactor
    // reaches Dest.
    return all_of((*this)->ops(), [=](SDValue Op) {
      return Op.reachesChainWithoutSideEffects(Dest, Depth - 1);
    });
  }

  // Loads don't have side effects, look through them.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(*this)) {
    if (!Ld->isVolatile())
      return Ld->getChain().reachesChainWithoutSideEffects(Dest, Depth-1);
  }
  return false;
}

bool SDNode::hasPredecessor(const SDNode *N) const {
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;
  Worklist.push_back(this);
  return hasPredecessorHelper(N, Visited, Worklist);
}

void SDNode::intersectFlagsWith(const SDNodeFlags Flags) {
  this->Flags.intersectWith(Flags);
}

SDValue SelectionDAG::UnrollVectorOp(SDNode *N, unsigned ResNE) {
  assert(N->getNumValues() == 1 &&
         "Can't unroll a vector with multiple results!");

  EVT VT = N->getValueType(0);
  unsigned NE = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDLoc dl(N);

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
        Operands[j] =
            getNode(ISD::EXTRACT_VECTOR_ELT, dl, OperandEltVT, Operand,
                    getConstant(i, dl, TLI->getVectorIdxTy(getDataLayout())));
      } else {
        // A scalar operand; just use it as is.
        Operands[j] = Operand;
      }
    }

    switch (N->getOpcode()) {
    default: {
      Scalars.push_back(getNode(N->getOpcode(), dl, EltVT, Operands,
                                N->getFlags()));
      break;
    }
    case ISD::VSELECT:
      Scalars.push_back(getNode(ISD::SELECT, dl, EltVT, Operands));
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

  EVT VecVT = EVT::getVectorVT(*getContext(), EltVT, ResNE);
  return getBuildVector(VecVT, dl, Scalars);
}

bool SelectionDAG::areNonVolatileConsecutiveLoads(LoadSDNode *LD,
                                                  LoadSDNode *Base,
                                                  unsigned Bytes,
                                                  int Dist) const {
  if (LD->isVolatile() || Base->isVolatile())
    return false;
  if (LD->isIndexed() || Base->isIndexed())
    return false;
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
    const MachineFrameInfo &MFI = getMachineFunction().getFrameInfo();
    int FI  = cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = cast<FrameIndexSDNode>(BaseLoc)->getIndex();
    int FS  = MFI.getObjectSize(FI);
    int BFS = MFI.getObjectSize(BFI);
    if (FS != BFS || FS != (int)Bytes) return false;
    return MFI.getObjectOffset(FI) == (MFI.getObjectOffset(BFI) + Dist*Bytes);
  }

  // Handle X + C.
  if (isBaseWithConstantOffset(Loc)) {
    int64_t LocOffset = cast<ConstantSDNode>(Loc.getOperand(1))->getSExtValue();
    if (Loc.getOperand(0) == BaseLoc) {
      // If the base location is a simple address with no offset itself, then
      // the second load's first add operand should be the base address.
      if (LocOffset == Dist * (int)Bytes)
        return true;
    } else if (isBaseWithConstantOffset(BaseLoc)) {
      // The base location itself has an offset, so subtract that value from the
      // second load's offset before comparing to distance * size.
      int64_t BOffset =
        cast<ConstantSDNode>(BaseLoc.getOperand(1))->getSExtValue();
      if (Loc.getOperand(0) == BaseLoc.getOperand(0)) {
        if ((LocOffset - BOffset) == Dist * (int)Bytes)
          return true;
      }
    }
  }
  const GlobalValue *GV1 = nullptr;
  const GlobalValue *GV2 = nullptr;
  int64_t Offset1 = 0;
  int64_t Offset2 = 0;
  bool isGA1 = TLI->isGAPlusOffset(Loc.getNode(), GV1, Offset1);
  bool isGA2 = TLI->isGAPlusOffset(BaseLoc.getNode(), GV2, Offset2);
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
  if (TLI->isGAPlusOffset(Ptr.getNode(), GV, GVOffset)) {
    unsigned PtrWidth = getDataLayout().getPointerTypeSizeInBits(GV->getType());
    KnownBits Known(PtrWidth);
    llvm::computeKnownBits(const_cast<GlobalValue *>(GV), Known,
                           getDataLayout());
    unsigned AlignBits = Known.Zero.countTrailingOnes();
    unsigned Align = AlignBits ? 1 << std::min(31U, AlignBits) : 0;
    if (Align)
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
    const MachineFrameInfo &MFI = getMachineFunction().getFrameInfo();
    unsigned FIInfoAlign = MinAlign(MFI.getObjectAlignment(FrameIdx),
                                    FrameOffset);
    return FIInfoAlign;
  }

  return 0;
}

/// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a type
/// which is split (or expanded) into two not necessarily identical pieces.
std::pair<EVT, EVT> SelectionDAG::GetSplitDestVTs(const EVT &VT) const {
  // Currently all types are split in half.
  EVT LoVT, HiVT;
  if (!VT.isVector())
    LoVT = HiVT = TLI->getTypeToTransformTo(*getContext(), VT);
  else
    LoVT = HiVT = VT.getHalfNumVectorElementsVT(*getContext());

  return std::make_pair(LoVT, HiVT);
}

/// SplitVector - Split the vector with EXTRACT_SUBVECTOR and return the
/// low/high part.
std::pair<SDValue, SDValue>
SelectionDAG::SplitVector(const SDValue &N, const SDLoc &DL, const EVT &LoVT,
                          const EVT &HiVT) {
  assert(LoVT.getVectorNumElements() + HiVT.getVectorNumElements() <=
         N.getValueType().getVectorNumElements() &&
         "More vector elements requested than available!");
  SDValue Lo, Hi;
  Lo = getNode(ISD::EXTRACT_SUBVECTOR, DL, LoVT, N,
               getConstant(0, DL, TLI->getVectorIdxTy(getDataLayout())));
  Hi = getNode(ISD::EXTRACT_SUBVECTOR, DL, HiVT, N,
               getConstant(LoVT.getVectorNumElements(), DL,
                           TLI->getVectorIdxTy(getDataLayout())));
  return std::make_pair(Lo, Hi);
}

void SelectionDAG::ExtractVectorElements(SDValue Op,
                                         SmallVectorImpl<SDValue> &Args,
                                         unsigned Start, unsigned Count) {
  EVT VT = Op.getValueType();
  if (Count == 0)
    Count = VT.getVectorNumElements();

  EVT EltVT = VT.getVectorElementType();
  EVT IdxTy = TLI->getVectorIdxTy(getDataLayout());
  SDLoc SL(Op);
  for (unsigned i = Start, e = Start + Count; i != e; ++i) {
    Args.push_back(getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                           Op, getConstant(i, SL, IdxTy)));
  }
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

bool BuildVectorSDNode::isConstantSplat(APInt &SplatValue, APInt &SplatUndef,
                                        unsigned &SplatBitSize,
                                        bool &HasAnyUndefs,
                                        unsigned MinSplatBits,
                                        bool IsBigEndian) const {
  EVT VT = getValueType(0);
  assert(VT.isVector() && "Expected a vector type");
  unsigned VecWidth = VT.getSizeInBits();
  if (MinSplatBits > VecWidth)
    return false;

  // FIXME: The widths are based on this node's type, but build vectors can
  // truncate their operands.
  SplatValue = APInt(VecWidth, 0);
  SplatUndef = APInt(VecWidth, 0);

  // Get the bits. Bits with undefined values (when the corresponding element
  // of the vector is an ISD::UNDEF value) are set in SplatUndef and cleared
  // in SplatValue. If any of the values are not constant, give up and return
  // false.
  unsigned int NumOps = getNumOperands();
  assert(NumOps > 0 && "isConstantSplat has 0-size build vector");
  unsigned EltWidth = VT.getScalarSizeInBits();

  for (unsigned j = 0; j < NumOps; ++j) {
    unsigned i = IsBigEndian ? NumOps - 1 - j : j;
    SDValue OpVal = getOperand(i);
    unsigned BitPos = j * EltWidth;

    if (OpVal.isUndef())
      SplatUndef.setBits(BitPos, BitPos + EltWidth);
    else if (auto *CN = dyn_cast<ConstantSDNode>(OpVal))
      SplatValue.insertBits(CN->getAPIntValue().zextOrTrunc(EltWidth), BitPos);
    else if (auto *CN = dyn_cast<ConstantFPSDNode>(OpVal))
      SplatValue.insertBits(CN->getValueAPF().bitcastToAPInt(), BitPos);
    else
      return false;
  }

  // The build_vector is all constants or undefs. Find the smallest element
  // size that splats the vector.
  HasAnyUndefs = (SplatUndef != 0);

  // FIXME: This does not work for vectors with elements less than 8 bits.
  while (VecWidth > 8) {
    unsigned HalfSize = VecWidth / 2;
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

    VecWidth = HalfSize;
  }

  SplatBitSize = VecWidth;
  return true;
}

SDValue BuildVectorSDNode::getSplatValue(BitVector *UndefElements) const {
  if (UndefElements) {
    UndefElements->clear();
    UndefElements->resize(getNumOperands());
  }
  SDValue Splatted;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    SDValue Op = getOperand(i);
    if (Op.isUndef()) {
      if (UndefElements)
        (*UndefElements)[i] = true;
    } else if (!Splatted) {
      Splatted = Op;
    } else if (Splatted != Op) {
      return SDValue();
    }
  }

  if (!Splatted) {
    assert(getOperand(0).isUndef() &&
           "Can only have a splat without a constant for all undefs.");
    return getOperand(0);
  }

  return Splatted;
}

ConstantSDNode *
BuildVectorSDNode::getConstantSplatNode(BitVector *UndefElements) const {
  return dyn_cast_or_null<ConstantSDNode>(getSplatValue(UndefElements));
}

ConstantFPSDNode *
BuildVectorSDNode::getConstantFPSplatNode(BitVector *UndefElements) const {
  return dyn_cast_or_null<ConstantFPSDNode>(getSplatValue(UndefElements));
}

int32_t
BuildVectorSDNode::getConstantFPSplatPow2ToLog2Int(BitVector *UndefElements,
                                                   uint32_t BitWidth) const {
  if (ConstantFPSDNode *CN =
          dyn_cast_or_null<ConstantFPSDNode>(getSplatValue(UndefElements))) {
    bool IsExact;
    APSInt IntVal(BitWidth);
    const APFloat &APF = CN->getValueAPF();
    if (APF.convertToInteger(IntVal, APFloat::rmTowardZero, &IsExact) !=
            APFloat::opOK ||
        !IsExact)
      return -1;

    return IntVal.exactLogBase2();
  }
  return -1;
}

bool BuildVectorSDNode::isConstant() const {
  for (const SDValue &Op : op_values()) {
    unsigned Opc = Op.getOpcode();
    if (Opc != ISD::UNDEF && Opc != ISD::Constant && Opc != ISD::ConstantFP)
      return false;
  }
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

// \brief Returns the SDNode if it is a constant integer BuildVector
// or constant integer.
SDNode *SelectionDAG::isConstantIntBuildVectorOrConstantInt(SDValue N) {
  if (isa<ConstantSDNode>(N))
    return N.getNode();
  if (ISD::isBuildVectorOfConstantSDNodes(N.getNode()))
    return N.getNode();
  // Treat a GlobalAddress supporting constant offset folding as a
  // constant integer.
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N))
    if (GA->getOpcode() == ISD::GlobalAddress &&
        TLI->isOffsetFoldingLegal(GA))
      return GA;
  return nullptr;
}

SDNode *SelectionDAG::isConstantFPBuildVectorOrConstantFP(SDValue N) {
  if (isa<ConstantFPSDNode>(N))
    return N.getNode();

  if (ISD::isBuildVectorOfConstantFPSDNodes(N.getNode()))
    return N.getNode();

  return nullptr;
}

#ifndef NDEBUG
static void checkForCyclesHelper(const SDNode *N,
                                 SmallPtrSetImpl<const SDNode*> &Visited,
                                 SmallPtrSetImpl<const SDNode*> &Checked,
                                 const llvm::SelectionDAG *DAG) {
  // If this node has already been checked, don't check it again.
  if (Checked.count(N))
    return;

  // If a node has already been visited on this depth-first walk, reject it as
  // a cycle.
  if (!Visited.insert(N).second) {
    errs() << "Detected cycle in SelectionDAG\n";
    dbgs() << "Offending node:\n";
    N->dumprFull(DAG); dbgs() << "\n";
    abort();
  }

  for (const SDValue &Op : N->op_values())
    checkForCyclesHelper(Op.getNode(), Visited, Checked, DAG);

  Checked.insert(N);
  Visited.erase(N);
}
#endif

void llvm::checkForCycles(const llvm::SDNode *N,
                          const llvm::SelectionDAG *DAG,
                          bool force) {
#ifndef NDEBUG
  bool check = force;
#ifdef EXPENSIVE_CHECKS
  check = true;
#endif  // EXPENSIVE_CHECKS
  if (check) {
    assert(N && "Checking nonexistent SDNode");
    SmallPtrSet<const SDNode*, 32> visited;
    SmallPtrSet<const SDNode*, 32> checked;
    checkForCyclesHelper(N, visited, checked, DAG);
  }
#endif  // !NDEBUG
}

void llvm::checkForCycles(const llvm::SelectionDAG *DAG, bool force) {
  checkForCycles(DAG->getRoot().getNode(), DAG, force);
}
