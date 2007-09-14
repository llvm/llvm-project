//===-- SelectionDAG.cpp - Implement the SelectionDAG data structures -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

/// makeVTList - Return an instance of the SDVTList struct initialized with the
/// specified members.
static SDVTList makeVTList(const MVT::ValueType *VTs, unsigned NumVTs) {
  SDVTList Res = {VTs, NumVTs};
  return Res;
}

//===----------------------------------------------------------------------===//
//                              ConstantFPSDNode Class
//===----------------------------------------------------------------------===//

/// isExactlyValue - We don't rely on operator== working on double values, as
/// it returns true for things that are clearly not equal, like -0.0 and 0.0.
/// As such, this method can be used to do an exact bit-for-bit comparison of
/// two floating point values.
bool ConstantFPSDNode::isExactlyValue(const APFloat& V) const {
  return Value.bitwiseIsEqual(V);
}

bool ConstantFPSDNode::isValueValidForType(MVT::ValueType VT, 
                                           const APFloat& Val) {
  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  switch (VT) {
  default:
    return false;         // These can't be represented as floating point!

  // FIXME rounding mode needs to be more flexible
  case MVT::f32:
    return &Val2.getSemantics() == &APFloat::IEEEsingle ||
           Val2.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven) == 
              APFloat::opOK;
  case MVT::f64:
    return &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           Val2.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven) == 
             APFloat::opOK;
  // TODO: Figure out how to test if we can use a shorter type instead!
  case MVT::f80:
  case MVT::f128:
  case MVT::ppcf128:
    return true;
  }
}

//===----------------------------------------------------------------------===//
//                              ISD Namespace
//===----------------------------------------------------------------------===//

/// isBuildVectorAllOnes - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are ~0 or undef.
bool ISD::isBuildVectorAllOnes(const SDNode *N) {
  // Look through a bit convert.
  if (N->getOpcode() == ISD::BIT_CONVERT)
    N = N->getOperand(0).Val;
  
  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;
  
  unsigned i = 0, e = N->getNumOperands();
  
  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;
  
  // Do not accept an all-undef vector.
  if (i == e) return false;
  
  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements.
  SDOperand NotZero = N->getOperand(i);
  if (isa<ConstantSDNode>(NotZero)) {
    if (!cast<ConstantSDNode>(NotZero)->isAllOnesValue())
      return false;
  } else if (isa<ConstantFPSDNode>(NotZero)) {
    MVT::ValueType VT = NotZero.getValueType();
    if (VT== MVT::f64) {
      if (((cast<ConstantFPSDNode>(NotZero)->getValueAPF().
                  convertToAPInt().getZExtValue())) != (uint64_t)-1)
        return false;
    } else {
      if ((uint32_t)cast<ConstantFPSDNode>(NotZero)->
                      getValueAPF().convertToAPInt().getZExtValue() != 
          (uint32_t)-1)
        return false;
    }
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
  if (N->getOpcode() == ISD::BIT_CONVERT)
    N = N->getOperand(0).Val;
  
  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;
  
  unsigned i = 0, e = N->getNumOperands();
  
  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;
  
  // Do not accept an all-undef vector.
  if (i == e) return false;
  
  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements.
  SDOperand Zero = N->getOperand(i);
  if (isa<ConstantSDNode>(Zero)) {
    if (!cast<ConstantSDNode>(Zero)->isNullValue())
      return false;
  } else if (isa<ConstantFPSDNode>(Zero)) {
    if (!cast<ConstantFPSDNode>(Zero)->getValueAPF().isPosZero())
      return false;
  } else
    return false;
  
  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != Zero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
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
                       (OldG << 2));        // New L bit.
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
    Operation &= ~8;     // Don't let N and U bits get set.
  return ISD::CondCode(Operation);
}


/// isSignedOp - For an integer comparison, return 1 if the comparison is a
/// signed operation and 2 if the result is an unsigned comparison.  Return zero
/// if the operation does not depend on the sign of the input (setne and seteq).
static int isSignedOp(ISD::CondCode Opcode) {
  switch (Opcode) {
  default: assert(0 && "Illegal integer setcc operation!");
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
    case ISD::SETUEQ: Result = ISD::SETEQ   ; break;  // SETUGE & SETULE
    case ISD::SETOLT: Result = ISD::SETULT  ; break;  // SETULT & SETNE
    case ISD::SETOGT: Result = ISD::SETUGT  ; break;  // SETUGT & SETNE
    }
  }
  
  return Result;
}

const TargetMachine &SelectionDAG::getTarget() const {
  return TLI.getTargetMachine();
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
void AddNodeIDValueTypes(FoldingSetNodeID &ID, SDVTList VTList) {
  ID.AddPointer(VTList.VTs);  
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              const SDOperand *Ops, unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops) {
    ID.AddPointer(Ops->Val);
    ID.AddInteger(Ops->ResNo);
  }
}

static void AddNodeIDNode(FoldingSetNodeID &ID,
                          unsigned short OpC, SDVTList VTList, 
                          const SDOperand *OpList, unsigned N) {
  AddNodeIDOpcode(ID, OpC);
  AddNodeIDValueTypes(ID, VTList);
  AddNodeIDOperands(ID, OpList, N);
}

/// AddNodeIDNode - Generic routine for adding a nodes info to the NodeID
/// data.
static void AddNodeIDNode(FoldingSetNodeID &ID, SDNode *N) {
  AddNodeIDOpcode(ID, N->getOpcode());
  // Add the return value info.
  AddNodeIDValueTypes(ID, N->getVTList());
  // Add the operand info.
  AddNodeIDOperands(ID, N->op_begin(), N->getNumOperands());

  // Handle SDNode leafs with special info.
  switch (N->getOpcode()) {
  default: break;  // Normal nodes don't need extra info.
  case ISD::TargetConstant:
  case ISD::Constant:
    ID.AddInteger(cast<ConstantSDNode>(N)->getValue());
    break;
  case ISD::TargetConstantFP:
  case ISD::ConstantFP: {
    APFloat V = cast<ConstantFPSDNode>(N)->getValueAPF();
    if (&V.getSemantics() == &APFloat::IEEEdouble)
      ID.AddDouble(V.convertToDouble());
    else if (&V.getSemantics() == &APFloat::IEEEsingle)
      ID.AddDouble((double)V.convertToFloat());
    else
      assert(0);
    break;
  }
  case ISD::TargetGlobalAddress:
  case ISD::GlobalAddress:
  case ISD::TargetGlobalTLSAddress:
  case ISD::GlobalTLSAddress: {
    GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);
    ID.AddPointer(GA->getGlobal());
    ID.AddInteger(GA->getOffset());
    break;
  }
  case ISD::BasicBlock:
    ID.AddPointer(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::Register:
    ID.AddInteger(cast<RegisterSDNode>(N)->getReg());
    break;
  case ISD::SRCVALUE: {
    SrcValueSDNode *SV = cast<SrcValueSDNode>(N);
    ID.AddPointer(SV->getValue());
    ID.AddInteger(SV->getOffset());
    break;
  }
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    ID.AddInteger(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::JumpTable:
  case ISD::TargetJumpTable:
    ID.AddInteger(cast<JumpTableSDNode>(N)->getIndex());
    break;
  case ISD::ConstantPool:
  case ISD::TargetConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(N);
    ID.AddInteger(CP->getAlignment());
    ID.AddInteger(CP->getOffset());
    if (CP->isMachineConstantPoolEntry())
      CP->getMachineCPVal()->AddSelectionDAGCSEId(ID);
    else
      ID.AddPointer(CP->getConstVal());
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(N);
    ID.AddInteger(LD->getAddressingMode());
    ID.AddInteger(LD->getExtensionType());
    ID.AddInteger((unsigned int)(LD->getLoadedVT()));
    ID.AddPointer(LD->getSrcValue());
    ID.AddInteger(LD->getSrcValueOffset());
    ID.AddInteger(LD->getAlignment());
    ID.AddInteger(LD->isVolatile());
    break;
  }
  case ISD::STORE: {
    StoreSDNode *ST = cast<StoreSDNode>(N);
    ID.AddInteger(ST->getAddressingMode());
    ID.AddInteger(ST->isTruncatingStore());
    ID.AddInteger((unsigned int)(ST->getStoredVT()));
    ID.AddPointer(ST->getSrcValue());
    ID.AddInteger(ST->getSrcValueOffset());
    ID.AddInteger(ST->getAlignment());
    ID.AddInteger(ST->isVolatile());
    break;
  }
  }
}

//===----------------------------------------------------------------------===//
//                              SelectionDAG Class
//===----------------------------------------------------------------------===//

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

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.back();
    DeadNodes.pop_back();
    
    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *Operand = I->Val;
      Operand->removeUser(N);
      
      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }
    if (N->OperandsNeedDelete)
      delete[] N->OperandList;
    N->OperandList = 0;
    N->NumOperands = 0;
    
    // Finally, remove N itself.
    AllNodes.erase(N);
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

void SelectionDAG::RemoveDeadNode(SDNode *N, std::vector<SDNode*> &Deleted) {
  SmallVector<SDNode*, 16> DeadNodes;
  DeadNodes.push_back(N);

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.back();
    DeadNodes.pop_back();
    
    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *Operand = I->Val;
      Operand->removeUser(N);
      
      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }
    if (N->OperandsNeedDelete)
      delete[] N->OperandList;
    N->OperandList = 0;
    N->NumOperands = 0;
    
    // Finally, remove N itself.
    Deleted.push_back(N);
    AllNodes.erase(N);
  }
}

void SelectionDAG::DeleteNode(SDNode *N) {
  assert(N->use_empty() && "Cannot delete a node that is not dead!");

  // First take this out of the appropriate CSE map.
  RemoveNodeFromCSEMaps(N);

  // Finally, remove uses due to operands of this node, remove from the 
  // AllNodes list, and delete the node.
  DeleteNodeNotInCSEMaps(N);
}

void SelectionDAG::DeleteNodeNotInCSEMaps(SDNode *N) {

  // Remove it from the AllNodes list.
  AllNodes.remove(N);
    
  // Drop all of the operands and decrement used nodes use counts.
  for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I)
    I->Val->removeUser(N);
  if (N->OperandsNeedDelete)
    delete[] N->OperandList;
  N->OperandList = 0;
  N->NumOperands = 0;
  
  delete N;
}

/// RemoveNodeFromCSEMaps - Take the specified node out of the CSE map that
/// correspond to it.  This is useful when we're about to delete or repurpose
/// the node.  We don't want future request for structurally identical nodes
/// to return N anymore.
void SelectionDAG::RemoveNodeFromCSEMaps(SDNode *N) {
  bool Erased = false;
  switch (N->getOpcode()) {
  case ISD::HANDLENODE: return;  // noop.
  case ISD::STRING:
    Erased = StringNodes.erase(cast<StringSDNode>(N)->getValue());
    break;
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != 0;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = 0;
    break;
  case ISD::ExternalSymbol:
    Erased = ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::TargetExternalSymbol:
    Erased =
      TargetExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::VALUETYPE:
    Erased = ValueTypeNodes[cast<VTSDNode>(N)->getVT()] != 0;
    ValueTypeNodes[cast<VTSDNode>(N)->getVT()] = 0;
    break;
  default:
    // Remove it from the CSE Map.
    Erased = CSEMap.RemoveNode(N);
    break;
  }
#ifndef NDEBUG
  // Verify that the node was actually in one of the CSE maps, unless it has a 
  // flag result (which cannot be CSE'd) or is one of the special cases that are
  // not subject to CSE.
  if (!Erased && N->getValueType(N->getNumValues()-1) != MVT::Flag &&
      !N->isTargetOpcode()) {
    N->dump(this);
    cerr << "\n";
    assert(0 && "Node is not in map!");
  }
#endif
}

/// AddNonLeafNodeToCSEMaps - Add the specified node back to the CSE maps.  It
/// has been taken out and modified in some way.  If the specified node already
/// exists in the CSE maps, do not modify the maps, but return the existing node
/// instead.  If it doesn't exist, add it and return null.
///
SDNode *SelectionDAG::AddNonLeafNodeToCSEMaps(SDNode *N) {
  assert(N->getNumOperands() && "This is a leaf node!");
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  SDNode *New = CSEMap.GetOrInsertNode(N);
  if (New != N) return New;  // Node already existed.
  return 0;
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, SDOperand Op,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  SDOperand Ops[] = { Op };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 1);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, 
                                           SDOperand Op1, SDOperand Op2,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
                                              
  SDOperand Ops[] = { Op1, Op2 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 2);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, 
                                           const SDOperand *Ops,unsigned NumOps,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, NumOps);
  
  if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    ID.AddInteger(LD->getAddressingMode());
    ID.AddInteger(LD->getExtensionType());
    ID.AddInteger((unsigned int)(LD->getLoadedVT()));
    ID.AddPointer(LD->getSrcValue());
    ID.AddInteger(LD->getSrcValueOffset());
    ID.AddInteger(LD->getAlignment());
    ID.AddInteger(LD->isVolatile());
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    ID.AddInteger(ST->getAddressingMode());
    ID.AddInteger(ST->isTruncatingStore());
    ID.AddInteger((unsigned int)(ST->getStoredVT()));
    ID.AddPointer(ST->getSrcValue());
    ID.AddInteger(ST->getSrcValueOffset());
    ID.AddInteger(ST->getAlignment());
    ID.AddInteger(ST->isVolatile());
  }
  
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


SelectionDAG::~SelectionDAG() {
  while (!AllNodes.empty()) {
    SDNode *N = AllNodes.begin();
    N->SetNextInBucket(0);
    if (N->OperandsNeedDelete)
      delete [] N->OperandList;
    N->OperandList = 0;
    N->NumOperands = 0;
    AllNodes.pop_front();
  }
}

SDOperand SelectionDAG::getZeroExtendInReg(SDOperand Op, MVT::ValueType VT) {
  if (Op.getValueType() == VT) return Op;
  int64_t Imm = ~0ULL >> (64-MVT::getSizeInBits(VT));
  return getNode(ISD::AND, Op.getValueType(), Op,
                 getConstant(Imm, Op.getValueType()));
}

SDOperand SelectionDAG::getString(const std::string &Val) {
  StringSDNode *&N = StringNodes[Val];
  if (!N) {
    N = new StringSDNode(Val);
    AllNodes.push_back(N);
  }
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstant(uint64_t Val, MVT::ValueType VT, bool isT) {
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");
  assert(!MVT::isVector(VT) && "Cannot create Vector ConstantSDNodes!");
  
  // Mask out any bits that are not valid for this constant.
  Val &= MVT::getIntVTBitMask(VT);

  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(Val);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantSDNode(isT, Val, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantFP(const APFloat& V, MVT::ValueType VT,
                                      bool isTarget) {
  assert(MVT::isFloatingPoint(VT) && "Cannot create integer FP constant!");
                                
  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;
  bool isDouble = (EltVT == MVT::f64);
  double Val = isDouble ? V.convertToDouble() : (double)V.convertToFloat();

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  // ?? Should we store float/double/longdouble separately in ID?
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), 0, 0);
  ID.AddDouble(Val);
  void *IP = 0;
  SDNode *N = NULL;
  if ((N = CSEMap.FindNodeOrInsertPos(ID, IP)))
    if (!MVT::isVector(VT))
      return SDOperand(N, 0);
  if (!N) {
    N = new ConstantFPSDNode(isTarget, 
      isDouble ? APFloat(Val) : APFloat((float)Val), EltVT);
    CSEMap.InsertNode(N, IP);
    AllNodes.push_back(N);
  }

  SDOperand Result(N, 0);
  if (MVT::isVector(VT)) {
    SmallVector<SDOperand, 8> Ops;
    Ops.assign(MVT::getVectorNumElements(VT), Result);
    Result = getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
  return Result;
}

SDOperand SelectionDAG::getConstantFP(double Val, MVT::ValueType VT,
                                      bool isTarget) {
  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;
  if (EltVT==MVT::f32)
    return getConstantFP(APFloat((float)Val), VT, isTarget);
  else
    return getConstantFP(APFloat(Val), VT, isTarget);
}

SDOperand SelectionDAG::getGlobalAddress(const GlobalValue *GV,
                                         MVT::ValueType VT, int Offset,
                                         bool isTargetGA) {
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  unsigned Opc;
  if (GVar && GVar->isThreadLocal())
    Opc = isTargetGA ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress;
  else
    Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddPointer(GV);
  ID.AddInteger(Offset);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
   return SDOperand(E, 0);
  SDNode *N = new GlobalAddressSDNode(isTargetGA, GV, VT, Offset);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getFrameIndex(int FI, MVT::ValueType VT,
                                      bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetFrameIndex : ISD::FrameIndex;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(FI);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new FrameIndexSDNode(FI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getJumpTable(int JTI, MVT::ValueType VT, bool isTarget){
  unsigned Opc = isTarget ? ISD::TargetJumpTable : ISD::JumpTable;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(JTI);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new JumpTableSDNode(JTI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantPool(Constant *C, MVT::ValueType VT,
                                        unsigned Alignment, int Offset,
                                        bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  ID.AddPointer(C);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantPoolSDNode(isTarget, C, VT, Offset, Alignment);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getConstantPool(MachineConstantPoolValue *C,
                                        MVT::ValueType VT,
                                        unsigned Alignment, int Offset,
                                        bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), 0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  C->AddSelectionDAGCSEId(ID);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantPoolSDNode(isTarget, C, VT, Offset, Alignment);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BasicBlock, getVTList(MVT::Other), 0, 0);
  ID.AddPointer(MBB);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new BasicBlockSDNode(MBB);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getValueType(MVT::ValueType VT) {
  if ((unsigned)VT >= ValueTypeNodes.size())
    ValueTypeNodes.resize(VT+1);
  if (ValueTypeNodes[VT] == 0) {
    ValueTypeNodes[VT] = new VTSDNode(VT);
    AllNodes.push_back(ValueTypeNodes[VT]);
  }

  return SDOperand(ValueTypeNodes[VT], 0);
}

SDOperand SelectionDAG::getExternalSymbol(const char *Sym, MVT::ValueType VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDOperand(N, 0);
  N = new ExternalSymbolSDNode(false, Sym, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetExternalSymbol(const char *Sym,
                                                MVT::ValueType VT) {
  SDNode *&N = TargetExternalSymbols[Sym];
  if (N) return SDOperand(N, 0);
  N = new ExternalSymbolSDNode(true, Sym, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getCondCode(ISD::CondCode Cond) {
  if ((unsigned)Cond >= CondCodeNodes.size())
    CondCodeNodes.resize(Cond+1);
  
  if (CondCodeNodes[Cond] == 0) {
    CondCodeNodes[Cond] = new CondCodeSDNode(Cond);
    AllNodes.push_back(CondCodeNodes[Cond]);
  }
  return SDOperand(CondCodeNodes[Cond], 0);
}

SDOperand SelectionDAG::getRegister(unsigned RegNo, MVT::ValueType VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::Register, getVTList(VT), 0, 0);
  ID.AddInteger(RegNo);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new RegisterSDNode(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getSrcValue(const Value *V, int Offset) {
  assert((!V || isa<PointerType>(V->getType())) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::SRCVALUE, getVTList(MVT::Other), 0, 0);
  ID.AddPointer(V);
  ID.AddInteger(Offset);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SrcValueSDNode(V, Offset);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::FoldSetCC(MVT::ValueType VT, SDOperand N1,
                                  SDOperand N2, ISD::CondCode Cond) {
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
    assert(!MVT::isInteger(N1.getValueType()) && "Illegal setcc for integer!");
    break;
  }
  
  if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val)) {
    uint64_t C2 = N2C->getValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val)) {
      uint64_t C1 = N1C->getValue();
      
      // Sign extend the operands if required
      if (ISD::isSignedIntSetCC(Cond)) {
        C1 = N1C->getSignExtended();
        C2 = N2C->getSignExtended();
      }
      
      switch (Cond) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, VT);
      case ISD::SETULT: return getConstant(C1 <  C2, VT);
      case ISD::SETUGT: return getConstant(C1 >  C2, VT);
      case ISD::SETULE: return getConstant(C1 <= C2, VT);
      case ISD::SETUGE: return getConstant(C1 >= C2, VT);
      case ISD::SETLT:  return getConstant((int64_t)C1 <  (int64_t)C2, VT);
      case ISD::SETGT:  return getConstant((int64_t)C1 >  (int64_t)C2, VT);
      case ISD::SETLE:  return getConstant((int64_t)C1 <= (int64_t)C2, VT);
      case ISD::SETGE:  return getConstant((int64_t)C1 >= (int64_t)C2, VT);
      }
    }
  }
  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.Val))
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2.Val)) {

      APFloat::cmpResult R = N1C->getValueAPF().compare(N2C->getValueAPF());
      switch (Cond) {
      default: break;
      case ISD::SETEQ:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOEQ: return getConstant(R==APFloat::cmpEqual, VT);
      case ISD::SETNE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETONE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpLessThan, VT);
      case ISD::SETLT:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOLT: return getConstant(R==APFloat::cmpLessThan, VT);
      case ISD::SETGT:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOGT: return getConstant(R==APFloat::cmpGreaterThan, VT);
      case ISD::SETLE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOLE: return getConstant(R==APFloat::cmpLessThan ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETGE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
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
      return getSetCC(VT, N2, N1, ISD::getSetCCSwappedOperands(Cond));
    }
      
  // Could not fold it.
  return SDOperand();
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool SelectionDAG::MaskedValueIsZero(SDOperand Op, uint64_t Mask, 
                                     unsigned Depth) const {
  // The masks are not wide enough to represent this type!  Should use APInt.
  if (Op.getValueType() == MVT::i128)
    return false;
  
  uint64_t KnownZero, KnownOne;
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}

/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
void SelectionDAG::ComputeMaskedBits(SDOperand Op, uint64_t Mask, 
                                     uint64_t &KnownZero, uint64_t &KnownOne,
                                     unsigned Depth) const {
  KnownZero = KnownOne = 0;   // Don't know anything.
  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.
  
  // The masks are not wide enough to represent this type!  Should use APInt.
  if (Op.getValueType() == MVT::i128)
    return;
  
  uint64_t KnownZero2, KnownOne2;

  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownZero;
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  case ISD::OR:
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownOne;
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
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
    uint64_t KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
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
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult)
      KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
    return;
  case ISD::SHL:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      ComputeMaskedBits(Op.getOperand(0), Mask >> SA->getValue(),
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= SA->getValue();
      KnownOne  <<= SA->getValue();
      KnownZero |= (1ULL << SA->getValue())-1;  // low bits known zero.
    }
    return;
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();

      uint64_t TypeMask = MVT::getIntVTBitMask(VT);
      ComputeMaskedBits(Op.getOperand(0), (Mask << ShAmt) & TypeMask,
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= ShAmt;
      KnownOne  >>= ShAmt;

      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= MVT::getSizeInBits(VT)-ShAmt;
      KnownZero |= HighBits;  // High bits known zero.
    }
    return;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();

      // Compute the new bits that are at the top now.
      uint64_t TypeMask = MVT::getIntVTBitMask(VT);

      uint64_t InDemandedMask = (Mask << ShAmt) & TypeMask;
      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= MVT::getSizeInBits(VT) - ShAmt;
      if (HighBits & Mask)
        InDemandedMask |= MVT::getIntVTSignBit(VT);
      
      ComputeMaskedBits(Op.getOperand(0), InDemandedMask, KnownZero, KnownOne,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= ShAmt;
      KnownOne  >>= ShAmt;
      
      // Handle the sign bits.
      uint64_t SignBit = MVT::getIntVTSignBit(VT);
      SignBit >>= ShAmt;  // Adjust to where it is now in the mask.
      
      if (KnownZero & SignBit) {       
        KnownZero |= HighBits;  // New bits are known zero.
      } else if (KnownOne & SignBit) {
        KnownOne  |= HighBits;  // New bits are known one.
      }
    }
    return;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    
    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    uint64_t NewBits = ~MVT::getIntVTBitMask(EVT) & Mask;

    uint64_t InSignBit = MVT::getIntVTSignBit(EVT);
    int64_t InputDemandedBits = Mask & MVT::getIntVTBitMask(EVT);
    
    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if (NewBits)
      InputDemandedBits |= InSignBit;
    
    ComputeMaskedBits(Op.getOperand(0), InputDemandedBits,
                      KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero & InSignBit) {          // Input sign bit known clear
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne & InSignBit) {    // Input sign bit known set
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
    MVT::ValueType VT = Op.getValueType();
    unsigned LowBits = Log2_32(MVT::getSizeInBits(VT))+1;
    KnownZero = ~((1ULL << LowBits)-1) & MVT::getIntVTBitMask(VT);
    KnownOne  = 0;
    return;
  }
  case ISD::LOAD: {
    if (ISD::isZEXTLoad(Op.Val)) {
      LoadSDNode *LD = cast<LoadSDNode>(Op);
      MVT::ValueType VT = LD->getLoadedVT();
      KnownZero |= ~MVT::getIntVTBitMask(VT) & Mask;
    }
    return;
  }
  case ISD::ZERO_EXTEND: {
    uint64_t InMask  = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    uint64_t NewBits = (~InMask) & Mask;
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero, 
                      KnownOne, Depth+1);
    KnownZero |= NewBits & Mask;
    KnownOne  &= ~NewBits;
    return;
  }
  case ISD::SIGN_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits    = MVT::getSizeInBits(InVT);
    uint64_t InMask    = MVT::getIntVTBitMask(InVT);
    uint64_t InSignBit = 1ULL << (InBits-1);
    uint64_t NewBits   = (~InMask) & Mask;
    uint64_t InDemandedBits = Mask & InMask;

    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if (NewBits & Mask)
      InDemandedBits |= InSignBit;
    
    ComputeMaskedBits(Op.getOperand(0), InDemandedBits, KnownZero, 
                      KnownOne, Depth+1);
    // If the sign bit is known zero or one, the  top bits match.
    if (KnownZero & InSignBit) {
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne & InSignBit) {
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {   // Otherwise, top bits aren't known.
      KnownOne  &= ~NewBits;
      KnownZero &= ~NewBits;
    }
    return;
  }
  case ISD::ANY_EXTEND: {
    MVT::ValueType VT = Op.getOperand(0).getValueType();
    ComputeMaskedBits(Op.getOperand(0), Mask & MVT::getIntVTBitMask(VT),
                      KnownZero, KnownOne, Depth+1);
    return;
  }
  case ISD::TRUNCATE: {
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    uint64_t OutMask = MVT::getIntVTBitMask(Op.getValueType());
    KnownZero &= OutMask;
    KnownOne &= OutMask;
    break;
  }
  case ISD::AssertZext: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    uint64_t InMask = MVT::getIntVTBitMask(VT);
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero, 
                      KnownOne, Depth+1);
    KnownZero |= (~InMask) & Mask;
    return;
  }
  case ISD::ADD: {
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    uint64_t KnownZeroOut = std::min(CountTrailingZeros_64(~KnownZero), 
                                     CountTrailingZeros_64(~KnownZero2));
    
    KnownZero = (1ULL << KnownZeroOut) - 1;
    KnownOne = 0;
    return;
  }
  case ISD::SUB: {
    ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0));
    if (!CLHS) return;

    // We know that the top bits of C-X are clear if X contains less bits
    // than C (i.e. no wrap-around can happen).  For example, 20-X is
    // positive if we can prove that X is >= 0 and < 16.
    MVT::ValueType VT = CLHS->getValueType(0);
    if ((CLHS->getValue() & MVT::getIntVTSignBit(VT)) == 0) {  // sign bit clear
      unsigned NLZ = CountLeadingZeros_64(CLHS->getValue()+1);
      uint64_t MaskV = (1ULL << (63-NLZ))-1; // NLZ can't be 64 with no sign bit
      MaskV = ~MaskV & MVT::getIntVTBitMask(VT);
      ComputeMaskedBits(Op.getOperand(1), MaskV, KnownZero, KnownOne, Depth+1);

      // If all of the MaskV bits are known to be zero, then we know the output
      // top bits are zero, because we now know that the output is from [0-C].
      if ((KnownZero & MaskV) == MaskV) {
        unsigned NLZ2 = CountLeadingZeros_64(CLHS->getValue());
        KnownZero = ~((1ULL << (64-NLZ2))-1) & Mask;  // Top bits known zero.
        KnownOne = 0;   // No one bits known.
      } else {
        KnownZero = KnownOne = 0;  // Otherwise, nothing known.
      }
    }
    return;
  }
  default:
    // Allow the target to implement this method for its nodes.
    if (Op.getOpcode() >= ISD::BUILTIN_OP_END) {
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
      TLI.computeMaskedBitsForTargetNode(Op, Mask, KnownZero, KnownOne, *this);
    }
    return;
  }
}

/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "SRA X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
unsigned SelectionDAG::ComputeNumSignBits(SDOperand Op, unsigned Depth) const{
  MVT::ValueType VT = Op.getValueType();
  assert(MVT::isInteger(VT) && "Invalid VT!");
  unsigned VTBits = MVT::getSizeInBits(VT);
  unsigned Tmp, Tmp2;
  
  if (Depth == 6)
    return 1;  // Limit search depth.

  switch (Op.getOpcode()) {
  default: break;
  case ISD::AssertSext:
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return VTBits-Tmp+1;
  case ISD::AssertZext:
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return VTBits-Tmp;
    
  case ISD::Constant: {
    uint64_t Val = cast<ConstantSDNode>(Op)->getValue();
    // If negative, invert the bits, then look at it.
    if (Val & MVT::getIntVTSignBit(VT))
      Val = ~Val;
    
    // Shift the bits so they are the leading bits in the int64_t.
    Val <<= 64-VTBits;
    
    // Return # leading zeros.  We use 'min' here in case Val was zero before
    // shifting.  We don't want to return '64' as for an i32 "0".
    return std::min(VTBits, CountLeadingZeros_64(Val));
  }
    
  case ISD::SIGN_EXTEND:
    Tmp = VTBits-MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return ComputeNumSignBits(Op.getOperand(0), Depth+1) + Tmp;
    
  case ISD::SIGN_EXTEND_INREG:
    // Max of the input and what this extends.
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    Tmp = VTBits-Tmp+1;
    
    Tmp2 = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    return std::max(Tmp, Tmp2);

  case ISD::SRA:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    // SRA X, C   -> adds C sign bits.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Tmp += C->getValue();
      if (Tmp > VTBits) Tmp = VTBits;
    }
    return Tmp;
  case ISD::SHL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (C->getValue() >= VTBits ||      // Bad shift.
          C->getValue() >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - C->getValue();
    }
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    return std::min(Tmp, Tmp2);

  case ISD::SELECT:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    return std::min(Tmp, Tmp2);
    
  case ISD::SETCC:
    // If setcc returns 0/-1, all bits are sign bits.
    if (TLI.getSetCCResultContents() ==
        TargetLowering::ZeroOrNegativeOneSetCCResult)
      return VTBits;
    break;
  case ISD::ROTL:
  case ISD::ROTR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned RotAmt = C->getValue() & (VTBits-1);
      
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
    if (ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
      if (CRHS->isAllOnesValue()) {
        uint64_t KnownZero, KnownOne;
        uint64_t Mask = MVT::getIntVTBitMask(VT);
        ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
        
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero|1) == Mask)
          return VTBits;
        
        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero & MVT::getIntVTSignBit(VT))
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
      if (CLHS->getValue() == 0) {
        uint64_t KnownZero, KnownOne;
        uint64_t Mask = MVT::getIntVTBitMask(VT);
        ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero|1) == Mask)
          return VTBits;
        
        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero & MVT::getIntVTSignBit(VT))
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
      Tmp = MVT::getSizeInBits(LD->getLoadedVT());
      return VTBits-Tmp+1;
    case ISD::ZEXTLOAD:    // '16' bits known
      Tmp = MVT::getSizeInBits(LD->getLoadedVT());
      return VTBits-Tmp;
    }
  }

  // Allow the target to implement this method for its nodes.
  if (Op.getOpcode() >= ISD::BUILTIN_OP_END ||
      Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN || 
      Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_VOID) {
    unsigned NumBits = TLI.ComputeNumSignBitsForTargetNode(Op, Depth);
    if (NumBits > 1) return NumBits;
  }
  
  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  uint64_t KnownZero, KnownOne;
  uint64_t Mask = MVT::getIntVTBitMask(VT);
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  
  uint64_t SignBit = MVT::getIntVTSignBit(VT);
  if (KnownZero & SignBit) {        // SignBit is 0
    Mask = KnownZero;
  } else if (KnownOne & SignBit) {  // SignBit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return 1;
  }
  
  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask ^= ~0ULL;
  Mask <<= 64-VTBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::min(VTBits, CountLeadingZeros_64(Mask));
}


/// getNode - Gets or creates the specified node.
///
SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opcode, getVTList(VT), 0, 0);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SDNode(Opcode, SDNode::getSDVTList(VT));
  CSEMap.InsertNode(N, IP);
  
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand Operand) {
  unsigned Tmp1;
  // Constant fold unary operations with an integer constant operand.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand.Val)) {
    uint64_t Val = C->getValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND: return getConstant(C->getSignExtended(), VT);
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND: return getConstant(Val, VT);
    case ISD::TRUNCATE:    return getConstant(Val, VT);
    case ISD::SINT_TO_FP:  return getConstantFP(C->getSignExtended(), VT);
    case ISD::UINT_TO_FP:  return getConstantFP(C->getValue(), VT);
    case ISD::BIT_CONVERT:
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(BitsToFloat(Val), VT);
      else if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(BitsToDouble(Val), VT);
      break;
    case ISD::BSWAP:
      switch(VT) {
      default: assert(0 && "Invalid bswap!"); break;
      case MVT::i16: return getConstant(ByteSwap_16((unsigned short)Val), VT);
      case MVT::i32: return getConstant(ByteSwap_32((unsigned)Val), VT);
      case MVT::i64: return getConstant(ByteSwap_64(Val), VT);
      }
      break;
    case ISD::CTPOP:
      switch(VT) {
      default: assert(0 && "Invalid ctpop!"); break;
      case MVT::i1: return getConstant(Val != 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val & 0xFF;
        return getConstant(CountPopulation_32(Tmp1), VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val & 0xFFFF;
        return getConstant(CountPopulation_32(Tmp1), VT);
      case MVT::i32:
        return getConstant(CountPopulation_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountPopulation_64(Val), VT);
      }
    case ISD::CTLZ:
      switch(VT) {
      default: assert(0 && "Invalid ctlz!"); break;
      case MVT::i1: return getConstant(Val == 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val & 0xFF;
        return getConstant(CountLeadingZeros_32(Tmp1)-24, VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val & 0xFFFF;
        return getConstant(CountLeadingZeros_32(Tmp1)-16, VT);
      case MVT::i32:
        return getConstant(CountLeadingZeros_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountLeadingZeros_64(Val), VT);
      }
    case ISD::CTTZ:
      switch(VT) {
      default: assert(0 && "Invalid cttz!"); break;
      case MVT::i1: return getConstant(Val == 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val | 0x100;
        return getConstant(CountTrailingZeros_32(Tmp1), VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val | 0x10000;
        return getConstant(CountTrailingZeros_32(Tmp1), VT);
      case MVT::i32:
        return getConstant(CountTrailingZeros_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountTrailingZeros_64(Val), VT);
      }
    }
  }

  // Constant fold unary operations with a floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.Val)) {
    APFloat V = C->getValueAPF();    // make copy
    switch (Opcode) {
    case ISD::FNEG:
      V.changeSign();
      return getConstantFP(V, VT);
    case ISD::FABS:
      V.clearSign();
      return getConstantFP(V, VT);
    case ISD::FP_ROUND:
    case ISD::FP_EXTEND:
      // This can return overflow, underflow, or inexact; we don't care.
      // FIXME need to be more flexible about rounding mode.
      (void) V.convert(VT==MVT::f32 ? APFloat::IEEEsingle : 
                                      APFloat::IEEEdouble,
                       APFloat::rmNearestTiesToEven);
      return getConstantFP(V, VT);
    case ISD::FP_TO_SINT:
    case ISD::FP_TO_UINT: {
      integerPart x;
      assert(integerPartWidth >= 64);
      // FIXME need to be more flexible about rounding mode.
      APFloat::opStatus s = V.convertToInteger(&x, 64U,
                            Opcode==ISD::FP_TO_SINT,
                            APFloat::rmTowardZero);
      if (s==APFloat::opInvalidOp)     // inexact is OK, in fact usual
        break;
      return getConstant(x, VT);
    }
    case ISD::BIT_CONVERT:
      if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
        return getConstant((uint32_t)V.convertToAPInt().getZExtValue(), VT);
      else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
        return getConstant(V.convertToAPInt().getZExtValue(), VT);
      break;
    }
  }

  unsigned OpOpcode = Operand.Val->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
    return Operand;         // Factor of one node?  No factor.
  case ISD::FP_ROUND:
  case ISD::FP_EXTEND:
    assert(MVT::isFloatingPoint(VT) &&
           MVT::isFloatingPoint(Operand.getValueType()) && "Invalid FP cast!");
    break;
  case ISD::SIGN_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid SIGN_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid sext node, dst < src!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ZERO_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid ZERO_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid zext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ANY_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid ANY_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid anyext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::TRUNCATE:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid TRUNCATE!");
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert(Operand.getValueType() > VT && "Invalid truncate node, src < dst!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
    else if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
             OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.Val->getOperand(0).getValueType() < VT)
        return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
      else if (Operand.Val->getOperand(0).getValueType() > VT)
        return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
      else
        return Operand.Val->getOperand(0);
    }
    break;
  case ISD::BIT_CONVERT:
    // Basic sanity checking.
    assert(MVT::getSizeInBits(VT) == MVT::getSizeInBits(Operand.getValueType())
           && "Cannot BIT_CONVERT between types of different sizes!");
    if (VT == Operand.getValueType()) return Operand;  // noop conversion.
    if (OpOpcode == ISD::BIT_CONVERT)  // bitconv(bitconv(x)) -> bitconv(x)
      return getNode(ISD::BIT_CONVERT, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
    break;
  case ISD::SCALAR_TO_VECTOR:
    assert(MVT::isVector(VT) && !MVT::isVector(Operand.getValueType()) &&
           MVT::getVectorElementType(VT) == Operand.getValueType() &&
           "Illegal SCALAR_TO_VECTOR node!");
    break;
  case ISD::FNEG:
    if (OpOpcode == ISD::FSUB)   // -(X-Y) -> (Y-X)
      return getNode(ISD::FSUB, VT, Operand.Val->getOperand(1),
                     Operand.Val->getOperand(0));
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.Val->getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, VT, Operand.Val->getOperand(0));
    break;
  }

  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) { // Don't CSE flag producing nodes
    FoldingSetNodeID ID;
    SDOperand Ops[1] = { Operand };
    AddNodeIDNode(ID, Opcode, VTs, Ops, 1);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new UnarySDNode(Opcode, VTs, Operand);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new UnarySDNode(Opcode, VTs, Operand);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2) {
#ifndef NDEBUG
  switch (Opcode) {
  case ISD::TokenFactor:
    assert(VT == MVT::Other && N1.getValueType() == MVT::Other &&
           N2.getValueType() == MVT::Other && "Invalid token factor!");
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::MULHU:
  case ISD::MULHS:
    assert(MVT::isInteger(VT) && "This operator does not apply to FP types!");
    // fall through
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::SREM:
    assert(MVT::isInteger(N1.getValueType()) && "Should use F* for FP ops");
    // fall through.
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FCOPYSIGN:   // N1 and result must match.  N1/N2 need not match.
    assert(N1.getValueType() == VT &&
           MVT::isFloatingPoint(N1.getValueType()) && 
           MVT::isFloatingPoint(N2.getValueType()) &&
           "Invalid FCOPYSIGN!");
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
    assert(VT == N1.getValueType() &&
           "Shift operators return type must be the same as their first arg");
    assert(MVT::isInteger(VT) && MVT::isInteger(N2.getValueType()) &&
           VT != MVT::i1 && "Shifts only work on integers");
    break;
  case ISD::FP_ROUND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg round!");
    assert(MVT::isFloatingPoint(VT) && MVT::isFloatingPoint(EVT) &&
           "Cannot FP_ROUND_INREG integer types");
    assert(EVT <= VT && "Not rounding down!");
    break;
  }
  case ISD::AssertSext:
  case ISD::AssertZext:
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(MVT::isInteger(VT) && MVT::isInteger(EVT) &&
           "Cannot *_EXTEND_INREG FP types");
    assert(EVT <= VT && "Not extending!");
  }

  default: break;
  }
#endif

  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  if (N1C) {
    if (Opcode == ISD::SIGN_EXTEND_INREG) {
      int64_t Val = N1C->getValue();
      unsigned FromBits = MVT::getSizeInBits(cast<VTSDNode>(N2)->getVT());
      Val <<= 64-FromBits;
      Val >>= 64-FromBits;
      return getConstant(Val, VT);
    }
    
    if (N2C) {
      uint64_t C1 = N1C->getValue(), C2 = N2C->getValue();
      switch (Opcode) {
      case ISD::ADD: return getConstant(C1 + C2, VT);
      case ISD::SUB: return getConstant(C1 - C2, VT);
      case ISD::MUL: return getConstant(C1 * C2, VT);
      case ISD::UDIV:
        if (C2) return getConstant(C1 / C2, VT);
        break;
      case ISD::UREM :
        if (C2) return getConstant(C1 % C2, VT);
        break;
      case ISD::SDIV :
        if (C2) return getConstant(N1C->getSignExtended() /
                                   N2C->getSignExtended(), VT);
        break;
      case ISD::SREM :
        if (C2) return getConstant(N1C->getSignExtended() %
                                   N2C->getSignExtended(), VT);
        break;
      case ISD::AND  : return getConstant(C1 & C2, VT);
      case ISD::OR   : return getConstant(C1 | C2, VT);
      case ISD::XOR  : return getConstant(C1 ^ C2, VT);
      case ISD::SHL  : return getConstant(C1 << C2, VT);
      case ISD::SRL  : return getConstant(C1 >> C2, VT);
      case ISD::SRA  : return getConstant(N1C->getSignExtended() >>(int)C2, VT);
      case ISD::ROTL : 
        return getConstant((C1 << C2) | (C1 >> (MVT::getSizeInBits(VT) - C2)),
                           VT);
      case ISD::ROTR : 
        return getConstant((C1 >> C2) | (C1 << (MVT::getSizeInBits(VT) - C2)), 
                           VT);
      default: break;
      }
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }
  }

  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2.Val);
  if (N1CFP) {
    if (N2CFP) {
      APFloat V1 = N1CFP->getValueAPF(), V2 = N2CFP->getValueAPF();
      APFloat::opStatus s;
      switch (Opcode) {
      case ISD::FADD: 
        s = V1.add(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp)
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
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1CFP, N2CFP);
        std::swap(N1, N2);
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
        if (!MVT::isVector(VT)) 
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
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::SUB:
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::FDIV:
    case ISD::FREM:
    case ISD::UDIV:
    case ISD::SDIV:
    case ISD::UREM:
    case ISD::SREM:
    case ISD::XOR:
      return N2;       // fold op(arg1, undef) -> undef
    case ISD::MUL: 
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      if (!MVT::isVector(VT)) 
        return getConstant(0, VT);  // fold op(arg1, undef) -> 0
      // For vectors, we can't easily build an all zero vector, just return
      // the LHS.
      return N1;
    case ISD::OR:
      if (!MVT::isVector(VT)) 
        return getConstant(MVT::getIntVTBitMask(VT), VT);
      // For vectors, we can't easily build an all one vector, just return
      // the LHS.
      return N1;
    case ISD::SRA:
      return N1;
    }
  }

  // Fold operations.
  switch (Opcode) {
  case ISD::TokenFactor:
    // Fold trivial token factors.
    if (N1.getOpcode() == ISD::EntryToken) return N2;
    if (N2.getOpcode() == ISD::EntryToken) return N1;
    break;
      
  case ISD::AND:
    // (X & 0) -> 0.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->getValue() == 0)
      return N2;
    break;
  case ISD::OR:
  case ISD::XOR:
    // (X ^| 0) -> X.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->getValue() == 0)
      return N1;
    break;
  case ISD::FP_ROUND_INREG:
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    if (EVT == VT) return N1;  // Not actually extending
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    assert(N2C && "Bad EXTRACT_VECTOR_ELT!");

    // EXTRACT_VECTOR_ELT of CONCAT_VECTORS is often formed while lowering is
    // expanding copies of large vectors from registers.
    if (N1.getOpcode() == ISD::CONCAT_VECTORS &&
        N1.getNumOperands() > 0) {
      unsigned Factor =
        MVT::getVectorNumElements(N1.getOperand(0).getValueType());
      return getNode(ISD::EXTRACT_VECTOR_ELT, VT,
                     N1.getOperand(N2C->getValue() / Factor),
                     getConstant(N2C->getValue() % Factor, N2.getValueType()));
    }

    // EXTRACT_VECTOR_ELT of BUILD_VECTOR is often formed while lowering is
    // expanding large vector constants.
    if (N1.getOpcode() == ISD::BUILD_VECTOR)
      return N1.getOperand(N2C->getValue());

    // EXTRACT_VECTOR_ELT of INSERT_VECTOR_ELT is often formed when vector
    // operations are lowered to scalars.
    if (N1.getOpcode() == ISD::INSERT_VECTOR_ELT)
      if (ConstantSDNode *IEC = dyn_cast<ConstantSDNode>(N1.getOperand(2))) {
        if (IEC == N2C)
          return N1.getOperand(1);
        else
          return getNode(ISD::EXTRACT_VECTOR_ELT, VT, N1.getOperand(0), N2);
      }
    break;
  case ISD::EXTRACT_ELEMENT:
    assert(N2C && (unsigned)N2C->getValue() < 2 && "Bad EXTRACT_ELEMENT!");
    
    // EXTRACT_ELEMENT of BUILD_PAIR is often formed while legalize is expanding
    // 64-bit integers into 32-bit parts.  Instead of building the extract of
    // the BUILD_PAIR, only to have legalize rip it apart, just do it now. 
    if (N1.getOpcode() == ISD::BUILD_PAIR)
      return N1.getOperand(N2C->getValue());
    
    // EXTRACT_ELEMENT of a constant int is also very common.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      unsigned Shift = MVT::getSizeInBits(VT) * N2C->getValue();
      return getConstant(C->getValue() >> Shift, VT);
    }
    break;

  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
#if 0
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    if (N2.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N2.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, VT, N1, N2.getOperand(0));
    else if (N2.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N2.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = MVT::getSizeInBits(VT);
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, VT, N1, N2.getOperand(0));
      }
    break;
#endif
  }

  // Memoize this node if possible.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) {
    SDOperand Ops[] = { N1, N2 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 2);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new BinarySDNode(Opcode, VTs, N1, N2);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new BinarySDNode(Opcode, VTs, N1, N2);
  }

  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  switch (Opcode) {
  case ISD::SETCC: {
    // Use FoldSetCC to simplify SETCC's.
    SDOperand Simp = FoldSetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get());
    if (Simp.Val) return Simp;
    break;
  }
  case ISD::SELECT:
    if (N1C)
      if (N1C->getValue())
        return N2;             // select true, X, Y -> X
      else
        return N3;             // select false, X, Y -> Y

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::BRCOND:
    if (N2C)
      if (N2C->getValue()) // Unconditional branch
        return getNode(ISD::BR, MVT::Other, N1, N3);
      else
        return N1;         // Never-taken branch
    break;
  case ISD::VECTOR_SHUFFLE:
    assert(VT == N1.getValueType() && VT == N2.getValueType() &&
           MVT::isVector(VT) && MVT::isVector(N3.getValueType()) &&
           N3.getOpcode() == ISD::BUILD_VECTOR &&
           MVT::getVectorNumElements(VT) == N3.getNumOperands() &&
           "Illegal VECTOR_SHUFFLE node!");
    break;
  case ISD::BIT_CONVERT:
    // Fold bit_convert nodes from a type to themselves.
    if (N1.getValueType() == VT)
      return N1;
    break;
  }

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) {
    SDOperand Ops[] = { N1, N2, N3 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 3);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new TernarySDNode(Opcode, VTs, N1, N2, N3);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new TernarySDNode(Opcode, VTs, N1, N2, N3);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4) {
  SDOperand Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, VT, Ops, 4);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4, SDOperand N5) {
  SDOperand Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, VT, Ops, 5);
}

SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr,
                                const Value *SV, int SVOffset,
                                bool isVolatile, unsigned Alignment) {
  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for load must be a pointer");
      Ty = PT->getElementType();
    }  
    assert(Ty && "Could not get type information for load");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(VT, MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops, 3);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(ISD::NON_EXTLOAD);
  ID.AddInteger((unsigned int)VT);
  ID.AddPointer(SV);
  ID.AddInteger(SVOffset);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new LoadSDNode(Ops, VTs, ISD::UNINDEXED,
                             ISD::NON_EXTLOAD, VT, SV, SVOffset, Alignment,
                             isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, MVT::ValueType VT,
                                   SDOperand Chain, SDOperand Ptr,
                                   const Value *SV,
                                   int SVOffset, MVT::ValueType EVT,
                                   bool isVolatile, unsigned Alignment) {
  // If they are asking for an extending load from/to the same thing, return a
  // normal load.
  if (VT == EVT)
    ExtType = ISD::NON_EXTLOAD;

  if (MVT::isVector(VT))
    assert(EVT == MVT::getVectorElementType(VT) && "Invalid vector extload!");
  else
    assert(EVT < VT && "Should only be an extending load, not truncating!");
  assert((ExtType == ISD::EXTLOAD || MVT::isInteger(VT)) &&
         "Cannot sign/zero extend a FP/Vector load!");
  assert(MVT::isInteger(VT) == MVT::isInteger(EVT) &&
         "Cannot convert from FP to Int or Int -> FP!");

  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for load must be a pointer");
      Ty = PT->getElementType();
    }  
    assert(Ty && "Could not get type information for load");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(VT, MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops, 3);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(ExtType);
  ID.AddInteger((unsigned int)EVT);
  ID.AddPointer(SV);
  ID.AddInteger(SVOffset);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new LoadSDNode(Ops, VTs, ISD::UNINDEXED, ExtType, EVT,
                             SV, SVOffset, Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand
SelectionDAG::getIndexedLoad(SDOperand OrigLoad, SDOperand Base,
                             SDOperand Offset, ISD::MemIndexedMode AM) {
  LoadSDNode *LD = cast<LoadSDNode>(OrigLoad);
  assert(LD->getOffset().getOpcode() == ISD::UNDEF &&
         "Load is already a indexed load!");
  MVT::ValueType VT = OrigLoad.getValueType();
  SDVTList VTs = getVTList(VT, Base.getValueType(), MVT::Other);
  SDOperand Ops[] = { LD->getChain(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops, 3);
  ID.AddInteger(AM);
  ID.AddInteger(LD->getExtensionType());
  ID.AddInteger((unsigned int)(LD->getLoadedVT()));
  ID.AddPointer(LD->getSrcValue());
  ID.AddInteger(LD->getSrcValueOffset());
  ID.AddInteger(LD->getAlignment());
  ID.AddInteger(LD->isVolatile());
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new LoadSDNode(Ops, VTs, AM,
                             LD->getExtensionType(), LD->getLoadedVT(),
                             LD->getSrcValue(), LD->getSrcValueOffset(),
                             LD->getAlignment(), LD->isVolatile());
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getStore(SDOperand Chain, SDOperand Val,
                                 SDOperand Ptr, const Value *SV, int SVOffset,
                                 bool isVolatile, unsigned Alignment) {
  MVT::ValueType VT = Val.getValueType();

  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for store must be a pointer");
      Ty = PT->getElementType();
    }
    assert(Ty && "Could not get type information for store");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(false);
  ID.AddInteger((unsigned int)VT);
  ID.AddPointer(SV);
  ID.AddInteger(SVOffset);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, ISD::UNINDEXED, false,
                              VT, SV, SVOffset, Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTruncStore(SDOperand Chain, SDOperand Val,
                                      SDOperand Ptr, const Value *SV,
                                      int SVOffset, MVT::ValueType SVT,
                                      bool isVolatile, unsigned Alignment) {
  MVT::ValueType VT = Val.getValueType();
  bool isTrunc = VT != SVT;

  assert(VT > SVT && "Not a truncation?");
  assert(MVT::isInteger(VT) == MVT::isInteger(SVT) &&
         "Can't do FP-INT conversion!");

  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for store must be a pointer");
      Ty = PT->getElementType();
    }
    assert(Ty && "Could not get type information for store");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(isTrunc);
  ID.AddInteger((unsigned int)SVT);
  ID.AddPointer(SV);
  ID.AddInteger(SVOffset);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, ISD::UNINDEXED, isTrunc,
                              SVT, SV, SVOffset, Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand
SelectionDAG::getIndexedStore(SDOperand OrigStore, SDOperand Base,
                              SDOperand Offset, ISD::MemIndexedMode AM) {
  StoreSDNode *ST = cast<StoreSDNode>(OrigStore);
  assert(ST->getOffset().getOpcode() == ISD::UNDEF &&
         "Store is already a indexed store!");
  SDVTList VTs = getVTList(Base.getValueType(), MVT::Other);
  SDOperand Ops[] = { ST->getChain(), ST->getValue(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(AM);
  ID.AddInteger(ST->isTruncatingStore());
  ID.AddInteger((unsigned int)(ST->getStoredVT()));
  ID.AddPointer(ST->getSrcValue());
  ID.AddInteger(ST->getSrcValueOffset());
  ID.AddInteger(ST->getAlignment());
  ID.AddInteger(ST->isVolatile());
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, AM,
                              ST->isTruncatingStore(), ST->getStoredVT(),
                              ST->getSrcValue(), ST->getSrcValueOffset(),
                              ST->getAlignment(), ST->isVolatile());
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getVAArg(MVT::ValueType VT,
                                 SDOperand Chain, SDOperand Ptr,
                                 SDOperand SV) {
  SDOperand Ops[] = { Chain, Ptr, SV };
  return getNode(ISD::VAARG, getVTList(VT, MVT::Other), Ops, 3);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                const SDOperand *Ops, unsigned NumOps) {
  switch (NumOps) {
  case 0: return getNode(Opcode, VT);
  case 1: return getNode(Opcode, VT, Ops[0]);
  case 2: return getNode(Opcode, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, VT, Ops[0], Ops[1], Ops[2]);
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
  if (VT != MVT::Flag) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, VTs, Ops, NumOps);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, VTs, Ops, NumOps);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                std::vector<MVT::ValueType> &ResultTys,
                                const SDOperand *Ops, unsigned NumOps) {
  return getNode(Opcode, getNodeValueTypes(ResultTys), ResultTys.size(),
                 Ops, NumOps);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                const MVT::ValueType *VTs, unsigned NumVTs,
                                const SDOperand *Ops, unsigned NumOps) {
  if (NumVTs == 1)
    return getNode(Opcode, VTs[0], Ops, NumOps);
  return getNode(Opcode, makeVTList(VTs, NumVTs), Ops, NumOps);
}  
  
SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                const SDOperand *Ops, unsigned NumOps) {
  if (VTList.NumVTs == 1)
    return getNode(Opcode, VTList.VTs[0], Ops, NumOps);

  switch (Opcode) {
  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
#if 0
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SHL_PARTS:
    if (N3.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N3.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, VT, N1, N2, N3.getOperand(0));
    else if (N3.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N3.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = MVT::getSizeInBits(VT)*2;
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, VT, N1, N2, N3.getOperand(0));
      }
    break;
#endif
  }

  // Memoize the node unless it returns a flag.
  SDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Flag) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    if (NumOps == 1)
      N = new UnarySDNode(Opcode, VTList, Ops[0]);
    else if (NumOps == 2)
      N = new BinarySDNode(Opcode, VTList, Ops[0], Ops[1]);
    else if (NumOps == 3)
      N = new TernarySDNode(Opcode, VTList, Ops[0], Ops[1], Ops[2]);
    else
      N = new SDNode(Opcode, VTList, Ops, NumOps);
    CSEMap.InsertNode(N, IP);
  } else {
    if (NumOps == 1)
      N = new UnarySDNode(Opcode, VTList, Ops[0]);
    else if (NumOps == 2)
      N = new BinarySDNode(Opcode, VTList, Ops[0], Ops[1]);
    else if (NumOps == 3)
      N = new TernarySDNode(Opcode, VTList, Ops[0], Ops[1], Ops[2]);
    else
      N = new SDNode(Opcode, VTList, Ops, NumOps);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDVTList SelectionDAG::getVTList(MVT::ValueType VT) {
  if (!MVT::isExtendedVT(VT))
    return makeVTList(SDNode::getValueTypeList(VT), 1);

  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 1 && (*I)[0] == VT)
      return makeVTList(&(*I)[0], 1);
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT);
  VTList.push_front(V);
  return makeVTList(&(*VTList.begin())[0], 1);
}

SDVTList SelectionDAG::getVTList(MVT::ValueType VT1, MVT::ValueType VT2) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 2 && (*I)[0] == VT1 && (*I)[1] == VT2)
      return makeVTList(&(*I)[0], 2);
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  VTList.push_front(V);
  return makeVTList(&(*VTList.begin())[0], 2);
}
SDVTList SelectionDAG::getVTList(MVT::ValueType VT1, MVT::ValueType VT2,
                                 MVT::ValueType VT3) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 3 && (*I)[0] == VT1 && (*I)[1] == VT2 &&
        (*I)[2] == VT3)
      return makeVTList(&(*I)[0], 3);
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  V.push_back(VT3);
  VTList.push_front(V);
  return makeVTList(&(*VTList.begin())[0], 3);
}

SDVTList SelectionDAG::getVTList(const MVT::ValueType *VTs, unsigned NumVTs) {
  switch (NumVTs) {
    case 0: assert(0 && "Cannot have nodes without results!");
    case 1: return getVTList(VTs[0]);
    case 2: return getVTList(VTs[0], VTs[1]);
    case 3: return getVTList(VTs[0], VTs[1], VTs[2]);
    default: break;
  }

  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() != NumVTs || VTs[0] != (*I)[0] || VTs[1] != (*I)[1]) continue;
   
    bool NoMatch = false;
    for (unsigned i = 2; i != NumVTs; ++i)
      if (VTs[i] != (*I)[i]) {
        NoMatch = true;
        break;
      }
    if (!NoMatch)
      return makeVTList(&*I->begin(), NumVTs);
  }
  
  VTList.push_front(std::vector<MVT::ValueType>(VTs, VTs+NumVTs));
  return makeVTList(&*VTList.begin()->begin(), NumVTs);
}


/// UpdateNodeOperands - *Mutate* the specified node in-place to have the
/// specified operands.  If the resultant node already exists in the DAG,
/// this does not modify the specified node, instead it returns the node that
/// already exists.  If the resultant node does not exist in the DAG, the
/// input node is returned.  As a degenerate case, if you specify the same
/// input operands as the node already has, the input node is returned.
SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand Op) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == 1 && "Update with wrong number of operands");
  
  // Check to see if there is no change.
  if (Op == N->getOperand(0)) return InN;
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  N->OperandList[0].Val->removeUser(N);
  Op.Val->addUser(N);
  N->OperandList[0] = Op;
  
  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand Op1, SDOperand Op2) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == 2 && "Update with wrong number of operands");
  
  // Check to see if there is no change.
  if (Op1 == N->getOperand(0) && Op2 == N->getOperand(1))
    return InN;   // No operands changed, just return the input node.
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op1, Op2, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  if (N->OperandList[0] != Op1) {
    N->OperandList[0].Val->removeUser(N);
    Op1.Val->addUser(N);
    N->OperandList[0] = Op1;
  }
  if (N->OperandList[1] != Op2) {
    N->OperandList[1].Val->removeUser(N);
    Op2.Val->addUser(N);
    N->OperandList[1] = Op2;
  }
  
  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2, SDOperand Op3) {
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return UpdateNodeOperands(N, Ops, 3);
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2, 
                   SDOperand Op3, SDOperand Op4) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4 };
  return UpdateNodeOperands(N, Ops, 4);
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2,
                   SDOperand Op3, SDOperand Op4, SDOperand Op5) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return UpdateNodeOperands(N, Ops, 5);
}


SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand *Ops, unsigned NumOps) {
  SDNode *N = InN.Val;
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
  if (!AnyChange) return InN;
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Ops, NumOps, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  for (unsigned i = 0; i != NumOps; ++i) {
    if (N->OperandList[i] != Ops[i]) {
      N->OperandList[i].Val->removeUser(N);
      Ops[i].Val->addUser(N);
      N->OperandList[i] = Ops[i];
    }
  }

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}


/// MorphNodeTo - This frees the operands of the current node, resets the
/// opcode, types, and operands to the specified value.  This should only be
/// used by the SelectionDAG class.
void SDNode::MorphNodeTo(unsigned Opc, SDVTList L,
                         const SDOperand *Ops, unsigned NumOps) {
  NodeType = Opc;
  ValueList = L.VTs;
  NumValues = L.NumVTs;
  
  // Clear the operands list, updating used nodes to remove this from their
  // use list.
  for (op_iterator I = op_begin(), E = op_end(); I != E; ++I)
    I->Val->removeUser(this);
  
  // If NumOps is larger than the # of operands we currently have, reallocate
  // the operand list.
  if (NumOps > NumOperands) {
    if (OperandsNeedDelete)
      delete [] OperandList;
    OperandList = new SDOperand[NumOps];
    OperandsNeedDelete = true;
  }
  
  // Assign the new operands.
  NumOperands = NumOps;
  
  for (unsigned i = 0, e = NumOps; i != e; ++i) {
    OperandList[i] = Ops[i];
    SDNode *N = OperandList[i].Val;
    N->Uses.push_back(this);
  }
}

/// SelectNodeTo - These are used for target selectors to *mutate* the
/// specified node to have the specified return type, Target opcode, and
/// operands.  Note that target opcodes are stored as
/// ISD::BUILTIN_OP_END+TargetOpcode in the node opcode field.
///
/// Note that SelectNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT) {
  SDVTList VTs = getVTList(VT);
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, 0, 0);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
   
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, 0, 0);

  CSEMap.InsertNode(N, IP);
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1 };
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 1);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 1);
  CSEMap.InsertNode(N, IP);
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1,
                                   SDOperand Op2) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1, Op2 };
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1,
                                   SDOperand Op2, SDOperand Op3) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);

  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, const SDOperand *Ops,
                                   unsigned NumOps) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, NumOps);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, NumOps);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc, 
                                   MVT::ValueType VT1, MVT::ValueType VT2,
                                   SDOperand Op1, SDOperand Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  FoldingSetNodeID ID;
  SDOperand Ops[] = { Op1, Op2 };
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;

  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT1, MVT::ValueType VT2,
                                   SDOperand Op1, SDOperand Op2, 
                                   SDOperand Op3) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;

  RemoveNodeFromCSEMaps(N);

  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}


/// getTargetNode - These are used for target selectors to create a new node
/// with specified return type(s), target opcode, and operands.
///
/// Note that getTargetNode returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    const SDOperand *Ops, unsigned NumOps) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, &Op1, 1).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, 3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2,
                                    const SDOperand *Ops, unsigned NumOps) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, 3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    const SDOperand *Ops, unsigned NumOps) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    MVT::ValueType VT4,
                                    const SDOperand *Ops, unsigned NumOps) {
  std::vector<MVT::ValueType> VTList;
  VTList.push_back(VT1);
  VTList.push_back(VT2);
  VTList.push_back(VT3);
  VTList.push_back(VT4);
  const MVT::ValueType *VTs = getNodeValueTypes(VTList);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 4, Ops, NumOps).Val;
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From/To have a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDOperand FromN, SDOperand ToN,
                                      std::vector<SDNode*> *Deleted) {
  SDNode *From = FromN.Val, *To = ToN.Val;
  assert(From->getNumValues() == 1 && To->getNumValues() == 1 &&
         "Cannot replace with this method!");
  assert(From != To && "Cannot replace uses of with self");
  
  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        From->removeUser(U);
        I->Val = To;
        To->addUser(U);
      }

    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
    }
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From/To have matching types and numbers of result
/// values.
///
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, SDNode *To,
                                      std::vector<SDNode*> *Deleted) {
  assert(From != To && "Cannot replace uses of with self");
  assert(From->getNumValues() == To->getNumValues() &&
         "Cannot use this version of ReplaceAllUsesWith!");
  if (From->getNumValues() == 1) {  // If possible, use the faster version.
    ReplaceAllUsesWith(SDOperand(From, 0), SDOperand(To, 0), Deleted);
    return;
  }
  
  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        From->removeUser(U);
        I->Val = To;
        To->addUser(U);
      }
        
    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
    }
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From,
                                      const SDOperand *To,
                                      std::vector<SDNode*> *Deleted) {
  if (From->getNumValues() == 1 && To[0].Val->getNumValues() == 1) {
    // Degenerate case handled above.
    ReplaceAllUsesWith(SDOperand(From, 0), To[0], Deleted);
    return;
  }

  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        const SDOperand &ToOp = To[I->ResNo];
        From->removeUser(U);
        *I = ToOp;
        ToOp.Val->addUser(U);
      }
        
    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
    }
  }
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.Val alone.  The Deleted vector is
/// handled the same was as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDOperand From, SDOperand To,
                                             std::vector<SDNode*> &Deleted) {
  assert(From != To && "Cannot replace a value with itself");
  // Handle the simple, trivial, case efficiently.
  if (From.Val->getNumValues() == 1 && To.Val->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To, &Deleted);
    return;
  }
  
  // Get all of the users of From.Val.  We want these in a nice,
  // deterministically ordered and uniqued set, so we use a SmallSetVector.
  SmallSetVector<SDNode*, 16> Users(From.Val->use_begin(), From.Val->use_end());

  while (!Users.empty()) {
    // We know that this user uses some value of From.  If it is the right
    // value, update it.
    SDNode *User = Users.back();
    Users.pop_back();
    
    for (SDOperand *Op = User->OperandList,
         *E = User->OperandList+User->NumOperands; Op != E; ++Op) {
      if (*Op == From) {
        // Okay, we know this user needs to be updated.  Remove its old self
        // from the CSE maps.
        RemoveNodeFromCSEMaps(User);
        
        // Update all operands that match "From".
        for (; Op != E; ++Op) {
          if (*Op == From) {
            From.Val->removeUser(User);
            *Op = To;
            To.Val->addUser(User);
          }
        }
                   
        // Now that we have modified User, add it back to the CSE maps.  If it
        // already exists there, recursively merge the results together.
        if (SDNode *Existing = AddNonLeafNodeToCSEMaps(User)) {
          unsigned NumDeleted = Deleted.size();
          ReplaceAllUsesWith(User, Existing, &Deleted);
          
          // User is now dead.
          Deleted.push_back(User);
          DeleteNodeNotInCSEMaps(User);
          
          // We have to be careful here, because ReplaceAllUsesWith could have
          // deleted a user of From, which means there may be dangling pointers
          // in the "Users" setvector.  Scan over the deleted node pointers and
          // remove them from the setvector.
          for (unsigned i = NumDeleted, e = Deleted.size(); i != e; ++i)
            Users.remove(Deleted[i]);
        }
        break;   // Exit the operand scanning loop.
      }
    }
  }
}


/// AssignNodeIds - Assign a unique node id for each node in the DAG based on
/// their allnodes order. It returns the maximum id.
unsigned SelectionDAG::AssignNodeIds() {
  unsigned Id = 0;
  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I){
    SDNode *N = I;
    N->setNodeId(Id++);
  }
  return Id;
}

/// AssignTopologicalOrder - Assign a unique node id for each node in the DAG
/// based on their topological order. It returns the maximum id and a vector
/// of the SDNodes* in assigned order by reference.
unsigned SelectionDAG::AssignTopologicalOrder(std::vector<SDNode*> &TopOrder) {
  unsigned DAGSize = AllNodes.size();
  std::vector<unsigned> InDegree(DAGSize);
  std::vector<SDNode*> Sources;

  // Use a two pass approach to avoid using a std::map which is slow.
  unsigned Id = 0;
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ++I){
    SDNode *N = I;
    N->setNodeId(Id++);
    unsigned Degree = N->use_size();
    InDegree[N->getNodeId()] = Degree;
    if (Degree == 0)
      Sources.push_back(N);
  }

  TopOrder.clear();
  while (!Sources.empty()) {
    SDNode *N = Sources.back();
    Sources.pop_back();
    TopOrder.push_back(N);
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *P = I->Val;
      unsigned Degree = --InDegree[P->getNodeId()];
      if (Degree == 0)
        Sources.push_back(P);
    }
  }

  // Second pass, assign the actual topological order as node ids.
  Id = 0;
  for (std::vector<SDNode*>::iterator TI = TopOrder.begin(),TE = TopOrder.end();
       TI != TE; ++TI)
    (*TI)->setNodeId(Id++);

  return Id;
}



//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//

// Out-of-line virtual method to give class a home.
void SDNode::ANCHOR() {}
void UnarySDNode::ANCHOR() {}
void BinarySDNode::ANCHOR() {}
void TernarySDNode::ANCHOR() {}
void HandleSDNode::ANCHOR() {}
void StringSDNode::ANCHOR() {}
void ConstantSDNode::ANCHOR() {}
void ConstantFPSDNode::ANCHOR() {}
void GlobalAddressSDNode::ANCHOR() {}
void FrameIndexSDNode::ANCHOR() {}
void JumpTableSDNode::ANCHOR() {}
void ConstantPoolSDNode::ANCHOR() {}
void BasicBlockSDNode::ANCHOR() {}
void SrcValueSDNode::ANCHOR() {}
void RegisterSDNode::ANCHOR() {}
void ExternalSymbolSDNode::ANCHOR() {}
void CondCodeSDNode::ANCHOR() {}
void VTSDNode::ANCHOR() {}
void LoadSDNode::ANCHOR() {}
void StoreSDNode::ANCHOR() {}

HandleSDNode::~HandleSDNode() {
  SDVTList VTs = { 0, 0 };
  MorphNodeTo(ISD::HANDLENODE, VTs, 0, 0);  // Drops operand uses.
}

GlobalAddressSDNode::GlobalAddressSDNode(bool isTarget, const GlobalValue *GA,
                                         MVT::ValueType VT, int o)
  : SDNode(isa<GlobalVariable>(GA) &&
           cast<GlobalVariable>(GA)->isThreadLocal() ?
           // Thread Local
           (isTarget ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress) :
           // Non Thread Local
           (isTarget ? ISD::TargetGlobalAddress : ISD::GlobalAddress),
           getSDVTList(VT)), Offset(o) {
  TheGlobal = const_cast<GlobalValue*>(GA);
}

/// Profile - Gather unique data for the node.
///
void SDNode::Profile(FoldingSetNodeID &ID) {
  AddNodeIDNode(ID, this);
}

/// getValueTypeList - Return a pointer to the specified value type.
///
MVT::ValueType *SDNode::getValueTypeList(MVT::ValueType VT) {
  static MVT::ValueType VTs[MVT::LAST_VALUETYPE];
  VTs[VT] = VT;
  return &VTs[VT];
}
  
/// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
/// indicated value.  This method ignores uses of other values defined by this
/// operation.
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  // If there is only one value, this is easy.
  if (getNumValues() == 1)
    return use_size() == NUses;
  if (use_size() < NUses) return false;

  SDOperand TheValue(const_cast<SDNode *>(this), Value);

  SmallPtrSet<SDNode*, 32> UsersHandled;

  for (SDNode::use_iterator UI = Uses.begin(), E = Uses.end(); UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getNumOperands() == 1 ||
        UsersHandled.insert(User))     // First time we've seen this?
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
        if (User->getOperand(i) == TheValue) {
          if (NUses == 0)
            return false;   // too many uses
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

  if (use_size() == 0) return false;

  SDOperand TheValue(const_cast<SDNode *>(this), Value);

  SmallPtrSet<SDNode*, 32> UsersHandled;

  for (SDNode::use_iterator UI = Uses.begin(), E = Uses.end(); UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getNumOperands() == 1 ||
        UsersHandled.insert(User))     // First time we've seen this?
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
        if (User->getOperand(i) == TheValue) {
          return true;
        }
  }

  return false;
}


/// isOnlyUse - Return true if this node is the only use of N.
///
bool SDNode::isOnlyUse(SDNode *N) const {
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
bool SDOperand::isOperand(SDNode *N) const {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (*this == N->getOperand(i))
      return true;
  return false;
}

bool SDNode::isOperand(SDNode *N) const {
  for (unsigned i = 0, e = N->NumOperands; i != e; ++i)
    if (this == N->OperandList[i].Val)
      return true;
  return false;
}

static void findPredecessor(SDNode *N, const SDNode *P, bool &found,
                            SmallPtrSet<SDNode *, 32> &Visited) {
  if (found || !Visited.insert(N))
    return;

  for (unsigned i = 0, e = N->getNumOperands(); !found && i != e; ++i) {
    SDNode *Op = N->getOperand(i).Val;
    if (Op == P) {
      found = true;
      return;
    }
    findPredecessor(Op, P, found, Visited);
  }
}

/// isPredecessor - Return true if this node is a predecessor of N. This node
/// is either an operand of N or it can be reached by recursively traversing
/// up the operands.
/// NOTE: this is an expensive method. Use it carefully.
bool SDNode::isPredecessor(SDNode *N) const {
  SmallPtrSet<SDNode *, 32> Visited;
  bool found = false;
  findPredecessor(N, this, found, Visited);
  return found;
}

uint64_t SDNode::getConstantOperandVal(unsigned Num) const {
  assert(Num < NumOperands && "Invalid child # of SDNode!");
  return cast<ConstantSDNode>(OperandList[Num])->getValue();
}

std::string SDNode::getOperationName(const SelectionDAG *G) const {
  switch (getOpcode()) {
  default:
    if (getOpcode() < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    else {
      if (G) {
        if (const TargetInstrInfo *TII = G->getTarget().getInstrInfo())
          if (getOpcode()-ISD::BUILTIN_OP_END < TII->getNumOpcodes())
            return TII->getName(getOpcode()-ISD::BUILTIN_OP_END);

        TargetLowering &TLI = G->getTargetLoweringInfo();
        const char *Name =
          TLI.getTargetNodeName(getOpcode());
        if (Name) return Name;
      }

      return "<<Unknown Target Node>>";
    }
   
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::READCYCLECOUNTER: return "ReadCycleCounter";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::AssertSext:    return "AssertSext";
  case ISD::AssertZext:    return "AssertZext";

  case ISD::STRING:        return "String";
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
  case ISD::EHSELECTION: return "EHSELECTION";
  case ISD::EH_RETURN: return "EH_RETURN";
  case ISD::ConstantPool:  return "ConstantPool";
  case ISD::ExternalSymbol: return "ExternalSymbol";
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(getOperand(0))->getValue();
    return Intrinsic::getName((Intrinsic::ID)IID);
  }
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(getOperand(1))->getValue();
    return Intrinsic::getName((Intrinsic::ID)IID);
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

  case ISD::CopyToReg:     return "CopyToReg";
  case ISD::CopyFromReg:   return "CopyFromReg";
  case ISD::UNDEF:         return "undef";
  case ISD::MERGE_VALUES:  return "merge_values";
  case ISD::INLINEASM:     return "inlineasm";
  case ISD::LABEL:         return "label";
  case ISD::HANDLENODE:    return "handlenode";
  case ISD::FORMAL_ARGUMENTS: return "formal_arguments";
  case ISD::CALL:          return "call";
    
  // Unary operators
  case ISD::FABS:   return "fabs";
  case ISD::FNEG:   return "fneg";
  case ISD::FSQRT:  return "fsqrt";
  case ISD::FSIN:   return "fsin";
  case ISD::FCOS:   return "fcos";
  case ISD::FPOWI:  return "fpowi";

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
  case ISD::FREM:   return "frem";
  case ISD::FCOPYSIGN: return "fcopysign";

  case ISD::SETCC:       return "setcc";
  case ISD::SELECT:      return "select";
  case ISD::SELECT_CC:   return "select_cc";
  case ISD::INSERT_VECTOR_ELT:   return "insert_vector_elt";
  case ISD::EXTRACT_VECTOR_ELT:  return "extract_vector_elt";
  case ISD::CONCAT_VECTORS:      return "concat_vectors";
  case ISD::EXTRACT_SUBVECTOR:   return "extract_subvector";
  case ISD::SCALAR_TO_VECTOR:    return "scalar_to_vector";
  case ISD::VECTOR_SHUFFLE:      return "vector_shuffle";
  case ISD::CARRY_FALSE:         return "carry_false";
  case ISD::ADDC:        return "addc";
  case ISD::ADDE:        return "adde";
  case ISD::SUBC:        return "subc";
  case ISD::SUBE:        return "sube";
  case ISD::SHL_PARTS:   return "shl_parts";
  case ISD::SRA_PARTS:   return "sra_parts";
  case ISD::SRL_PARTS:   return "srl_parts";
  
  case ISD::EXTRACT_SUBREG:     return "extract_subreg";
  case ISD::INSERT_SUBREG:      return "insert_subreg";
  
  // Conversion operators.
  case ISD::SIGN_EXTEND: return "sign_extend";
  case ISD::ZERO_EXTEND: return "zero_extend";
  case ISD::ANY_EXTEND:  return "any_extend";
  case ISD::SIGN_EXTEND_INREG: return "sign_extend_inreg";
  case ISD::TRUNCATE:    return "truncate";
  case ISD::FP_ROUND:    return "fp_round";
  case ISD::FP_ROUND_INREG: return "fp_round_inreg";
  case ISD::FP_EXTEND:   return "fp_extend";

  case ISD::SINT_TO_FP:  return "sint_to_fp";
  case ISD::UINT_TO_FP:  return "uint_to_fp";
  case ISD::FP_TO_SINT:  return "fp_to_sint";
  case ISD::FP_TO_UINT:  return "fp_to_uint";
  case ISD::BIT_CONVERT: return "bit_convert";

    // Control flow instructions
  case ISD::BR:      return "br";
  case ISD::BRIND:   return "brind";
  case ISD::BR_JT:   return "br_jt";
  case ISD::BRCOND:  return "brcond";
  case ISD::BR_CC:   return "br_cc";
  case ISD::RET:     return "ret";
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
    
  // Block memory operations.
  case ISD::MEMSET:  return "memset";
  case ISD::MEMCPY:  return "memcpy";
  case ISD::MEMMOVE: return "memmove";

  // Bit manipulation
  case ISD::BSWAP:   return "bswap";
  case ISD::CTPOP:   return "ctpop";
  case ISD::CTTZ:    return "cttz";
  case ISD::CTLZ:    return "ctlz";

  // Debug info
  case ISD::LOCATION: return "location";
  case ISD::DEBUG_LOC: return "debug_loc";

  // Trampolines
  case ISD::TRAMPOLINE: return "trampoline";

  case ISD::CONDCODE:
    switch (cast<CondCodeSDNode>(this)->get()) {
    default: assert(0 && "Unknown setcc condition!");
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

void SDNode::dump() const { dump(0); }
void SDNode::dump(const SelectionDAG *G) const {
  cerr << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) cerr << ",";
    if (getValueType(i) == MVT::Other)
      cerr << "ch";
    else
      cerr << MVT::getValueTypeString(getValueType(i));
  }
  cerr << " = " << getOperationName(G);

  cerr << " ";
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i) cerr << ", ";
    cerr << (void*)getOperand(i).Val;
    if (unsigned RN = getOperand(i).ResNo)
      cerr << ":" << RN;
  }

  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(this)) {
    cerr << "<" << CSDN->getValue() << ">";
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(this)) {
    cerr << "<" << (&CSDN->getValueAPF().getSemantics()==&APFloat::IEEEsingle ? 
                    CSDN->getValueAPF().convertToFloat() :
                    CSDN->getValueAPF().convertToDouble()) << ">";
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(this)) {
    int offset = GADN->getOffset();
    cerr << "<";
    WriteAsOperand(*cerr.stream(), GADN->getGlobal()) << ">";
    if (offset > 0)
      cerr << " + " << offset;
    else
      cerr << " " << offset;
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    cerr << "<" << FIDN->getIndex() << ">";
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(this)) {
    cerr << "<" << JTDN->getIndex() << ">";
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    int offset = CP->getOffset();
    if (CP->isMachineConstantPoolEntry())
      cerr << "<" << *CP->getMachineCPVal() << ">";
    else
      cerr << "<" << *CP->getConstVal() << ">";
    if (offset > 0)
      cerr << " + " << offset;
    else
      cerr << " " << offset;
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      cerr << LBB->getName() << " ";
    cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    if (G && R->getReg() && MRegisterInfo::isPhysicalRegister(R->getReg())) {
      cerr << " " <<G->getTarget().getRegisterInfo()->getName(R->getReg());
    } else {
      cerr << " #" << R->getReg();
    }
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    cerr << "'" << ES->getSymbol() << "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      cerr << "<" << M->getValue() << ":" << M->getOffset() << ">";
    else
      cerr << "<null:" << M->getOffset() << ">";
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(this)) {
    cerr << ":" << MVT::getValueTypeString(N->getVT());
  } else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(this)) {
    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:
      cerr << " <anyext ";
      break;
    case ISD::SEXTLOAD:
      cerr << " <sext ";
      break;
    case ISD::ZEXTLOAD:
      cerr << " <zext ";
      break;
    }
    if (doExt)
      cerr << MVT::getValueTypeString(LD->getLoadedVT()) << ">";

    const char *AM = getIndexedModeName(LD->getAddressingMode());
    if (*AM)
      cerr << " " << AM;
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(this)) {
    if (ST->isTruncatingStore())
      cerr << " <trunc "
           << MVT::getValueTypeString(ST->getStoredVT()) << ">";

    const char *AM = getIndexedModeName(ST->getAddressingMode());
    if (*AM)
      cerr << " " << AM;
  }
}

static void DumpNodes(const SDNode *N, unsigned indent, const SelectionDAG *G) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i).Val->hasOneUse())
      DumpNodes(N->getOperand(i).Val, indent+2, G);
    else
      cerr << "\n" << std::string(indent+2, ' ')
           << (void*)N->getOperand(i).Val << ": <multiple use>";


  cerr << "\n" << std::string(indent, ' ');
  N->dump(G);
}

void SelectionDAG::dump() const {
  cerr << "SelectionDAG has " << AllNodes.size() << " nodes:";
  std::vector<const SDNode*> Nodes;
  for (allnodes_const_iterator I = allnodes_begin(), E = allnodes_end();
       I != E; ++I)
    Nodes.push_back(I);
  
  std::sort(Nodes.begin(), Nodes.end());

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    if (!Nodes[i]->hasOneUse() && Nodes[i] != getRoot().Val)
      DumpNodes(Nodes[i], 2, this);
  }

  if (getRoot().Val) DumpNodes(getRoot().Val, 2, this);

  cerr << "\n\n";
}

const Type *ConstantPoolSDNode::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}
