//===-- llvm/CodeGen/SelectionDAGAddressAnalysis.cpp ------- DAG Address
//Analysis ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//

#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {

bool BaseIndexOffset::equalBaseIndex(BaseIndexOffset &Other,
                                     const SelectionDAG &DAG, int64_t &Off) {
  // Obvious equivalent
  Off = Other.Offset - Offset;
  if (Other.Base == Base && Other.Index == Index &&
      Other.IsIndexSignExt == IsIndexSignExt)
    return true;

  // Match GlobalAddresses
  if (Index == Other.Index)
    if (GlobalAddressSDNode *A = dyn_cast<GlobalAddressSDNode>(Base))
      if (GlobalAddressSDNode *B = dyn_cast<GlobalAddressSDNode>(Other.Base))
        if (A->getGlobal() == B->getGlobal()) {
          Off += B->getOffset() - A->getOffset();
          return true;
        }

  // TODO: we should be able to add FrameIndex analysis improvements here.

  return false;
}

/// Parses tree in Ptr for base, index, offset addresses.
BaseIndexOffset BaseIndexOffset::match(SDValue Ptr) {
  // (((B + I*M) + c)) + c ...
  SDValue Base = Ptr;
  SDValue Index = SDValue();
  int64_t Offset = 0;
  bool IsIndexSignExt = false;

  // Consume constant adds
  while (Base->getOpcode() == ISD::ADD &&
         isa<ConstantSDNode>(Base->getOperand(1))) {
    int64_t POffset = cast<ConstantSDNode>(Base->getOperand(1))->getSExtValue();
    Offset += POffset;
    Base = Base->getOperand(0);
  }

  if (Base->getOpcode() == ISD::ADD) {
    // TODO: The following code appears to be needless as it just
    //       bails on some Ptrs early, reducing the cases where we
    //       find equivalence. We should be able to remove this.
    // Inside a loop the current BASE pointer is calculated using an ADD and a
    // MUL instruction. In this case Base is the actual BASE pointer.
    // (i64 add (i64 %array_ptr)
    //          (i64 mul (i64 %induction_var)
    //                   (i64 %element_size)))
    if (Base->getOperand(1)->getOpcode() == ISD::MUL)
      return BaseIndexOffset(Base, Index, Offset, IsIndexSignExt);

    // Look at Base + Index + Offset cases.
    Index = Base->getOperand(1);
    SDValue PotentialBase = Base->getOperand(0);

    // Skip signextends.
    if (Index->getOpcode() == ISD::SIGN_EXTEND) {
      Index = Index->getOperand(0);
      IsIndexSignExt = true;
    }

    // Check if Index Offset pattern
    if (Index->getOpcode() != ISD::ADD ||
        !isa<ConstantSDNode>(Index->getOperand(1)))
      return BaseIndexOffset(PotentialBase, Index, Offset, IsIndexSignExt);

    Offset += cast<ConstantSDNode>(Index->getOperand(1))->getSExtValue();
    Index = Index->getOperand(0);
    if (Index->getOpcode() == ISD::SIGN_EXTEND) {
      Index = Index->getOperand(0);
      IsIndexSignExt = true;
    } else
      IsIndexSignExt = false;
    Base = PotentialBase;
  }
  return BaseIndexOffset(Base, Index, Offset, IsIndexSignExt);
}
} // end namespace llvm
