//==- llvm/CodeGen/SelectionDAGAddressAnalysis.cpp - DAG Address Analysis --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

using namespace llvm;

bool BaseIndexOffset::equalBaseIndex(BaseIndexOffset &Other,
                                     const SelectionDAG &DAG, int64_t &Off) {
  // Initial Offset difference.
  Off = Other.Offset - Offset;

  if ((Other.Index == Index) && (Other.IsIndexSignExt == IsIndexSignExt)) {
    // Trivial match.
    if (Other.Base == Base)
      return true;

    // Match GlobalAddresses
    if (auto *A = dyn_cast<GlobalAddressSDNode>(Base))
      if (auto *B = dyn_cast<GlobalAddressSDNode>(Other.Base))
        if (A->getGlobal() == B->getGlobal()) {
          Off += B->getOffset() - A->getOffset();
          return true;
        }

    const MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();

    // Match non-equal FrameIndexes - If both frame indices are fixed
    // we know their relative offsets and can compare them. Otherwise
    // we must be conservative.
    if (auto *A = dyn_cast<FrameIndexSDNode>(Base))
      if (auto *B = dyn_cast<FrameIndexSDNode>(Other.Base))
        if (MFI.isFixedObjectIndex(A->getIndex()) &&
            MFI.isFixedObjectIndex(B->getIndex())) {
          Off += MFI.getObjectOffset(B->getIndex()) -
                 MFI.getObjectOffset(A->getIndex());
          return true;
        }
  }
  return false;
}

/// Parses tree in Ptr for base, index, offset addresses.
BaseIndexOffset BaseIndexOffset::match(SDValue Ptr, const SelectionDAG &DAG) {
  // (((B + I*M) + c)) + c ...
  SDValue Base = DAG.getTargetLoweringInfo().unwrapAddress(Ptr);
  SDValue Index = SDValue();
  int64_t Offset = 0;
  bool IsIndexSignExt = false;

  // Consume constant adds & ors with appropriate masking.
  while (Base->getOpcode() == ISD::ADD || Base->getOpcode() == ISD::OR) {
    if (auto *C = dyn_cast<ConstantSDNode>(Base->getOperand(1))) {
      // Only consider ORs which act as adds.
      if (Base->getOpcode() == ISD::OR &&
          !DAG.MaskedValueIsZero(Base->getOperand(0), C->getAPIntValue()))
        break;
      Offset += C->getSExtValue();
      Base = Base->getOperand(0);
      continue;
    }
    break;
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
