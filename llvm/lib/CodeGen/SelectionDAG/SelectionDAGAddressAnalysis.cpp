//==- llvm/CodeGen/SelectionDAGAddressAnalysis.cpp - DAG Address Analysis --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

using namespace llvm;

bool BaseIndexOffset::equalBaseIndex(const BaseIndexOffset &Other,
                                     const SelectionDAG &DAG,
                                     int64_t &Off) const {
  // Conservatively fail if we a match failed..
  if (!Base.getNode() || !Other.Base.getNode())
    return false;
  if (!hasValidOffset() || !Other.hasValidOffset())
    return false;
  // Initial Offset difference.
  Off = *Other.Offset - *Offset;

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

    // Match Constants
    if (auto *A = dyn_cast<ConstantPoolSDNode>(Base))
      if (auto *B = dyn_cast<ConstantPoolSDNode>(Other.Base)) {
        bool IsMatch =
            A->isMachineConstantPoolEntry() == B->isMachineConstantPoolEntry();
        if (IsMatch) {
          if (A->isMachineConstantPoolEntry())
            IsMatch = A->getMachineCPVal() == B->getMachineCPVal();
          else
            IsMatch = A->getConstVal() == B->getConstVal();
        }
        if (IsMatch) {
          Off += B->getOffset() - A->getOffset();
          return true;
        }
      }

    const MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();

    // Match FrameIndexes.
    if (auto *A = dyn_cast<FrameIndexSDNode>(Base))
      if (auto *B = dyn_cast<FrameIndexSDNode>(Other.Base)) {
        // Equal FrameIndexes - offsets are directly comparable.
        if (A->getIndex() == B->getIndex())
          return true;
        // Non-equal FrameIndexes - If both frame indices are fixed
        // we know their relative offsets and can compare them. Otherwise
        // we must be conservative.
        if (MFI.isFixedObjectIndex(A->getIndex()) &&
            MFI.isFixedObjectIndex(B->getIndex())) {
          Off += MFI.getObjectOffset(B->getIndex()) -
                 MFI.getObjectOffset(A->getIndex());
          return true;
        }
      }
  }
  return false;
}

bool BaseIndexOffset::computeAliasing(const SDNode *Op0,
                                      const Optional<int64_t> NumBytes0,
                                      const SDNode *Op1,
                                      const Optional<int64_t> NumBytes1,
                                      const SelectionDAG &DAG, bool &IsAlias) {

  BaseIndexOffset BasePtr0 = match(Op0, DAG);
  BaseIndexOffset BasePtr1 = match(Op1, DAG);

  if (!(BasePtr0.getBase().getNode() && BasePtr1.getBase().getNode()))
    return false;
  int64_t PtrDiff;
  if (NumBytes0.hasValue() && NumBytes1.hasValue() &&
      BasePtr0.equalBaseIndex(BasePtr1, DAG, PtrDiff)) {
    // If the size of memory access is unknown, do not use it to analysis.
    // One example of unknown size memory access is to load/store scalable
    // vector objects on the stack.
    // BasePtr1 is PtrDiff away from BasePtr0. They alias if none of the
    // following situations arise:
    if (PtrDiff >= 0 &&
        *NumBytes0 != static_cast<int64_t>(MemoryLocation::UnknownSize)) {
      // [----BasePtr0----]
      //                         [---BasePtr1--]
      // ========PtrDiff========>
      IsAlias = !(*NumBytes0 <= PtrDiff);
      return true;
    }
    if (PtrDiff < 0 &&
        *NumBytes1 != static_cast<int64_t>(MemoryLocation::UnknownSize)) {
      //                     [----BasePtr0----]
      // [---BasePtr1--]
      // =====(-PtrDiff)====>
      IsAlias = !((PtrDiff + *NumBytes1) <= 0);
      return true;
    }
    return false;
  }
  // If both BasePtr0 and BasePtr1 are FrameIndexes, we will not be
  // able to calculate their relative offset if at least one arises
  // from an alloca. However, these allocas cannot overlap and we
  // can infer there is no alias.
  if (auto *A = dyn_cast<FrameIndexSDNode>(BasePtr0.getBase()))
    if (auto *B = dyn_cast<FrameIndexSDNode>(BasePtr1.getBase())) {
      MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
      // If the base are the same frame index but the we couldn't find a
      // constant offset, (indices are different) be conservative.
      if (A != B && (!MFI.isFixedObjectIndex(A->getIndex()) ||
                     !MFI.isFixedObjectIndex(B->getIndex()))) {
        IsAlias = false;
        return true;
      }
    }

  bool IsFI0 = isa<FrameIndexSDNode>(BasePtr0.getBase());
  bool IsFI1 = isa<FrameIndexSDNode>(BasePtr1.getBase());
  bool IsGV0 = isa<GlobalAddressSDNode>(BasePtr0.getBase());
  bool IsGV1 = isa<GlobalAddressSDNode>(BasePtr1.getBase());
  bool IsCV0 = isa<ConstantPoolSDNode>(BasePtr0.getBase());
  bool IsCV1 = isa<ConstantPoolSDNode>(BasePtr1.getBase());

  // If of mismatched base types or checkable indices we can check
  // they do not alias.
  if ((BasePtr0.getIndex() == BasePtr1.getIndex() || (IsFI0 != IsFI1) ||
       (IsGV0 != IsGV1) || (IsCV0 != IsCV1)) &&
      (IsFI0 || IsGV0 || IsCV0) && (IsFI1 || IsGV1 || IsCV1)) {
    IsAlias = false;
    return true;
  }
  return false; // Cannot determine whether the pointers alias.
}

bool BaseIndexOffset::contains(const SelectionDAG &DAG, int64_t BitSize,
                               const BaseIndexOffset &Other,
                               int64_t OtherBitSize, int64_t &BitOffset) const {
  int64_t Offset;
  if (!equalBaseIndex(Other, DAG, Offset))
    return false;
  if (Offset >= 0) {
    // Other is after *this:
    // [-------*this---------]
    //            [---Other--]
    // ==Offset==>
    BitOffset = 8 * Offset;
    return BitOffset + OtherBitSize <= BitSize;
  }
  // Other starts strictly before *this, it cannot be fully contained.
  //    [-------*this---------]
  // [--Other--]
  return false;
}

/// Parses tree in Ptr for base, index, offset addresses.
static BaseIndexOffset matchLSNode(const LSBaseSDNode *N,
                                   const SelectionDAG &DAG) {
  SDValue Ptr = N->getBasePtr();

  // (((B + I*M) + c)) + c ...
  SDValue Base = DAG.getTargetLoweringInfo().unwrapAddress(Ptr);
  SDValue Index = SDValue();
  int64_t Offset = 0;
  bool IsIndexSignExt = false;

  // pre-inc/pre-dec ops are components of EA.
  if (N->getAddressingMode() == ISD::PRE_INC) {
    if (auto *C = dyn_cast<ConstantSDNode>(N->getOffset()))
      Offset += C->getSExtValue();
    else // If unknown, give up now.
      return BaseIndexOffset(SDValue(), SDValue(), 0, false);
  } else if (N->getAddressingMode() == ISD::PRE_DEC) {
    if (auto *C = dyn_cast<ConstantSDNode>(N->getOffset()))
      Offset -= C->getSExtValue();
    else // If unknown, give up now.
      return BaseIndexOffset(SDValue(), SDValue(), 0, false);
  }

  // Consume constant adds & ors with appropriate masking.
  while (true) {
    switch (Base->getOpcode()) {
    case ISD::OR:
      // Only consider ORs which act as adds.
      if (auto *C = dyn_cast<ConstantSDNode>(Base->getOperand(1)))
        if (DAG.MaskedValueIsZero(Base->getOperand(0), C->getAPIntValue())) {
          Offset += C->getSExtValue();
          Base = DAG.getTargetLoweringInfo().unwrapAddress(Base->getOperand(0));
          continue;
        }
      break;
    case ISD::ADD:
      if (auto *C = dyn_cast<ConstantSDNode>(Base->getOperand(1))) {
        Offset += C->getSExtValue();
        Base = DAG.getTargetLoweringInfo().unwrapAddress(Base->getOperand(0));
        continue;
      }
      break;
    case ISD::LOAD:
    case ISD::STORE: {
      auto *LSBase = cast<LSBaseSDNode>(Base.getNode());
      unsigned int IndexResNo = (Base->getOpcode() == ISD::LOAD) ? 1 : 0;
      if (LSBase->isIndexed() && Base.getResNo() == IndexResNo)
        if (auto *C = dyn_cast<ConstantSDNode>(LSBase->getOffset())) {
          auto Off = C->getSExtValue();
          if (LSBase->getAddressingMode() == ISD::PRE_DEC ||
              LSBase->getAddressingMode() == ISD::POST_DEC)
            Offset -= Off;
          else
            Offset += Off;
          Base = DAG.getTargetLoweringInfo().unwrapAddress(LSBase->getBasePtr());
          continue;
        }
      break;
    }
    }
    // If we get here break out of the loop.
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

BaseIndexOffset BaseIndexOffset::match(const SDNode *N,
                                       const SelectionDAG &DAG) {
  if (const auto *LS0 = dyn_cast<LSBaseSDNode>(N))
    return matchLSNode(LS0, DAG);
  if (const auto *LN = dyn_cast<LifetimeSDNode>(N)) {
    if (LN->hasOffset())
      return BaseIndexOffset(LN->getOperand(1), SDValue(), LN->getOffset(),
                             false);
    return BaseIndexOffset(LN->getOperand(1), SDValue(), false);
  }
  return BaseIndexOffset();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

LLVM_DUMP_METHOD void BaseIndexOffset::dump() const {
  print(dbgs());
}

void BaseIndexOffset::print(raw_ostream& OS) const {
  OS << "BaseIndexOffset base=[";
  Base->print(OS);
  OS << "] index=[";
  if (Index)
    Index->print(OS);
  OS << "] offset=" << Offset;
}

#endif
