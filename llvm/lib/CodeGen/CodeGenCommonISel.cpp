//===-- CodeGenCommonISel.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilies that are shared between SelectionDAG and
// GlobalISel frameworks.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CodeGenCommonISel.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"

using namespace llvm;

/// Add a successor MBB to ParentMBB< creating a new MachineBB for BB if SuccMBB
/// is 0.
MachineBasicBlock *
StackProtectorDescriptor::addSuccessorMBB(
    const BasicBlock *BB, MachineBasicBlock *ParentMBB, bool IsLikely,
    MachineBasicBlock *SuccMBB) {
  // If SuccBB has not been created yet, create it.
  if (!SuccMBB) {
    MachineFunction *MF = ParentMBB->getParent();
    MachineFunction::iterator BBI(ParentMBB);
    SuccMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(++BBI, SuccMBB);
  }
  // Add it as a successor of ParentMBB.
  ParentMBB->addSuccessor(
      SuccMBB, BranchProbabilityInfo::getBranchProbStackProtector(IsLikely));
  return SuccMBB;
}

/// Given that the input MI is before a partial terminator sequence TSeq, return
/// true if M + TSeq also a partial terminator sequence.
///
/// A Terminator sequence is a sequence of MachineInstrs which at this point in
/// lowering copy vregs into physical registers, which are then passed into
/// terminator instructors so we can satisfy ABI constraints. A partial
/// terminator sequence is an improper subset of a terminator sequence (i.e. it
/// may be the whole terminator sequence).
static bool MIIsInTerminatorSequence(const MachineInstr &MI) {
  // If we do not have a copy or an implicit def, we return true if and only if
  // MI is a debug value.
  if (!MI.isCopy() && !MI.isImplicitDef()) {
    // Sometimes DBG_VALUE MI sneak in between the copies from the vregs to the
    // physical registers if there is debug info associated with the terminator
    // of our mbb. We want to include said debug info in our terminator
    // sequence, so we return true in that case.
    if (MI.isDebugInstr())
      return true;

    // For GlobalISel, we may have extension instructions for arguments within
    // copy sequences. Allow these.
    switch (MI.getOpcode()) {
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_MERGE_VALUES:
    case TargetOpcode::G_UNMERGE_VALUES:
    case TargetOpcode::G_CONCAT_VECTORS:
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_EXTRACT:
      return true;
    default:
      return false;
    }
  }

  // We have left the terminator sequence if we are not doing one of the
  // following:
  //
  // 1. Copying a vreg into a physical register.
  // 2. Copying a vreg into a vreg.
  // 3. Defining a register via an implicit def.

  // OPI should always be a register definition...
  MachineInstr::const_mop_iterator OPI = MI.operands_begin();
  if (!OPI->isReg() || !OPI->isDef())
    return false;

  // Defining any register via an implicit def is always ok.
  if (MI.isImplicitDef())
    return true;

  // Grab the copy source...
  MachineInstr::const_mop_iterator OPI2 = OPI;
  ++OPI2;
  assert(OPI2 != MI.operands_end()
         && "Should have a copy implying we should have 2 arguments.");

  // Make sure that the copy dest is not a vreg when the copy source is a
  // physical register.
  if (!OPI2->isReg() || (!Register::isPhysicalRegister(OPI->getReg()) &&
                         Register::isPhysicalRegister(OPI2->getReg())))
    return false;

  return true;
}

/// Find the split point at which to splice the end of BB into its success stack
/// protector check machine basic block.
///
/// On many platforms, due to ABI constraints, terminators, even before register
/// allocation, use physical registers. This creates an issue for us since
/// physical registers at this point can not travel across basic
/// blocks. Luckily, selectiondag always moves physical registers into vregs
/// when they enter functions and moves them through a sequence of copies back
/// into the physical registers right before the terminator creating a
/// ``Terminator Sequence''. This function is searching for the beginning of the
/// terminator sequence so that we can ensure that we splice off not just the
/// terminator, but additionally the copies that move the vregs into the
/// physical registers.
MachineBasicBlock::iterator
llvm::findSplitPointForStackProtector(MachineBasicBlock *BB,
                                      const TargetInstrInfo &TII) {
  MachineBasicBlock::iterator SplitPoint = BB->getFirstTerminator();
  if (SplitPoint == BB->begin())
    return SplitPoint;

  MachineBasicBlock::iterator Start = BB->begin();
  MachineBasicBlock::iterator Previous = SplitPoint;
  do {
    --Previous;
  } while (Previous != Start && Previous->isDebugInstr());

  if (TII.isTailCall(*SplitPoint) &&
      Previous->getOpcode() == TII.getCallFrameDestroyOpcode()) {
    // Call frames cannot be nested, so if this frame is describing the tail
    // call itself, then we must insert before the sequence even starts. For
    // example:
    //     <split point>
    //     ADJCALLSTACKDOWN ...
    //     <Moves>
    //     ADJCALLSTACKUP ...
    //     TAILJMP somewhere
    // On the other hand, it could be an unrelated call in which case this tail
    // call has no register moves of its own and should be the split point. For
    // example:
    //     ADJCALLSTACKDOWN
    //     CALL something_else
    //     ADJCALLSTACKUP
    //     <split point>
    //     TAILJMP somewhere
    do {
      --Previous;
      if (Previous->isCall())
        return SplitPoint;
    } while(Previous->getOpcode() != TII.getCallFrameSetupOpcode());

    return Previous;
  }

  while (MIIsInTerminatorSequence(*Previous)) {
    SplitPoint = Previous;
    if (Previous == Start)
      break;
    --Previous;
  }

  return SplitPoint;
}
