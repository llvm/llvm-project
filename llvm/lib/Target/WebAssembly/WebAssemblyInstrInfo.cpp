//===-- WebAssemblyInstrInfo.cpp - WebAssembly Instruction Information ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of the
/// TargetInstrInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyInstrInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-instr-info"

#define GET_INSTRINFO_CTOR_DTOR
#include "WebAssemblyGenInstrInfo.inc"

WebAssemblyInstrInfo::WebAssemblyInstrInfo(const WebAssemblySubtarget &STI)
    : RI(STI.getTargetTriple()) {}

void WebAssemblyInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       DebugLoc DL, unsigned DestReg,
                                       unsigned SrcReg, bool KillSrc) const {
  BuildMI(MBB, I, DL, get(WebAssembly::COPY), DestReg)
      .addReg(SrcReg, KillSrc ? RegState::Kill : 0);
}

// Branch analysis.
bool WebAssemblyInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                         MachineBasicBlock *&TBB,
                                         MachineBasicBlock *&FBB,
                                         SmallVectorImpl<MachineOperand> &Cond,
                                         bool AllowModify) const {
  bool HaveCond = false;
  for (MachineInstr &MI : iterator_range<MachineBasicBlock::instr_iterator>(
           MBB.getFirstInstrTerminator(), MBB.instr_end())) {
    switch (MI.getOpcode()) {
    default:
      // Unhandled instruction; bail out.
      return true;
    case WebAssembly::BR_IF:
      if (HaveCond)
        return true;
      Cond.push_back(MI.getOperand(1));
      TBB = MI.getOperand(0).getMBB();
      HaveCond = true;
      break;
    case WebAssembly::BR:
      if (!HaveCond)
        TBB = MI.getOperand(0).getMBB();
      else
        FBB = MI.getOperand(0).getMBB();
      break;
    }
    if (MI.isBarrier())
      break;
  }

  return false;
}

unsigned WebAssemblyInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::instr_iterator I = MBB.instr_end();
  unsigned Count = 0;

  while (I != MBB.instr_begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (!I->isTerminator())
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.instr_end();
    ++Count;
  }

  return Count;
}

unsigned WebAssemblyInstrInfo::InsertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, DebugLoc DL) const {
  assert(Cond.size() <= 1);

  if (Cond.empty()) {
    if (!TBB)
      return 0;

    BuildMI(&MBB, DL, get(WebAssembly::BR)).addMBB(TBB);
    return 1;
  }

  BuildMI(&MBB, DL, get(WebAssembly::BR_IF))
      .addMBB(TBB)
      .addOperand(Cond[0]);
  if (!FBB)
    return 1;

  BuildMI(&MBB, DL, get(WebAssembly::BR)).addMBB(FBB);
  return 2;
}

bool WebAssemblyInstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1);

  // TODO: Add branch reversal here... And re-enable MachineBlockPlacementID
  // when we do.

  return true;
}
