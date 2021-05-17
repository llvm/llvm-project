//===-- WebAssemblyInstrInfo.cpp - WebAssembly Instruction Information ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the WebAssembly implementation of the
/// TargetInstrInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyInstrInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-instr-info"

#define GET_INSTRINFO_CTOR_DTOR
#include "WebAssemblyGenInstrInfo.inc"

// defines WebAssembly::getNamedOperandIdx
#define GET_INSTRINFO_NAMED_OPS
#include "WebAssemblyGenInstrInfo.inc"

WebAssemblyInstrInfo::WebAssemblyInstrInfo(const WebAssemblySubtarget &STI)
    : WebAssemblyGenInstrInfo(WebAssembly::ADJCALLSTACKDOWN,
                              WebAssembly::ADJCALLSTACKUP,
                              WebAssembly::CATCHRET),
      RI(STI.getTargetTriple()) {}

bool WebAssemblyInstrInfo::isReallyTriviallyReMaterializable(
    const MachineInstr &MI, AAResults *AA) const {
  switch (MI.getOpcode()) {
  case WebAssembly::CONST_I32:
  case WebAssembly::CONST_I64:
  case WebAssembly::CONST_F32:
  case WebAssembly::CONST_F64:
    // isReallyTriviallyReMaterializableGeneric misses these because of the
    // ARGUMENTS implicit def, so we manualy override it here.
    return true;
  default:
    return false;
  }
}

void WebAssemblyInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       const DebugLoc &DL, MCRegister DestReg,
                                       MCRegister SrcReg, bool KillSrc) const {
  // This method is called by post-RA expansion, which expects only pregs to
  // exist. However we need to handle both here.
  auto &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
      Register::isVirtualRegister(DestReg)
          ? MRI.getRegClass(DestReg)
          : MRI.getTargetRegisterInfo()->getMinimalPhysRegClass(DestReg);

  unsigned CopyOpcode;
  if (RC == &WebAssembly::I32RegClass)
    CopyOpcode = WebAssembly::COPY_I32;
  else if (RC == &WebAssembly::I64RegClass)
    CopyOpcode = WebAssembly::COPY_I64;
  else if (RC == &WebAssembly::F32RegClass)
    CopyOpcode = WebAssembly::COPY_F32;
  else if (RC == &WebAssembly::F64RegClass)
    CopyOpcode = WebAssembly::COPY_F64;
  else if (RC == &WebAssembly::V128RegClass)
    CopyOpcode = WebAssembly::COPY_V128;
  else if (RC == &WebAssembly::FUNCREFRegClass)
    CopyOpcode = WebAssembly::COPY_FUNCREF;
  else if (RC == &WebAssembly::EXTERNREFRegClass)
    CopyOpcode = WebAssembly::COPY_EXTERNREF;
  else
    llvm_unreachable("Unexpected register class");

  BuildMI(MBB, I, DL, get(CopyOpcode), DestReg)
      .addReg(SrcReg, KillSrc ? RegState::Kill : 0);
}

MachineInstr *WebAssemblyInstrInfo::commuteInstructionImpl(
    MachineInstr &MI, bool NewMI, unsigned OpIdx1, unsigned OpIdx2) const {
  // If the operands are stackified, we can't reorder them.
  WebAssemblyFunctionInfo &MFI =
      *MI.getParent()->getParent()->getInfo<WebAssemblyFunctionInfo>();
  if (MFI.isVRegStackified(MI.getOperand(OpIdx1).getReg()) ||
      MFI.isVRegStackified(MI.getOperand(OpIdx2).getReg()))
    return nullptr;

  // Otherwise use the default implementation.
  return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
}

// Branch analysis.
bool WebAssemblyInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                         MachineBasicBlock *&TBB,
                                         MachineBasicBlock *&FBB,
                                         SmallVectorImpl<MachineOperand> &Cond,
                                         bool /*AllowModify*/) const {
  const auto &MFI = *MBB.getParent()->getInfo<WebAssemblyFunctionInfo>();
  // WebAssembly has control flow that doesn't have explicit branches or direct
  // fallthrough (e.g. try/catch), which can't be modeled by analyzeBranch. It
  // is created after CFGStackify.
  if (MFI.isCFGStackified())
    return true;

  bool HaveCond = false;
  for (MachineInstr &MI : MBB.terminators()) {
    switch (MI.getOpcode()) {
    default:
      // Unhandled instruction; bail out.
      return true;
    case WebAssembly::BR_IF:
      if (HaveCond)
        return true;
      Cond.push_back(MachineOperand::CreateImm(true));
      Cond.push_back(MI.getOperand(1));
      TBB = MI.getOperand(0).getMBB();
      HaveCond = true;
      break;
    case WebAssembly::BR_UNLESS:
      if (HaveCond)
        return true;
      Cond.push_back(MachineOperand::CreateImm(false));
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

unsigned WebAssemblyInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                            int *BytesRemoved) const {
  assert(!BytesRemoved && "code size not handled");

  MachineBasicBlock::instr_iterator I = MBB.instr_end();
  unsigned Count = 0;

  while (I != MBB.instr_begin()) {
    --I;
    if (I->isDebugInstr())
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

unsigned WebAssemblyInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  assert(!BytesAdded && "code size not handled");

  if (Cond.empty()) {
    if (!TBB)
      return 0;

    BuildMI(&MBB, DL, get(WebAssembly::BR)).addMBB(TBB);
    return 1;
  }

  assert(Cond.size() == 2 && "Expected a flag and a successor block");

  if (Cond[0].getImm())
    BuildMI(&MBB, DL, get(WebAssembly::BR_IF)).addMBB(TBB).add(Cond[1]);
  else
    BuildMI(&MBB, DL, get(WebAssembly::BR_UNLESS)).addMBB(TBB).add(Cond[1]);
  if (!FBB)
    return 1;

  BuildMI(&MBB, DL, get(WebAssembly::BR)).addMBB(FBB);
  return 2;
}

bool WebAssemblyInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Expected a flag and a condition expression");
  Cond.front() = MachineOperand::CreateImm(!Cond.front().getImm());
  return false;
}

ArrayRef<std::pair<int, const char *>>
WebAssemblyInstrInfo::getSerializableTargetIndices() const {
  static const std::pair<int, const char *> TargetIndices[] = {
      {WebAssembly::TI_LOCAL, "wasm-local"},
      {WebAssembly::TI_GLOBAL_FIXED, "wasm-global-fixed"},
      {WebAssembly::TI_OPERAND_STACK, "wasm-operand-stack"},
      {WebAssembly::TI_GLOBAL_RELOC, "wasm-global-reloc"},
      {WebAssembly::TI_LOCAL_INDIRECT, "wasm-local-indirect"}};
  return makeArrayRef(TargetIndices);
}
