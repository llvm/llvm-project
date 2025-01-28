//===- AArch64WinFixupBufferSecurityCheck.cpp Fixup Buffer Security Check -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Buffer Security Check implementation inserts windows specific callback into
// code. On windows, __security_check_cookie call gets call everytime function
// is return without fixup. Since this function is defined in runtime library,
// it incures cost of call in dll which simply does comparison and returns most
// time. With Fixup, We selective move to call in DLL only if comparison fails.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Module.h"

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-win-fixup-bscheck"

namespace {

class AArch64WinFixupBufferSecurityCheckPass : public MachineFunctionPass {
public:
  static char ID;

  AArch64WinFixupBufferSecurityCheckPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AArch64 Windows Fixup Buffer Security Check";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  std::pair<MachineInstr *, MachineInstr *>
  findSecurityCheckAndLoadStackGuard(MachineFunction &MF);

  MachineInstr *cloneLoadStackGuard(MachineFunction &MF, MachineInstr *MI);

  bool getGuardCheckSequence(MachineInstr *CheckCall, MachineInstr *SeqMI[5]);

  void finishBlock(MachineBasicBlock *MBB);

  void finishFunction(MachineBasicBlock *FailMBB, MachineBasicBlock *NewRetMBB);
};
} // end anonymous namespace

char AArch64WinFixupBufferSecurityCheckPass::ID = 0;

INITIALIZE_PASS(AArch64WinFixupBufferSecurityCheckPass, DEBUG_TYPE, DEBUG_TYPE,
                false, false)

FunctionPass *llvm::createAArch64WinFixupBufferSecurityCheckPass() {
  return new AArch64WinFixupBufferSecurityCheckPass();
}

void AArch64WinFixupBufferSecurityCheckPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addUsedIfAvailable<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

std::pair<MachineInstr *, MachineInstr *>
AArch64WinFixupBufferSecurityCheckPass::findSecurityCheckAndLoadStackGuard(
    MachineFunction &MF) {

  MachineInstr *SecurityCheckCall = nullptr;
  MachineInstr *LoadStackGuard = nullptr;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (!LoadStackGuard && MI.getOpcode() == TargetOpcode::LOAD_STACK_GUARD) {
        LoadStackGuard = &MI;
      }

      if (MI.isCall() && MI.getNumExplicitOperands() == 1) {
        auto MO = MI.getOperand(0);
        if (MO.isGlobal()) {
          auto Callee = dyn_cast<Function>(MO.getGlobal());
          if (Callee && Callee->getName() == "__security_check_cookie") {
            SecurityCheckCall = &MI;
          }
        }
      }

      // If both are found, return them
      if (LoadStackGuard && SecurityCheckCall) {
        return std::make_pair(LoadStackGuard, SecurityCheckCall);
      }
    }
  }

  return std::make_pair(nullptr, nullptr);
}

MachineInstr *
AArch64WinFixupBufferSecurityCheckPass::cloneLoadStackGuard(MachineFunction &MF,
                                                            MachineInstr *MI) {

  MachineInstr *ClonedInstr = MF.CloneMachineInstr(MI);

  // Get the register class of the original destination register
  Register OrigReg = MI->getOperand(0).getReg();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterClass *RegClass = MRI.getRegClass(OrigReg);

  // Create a new virtual register in the same register class
  Register NewReg = MRI.createVirtualRegister(RegClass);

  // Update operand 0 (destination) of the cloned instruction
  MachineOperand &DestOperand = ClonedInstr->getOperand(0);
  if (DestOperand.isReg() && DestOperand.isDef()) {
    DestOperand.setReg(NewReg); // Set the new virtual register
  }

  return ClonedInstr;
}

bool AArch64WinFixupBufferSecurityCheckPass::getGuardCheckSequence(
    MachineInstr *CheckCall, MachineInstr *SeqMI[5]) {

  MachineBasicBlock *MBB = CheckCall->getParent();

  MachineBasicBlock::iterator UIt(CheckCall);
  MachineBasicBlock::reverse_iterator DIt(CheckCall);

  // Move forward to find the stack adjustment after the call
  ++UIt;
  if (UIt == MBB->end() || UIt->getOpcode() != AArch64::ADJCALLSTACKUP) {
    return false;
  }
  SeqMI[4] = &*UIt;

  // Assign the BL instruction (call to __security_check_cookie)
  SeqMI[3] = CheckCall;

  // Move backward to find the COPY instruction for the function slot cookie
  // argument passing
  ++DIt;
  if (DIt == MBB->rend() || DIt->getOpcode() != AArch64::COPY) {
    return false;
  }
  SeqMI[2] = &*DIt;

  // Move backward to find the instruction that loads the security cookie from
  // the stack
  ++DIt;
  if (DIt == MBB->rend() || DIt->getOpcode() != AArch64::LDRXui) {
    return false;
  }
  SeqMI[1] = &*DIt;

  // Move backward to find the stack adjustment before the call
  ++DIt;
  if (DIt == MBB->rend() || DIt->getOpcode() != AArch64::ADJCALLSTACKDOWN) {
    return false;
  }
  SeqMI[0] = &*DIt;

  // If all instructions are matched and stored, the sequence is valid
  return true;
}

void AArch64WinFixupBufferSecurityCheckPass::finishBlock(
    MachineBasicBlock *MBB) {
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *MBB);
}

void AArch64WinFixupBufferSecurityCheckPass::finishFunction(
    MachineBasicBlock *FailMBB, MachineBasicBlock *NewRetMBB) {
  FailMBB->getParent()->RenumberBlocks();
  // FailMBB includes call to MSCV RT  where __security_check_cookie
  // function is called. This function uses regcall and it expects cookie
  // value from stack slot.( even if this is modified)
  // Before going further we compute back livein for this block to make sure
  // it is live and provided.
  finishBlock(FailMBB);
  finishBlock(NewRetMBB);
}

bool AArch64WinFixupBufferSecurityCheckPass::runOnMachineFunction(
    MachineFunction &MF) {
  bool Changed = false;
  const AArch64Subtarget &STI = MF.getSubtarget<AArch64Subtarget>();

  if (!STI.getTargetTriple().isWindowsMSVCEnvironment())
    return Changed;

  // Check if security cookie was installed or not
  Module &M = *MF.getFunction().getParent();
  GlobalVariable *GV = M.getGlobalVariable("__security_cookie");
  if (!GV)
    return Changed;

  // Find LOAD_STACK_GUARD and __security_check_cookie instructions
  auto [StackGuard, CheckCall] = findSecurityCheckAndLoadStackGuard(MF);
  if (!CheckCall || !StackGuard)
    return Changed;

  // Get sequence of instructions in current basic block responsible for calling
  // __security_check_cookie
  MachineInstr *SeqMI[5];
  if (!getGuardCheckSequence(CheckCall, SeqMI))
    return Changed;

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  MachineBasicBlock *CurMBB = CheckCall->getParent();

  // Find LOAD_STACK_GUARD in CurrMBB and build a new LOAD_STACK_GUARD
  // instruction with new destination register
  MachineInstr *ClonedInstr = cloneLoadStackGuard(MF, StackGuard);
  if (!ClonedInstr)
    return Changed;

  // Insert cloned LOAD_STACK_GUARD right before the call to
  // __security_check_cookie
  MachineBasicBlock::iterator InsertPt(SeqMI[0]);
  CurMBB->insert(InsertPt, ClonedInstr);

  auto CookieLoadReg = SeqMI[1]->getOperand(0).getReg();
  auto GlobalCookieReg = ClonedInstr->getOperand(0).getReg();

  // Move LDRXui that loads __security_cookie from stack, right after
  // the cloned LOAD_STACK_GUARD
  CurMBB->splice(InsertPt, CurMBB, std::next(InsertPt));

  // Create a new virtual register for the CMP instruction result
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register DiscardReg = MRI.createVirtualRegister(&AArch64::GPR64RegClass);

  // Emit the CMP instruction to compare stack cookie with global cookie
  BuildMI(*CurMBB, InsertPt, DebugLoc(), TII->get(AArch64::SUBSXrr))
      .addReg(DiscardReg,
              RegState::Define | RegState::Dead) // Result discarded
      .addReg(CookieLoadReg)                     // First operand: stack cookie
      .addReg(GlobalCookieReg); // Second operand: global cookie

  // Create FailMBB basic block to call __security_check_cookie
  MachineBasicBlock *FailMBB = MF.CreateMachineBasicBlock();
  MF.insert(MF.end(), FailMBB);

  // Create NewRetMBB basic block to skip call to __security_check_cookie
  MachineBasicBlock *NewRetMBB = MF.CreateMachineBasicBlock();
  MF.insert(MF.end(), NewRetMBB);

  // Conditional branch to FailMBB if cookies do not match
  BuildMI(*CurMBB, InsertPt, DebugLoc(), TII->get(AArch64::Bcc))
      .addImm(AArch64CC::NE) // Condition: Not Equal
      .addMBB(FailMBB);      // Failure block

  // Add an unconditional branch to NewRetMBB.
  BuildMI(*CurMBB, InsertPt, DebugLoc(), TII->get(AArch64::B))
      .addMBB(NewRetMBB);

  // Move fail check squence from CurMBB to FailMBB
  MachineBasicBlock::iterator U2It(SeqMI[4]);
  ++U2It;
  FailMBB->splice(FailMBB->end(), CurMBB, InsertPt, U2It);

  // Insert a BRK instruction at the end of the FailMBB
  BuildMI(*FailMBB, FailMBB->end(), DebugLoc(), TII->get(AArch64::BRK))
      .addImm(0); // Immediate value for BRK

  // Move remaining instructions after CheckCall to NewRetMBB.
  NewRetMBB->splice(NewRetMBB->end(), CurMBB, U2It, CurMBB->end());

  // Restructure Basic Blocks
  CurMBB->addSuccessor(NewRetMBB);
  CurMBB->addSuccessor(FailMBB);

  MachineDominatorTreeWrapperPass *WrapperPass =
      getAnalysisIfAvailable<MachineDominatorTreeWrapperPass>();
  MachineDominatorTree *MDT =
      WrapperPass ? &WrapperPass->getDomTree() : nullptr;
  if (MDT) {
    MDT->addNewBlock(FailMBB, CurMBB);
    MDT->addNewBlock(NewRetMBB, CurMBB);
  }

  finishFunction(FailMBB, NewRetMBB);

  return !Changed;
}
