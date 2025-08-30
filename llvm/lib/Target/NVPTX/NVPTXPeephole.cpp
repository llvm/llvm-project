//===-- NVPTXPeephole.cpp - NVPTX Peephole Optimiztions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In NVPTX, NVPTXFrameLowering will emit following instruction at the beginning
// of a MachineFunction.
//
//   mov %SPL, %depot
//   cvta.local %SP, %SPL
//
// Allocas are local addresses, if multiple alloca addresses need to be
// converted to generic ones, then multiple cvta.local instructions will be
// emitted. To eliminate these redundant cvta.local instructions, we need to
// combine them into a single cvta.local instruction.
//
// This peephole pass optimizes these cases, for example
//
// It will transform the following pattern
//    %0 = LEA_ADDRi64 %VRFrameLocal64, 0
//    %1 = LEA_ADDRi64 %VRFrameLocal64, 4
//    %2 = cvta_to_local_64 %0
//    %3 = cvta_to_local_64 %1
//
// into
//    %0 = LEA_ADDRi64 %VRFrame64, 0
//    %1 = LEA_ADDRi64 %VRFrame64, 4
//
// %VRFrameLocal64 is the virtual register name of %SPL
// %VRFrame64 is the virtual register name of %SP
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-peephole"

namespace {
struct NVPTXPeephole : public MachineFunctionPass {
 public:
  static char ID;
  NVPTXPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "NVPTX optimize redundant cvta.to.local instruction";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
}

char NVPTXPeephole::ID = 0;

INITIALIZE_PASS(NVPTXPeephole, "nvptx-peephole", "NVPTX Peephole", false, false)

static bool isCVTALocalCombinationCandidate(MachineInstr &Root) {
  auto &MBB = *Root.getParent();
  auto &MF = *MBB.getParent();
  // Check current instruction is cvta.local
  if (Root.getOpcode() != NVPTX::cvta_local_64 &&
      Root.getOpcode() != NVPTX::cvta_local)
    return false;

  auto &Op = Root.getOperand(1);
  const auto &MRI = MF.getRegInfo();
  MachineInstr *LocalAddrDef = nullptr;
  if (Op.isReg() && Op.getReg().isVirtual()) {
    LocalAddrDef = MRI.getUniqueVRegDef(Op.getReg());
  }

  if (!LocalAddrDef || LocalAddrDef->getParent() != &MBB)
    return false;

  //  With -nvptx-short-ptr there's an extra cvta.u64.u32 instruction
  // between the LEA_ADDRi and the cvta.local.
  if (LocalAddrDef->getOpcode() == NVPTX::CVT_u64_u32) {
    auto &Op = LocalAddrDef->getOperand(1);
    if (Op.isReg() && Op.getReg().isVirtual())
      LocalAddrDef = MRI.getUniqueVRegDef(Op.getReg());
  }

  // Check the register operand is uniquely defined by LEA_ADDRi instruction
  if (!LocalAddrDef || LocalAddrDef->getParent() != &MBB ||
      (LocalAddrDef->getOpcode() != NVPTX::LEA_ADDRi64 &&
       LocalAddrDef->getOpcode() != NVPTX::LEA_ADDRi)) {
    return false;
  }

  const NVPTXRegisterInfo *NRI =
      MF.getSubtarget<NVPTXSubtarget>().getRegisterInfo();

  // Check the LEA_ADDRi operand is Frame index
  auto &BaseAddrOp = LocalAddrDef->getOperand(1);
  if (BaseAddrOp.isReg() &&
      BaseAddrOp.getReg() == NRI->getFrameLocalRegister(MF)) {
    return true;
  }

  return false;
}

static void CombineCVTALocal(MachineInstr &CVTALocalInstr) {
  auto &MBB = *CVTALocalInstr.getParent();
  auto &MF = *MBB.getParent();
  const auto &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  auto *LeaInstr = MRI.getUniqueVRegDef(CVTALocalInstr.getOperand(1).getReg());
  MachineInstr *CVTInstr = nullptr;
  if (LeaInstr->getOpcode() == NVPTX::CVT_u64_u32) {
    CVTInstr = LeaInstr;
    LeaInstr = MRI.getUniqueVRegDef(LeaInstr->getOperand(1).getReg());
    assert((LeaInstr->getOpcode() == NVPTX::LEA_ADDRi64 ||
            LeaInstr->getOpcode() == NVPTX::LEA_ADDRi) &&
           "Expected LEA_ADDRi64 or LEA_ADDRi");
  }

  const NVPTXRegisterInfo *NRI =
      MF.getSubtarget<NVPTXSubtarget>().getRegisterInfo();

  MachineInstrBuilder MIB =
      BuildMI(MF, CVTALocalInstr.getDebugLoc(), TII->get(LeaInstr->getOpcode()),
              CVTALocalInstr.getOperand(0).getReg())
          .addReg(NRI->getFrameRegister(MF))
          .add(LeaInstr->getOperand(2));

  MBB.insert((MachineBasicBlock::iterator)&CVTALocalInstr, MIB);

  // Check if we can erase the cvta.u64.u32 or LEA_ADDRi instructions
  if (CVTInstr) {
    // Check if the cvta.u64.u32 instruction has only one non dbg use
    // which is the cvta.local instruction.
    if (MRI.hasOneNonDBGUse(CVTInstr->getOperand(0).getReg()))
      CVTInstr->eraseFromParent();

    // Check if the LEA_ADDRi instruction has no other non dbg uses
    // (i.e. cvta.u64.u32 was the only non dbg use)
    if (MRI.use_nodbg_empty(CVTInstr->getOperand(0).getReg()))
      CVTInstr->eraseFromParent();

  } else if (MRI.hasOneNonDBGUse(LeaInstr->getOperand(0).getReg())) {
    // Check if the LEA_ADDRi instruction has only one non dbg use
    // which is the cvta.u64.u32 instruction.
    LeaInstr->eraseFromParent();
  }

  CVTALocalInstr.eraseFromParent();
}

bool NVPTXPeephole::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  bool Changed = false;
  // Loop over all of the basic blocks.
  for (auto &MBB : MF) {
    // Traverse the basic block.
    auto BlockIter = MBB.begin();

    while (BlockIter != MBB.end()) {
      auto &MI = *BlockIter++;
      if (isCVTALocalCombinationCandidate(MI)) {
        CombineCVTALocal(MI);
        Changed = true;
      }
    }  // Instruction
  }    // Basic Block

  const NVPTXRegisterInfo *NRI =
      MF.getSubtarget<NVPTXSubtarget>().getRegisterInfo();

  const auto &MRI = MF.getRegInfo();

  // Remove unnecessary %VRFrame = cvta.local %VRFrame
  if (MRI.hasOneNonDBGUse(NRI->getFrameRegister(MF)))
    if (auto *MI = MRI.getOneNonDBGUser(NRI->getFrameRegister(MF)))
      if (MI->getOpcode() == NVPTX::cvta_local_64 ||
          MI->getOpcode() == NVPTX::cvta_local)
        MI->eraseFromParent();

  // Remove unnecessary %VRFrame = cvt.u64.u32 %VRFrameLocal
  if (MRI.use_empty(NRI->getFrameRegister(MF)))
    if (auto *MI = MRI.getUniqueVRegDef(NRI->getFrameRegister(MF)))
      MI->eraseFromParent();

  // Remove unnecessary %VRFrameLocal = LEA_ADDRi %depot
  if (MRI.use_empty(NRI->getFrameLocalRegister(MF)))
    if (auto *MI = MRI.getUniqueVRegDef(NRI->getFrameLocalRegister(MF)))
      MI->eraseFromParent();

  return Changed;
}

MachineFunctionPass *llvm::createNVPTXPeephole() { return new NVPTXPeephole(); }
