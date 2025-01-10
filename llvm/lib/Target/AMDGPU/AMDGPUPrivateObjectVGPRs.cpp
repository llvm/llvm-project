//===----------- AMDGPUPrivateObjectVGPRs.cpp - Private object VGPRs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Add implicit use/def operands to V_LOAD/STORE_IDX pseudos for VGPRs
/// allocated to promoted private objects and thus prevent the register
/// allocator from using these VGPRs where the private objects are live.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

namespace {

class AMDGPUPrivateObjectVGPRs : public MachineFunctionPass {
  struct LiveInRange {
    MachineBasicBlock *MBB;
    MCPhysReg BaseReg;
    unsigned NumRegs;
  };

public:
  static char ID;

  AMDGPUPrivateObjectVGPRs() : MachineFunctionPass(ID) {
    initializeAMDGPUPrivateObjectVGPRsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Def/use private object VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool processMI(MachineInstr &MI, SmallVectorImpl<LiveInRange> &LiveIns);
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRs, DEBUG_TYPE,
                "AMDGPU Add defs/uses for private object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRs::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRs::ID;

FunctionPass *llvm::createAMDGPUPrivateObjectVGPRsPass() {
  return new AMDGPUPrivateObjectVGPRs();
}

bool AMDGPUPrivateObjectVGPRs::processMI(
    MachineInstr &MI, SmallVectorImpl<LiveInRange> &LiveIns) {
  if (MI.getOpcode() != AMDGPU::V_LOAD_IDX &&
      MI.getOpcode() != AMDGPU::V_STORE_IDX)
    return false;

  const MachineMemOperand *MMO = *MI.memoperands_begin();
  assert(MMO);
  const Value *Ptr = MMO->getValue();
  if (!Ptr)
    return false;

  if (auto *GEP = dyn_cast<GEPOperator>(Ptr))
    Ptr = GEP->getPointerOperand();
  const auto *Alloca = dyn_cast<AllocaInst>(Ptr);
  if (!Alloca)
    return false;

  const MDNode *AllocatedVGPRs = Alloca->getMetadata("amdgpu.allocated.vgprs");
  if (!AllocatedVGPRs)
    return false;

  unsigned Offset =
      cast<ConstantInt>(
          cast<ConstantAsMetadata>(AllocatedVGPRs->getOperand(0))->getValue())
          ->getZExtValue();
  unsigned Size =
      cast<ConstantInt>(
          cast<ConstantAsMetadata>(AllocatedVGPRs->getOperand(1))->getValue())
          ->getZExtValue();

  assert(Offset % 4 == 0 && Size % 4 == 0);
  MCPhysReg BaseReg = AMDGPU::VGPR0 + Offset / 4;
  unsigned NumRegs = Size / 4;
  LiveIns.push_back({MI.getParent(), BaseReg, NumRegs});

  for (unsigned I : seq(NumRegs)) {
    // In general case, we don't know which VGPRs are read or written, so
    // we conservatively assume V_LOAD_IDX pseudos load all of them and
    // V_STORE_IDX store only some of them, meaning V_STORE_IDX have to
    // have both defs and uses for all the registers.
    // TODO: In cases with constant GEPs where we can realiably determine
    // the accessed VGPRs we don't need to add defs/uses for all registers
    // and V_STORE_IDX don't need to have implicit uses.
    MCPhysReg Reg = BaseReg + I;
    MI.addOperand(
        MachineOperand::CreateReg(Reg, /*isDef=*/false, /*isImp=*/true));

    if (MI.getOpcode() == AMDGPU::V_STORE_IDX) {
      MI.addOperand(
          MachineOperand::CreateReg(Reg, /*isDef=*/true, /*isImp=*/true));
    }
  }

  return true;
}

bool AMDGPUPrivateObjectVGPRs::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasVGPRIndexingRegisters())
    return false;

  SmallVector<LiveInRange, 16> LiveIns;
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.instrs())
      Changed |= processMI(MI, LiveIns);
  }

  // Add live-ins and propagate them through all predecessors up to the
  // entry block.
  while (!LiveIns.empty()) {
    auto [MBB, BaseReg, NumRegs] = LiveIns.pop_back_val();
    if (MBB->isLiveIn(BaseReg))
      continue;

    for (unsigned I : seq(NumRegs))
      MBB->addLiveIn(BaseReg + I);
    for (MachineBasicBlock *Pred : MBB->predecessors())
      LiveIns.push_back({Pred, BaseReg, NumRegs});
  }

  return Changed;
}
