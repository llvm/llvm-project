//===- AMDGPUTemporalDivergenceLowering.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "temporal-divergence-lowering"

using namespace llvm;

namespace {

class AMDGPUTemporalDivergenceLowering : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUTemporalDivergenceLowering() : MachineFunctionPass(ID) {
    initializeAMDGPUTemporalDivergenceLoweringPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "Temporal divergence lowering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.addRequired<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUTemporalDivergenceLowering, DEBUG_TYPE,
                      "Temporal divergence lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(AMDGPUTemporalDivergenceLowering, DEBUG_TYPE,
                    "Temporal divergence lowering", false, false)

char AMDGPUTemporalDivergenceLowering::ID = 0;

char &llvm::AMDGPUTemporalDivergenceLoweringID =
    AMDGPUTemporalDivergenceLowering::ID;

FunctionPass *llvm::createAMDGPUTemporalDivergenceLoweringPass() {
  return new AMDGPUTemporalDivergenceLowering();
}

static void replaceUseRegisterWith(const MachineInstr *MI, Register Reg,
                                   Register Newreg) {
  for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
    const MachineOperand &Op = MI->getOperand(i);
    if (Op.isReg() && Op.getReg() == Reg) {
      const_cast<MachineInstr *>(MI)->getOperand(i).setReg(Newreg);
    }
  }
}
// Get poiners to build instruction just after MI (skips phis if needed)
static std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
getInsertAfterPtrs(MachineInstr *MI) {
  MachineBasicBlock *InsertMBB = MI->getParent();
  return std::make_pair(
      InsertMBB, InsertMBB->SkipPHIsAndLabels(std::next(MI->getIterator())));
}

bool AMDGPUTemporalDivergenceLowering::runOnMachineFunction(
    MachineFunction &MF) {

  MachineCycleInfo &CycleInfo =
      getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  MachineDominatorTree &DomTree = getAnalysis<MachineDominatorTree>();

  MachineUniformityInfo MUI =
      computeMachineUniformityInfo(MF, CycleInfo, DomTree.getBase(), true);

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const GCNSubtarget &Subtarget = MF.getSubtarget<GCNSubtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  const SIRegisterInfo &TRI = *Subtarget.getRegisterInfo();

  // Temporal divergence lowering is required for uniform UniformSourceReg
  // and divergent UserInstr. UserInstr is uniform only when loop is uniform.
  for (auto [SrcReg, UserInstr, CycleExitBlocks] : MUI.uses_outside_cycle()) {
    if (!MUI.isUniform(SrcReg) || !MUI.isDivergent(UserInstr))
      continue;

    MachineInstr *UniformSourceInstr = MRI.getVRegDef(SrcReg);

    // FixMe: SrcReg is lane mask in this case. Find a better way to detect it.
    if (UniformSourceInstr->getOpcode() == AMDGPU::SI_IF_BREAK ||
        UserInstr->getOpcode() == AMDGPU::SI_IF)
      continue;

    unsigned Size = TRI.getRegSizeInBits(*MRI.getRegClassOrNull(SrcReg));
    Register VgprDst =
        MRI.createVirtualRegister(TRI.getVGPRClassForBitWidth(Size));

    auto [MBB, AfterUniformSourceReg] = getInsertAfterPtrs(UniformSourceInstr);
    BuildMI(*MBB, AfterUniformSourceReg, {}, TII.get(AMDGPU::COPY))
        .addDef(VgprDst)
        .addReg(SrcReg)
        .addReg(AMDGPU::EXEC, RegState::Implicit);

    replaceUseRegisterWith(UserInstr, SrcReg, VgprDst);
  }

  return true;
}
