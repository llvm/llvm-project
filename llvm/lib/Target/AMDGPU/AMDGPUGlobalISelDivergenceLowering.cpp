//===-- AMDGPUGlobalISelDivergenceLowering.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// GlobalISel pass that selects divergent i1 phis as lane mask phis.
/// Lane mask merging uses same algorithm as SDAG in SILowerI1Copies.
/// Handles all cases of temporal divergence.
/// For divergent non-phi i1 and uniform i1 uses outside of the cycle this pass
/// currently depends on LCSSA to insert phis with one incoming.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SILowerI1Copies.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-global-isel-divergence-lowering"

using namespace llvm;

namespace {

class AMDGPUGlobalISelDivergenceLowering : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUGlobalISelDivergenceLowering() : MachineFunctionPass(ID) {
    initializeAMDGPUGlobalISelDivergenceLoweringPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU GlobalISel divergence lowering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class DivergenceLoweringHelper : public PhiLoweringHelper {
public:
  DivergenceLoweringHelper(MachineFunction *MF, MachineDominatorTree *DT,
                           MachinePostDominatorTree *PDT,
                           MachineUniformityInfo *MUI);

private:
  MachineUniformityInfo *MUI = nullptr;

public:
  void markAsLaneMask(Register DstReg) const override;
  void getCandidatesForLowering(
      SmallVectorImpl<MachineInstr *> &Vreg1Phis) const override;
  void collectIncomingValuesFromPhi(
      const MachineInstr *MI,
      SmallVectorImpl<Incoming> &Incomings) const override;
  void replaceDstReg(Register NewReg, Register OldReg,
                     MachineBasicBlock *MBB) override;
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg,
                           Register CurReg) override;
  void constrainAsLaneMask(Incoming &In) override;
};

DivergenceLoweringHelper::DivergenceLoweringHelper(
    MachineFunction *MF, MachineDominatorTree *DT,
    MachinePostDominatorTree *PDT, MachineUniformityInfo *MUI)
    : PhiLoweringHelper(MF, DT, PDT), MUI(MUI) {}

// _(s1) -> SReg_32/64(s1)
void DivergenceLoweringHelper::markAsLaneMask(Register DstReg) const {
  assert(MRI->getType(DstReg) == LLT::scalar(1));

  if (MRI->getRegClassOrNull(DstReg)) {
    MRI->constrainRegClass(DstReg, ST->getBoolRC());
    return;
  }

  MRI->setRegClass(DstReg, ST->getBoolRC());
}

void DivergenceLoweringHelper::getCandidatesForLowering(
    SmallVectorImpl<MachineInstr *> &Vreg1Phis) const {
  LLT S1 = LLT::scalar(1);

  // Add divergent i1 phis to the list
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB.phis()) {
      Register Dst = MI.getOperand(0).getReg();
      if (MRI->getType(Dst) == S1 && MUI->isDivergent(Dst))
        Vreg1Phis.push_back(&MI);
    }
  }
}

void DivergenceLoweringHelper::collectIncomingValuesFromPhi(
    const MachineInstr *MI, SmallVectorImpl<Incoming> &Incomings) const {
  for (unsigned i = 1; i < MI->getNumOperands(); i += 2) {
    Incomings.emplace_back(MI->getOperand(i).getReg(),
                           MI->getOperand(i + 1).getMBB(), Register());
  }
}

void DivergenceLoweringHelper::replaceDstReg(Register NewReg, Register OldReg,
                                             MachineBasicBlock *MBB) {
  BuildMI(*MBB, MBB->getFirstNonPHI(), {}, TII->get(AMDGPU::COPY), OldReg)
      .addReg(NewReg);
}

// Get pointers to build instruction just after MI (skips phis if needed)
static std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
getInsertAfterPtrs(MachineInstr *MI) {
  MachineBasicBlock *InsertMBB = MI->getParent();
  return {InsertMBB,
          InsertMBB->SkipPHIsAndLabels(std::next(MI->getIterator()))};
}

// bb.previous
// %PrevReg = ...
//
// bb.current
// %CurReg = ...
//
// %DstReg - not defined
//
// -> (wave32 example, new registers have sreg_32 reg class and S1 LLT)
//
// bb.previous
// %PrevReg = ...
// %PrevRegCopy:sreg_32(s1) = COPY %PrevReg
//
// bb.current
// %CurReg = ...
// %CurRegCopy:sreg_32(s1) = COPY %CurReg
// ...
// %PrevMaskedReg:sreg_32(s1) = ANDN2 %PrevRegCopy, ExecReg - active lanes 0
// %CurMaskedReg:sreg_32(s1)  = AND %ExecReg, CurRegCopy - inactive lanes to 0
// %DstReg:sreg_32(s1)        = OR %PrevMaskedReg, CurMaskedReg
//
// DstReg = for active lanes rewrite bit in PrevReg with bit from CurReg
void DivergenceLoweringHelper::buildMergeLaneMasks(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, const DebugLoc &DL,
    Register DstReg, Register PrevReg, Register CurReg) {
  // DstReg = (PrevReg & !EXEC) | (CurReg & EXEC)
  // TODO: check if inputs are constants or results of a compare.

  Register PrevRegCopy = createLaneMaskReg(MRI, LaneMaskRegAttrs);
  auto [PrevMBB, AfterPrevReg] = getInsertAfterPtrs(MRI->getVRegDef(PrevReg));
  BuildMI(*PrevMBB, AfterPrevReg, DL, TII->get(AMDGPU::COPY), PrevRegCopy)
      .addReg(PrevReg);
  Register PrevMaskedReg = createLaneMaskReg(MRI, LaneMaskRegAttrs);
  BuildMI(MBB, I, DL, TII->get(AndN2Op), PrevMaskedReg)
      .addReg(PrevRegCopy)
      .addReg(ExecReg);

  Register CurRegCopy = createLaneMaskReg(MRI, LaneMaskRegAttrs);
  auto [CurMBB, AfterCurReg] = getInsertAfterPtrs(MRI->getVRegDef(CurReg));
  BuildMI(*CurMBB, AfterCurReg, DL, TII->get(AMDGPU::COPY), CurRegCopy)
      .addReg(CurReg);
  Register CurMaskedReg = createLaneMaskReg(MRI, LaneMaskRegAttrs);
  BuildMI(MBB, I, DL, TII->get(AndOp), CurMaskedReg)
      .addReg(ExecReg)
      .addReg(CurRegCopy);

  BuildMI(MBB, I, DL, TII->get(OrOp), DstReg)
      .addReg(PrevMaskedReg)
      .addReg(CurMaskedReg);
}

void DivergenceLoweringHelper::constrainAsLaneMask(Incoming &In) { return; }

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUGlobalISelDivergenceLowering, DEBUG_TYPE,
                      "AMDGPU GlobalISel divergence lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPUGlobalISelDivergenceLowering, DEBUG_TYPE,
                    "AMDGPU GlobalISel divergence lowering", false, false)

char AMDGPUGlobalISelDivergenceLowering::ID = 0;

char &llvm::AMDGPUGlobalISelDivergenceLoweringID =
    AMDGPUGlobalISelDivergenceLowering::ID;

FunctionPass *llvm::createAMDGPUGlobalISelDivergenceLoweringPass() {
  return new AMDGPUGlobalISelDivergenceLowering();
}

bool AMDGPUGlobalISelDivergenceLowering::runOnMachineFunction(
    MachineFunction &MF) {
  MachineDominatorTree &DT = getAnalysis<MachineDominatorTree>();
  MachinePostDominatorTree &PDT = getAnalysis<MachinePostDominatorTree>();
  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();

  DivergenceLoweringHelper Helper(&MF, &DT, &PDT, &MUI);

  bool Changed = false;
  Changed |= Helper.lowerPhis();
  return Changed;
}
