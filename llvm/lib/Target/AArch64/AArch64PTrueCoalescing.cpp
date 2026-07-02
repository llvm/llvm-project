//===- AArch64PTrueCoalescing.cpp - Coalesce SVE PTRUEs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass coalesces compatible all-active SVE PTRUE instructions.
//
// Consider two all-active PTRUE instructions X and Y with element sizes XSize
// and YSize. If X dominates Y and XSize <= YSize, then every predicate bit that
// Y sets is also set by X. In that case, uses of Y can be redirected to X as
// long as each user of Y only reads predicate bits at YSize granularity or
// larger.
//
// If the dominating PTRUE has a larger element size, we can coalesce the pair
// by changing the dominating PTRUE to the smaller element size, provided that
// all of its existing users are also safe with that granularity.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-ptrue-coalesce"

static cl::opt<bool> EnablePTrueCoalescing(
    "aarch64-enable-ptrue-coalescing", cl::init(false), cl::Hidden,
    cl::desc("Enable coalescing of compatible AArch64 SVE PTRUE instructions"));

namespace {

static bool isAllActivePTrue(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::PTRUE_B:
  case AArch64::PTRUE_H:
  case AArch64::PTRUE_S:
  case AArch64::PTRUE_D:
    return MI.getOperand(1).getImm() == 31;
  }
}

class AArch64PTrueCoalescingImpl {
  const AArch64InstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *MDT = nullptr;

public:
  explicit AArch64PTrueCoalescingImpl(MachineDominatorTree &MDT) : MDT(&MDT) {}

  bool run(MachineFunction &MF);

private:
  bool allUsersSafeForElementSize(Register Reg, uint64_t ElementSize) const;
  bool tryCoalesce(MachineInstr &DomPTrue, MachineInstr &PTrue) const;
};

class AArch64PTrueCoalescingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AArch64PTrueCoalescingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AArch64 PTRUE Coalescing"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char AArch64PTrueCoalescingLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AArch64PTrueCoalescingLegacy, DEBUG_TYPE,
                      "AArch64 PTRUE Coalescing", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AArch64PTrueCoalescingLegacy, DEBUG_TYPE,
                    "AArch64 PTRUE Coalescing", false, false)

bool AArch64PTrueCoalescingImpl::allUsersSafeForElementSize(
    Register Reg, uint64_t ElementSize) const {
  for (MachineOperand &UseMO : MRI->use_nodbg_operands(Reg)) {
    if (UseMO.getSubReg())
      return false;

    MachineInstr *UseMI = UseMO.getParent();
    uint64_t UseElementSize = TII->getElementSizeForOpcode(UseMI->getOpcode());
    if (UseElementSize == AArch64::ElementSizeNone ||
        UseElementSize < ElementSize)
      return false;
  }

  return true;
}

bool AArch64PTrueCoalescingImpl::tryCoalesce(MachineInstr &DomPTrue,
                                             MachineInstr &PTrue) const {
  assert(isAllActivePTrue(DomPTrue) && "Expected all-active PTRUE");
  assert(isAllActivePTrue(PTrue) && "Expected all-active PTRUE");

  if (&DomPTrue == &PTrue || !MDT->dominates(&DomPTrue, &PTrue))
    return false;

  Register DomReg = DomPTrue.getOperand(0).getReg();
  Register Reg = PTrue.getOperand(0).getReg();

  uint64_t DomElementSize = TII->getElementSizeForOpcode(DomPTrue.getOpcode());
  uint64_t ElementSize = TII->getElementSizeForOpcode(PTrue.getOpcode());
  assert(DomElementSize != AArch64::ElementSizeNone &&
         "PTRUE should have an element size");
  assert(ElementSize != AArch64::ElementSizeNone &&
         "PTRUE should have an element size");

  if (!MRI->constrainRegClass(DomReg, MRI->getRegClass(Reg)))
    return false;

  bool MutateDomPTrue = false;
  if (DomElementSize < ElementSize) {
    // DomPTrue sets all lanes set by PTrue, plus extra lanes. Prefer to reuse
    // DomPTrue as-is when PTrue's users do not observe those extra lanes.
    if (!allUsersSafeForElementSize(Reg, ElementSize)) {
      if (!allUsersSafeForElementSize(DomReg, ElementSize))
        return false;
      MutateDomPTrue = true;
    }
  } else if (DomElementSize > ElementSize) {
    if (!allUsersSafeForElementSize(DomReg, ElementSize))
      return false;
    MutateDomPTrue = true;
  }

  LLVM_DEBUG(dbgs() << "Coalescing PTRUE: " << PTrue);
  LLVM_DEBUG(dbgs() << "            with: " << DomPTrue);

  if (MutateDomPTrue) {
    LLVM_DEBUG(dbgs() << "        updated: " << DomPTrue);
    DomPTrue.setDesc(TII->get(PTrue.getOpcode()));
    LLVM_DEBUG(dbgs() << "             to: " << DomPTrue);
  }

  MRI->replaceRegWith(Reg, DomReg);
  MRI->clearKillFlags(DomReg);
  PTrue.eraseFromParent();
  return true;
}

bool AArch64PTrueCoalescingImpl::run(MachineFunction &MF) {
  if (!EnablePTrueCoalescing ||
      !MF.getSubtarget<AArch64Subtarget>().isSVEorStreamingSVEAvailable())
    return false;

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();

  assert(MRI->isSSA() && "Expected to be run on SSA form!");

  SmallVector<MachineInstr *, 8> PTrues;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (isAllActivePTrue(MI))
        PTrues.push_back(&MI);

  bool Changed = false;
  auto tryCoalescePTrue = [&](MachineInstr *PTrue) {
    for (MachineInstr *Candidate : PTrues)
      if (tryCoalesce(*Candidate, *PTrue))
        return true;
    return false;
  };

  for (auto I = PTrues.begin(); I != PTrues.end();) {
    if (tryCoalescePTrue(*I)) {
      I = PTrues.erase(I);
      Changed = true;
    } else {
      ++I;
    }
  }

  return Changed;
}

bool AArch64PTrueCoalescingLegacy::runOnMachineFunction(MachineFunction &MF) {
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  return AArch64PTrueCoalescingImpl(MDT).run(MF);
}

FunctionPass *llvm::createAArch64PTrueCoalescingLegacyPass() {
  return new AArch64PTrueCoalescingLegacy();
}

PreservedAnalyses
AArch64PTrueCoalescingPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  const bool Changed = AArch64PTrueCoalescingImpl(MDT).run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MachineDominatorTreeAnalysis>();
  return PA;
}
