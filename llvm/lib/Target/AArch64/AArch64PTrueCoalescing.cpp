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

class AArch64PTrueCoalescingImpl {
  const AArch64InstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *MDT = nullptr;

public:
  explicit AArch64PTrueCoalescingImpl(MachineDominatorTree &MDT) : MDT(&MDT) {}

  bool run(MachineFunction &MF);

private:
  struct PredicateInfo {
    // Instruction that created the predicate.
    MachineInstr *MI = nullptr;
    // Element size of the MI.
    unsigned ElementSize = AArch64::ElementSizeNone;
    // Smallest element size of all instructions that use the predicate.
    unsigned SmallestUsedElementSize = AArch64::ElementSizeNone;

    bool isValid() const {
      assert(ElementSize != AArch64::ElementSizeNone &&
             "PTRUE missing element size!");
      return MI && SmallestUsedElementSize != AArch64::ElementSizeNone;
    }

    void invalidate() {
      assert(isValid());
      MI = nullptr;
    }
  };

  std::optional<PredicateInfo> createPredicateInfo(MachineInstr &MI) const {
    // TODO: Extend support beyond "PTRUE all"?
    if (!isPTrueOpcode(MI.getOpcode()) || MI.getOperand(1).getImm() != 31)
      return std::nullopt;

    Register Pred = MI.getOperand(0).getReg();
    unsigned SmallestUsedElementSize = getSmallestElementSizeInUse(Pred);
    unsigned ElementSize = TII->getElementSizeForOpcode(MI.getOpcode());
    assert(ElementSize != AArch64::ElementSizeNone &&
           "PTRUE missing element size!");

    if (SmallestUsedElementSize == AArch64::ElementSizeNone)
      return std::nullopt;

    return PredicateInfo{&MI, ElementSize, SmallestUsedElementSize};
  }

  // Return the smallest element size of all instructions that use Reg, or
  // AArch64::ElementSizeNone when unknown.
  unsigned getSmallestElementSizeInUse(Register Reg) const;

  // Try to replace uses of CanPred with DomPred. In some cases that means
  // modifying DomPred to support smaller element types.
  bool tryCoalesce(PredicateInfo &DomPred, PredicateInfo &CanPred) const;
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

unsigned
AArch64PTrueCoalescingImpl::getSmallestElementSizeInUse(Register Reg) const {
  // SSA form only applies to virtual registers.
  if (!Reg.isVirtual())
    return AArch64::ElementSizeNone;

  unsigned SmallestElementSize = AArch64::ElementSizeNone;

  for (MachineOperand &UseMO : MRI->use_nodbg_operands(Reg)) {
    assert(UseMO.getSubReg() == 0 && "Unexpected SubReg!");
    MachineInstr *UseMI = UseMO.getParent();

    unsigned ElementSize = TII->getElementSizeForOpcode(UseMI->getOpcode());
    if (ElementSize == AArch64::ElementSizeNone)
      return AArch64::ElementSizeNone;

    if (SmallestElementSize == AArch64::ElementSizeNone ||
        SmallestElementSize > ElementSize)
      SmallestElementSize = ElementSize;
  }

  return SmallestElementSize;
}

bool AArch64PTrueCoalescingImpl::tryCoalesce(PredicateInfo &DomPI,
                                             PredicateInfo &CanPI) const {
  assert(DomPI.isValid() && CanPI.isValid());
  MachineInstr *DomMI = DomPI.MI;
  MachineInstr *CanMI = CanPI.MI;

  if (DomMI == CanMI || !MDT->dominates(DomMI, CanMI))
    return false;

  // A predicate's observable shape is the larger of the element size of the
  // instruction writing the predicate and the one reading it. First check if
  // DomPI can replace CanPI as-is for CanPI's users. If not, try changing DomPI
  // to CanPI's element size, but only if DomPI's existing users would observe
  // the same shape after that change.

  bool MutateDomPTrue = false;
  if (std::max(CanPI.ElementSize, CanPI.SmallestUsedElementSize) !=
      std::max(DomPI.ElementSize, CanPI.SmallestUsedElementSize)) {
    if (std::max(CanPI.ElementSize, DomPI.SmallestUsedElementSize) !=
        std::max(DomPI.ElementSize, DomPI.SmallestUsedElementSize))
      return false;

    MutateDomPTrue = true;
  }

  Register DomReg = DomMI->getOperand(0).getReg();
  Register CanReg = CanMI->getOperand(0).getReg();
  if (!MRI->constrainRegClass(DomReg, MRI->getRegClass(CanReg)))
    return false;

  LLVM_DEBUG(dbgs() << "Coalescing PTRUE: " << CanMI);
  LLVM_DEBUG(dbgs() << "            with: " << DomMI);

  if (MutateDomPTrue) {
    LLVM_DEBUG(dbgs() << "        updated: " << DomMI);
    DomMI->setDesc(TII->get(CanMI->getOpcode()));
    DomPI.ElementSize = CanPI.ElementSize;
    LLVM_DEBUG(dbgs() << "             to: " << DomMI);
  }

  MRI->replaceRegWith(CanReg, DomReg);
  MRI->clearKillFlags(DomReg);
  CanMI->eraseFromParent();

  // Update DomPI based on uses inherited from CanPI.
  if (CanPI.SmallestUsedElementSize < DomPI.SmallestUsedElementSize)
    DomPI.SmallestUsedElementSize = CanPI.SmallestUsedElementSize;
  CanPI.invalidate();
  return true;
}

bool AArch64PTrueCoalescingImpl::run(MachineFunction &MF) {
  if (!EnablePTrueCoalescing ||
      !MF.getSubtarget<AArch64Subtarget>().isSVEorStreamingSVEAvailable())
    return false;

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();

  assert(MRI->isSSA() && "Expected to be run on SSA form!");

  // TODO: Until we prove candidates share the same VG definition, do not
  // coalesce in functions that define VG.
  if (!MRI->def_empty(AArch64::VG))
    return false;

  // A list of predicate setting instructions with some usage information.
  SmallVector<PredicateInfo, 8> PIs;

  // Build a list of predicates whose uses all have a known size.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (auto PI = createPredicateInfo(MI))
        PIs.push_back(*PI);

  LLVM_DEBUG(dbgs() << "Coalescable PTRUE candidates: " << PIs.size() << "\n");
  bool Changed = false;

  for (PredicateInfo &DominantPI : PIs) {
    if (!DominantPI.isValid())
      continue;

    for (PredicateInfo &CandidatePI : PIs) {
      if (!CandidatePI.isValid())
        continue;

      Changed |= tryCoalesce(DominantPI, CandidatePI);
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

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineDominatorTreeAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
