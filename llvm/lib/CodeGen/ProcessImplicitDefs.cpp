//===---------------------- ProcessImplicitDefs.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ProcessImplicitDefs.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "processimpdefs"

namespace {
/// Process IMPLICIT_DEF instructions and make sure there is one implicit_def
/// for each use. Add isUndef marker to implicit_def defs and their uses.
class ProcessImplicitDefsLegacy : public MachineFunctionPass {
public:
  static char ID;

  ProcessImplicitDefsLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

class ProcessImplicitDefs {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  SmallSetVector<MachineInstr *, 16> WorkList;

  void processImplicitDef(MachineInstr *MI);
  bool canTurnIntoImplicitDef(MachineInstr *MI);

public:
  bool run(MachineFunction &MF);
};
} // end anonymous namespace

char ProcessImplicitDefsLegacy::ID = 0;
char &llvm::ProcessImplicitDefsID = ProcessImplicitDefsLegacy::ID;

INITIALIZE_PASS(ProcessImplicitDefsLegacy, DEBUG_TYPE,
                "Process Implicit Definitions", false, false)

void ProcessImplicitDefsLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<AAResultsWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool ProcessImplicitDefs::canTurnIntoImplicitDef(MachineInstr *MI) {
  if (!MI->isCopyLike() &&
      !MI->isInsertSubreg() &&
      !MI->isRegSequence() &&
      !MI->isPHI())
    return false;
  for (const MachineOperand &MO : MI->all_uses())
    if (MO.readsReg())
      return false;
  return true;
}

void ProcessImplicitDefs::processImplicitDef(MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "Processing " << *MI);
  Register Reg = MI->getOperand(0).getReg();

  if (Reg.isVirtual()) {
    // For virtual registers, mark all uses as <undef>, and convert users to
    // implicit-def when possible.
    for (MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
      MO.setIsUndef();
      MachineInstr *UserMI = MO.getParent();
      if (!canTurnIntoImplicitDef(UserMI))
        continue;
      LLVM_DEBUG(dbgs() << "Converting to IMPLICIT_DEF: " << *UserMI);
      UserMI->setDesc(TII->get(TargetOpcode::IMPLICIT_DEF));
      WorkList.insert(UserMI);
    }
    MI->eraseFromParent();
    return;
  }

  // This is a physreg implicit-def.
  // Trim any extra operands.
  for (unsigned i = MI->getNumOperands() - 1; i; --i)
    MI->removeOperand(i);

  // Try to add undef flag to all uses.  If all uses are updated remove
  // implicit-def.
  MachineBasicBlock::instr_iterator SearchMI = MI->getIterator();
  MachineBasicBlock::instr_iterator SearchE = MI->getParent()->instr_end();
  bool ImplicitDefIsDead = false;
  bool SearchedWholeBlock = true;
  constexpr unsigned SearchLimit = 35;
  unsigned Count = 0;
  for (++SearchMI; SearchMI != SearchE; ++SearchMI) {
    if (SearchMI->isDebugInstr())
      continue;
    if (++Count > SearchLimit) {
      SearchedWholeBlock = false;
      break;
    }
    for (MachineOperand &MO : SearchMI->operands()) {
      if (!MO.isReg())
        continue;
      Register SearchReg = MO.getReg();
      if (!SearchReg.isPhysical() || !TRI->regsOverlap(Reg, SearchReg))
        continue;
      // SearchMI uses or redefines Reg. Set <undef> flags on all uses.
      if (MO.isUse()) {
        if (TRI->isSubRegisterEq(Reg, SearchReg)) {
          MO.setIsUndef();
        } else {
          // Use is larger than Reg.  It is not safe to add undef to this use.
          return;
        }
      }
      if (MO.isDef()) {
        if (TRI->isSubRegisterEq(SearchReg, Reg)) {
          ImplicitDefIsDead = true;
        } else {
          // Reg is larger than definition.  It is not safe to add undef to any
          // subsequent uses of Reg.
          return;
        }
      }
    }
    if (ImplicitDefIsDead) {
      LLVM_DEBUG(dbgs() << "Physreg redefine: " << *SearchMI);
      break;
    }
  }

  // If we have added an undef flag to all uses (i.e. we have found a redefining
  // MI or there are no successors), we can erase the IMPLICIT_DEF.
  if (ImplicitDefIsDead ||
      (SearchedWholeBlock && MI->getParent()->succ_empty())) {
    MI->eraseFromParent();
    LLVM_DEBUG(dbgs() << "Deleting implicit-def: " << *MI);
  }
}

bool ProcessImplicitDefsLegacy::runOnMachineFunction(MachineFunction &MF) {
  return ProcessImplicitDefs().run(MF);
}

PreservedAnalyses
ProcessImplicitDefsPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  if (!ProcessImplicitDefs().run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<AAManager>();
}

/// processImplicitDefs - Process IMPLICIT_DEF instructions and turn them into
/// <undef> operands.
bool ProcessImplicitDefs::run(MachineFunction &MF) {

  LLVM_DEBUG(dbgs() << "********** PROCESS IMPLICIT DEFS **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  bool Changed = false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  assert(WorkList.empty() && "Inconsistent worklist state");

  for (MachineBasicBlock &MBB : MF) {
    // Scan the basic block for implicit defs.
    for (MachineInstr &MI : MBB)
      if (MI.isImplicitDef())
        WorkList.insert(&MI);

    if (WorkList.empty())
      continue;

    LLVM_DEBUG(dbgs() << printMBBReference(MBB) << " has " << WorkList.size()
                      << " implicit defs.\n");
    Changed = true;

    // Drain the WorkList to recursively process any new implicit defs.
    do processImplicitDef(WorkList.pop_back_val());
    while (!WorkList.empty());
  }
  return Changed;
}
