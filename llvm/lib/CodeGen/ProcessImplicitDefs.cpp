//===---------------------- ProcessImplicitDefs.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ProcessImplicitDefs.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "process-implicit-defs"

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

  void processImplicitDef(MachineInstr *MI, const ImplicitDefInfo &Info);

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

static bool canBecomeImplicitDef(const MachineInstr &MI) {
  return MI.isCopyLike() || MI.isInsertSubreg() || MI.isRegSequence() ||
         MI.isPHI();
}

ImplicitDefInfo::ImplicitDefInfo(const MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const MachineInstr *> WorkList;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (MI.isImplicitDef() && MI.getOperand(0).getReg().isVirtual())
        WorkList.push_back(&MI);

  while (!WorkList.empty()) {
    const MachineInstr *MI = WorkList.pop_back_val();
    Register Reg = MI->getOperand(0).getReg();
    bool AllUsesUndef = true;
    for (const MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      const MachineInstr &UseMI = *MO.getParent();
      if (UseMI.hasTiedAndOtherReadOf(Reg, MO.getSubReg())) {
        AllUsesUndef = false;
        continue;
      }
      UndefUses.insert(&MO);
      if (canBecomeImplicitDef(UseMI) && !NewImplicitDefs.contains(&UseMI) &&
          all_of(UseMI.all_uses(), [&](const MachineOperand &Use) {
            return !Use.readsReg() || UndefUses.contains(&Use);
          })) {
        NewImplicitDefs.insert(&UseMI);
        if (UseMI.getOperand(0).getReg().isVirtual())
          WorkList.push_back(&UseMI);
      }
    }
    if (AllUsesUndef)
      ErasedImplicitDefs.insert(MI);
  }
}

void ProcessImplicitDefs::processImplicitDef(MachineInstr *MI,
                                             const ImplicitDefInfo &Info) {
  LLVM_DEBUG(dbgs() << "Processing " << *MI);
  Register Reg = MI->getOperand(0).getReg();
  // Trim any extra operands.
  for (unsigned i = MI->getNumOperands() - 1; i; --i)
    MI->removeOperand(i);

  MachineBasicBlock *MBB = MI->getParent();

  if (Reg.isVirtual()) {
    if (Info.isErased(*MI)) {
      MI->eraseFromParent();
      return;
    }
    // A kept PHI, now an IMPLICIT_DEF, leaves the PHI block.
    MachineBasicBlock::iterator Next = std::next(MI->getIterator());
    if (Next != MBB->end() && Next->isPHI())
      MBB->splice(MBB->SkipPHIsAndLabels(Next), MBB, MI);
    return;
  }

  // This is a physreg implicit-def.
  // Try to add undef flag to all uses.  If all uses are updated remove
  // implicit-def.
  MachineBasicBlock::instr_iterator SearchMI = MI->getIterator();
  MachineBasicBlock::instr_iterator SearchE = MBB->instr_end();
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
  if (ImplicitDefIsDead || (SearchedWholeBlock && MBB->succ_empty())) {
    LLVM_DEBUG(dbgs() << "Deleting implicit-def: " << *MI);
    MI->eraseFromParent();
    return;
  }

  if (SearchedWholeBlock && MRI->tracksLiveness() && !MRI->isReserved(Reg) &&
      none_of(MBB->successors(), [&](MachineBasicBlock *Succ) {
        return Succ->isLiveInOverlappedWith(Reg, *TRI);
      }))
    MI->eraseFromParent();
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

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  ImplicitDefInfo Info(MF);
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.all_uses())
        if (Info.isUndef(MO))
          MO.setIsUndef();
      if (Info.becomesImplicitDef(MI)) {
        LLVM_DEBUG(dbgs() << "Converting to IMPLICIT_DEF: " << MI);
        MI.setDesc(TII->get(TargetOpcode::IMPLICIT_DEF));
      }
    }
  }

  // Bottom-up, so that erased IMPLICIT_DEFs don't count towards the search
  // limit for physical registers.
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {
      if (MI.isImplicitDef()) {
        processImplicitDef(&MI, Info);
        Changed = true;
      }
    }
  }
  return Changed;
}
