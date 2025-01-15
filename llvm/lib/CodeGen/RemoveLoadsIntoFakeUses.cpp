//===---- RemoveLoadsIntoFakeUses.cpp - Remove loads with no real uses ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The FAKE_USE instruction is used to preserve certain values through
/// optimizations for the sake of debugging. This may result in spilled values
/// being loaded into registers that are only used by FAKE_USEs; this is not
/// necessary for debugging purposes, because at that point the value must be on
/// the stack and hence available for debugging. Therefore, this pass removes
/// loads that are only used by FAKE_USEs.
///
/// This pass should run very late, to ensure that we don't inadvertently
/// shorten stack lifetimes by removing these loads, since the FAKE_USEs will
/// also no longer be in effect. Running immediately before LiveDebugValues
/// ensures that LDV will have accurate information of the machine location of
/// debug values.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "remove-loads-into-fake-uses"

STATISTIC(NumLoadsDeleted, "Number of dead load instructions deleted");
STATISTIC(NumFakeUsesDeleted, "Number of FAKE_USE instructions deleted");

class RemoveLoadsIntoFakeUses : public MachineFunctionPass {
public:
  static char ID;

  RemoveLoadsIntoFakeUses() : MachineFunctionPass(ID) {
    initializeRemoveLoadsIntoFakeUsesPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "Remove Loads Into Fake Uses";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

char RemoveLoadsIntoFakeUses::ID = 0;
char &llvm::RemoveLoadsIntoFakeUsesID = RemoveLoadsIntoFakeUses::ID;

INITIALIZE_PASS_BEGIN(RemoveLoadsIntoFakeUses, DEBUG_TYPE,
                      "Remove Loads Into Fake Uses", false, false)
INITIALIZE_PASS_END(RemoveLoadsIntoFakeUses, DEBUG_TYPE,
                    "Remove Loads Into Fake Uses", false, false)

bool RemoveLoadsIntoFakeUses::runOnMachineFunction(MachineFunction &MF) {
  // Only run this for functions that have fake uses.
  if (!MF.hasFakeUses() || skipFunction(MF.getFunction()))
    return false;

  bool AnyChanges = false;

  LiveRegUnits LivePhysRegs;
  const MachineRegisterInfo *MRI = &MF.getRegInfo();
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  const TargetInstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();

  SmallDenseMap<Register, SmallVector<MachineInstr *>> RegFakeUses;
  LivePhysRegs.init(*TRI);
  SmallVector<MachineInstr *, 16> Statepoints;
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    LivePhysRegs.addLiveOuts(*MBB);

    for (MachineInstr &MI : make_early_inc_range(reverse(*MBB))) {
      if (MI.isFakeUse()) {
        for (const MachineOperand &MO : MI.operands()) {
          // Track the Fake Uses that use this register so that we can delete
          // them if we delete the corresponding load.
          if (MO.isReg())
            RegFakeUses[MO.getReg()].push_back(&MI);
        }
        // Do not record FAKE_USE uses in LivePhysRegs so that we can recognize
        // otherwise-unused loads.
        continue;
      }

      // If the restore size is not std::nullopt then we are dealing with a
      // reload of a spilled register.
      if (MI.getRestoreSize(TII)) {
        Register Reg = MI.getOperand(0).getReg();
        assert(Reg.isPhysical() && "VReg seen in function with NoVRegs set?");
        // Don't delete live physreg defs, or any reserved register defs.
        if (!LivePhysRegs.available(Reg) || MRI->isReserved(Reg))
          continue;
        // There should be an exact match between the loaded register and the
        // FAKE_USE use. If not, this is a load that is unused by anything? It
        // should probably be deleted, but that's outside of this pass' scope.
        if (RegFakeUses.contains(Reg)) {
          LLVM_DEBUG(dbgs() << "RemoveLoadsIntoFakeUses: DELETING: " << MI);
          // It is possible that some DBG_VALUE instructions refer to this
          // instruction. They will be deleted in the live debug variable
          // analysis.
          MI.eraseFromParent();
          AnyChanges = true;
          ++NumLoadsDeleted;
          // Each FAKE_USE now appears to be a fake use of the previous value
          // of the loaded register; delete them to avoid incorrectly
          // interpreting them as such.
          for (MachineInstr *FakeUse : RegFakeUses[Reg]) {
            LLVM_DEBUG(dbgs()
                       << "RemoveLoadsIntoFakeUses: DELETING: " << *FakeUse);
            FakeUse->eraseFromParent();
          }
          NumFakeUsesDeleted += RegFakeUses[Reg].size();
          RegFakeUses[Reg].clear();
        }
        continue;
      }

      // In addition to tracking LivePhysRegs, we need to clear RegFakeUses each
      // time a register is defined, as existing FAKE_USEs no longer apply to
      // that register.
      if (!RegFakeUses.empty()) {
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.isDef()) {
            Register Reg = MO.getReg();
            assert(Reg.isPhysical() &&
                   "VReg seen in function with NoVRegs set?");
            for (MCRegUnit Unit : TRI->regunits(Reg))
              RegFakeUses.erase(Unit);
          }
        }
      }
      LivePhysRegs.stepBackward(MI);
    }
  }

  return AnyChanges;
}
