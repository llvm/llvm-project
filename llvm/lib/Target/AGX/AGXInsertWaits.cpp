//===-- AGXInsertWaits.cpp - Insert scoreboard waits ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory loads, stack loads and atomics write their results asynchronously,
// tracked by a scoreboard slot. Like SIInsertWaitcnts, this pass inserts a
// wait on the slot before any instruction reads or overwrites a register
// with a pending result, before barriers and at the end of every block with
// pending results (so that no wait is needed across control flow). At most
// eight requests are kept outstanding.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "agx-insert-waits"

namespace {

class AGXInsertWaits : public MachineFunctionPass {
public:
  static char ID;
  AGXInsertWaits() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Insert Waits"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXInsertWaits::ID = 0;

INITIALIZE_PASS(AGXInsertWaits, DEBUG_TYPE, "AGX Insert Waits", false, false)

FunctionPass *llvm::createAGXInsertWaitsPass() { return new AGXInsertWaits(); }

static constexpr unsigned MaxPending = 8;

static bool isBarrier(const MachineInstr &MI) {
  return MI.getOpcode() == AGX::THREADGROUP_BARRIER ||
         MI.getOpcode() == AGX::MEMORY_BARRIER;
}

bool AGXInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo &TII = *ST.getInstrInfo();
  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    BitVector Pending(TRI.getNumRegUnits());
    unsigned NumPending = 0;

    auto Wait = [&](MachineBasicBlock::iterator I) {
      BuildMI(MBB, I, DebugLoc(), TII.get(AGX::WAIT)).addImm(0);
      Pending.reset();
      NumPending = 0;
      Changed = true;
    };

    auto Touches = [&](const MachineInstr &MI) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.getReg() == AGX::R0L)
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
          if (Pending.test(static_cast<unsigned>(U)))
            return true;
      }
      return false;
    };

    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AGX::WAIT) {
        Pending.reset();
        NumPending = 0;
        continue;
      }
      if (NumPending &&
          (Touches(MI) || isBarrier(MI) || MI.isTerminator() ||
           (AGXInstrInfo::isAsync(MI) && NumPending >= MaxPending)))
        Wait(MI.getIterator());
      // Asynchronous instructions without a result (atomics whose old value
      // is unused) leave nothing to wait for, like stores.
      if (AGXInstrInfo::isAsync(MI) && MI.getNumExplicitDefs()) {
        for (const MachineOperand &MO : MI.defs())
          for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
            Pending.set(static_cast<unsigned>(U));
        ++NumPending;
      }
    }
    if (NumPending)
      Wait(MBB.end());
  }
  return Changed;
}
