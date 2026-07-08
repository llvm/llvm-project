//===-- ARMPostRAPeephole.cpp - Sink CMP instructions past COPYs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass sinks tCMP instructions past COPY instructions to reduce CPSR
// liveness. This is particularly important for Thumb-1. In Thumb-1, low-to-low
// register moves (COPY) are expanded to the efficient 'movs' instruction if
// CPSR is dead. If CPSR is live, Thumb-1 copy lowering cannot use the efficient
// movs encoding for low-register copies and may require an alternate sequence.
// By sinking the CMP, we shorten the CPSR live range and allow 'movs' to be
// emitted.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "arm-postra-peephole"

namespace {
struct ARMPostRAPeephole : public MachineFunctionPass {
  static char ID;
  ARMPostRAPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return "ARM Post-RA Peephole"; }
};

char ARMPostRAPeephole::ID = 0;
} // namespace

bool ARMPostRAPeephole::runOnMachineFunction(MachineFunction &MF) {
  const ARMSubtarget &STI = MF.getSubtarget<ARMSubtarget>();
  if (!STI.isThumb1Only() || STI.hasV6Ops())
    return false;

  bool Changed = false;
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;
      ++I;

      switch (MI.getOpcode()) {
      case ARM::tCMPr:
      case ARM::tCMPi8:
      case ARM::tCMPhir:
      case ARM::tCMN:
      case ARM::tTST: {
        auto NextI = I;
        bool CanSink = true;
        bool AnyBeneficial = false;

        while (NextI != E && NextI->isCopy()) {
          MachineInstr &Copy = *NextI;

          // Don't move the compare across a COPY that reads or writes CPSR.
          if (Copy.modifiesRegister(ARM::CPSR, TRI) ||
              Copy.readsRegister(ARM::CPSR, TRI)) {
            CanSink = false;
            break;
          }

          // Check if COPY modifies any register used by the CMP
          for (const MachineOperand &MO : MI.operands()) {
            if (!MO.isReg() || !MO.isUse())
              continue;
            if (Copy.modifiesRegister(MO.getReg(), TRI)) {
              CanSink = false;
              break;
            }
          }

          if (!CanSink)
            break;

          assert(Copy.isCopy());
          assert(Copy.getOperand(0).isReg());
          assert(Copy.getOperand(1).isReg());

          // copyPhysReg only ever consults CPSR liveness for a low-to-low copy;
          // anything else gets tMOVr regardless of where the CMP ends up, so
          // it's safe to sink past but shouldn't by itself justify sinking.
          Register Dst = Copy.getOperand(0).getReg();
          Register Src = Copy.getOperand(1).getReg();
          if (!ARM::hGPRRegClass.contains(Src) &&
              ARM::tGPRRegClass.contains(Dst))
            AnyBeneficial = true;

          ++NextI;
        }

        if (!CanSink || NextI == I || !AnyBeneficial)
          continue;

        // Clear kill flags on the CMP since we are moving it later.
        MI.clearKillInfo();

        // Also selectively clear kill flags on the COPYs we skip over,
        // but only for registers that the CMP uses.
        for (auto SkipI = I; SkipI != NextI; ++SkipI) {
          for (const MachineOperand &MO : MI.operands()) {
            if (MO.isReg() && MO.isUse())
              SkipI->clearRegisterKills(MO.getReg(), TRI);
          }
        }

        MBB.splice(NextI, &MBB, MI.getIterator());
        Changed = true;
        break;
      }
      default:
        continue;
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createARMPostRAPeepholePass() {
  return new ARMPostRAPeephole();
}

INITIALIZE_PASS(ARMPostRAPeephole, DEBUG_TYPE, "ARM Post-RA Peephole", false,
                false)
