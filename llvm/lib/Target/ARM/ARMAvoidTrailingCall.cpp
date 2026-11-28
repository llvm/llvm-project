//===-- ARMAvoidTrailingCall.cpp - Trap after trailing calls on Windows ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Windows unwinder finds the function or funclet of a frame from its
// return address. A call that ends a function or funclet returns into the
// next one, so put a trap after it.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "arm-avoid-trailing-call"
#define ARM_AVOID_TRAILING_CALL_NAME "ARM avoid trailing call"

namespace {
class ARMAvoidTrailingCall : public MachineFunctionPass {
public:
  static char ID;
  ARMAvoidTrailingCall() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return ARM_AVOID_TRAILING_CALL_NAME;
  }
};
} // end anonymous namespace

char ARMAvoidTrailingCall::ID = 0;

INITIALIZE_PASS(ARMAvoidTrailingCall, DEBUG_TYPE, ARM_AVOID_TRAILING_CALL_NAME,
                false, false)

FunctionPass *llvm::createARMAvoidTrailingCallPass() {
  return new ARMAvoidTrailingCall();
}

bool ARMAvoidTrailingCall::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.usesWindowsUnwindInfo())
    return false;

  const ARMSubtarget &STI = MF.getSubtarget<ARMSubtarget>();
  const ARMBaseInstrInfo &TII = *STI.getInstrInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *Next = MBB.getNextNode();
    if (Next && !Next->isEHFuncletEntry())
      continue;
    auto Last = find_if(reverse(MBB), [](const MachineInstr &MI) {
      return !MI.isMetaInstruction() && !isSEHInstruction(MI);
    });
    // An empty block needs a trap too: an invoke of a noreturn function falls
    // through to its empty normal destination.
    bool IsEmpty = Last == MBB.rend();
    bool IsCall = !IsEmpty && Last->isCall() && !Last->isReturn();
    if (!IsEmpty && !IsCall)
      continue;
    BuildMI(MBB, MBB.end(), DebugLoc(),
            TII.get(STI.isThumb() ? ARM::tTRAP : ARM::TRAP));
    Changed = true;
  }
  return Changed;
}
