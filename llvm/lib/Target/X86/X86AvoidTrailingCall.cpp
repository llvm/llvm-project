//===----- X86AvoidTrailingCall.cpp - Insert int3 after trailing calls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Windows x64 unwinder has trouble unwinding the stack when a return
// address points to the end of the function. This pass maintains the invariant
// that every return address is inside the bounds of its parent function or
// funclet by inserting int3 if the last instruction would otherwise be a call.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

#define DEBUG_TYPE "x86-avoid-trailing-call"

using namespace llvm;

namespace {

class X86AvoidTrailingCallPass : public MachineFunctionPass {
public:
  X86AvoidTrailingCallPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  StringRef getPassName() const override {
    return "X86 avoid trailing call pass";
  }
  static char ID;
};

char X86AvoidTrailingCallPass::ID = 0;

} // end anonymous namespace

FunctionPass *llvm::createX86AvoidTrailingCallPass() {
  return new X86AvoidTrailingCallPass();
}

// A real instruction is a non-meta, non-pseudo instruction.  Some pseudos
// expand to nothing, and some expand to code. This logic conservatively assumes
// they might expand to nothing.
static bool isRealInstruction(MachineInstr &MI) {
  return !MI.isPseudo() && !MI.isMetaInstruction();
}

// Return true if this is a call instruction, but not a tail call.
static bool isCallInstruction(const MachineInstr &MI) {
  return MI.isCall() && !MI.isReturn();
}

bool X86AvoidTrailingCallPass::runOnMachineFunction(MachineFunction &MF) {
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo &TII = *STI.getInstrInfo();
  assert(STI.isTargetWin64() && "pass only runs on Win64");

  // FIXME: Perhaps this pass should also replace SEH_Epilogue by inserting nops
  // before epilogues.

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Look for basic blocks that precede funclet entries or are at the end of
    // the function.
    MachineBasicBlock *NextMBB = MBB.getNextNode();
    if (NextMBB && !NextMBB->isEHFuncletEntry())
      continue;

    // Find the last real instruction in this block, or previous blocks if this
    // block is empty.
    MachineBasicBlock::reverse_iterator LastRealInstr;
    for (MachineBasicBlock &RMBB :
         make_range(MBB.getReverseIterator(), MF.rend())) {
      LastRealInstr = llvm::find_if(reverse(RMBB), isRealInstruction);
      if (LastRealInstr != RMBB.rend())
        break;
    }

    // Do nothing if this function or funclet has no instructions.
    if (LastRealInstr == MF.begin()->rend())
      continue;

    // If this is a call instruction, insert int3 right after it with the same
    // DebugLoc. Convert back to a forward iterator and advance the insertion
    // position once.
    if (isCallInstruction(*LastRealInstr)) {
      LLVM_DEBUG({
        dbgs() << "inserting int3 after trailing call instruction:\n";
        LastRealInstr->dump();
        dbgs() << '\n';
      });

      MachineBasicBlock::iterator MBBI = std::next(LastRealInstr.getReverse());
      BuildMI(*LastRealInstr->getParent(), MBBI, LastRealInstr->getDebugLoc(),
              TII.get(X86::INT3));
      Changed = true;
    }
  }

  return Changed;
}
