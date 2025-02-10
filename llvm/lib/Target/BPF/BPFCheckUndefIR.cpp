//===---------------- BPFAdjustOpt.cpp - Adjust Optimization --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check 'undef' and 'unreachable' IRs and issue proper warnings.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "bpf-check-undef-ir"

using namespace llvm;

namespace {

class BPFCheckUndefIR final : public ModulePass {
  bool runOnModule(Module &F) override;

public:
  static char ID;
  BPFCheckUndefIR() : ModulePass(ID) {}

private:
  void BPFCheckUndefIRImpl(Function &F);
  void BPFCheckInst(Function &F, BasicBlock &BB, Instruction &I);
  void HandleReturnInsn(Function &F, ReturnInst *I);
  void HandleUnreachableInsn(Function &F, BasicBlock &BB, Instruction &I);
};
} // End anonymous namespace

char BPFCheckUndefIR::ID = 0;
INITIALIZE_PASS(BPFCheckUndefIR, DEBUG_TYPE, "BPF Check Undef IRs", false,
                false)

ModulePass *llvm::createBPFCheckUndefIR() { return new BPFCheckUndefIR(); }

void BPFCheckUndefIR::HandleReturnInsn(Function &F, ReturnInst *I) {
  Value *RetValue = I->getReturnValue();
  // PoisonValue is a special UndefValue where compiler intentionally to
  // poisons a value since it shouldn't be used.
  if (!RetValue || isa<PoisonValue>(RetValue) || !isa<UndefValue>(RetValue))
    return;

  dbgs() << "WARNING: return undefined value in func " << F.getName()
         << ", due to uninitialized variable?\n";
}

void BPFCheckUndefIR::HandleUnreachableInsn(Function &F, BasicBlock &BB,
                                            Instruction &I) {
  // LLVM may create a switch statement with default to a 'unreachable' basic
  // block. Do not warn for such cases.
  unsigned NumNoSwitches = 0, NumSwitches = 0;
  for (BasicBlock *Pred : predecessors(&BB)) {
    const Instruction *Term = Pred->getTerminator();
    if (Term && Term->getOpcode() == Instruction::Switch) {
      NumSwitches++;
      continue;
    }
    NumNoSwitches++;
  }
  if (NumSwitches > 0 && NumNoSwitches == 0)
    return;

  // If the previous insn is no return, do not warn for such cases.
  // One example is __bpf_unreachable from libbpf bpf_headers.h.
  Instruction *PrevI = I.getPrevNonDebugInstruction();
  if (PrevI) {
    auto *CI = dyn_cast<CallInst>(PrevI);
    if (CI && CI->doesNotReturn())
      return;
  }

  dbgs() << "WARNING: unreachable in func " << F.getName()
         << ", due to uninitialized variable?\n";
}

void BPFCheckUndefIR::BPFCheckInst(Function &F, BasicBlock &BB,
                                   Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::Ret:
    HandleReturnInsn(F, cast<ReturnInst>(&I));
    break;
  case Instruction::Unreachable:
    HandleUnreachableInsn(F, BB, I);
    break;
  default:
    break;
  }
}

void BPFCheckUndefIR::BPFCheckUndefIRImpl(Function &F) {
  // A 'unreachable' will be added to the end of naked function.
  // Let ignore these naked functions.
  if (F.hasFnAttribute(Attribute::Naked))
    return;

  for (auto &BB : F) {
    for (auto &I : BB)
      BPFCheckInst(F, BB, I);
  }
}

bool BPFCheckUndefIR::runOnModule(Module &M) {
  for (Function &F : M)
    BPFCheckUndefIRImpl(F);
  return false;
}
