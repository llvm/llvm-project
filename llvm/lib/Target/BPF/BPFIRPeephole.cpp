//===------------ BPFIRPeephole.cpp - IR Peephole Transformation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR level peephole optimization, specifically removing @llvm.stacksave() and
// @llvm.stackrestore().
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "bpf-ir-peephole"

using namespace llvm;

namespace {

static bool BPFIRPeepholeImpl(Function &F) {
  LLVM_DEBUG(dbgs() << "******** BPF IR Peephole ********\n");

  Instruction *ToErase = nullptr;
  for (auto &BB : F) {
    for (auto &I : BB) {
      // The following code pattern is handled:
      //     %3 = call i8* @llvm.stacksave()
      //     store i8* %3, i8** %saved_stack, align 8
      //     ...
      //     %4 = load i8*, i8** %saved_stack, align 8
      //     call void @llvm.stackrestore(i8* %4)
      //     ...
      // The goal is to remove the above four instructions,
      // so we won't have instructions with r11 (stack pointer)
      // if eventually there is no variable length stack allocation.
      // InstrCombine also tries to remove the above instructions,
      // if it is proven safe (constant alloca etc.), but depending
      // on code pattern, it may still miss some.
      //
      // With unconditionally removing these instructions, if alloca is
      // constant, we are okay then. Otherwise, SelectionDag will complain
      // since BPF does not support dynamic allocation yet.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand())) {
          if (!GV->getName().equals("llvm.stacksave"))
            continue;
          if (!Call->hasOneUser())
            continue;
          auto *Inst = cast<Instruction>(*Call->user_begin());
          LLVM_DEBUG(dbgs() << "Remove:"; I.dump());
          LLVM_DEBUG(dbgs() << "Remove:"; Inst->dump(); dbgs() << '\n');
          Inst->eraseFromParent();
          ToErase = &I;
        }
        continue;
      }

      if (auto *LD = dyn_cast<LoadInst>(&I)) {
        if (!LD->hasOneUser())
          continue;
        auto *Call = dyn_cast<CallInst>(*LD->user_begin());
        if (!Call)
          continue;
        auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand());
        if (!GV)
          continue;
        if (!GV->getName().equals("llvm.stackrestore"))
          continue;
        LLVM_DEBUG(dbgs() << "Remove:"; I.dump());
        LLVM_DEBUG(dbgs() << "Remove:"; Call->dump(); dbgs() << '\n');
        Call->eraseFromParent();
        ToErase = &I;
      }
    }
  }

  return false;
}

class BPFIRPeephole final : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID;
  BPFIRPeephole() : FunctionPass(ID) {}
};
} // End anonymous namespace

char BPFIRPeephole::ID = 0;
INITIALIZE_PASS(BPFIRPeephole, DEBUG_TYPE, "BPF IR Peephole", false, false)

FunctionPass *llvm::createBPFIRPeephole() { return new BPFIRPeephole(); }

bool BPFIRPeephole::runOnFunction(Function &F) { return BPFIRPeepholeImpl(F); }

PreservedAnalyses BPFIRPeepholePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  return BPFIRPeepholeImpl(F) ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
}
