//===-- NVPTXImageOptimizer.cpp - Image optimization pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR-level optimizations of image access code,
// including:
//
// 1. Eliminate istypep intrinsics when image access qualifier is known
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVVMProperties.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class NVPTXImageOptimizer {
  SmallVector<Instruction *, 4> InstrToDelete;

public:
  bool run(Function &F);

private:
  bool replaceIsTypeP(Instruction &I, PTXOpaqueType Expected);
  Value *cleanupValue(Value *V);
  void replaceWith(Instruction *From, ConstantInt *To);
};
} // namespace

bool NVPTXImageOptimizer::run(Function &F) {
  bool Changed = false;
  InstrToDelete.clear();

  // Look for call instructions in the function
  for (BasicBlock &BB : F) {
    for (Instruction &Instr : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&Instr)) {
        Function *CalledF = CI->getCalledFunction();
        if (CalledF && CalledF->isIntrinsic()) {
          // This is an intrinsic function call, check if its an istypep
          switch (CalledF->getIntrinsicID()) {
          default: break;
          case Intrinsic::nvvm_istypep_sampler:
            Changed |= replaceIsTypeP(Instr, PTXOpaqueType::Sampler);
            break;
          case Intrinsic::nvvm_istypep_surface:
            Changed |= replaceIsTypeP(Instr, PTXOpaqueType::Surface);
            break;
          case Intrinsic::nvvm_istypep_texture:
            Changed |= replaceIsTypeP(Instr, PTXOpaqueType::Texture);
            break;
          }
        }
      }
    }
  }

  // Delete any istypep instances we replaced in the IR
  for (Instruction *I : InstrToDelete)
    I->eraseFromParent();

  return Changed;
}

bool NVPTXImageOptimizer::replaceIsTypeP(Instruction &I,
                                         PTXOpaqueType Expected) {
  PTXOpaqueType OT = getPTXOpaqueType(*cleanupValue(I.getOperand(0)));
  if (OT == PTXOpaqueType::None)
    return false;
  replaceWith(&I, ConstantInt::getBool(I.getContext(), OT == Expected));
  return true;
}

void NVPTXImageOptimizer::replaceWith(Instruction *From, ConstantInt *To) {
  // We implement "poor man's DCE" here to make sure any code that is no longer
  // live is actually unreachable and can be trivially eliminated by the
  // unreachable block elimination pass.
  for (Use &U : From->uses()) {
    if (CondBrInst *BI = dyn_cast<CondBrInst>(U)) {
      BasicBlock *Dest = BI->getSuccessor(To->isZero() ? 1 : 0);
      UncondBrInst::Create(Dest, BI->getIterator());
      InstrToDelete.push_back(BI);
    }
  }
  From->replaceAllUsesWith(To);
  InstrToDelete.push_back(From);
}

Value *NVPTXImageOptimizer::cleanupValue(Value *V) {
  if (ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(V)) {
    return cleanupValue(EVI->getAggregateOperand());
  }
  return V;
}

namespace {
class NVPTXImageOptimizerLegacyPass : public FunctionPass {
public:
  static char ID;
  NVPTXImageOptimizerLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    return NVPTXImageOptimizer().run(F);
  }

  StringRef getPassName() const override { return "NVPTX Image Optimizer"; }
};
} // namespace

char NVPTXImageOptimizerLegacyPass::ID = 0;

FunctionPass *llvm::createNVPTXImageOptimizerLegacyPass() {
  return new NVPTXImageOptimizerLegacyPass();
}

PreservedAnalyses NVPTXImageOptimizerPass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  // The transform replaces conditional branches with unconditional ones, so
  // the CFG is not preserved.
  return NVPTXImageOptimizer().run(F) ? PreservedAnalyses::none()
                                      : PreservedAnalyses::all();
}
