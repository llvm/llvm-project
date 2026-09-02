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
class NVPTXImageOptimizer : public FunctionPass {
private:
  static char ID;
  SmallVector<Instruction*, 4> InstrToDelete;

public:
  NVPTXImageOptimizer();

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "NVPTX Image Optimizer"; }

private:
  bool replaceIsTypeP(Instruction &I, PTXOpaqueType Expected);
  Value *cleanupValue(Value *V);
  void replaceWith(Instruction *From, ConstantInt *To);
};
}

char NVPTXImageOptimizer::ID = 0;

NVPTXImageOptimizer::NVPTXImageOptimizer()
  : FunctionPass(ID) {}

bool NVPTXImageOptimizer::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

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

FunctionPass *llvm::createNVPTXImageOptimizerPass() {
  return new NVPTXImageOptimizer();
}
