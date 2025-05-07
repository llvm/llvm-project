//===--------- IRSanitizer.cpp - IRSanitizer instrumentation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IRSanitizer class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/IRSanitizer.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace llvm {

IRSanitizerPass::IRSanitizerPass() {}

PreservedAnalyses IRSanitizerPass::sanitizeFunction(Function &F, FunctionAnalysisManager &FAM) {
  Function *TrapFunc = Intrinsic::getOrInsertDeclaration(M, llvm::Intrinsic::trap);

  std::vector<LoadInst *> LoadsToInstr;
  for (auto &Inst : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
      LoadsToInstr.push_back(LI);
    }
  }

  for (LoadInst *LI : LoadsToInstr) {
    IRBuilder<> IRB(LI);
    Align align = LI->getAlign();
    Value *Ptr = LI->getOperand(0);

    Value *PtrAsInt = IRB.CreatePtrToInt(Ptr, PtrIntTy);
    Value *BottomBits = IRB.CreateAnd(PtrAsInt, align.value() - 1);
    Value *IsUnaligned = IRB.CreateICmpNE(BottomBits, Constant::getNullValue(PtrIntTy));

    Instruction *Then = SplitBlockAndInsertIfThen(IsUnaligned, LI, true);

    IRBuilder<> FailIRB(Then);
    FailIRB.CreateCall(TrapFunc);
  }
  
  return PreservedAnalyses::all();
}

PreservedAnalyses IRSanitizerPass::run(Module &M, ModuleAnalysisManager &MAM) {
  this->M = &M;
  this->PtrIntTy = IntegerType::get(M.getContext(), M.getDataLayout().getPointerSizeInBits());

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  PreservedAnalyses PA = PreservedAnalyses::all();
  for (Function &F : M) {
    PA.intersect(sanitizeFunction(F, FAM));
  }

  return PA;
}

}  // namespace llvm
