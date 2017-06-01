//===-- PreISelIntrinsicLowering.cpp - Pre-ISel intrinsic lowering pass ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR lowering for the llvm.load.relative intrinsic.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

bool lowerLoadRelative(Function &F) {
  if (F.use_empty())
    return false;

  bool Changed = false;
  Type *Int32Ty = Type::getInt32Ty(F.getContext());
  Type *Int32PtrTy = Int32Ty->getPointerTo();
  Type *Int8Ty = Type::getInt8Ty(F.getContext());

  for (auto I = F.use_begin(), E = F.use_end(); I != E;) {
    auto CI = dyn_cast<CallInst>(I->getUser());
    ++I;
    if (!CI || CI->getCalledValue() != &F)
      continue;

    IRBuilder<> B(CI);
    Value *OffsetPtr =
        B.CreateGEP(Int8Ty, CI->getArgOperand(0), CI->getArgOperand(1));
    Value *OffsetPtrI32 = B.CreateBitCast(OffsetPtr, Int32PtrTy);
    Value *OffsetI32 = B.CreateAlignedLoad(OffsetPtrI32, 4);

    Value *ResultPtr = B.CreateGEP(Int8Ty, CI->getArgOperand(0), OffsetI32);

    CI->replaceAllUsesWith(ResultPtr);
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool lowerIntrinsics(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.getName().startswith("llvm.load.relative."))
      Changed |= lowerLoadRelative(F);
  }
  return Changed;
}

class PreISelIntrinsicLoweringLegacyPass : public ModulePass {
public:
  static char ID;
  PreISelIntrinsicLoweringLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) { return lowerIntrinsics(M); }
};

char PreISelIntrinsicLoweringLegacyPass::ID;
}

INITIALIZE_PASS(PreISelIntrinsicLoweringLegacyPass,
                "pre-isel-intrinsic-lowering", "Pre-ISel Intrinsic Lowering",
                false, false)

namespace llvm {
ModulePass *createPreISelIntrinsicLoweringPass() {
  return new PreISelIntrinsicLoweringLegacyPass;
}

PreservedAnalyses PreISelIntrinsicLoweringPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (!lowerIntrinsics(M))
    return PreservedAnalyses::all();
  else
    return PreservedAnalyses::none();
}
} // End llvm namespace
