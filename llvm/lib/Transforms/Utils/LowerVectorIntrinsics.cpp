//===- LowerVectorIntrinsics.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerVectorIntrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "lower-vector-intrinsics"

using namespace llvm;

bool llvm::lowerUnaryVectorIntrinsicAsLoop(Module &M, CallInst *CI) {
  Type *RetTy = CI->getType();
  auto *StructRetTy = dyn_cast<StructType>(RetTy);
  unsigned NumResults = StructRetTy ? StructRetTy->getNumElements() : 1;
  auto *VecTy = cast<VectorType>(StructRetTy ? StructRetTy->getElementType(0)
                                             : RetTy);

  BasicBlock *PreLoopBB = CI->getParent();
  BasicBlock *PostLoopBB = nullptr;
  Function *ParentFunc = PreLoopBB->getParent();
  LLVMContext &Ctx = PreLoopBB->getContext();
  Type *IdxTy = M.getDataLayout().getIndexType(Ctx, 0);

  PostLoopBB = PreLoopBB->splitBasicBlock(CI);
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "", ParentFunc, PostLoopBB);
  PreLoopBB->getTerminator()->setSuccessor(0, LoopBB);

  // Loop preheader
  IRBuilder<> PreLoopBuilder(PreLoopBB->getTerminator());
  Value *LoopEnd =
      PreLoopBuilder.CreateElementCount(IdxTy, VecTy->getElementCount());

  // Loop body
  IRBuilder<> LoopBuilder(LoopBB);

  PHINode *LoopIndex = LoopBuilder.CreatePHI(IdxTy, 2);
  LoopIndex->addIncoming(ConstantInt::get(IdxTy, 0U), PreLoopBB);

  SmallVector<PHINode *, 2> ResultPhis(NumResults);
  for (unsigned I = 0; I != NumResults; ++I) {
    ResultPhis[I] = LoopBuilder.CreatePHI(VecTy, 2);
    ResultPhis[I]->addIncoming(PoisonValue::get(VecTy), PreLoopBB);
  }

  Value *Elem =
      LoopBuilder.CreateExtractElement(CI->getArgOperand(0), LoopIndex);
  Function *Fn = Intrinsic::getOrInsertDeclaration(&M, CI->getIntrinsicID(),
                                                   VecTy->getElementType());

  CallInst *ScalarCall = LoopBuilder.CreateCall(Fn, Elem);
  if (isa<FPMathOperator>(CI))
    ScalarCall->copyFastMathFlags(CI);

  SmallVector<Value *, 2> NewVecs(NumResults);
  for (unsigned I = 0; I != NumResults; ++I) {
    Value *ScalarRes = ScalarCall;
    if (StructRetTy)
      ScalarRes = LoopBuilder.CreateExtractValue(ScalarCall, I);
    NewVecs[I] =
        LoopBuilder.CreateInsertElement(ResultPhis[I], ScalarRes, LoopIndex);
    ResultPhis[I]->addIncoming(NewVecs[I], LoopBB);
  }

  Value *One = ConstantInt::get(IdxTy, 1U);
  Value *NextLoopIndex = LoopBuilder.CreateAdd(LoopIndex, One);
  LoopIndex->addIncoming(NextLoopIndex, LoopBB);

  Value *ExitCond =
      LoopBuilder.CreateICmp(CmpInst::ICMP_EQ, NextLoopIndex, LoopEnd);
  LoopBuilder.CreateCondBr(ExitCond, PostLoopBB, LoopBB);

  Value *Res = NewVecs[0];
  if (StructRetTy) {
    IRBuilder<> PostLoopBuilder(CI);
    Res = PoisonValue::get(RetTy);
    for (unsigned I = 0; I != NumResults; ++I)
      Res = PostLoopBuilder.CreateInsertValue(Res, NewVecs[I], I);
  }

  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return true;
}
