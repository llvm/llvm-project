//===- LowerVectorIntrinsics.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerVectorIntrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "lower-vector-intrinsics"

using namespace llvm;

bool llvm::lowerUnaryVectorIntrinsicAsLoop(Module &M, CallInst *CI) {
  VectorType *VecTy = cast<VectorType>(CI->getArgOperand(0)->getType());
  Type *IdxTy = M.getDataLayout().getIndexType(CI->getContext(), 0);

  IRBuilder<> Builder(CI);
  Value *LoopEnd = Builder.CreateElementCount(IdxTy, VecTy->getElementCount());

  auto [BodyIP, IV] =
      SplitBlockAndInsertSimpleForLoop(LoopEnd, CI->getIterator());

  BasicBlock *LoopBB = BodyIP->getParent();
  auto *IVPhi = cast<PHINode>(IV);
  BasicBlock *Preheader =
      IVPhi->getIncomingBlock(IVPhi->getIncomingBlock(0) == LoopBB);

  PHINode *Vec = PHINode::Create(VecTy, 2, "", BodyIP->getIterator());
  Vec->addIncoming(CI->getArgOperand(0), Preheader);

  Builder.SetInsertPoint(BodyIP);
  Value *Elem = Builder.CreateExtractElement(Vec, IV);
  Function *Exp = Intrinsic::getOrInsertDeclaration(&M, CI->getIntrinsicID(),
                                                    VecTy->getElementType());
  Value *Res = Builder.CreateCall(Exp, Elem);
  Value *NewVec = Builder.CreateInsertElement(Vec, Res, IV);
  Vec->addIncoming(NewVec, LoopBB);

  CI->replaceAllUsesWith(NewVec);
  CI->eraseFromParent();
  return true;
}
