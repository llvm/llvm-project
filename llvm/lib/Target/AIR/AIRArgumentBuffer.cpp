//===- AIRArgumentBuffer.cpp - Pack kernel buffers into an arg buffer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRArgumentBuffer.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "air-argument-buffer"

namespace {

static bool isArgBufferParam(Type *T) {
  if (!T->isPointerTy())
    return false;
  unsigned AS = cast<PointerType>(T)->getAddressSpace();
  return AS == 1 || AS == 2;
}

static Function *packKernel(Module &M, Function &F) {
  SmallVector<unsigned, 8> BufIdx;
  for (unsigned I = 0; I < F.arg_size(); ++I)
    if (isArgBufferParam(F.getArg(I)->getType()))
      BufIdx.push_back(I);

  if (BufIdx.empty())
    return &F;

  LLVMContext &Ctx = M.getContext();

  SmallVector<Type *, 8> Members;
  for (unsigned I : BufIdx)
    Members.push_back(F.getArg(I)->getType());

  StructType *ArgStructTy =
      StructType::create(Ctx, Members, ("argbuf." + F.getName()).str());
  auto *ArgBufPtrTy = PointerType::get(Ctx, 2);

  SmallVector<Type *, 8> NewParams;
  NewParams.push_back(ArgBufPtrTy);
  SmallDenseSet<unsigned, 8> BufSet(BufIdx.begin(), BufIdx.end());
  for (unsigned I = 0; I < F.arg_size(); ++I)
    if (!BufSet.count(I))
      NewParams.push_back(F.getArg(I)->getType());

  auto *NewFTy = FunctionType::get(F.getReturnType(), NewParams, F.isVarArg());
  auto *NewF =
      Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", &M);
  NewF->copyAttributesFrom(&F);
  NewF->setCallingConv(F.getCallingConv());
  NewF->takeName(&F);

  NewF->splice(NewF->begin(), &F);

  Argument *ArgBuf = NewF->getArg(0);
  ArgBuf->setName("argbuf");

  {
    unsigned NewI = 1;
    for (unsigned I = 0; I < F.arg_size(); ++I) {
      if (BufSet.count(I))
        continue;
      Argument *Old = F.getArg(I);
      Argument *New = NewF->getArg(NewI++);
      New->takeName(Old);
      Old->replaceAllUsesWith(New);
    }
  }

  BasicBlock &Entry = NewF->getEntryBlock();
  IRBuilder<> B(&Entry, Entry.begin());
  for (auto [Field, OrigArgNo] : enumerate(BufIdx)) {
    Argument *Old = F.getArg(OrigArgNo);
    Value *GEP = B.CreateInBoundsGEP(
        ArgStructTy, ArgBuf,
        {B.getInt32(0), B.getInt32(static_cast<uint32_t>(Field))},
        Old->getName() + ".addr");
    LoadInst *Ld = B.CreateLoad(Old->getType(), GEP, Old->getName());
    Old->replaceAllUsesWith(Ld);
  }

  auto *I32 = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 8> MemberAS;
  for (Type *M : Members)
    MemberAS.push_back(ConstantAsMetadata::get(ConstantInt::get(
        I32, cast<PointerType>(M)->getAddressSpace())));
  NewF->setMetadata("air.argbuf.member_as", MDNode::get(Ctx, MemberAS));

  if (!F.use_empty())
    F.replaceAllUsesWith(UndefValue::get(F.getType()));
  F.eraseFromParent();

  return NewF;
}

static bool runImpl(Module &M) {
  SmallVector<Function *, 2> Kernels;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (!F.hasFnAttribute("air-from-generic-gpu"))
      continue;
    Kernels.push_back(&F);
  }

  bool Changed = false;
  for (Function *F : Kernels)
    Changed |= (packKernel(M, *F) != F);
  return Changed;
}

} // namespace

PreservedAnalyses AIRArgumentBufferPass::run(Module &M,
                                             ModuleAnalysisManager &) {
  return runImpl(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRArgumentBufferLegacy::runOnModule(Module &M) { return runImpl(M); }

char AIRArgumentBufferLegacy::ID = 0;

INITIALIZE_PASS(AIRArgumentBufferLegacy, DEBUG_TYPE,
                "Pack kernel buffers into a Metal argument buffer", false, false)

ModulePass *llvm::createAIRArgumentBufferLegacyPass() {
  return new AIRArgumentBufferLegacy();
}
