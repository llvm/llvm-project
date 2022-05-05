//===- CoroCleanup.cpp - Coroutine Cleanup Pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "CoroInternal.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "coro-cleanup"

namespace {
// Created on demand if CoroCleanup pass has work to do.
struct Lowerer : coro::LowererBase {
  IRBuilder<> Builder;
  Lowerer(Module &M) : LowererBase(M), Builder(Context) {}
  void lower(Function &F);
};
}

static void simplifyCFG(Function &F) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createCFGSimplificationPass());

  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();
}

static void lowerSubFn(IRBuilder<> &Builder, CoroSubFnInst *SubFn) {
  Builder.SetInsertPoint(SubFn);
  Value *FrameRaw = SubFn->getFrame();
  int Index = SubFn->getIndex();

  auto *FrameTy = StructType::get(
      SubFn->getContext(), {Builder.getInt8PtrTy(), Builder.getInt8PtrTy()});
  PointerType *FramePtrTy = FrameTy->getPointerTo();

  Builder.SetInsertPoint(SubFn);
  auto *FramePtr = Builder.CreateBitCast(FrameRaw, FramePtrTy);
  auto *Gep = Builder.CreateConstInBoundsGEP2_32(FrameTy, FramePtr, 0, Index);
  auto *Load = Builder.CreateLoad(FrameTy->getElementType(Index), Gep);

  SubFn->replaceAllUsesWith(Load);
}

void Lowerer::lower(Function &F) {
  bool IsPrivateAndUnprocessed =
      F.hasFnAttribute(CORO_PRESPLIT_ATTR) && F.hasLocalLinkage();

  for (Instruction &I : llvm::make_early_inc_range(instructions(F))) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::coro_begin:
        II->replaceAllUsesWith(II->getArgOperand(1));
        break;
      case Intrinsic::coro_free:
        II->replaceAllUsesWith(II->getArgOperand(1));
        break;
      case Intrinsic::coro_alloc:
        II->replaceAllUsesWith(ConstantInt::getTrue(Context));
        break;
      case Intrinsic::coro_async_resume:
        II->replaceAllUsesWith(
            ConstantPointerNull::get(cast<PointerType>(I.getType())));
        break;
      case Intrinsic::coro_id:
      case Intrinsic::coro_id_retcon:
      case Intrinsic::coro_id_retcon_once:
      case Intrinsic::coro_id_async:
        II->replaceAllUsesWith(ConstantTokenNone::get(Context));
        break;
      case Intrinsic::coro_subfn_addr:
        lowerSubFn(Builder, cast<CoroSubFnInst>(II));
        break;
      case Intrinsic::coro_end:
      case Intrinsic::coro_suspend_retcon:
        if (IsPrivateAndUnprocessed) {
          II->replaceAllUsesWith(UndefValue::get(II->getType()));
        } else
          continue;
        break;
      case Intrinsic::coro_async_size_replace:
        auto *Target = cast<ConstantStruct>(
            cast<GlobalVariable>(II->getArgOperand(0)->stripPointerCasts())
                ->getInitializer());
        auto *Source = cast<ConstantStruct>(
            cast<GlobalVariable>(II->getArgOperand(1)->stripPointerCasts())
                ->getInitializer());
        auto *TargetSize = Target->getOperand(1);
        auto *SourceSize = Source->getOperand(1);
        if (TargetSize->isElementWiseEqual(SourceSize)) {
          break;
        }
        auto *TargetRelativeFunOffset = Target->getOperand(0);
        auto *NewFuncPtrStruct = ConstantStruct::get(
            Target->getType(), TargetRelativeFunOffset, SourceSize);
        Target->replaceAllUsesWith(NewFuncPtrStruct);
        break;
      }
      II->eraseFromParent();
    }
  }

  // After replacement were made we can cleanup the function body a little.
  simplifyCFG(F);
}

static bool declaresCoroCleanupIntrinsics(const Module &M) {
  return coro::declaresIntrinsics(
      M, {"llvm.coro.alloc", "llvm.coro.begin", "llvm.coro.subfn.addr",
          "llvm.coro.free", "llvm.coro.id", "llvm.coro.id.retcon",
          "llvm.coro.id.retcon.once", "llvm.coro.async.size.replace",
          "llvm.coro.async.resume"});
}

PreservedAnalyses CoroCleanupPass::run(Module &M,
                                       ModuleAnalysisManager &MAM) {
  if (!declaresCoroCleanupIntrinsics(M))
    return PreservedAnalyses::all();

  Lowerer L(M);
  for (auto &F : M)
    L.lower(F);

  return PreservedAnalyses::none();
}
