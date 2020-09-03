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
  bool lowerRemainingCoroIntrinsics(Function &F);
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

bool Lowerer::lowerRemainingCoroIntrinsics(Function &F) {
  bool Changed = false;

  for (auto IB = inst_begin(F), E = inst_end(F); IB != E;) {
    Instruction &I = *IB++;
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
      case Intrinsic::coro_id:
      case Intrinsic::coro_id_retcon:
      case Intrinsic::coro_id_retcon_once:
        II->replaceAllUsesWith(ConstantTokenNone::get(Context));
        break;
      case Intrinsic::coro_subfn_addr:
        lowerSubFn(Builder, cast<CoroSubFnInst>(II));
        break;
      }
      II->eraseFromParent();
      Changed = true;
    }
  }

  if (Changed) {
    // After replacement were made we can cleanup the function body a little.
    simplifyCFG(F);
  }

  return Changed;
}

static bool declaresCoroCleanupIntrinsics(const Module &M) {
  return coro::declaresIntrinsics(M, {"llvm.coro.alloc", "llvm.coro.begin",
                                      "llvm.coro.subfn.addr", "llvm.coro.free",
                                      "llvm.coro.id", "llvm.coro.id.retcon",
                                      "llvm.coro.id.retcon.once"});
}

PreservedAnalyses CoroCleanupPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &M = *F.getParent();
  if (!declaresCoroCleanupIntrinsics(M) ||
      !Lowerer(M).lowerRemainingCoroIntrinsics(F))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

namespace {

struct CoroCleanupLegacy : FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  CoroCleanupLegacy() : FunctionPass(ID) {
    initializeCoroCleanupLegacyPass(*PassRegistry::getPassRegistry());
  }

  std::unique_ptr<Lowerer> L;

  // This pass has work to do only if we find intrinsics we are going to lower
  // in the module.
  bool doInitialization(Module &M) override {
    if (declaresCoroCleanupIntrinsics(M))
      L = std::make_unique<Lowerer>(M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (L)
      return L->lowerRemainingCoroIntrinsics(F);
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (!L)
      AU.setPreservesAll();
  }
  StringRef getPassName() const override { return "Coroutine Cleanup"; }
};
}

char CoroCleanupLegacy::ID = 0;
INITIALIZE_PASS(CoroCleanupLegacy, "coro-cleanup",
                "Lower all coroutine related intrinsics", false, false)

Pass *llvm::createCoroCleanupLegacyPass() { return new CoroCleanupLegacy(); }
