//===- AIRAsyncCopyToCooperative.cpp - lower async copy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRAsyncCopyToCooperative.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-async-copy-to-cooperative"

static constexpr StringLiteral kAsyncCopyPrefix("air.simdgroup_async_copy_2d.");
static constexpr StringLiteral kWaitEvents("air.wait_simdgroup_events");
static constexpr StringLiteral kMMAPrefix("air.simdgroup_matrix_8x8_");
static constexpr StringLiteral kTidTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral kBarrier("air.wg.barrier");

static bool moduleUsesMMA(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kMMAPrefix))
      return true;
  return false;
}

static bool moduleHasAsyncCopy(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kAsyncCopyPrefix))
      return true;
  return false;
}

static GlobalVariable *threadgroupGlobalOf(Value *V) {
  SmallVector<Value *, 8> Work{V};
  SmallPtrSet<Value *, 8> Seen;
  while (!Work.empty()) {
    Value *Cur = Work.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    Cur = Cur->stripPointerCasts();
    if (auto *GV = dyn_cast<GlobalVariable>(Cur)) {
      if (GV->getAddressSpace() == metal::AS::Threadgroup)
        return GV;
      continue;
    }
    if (auto *GEP = dyn_cast<GEPOperator>(Cur))
      Work.push_back(GEP->getPointerOperand());
    else if (auto *Sel = dyn_cast<SelectInst>(Cur)) {
      Work.push_back(Sel->getTrueValue());
      Work.push_back(Sel->getFalseValue());
    } else if (auto *PN = dyn_cast<PHINode>(Cur)) {
      for (Value *In : PN->incoming_values())
        Work.push_back(In);
    }
  }
  return nullptr;
}

static bool mmaReadsAsyncArena(Module &M) {
  SmallPtrSet<GlobalVariable *, 4> AsyncTargets;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        if (!CI->getCalledFunction()->getName().starts_with(kAsyncCopyPrefix))
          continue;
        if (CI->arg_size() > 2)
          if (auto *GV = threadgroupGlobalOf(CI->getArgOperand(2)))
            AsyncTargets.insert(GV);
      }
  }
  if (AsyncTargets.empty())
    return false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.starts_with(kMMAPrefix) || !Name.contains("_load"))
          if (!Name.starts_with(kMMAPrefix) || !Name.contains("_store"))
            continue;
        unsigned PtrArg = Name.contains("_store") ? 1u : 0u;
        if (PtrArg >= CI->arg_size())
          continue;
        if (auto *GV = threadgroupGlobalOf(CI->getArgOperand(PtrArg)))
          if (AsyncTargets.count(GV))
            return true;
      }
  }
  return false;
}

static Function *getTidFn(Module &M) {
  if (auto *F = M.getFunction(kTidTG))
    return F;
  auto &Ctx = M.getContext();
  auto *RetTy = ArrayType::get(Type::getInt32Ty(Ctx), 3);
  auto *FT = FunctionType::get(RetTy, {}, false);
  return Function::Create(FT, Function::ExternalLinkage, kTidTG, &M);
}

static Function *getBarrierFn(Module &M) {
  if (auto *F = M.getFunction(kBarrier))
    return F;
  auto &Ctx = M.getContext();
  auto *I32 = Type::getInt32Ty(Ctx);
  auto *FT = FunctionType::get(Type::getVoidTy(Ctx), {I32, I32}, false);
  return Function::Create(FT, Function::ExternalLinkage, kBarrier, &M);
}

static Value *getFlatTid(IRBuilder<> &B, Module &M) {
  auto *TidFn = getTidFn(M);
  Value *Tid = B.CreateCall(TidFn, {}, "tid");
  return B.CreateExtractValue(Tid, {0}, "tid.x");
}

static int64_t constVecElem(Value *V, unsigned Idx) {
  if (auto *CV = dyn_cast<Constant>(V)) {
    if (auto *CE = CV->getAggregateElement(Idx))
      if (auto *CI = dyn_cast<ConstantInt>(CE))
        return (int64_t)CI->getZExtValue();
  }
  return -1;
}

static bool lowerAsyncCopy(CallInst *CI, Module &M, unsigned TGSize) {
  auto &Ctx = M.getContext();
  IRBuilder<> B(CI);
  auto *I64 = Type::getInt64Ty(Ctx);
  auto *F32 = Type::getFloatTy(Ctx);

  Value *Dst = CI->getArgOperand(2);
  Value *DstStrideBytes = CI->getArgOperand(3);
  Value *DstTile = CI->getArgOperand(5);
  Value *Src = CI->getArgOperand(6);
  Value *SrcStrideBytes = CI->getArgOperand(7);

  int64_t WidthBytes = constVecElem(DstTile, 0);
  int64_t Rows = constVecElem(DstTile, 1);
  if (WidthBytes < 0 || Rows < 0 || (WidthBytes % 4) != 0)
    return false;
  int64_t WidthF = WidthBytes / 4;
  int64_t Total = Rows * WidthF;
  unsigned Step = TGSize ? TGSize : 32;

  Value *Flat = getFlatTid(B, M);
  Value *FlatI64 = B.CreateZExt(Flat, I64, "tid64");
  Value *Four = ConstantInt::get(I64, 4);
  Value *WF = ConstantInt::get(I64, WidthF);
  Value *TotalC = ConstantInt::get(I64, Total);

  Value *SrcPitchF = B.CreateUDiv(SrcStrideBytes, Four, "ci.srcpitchf");
  Value *DstPitchF = B.CreateUDiv(DstStrideBytes, Four, "ci.dstpitchf");
  int64_t KMax = (Total + (int64_t)Step - 1) / (int64_t)Step;
  for (int64_t k = 0; k < KMax; ++k) {
    Value *I =
        B.CreateAdd(FlatI64, ConstantInt::get(I64, k * (int64_t)Step), "ci.i");
    Value *Valid = B.CreateICmpULT(I, TotalC, "ci.valid");
    Value *II = B.CreateSelect(Valid, I, ConstantInt::get(I64, 0), "ci.ii");
    Value *R = B.CreateUDiv(II, WF, "ci.r");
    Value *C = B.CreateURem(II, WF, "ci.c");
    Value *DstIdx = B.CreateAdd(B.CreateMul(R, DstPitchF), C, "ci.dstidx");
    Value *SrcIdx = B.CreateAdd(B.CreateMul(R, SrcPitchF), C, "ci.srcidx");
    Value *DstP = B.CreateGEP(F32, Dst, DstIdx, "ci.dstp");
    Value *SrcP = B.CreateGEP(F32, Src, SrcIdx, "ci.srcp");
    Value *V = B.CreateAlignedLoad(F32, SrcP, Align(4), "ci.v");
    B.CreateAlignedStore(V, DstP, Align(4));
  }

  CI->replaceAllUsesWith(Dst);
  CI->eraseFromParent();
  return true;
}

static bool asyncCopyToCooperative(Module &M, unsigned TGSize) {
  return false;
  if (!moduleUsesMMA(M) || !moduleHasAsyncCopy(M))
    return false;
  if (!mmaReadsAsyncArena(M))
    return false;

  bool Changed = false;

  SmallVector<CallInst *, 8> Waits;
  SmallVector<CallInst *, 8> Copies;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (Name == kWaitEvents)
          Waits.push_back(CI);
        else if (Name.starts_with(kAsyncCopyPrefix))
          Copies.push_back(CI);
      }
  }

  auto *I32 = Type::getInt32Ty(M.getContext());
  for (CallInst *W : Waits) {
    IRBuilder<> B(W);
    Function *Barr = getBarrierFn(M);
    B.CreateCall(Barr, {ConstantInt::get(I32, 2), ConstantInt::get(I32, 1)});
    W->eraseFromParent();
    Changed = true;
  }

  bool AllLowered = true;
  for (CallInst *C : Copies) {
    if (lowerAsyncCopy(C, M, TGSize))
      Changed = true;
    else
      AllLowered = false;
  }

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<AllocaInst *, 2> DeadAllocas;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          if (!AI->getAllocatedType()->isPointerTy())
            continue;
          if (AI->getAllocatedType()->getPointerAddressSpace() != 3)
            continue;
          bool OnlyStores = true;
          for (User *U : AI->users())
            if (!isa<StoreInst>(U)) {
              OnlyStores = false;
              break;
            }
          if (OnlyStores)
            DeadAllocas.push_back(AI);
        }
    for (AllocaInst *AI : DeadAllocas) {
      SmallVector<Instruction *, 4> Stores;
      for (User *U : AI->users())
        Stores.push_back(cast<Instruction>(U));
      for (Instruction *S : Stores)
        S->eraseFromParent();
      AI->eraseFromParent();
      Changed = true;
    }
  }

  if (AllLowered) {
    SmallVector<Function *, 4> Dead;
    for (Function &F : M)
      if ((F.getName().starts_with(kAsyncCopyPrefix) ||
           F.getName() == kWaitEvents) &&
          F.use_empty())
        Dead.push_back(&F);
    for (Function *F : Dead) {
      F->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

static unsigned getThreadgroupSize(Module &M) {
  (void)M;
  return 32;
}

PreservedAnalyses
AIRAsyncCopyToCooperativePass::run(Module &M, ModuleAnalysisManager &AM) {
  return asyncCopyToCooperative(M, getThreadgroupSize(M))
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

bool AIRAsyncCopyToCooperativeLegacy::runOnModule(Module &M) {
  return asyncCopyToCooperative(M, getThreadgroupSize(M));
}

char AIRAsyncCopyToCooperativeLegacy::ID = 0;

INITIALIZE_PASS(AIRAsyncCopyToCooperativeLegacy, DEBUG_TYPE,
                "AIR Async Copy To Cooperative", false, false)

ModulePass *llvm::createAIRAsyncCopyToCooperativeLegacyPass() {
  return new AIRAsyncCopyToCooperativeLegacy();
}
