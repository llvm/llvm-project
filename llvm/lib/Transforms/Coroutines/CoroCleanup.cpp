//===- CoroCleanup.cpp - Coroutine Cleanup Pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "CoroInternal.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace llvm;

#define DEBUG_TYPE "coro-cleanup"

namespace {
// Created on demand if CoroCleanup pass has work to do.
struct Lowerer : coro::LowererBase {
  IRBuilder<> Builder;
  Constant *NoopCoro = nullptr;

  Lowerer(Module &M) : LowererBase(M), Builder(Context) {}
  bool lower(Function &F);

private:
  void elideCoroNoop(IntrinsicInst *II);
  void lowerCoroNoop(IntrinsicInst *II);
};
}

static void lowerSubFn(IRBuilder<> &Builder, CoroSubFnInst *SubFn) {
  Builder.SetInsertPoint(SubFn);
  Value *FramePtr = SubFn->getFrame();
  int Index = SubFn->getIndex();

  auto *FrameTy = StructType::get(SubFn->getContext(),
                                  {Builder.getPtrTy(), Builder.getPtrTy()});

  Builder.SetInsertPoint(SubFn);
  auto *Gep = Builder.CreateConstInBoundsGEP2_32(FrameTy, FramePtr, 0, Index);
  auto *Load = Builder.CreateLoad(FrameTy->getElementType(Index), Gep);

  SubFn->replaceAllUsesWith(Load);
}

static void buildDebugInfoForNoopResumeDestroyFunc(Function *NoopFn) {
  Module &M = *NoopFn->getParent();
  if (M.debug_compile_units().empty())
    return;

  DICompileUnit *CU = *M.debug_compile_units_begin();
  DIBuilder DB(M, /*AllowUnresolved*/ false, CU);
  std::array<Metadata *, 2> Params{nullptr, nullptr};
  auto *SubroutineType =
      DB.createSubroutineType(DB.getOrCreateTypeArray(Params));
  StringRef Name = NoopFn->getName();
  auto *SP = DB.createFunction(
      CU, /*Name=*/Name, /*LinkageName=*/Name, /*File=*/CU->getFile(),
      /*LineNo=*/0, SubroutineType, /*ScopeLine=*/0, DINode::FlagArtificial,
      DISubprogram::SPFlagDefinition);
  NoopFn->setSubprogram(SP);
  DB.finalize();
}

bool Lowerer::lower(Function &F) {
  bool IsPrivateAndUnprocessed = F.isPresplitCoroutine() && F.hasLocalLinkage();
  bool Changed = false;

  SmallPtrSet<Instruction *, 8> DeadInsts{};
  for (Instruction &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::coro_begin:
      case Intrinsic::coro_begin_custom_abi:
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
      case Intrinsic::coro_noop:
        elideCoroNoop(II);
        if (!II->user_empty())
          lowerCoroNoop(II);
        break;
      case Intrinsic::coro_subfn_addr:
        lowerSubFn(Builder, cast<CoroSubFnInst>(II));
        break;
      case Intrinsic::coro_suspend_retcon:
      case Intrinsic::coro_is_in_ramp:
        if (IsPrivateAndUnprocessed) {
          II->replaceAllUsesWith(PoisonValue::get(II->getType()));
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
      DeadInsts.insert(II);
      Changed = true;
    }
  }

  for (auto *I : DeadInsts)
    I->eraseFromParent();
  return Changed;
}

void Lowerer::elideCoroNoop(IntrinsicInst *II) {
  for (User *U : make_early_inc_range(II->users())) {
    auto *Fn = dyn_cast<CoroSubFnInst>(U);
    if (Fn == nullptr)
      continue;

    auto *User = Fn->getUniqueUndroppableUser();
    if (auto *Call = dyn_cast<CallInst>(User)) {
      Call->eraseFromParent();
      Fn->eraseFromParent();
      continue;
    }

    if (auto *I = dyn_cast<InvokeInst>(User)) {
      Builder.SetInsertPoint(I);
      Builder.CreateBr(I->getNormalDest());
      I->eraseFromParent();
      Fn->eraseFromParent();
    }
  }
}

void Lowerer::lowerCoroNoop(IntrinsicInst *II) {
  if (!NoopCoro) {
    LLVMContext &C = Builder.getContext();
    Module &M = *II->getModule();

    // Create a noop.frame struct type.
    auto *FnTy = FunctionType::get(Type::getVoidTy(C), Builder.getPtrTy(0),
                                   /*isVarArg=*/false);
    auto *FnPtrTy = Builder.getPtrTy(0);
    StructType *FrameTy =
        StructType::create({FnPtrTy, FnPtrTy}, "NoopCoro.Frame");

    // Create a Noop function that does nothing.
    Function *NoopFn = Function::createWithDefaultAttr(
        FnTy, GlobalValue::LinkageTypes::InternalLinkage,
        M.getDataLayout().getProgramAddressSpace(), "__NoopCoro_ResumeDestroy",
        &M);
    NoopFn->setCallingConv(CallingConv::Fast);
    buildDebugInfoForNoopResumeDestroyFunc(NoopFn);
    auto *Entry = BasicBlock::Create(C, "entry", NoopFn);
    ReturnInst::Create(C, Entry);

    // Create a constant struct for the frame.
    Constant *Values[] = {NoopFn, NoopFn};
    Constant *NoopCoroConst = ConstantStruct::get(FrameTy, Values);
    NoopCoro = new GlobalVariable(
        M, NoopCoroConst->getType(), /*isConstant=*/true,
        GlobalVariable::PrivateLinkage, NoopCoroConst, "NoopCoro.Frame.Const");
    cast<GlobalVariable>(NoopCoro)->setNoSanitizeMetadata();
  }

  Builder.SetInsertPoint(II);
  auto *NoopCoroVoidPtr = Builder.CreateBitCast(NoopCoro, Int8Ptr);
  II->replaceAllUsesWith(NoopCoroVoidPtr);
}

static bool declaresCoroCleanupIntrinsics(const Module &M) {
  return coro::declaresIntrinsics(
      M,
      {Intrinsic::coro_alloc, Intrinsic::coro_begin, Intrinsic::coro_subfn_addr,
       Intrinsic::coro_free, Intrinsic::coro_id, Intrinsic::coro_id_retcon,
       Intrinsic::coro_id_async, Intrinsic::coro_id_retcon_once,
       Intrinsic::coro_noop, Intrinsic::coro_async_size_replace,
       Intrinsic::coro_async_resume, Intrinsic::coro_begin_custom_abi});
}

PreservedAnalyses CoroCleanupPass::run(Module &M,
                                       ModuleAnalysisManager &MAM) {
  if (!declaresCoroCleanupIntrinsics(M))
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  FunctionPassManager FPM;
  FPM.addPass(SimplifyCFGPass());

  PreservedAnalyses FuncPA;
  FuncPA.preserveSet<CFGAnalyses>();

  Lowerer L(M);
  for (auto &F : M) {
    if (L.lower(F)) {
      FAM.invalidate(F, FuncPA);
      FPM.run(F, FAM);
    }
  }

  return PreservedAnalyses::none();
}
