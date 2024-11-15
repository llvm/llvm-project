//===- CoroAnnotationElide.cpp - Elide attributed safe coroutine calls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This pass transforms all Call or Invoke instructions that are annotated
// "coro_elide_safe" to call the `.noalloc` variant of coroutine instead.
// The frame of the callee coroutine is allocated inside the caller. A pointer
// to the allocated frame will be passed into the `.noalloc` ramp function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroAnnotationElide.h"

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "coro-annotation-elide"

static Instruction *getFirstNonAllocaInTheEntryBlock(Function *F) {
  for (Instruction &I : F->getEntryBlock())
    if (!isa<AllocaInst>(&I))
      return &I;
  llvm_unreachable("no terminator in the entry block");
}

// Create an alloca in the caller, using FrameSize and FrameAlign as the callee
// coroutine's activation frame.
static Value *allocateFrameInCaller(Function *Caller, uint64_t FrameSize,
                                    Align FrameAlign) {
  LLVMContext &C = Caller->getContext();
  BasicBlock::iterator InsertPt =
      getFirstNonAllocaInTheEntryBlock(Caller)->getIterator();
  const DataLayout &DL = Caller->getDataLayout();
  auto FrameTy = ArrayType::get(Type::getInt8Ty(C), FrameSize);
  auto *Frame = new AllocaInst(FrameTy, DL.getAllocaAddrSpace(), "", InsertPt);
  Frame->setAlignment(FrameAlign);
  return Frame;
}

// Given a call or invoke instruction to the elide safe coroutine, this function
// does the following:
//  - Allocate a frame for the callee coroutine in the caller using alloca.
//  - Replace the old CB with a new Call or Invoke to `NewCallee`, with the
//    pointer to the frame as an additional argument to NewCallee.
static void processCall(CallBase *CB, Function *Caller, Function *NewCallee,
                        uint64_t FrameSize, Align FrameAlign) {
  // TODO: generate the lifetime intrinsics for the new frame. This will require
  // introduction of two pesudo lifetime intrinsics in the frontend around the
  // `co_await` expression and convert them to real lifetime intrinsics here.
  auto *FramePtr = allocateFrameInCaller(Caller, FrameSize, FrameAlign);
  auto NewCBInsertPt = CB->getIterator();
  llvm::CallBase *NewCB = nullptr;
  SmallVector<Value *, 4> NewArgs;
  NewArgs.append(CB->arg_begin(), CB->arg_end());
  NewArgs.push_back(FramePtr);

  if (auto *CI = dyn_cast<CallInst>(CB)) {
    auto *NewCI = CallInst::Create(NewCallee->getFunctionType(), NewCallee,
                                   NewArgs, "", NewCBInsertPt);
    NewCI->setTailCallKind(CI->getTailCallKind());
    NewCB = NewCI;
  } else if (auto *II = dyn_cast<InvokeInst>(CB)) {
    NewCB = InvokeInst::Create(NewCallee->getFunctionType(), NewCallee,
                               II->getNormalDest(), II->getUnwindDest(),
                               NewArgs, {}, "", NewCBInsertPt);
  } else {
    llvm_unreachable("CallBase should either be Call or Invoke!");
  }

  NewCB->setCalledFunction(NewCallee->getFunctionType(), NewCallee);
  NewCB->setCallingConv(CB->getCallingConv());
  NewCB->setAttributes(CB->getAttributes());
  NewCB->setDebugLoc(CB->getDebugLoc());
  std::copy(CB->bundle_op_info_begin(), CB->bundle_op_info_end(),
            NewCB->bundle_op_info_begin());

  NewCB->removeFnAttr(llvm::Attribute::CoroElideSafe);
  CB->replaceAllUsesWith(NewCB);
  CB->eraseFromParent();
}

PreservedAnalyses CoroAnnotationElidePass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  bool Changed = false;

  Function *NewCallee =
      F.getParent()->getFunction((F.getName() + ".noalloc").str());

  if (!NewCallee)
    return PreservedAnalyses::all();

  auto FramePtrArgPosition = NewCallee->arg_size() - 1;
  auto FrameSize = NewCallee->getParamDereferenceableBytes(FramePtrArgPosition);
  auto FrameAlign = NewCallee->getParamAlign(FramePtrArgPosition).valueOrOne();

  SmallVector<CallBase *, 4> Users;
  for (auto *U : F.users()) {
    if (auto *CB = dyn_cast<CallBase>(U)) {
      if (CB->getCalledFunction() == &F)
        Users.push_back(CB);
    }
  }

  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  for (auto *CB : Users) {
    auto *Caller = CB->getFunction();
    if (!Caller)
      continue;

    bool IsCallerPresplitCoroutine = Caller->isPresplitCoroutine();
    bool HasAttr = CB->hasFnAttr(llvm::Attribute::CoroElideSafe);
    if (IsCallerPresplitCoroutine && HasAttr) {
      processCall(CB, Caller, NewCallee, FrameSize, FrameAlign);

      ORE.emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "CoroAnnotationElide", Caller)
               << "'" << ore::NV("callee", F.getName()) << "' elided in '"
               << ore::NV("caller", Caller->getName()) << "'";
      });

      FAM.invalidate(*Caller, PreservedAnalyses::none());
      Changed = true;
    } else {
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "CoroAnnotationElide",
                                        Caller)
               << "'" << ore::NV("callee", F.getName()) << "' not elided in '"
               << ore::NV("caller", Caller->getName()) << "' (caller_presplit="
               << ore::NV("caller_presplit", IsCallerPresplitCoroutine)
               << ", elide_safe_attr=" << ore::NV("elide_safe_attr", HasAttr)
               << ")";
      });
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
