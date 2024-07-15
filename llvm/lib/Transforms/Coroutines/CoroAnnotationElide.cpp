//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#define CORO_MUST_ELIDE_ANNOTATION "coro_must_elide"

static Instruction *getFirstNonAllocaInTheEntryBlock(Function *F) {
  for (Instruction &I : F->getEntryBlock())
    if (!isa<AllocaInst>(&I))
      return &I;
  llvm_unreachable("no terminator in the entry block");
}

static Value *allocateFrameInCaller(Function *Caller, uint64_t FrameSize,
                                    Align FrameAlign) {
  LLVMContext &C = Caller->getContext();
  BasicBlock::iterator InsertPt =
      getFirstNonAllocaInTheEntryBlock(Caller)->getIterator();
  const DataLayout &DL = Caller->getDataLayout();
  auto FrameTy = ArrayType::get(Type::getInt8Ty(C), FrameSize);
  auto *Frame = new AllocaInst(FrameTy, DL.getAllocaAddrSpace(), "", InsertPt);
  Frame->setAlignment(FrameAlign);
  return new BitCastInst(Frame, PointerType::getUnqual(C), "vFrame", InsertPt);
}

static void processCall(CallBase *CB, Function *Caller, Function *NewCallee,
                        uint64_t FrameSize, Align FrameAlign) {
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
                               NewArgs, std::nullopt, "", NewCBInsertPt);
  } else {
    llvm_unreachable("CallBase should either be Call or Invoke!");
  }

  NewCB->setCalledFunction(NewCallee->getFunctionType(), NewCallee);
  NewCB->setCallingConv(CB->getCallingConv());
  NewCB->setAttributes(CB->getAttributes());
  NewCB->setDebugLoc(CB->getDebugLoc());
  std::copy(CB->bundle_op_info_begin(), CB->bundle_op_info_end(),
            NewCB->bundle_op_info_begin());

  NewCB->removeFnAttr(llvm::Attribute::CoroMustElide);
  CB->replaceAllUsesWith(NewCB);
  CB->eraseFromParent();
}

PreservedAnalyses CoroAnnotationElidePass::run(LazyCallGraph::SCC &C,
                                               CGSCCAnalysisManager &AM,
                                               LazyCallGraph &CG,
                                               CGSCCUpdateResult &UR) {
  bool Changed = false;
  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);

  auto &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  for (LazyCallGraph::Node &N : C) {
    Function *Callee = &N.getFunction();
    Function *NewCallee = Callee->getParent()->getFunction(
        (Callee->getName() + ".noalloc").str());
    if (!NewCallee) {
      continue;
    }

    auto FramePtrArgPosition = NewCallee->arg_size() - 1;
    auto FrameSize =
        NewCallee->getParamDereferenceableBytes(FramePtrArgPosition);
    auto FrameAlign =
        NewCallee->getParamAlign(FramePtrArgPosition).valueOrOne();

    SmallVector<CallBase *, 4> Users;
    for (auto *U : Callee->users()) {
      if (auto *CB = dyn_cast<CallBase>(U)) {
        Users.push_back(CB);
      }
    }

    auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(*Callee);

    for (auto *CB : Users) {
      auto *Caller = CB->getFunction();
      if (Caller && Caller->isPresplitCoroutine() &&
          CB->hasFnAttr(llvm::Attribute::CoroMustElide)) {
        processCall(CB, Caller, NewCallee, FrameSize, FrameAlign);
        CGUpdater.reanalyzeFunction(*Caller);

        ORE.emit([&]() {
          return OptimizationRemark(DEBUG_TYPE, "CoroAnnotationElide", Caller)
                 << "'" << ore::NV("callee", Callee->getName())
                 << "' elided in '" << ore::NV("caller", Caller->getName());
        });
        Changed = true;
      }
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
