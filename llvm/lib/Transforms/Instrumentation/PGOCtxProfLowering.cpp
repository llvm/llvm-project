//===- PGOCtxProfLowering.cpp - Contextual  PGO Instrumentation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "llvm/Transforms/Instrumentation/PGOCtxProfLowering.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

using namespace llvm;

static cl::list<std::string> ContextRoots("profile-context-root");

bool PGOCtxProfLoweringPass::isContextualIRPGOEnabled() {
  return !ContextRoots.empty();
}

PreservedAnalyses PGOCtxProfLoweringPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  ContextRootTy = nullptr;
  auto *PointerTy = PointerType::get(M.getContext(), 0);
  auto *SanitizerMutexType = Type::getInt8Ty(M.getContext());
  auto *I32Ty = Type::getInt32Ty(M.getContext());
  auto *I64Ty = Type::getInt64Ty(M.getContext());

  ContextRootTy =
      StructType::get(M.getContext(), {
                                          PointerTy,          /*FirstNode*/
                                          PointerTy,          /*FirstMemBlock*/
                                          PointerTy,          /*CurrentMem*/
                                          SanitizerMutexType, /*Taken*/
                                      });
  ContextNodeTy = StructType::get(M.getContext(), {
                                                      I64Ty,     /*Guid*/
                                                      PointerTy, /*Next*/
                                                      I32Ty,     /*NrCounters*/
                                                      I32Ty,     /*NrCallsites*/
                                                  });

  for (const auto &Fname : ContextRoots) {
    if (const auto *F = M.getFunction(Fname)) {
      if (F->isDeclaration())
        continue;
      auto *G = M.getOrInsertGlobal(Fname + "_ctx_root", ContextRootTy);
      cast<GlobalVariable>(G)->setInitializer(
          Constant::getNullValue(ContextRootTy));
      ContextRootMap.insert(std::make_pair(F, G));
    }
  }

  StartCtx = cast<Function>(
      M.getOrInsertFunction(
           "__llvm_instrprof_start_context",
           FunctionType::get(ContextNodeTy->getPointerTo(),
                             {ContextRootTy->getPointerTo(), /*ContextRoot*/
                              I64Ty, /*Guid*/ I32Ty,
                              /*NrCounters*/ I32Ty /*NrCallsites*/},
                             false))
          .getCallee());
  GetCtx = cast<Function>(
      M.getOrInsertFunction("__llvm_instrprof_get_context",
                            FunctionType::get(ContextNodeTy->getPointerTo(),
                                              {PointerTy, /*Callee*/
                                               I64Ty,     /*Guid*/
                                               I32Ty,     /*NrCounters*/
                                               I32Ty},    /*NrCallsites*/
                                              false))
          .getCallee());
  ReleaseCtx = cast<Function>(
      M.getOrInsertFunction(
           "__llvm_instrprof_release_context",
           FunctionType::get(Type::getVoidTy(M.getContext()),
                             {
                                 ContextRootTy->getPointerTo(), /*ContextRoot*/
                             },
                             false))
          .getCallee());
  CallsiteInfoTLS =
      new GlobalVariable(M, PointerTy, false, GlobalValue::ExternalLinkage,
                         nullptr, "__llvm_instrprof_callsite");
  CallsiteInfoTLS->setThreadLocal(true);
  CallsiteInfoTLS->setVisibility(llvm::GlobalValue::HiddenVisibility);
  ExpectedCalleeTLS =
      new GlobalVariable(M, PointerTy, false, GlobalValue::ExternalLinkage,
                         nullptr, "__llvm_instrprof_expected_callee");
  ExpectedCalleeTLS->setThreadLocal(true);
  ExpectedCalleeTLS->setVisibility(llvm::GlobalValue::HiddenVisibility);
  
  for (auto &F : M)
    lowerFunction(F);
  return PreservedAnalyses::none();
}

void PGOCtxProfLoweringPass::lowerFunction(Function &F) {
  if (F.isDeclaration())
    return;

  Value *Guid = nullptr;
  uint32_t NrCounters = 0;
  uint32_t NrCallsites = 0;
  [&]() {
    for (const auto &BB : F)
      for (const auto &I : BB) {
        if (const auto *Incr = dyn_cast<InstrProfIncrementInst>(&I)) {
          if (!NrCounters)
            NrCounters = static_cast<uint32_t>(Incr->getNumCounters()->getZExtValue());
        } else if (const auto *CSIntr = dyn_cast<InstrProfCallsite>(&I)) {
          if (!NrCallsites)
            NrCallsites =
                static_cast<uint32_t>(CSIntr->getNumCounters()->getZExtValue());
        }
        if (NrCounters && NrCallsites)
          return;
      }
  }();

  Value *Context = nullptr;
  Value *RealContext = nullptr;

  StructType *ThisContextType = nullptr;
  Value* TheRootContext = nullptr;
  Value *ExpectedCalleeTLSAddr = nullptr;
  Value *CallsiteInfoTLSAddr = nullptr;

  auto &Head = F.getEntryBlock();
  for (auto &I : Head) {
    if (auto *Mark = dyn_cast<InstrProfIncrementInst>(&I)) {
      assert(Mark->getIndex()->isZero());

      IRBuilder<> Builder(Mark);
      // TODO!!!! use InstrProfSymtab::getCanonicalName
      Guid = Builder.getInt64(F.getGUID());
      ThisContextType = StructType::get(
          F.getContext(),
          {ContextNodeTy, ArrayType::get(Builder.getInt64Ty(), NrCounters),
           ArrayType::get(Builder.getPtrTy(), NrCallsites)});
      auto Iter = ContextRootMap.find(&F);
      if (Iter != ContextRootMap.end()) {
        TheRootContext = Iter->second;
        Context = Builder.CreateCall(
            StartCtx,
            {TheRootContext, Guid, Builder.getInt32(NrCounters),
             Builder.getInt32(NrCallsites)});
      } else {
        Context =
            Builder.CreateCall(GetCtx, {&F, Guid, Builder.getInt32(NrCounters),
                                        Builder.getInt32(NrCallsites)});
      }
      auto *CtxAsInt = Builder.CreatePtrToInt(Context, Builder.getInt64Ty());
      if (NrCallsites > 0) {
        auto *Index = Builder.CreateAnd(CtxAsInt, Builder.getInt64(1));
        ExpectedCalleeTLSAddr = Builder.CreateGEP(
            Builder.getInt8Ty()->getPointerTo(),
            Builder.CreateThreadLocalAddress(ExpectedCalleeTLS), {Index});
        CallsiteInfoTLSAddr = Builder.CreateGEP(
            Builder.getInt32Ty(),
            Builder.CreateThreadLocalAddress(CallsiteInfoTLS), {Index});
      }
      RealContext = Builder.CreateIntToPtr(
          Builder.CreateAnd(CtxAsInt, Builder.getInt64(-2)),
          ThisContextType->getPointerTo());
      I.eraseFromParent();
      break;
    }
  }
  if(!Context) {
    dbgs() << "[instprof] Function doesn't have instrumentation, skipping "
           << F.getName() << "\n";
    return;
  }

  for (auto &BB : F) {
    for (auto &I : llvm::make_early_inc_range(BB)) {
      if (auto *Instr = dyn_cast<InstrProfCntrInstBase>(&I)) {
        IRBuilder<> Builder(Instr);
        switch (Instr->getIntrinsicID()) {
        case llvm::Intrinsic::instrprof_increment:
        case llvm::Intrinsic::instrprof_increment_step: {
          auto *AsStep = cast<InstrProfIncrementInst>(Instr);
          auto *GEP = Builder.CreateGEP(
              ThisContextType, RealContext,
              {Builder.getInt32(0), Builder.getInt32(1), AsStep->getIndex()});
          Builder.CreateStore(
              Builder.CreateAdd(Builder.CreateLoad(Builder.getInt64Ty(), GEP),
                                AsStep->getStep()),
              GEP);
        } break;
        case llvm::Intrinsic::instrprof_callsite:
          auto *CSIntrinsic = dyn_cast<InstrProfCallsite>(Instr);
          Builder.CreateStore(CSIntrinsic->getCallee(), ExpectedCalleeTLSAddr,
                              true);
          Builder.CreateStore(
              Builder.CreateGEP(ThisContextType, Context,
                                {Builder.getInt32(0), Builder.getInt32(2),
                                 CSIntrinsic->getIndex()}),
              CallsiteInfoTLSAddr, true);
          break;
        }
        I.eraseFromParent();
      } else if (TheRootContext && isa<ReturnInst>(I)) {
        IRBuilder<> Builder(&I);
        Builder.CreateCall(ReleaseCtx, {TheRootContext});
      }
    }
  }
}