//===-- EJitWrapperGen.cpp - EmbeddedJIT Wrapper Code Generation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  PASS3: Insert wrapper prologue in every ejit_entry function. Uses the
//  single-function mixed scheme: wraps the original body in a fallback
//  block with a JIT dispatch path. No separate wrapper function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-wrapper-gen"

static cl::opt<bool> EJitNoInlineEntry(
    "ejit-noinline-entry", cl::init(true), cl::Hidden,
    cl::desc("Add noinline attribute to ejit_entry functions to prevent the "
             "CGSCC inliner from duplicating the JIT wrapper into callers"));

namespace {

struct PeriodArrIndInfo {
  std::string PeriodName;
  unsigned ArgIndex;
};

static SmallVector<PeriodArrIndInfo, 4>
getPeriodArrIndInfo(const Function &F) {
  SmallVector<PeriodArrIndInfo, 4> Result;
  MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
  if (!MD)
    return Result;

  for (const MDOperand &Op : MD->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() < 3)
      continue;
    if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
      if (Tag->getString() == TAG_EJIT_PERIOD_ARR_IND) {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        auto *IdxC = dyn_cast<ConstantAsMetadata>(Sub->getOperand(2));
        if (PN && IdxC)
          if (auto *CI = dyn_cast<ConstantInt>(IdxC->getValue()))
            Result.push_back({PN->getString().str(),
                              static_cast<unsigned>(CI->getZExtValue())});
      }
    }
  }
  return Result;
}

} // anonymous namespace

PreservedAnalyses
EJitWrapperGenPass::run(Module &M, ModuleAnalysisManager &AM) {
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);

  SmallVector<Function *, 4> EntryFuncs;
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (hasMDStringEntry(MD, TAG_EJIT_ENTRY) && !F.isDeclaration())
      EntryFuncs.push_back(&F);
  }

  if (EntryFuncs.empty())
    return PreservedAnalyses::all();

  // Declare ejit_compile_or_get (only if we have entry functions)
  M.getOrInsertFunction(FN_COMPILE_OR_GET,
      FunctionType::get(PtrTy, {PtrTy, PtrTy, Type::getInt32Ty(Ctx), PtrTy}, false));

  auto isAlreadyWrapped = [](Function &F) -> bool {
    if (!F.getEntryBlock().getName().starts_with("jit_entry"))
      return false;
    for (Instruction &I : F.getEntryBlock())
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (CB->getCalledFunction() &&
            CB->getCalledFunction()->getName() == FN_COMPILE_OR_GET)
          return true;
    return false;
  };

  LLVM_DEBUG(dbgs() << "ejit-wrapper-gen: " << EntryFuncs.size()
                    << " entry function(s)\n");
  bool Changed = false;
  for (Function *F : EntryFuncs) {
    LLVM_DEBUG(dbgs() << "ejit-wrapper-gen: wrapping " << F->getName() << "\n");
    // Idempotency guard: skip functions already wrapped by an earlier pass run.
    // PASS3 may be invoked multiple times via EJitAotModulePass (e.g. O1+O2
    // pipelines), and re-wrapping produces broken PHI nodes referencing stale
    // predecessor blocks.
    if (isAlreadyWrapped(*F))
      continue;

    // Prevent the CGSCC inliner from inlining the wrapped function into
    // callers. Each call site would duplicate the JIT dispatch logic (alloca,
    // ejit_compile_or_get call, indirect call) and the inliner may produce
    // inconsistent AOT fallback code depending on call-site context.
    if (EJitNoInlineEntry)
      F->addFnAttr(Attribute::NoInline);

    auto PeriodInds = getPeriodArrIndInfo(*F);
    unsigned DimCount = PeriodInds.size();

    // Save original entry block
    BasicBlock &OrigEntry = F->getEntryBlock();

    // Create three new blocks
    auto *JitEntry = BasicBlock::Create(Ctx, "jit_entry", F, &OrigEntry);
    auto *JitFallback = BasicBlock::Create(Ctx, "jit_fallback", F);
    auto *JitDispatch = BasicBlock::Create(Ctx, "jit_dispatch", F);

    // Splice all instructions from OrigEntry to jit_fallback
    JitFallback->splice(JitFallback->end(), &OrigEntry, OrigEntry.begin(),
                        OrigEntry.end());

    // Fix PHI nodes in successor blocks that reference the original entry block.
    // Optimization passes may have created PHI nodes pointing at the entry block,
    // and after we replace it with jit_fallback those PHI entries must be updated.
    OrigEntry.replaceAllUsesWith(JitFallback);

    // Delete the now-empty original entry block
    OrigEntry.eraseFromParent();

    // Build wrapper prologue in jit_entry
    IRBuilder<> Builder(JitEntry);

    // Build dims array (if any dimensions)
    Value *DimsPtr = ConstantPointerNull::get(PtrTy);
    Value *CountVal = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

    if (DimCount > 0) {
      // ejit_dim_t = { ptr, i8 }  (matches EJitRuntime.h: uint8_t index)
      auto *DimTy = StructType::get(Ctx, {PtrTy, Type::getInt8Ty(Ctx)});
      auto *DimsAlloca = Builder.CreateAlloca(DimTy,
          ConstantInt::get(Type::getInt32Ty(Ctx), DimCount));

      for (unsigned I = 0; I < DimCount; ++I) {
        Value *Idx = ConstantInt::get(Type::getInt32Ty(Ctx), I);
        Value *DimPtr = Builder.CreateInBoundsGEP(DimTy, DimsAlloca, {Idx});

        // name field
        Value *NamePtr = Builder.CreateInBoundsGEP(
            DimTy, DimPtr, {ConstantInt::get(Type::getInt32Ty(Ctx), 0),
                             ConstantInt::get(Type::getInt32Ty(Ctx), 0)});
        Value *NameStr = Builder.CreateGlobalString(PeriodInds[I].PeriodName);
        Builder.CreateStore(NameStr, NamePtr);

        // index field
        Value *IdxFieldPtr = Builder.CreateInBoundsGEP(
            DimTy, DimPtr, {ConstantInt::get(Type::getInt32Ty(Ctx), 0),
                             ConstantInt::get(Type::getInt32Ty(Ctx), 1)});
        Value *ArgVal = F->getArg(PeriodInds[I].ArgIndex);
        Value *IdxVal = Builder.CreateZExtOrTrunc(ArgVal, Type::getInt8Ty(Ctx));
        Builder.CreateStore(IdxVal, IdxFieldPtr);
      }

      DimsPtr = Builder.CreateBitCast(DimsAlloca, PtrTy);
      CountVal = ConstantInt::get(Type::getInt32Ty(Ctx), DimCount);
    }

    // Build funcName
    Value *FuncNameStr = Builder.CreateGlobalString(F->getName());

    // Call ejit_compile_or_get(funcName, dims, count, null)
    FunctionCallee CompileFn = M.getFunction(FN_COMPILE_OR_GET);
    Value *JitResult = Builder.CreateCall(
        CompileFn, {FuncNameStr, DimsPtr, CountVal,
                    ConstantPointerNull::get(PtrTy)});

    // Null check branch
    Value *IsNull = Builder.CreateIsNull(JitResult);
    Builder.CreateCondBr(IsNull, JitFallback, JitDispatch);

    // jit_dispatch: cast function pointer and call
    Builder.SetInsertPoint(JitDispatch);

    // Build argument list for indirect call
    SmallVector<Value *, 8> Args;
    for (auto &Arg : F->args())
      Args.push_back(&Arg);

    if (F->getReturnType()->isVoidTy()) {
      Builder.CreateCall(F->getFunctionType(), JitResult, Args);
      Builder.CreateRetVoid();
    } else {
      Value *RetVal = Builder.CreateCall(F->getFunctionType(), JitResult, Args);
      Builder.CreateRet(RetVal);
    }

    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
