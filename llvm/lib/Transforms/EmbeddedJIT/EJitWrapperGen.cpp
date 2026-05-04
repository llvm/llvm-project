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

using namespace llvm;
using namespace llvm::ejit;

namespace {

static bool hasMDStringEntry(const MDNode *Node, StringRef Name) {
  if (!Node)
    return false;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() == 0)
      continue;
    if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
      if (S->getString() == Name)
        return true;
  }
  return false;
}

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

  bool Changed = false;
  for (Function *F : EntryFuncs) {
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

    // Delete the now-empty original entry block
    OrigEntry.eraseFromParent();

    // Build wrapper prologue in jit_entry
    IRBuilder<> Builder(JitEntry);

    // Build dims array (if any dimensions)
    Value *DimsPtr = ConstantPointerNull::get(PtrTy);
    Value *CountVal = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

    if (DimCount > 0) {
      // ejit_dim_t = { ptr, i32 }
      auto *DimTy = StructType::get(Ctx, {PtrTy, Type::getInt32Ty(Ctx)});
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
        Value *IdxVal = Builder.CreateZExtOrTrunc(ArgVal, Type::getInt32Ty(Ctx));
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

    Value *Pfn = Builder.CreatePointerCast(
        JitResult, PointerType::get(F->getFunctionType(), 0));

    if (F->getReturnType()->isVoidTy()) {
      Builder.CreateCall(F->getFunctionType(), Pfn, Args);
      Builder.CreateRetVoid();
    } else {
      Value *RetVal = Builder.CreateCall(F->getFunctionType(), Pfn, Args);
      Builder.CreateRet(RetVal);
    }

    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
