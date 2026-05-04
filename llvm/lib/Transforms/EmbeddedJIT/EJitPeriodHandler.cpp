//===-- EJitPeriodHandler.cpp - EmbeddedJIT Lifecycle Handler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  PASS4: Insert ejit_deactivate_array at function entry and
//  ejit_activate_array before every return in ejit_period_lc functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

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

struct LifecycleInfo {
  std::string PeriodName;
  unsigned ArgIndex;
  GlobalVariable *ArrayGV;
};

static SmallVector<LifecycleInfo, 4>
collectLifecycleInfo(Module &M, Function &F) {
  SmallVector<LifecycleInfo, 4> Result;
  MDNode *MD = F.getMetadata("ejit.metadata");
  if (!MD)
    return Result;

  // Collect period names from ejit_period_lc entries
  for (const MDOperand &Op : MD->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() < 2)
      continue;
    if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
      if (Tag->getString() == "ejit_period_lc") {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        if (PN)
          Result.push_back({PN->getString().str(), 0, nullptr});
      }
    }
  }

  // Find arg index from ejit_period_arr_ind entries
  for (const MDOperand &Op : MD->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() < 3)
      continue;
    if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
      if (Tag->getString() == "ejit_period_arr_ind") {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        if (auto *IdxC = dyn_cast<ConstantAsMetadata>(Sub->getOperand(2)))
          if (auto *CI = dyn_cast<ConstantInt>(IdxC->getValue()))
            for (auto &LC : Result)
              if (LC.PeriodName == PN->getString())
                LC.ArgIndex = static_cast<unsigned>(CI->getZExtValue());
      }
    }
  }

  // Find matching global array for each period
  for (GlobalVariable &GV : M.globals()) {
    MDNode *GMD = GV.getMetadata("ejit.metadata");
    if (!GMD)
      continue;
    for (auto &LC : Result) {
      for (const MDOperand &Op : GMD->operands()) {
        auto *Sub = dyn_cast<MDNode>(Op.get());
        if (!Sub || Sub->getNumOperands() < 2)
          continue;
        if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
          if (Tag->getString() == "ejit_period_arr") {
            auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
            if (PN && PN->getString() == LC.PeriodName)
              LC.ArrayGV = &GV;
          }
        }
      }
    }
  }

  return Result;
}

} // anonymous namespace

PreservedAnalyses
EJitPeriodHandlerPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool Changed = false;
  SmallVector<Function *, 4> LcFuncs;
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata("ejit.metadata");
    if (hasMDStringEntry(MD, "ejit_period_lc") && !F.isDeclaration())
      LcFuncs.push_back(&F);
  }

  if (LcFuncs.empty())
    return PreservedAnalyses::all();

  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *VoidTy = Type::getVoidTy(Ctx);

  // Declare runtime functions (only if we have lc functions)
  M.getOrInsertFunction("ejit_deactivate_array",
      FunctionType::get(VoidTy, {PtrTy, PtrTy, I32Ty}, false));
  M.getOrInsertFunction("ejit_activate_array",
      FunctionType::get(VoidTy, {PtrTy, PtrTy, I32Ty}, false));

  for (Function *F : LcFuncs) {
    auto LcInfos = collectLifecycleInfo(M, *F);
    if (LcInfos.empty())
      continue;

    FunctionCallee DeactivateFn = M.getFunction("ejit_deactivate_array");
    FunctionCallee ActivateFn = M.getFunction("ejit_activate_array");

    // Insert deactivate at entry (after allocas, before first real instruction)
    BasicBlock &EntryBB = F->getEntryBlock();
    Instruction *InsertPt = &*EntryBB.begin();
    while (isa<AllocaInst>(InsertPt) && InsertPt != EntryBB.getTerminator())
      InsertPt = InsertPt->getNextNode();

    for (auto &LC : LcInfos) {
      IRBuilder<> Builder(InsertPt);
      Value *PN = Builder.CreateGlobalString(LC.PeriodName);
      Value *AP = LC.ArrayGV
          ? Builder.CreateBitCast(LC.ArrayGV, PtrTy)
          : ConstantPointerNull::get(PtrTy);
      Value *Idx = Builder.getInt32(LC.ArgIndex);
      Builder.CreateCall(DeactivateFn, {PN, AP, Idx});
    }

    // Insert activate before each return (reverse order for RAII pairing)
    SmallVector<ReturnInst *, 4> Returns;
    for (BasicBlock &BB : *F)
      if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
        Returns.push_back(RI);

    for (ReturnInst *RI : Returns) {
      // Reverse iterate over LcInfos
      for (auto It = LcInfos.rbegin(); It != LcInfos.rend(); ++It) {
        IRBuilder<> Builder(RI);
        Value *PN = Builder.CreateGlobalString(It->PeriodName);
        Value *AP = It->ArrayGV
            ? Builder.CreateBitCast(It->ArrayGV, PtrTy)
            : ConstantPointerNull::get(PtrTy);
        Value *Idx = Builder.getInt32(It->ArgIndex);
        Builder.CreateCall(ActivateFn, {PN, AP, Idx});
      }
    }

    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
