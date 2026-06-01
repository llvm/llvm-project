//===-- EJitRegisterPeriod.cpp - EmbeddedJIT Period Registration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  PASS2: Scan the module for ejit_period and ejit_period_arr global
//  variables and generate runtime registration calls in ejit_auto_register.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::ejit;

extern cl::opt<bool> EJitNoGlobalCtors;

static void
generateRegistryTablePeriod(
    Module &M,
    const SmallVectorImpl<std::tuple<GlobalVariable *, std::string,
                                     std::string, uint32_t>> &PeriodArrays,
    const SmallVectorImpl<std::pair<GlobalVariable *, std::string>> &StaticVars);

#define DEBUG_TYPE "ejit-register-period"

PreservedAnalyses
EJitRegisterPeriodPass::run(Module &M, ModuleAnalysisManager &AM) {
  // Collect period variables
  SmallVector<std::tuple<GlobalVariable *, std::string, std::string, uint32_t>, 8>
      PeriodArrays; // {GV, periodName, varName, arraySize}
  SmallVector<std::pair<GlobalVariable *, std::string>, 8>
      StaticVars; // {GV, varName}

  for (GlobalVariable &GV : M.globals()) {
    MDNode *MD = GV.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;

    std::string varName = GV.getName().str();

    if (hasMDStringEntry(MD, TAG_EJIT_PERIOD_ARR)) {
      StringRef periodName = getMDStringValue(MD, TAG_EJIT_PERIOD_ARR);
      uint32_t size = getMDIntValue(MD, TAG_EJIT_PERIOD_ARR);
      PeriodArrays.push_back({&GV, periodName.str(), varName, size});
    }

    if (hasMDStringEntry(MD, TAG_EJIT_PERIOD)) {
      StaticVars.push_back({&GV, varName});
    }
  }

  if (PeriodArrays.empty() && StaticVars.empty()) {
    LLVM_DEBUG(dbgs() << "ejit-register-period: no period vars\n");
    return PreservedAnalyses::all();
  }
  LLVM_DEBUG(dbgs() << "ejit-register-period: " << PeriodArrays.size()
                    << " arrays, " << StaticVars.size() << " static vars\n");

  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);

  // Declare runtime functions
  M.getOrInsertFunction(FN_REGISTER_PERIOD_ARRAY,
      FunctionType::get(Type::getVoidTy(Ctx),
                        {PtrTy, PtrTy, PtrTy, Type::getInt64Ty(Ctx)}, false));
  M.getOrInsertFunction(FN_REGISTER_STATIC_VAR,
      FunctionType::get(Type::getVoidTy(Ctx), {PtrTy, PtrTy}, false));

  // Find or create ejit_auto_register
  Function *AutoReg = M.getFunction(FN_AUTO_REGISTER);
  if (!AutoReg) {
    auto *AutoRegTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    AutoReg = Function::Create(AutoRegTy, GlobalValue::InternalLinkage,
                               FN_AUTO_REGISTER, &M);
    BasicBlock::Create(Ctx, "entry", AutoReg);
    ReturnInst::Create(Ctx, &AutoReg->getEntryBlock());
  }

  // Insert calls before return
  BasicBlock *EntryBB = &AutoReg->getEntryBlock();
  Instruction *Ret = EntryBB->getTerminator();
  FunctionCallee FnRegArr = M.getFunction(FN_REGISTER_PERIOD_ARRAY);
  FunctionCallee FnRegSV = M.getFunction(FN_REGISTER_STATIC_VAR);

  for (auto &[GV, PeriodName, VarName, Size] : PeriodArrays) {
    IRBuilder<> Builder(Ret);
    Value *PN = Builder.CreateGlobalString(PeriodName);
    Value *VN = Builder.CreateGlobalString(VarName);
    Value *BA = Builder.CreateBitCast(GV, PtrTy);
    Builder.CreateCall(FnRegArr, {PN, VN, BA, ConstantInt::get(Type::getInt64Ty(Ctx), Size)});
  }

  for (auto &[GV, VarName] : StaticVars) {
    IRBuilder<> Builder(Ret);
    Value *VN = Builder.CreateGlobalString(VarName);
    Value *VA = Builder.CreateBitCast(GV, PtrTy);
    Builder.CreateCall(FnRegSV, {VN, VA});
  }

  if (!EJitNoGlobalCtors)
    appendToGlobalCtors(M, AutoReg, EJIT_CTOR_PRIORITY);

  // Always build the static registry table for bare-metal / testing fallback.
  generateRegistryTablePeriod(M, PeriodArrays, StaticVars);

  return PreservedAnalyses::none();
}

/// Build a global constant array __ejit_registry_period[] that ejit_init()
/// walks on bare-metal where global constructors are unavailable.
static void
generateRegistryTablePeriod(
    Module &M,
    const SmallVectorImpl<std::tuple<GlobalVariable *, std::string,
                                     std::string, uint32_t>> &PeriodArrays,
    const SmallVectorImpl<std::pair<GlobalVariable *, std::string>> &StaticVars) {
  LLVMContext &Ctx = M.getContext();
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  StructType *EntryTy = StructType::get(
      Ctx, {I32Ty, PtrTy, PtrTy, PtrTy, I64Ty}, /*isPacked=*/false);

  SmallVector<Constant *, 16> Entries;

  // Period array entries
  for (auto &[GV, PeriodName, VarName, Size] : PeriodArrays) {
    Entries.push_back(ConstantStruct::get(EntryTy, {
        ConstantInt::get(I32Ty, 1),                  // EJIT_REG_PERIOD_ARRAY
        ConstantExpr::getBitCast(
            M.getOrInsertGlobal(PeriodName, PtrTy), PtrTy),
        ConstantExpr::getBitCast(
            M.getOrInsertGlobal(VarName, PtrTy), PtrTy),
        ConstantExpr::getBitCast(GV, PtrTy),         // base address
        ConstantInt::get(I64Ty, Size),
    }));
  }

  // Static var entries
  for (auto &[GV, VarName] : StaticVars) {
    Entries.push_back(ConstantStruct::get(EntryTy, {
        ConstantInt::get(I32Ty, 2),                  // EJIT_REG_STATIC_VAR
        ConstantExpr::getBitCast(
            M.getOrInsertGlobal(VarName, PtrTy), PtrTy),
        ConstantPointerNull::get(PtrTy),
        ConstantExpr::getBitCast(GV, PtrTy),
        ConstantInt::get(I64Ty, 0),
    }));
  }

  // Sentinel
  Entries.push_back(ConstantStruct::get(EntryTy, {
      ConstantInt::get(I32Ty, 4),                    // EJIT_REG_NONE
      ConstantPointerNull::get(PtrTy),
      ConstantPointerNull::get(PtrTy),
      ConstantPointerNull::get(PtrTy),
      ConstantInt::get(I64Ty, 0),
  }));

  ArrayType *ArrayTy = ArrayType::get(EntryTy, Entries.size());
  Constant *ArrayInit = ConstantArray::get(ArrayTy, Entries);

  (void)new GlobalVariable(M, ArrayTy, /*isConstant=*/true,
                           GlobalValue::ExternalLinkage, ArrayInit,
                           "__ejit_registry_period");
}
