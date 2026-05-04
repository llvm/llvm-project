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

static StringRef getMDStringValue(const MDNode *Node, StringRef Tag) {
  if (!Node)
    return {};
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (Sub && Sub->getNumOperands() >= 2) {
      if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
        if (S->getString() == Tag)
          if (auto *V = dyn_cast<MDString>(Sub->getOperand(1)))
            return V->getString();
    }
  }
  return {};
}

static uint32_t getMDIntValue(const MDNode *Node, StringRef Tag) {
  if (!Node)
    return 0;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (Sub && Sub->getNumOperands() >= 3) {
      if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
        if (S->getString() == Tag)
          if (auto *C = dyn_cast<ConstantAsMetadata>(Sub->getOperand(2)))
            if (auto *CI = dyn_cast<ConstantInt>(C->getValue()))
              return static_cast<uint32_t>(CI->getZExtValue());
    }
  }
  return 0;
}

} // anonymous namespace

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

  if (PeriodArrays.empty() && StaticVars.empty())
    return PreservedAnalyses::all();

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

  // Register in global_ctors
  bool HasEntry = false;
  if (GlobalVariable *Ctors = M.getGlobalVariable(CTORS_GLOBAL)) {
    if (auto *Arr = dyn_cast<ConstantArray>(Ctors->getInitializer()))
      for (auto &Op : Arr->operands())
        if (auto *CS = dyn_cast<ConstantStruct>(Op))
          if (auto *F = dyn_cast<Function>(CS->getOperand(1)))
            if (F == AutoReg) { HasEntry = true; break; }
  }

  if (!HasEntry) {
    auto *Ty = StructType::get(Ctx, {Type::getInt32Ty(Ctx),
                                      AutoReg->getType(), PtrTy});
    auto *Entry = ConstantStruct::get(Ty,
        {ConstantInt::get(Type::getInt32Ty(Ctx), EJIT_CTOR_PRIORITY),
         AutoReg, ConstantPointerNull::get(PtrTy)});

    if (GlobalVariable *Ctors = M.getGlobalVariable(CTORS_GLOBAL)) {
      if (auto *Arr = dyn_cast<ConstantArray>(Ctors->getInitializer())) {
        SmallVector<Constant *, 8> E;
        for (auto &Op : Arr->operands()) E.push_back(cast<Constant>(Op));
        E.push_back(Entry);
        Ctors->setInitializer(ConstantArray::get(
            ArrayType::get(Ty, E.size()), E));
      }
    } else {
      auto *ATy = ArrayType::get(Ty, 1);
      new GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(ATy, {Entry}), CTORS_GLOBAL);
    }
  }

  return PreservedAnalyses::none();
}
