//===-- OffloadSanitizer.cpp - Offload sanitizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/OffloadSanitizer.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "offload-sanitizer"

namespace {

class OffloadSanitizerImpl final {
public:
  OffloadSanitizerImpl(Module &M, FunctionAnalysisManager &FAM)
      : M(M), FAM(FAM), Ctx(M.getContext()) {}

  bool instrument();

private:
  bool shouldInstrumentFunction(Function &Fn);
  bool instrumentFunction(Function &Fn);
  bool instrumentTrapInstructions(SmallVectorImpl<IntrinsicInst *> &TrapCalls);
  bool instrumentUnreachableInstructions(
      SmallVectorImpl<UnreachableInst *> &UnreachableInsts);

  FunctionCallee getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
    }
    return FC;
  }

  /// void __offload_san_trap_info(Int64Ty);
  FunctionCallee TrapInfoFn;
  FunctionCallee getTrapInfoFn() {
    return getOrCreateFn(TrapInfoFn, "__offload_san_trap_info", VoidTy,
                         {/*PC*/ Int64Ty});
  }

  /// void __offload_san_unreachable_info(Int64Ty);
  FunctionCallee UnreachableInfoFn;
  FunctionCallee getUnreachableInfoFn() {
    return getOrCreateFn(UnreachableInfoFn, "__offload_san_unreachable_info",
                         VoidTy, {/*PC*/ Int64Ty});
  }

  CallInst *createCall(IRBuilder<> &IRB, FunctionCallee Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "") {
    Calls.push_back(IRB.CreateCall(Callee, Args, Name));
    return Calls.back();
  }
  SmallVector<CallInst *> Calls;

  Value *getPC(IRBuilder<> &IRB) {
    return IRB.CreateIntrinsic(Int64Ty, Intrinsic::amdgcn_s_getpc, {}, nullptr,
                               "PC");
  }

  Module &M;
  FunctionAnalysisManager &FAM;
  LLVMContext &Ctx;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();
};

} // end anonymous namespace

bool OffloadSanitizerImpl::shouldInstrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  if (Fn.getName().contains("ompx") || Fn.getName().contains("__kmpc") ||
      Fn.getName().starts_with("rpc_"))
    return false;
  return !Fn.hasFnAttribute(Attribute::DisableSanitizerInstrumentation);
}

bool OffloadSanitizerImpl::instrumentTrapInstructions(
    SmallVectorImpl<IntrinsicInst *> &TrapCalls) {
  bool Changed = false;
  for (auto *II : TrapCalls) {
    IRBuilder<> IRB(II);
    createCall(IRB, getTrapInfoFn(), {getPC(IRB)});
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentUnreachableInstructions(
    SmallVectorImpl<UnreachableInst *> &UnreachableInsts) {
  bool Changed = false;
  for (auto *II : UnreachableInsts) {
    // Skip unreachables after traps since we instrument those as well.
    if (&II->getParent()->front() != II)
      if (auto *CI = dyn_cast<CallInst>(II->getPrevNode()))
        if (CI->getIntrinsicID() == Intrinsic::trap)
          continue;
    IRBuilder<> IRB(II);
    createCall(IRB, getUnreachableInfoFn(), {getPC(IRB)});
  }
  return Changed;
}

bool OffloadSanitizerImpl::instrumentFunction(Function &Fn) {
  if (!shouldInstrumentFunction(Fn))
    return false;

  SmallVector<UnreachableInst *> UnreachableInsts;
  SmallVector<IntrinsicInst *> TrapCalls;

  bool Changed = false;
  for (auto &I : instructions(Fn)) {
    switch (I.getOpcode()) {
    case Instruction::Unreachable:
      UnreachableInsts.push_back(cast<UnreachableInst>(&I));
      break;
    case Instruction::Call: {
      auto &CI = cast<CallInst>(I);
      if (auto *II = dyn_cast<IntrinsicInst>(&CI))
        if (II->getIntrinsicID() == Intrinsic::trap)
          TrapCalls.push_back(II);
      break;
    }
    default:
      break;
    }
  }

  Changed |= instrumentTrapInstructions(TrapCalls);
  Changed |= instrumentUnreachableInstructions(UnreachableInsts);

  return Changed;
}

bool OffloadSanitizerImpl::instrument() {
  bool Changed = false;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  removeFromUsedLists(M, [&](Constant *C) {
    if (!C->getName().starts_with("__offload_san"))
      return false;
    return Changed = true;
  });

  return Changed;
}

PreservedAnalyses OffloadSanitizerPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  OffloadSanitizerImpl Impl(M, FAM);
  if (!Impl.instrument())
    return PreservedAnalyses::all();
  LLVM_DEBUG(M.dump());
  return PreservedAnalyses::none();
}
