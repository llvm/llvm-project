//===- MetalScalarStoreGuard.cpp - Guard scalar device stores -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalScalarStoreGuard.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-scalar-store-guard"

// Metal device address space. Inlined here since this pass only needs the
// one constant.

// AIR per-thread index intrinsics and the `!air.kernel` named-metadata key.
static constexpr StringLiteral kCallTid("air.thread_position_in_grid");
static constexpr StringLiteral kCallTidTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral kCallSimdlane("air.thread_index_in_simdgroup");
static constexpr StringLiteral kAtomicGlobalPrefix("air.atomic.global");
static constexpr StringLiteral kNMDKernel("air.kernel");

static bool isPerThreadIndexCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  return Name.starts_with(kCallTid) || Name.starts_with(kCallTidTG) ||
         Name.starts_with(kCallSimdlane);
}

static bool isDeviceWriteIntrinsic(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  return Callee->getName().starts_with(kAtomicGlobalPrefix);
}

static bool isScalarKernel(Function &F) {
  if (F.isDeclaration())
    return false;
  bool HasDeviceWrite = false;
  bool HasPerThreadIdx = false;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (SI->getPointerAddressSpace() == metal::AS::Device)
          HasDeviceWrite = true;
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (isPerThreadIndexCall(CI))
          HasPerThreadIdx = true;
        if (isDeviceWriteIntrinsic(CI))
          HasDeviceWrite = true;
      }
    }
  }
  return HasDeviceWrite && !HasPerThreadIdx;
}

static bool scalarStoreGuard(Module &M) {
  // Skip if IR already has !air.kernel metadata (pre-lowered).
  if (M.getNamedMetadata(kNMDKernel))
    return false;

  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  for (Function &F : M) {
    if (!isScalarKernel(F))
      continue;

    // Insert: if (tid.x != 0) ret; before the existing entry body.
    BasicBlock &Entry = F.getEntryBlock();
    auto *TidTGTy = ArrayType::get(I32, 3);
    FunctionType *TidFTy = FunctionType::get(TidTGTy, {}, false);
    FunctionCallee TidFC = M.getOrInsertFunction(kCallTidTG, TidFTy);

    BasicBlock *BodyBB =
        Entry.splitBasicBlock(Entry.getFirstNonPHIIt(), "body");
    BasicBlock *ExitBB = BasicBlock::Create(Ctx, "exit", &F);
    IRBuilder<> ExitB(ExitBB);
    ExitB.CreateRetVoid();

    Entry.getTerminator()->eraseFromParent();
    IRBuilder<> B(&Entry);
    Value *TidResult = B.CreateCall(TidFC, {}, "guard_tid");
    Value *TidX = B.CreateExtractValue(TidResult, {0}, "guard_tid_x");
    Value *IsT0 = B.CreateICmpEQ(TidX, ConstantInt::get(I32, 0), "guard_is_t0");
    B.CreateCondBr(IsT0, BodyBB, ExitBB);

    Changed = true;
  }

  return Changed;
}

PreservedAnalyses MetalScalarStoreGuardPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return scalarStoreGuard(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

bool MetalScalarStoreGuardLegacy::runOnModule(Module &M) {
  return scalarStoreGuard(M);
}

char MetalScalarStoreGuardLegacy::ID = 0;

INITIALIZE_PASS(MetalScalarStoreGuardLegacy, DEBUG_TYPE,
                "Metal Scalar Store Guard", false, false)

ModulePass *llvm::createMetalScalarStoreGuardLegacyPass() {
  return new MetalScalarStoreGuardLegacy();
}
