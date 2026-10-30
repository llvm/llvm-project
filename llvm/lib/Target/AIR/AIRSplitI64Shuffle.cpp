//===- AIRSplitI64Shuffle.cpp - Rewrite i64 simd shuffle ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRSplitI64Shuffle.h"
#include "AIRKernelParams.h"
#include "AIR.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-split-i64-shuffle"

static constexpr StringLiteral kShufflePrefix("air.simd_shuffle");
static constexpr StringLiteral kI64Suffix(".i64");
static constexpr StringLiteral kV2I32Suffix(".v2i32");

static bool isI64ShuffleDecl(const Function &F) {
  return F.isDeclaration() && F.getName().starts_with(kShufflePrefix) &&
         F.getName().ends_with(kI64Suffix);
}


static constexpr StringLiteral kSimdlaneName("air.thread_index_in_simdgroup");

// Whether V has the same value in every thread: constants, buffer pointers and
// what is computed from them only (loads from the constant address space at a
// uniform address included). Conservative: anything else is per-lane.
static bool isUniform(Value *V, unsigned Depth = 0) {
  if (isa<Constant>(V))
    return true;
  if (Depth > 8)
    return false;
  // (A kernel's buffer; a function the kernel calls may be passed anything.)
  if (auto *A = dyn_cast<Argument>(V))
    return A->getType()->isPointerTy() && !isCalledFunction(*A->getParent());
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace() == 2 && !LI->isVolatile() &&
           isUniform(LI->getPointerOperand(), Depth + 1);
  if (isa<BinaryOperator, CastInst, GetElementPtrInst, CmpInst, SelectInst,
          ExtractElementInst, ExtractValueInst>(I))
    return all_of(I->operands(),
                  [&](Value *Op) { return isUniform(Op, Depth + 1); });
  return false;
}

// The thread's index in its SIMD-group in F: a call AIRSystemValues turns
// into a parameter (the kernel's, passed down to F when F is a function it
// calls).
static Value *simdLane(Function *F) {
  Module &M = *F->getParent();
  IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
  return B.CreateCall(
      M.getOrInsertFunction(kSimdlaneName, B.getInt32Ty()), {}, "simdlane");
}

// Metal takes the delta of air.simd_shuffle_{up,down,xor} (and of the quad
// forms) from one lane for the whole SIMD-group: with a per-lane delta, lanes
// read the wrong lane (PTX shfl.sync with a per-lane b operand,
// PTX-formalization warp-0, warp-1, warp-2). air.simd_shuffle and
// air.quad_shuffle take a per-lane index; a relative shuffle whose delta is
// not uniform becomes one of them, at the lane the relative one reads (its
// own when that is outside the SIMD-group or quad, as Metal and AGX do).
static bool perLaneRelativeShuffles(Module &M) {
  MapVector<Function *, SmallVector<CallInst *, 4>> Calls;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction()) {
          StringRef N = Callee->getName();
          if ((N.starts_with("air.simd_shuffle_") ||
               N.starts_with("air.quad_shuffle_")) &&
              CI->arg_size() == 2 && !isUniform(CI->getArgOperand(1)))
            Calls[&F].push_back(CI);
        }
  for (auto &[FKey, FCalls] : Calls) {
    Function *F = FKey;
    Value *Lane = simdLane(F);
    for (CallInst *CI : FCalls) {
      StringRef N = CI->getCalledFunction()->getName();
      bool Quad = N.starts_with("air.quad_");
      StringRef Rest = N.drop_front(StringRef("air.simd_shuffle_").size());
      auto [Op, Sfx] = Rest.split('.');
      if (Op != "up" && Op != "down" && Op != "xor")
        continue;
      IRBuilder<> B(CI);
      Type *I32 = B.getInt32Ty();
      unsigned Width = Quad ? 4 : 32;
      Value *Pos = Quad ? B.CreateAnd(Lane, 3) : Lane;
      Value *D = B.CreateZExt(CI->getArgOperand(1), I32);
      Value *Src, *Ok;
      if (Op == "up") {
        Src = B.CreateSub(Pos, D);
        Ok = B.CreateICmpUGE(Pos, D);
      } else if (Op == "down") {
        Src = B.CreateAdd(Pos, D);
        Ok = B.CreateICmpULT(Src, B.getInt32(Width));
      } else {
        Src = B.CreateXor(Pos, D);
        Ok = B.CreateICmpULT(Src, B.getInt32(Width));
      }
      Value *Idx = B.CreateTrunc(B.CreateSelect(Ok, Src, Pos), B.getInt16Ty(),
                                 CI->getName() + ".lane");
      std::string Name =
          ((Quad ? "air.quad_shuffle." : "air.simd_shuffle.") + Sfx).str();
      Function *Decl = M.getFunction(Name);
      if (!Decl) {
        Decl = Function::Create(CI->getFunctionType(),
                                GlobalValue::ExternalLinkage, Name, M);
        Decl->copyAttributesFrom(CI->getCalledFunction());
      }
      CallInst *New = B.CreateCall(Decl, {CI->getArgOperand(0), Idx});
      New->setAttributes(CI->getAttributes());
      New->takeName(CI);
      CI->replaceAllUsesWith(New);
      Function *Old = CI->getCalledFunction();
      CI->eraseFromParent();
      if (Old->use_empty())
        Old->eraseFromParent();
    }
  }
  return !Calls.empty();
}

static bool splitI64Shuffle(Module &M) {
  bool Changed = perLaneRelativeShuffles(M);
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I16 = Type::getInt16Ty(Ctx);
  Type *V2I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 2);

  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (isI64ShuffleDecl(*Callee))
              Calls.push_back(CI);

  for (CallInst *CI : Calls) {
    // Swap the trailing ".i64" for ".v2i32"; the JIT accepts the vector form.
    std::string NewName = CI->getCalledFunction()->getName().str();
    NewName.replace(NewName.size() - kI64Suffix.size(), kI64Suffix.size(),
                    kV2I32Suffix);

    FunctionType *NewFTy = FunctionType::get(V2I32, {V2I32, I16}, false);
    FunctionCallee NewFC = M.getOrInsertFunction(NewName, NewFTy);

    IRBuilder<> B(CI);
    Value *Vec =
        B.CreateBitCast(CI->getArgOperand(0), V2I32, CI->getName() + "_vv");
    Value *Shuf =
        B.CreateCall(NewFC, {Vec, CI->getArgOperand(1)}, CI->getName() + "_sv");
    Value *Result = B.CreateBitCast(Shuf, I64, CI->getName());

    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
  }

  SmallVector<Function *, 4> DeadDecls;
  for (Function &F : M)
    if (isI64ShuffleDecl(F) && F.use_empty())
      DeadDecls.push_back(&F);
  for (Function *F : DeadDecls)
    F->eraseFromParent();

  return Changed || !Calls.empty();
}

PreservedAnalyses AIRSplitI64ShufflePass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return splitI64Shuffle(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool AIRSplitI64ShuffleLegacy::runOnModule(Module &M) {
  return splitI64Shuffle(M);
}

char AIRSplitI64ShuffleLegacy::ID = 0;

INITIALIZE_PASS(AIRSplitI64ShuffleLegacy, DEBUG_TYPE,
                "AIR rewrite i64 simd shuffle to v2i32", false, false)

ModulePass *llvm::createAIRSplitI64ShuffleLegacyPass() {
  return new AIRSplitI64ShuffleLegacy();
}
