//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation implements the well known scalar replacement of
/// aggregates transformation but for logical pointers.
/// It tries to identify promotable elements of an aggregate alloca, and
/// promote them to multiple allocas of scalar type.
///
/// FIXME: nested aggregates are not fully optimized (#192619).
/// FIXME: array are not optimized (#192620).
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LogicalSROA.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "logical-sroa"

// Return all lifetime intrinsics with the instruction I as operand.
static SmallVector<LifetimeIntrinsic *>
collectLifetimeIntrinsicsUsing(Instruction &I) {
  SmallVector<LifetimeIntrinsic *> Output;

  for (User *U : I.users()) {
    if (auto *LI = dyn_cast<LifetimeIntrinsic>(U))
      Output.push_back(LI);
  }

  return Output;
}

// Returns true if all direct and indirect users of the alloca
// allow the split.
static bool isAllocaSplittable(StructuredAllocaInst &SAI) {
  SmallVector<Value *> WorkList(SAI.users());
  DenseSet<Value *> Visited;

  // Helper function to enqueue all non-visited users of `I`.
  auto enqueueAllUsers = [&](Instruction *I) {
    for (auto *U : I->users()) {
      if (Visited.contains(U))
        continue;
      WorkList.push_back(U);
    }
  };

  while (!WorkList.empty()) {
    Instruction *I = dyn_cast<Instruction>(WorkList.back());
    WorkList.pop_back();

    // User is not an instruction. Not sure what it it, in
    // doubt, don't split.
    if (!I)
      return false;

    Visited.insert(I);

    // Those allow the alloca split.
    if (isa<LifetimeIntrinsic>(I))
      continue;

    // If we load the whole alloca, we cannot split,
    // otherwise, we can stop looking into derived users.
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (LI->getPointerOperand() == &SAI)
        return false;
      continue;
    }

    // If we store to whole alloca, we cannot split,
    // otherwise, we can stop looking into derived users.
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      if (SI->getPointerOperand() == &SAI)
        return false;
      continue;
    }

    // PHI and Select instruction are not inherently preventing
    // the split, but correctly handling those requires more testing,
    // so postponing this (See #193749)
    if (isa<PHINode>(I) || isa<SelectInst>(I))
      return false;

    if (auto *SGEP = dyn_cast<StructuredGEPInst>(I)) {
      // If the SGEP has no indices and is still there, this probably means the
      // ptr is escaping or uses as-is. For now, we bail out.
      if (SGEP->getNumIndices() == 0)
        return false;

      enqueueAllUsers(SGEP);
      continue;
    }

    // Any other users prevents the split (call, escape, etc).
    return false;
  }

  return true;
}

// Returns a vector with one element for each field of the struct allocated by
// SAI. Each element is a vector of SGEP instruction referencing this field.
// This function ignores lifetime intrinsics.
static SmallVector<SmallVector<StructuredGEPInst *>>
collectPerFieldSGEP(StructuredAllocaInst &SAI) {
  StructType *ST = cast<StructType>(SAI.getAllocationType());
  SmallVector<SmallVector<StructuredGEPInst *>> Output(ST->getNumElements());

  for (User *U : SAI.users()) {
    if (isa<LifetimeIntrinsic>(U))
      continue;

    auto *SGEP = cast<StructuredGEPInst>(U);

    // IR rule: SGEP on struct can only use constant int as indices.
    ConstantInt *Index = cast<ConstantInt>(SGEP->getIndexOperand(0));
    assert(Index->getZExtValue() < Output.size());
    Output[Index->getZExtValue()].push_back(SGEP);
  }

  return Output;
}

// For each lifetime intrinsic in LifetimeIntrinsics, creates a new one, but
// uses V as operand.
static void copyLifetimeIntrinsicFor(IRBuilder<> &B, LifetimeIntrinsic *II,
                                     Value *V) {
  B.SetInsertPoint(II);

  if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
    B.CreateLifetimeStart(V);
  } else if (II->getIntrinsicID() == Intrinsic::lifetime_end) {
    B.CreateLifetimeEnd(V);
  } else
    llvm_unreachable("invalid argument: expected a lifetime intrinsic");
}

static void rewriteSGEPChain(IRBuilder<> &B, StructuredGEPInst *SGEP,
                             StructuredAllocaInst *FieldAlloca) {
  if (SGEP->getNumIndices() == 1) {
    SGEP->replaceAllUsesWith(FieldAlloca);
    SGEP->eraseFromParent();
    return;
  }

  SmallVector<Value *, 4> Indices(llvm::drop_begin(SGEP->indices()));
  B.SetInsertPoint(SGEP);
  auto *I = B.CreateStructuredGEP(FieldAlloca->getAllocationType(), FieldAlloca,
                                  Indices, SGEP->getName());
  SGEP->replaceAllUsesWith(I);
  SGEP->eraseFromParent();
}

static bool runOnStructuredAlloca(StructuredAllocaInst &SAI) {
  // For now, LogicalSROA only handles SGEP on structs.
  StructType *ST = dyn_cast<StructType>(SAI.getAllocationType());
  if (!ST)
    return false;

  if (!isAllocaSplittable(SAI))
    return false;

  auto PerFieldSGEP = collectPerFieldSGEP(SAI);
  assert(PerFieldSGEP.size() == ST->getNumElements());

  auto LifetimeIntrinsics = collectLifetimeIntrinsicsUsing(SAI);
  IRBuilder B(&SAI);
  for (const auto &[FieldIndex, Users] : llvm::enumerate(PerFieldSGEP)) {
    if (Users.empty())
      continue;

    B.SetInsertPoint(&SAI);
    auto *FieldAlloca = cast<StructuredAllocaInst>(
        B.CreateStructuredAlloca(ST->getElementType(FieldIndex)));

    for (auto II : LifetimeIntrinsics)
      copyLifetimeIntrinsicFor(B, II, FieldAlloca);

    for (StructuredGEPInst *SGEP : Users)
      rewriteSGEPChain(B, SGEP, FieldAlloca);
  }

  for (auto *II : LifetimeIntrinsics)
    II->eraseFromParent();
  SAI.eraseFromParent();
  return true;
}

static bool runLogicalSROA(Function &F) {
  SmallVector<StructuredAllocaInst *> Worklist;
  BasicBlock &EntryBB = F.getEntryBlock();
  for (Instruction &I : EntryBB) {
    if (StructuredAllocaInst *SAI = dyn_cast<StructuredAllocaInst>(&I))
      Worklist.push_back(SAI);
  }

  bool Changed = false;
  for (StructuredAllocaInst *SAI : Worklist)
    Changed |= runOnStructuredAlloca(*SAI);
  return Changed;
}

PreservedAnalyses LogicalSROAPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  if (!runLogicalSROA(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

LogicalSROAPass::LogicalSROAPass() {}
