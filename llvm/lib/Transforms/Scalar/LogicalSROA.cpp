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

namespace {
struct FieldSGEPs {
  /// All GEPs that access this specific field.
  SmallVector<StructuredGEPInst *, 1> SGEPs;
  /// Type of the access.
  Type *Ty;
  /// The number of index arguments common to the collection of GEPs.
  unsigned NumIndices;
};
} // namespace

/// Returns a vector with one element for each field that is independently
/// accessed of an SAI. Each element catalogues the list of GEPs for this field
/// as well as the information needed to rewrite the GEP to a smaller alloca.
/// This function ignores lifetime intrinsics.
static SmallVector<FieldSGEPs> collectPerFieldSGEPs(StructuredAllocaInst &SAI) {
  SmallVector<FieldSGEPs> PerFieldSGEPs;
  SmallVector<FieldSGEPs> Worklist;

  if (SAI.user_empty())
    return PerFieldSGEPs;

  Worklist.push_back({{}, SAI.getAllocationType(), /*NumSharedIndices=*/0});
  for (User *U : SAI.users())
    if (auto *SGEP = dyn_cast<StructuredGEPInst>(U))
      Worklist.back().SGEPs.push_back(SGEP);

  SmallVector<ConstantInt *> IndicesAtLevel;
  while (!Worklist.empty()) {
    FieldSGEPs Cur = Worklist.pop_back_val();

    // When we run out of constant indices we're at the maximum depth we can
    // split accesses at.
    if (llvm::any_of(Cur.SGEPs, [&Cur](const auto *SGEP) {
          return SGEP->getNumIndices() == Cur.NumIndices ||
                 !isa<ConstantInt>(SGEP->getIndexOperand(Cur.NumIndices));
        })) {
      PerFieldSGEPs.push_back(std::move(Cur));
      continue;
    }

    IndicesAtLevel.clear();
    for (StructuredGEPInst *SGEP : Cur.SGEPs)
      IndicesAtLevel.push_back(
          cast<ConstantInt>(SGEP->getIndexOperand(Cur.NumIndices)));

    // We need to operate on the unique indices that are accessed at this level
    // of the GEPs. We sort by integer value rather than pointer identity so
    // that the order we process these later will be deterministic.
    llvm::sort(IndicesAtLevel, [](const auto &LHS, const auto &RHS) {
      return LHS->getZExtValue() < RHS->getZExtValue();
    });
    IndicesAtLevel.erase(llvm::unique(IndicesAtLevel), IndicesAtLevel.end());

    // Enqueue the next level of indices in pre-order.
    for (const ConstantInt *CI : llvm::reverse(IndicesAtLevel)) {
      Worklist.push_back({{},
                          StructuredGEPInst::getTypeAtIndex(Cur.Ty, CI),
                          Cur.NumIndices + 1});
      for (StructuredGEPInst *SGEP : Cur.SGEPs)
        if (SGEP->getIndexOperand(Cur.NumIndices) == CI)
          Worklist.back().SGEPs.push_back(SGEP);
    }
  }

  return PerFieldSGEPs;
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

static void rewriteSGEPChain(IRBuilder<> &B, StructuredAllocaInst *FieldAlloca,
                             StructuredGEPInst *SGEP, unsigned NumIndices) {
  if (SGEP->getNumIndices() == NumIndices) {
    SGEP->replaceAllUsesWith(FieldAlloca);
    SGEP->eraseFromParent();
    return;
  }

  SmallVector<Value *, 4> Indices(
      llvm::drop_begin(SGEP->indices(), NumIndices));
  B.SetInsertPoint(SGEP);
  auto *I = B.CreateStructuredGEP(FieldAlloca->getAllocationType(), FieldAlloca,
                                  Indices, SGEP->getName());
  SGEP->replaceAllUsesWith(I);
  SGEP->eraseFromParent();
}

static bool runOnStructuredAlloca(StructuredAllocaInst &SAI) {
  Type *AllocaTy = SAI.getAllocationType();
  // We only need to do anything with aggregate types.
  if (!isa<ArrayType, StructType, VectorType>(AllocaTy))
    return false;

  if (!isAllocaSplittable(SAI))
    return false;

  SmallVector<FieldSGEPs> PerFieldSGEPs = collectPerFieldSGEPs(SAI);
  SmallVector<LifetimeIntrinsic *> LifetimeIntrinsics =
      collectLifetimeIntrinsicsUsing(SAI);

  IRBuilder B(&SAI);
  for (const FieldSGEPs &Field : PerFieldSGEPs) {
    B.SetInsertPoint(&SAI);
    auto *FieldAlloca =
        cast<StructuredAllocaInst>(B.CreateStructuredAlloca(Field.Ty));

    for (auto II : LifetimeIntrinsics)
      copyLifetimeIntrinsicFor(B, II, FieldAlloca);

    for (StructuredGEPInst *SGEP : Field.SGEPs)
      rewriteSGEPChain(B, FieldAlloca, SGEP, Field.NumIndices);
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
