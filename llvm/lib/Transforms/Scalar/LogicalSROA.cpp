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
    auto II = dyn_cast<IntrinsicInst>(U);
    if (II && isLifetimeIntrinsic(II->getIntrinsicID()))
      Output.push_back(cast<LifetimeIntrinsic>(II));
  }

  return Output;
}

// Returns a vector with one element for each field of the struct allocated by
// SAI. Each element is a vector of SGEP instruction referencing this field.
//
// If any user of SAI is not an SGEP, or an SGEP referencing the whole struct,
// this function returns an empty array. This function ignores lifetime
// intrinsics.
static SmallVector<SmallVector<StructuredGEPInst *>>
collectPerFieldSGEP(StructuredAllocaInst &SAI) {
  StructType *ST = cast<StructType>(SAI.getAllocationType());
  SmallVector<SmallVector<StructuredGEPInst *>> Output(ST->getNumElements());

  for (User *U : SAI.users()) {
    auto II = dyn_cast<IntrinsicInst>(U);
    if (II && II->isLifetimeStartOrEnd())
      continue;

    auto SGEP = dyn_cast<StructuredGEPInst>(U);
    if (!SGEP)
      return {};

    // If the SGEP has no indices, this means we have a pointer on the whole
    // struct. For now, we bail out: if it was not used, it would be DCE'd, so
    // there is probably a reference to the whole struct somewhere.
    if (SGEP->getNumIndices() == 0)
      return {};

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

  auto PerFieldSGEP = collectPerFieldSGEP(SAI);
  if (PerFieldSGEP.empty())
    return false;

  auto LifetimeIntrinsics = collectLifetimeIntrinsicsUsing(SAI);
  IRBuilder B(&SAI);
  for (const auto &[FieldIndex, Users] : llvm::enumerate(PerFieldSGEP)) {
    if (Users.empty())
      continue;

    B.SetInsertPoint(&SAI);
    StructuredAllocaInst *FieldAlloca = cast<StructuredAllocaInst>(
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
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (StructuredAllocaInst *SAI = dyn_cast<StructuredAllocaInst>(&I))
        Worklist.push_back(SAI);
    }
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

namespace {

/// A legacy pass for the legacy pass manager that wraps the LogicalSROA pass.
class LogicalSROALegacyPass : public FunctionPass {
public:
  static char ID;

  LogicalSROALegacyPass() : FunctionPass(ID) {
    initializeLogicalSROALegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    return runLogicalSROA(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  StringRef getPassName() const override { return "LogicalSROA"; }
};

} // end anonymous namespace

char LogicalSROALegacyPass::ID = 0;

FunctionPass *llvm::createLogicalSROAPass() {
  return new LogicalSROALegacyPass();
}

INITIALIZE_PASS_BEGIN(LogicalSROALegacyPass, "logical-sroa",
                      "Logical Scalar Replacement Of Aggregates", false, false)
INITIALIZE_PASS_END(LogicalSROALegacyPass, "logical-sroa",
                    "Logical Scalar Replacement Of Aggregates", false, false)
