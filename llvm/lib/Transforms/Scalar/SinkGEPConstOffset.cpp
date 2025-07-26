//===- SinkGEPConstOffset.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SinkGEPConstOffset.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>
#include <string>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

/// A pass that tries to sink const offset in GEP chain to tail.
/// It is a FunctionPass because searching for the constant offset may inspect
/// other basic blocks.
class SinkGEPConstOffsetLegacyPass : public FunctionPass {
public:
  static char ID;

  SinkGEPConstOffsetLegacyPass() : FunctionPass(ID) {
    initializeSinkGEPConstOffsetLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

char SinkGEPConstOffsetLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    SinkGEPConstOffsetLegacyPass, "sink-gep-const-offset",
    "Sink const offsets down the GEP chain to the tail for reduction of "
    "register usage", false, false)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(
    SinkGEPConstOffsetLegacyPass, "sink-gep-const-offset",
    "Sink const offsets down the GEP chain to the tail for reduction of "
    "register usage", false, false)

FunctionPass *llvm::createSinkGEPConstOffsetPass() {
  return new SinkGEPConstOffsetLegacyPass();
}

/// The purpose of this function is to sink the constant offsets in the base
/// GEP to current GEP.
///
/// A simple example is given:
///
/// %gep0 = getelementptr half, ptr addrspace(3) %ptr, i32 512
/// %gep1 = getelementptr half, ptr addrspace(3) %gep0, i32 %ofst0
/// %data = load half, ptr addrspace(3) %gep1, align 2
/// ==>
/// %gep0 = getelementptr half, ptr addrspace(3) %ptr, i32 %ofst0
/// %gep1 = getelementptr half, ptr addrspace(3) %gep0, i32 512
/// %data = load half, ptr addrspace(3) %gep1, align 2
static bool sinkGEPConstantOffset(Value *Ptr, const DataLayout *DL) {
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return false;

  if (GEP->getNumIndices() != 1)
    return false;

  GetElementPtrInst *BaseGEP =
      dyn_cast<GetElementPtrInst>(GEP->getPointerOperand());
  if (!BaseGEP)
    return false;

  if (BaseGEP->getNumIndices() != 1)
    return false;

  Value *Idx = GEP->getOperand(1);
  Value *BaseIdx = BaseGEP->getOperand(1);

  ConstantInt *BaseIdxAsCI = dyn_cast<ConstantInt>(BaseIdx);
  if (!BaseIdxAsCI)
    return false;

  Type *ResTy = GEP->getResultElementType();
  Type *BaseResTy = BaseGEP->getResultElementType();

  ConstantInt *IdxAsCI = dyn_cast<ConstantInt>(Idx);
  if (IdxAsCI) {
    // %gep0 = getelementptr half, ptr addrspace(3) %ptr, i32 8
    // %gep1 = getelementptr half, ptr addrspace(3) %gep0, i32 4
    // as:
    // %gep1 = getelementptr half, ptr addrspace(3) %ptr, i32 12
    Type *NewResTy = nullptr;
    int64_t NewIdxValue = 0;
    if (ResTy == BaseResTy) {
      NewResTy = ResTy;
      NewIdxValue = BaseIdxAsCI->getSExtValue() + IdxAsCI->getSExtValue();
    } else {
      NewResTy = Type::getInt8Ty(GEP->getContext());
      NewIdxValue = (BaseIdxAsCI->getSExtValue() *
                     DL->getTypeAllocSize(BaseResTy)) +
                    (IdxAsCI->getSExtValue() *
                     DL->getTypeAllocSize(ResTy));
    }
    assert(NewResTy);
    Type *NewIdxType = (Idx->getType()->getPrimitiveSizeInBits() >
                      BaseIdx->getType()->getPrimitiveSizeInBits())
                         ? Idx->getType() : BaseIdx->getType();
    Constant *NewIdx = ConstantInt::get(NewIdxType, NewIdxValue);
    auto *NewGEP = GetElementPtrInst::Create(
        NewResTy, BaseGEP->getPointerOperand(), NewIdx);
    NewGEP->setIsInBounds(GEP->isInBounds());
    NewGEP->insertBefore(GEP->getIterator());
    NewGEP->takeName(GEP);

    GEP->replaceAllUsesWith(NewGEP);
    RecursivelyDeleteTriviallyDeadInstructions(GEP);

    return true;
  }

  // %gep0 = getelementptr half, ptr addrspace(3) %ptr, i32 8
  // %gep1 = getelementptr half, ptr addrspace(3) %gep0, i32 %idx
  // as:
  // %gepx0 = getelementptr half, ptr addrspace(3) %ptr, i32 %idx
  // %gepx1 = getelementptr half, ptr addrspace(3) %gepx0, i32 8
  auto *GEPX0 =
      GetElementPtrInst::Create(ResTy, BaseGEP->getPointerOperand(), Idx);
  GEPX0->insertBefore(GEP->getIterator());
  auto *GEPX1 = GetElementPtrInst::Create(BaseResTy, GEPX0, BaseIdx);
  GEPX1->setIsInBounds(GEP->isInBounds());
  GEPX1->insertBefore(GEP->getIterator());
  GEPX1->takeName(GEP);

  GEP->replaceAllUsesWith(GEPX1);
  RecursivelyDeleteTriviallyDeadInstructions(GEP);

  return true;
}

static bool sinkGEPConstantOffset(Function &F) {
  const DataLayout *DL = &F.getDataLayout();
  ReversePostOrderTraversal<Function*> RPOT(&F);
  bool Changed = false;
  for (auto *BB : RPOT)
    for (Instruction &I : llvm::make_early_inc_range(*BB))
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I))
        Changed |= sinkGEPConstantOffset(GEP, DL);

  return Changed;
}

bool SinkGEPConstOffsetLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  return sinkGEPConstantOffset(F);
}

void SinkGEPConstOffsetPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<SinkGEPConstOffsetPass> *>(this)
      ->printPipeline(OS, MapClassName2PassName);
}

PreservedAnalyses
SinkGEPConstOffsetPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!sinkGEPConstantOffset(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
