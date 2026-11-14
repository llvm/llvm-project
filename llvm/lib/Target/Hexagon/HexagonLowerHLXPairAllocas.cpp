//===- HexagonLowerHLXPairAllocas.cpp - Lower HLX pair allocas ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HLX pair registers are interleaved, so a plain load or store of an HLX
// VectorPair is not two HLX vector memory accesses. Clang only creates pair
// memory for compiler-generated allocas (e.g. -O0 locals); rewrite the
// accesses to those allocas that survive optimization into the LL16b.tmp and
// SS16b.new intrinsics, which define the in-memory layout of a pair.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-lower-hlx-pair-allocas"

namespace {

class HexagonLowerHLXPairAllocas : public FunctionPass {
public:
  static char ID;

  HexagonLowerHLXPairAllocas() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Hexagon Lower HLX Pair Allocas";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
    FunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char HexagonLowerHLXPairAllocas::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonLowerHLXPairAllocas, DEBUG_TYPE,
                      "Hexagon Lower HLX Pair Allocas", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(HexagonLowerHLXPairAllocas, DEBUG_TYPE,
                    "Hexagon Lower HLX Pair Allocas", false, false)

// Return the alloca accessed by I if I is a simple, aligned, temporal load or
// store of PairTy directly from a PairTy alloca.
static AllocaInst *getPairAlloca(Instruction &I, Type *PairTy,
                                 unsigned VectorLength) {
  if (!isa<LoadInst, StoreInst>(I) || getLoadStoreType(&I) != PairTy ||
      I.isVolatile() || I.isAtomic() ||
      I.getMetadata(LLVMContext::MD_nontemporal) ||
      getLoadStoreAlignment(&I).value() < VectorLength)
    return nullptr;
  auto *AI = dyn_cast<AllocaInst>(
      getLoadStorePointerOperand(&I)->stripPointerCastsSameRepresentation());
  if (!AI || AI->getAllocatedType() != PairTy ||
      AI->getAlign().value() < VectorLength)
    return nullptr;
  return AI;
}

bool HexagonLowerHLXPairAllocas::runOnFunction(Function &F) {
  auto &TM = getAnalysis<TargetPassConfig>().getTM<HexagonTargetMachine>();
  const HexagonSubtarget *HST = TM.getSubtargetImpl(F);
  if (!HST->useHLXOps())
    return false;

  unsigned VectorLength = HST->getHLXVectorLength();
  // Only the i32 VectorPair type is lowered; other HLX types are rejected or
  // handled by ISel.
  Type *PairTy =
      FixedVectorType::get(Type::getInt32Ty(F.getContext()), VectorLength / 2);
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    AllocaInst *AI = getPairAlloca(I, PairTy, VectorLength);
    if (!AI)
      continue;

    IRBuilder<> Builder(&I);
    CallInst *Call;
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      Call = Builder.CreateCall(
          Intrinsic::getOrInsertDeclaration(F.getParent(),
                                            Intrinsic::hexagon_LX_SS16b_new),
          {AI, SI->getValueOperand()});
    } else {
      Call = Builder.CreateCall(
          Intrinsic::getOrInsertDeclaration(F.getParent(),
                                            Intrinsic::hexagon_LX_LL16b_tmp),
          AI);
      I.replaceAllUsesWith(Call);
    }
    Call->setAAMetadata(I.getAAMetadata());
    Call->copyMetadata(I, {LLVMContext::MD_access_group});
    I.eraseFromParent();
    Changed = true;
  }
  return Changed;
}

FunctionPass *llvm::createHexagonLowerHLXPairAllocas() {
  return new HexagonLowerHLXPairAllocas();
}
