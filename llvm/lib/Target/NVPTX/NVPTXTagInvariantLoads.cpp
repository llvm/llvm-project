//===------ NVPTXTagInvariantLoads.cpp - Tag invariant loads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements invaraint load tagging. It traverses load instructions
// in a function, and determines if each load can be tagged as invariant.
//
// We currently infer invariance for loads from
//  - constant global variables, and
//  - kernel function pointer params that are noalias (i.e. __restrict) and
//    never written to.
//
// TODO: Perform a more powerful invariance analysis (ideally IPO).
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

static bool isInvariantLoad(const Instruction *I, const Value *Ptr,
                            const bool IsKernelFn) {
  // Don't bother with non-global loads
  if (Ptr->getType()->getPointerAddressSpace() != NVPTXAS::ADDRESS_SPACE_GLOBAL)
    return false;

  // If the load is already marked as invariant, we don't need to do anything
  if (I->getMetadata(LLVMContext::MD_invariant_load))
    return false;

  // We use getUnderlyingObjects() here instead of getUnderlyingObject()
  // mainly because the former looks through phi nodes while the latter does
  // not. We need to look through phi nodes to handle pointer induction
  // variables.
  SmallVector<const Value *, 8> Objs;
  getUnderlyingObjects(Ptr, Objs);

  return all_of(Objs, [&](const Value *V) {
    if (const auto *A = dyn_cast<const Argument>(V))
      return IsKernelFn && ((A->onlyReadsMemory() && A->hasNoAliasAttr()) ||
                            isParamGridConstant(*A));
    if (const auto *GV = dyn_cast<const GlobalVariable>(V))
      return GV->isConstant();
    return false;
  });
}

static void markLoadsAsInvariant(Instruction *I) {
  I->setMetadata(LLVMContext::MD_invariant_load,
                 MDNode::get(I->getContext(), {}));
}

static bool tagInvariantLoads(Function &F) {
  const bool IsKernelFn = isKernelFunction(F);

  bool Changed = false;
  for (auto &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I))
      if (isInvariantLoad(LI, LI->getPointerOperand(), IsKernelFn)) {
        markLoadsAsInvariant(LI);
        Changed = true;
      }
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::masked_load &&
          isInvariantLoad(II, II->getOperand(0), IsKernelFn)) {
        markLoadsAsInvariant(II);
        Changed = true;
      }
  }
  return Changed;
}

namespace {

struct NVPTXTagInvariantLoadLegacyPass : public FunctionPass {
  static char ID;

  NVPTXTagInvariantLoadLegacyPass() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

INITIALIZE_PASS(NVPTXTagInvariantLoadLegacyPass, "nvptx-tag-invariant-loads",
                "NVPTX Tag Invariant Loads", false, false)

bool NVPTXTagInvariantLoadLegacyPass::runOnFunction(Function &F) {
  return tagInvariantLoads(F);
}

char NVPTXTagInvariantLoadLegacyPass::ID = 0;

FunctionPass *llvm::createNVPTXTagInvariantLoadsPass() {
  return new NVPTXTagInvariantLoadLegacyPass();
}

PreservedAnalyses NVPTXTagInvariantLoadsPass::run(Function &F,
                                                  FunctionAnalysisManager &) {
  return tagInvariantLoads(F) ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
}
