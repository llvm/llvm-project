//===-- NVPTXLowerAlloca.cpp - Make alloca to use local memory =====--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace each generic alloca with an equivalent alloca in the local address
// space, followed by an addrspacecast back to generic for its users. For
// example,
//
//   %A = alloca i32
//   store i32 0, ptr %A ; emits st.u32
//
// is transformed to
//
//   %A = alloca i32, addrspace(5)
//   %A.generic = addrspacecast ptr addrspace(5) %A to ptr
//   store i32 0, ptr %A.generic
//
// This gives the alloca a local frame index, which stack lowering addresses
// through the local frame pointer (%SPL). When NVPTXInferAddressSpaces runs
// after this pass, it propagates the local address space into the users and
// folds the cast away where possible (so the store above becomes st.local.u32).
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class NVPTXLowerAlloca : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerAlloca() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "convert address space of alloca'ed memory to local";
  }
};
} // namespace

char NVPTXLowerAlloca::ID = 0;

INITIALIZE_PASS(NVPTXLowerAlloca, "nvptx-lower-alloca", "Lower Alloca", false,
                false)

// =============================================================================
// Main function for this pass.
// =============================================================================
bool NVPTXLowerAlloca::runOnFunction(Function &F) {
  // Mandatory lowering: later stack lowering relies on local allocas, so run
  // even for optnone functions (skipFunction is intentionally not called).
  SmallVector<AllocaInst *, 8> GenericAllocas;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *AI = dyn_cast<AllocaInst>(&I);
          AI && AI->getAddressSpace() == ADDRESS_SPACE_GENERIC)
        GenericAllocas.push_back(AI);

  for (AllocaInst *AI : GenericAllocas) {
    // Create an equivalent alloca in the local address space.
    auto *LocalAlloca = new AllocaInst(AI->getAllocatedType(),
                                       ADDRESS_SPACE_LOCAL, AI->getArraySize(),
                                       AI->getAlign(), "", AI->getIterator());
    LocalAlloca->setDebugLoc(AI->getDebugLoc());
    LocalAlloca->copyMetadata(*AI);
    LocalAlloca->setUsedWithInAlloca(AI->isUsedWithInAlloca());
    LocalAlloca->setSwiftError(AI->isSwiftError());

    // Debug records and lifetime markers have to reference the alloca itself,
    // not a cast of it, so retarget them to the local alloca before rewriting
    // the remaining users through a generic addrspacecast below:
    //   - the verifier requires an alloca operand for lifetime markers, and
    //   - pointing debug records at the alloca keeps the variable described by
    //     its (stable) stack slot rather than the cvta.local result.
    SmallVector<DbgVariableRecord *, 2> DbgUsers;
    findDbgUsers(AI, DbgUsers);
    for (DbgVariableRecord *DVR : DbgUsers)
      DVR->replaceVariableLocationOp(AI, LocalAlloca);

    for (Use &U : llvm::make_early_inc_range(AI->uses())) {
      auto *II = dyn_cast<IntrinsicInst>(U.getUser());
      if (!II || !isLifetimeIntrinsic(II->getIntrinsicID()))
        continue;
      U.set(LocalAlloca);
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          II->getModule(), II->getIntrinsicID(), {LocalAlloca->getType()});
      II->setCalledFunction(Decl);
    }

    // Everything else can go through a single generic addrspacecast.
    // replaceAllUsesWith leaves the (already retargeted) lifetime markers and
    // debug records untouched. NVPTXInferAddressSpaces folds the cast into the
    // users that can operate on local memory directly.
    auto *GenericPtr = new AddrSpaceCastInst(LocalAlloca, AI->getType(), "",
                                             AI->getIterator());
    GenericPtr->setDebugLoc(AI->getDebugLoc());
    AI->replaceAllUsesWith(GenericPtr);
    LocalAlloca->takeName(AI);
    AI->eraseFromParent();
  }

  return !GenericAllocas.empty();
}

FunctionPass *llvm::createNVPTXLowerAllocaPass() {
  return new NVPTXLowerAlloca();
}
