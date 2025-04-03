//===-- SPIRVLegalizeAddrspaceCast.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace llvm {
void initializeSPIRVLegalizeAddrspaceCastPass(PassRegistry &);
}

class SPIRVLegalizeAddrspaceCast : public FunctionPass {

public:
  SPIRVLegalizeAddrspaceCast(SPIRVTargetMachine *TM)
      : FunctionPass(ID), TM(TM) {
    initializeSPIRVLegalizeAddrspaceCastPass(*PassRegistry::getPassRegistry());
  };

  void gatherAddrspaceCast(Function &F) {
    WorkList.clear();
    std::vector<User *> ToVisit;
    for (auto &BB : F)
      for (auto &I : BB)
        ToVisit.push_back(&I);

    std::unordered_set<User *> Visited;
    while (ToVisit.size() > 0) {
      User *I = ToVisit.back();
      ToVisit.pop_back();
      if (Visited.count(I) != 0)
        continue;
      Visited.insert(I);

      if (AddrSpaceCastInst *AI = dyn_cast<AddrSpaceCastInst>(I))
        WorkList.insert(AI);
      else if (auto *AO = dyn_cast<AddrSpaceCastOperator>(I))
        WorkList.insert(AO);

      for (auto &O : I->operands())
        if (User *U = dyn_cast<User>(&O))
          ToVisit.push_back(U);
    }
  }

  void propagateAddrspace(User *U) {
    if (!U->getType()->isPointerTy())
      return;

    if (AddrSpaceCastOperator *AO = dyn_cast<AddrSpaceCastOperator>(U)) {
      for (auto &Use : AO->uses())
        WorkList.insert(Use.getUser());

      AO->mutateType(AO->getPointerOperand()->getType());
      AO->replaceAllUsesWith(AO->getPointerOperand());
      DeadUsers.insert(AO);
      return;
    }

    if (AddrSpaceCastInst *AC = dyn_cast<AddrSpaceCastInst>(U)) {
      for (auto &Use : AC->uses())
        WorkList.insert(Use.getUser());

      AC->mutateType(AC->getPointerOperand()->getType());
      AC->replaceAllUsesWith(AC->getPointerOperand());
      return;
    }

    PointerType *NewType = nullptr;
    for (Use &U : U->operands()) {
      PointerType *PT = dyn_cast<PointerType>(U.get()->getType());
      if (!PT)
        continue;

      if (NewType == nullptr)
        NewType = PT;
      else {
        // We could imagine a function calls taking 2 pointers to distinct
        // address spaces which returns a pointer. But we want to run this
        // pass after inlining, so we'll assume this doesn't happen.
        assert(NewType->getAddressSpace() == PT->getAddressSpace());
      }
    }

    assert(NewType != nullptr);
    U->mutateType(NewType);
  }

  virtual bool runOnFunction(Function &F) override {
    const SPIRVSubtarget &ST = TM->getSubtarget<SPIRVSubtarget>(F);
    GR = ST.getSPIRVGlobalRegistry();

    DeadUsers.clear();
    gatherAddrspaceCast(F);

    while (WorkList.size() > 0) {
      User *U = *WorkList.begin();
      WorkList.erase(U);
      propagateAddrspace(U);
    }

    for (User *U : DeadUsers) {
      if (Instruction *I = dyn_cast<Instruction>(U))
        I->eraseFromParent();
    }
    return DeadUsers.size() != 0;
  }

private:
  SPIRVTargetMachine *TM = nullptr;
  SPIRVGlobalRegistry *GR = nullptr;
  std::unordered_set<User *> WorkList;
  std::unordered_set<User *> DeadUsers;

public:
  static char ID;
};

char SPIRVLegalizeAddrspaceCast::ID = 0;
INITIALIZE_PASS(SPIRVLegalizeAddrspaceCast, "spirv-legalize-addrspacecast",
                "SPIRV legalize addrspacecast", false, false)

FunctionPass *
llvm::createSPIRVLegalizeAddrspaceCastPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizeAddrspaceCast(TM);
}
