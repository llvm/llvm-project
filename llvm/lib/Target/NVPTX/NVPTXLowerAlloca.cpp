//===-- NVPTXLowerAlloca.cpp - Make alloca to use local memory =====--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Change the Module's DataLayout to have the local address space for alloca's.
// Change the address space of each alloca to local and add an addrspacecast to
// generic address space. For example,
//
//   %A = alloca i32
//   store i32 0, i32* %A ; emits st.u32
//
// will be transformed to
//
//   %A = alloca i32, addrspace(5)
//   %Generic = addrspacecast i32 addrspace(5)* %A to i32*
//   store i32 0, i32 addrspace(5)* %Generic ; emits st.local.u32
//
// And we will rely on NVPTXInferAddressSpaces to combine the last two
// instructions.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;
using namespace NVPTXAS;

namespace {
class NVPTXLowerAlloca : public ModulePass {
  bool changeDataLayout(Module &M);
  bool lowerFunctionAllocas(Function &F);

public:
  static char ID;
  NVPTXLowerAlloca() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override {
    return "convert address space of alloca'ed memory to local";
  }
};
} // namespace

char NVPTXLowerAlloca::ID = 1;

INITIALIZE_PASS(NVPTXLowerAlloca, "nvptx-lower-alloca", "Lower Alloca", false,
                false)

bool NVPTXLowerAlloca::runOnModule(Module &M) {
  bool Changed = changeDataLayout(M);
  for (auto &F : M)
    Changed |= lowerFunctionAllocas(F);
  return Changed;
}

bool NVPTXLowerAlloca::lowerFunctionAllocas(Function &F) {
  SmallVector<AllocaInst *, 16> Allocas;
  for (auto &I : instructions(F))
    if (auto *Alloca = dyn_cast<AllocaInst>(&I))
      if (Alloca->getAddressSpace() != ADDRESS_SPACE_LOCAL)
        Allocas.push_back(Alloca);

  if (Allocas.empty())
    return false;

  IRBuilder<> Builder(F.getContext());
  for (AllocaInst *Alloca : Allocas) {
    Builder.SetInsertPoint(Alloca);
    auto *NewAlloca =
        Builder.CreateAlloca(Alloca->getAllocatedType(), ADDRESS_SPACE_LOCAL,
                             Alloca->getArraySize(), Alloca->getName());
    NewAlloca->setAlignment(Alloca->getAlign());
    auto *Cast = Builder.CreateAddrSpaceCast(
        NewAlloca,
        PointerType::get(Alloca->getAllocatedType()->getContext(),
                         ADDRESS_SPACE_GENERIC),
        "");
    for (auto &U : llvm::make_early_inc_range(Alloca->uses())) {
      auto *II = dyn_cast<IntrinsicInst>(U.getUser());
      if (!II || !II->isLifetimeStartOrEnd())
        continue;

      Builder.SetInsertPoint(II);
      Builder.CreateIntrinsic(II->getIntrinsicID(), {NewAlloca->getType()},
                              {NewAlloca});
      II->eraseFromParent();
    }
    SmallVector<DbgVariableRecord *, 4> DbgVariableUses;
    findDbgValues(Alloca, DbgVariableUses);
    for (auto *Dbg : DbgVariableUses)
      Dbg->replaceVariableLocationOp(Alloca, NewAlloca);

    Alloca->replaceAllUsesWith(Cast);
    Alloca->eraseFromParent();
  }
  return true;
}

bool NVPTXLowerAlloca::changeDataLayout(Module &M) {
  const auto &DL = M.getDataLayout();
  if (DL.getAllocaAddrSpace() == ADDRESS_SPACE_LOCAL)
    return false;
  auto DLStr = DL.getStringRepresentation();

  auto AddrSpaceStr = "A" + std::to_string(ADDRESS_SPACE_LOCAL);
  if (StringRef(DLStr).contains("A"))
    report_fatal_error("DataLayout should not contain A");
  M.setDataLayout(DLStr.empty() ? AddrSpaceStr : DLStr + "-" + AddrSpaceStr);
  return true;
}

ModulePass *llvm::createNVPTXLowerAllocaPass() {
  return new NVPTXLowerAlloca();
}
