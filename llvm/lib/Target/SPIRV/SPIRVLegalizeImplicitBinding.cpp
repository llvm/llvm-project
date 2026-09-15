//===- SPIRVLegalizeImplicitBinding.cpp - Legalize implicit bindings ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes the @llvm.spv.resource.handlefromimplicitbinding
// intrinsic by replacing it with a call to
// @llvm.spv.resource.handlefrombinding.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <vector>

using namespace llvm;

namespace {
class SPIRVLegalizeImplicitBindingImpl {
public:
  bool runOnModule(Module &M);

private:
  void collectBindingInfo(Module &M);
  uint32_t getAndReserveFirstUnusedBinding(uint32_t DescSet);
  bool replaceImplicitBindingCalls(Module &M);

  // A map from descriptor set to a bit vector of used binding numbers.
  std::vector<BitVector> UsedBindings;

  // Set to true by collectBindingInfo() if there are any implicit binding
  // declarations in the module.
  bool MayHaveImplicitBindings = false;
};

class SPIRVLegalizeImplicitBindingLegacy : public ModulePass {
public:
  static char ID;
  SPIRVLegalizeImplicitBindingLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "SPIRV Legalize Implicit Binding";
  }
  bool runOnModule(Module &M) override {
    return SPIRVLegalizeImplicitBindingImpl().runOnModule(M);
  }
};

static uint32_t getDescSet(const CallInst *CI) {
  uint32_t DescSetArgIdx;
  switch (CI->getIntrinsicID()) {
  case Intrinsic::spv_resource_handlefromimplicitbinding:
    DescSetArgIdx = 1;
    break;
  case Intrinsic::spv_resource_counterhandlefromimplicitbinding:
    DescSetArgIdx = 2;
    break;
  default:
    llvm_unreachable("CallInst is not an implicit binding intrinsic");
  }
  return cast<ConstantInt>(CI->getArgOperand(DescSetArgIdx))->getZExtValue();
}

// Collect all of the bindings used by llvm.spv.resource.handlefrombinding
// and llvm.spv.resource.counterhandlefrombinding calls. Also check if there
// are any implicit binding calls.
void SPIRVLegalizeImplicitBindingImpl::collectBindingInfo(Module &M) {

  auto addBinding = [&](uint32_t DescSet, uint32_t Binding) {
    if (UsedBindings.size() <= DescSet) {
      UsedBindings.resize(DescSet + 1);
      UsedBindings[DescSet].resize(64);
    }
    if (UsedBindings[DescSet].size() <= Binding) {
      UsedBindings[DescSet].resize(2 * Binding + 1);
    }
    UsedBindings[DescSet].set(Binding);
  };

  auto collectBinding = [&](Function &F, uint32_t ArgDescSetIdx,
                            uint32_t ArgBindingIdx) {
    for (User *U : F.users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        const uint32_t DescSet =
            cast<ConstantInt>(CI->getArgOperand(ArgDescSetIdx))->getZExtValue();
        const uint32_t Binding =
            cast<ConstantInt>(CI->getArgOperand(ArgBindingIdx))->getZExtValue();
        addBinding(DescSet, Binding);
      }
    }
  };

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    switch (F.getIntrinsicID()) {
    case Intrinsic::spv_resource_handlefrombinding:
      collectBinding(F, /*ArgDescSetIdx*/ 0, /*ArgBindingIdx*/ 1);
      break;
    case Intrinsic::spv_resource_counterhandlefrombinding:
      collectBinding(F, /*ArgDescSetIdx*/ 1, /*ArgBindingIdx*/ 2);
      break;
    case Intrinsic::spv_resource_handlefromimplicitbinding:
    case Intrinsic::spv_resource_counterhandlefromimplicitbinding:
      MayHaveImplicitBindings = true;
      break;
    default:
      break;
    }
  }
}

uint32_t SPIRVLegalizeImplicitBindingImpl::getAndReserveFirstUnusedBinding(
    uint32_t DescSet) {
  if (UsedBindings.size() <= DescSet) {
    UsedBindings.resize(DescSet + 1);
    UsedBindings[DescSet].resize(64);
  }

  int NewBinding = UsedBindings[DescSet].find_first_unset();
  if (NewBinding == -1) {
    NewBinding = UsedBindings[DescSet].size();
    UsedBindings[DescSet].resize(2 * NewBinding + 1);
  }

  UsedBindings[DescSet].set(NewBinding);
  return NewBinding;
}

// Replace the implicit binding call with a new call using explicit binding.
static void replaceWithHandleFromBinding(Module &M, CallInst *CI,
                                         uint32_t DescSet, uint32_t Binding,
                                         Value *IndexOp, Value *RangeOp,
                                         Value *Name) {
  assert(CI->getIntrinsicID() ==
             Intrinsic::spv_resource_handlefromimplicitbinding &&
         "unexpected implicit binding intrinsic");
  IRBuilder<> Builder(CI);
  Value *DescSetOp = Builder.getInt32(DescSet);
  Value *BindingOp = Builder.getInt32(Binding);
  Function *NewFunc = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {CI->getType()});
  CallInst *NewCI = Builder.CreateCall(
      NewFunc, {DescSetOp, BindingOp, IndexOp, RangeOp, Name});
  NewCI->setCallingConv(CI->getCallingConv());
  CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
}

// Replace the implicit counter binding call with a new call using explicit
// binding.
static void replaceWithCounterHandleFromBinding(Module &M, CallInst *CI,
                                                uint32_t DescSet,
                                                uint32_t Binding) {
  assert(CI->getIntrinsicID() ==
             Intrinsic::spv_resource_counterhandlefromimplicitbinding &&
         "unexpected implicit binding intrinsic");
  IRBuilder<> Builder(CI);
  Value *DescSetOp = Builder.getInt32(DescSet);
  Value *BindingOp = Builder.getInt32(Binding);
  Value *MainHandle = CI->getArgOperand(0);
  Type *OverloadTys[] = {CI->getType(), MainHandle->getType()};
  Function *NewFunc = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_counterhandlefrombinding, OverloadTys);
  CallInst *NewCI =
      Builder.CreateCall(NewFunc, {MainHandle, DescSetOp, BindingOp});
  NewCI->setCallingConv(CI->getCallingConv());
  CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
}

bool SPIRVLegalizeImplicitBindingImpl::replaceImplicitBindingCalls(Module &M) {
  // Collect all implicit binding calls.
  SmallVector<std::pair<uint32_t, CallInst *>> IBCalls;
  bool Changed = false;
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    uint32_t OrderIdIdx;
    if (F.getIntrinsicID() == Intrinsic::spv_resource_handlefromimplicitbinding)
      OrderIdIdx = 0;
    else if (F.getIntrinsicID() ==
             Intrinsic::spv_resource_counterhandlefromimplicitbinding)
      OrderIdIdx = 1;
    else
      continue;

    for (User *U : F.users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        ConstantInt *OrderId = cast<ConstantInt>(CI->getArgOperand(OrderIdIdx));
        IBCalls.emplace_back(OrderId->getZExtValue(), CI);
      }
    }
  }

  if (IBCalls.empty())
    return false;

  // Sort the collected calls by their order ID.
  llvm::sort(IBCalls, llvm::less_first());

  // Assign bindings based on the order ID. Same order ID gets the same binding.
  // Also make sure that calls with the same order ID have the same descriptor
  // set.
  uint32_t LastOrderId = -1;
  uint32_t LastBinding = -1;
  uint32_t LastDescSet = -1;
  for (auto &[OrderId, CI] : IBCalls) {
    uint32_t Binding;
    uint32_t DescSet = getDescSet(CI);
    if (OrderId == LastOrderId) {
      if (DescSet != LastDescSet)
        report_fatal_error("Implicit binding calls with the same order ID must "
                           "have the same descriptor set");
      Binding = LastBinding;
    } else {
      Binding = getAndReserveFirstUnusedBinding(DescSet);
    }

    // Replace the implicit binding call with an explicit binding call.
    if (CI->getIntrinsicID() ==
        Intrinsic::spv_resource_handlefromimplicitbinding)
      replaceWithHandleFromBinding(M, CI, DescSet, Binding,
                                   CI->getArgOperand(2), CI->getArgOperand(3),
                                   CI->getArgOperand(4));
    else
      replaceWithCounterHandleFromBinding(M, CI, DescSet, Binding);
    Changed = true;

    LastOrderId = OrderId;
    LastBinding = Binding;
    LastDescSet = DescSet;
  }
  return Changed;
}

bool SPIRVLegalizeImplicitBindingImpl::runOnModule(Module &M) {
  collectBindingInfo(M);

  bool Changed = false;
  if (MayHaveImplicitBindings)
    Changed |= replaceImplicitBindingCalls(M);

  return Changed;
}
} // namespace

PreservedAnalyses
SPIRVLegalizeImplicitBindingPass::run(Module &M, ModuleAnalysisManager &AM) {
  return SPIRVLegalizeImplicitBindingImpl().runOnModule(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

char SPIRVLegalizeImplicitBindingLegacy::ID = 0;

INITIALIZE_PASS(SPIRVLegalizeImplicitBindingLegacy,
                "legalize-spirv-implicit-binding",
                "Legalize SPIR-V implicit bindings", false, false)

ModulePass *llvm::createSPIRVLegalizeImplicitBindingPass() {
  return new SPIRVLegalizeImplicitBindingLegacy();
}
