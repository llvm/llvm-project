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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <vector>

using namespace llvm;

namespace {
class SPIRVLegalizeImplicitBinding : public ModulePass {
public:
  static char ID;
  SPIRVLegalizeImplicitBinding() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

private:
  void collectBindingInfo(Module &M);
  uint32_t getAndReserveFirstUnusedBinding(uint32_t DescSet);
  void replaceImplicitBindingCalls(Module &M);
  void verifyUniqueOrderIdPerResource(SmallVectorImpl<CallInst *> &Calls);

  // A map from descriptor set to a bit vector of used binding numbers.
  std::vector<BitVector> UsedBindings;
  // A list of all implicit binding calls, to be sorted by order ID.
  SmallVector<CallInst *, 16> ImplicitBindingCalls;
};

struct BindingInfoCollector : public InstVisitor<BindingInfoCollector> {
  std::vector<BitVector> &UsedBindings;
  SmallVector<CallInst *, 16> &ImplicitBindingCalls;

  BindingInfoCollector(std::vector<BitVector> &UsedBindings,
                       SmallVector<CallInst *, 16> &ImplicitBindingCalls)
      : UsedBindings(UsedBindings), ImplicitBindingCalls(ImplicitBindingCalls) {
  }

  void visitCallInst(CallInst &CI) {
    if (CI.getIntrinsicID() == Intrinsic::spv_resource_handlefrombinding) {
      const uint32_t DescSet =
          cast<ConstantInt>(CI.getArgOperand(0))->getZExtValue();
      const uint32_t Binding =
          cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();

      if (UsedBindings.size() <= DescSet) {
        UsedBindings.resize(DescSet + 1);
        UsedBindings[DescSet].resize(64);
      }
      if (UsedBindings[DescSet].size() <= Binding) {
        UsedBindings[DescSet].resize(2 * Binding + 1);
      }
      UsedBindings[DescSet].set(Binding);
    } else if (CI.getIntrinsicID() ==
               Intrinsic::spv_resource_handlefromimplicitbinding) {
      ImplicitBindingCalls.push_back(&CI);
    }
  }
};

void SPIRVLegalizeImplicitBinding::collectBindingInfo(Module &M) {
  BindingInfoCollector InfoCollector(UsedBindings, ImplicitBindingCalls);
  InfoCollector.visit(M);

  // Sort the collected calls by their order ID.
  std::sort(
      ImplicitBindingCalls.begin(), ImplicitBindingCalls.end(),
      [](const CallInst *A, const CallInst *B) {
        const uint32_t OrderIdArgIdx = 0;
        const uint32_t OrderA =
            cast<ConstantInt>(A->getArgOperand(OrderIdArgIdx))->getZExtValue();
        const uint32_t OrderB =
            cast<ConstantInt>(B->getArgOperand(OrderIdArgIdx))->getZExtValue();
        return OrderA < OrderB;
      });
}

void SPIRVLegalizeImplicitBinding::verifyUniqueOrderIdPerResource(
    SmallVectorImpl<CallInst *> &Calls) {
  // Check that the order Id is unique per resource.
  for (uint32_t i = 1; i < Calls.size(); ++i) {
    const uint32_t OrderIdArgIdx = 0;
    const uint32_t DescSetArgIdx = 1;
    const uint32_t OrderA =
        cast<ConstantInt>(Calls[i - 1]->getArgOperand(OrderIdArgIdx))
            ->getZExtValue();
    const uint32_t OrderB =
        cast<ConstantInt>(Calls[i]->getArgOperand(OrderIdArgIdx))
            ->getZExtValue();
    if (OrderA == OrderB) {
      const uint32_t DescSetA =
          cast<ConstantInt>(Calls[i - 1]->getArgOperand(DescSetArgIdx))
              ->getZExtValue();
      const uint32_t DescSetB =
          cast<ConstantInt>(Calls[i]->getArgOperand(DescSetArgIdx))
              ->getZExtValue();
      if (DescSetA != DescSetB) {
        report_fatal_error("Implicit binding calls with the same order ID must "
                           "have the same descriptor set");
      }
    }
  }
}

uint32_t SPIRVLegalizeImplicitBinding::getAndReserveFirstUnusedBinding(
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

void SPIRVLegalizeImplicitBinding::replaceImplicitBindingCalls(Module &M) {
  uint32_t lastOrderId = -1;
  uint32_t lastBindingNumber = -1;

  for (CallInst *OldCI : ImplicitBindingCalls) {
    IRBuilder<> Builder(OldCI);
    const uint32_t OrderId =
        cast<ConstantInt>(OldCI->getArgOperand(0))->getZExtValue();
    const uint32_t DescSet =
        cast<ConstantInt>(OldCI->getArgOperand(1))->getZExtValue();

    // Reuse an existing binding for this order ID, if one was already assigned.
    // Otherwise, assign a new binding.
    const uint32_t NewBinding = (lastOrderId == OrderId)
                                    ? lastBindingNumber
                                    : getAndReserveFirstUnusedBinding(DescSet);
    lastOrderId = OrderId;
    lastBindingNumber = NewBinding;

    SmallVector<Value *, 8> Args;
    Args.push_back(Builder.getInt32(DescSet));
    Args.push_back(Builder.getInt32(NewBinding));

    // Copy the remaining arguments from the old call.
    for (uint32_t i = 2; i < OldCI->arg_size(); ++i) {
      Args.push_back(OldCI->getArgOperand(i));
    }

    Function *NewFunc = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_handlefrombinding, OldCI->getType());
    CallInst *NewCI = Builder.CreateCall(NewFunc, Args);
    NewCI->setCallingConv(OldCI->getCallingConv());

    OldCI->replaceAllUsesWith(NewCI);
    OldCI->eraseFromParent();
  }
}

bool SPIRVLegalizeImplicitBinding::runOnModule(Module &M) {
  collectBindingInfo(M);
  if (ImplicitBindingCalls.empty()) {
    return false;
  }
  verifyUniqueOrderIdPerResource(ImplicitBindingCalls);

  replaceImplicitBindingCalls(M);
  return true;
}
} // namespace

char SPIRVLegalizeImplicitBinding::ID = 0;

INITIALIZE_PASS(SPIRVLegalizeImplicitBinding, "legalize-spirv-implicit-binding",
                "Legalize SPIR-V implicit bindings", false, false)

ModulePass *llvm::createSPIRVLegalizeImplicitBindingPass() {
  return new SPIRVLegalizeImplicitBinding();
}