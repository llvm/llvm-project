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
#include "llvm/ADT/DenseMap.h"
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
  void collectAndSortImplicitBindings(Module &M);
  unsigned findAndReserveFirstUnusedBinding(unsigned DescSet);
  void replaceImplicitBindingCalls(Module &M);

  // A map from descriptor set to a bit vector of used binding numbers.
  std::vector<BitVector> UsedBindings;
  // A list of all implicit binding calls, to be sorted by order ID.
  SmallVector<CallInst *, 16> ImplicitBindingCalls;
};

struct ImplicitBindingInfoCollector
    : public InstVisitor<ImplicitBindingInfoCollector> {
  std::vector<BitVector> &UsedBindings;
  SmallVector<CallInst *, 16> &ImplicitBindingCalls;

  ImplicitBindingInfoCollector(
      std::vector<BitVector> &UsedBindings,
      SmallVector<CallInst *, 16> &ImplicitBindingCalls)
      : UsedBindings(UsedBindings), ImplicitBindingCalls(ImplicitBindingCalls) {
  }

  void visitCallInst(CallInst &CI) {
    if (CI.getIntrinsicID() == Intrinsic::spv_resource_handlefrombinding) {
      // Extract descriptor set and binding.
      // Arguments are: desc set, binding, ...
      const unsigned DescSet =
          cast<ConstantInt>(CI.getArgOperand(0))->getZExtValue();
      const unsigned Binding =
          cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();

      if (UsedBindings.size() <= DescSet) {
        UsedBindings.resize(DescSet + 1);
      }
      if (UsedBindings[DescSet].size() <= Binding) {
        UsedBindings[DescSet].resize(Binding + 1);
      }
      UsedBindings[DescSet].set(Binding);
    } else if (CI.getIntrinsicID() ==
               Intrinsic::spv_resource_handlefromimplicitbinding) {
      ImplicitBindingCalls.push_back(&CI);
    }
  }
};

void SPIRVLegalizeImplicitBinding::collectAndSortImplicitBindings(Module &M) {
  ImplicitBindingInfoCollector InfoCollector(UsedBindings,
                                             ImplicitBindingCalls);
  InfoCollector.visit(M);

  // Sort the collected calls by their order ID (the first argument).
  std::sort(ImplicitBindingCalls.begin(), ImplicitBindingCalls.end(),
            [](const CallInst *A, const CallInst *B) {
              const unsigned OrderA =
                  cast<ConstantInt>(A->getArgOperand(0))->getZExtValue();
              const unsigned OrderB =
                  cast<ConstantInt>(B->getArgOperand(0))->getZExtValue();
              return OrderA < OrderB;
            });
}

unsigned SPIRVLegalizeImplicitBinding::findAndReserveFirstUnusedBinding(
    unsigned DescSet) {
  if (UsedBindings.size() <= DescSet) {
    UsedBindings.resize(DescSet + 1);
  }

  int NewBinding = UsedBindings[DescSet].find_first_unset();
  if (NewBinding == -1) {
    NewBinding = UsedBindings[DescSet].size();
    UsedBindings[DescSet].resize(NewBinding + 1);
  }

  UsedBindings[DescSet].set(NewBinding);
  return NewBinding;
}

void SPIRVLegalizeImplicitBinding::replaceImplicitBindingCalls(Module &M) {
  for (CallInst *OldCI : ImplicitBindingCalls) {
    IRBuilder<> Builder(OldCI);
    const unsigned DescSet =
        cast<ConstantInt>(OldCI->getArgOperand(1))->getZExtValue();
    const unsigned NewBinding = findAndReserveFirstUnusedBinding(DescSet);

    SmallVector<Value *, 8> Args;
    Args.push_back(Builder.getInt32(DescSet));
    Args.push_back(Builder.getInt32(NewBinding));

    // Copy the remaining arguments from the old call.
    for (unsigned i = 2; i < OldCI->arg_size(); ++i) {
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
  collectAndSortImplicitBindings(M);
  if (ImplicitBindingCalls.empty()) {
    return false;
  }

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