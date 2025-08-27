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
  void replaceResourceHandleCall(Module &M, CallInst *OldCI);
  void replaceCounterHandleCall(Module &M, CallInst *OldCI);

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

  void addBinding(uint32_t DescSet, uint32_t Binding) {
    if (UsedBindings.size() <= DescSet) {
      UsedBindings.resize(DescSet + 1);
      UsedBindings[DescSet].resize(64);
    }
    if (UsedBindings[DescSet].size() <= Binding) {
      UsedBindings[DescSet].resize(2 * Binding + 1);
    }
    UsedBindings[DescSet].set(Binding);
  }

  void visitCallInst(CallInst &CI) {
    if (CI.getIntrinsicID() == Intrinsic::spv_resource_handlefrombinding) {
      const uint32_t DescSet =
          cast<ConstantInt>(CI.getArgOperand(0))->getZExtValue();
      const uint32_t Binding =
          cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();
      addBinding(DescSet, Binding);
    } else if (CI.getIntrinsicID() ==
               Intrinsic::spv_resource_handlefromimplicitbinding) {
      ImplicitBindingCalls.push_back(&CI);
    } else if (CI.getIntrinsicID() ==
               Intrinsic::spv_resource_counterhandlefrombinding) {
      const uint32_t DescSet =
          cast<ConstantInt>(CI.getArgOperand(2))->getZExtValue();
      const uint32_t Binding =
          cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();
      addBinding(DescSet, Binding);
    } else if (CI.getIntrinsicID() ==
               Intrinsic::spv_resource_counterhandlefromimplicitbinding) {
      ImplicitBindingCalls.push_back(&CI);
    }
  }
};

static uint32_t getOrderId(const CallInst *CI) {
  switch (CI->getIntrinsicID()) {
  case Intrinsic::spv_resource_handlefromimplicitbinding:
    return cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
  case Intrinsic::spv_resource_counterhandlefromimplicitbinding:
    return cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
  default:
    llvm_unreachable("CallInst is not an implicit binding intrinsic");
  }
}

void SPIRVLegalizeImplicitBinding::collectBindingInfo(Module &M) {
  BindingInfoCollector InfoCollector(UsedBindings, ImplicitBindingCalls);
  InfoCollector.visit(M);

  // Sort the collected calls by their order ID.
  std::sort(ImplicitBindingCalls.begin(), ImplicitBindingCalls.end(),
            [](const CallInst *A, const CallInst *B) {
              return getOrderId(A) < getOrderId(B);
            });
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
  for (CallInst *OldCI : ImplicitBindingCalls) {
    if (OldCI->getIntrinsicID() ==
        Intrinsic::spv_resource_handlefromimplicitbinding) {
      replaceResourceHandleCall(M, OldCI);
    } else {
      assert(OldCI->getIntrinsicID() ==
                 Intrinsic::spv_resource_counterhandlefromimplicitbinding &&
             "Unexpected implicit binding intrinsic");
      replaceCounterHandleCall(M, OldCI);
    }
  }
}

bool SPIRVLegalizeImplicitBinding::runOnModule(Module &M) {
  collectBindingInfo(M);
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

void SPIRVLegalizeImplicitBinding::replaceResourceHandleCall(Module &M,
                                                             CallInst *OldCI) {
  IRBuilder<> Builder(OldCI);
  const uint32_t DescSet =
      cast<ConstantInt>(OldCI->getArgOperand(1))->getZExtValue();
  const uint32_t NewBinding = getAndReserveFirstUnusedBinding(DescSet);

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

void SPIRVLegalizeImplicitBinding::replaceCounterHandleCall(Module &M,
                                                            CallInst *OldCI) {
  IRBuilder<> Builder(OldCI);
  const uint32_t DescSet =
      cast<ConstantInt>(OldCI->getArgOperand(2))->getZExtValue();
  const uint32_t NewBinding = getAndReserveFirstUnusedBinding(DescSet);

  SmallVector<Value *, 8> Args;
  Args.push_back(OldCI->getArgOperand(0));
  Args.push_back(Builder.getInt32(NewBinding));
  Args.push_back(Builder.getInt32(DescSet));

  Type *Tys[] = {OldCI->getType(), OldCI->getArgOperand(0)->getType()};
  Function *NewFunc = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_counterhandlefrombinding, Tys);
  CallInst *NewCI = Builder.CreateCall(NewFunc, Args);
  NewCI->setCallingConv(OldCI->getCallingConv());

  OldCI->replaceAllUsesWith(NewCI);
  OldCI->eraseFromParent();
}
