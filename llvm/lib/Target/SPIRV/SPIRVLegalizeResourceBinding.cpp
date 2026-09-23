//===- SPIRVLegalizeResourceBinding.cpp - Legalize resource bindings ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes the @llvm.spv.resource.handlefromimplicitbinding
// and @llvm.spv.resource.handlefromheap intrinsics by replacing them with a
// call to @llvm.spv.resource.handlefrombinding.
// It also replaces any @llvm.spv.resource.counterhandlefromimplicitbinding and
// @llvm.spv.resource.counterhandlefromheap intrinsics with calls to
// @llvm.spv.resource.counterhandlefrombinding.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

namespace {
class SPIRVLegalizeResourceBindingImpl {
public:
  bool runOnModule(Module &M);

private:
  void collectBindingInfo(Module &M);
  uint32_t getAndReserveFirstUnusedBinding(uint32_t DescSet);
  bool replaceImplicitBindingCalls(Module &M);
  bool replaceHeapBindingCalls(Module &M);

  // A map from descriptor set to a bit vector of used binding numbers.
  std::vector<BitVector> UsedBindings;

  // Set to true by collectBindingInfo() if there are possibly any implicit
  // binding or heap binding calls in the module (if the module contains a
  // declaration of implicit binding or heap intrinsic).
  bool MayHaveImplicitBindings = false;
  bool MayHaveHeapBindings = false;
};

class SPIRVLegalizeResourceBindingLegacy : public ModulePass {
public:
  static char ID;
  SPIRVLegalizeResourceBindingLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "SPIRV Legalize Resource Binding";
  }
  bool runOnModule(Module &M) override {
    return SPIRVLegalizeResourceBindingImpl().runOnModule(M);
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
void SPIRVLegalizeResourceBindingImpl::collectBindingInfo(Module &M) {

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
    case Intrinsic::spv_resource_handlefromheap:
    case Intrinsic::spv_resource_counterhandlefromheap:
      MayHaveHeapBindings = true;
      break;
    default:
      break;
    }
  }
}

uint32_t SPIRVLegalizeResourceBindingImpl::getAndReserveFirstUnusedBinding(
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
                                         Value *RangeOp, Value *IndexOp,
                                         Value *Name) {
  assert((CI->getIntrinsicID() ==
              Intrinsic::spv_resource_handlefromimplicitbinding ||
          CI->getIntrinsicID() == Intrinsic::spv_resource_handlefromheap) &&
         "unexpected binding intrinsic");
  IRBuilder<> Builder(CI);
  Value *DescSetOp = Builder.getInt32(DescSet);
  Value *BindingOp = Builder.getInt32(Binding);
  Function *NewFunc = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {CI->getType()});
  CallInst *NewCI = Builder.CreateCall(
      NewFunc, {DescSetOp, BindingOp, RangeOp, IndexOp, Name});
  NewCI->setCallingConv(CI->getCallingConv());
  CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
}

// Replace the implicit counter binding call with a new call using explicit
// binding.
static void replaceWithCounterHandleFromBinding(Module &M, CallInst *CI,
                                                uint32_t DescSet,
                                                uint32_t Binding) {
  assert(
      (CI->getIntrinsicID() ==
           Intrinsic::spv_resource_counterhandlefromimplicitbinding ||
       CI->getIntrinsicID() == Intrinsic::spv_resource_counterhandlefromheap) &&
      "unexpected binding intrinsic");
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

bool SPIRVLegalizeResourceBindingImpl::replaceImplicitBindingCalls(Module &M) {
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

bool moduleContainsConstantString(Module &M, StringRef Str) {
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasInitializer())
      continue;
    if (ConstantDataArray *CDA =
            dyn_cast<ConstantDataArray>(GV.getInitializer())) {
      if (CDA->isString() && CDA->getAsCString() == Str)
        return true;
    }
  }
  return false;
}

// Creates unique global variable for the heap name string, making
// sure that each heap name string is unique within the module.
GlobalVariable *createHeapNameString(Module &M, StringRef Name) {
  SmallString<32> GlobalStringName(Name);
  uint32_t HeapNameLen = Name.size();
  StringRef HeapName;
  for (unsigned Suffix = 1;; ++Suffix) {
    GlobalStringName.append(".str");
    if (!M.getNamedValue(GlobalStringName)) {
      // Make sure the module does not already have a constant
      // string with this value.
      HeapName = GlobalStringName.substr(0, HeapNameLen);
      if (!moduleContainsConstantString(M, HeapName))
        break;
    }
    GlobalStringName.resize(Name.size());
    raw_svector_ostream(GlobalStringName) << '.' << Suffix;
    HeapNameLen = GlobalStringName.size();
  }
  Constant *Init = ConstantDataArray::getString(M.getContext(), HeapName);
  GlobalVariable *HeapNameGV = new GlobalVariable(
      M, Init->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
      Init, GlobalStringName, /*InsertBefore=*/nullptr,
      GlobalVariable::NotThreadLocal, /*AddressSpace=*/0);
  HeapNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  HeapNameGV->setAlignment(Align(1));
  return HeapNameGV;
}

// The SPIR-V backend represents dynamic resources as unbounded resource arrays.
// This function scans the module for calls to
// `llvm.spv.resource.handlefromheap` and groups them according to whether they
// create CBV/SRV/UAV resources or samplers. It also collects calls to
// `llvm.spv.resource.counterhandlefromheap` intrinsics that form a third group.
//
// The function assigns the first available binding to each non-empty heap group
// in this order: CBV/SRV/UAV resources, samplers, and counters. For each group,
// it will replace the heap intrinsic calls with their explicit
// `handlefrombinding` equivalents using the assigned binding.
//
// The function does not actually create the unbounded resource-array globals
// itself. It only assigns a unique name that is shared by all resources
// belonging to the same heap type; the existing `SPIRVInstructionSelector` will
// create the globals.
//
// Because the CBV/SRV/UAV group can contain resources of different types, these
// resources must be represented by separate arrays, one for each unique
// resource type. All of these arrays will use the same binding and therefore
// overlap.
bool SPIRVLegalizeResourceBindingImpl::replaceHeapBindingCalls(Module &M) {
  // First we collect all used heap binding declarations and group them based on
  // their kind.
  SmallVector<Function *, 8> CbvSrvUavs;
  SmallVector<Function *, 8> Samplers;
  SmallVector<Function *, 8> Counters;
  bool Changed = false;

  for (Function &F : M) {
    if (!F.isDeclaration() || F.user_empty())
      continue;

    if (F.getIntrinsicID() == Intrinsic::spv_resource_handlefromheap) {
      TargetExtType *ResType = cast<TargetExtType>(F.getReturnType());
      if (ResType->getName() == "spirv.Sampler")
        Samplers.emplace_back(&F);
      else
        CbvSrvUavs.emplace_back(&F);
    } else if (F.getIntrinsicID() ==
               Intrinsic::spv_resource_counterhandlefromheap) {
      Counters.emplace_back(&F);
    } else
      continue;
  }

  if (CbvSrvUavs.empty() && Samplers.empty() && Counters.empty())
    return false;

  // Heap resources are always mapped to descriptor set 0 as an unbounded
  // runtime array.
  constexpr uint32_t DescSet = 0;
  Value *Zero =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 0);

  if (!CbvSrvUavs.empty()) {
    // For CBV/UAV/SRV resources we need to create a different
    // ResourceDescriptorHeap name for each unique resource type. They will all
    // share the same binding and will overlap.
    uint32_t Binding = getAndReserveFirstUnusedBinding(DescSet);
    SmallDenseMap<TargetExtType *, GlobalVariable *> ResourceDescriptorHeaps;
    for (Function *F : CbvSrvUavs) {
      TargetExtType *ResType = cast<TargetExtType>(F->getReturnType());
      GlobalVariable *HeapNameGV = nullptr;
      auto It = ResourceDescriptorHeaps.find(ResType);
      if (It == ResourceDescriptorHeaps.end()) {
        HeapNameGV = createHeapNameString(M, "ResourceDescriptorHeap");
        [[maybe_unused]] auto [InsertedIt, Inserted] =
            ResourceDescriptorHeaps.try_emplace(ResType, HeapNameGV);
        assert(Inserted && "resource heap name already exists");
      } else {
        HeapNameGV = It->second;
      }

      for (User *U : make_early_inc_range(F->users())) {
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          Value *HeapIdx = CI->getArgOperand(0);
          replaceWithHandleFromBinding(M, CI, DescSet, Binding, Zero, HeapIdx,
                                       HeapNameGV);
          Changed = true;
        }
      }
      F->eraseFromParent();
    }
  }

  if (!Samplers.empty()) {
    uint32_t Binding = getAndReserveFirstUnusedBinding(DescSet);
    // The Sampler handle type should be the same for all samplers
    // (target("spirv.Sampler")).
    [[maybe_unused]] TargetExtType *SamplerHandleType =
        cast<TargetExtType>(Samplers.front()->getReturnType());

    GlobalVariable *HeapNameGV =
        createHeapNameString(M, "SamplerDescriptorHeap");
    for (Function *F : Samplers) {
      assert(F->getReturnType() == SamplerHandleType &&
             "sampler handle type mismatch");
      for (User *U : make_early_inc_range(F->users())) {
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          Value *HeapIdx = CI->getArgOperand(0);
          replaceWithHandleFromBinding(M, CI, DescSet, Binding, Zero, HeapIdx,
                                       HeapNameGV);
          Changed = true;
        }
      }
      F->eraseFromParent();
    }
  }

  if (!Counters.empty()) {
    uint32_t Binding = getAndReserveFirstUnusedBinding(DescSet);
    [[maybe_unused]] Type *CounterHandleTy = Counters.front()->getReturnType();
    for (Function *F : Counters) {
      // The counter handle type should be the same for all resource types
      // that have a counter (target("spirv.VulkanBuffer", i32, 12, 1)).
      assert(F->getReturnType() == CounterHandleTy &&
             "counter handle type mismatch");
      for (User *U : make_early_inc_range(F->users())) {
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          replaceWithCounterHandleFromBinding(M, CI, DescSet, Binding);
          Changed = true;
        }
      }
      F->eraseFromParent();
    }
  }

  return Changed;
}

bool SPIRVLegalizeResourceBindingImpl::runOnModule(Module &M) {
  collectBindingInfo(M);

  bool Changed = false;
  if (MayHaveImplicitBindings)
    Changed |= replaceImplicitBindingCalls(M);
  if (MayHaveHeapBindings)
    Changed |= replaceHeapBindingCalls(M);

  return Changed;
}
} // namespace

PreservedAnalyses
SPIRVLegalizeResourceBindingPass::run(Module &M, ModuleAnalysisManager &AM) {
  return SPIRVLegalizeResourceBindingImpl().runOnModule(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

char SPIRVLegalizeResourceBindingLegacy::ID = 0;

INITIALIZE_PASS(SPIRVLegalizeResourceBindingLegacy,
                "legalize-spirv-resource-binding",
                "Legalize SPIR-V resource bindings", false, false)

ModulePass *llvm::createSPIRVLegalizeResourceBindingPass() {
  return new SPIRVLegalizeResourceBindingLegacy();
}
