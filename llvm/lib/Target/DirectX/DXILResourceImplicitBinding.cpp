//===- DXILResourceImplicitBinding.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILResourceImplicitBinding.h"
#include "DirectX.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include <cstdint>

#define DEBUG_TYPE "dxil-resource-implicit-binding"

using namespace llvm;
using namespace llvm::dxil;

namespace {

static void diagnoseImplicitBindingNotFound(CallInst *ImplBindingCall) {
  Function *F = ImplBindingCall->getFunction();
  LLVMContext &Context = F->getParent()->getContext();
  // FIXME: include the name of the resource in the error message
  // (llvm/llvm-project#137868)
  Context.diagnose(
      DiagnosticInfoGenericWithLoc("resource cannot be allocated", *F,
                                   ImplBindingCall->getDebugLoc(), DS_Error));
}

static bool assignBindings(Module &M, DXILResourceBindingInfo &DRBI,
                           DXILResourceTypeMap &DRTM) {
  struct ImplicitBindingCall {
    int OrderID;
    CallInst *Call;
    ImplicitBindingCall(int OrderID, CallInst *Call)
        : OrderID(OrderID), Call(Call) {}
  };
  SmallVector<ImplicitBindingCall> Calls;
  SmallVector<Function *> FunctionsToMaybeRemove;

  // collect all of the llvm.dx.resource.handlefromImplicitbinding calls
  for (Function &F : M.functions()) {
    if (!F.isDeclaration())
      continue;

    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefromimplicitbinding)
      continue;

    for (User *U : F.users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        int OrderID = cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
        Calls.emplace_back(OrderID, CI);
      }
    }
    FunctionsToMaybeRemove.emplace_back(&F);
  }

  // sort all the collected implicit bindings by OrderID
  llvm::stable_sort(
      Calls, [](auto &LHS, auto &RHS) { return LHS.OrderID < RHS.OrderID; });

  // iterate over sorted calls, find binding for each new OrderID and replace
  // each call with dx_resource_handlefrombinding using the new binding
  int LastOrderID = -1;
  llvm::TargetExtType *HandleTy = nullptr;
  ConstantInt *RegSlotOp = nullptr;
  bool AllBindingsAssigned = true;
  bool Changed = false;

  for (ImplicitBindingCall &IB : Calls) {
    IRBuilder<> Builder(IB.Call);

    if (IB.OrderID != LastOrderID) {
      LastOrderID = IB.OrderID;
      HandleTy = cast<TargetExtType>(IB.Call->getType());
      ResourceTypeInfo &RTI = DRTM[HandleTy];

      uint32_t Space =
          cast<ConstantInt>(IB.Call->getArgOperand(1))->getZExtValue();
      int32_t Size =
          cast<ConstantInt>(IB.Call->getArgOperand(2))->getZExtValue();

      std::optional<uint32_t> RegSlot =
          DRBI.findAvailableBinding(RTI.getResourceClass(), Space, Size);
      if (!RegSlot) {
        diagnoseImplicitBindingNotFound(IB.Call);
        AllBindingsAssigned = false;
        continue;
      }
      RegSlotOp = ConstantInt::get(Builder.getInt32Ty(), RegSlot.value());
    }

    if (!RegSlotOp)
      continue;

    auto *NewCall = Builder.CreateIntrinsic(
        HandleTy, Intrinsic::dx_resource_handlefrombinding,
        {IB.Call->getOperand(1),   /* space */
         RegSlotOp,                /* register slot */
         IB.Call->getOperand(2),   /* size */
         IB.Call->getOperand(3),   /* index */
         IB.Call->getOperand(4)}); /* name */
    IB.Call->replaceAllUsesWith(NewCall);
    IB.Call->eraseFromParent();
    Changed = true;
  }

  for (Function *F : FunctionsToMaybeRemove) {
    if (F->user_empty()) {
      F->eraseFromParent();
      Changed = true;
    }
  }

  DRBI.setHasImplicitBinding(!AllBindingsAssigned);
  return Changed;
}

} // end anonymous namespace

PreservedAnalyses DXILResourceImplicitBinding::run(Module &M,
                                                   ModuleAnalysisManager &AM) {

  DXILResourceBindingInfo &DRBI = AM.getResult<DXILResourceBindingAnalysis>(M);
  DXILResourceTypeMap &DRTM = AM.getResult<DXILResourceTypeAnalysis>(M);

  if (!DRBI.hasImplicitBinding())
    return PreservedAnalyses::all();

  if (!assignBindings(M, DRBI, DRTM))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DXILResourceBindingAnalysis>();
  PA.preserve<DXILResourceTypeAnalysis>();
  return PA;
}

namespace {

class DXILResourceImplicitBindingLegacy : public ModulePass {
public:
  DXILResourceImplicitBindingLegacy() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    DXILResourceTypeMap &DRTM =
        getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
    DXILResourceBindingInfo &DRBI =
        getAnalysis<DXILResourceBindingWrapperPass>().getBindingInfo();

    if (DRBI.hasImplicitBinding())
      return assignBindings(M, DRBI, DRTM);
    return false;
  }

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addRequired<DXILResourceBindingWrapperPass>();
    AU.addPreserved<DXILResourceTypeWrapperPass>();
    AU.addPreserved<DXILResourceBindingWrapperPass>();
  }
};

char DXILResourceImplicitBindingLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILResourceImplicitBindingLegacy, DEBUG_TYPE,
                      "DXIL Resource Implicit Binding", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceBindingWrapperPass)
INITIALIZE_PASS_END(DXILResourceImplicitBindingLegacy, DEBUG_TYPE,
                    "DXIL Resource Implicit Binding", false, false)

ModulePass *llvm::createDXILResourceImplicitBindingLegacyPass() {
  return new DXILResourceImplicitBindingLegacy();
}
