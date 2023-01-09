//===-- AMDGPULowerKernelCalls.cpp - Fix kernel calls ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
//
// \brief replace calls to OpenCL kernels with equivalent non-kernel
//        functions
//
// In OpenCL, a kernel may call another kernel as if it was a non-kernel
// function. However, kernels and functions have different ABI. To fix this,
// we copy the body of kernel A into a new non-kernel function fA, if we
// encounter a call to A. All calls to A are then transferred to fA.
//
//===----------------------------------------------------------------------===//
#include "AMDGPU.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {
class AMDGPULowerKernelCalls : public ModulePass {
public:
  static char ID;
  explicit AMDGPULowerKernelCalls();

  StringRef getPassName() const override {
    return "AMDGPU Lower Kernel Calls";
  }

private:
  bool runOnModule(Module &M) override;
};
} // end anonymous namespace

char AMDGPULowerKernelCalls::ID = 0;

namespace llvm {
void initializeAMDGPULowerKernelCallsPass(PassRegistry &);

ModulePass *createAMDGPULowerKernelCallsPass() {
  return new AMDGPULowerKernelCalls();
}
}

char &llvm::AMDGPULowerKernelCallsID = AMDGPULowerKernelCalls::ID;

INITIALIZE_PASS(
    AMDGPULowerKernelCalls, "amdgpu-lower-kernel-calls",
    "Lower calls to kernel functions into non-kernel function calls.", false,
    false)

AMDGPULowerKernelCalls::AMDGPULowerKernelCalls() : ModulePass(ID) {
  initializeAMDGPULowerKernelCallsPass(*PassRegistry::getPassRegistry());
}

static void setNameForBody(Function *FBody, const Function &FKernel) {
  StringRef Name = FKernel.getName();
  SmallString<128> NewName("__amdgpu_");
  NewName += Name;
  NewName += "_kernel_body";

  FBody->setName(NewName.str());
}

static Function *cloneKernel(Function &F) {
  ValueToValueMapTy ignored;
  Function *NewF = F.empty()
                       ? Function::Create(
                             F.getFunctionType(), Function::ExternalLinkage, "",
                             F.getParent())
                       : CloneFunction(&F, ignored);
  NewF->setCallingConv(CallingConv::C);
  // If we are copying a definition, we know there are no external references
  // and we can force internal linkage.
  if (!NewF->isDeclaration()) {
    NewF->setVisibility(GlobalValue::DefaultVisibility);
    NewF->setLinkage(GlobalValue::InternalLinkage);
  }
  setNameForBody(NewF, F);
  return NewF;
}

bool AMDGPULowerKernelCalls::runOnModule(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (CallingConv::AMDGPU_KERNEL != F.getCallingConv())
      continue;
    Function *FBody = nullptr;
    for (Use &U : make_early_inc_range(F.uses())) {
      CallBase *CI = dyn_cast<CallBase>(U.getUser());
      if (!CI || !CI->isCallee(&U))
        continue;
      if (!FBody)
        FBody = cloneKernel(F);
      CI->setCalledFunction(FBody);
      CI->setCallingConv(CallingConv::C);
      Changed = true;
    }
  }

  return Changed;
}
