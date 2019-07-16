//===-- AMDGPUPromotePointerKernArgsToGlobal.cpp - Promote pointer args ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Generic pointer kernel arguments need promoting to global ones.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-promote-pointer-kernargs"

namespace {

class AMDGPUPromotePointerKernArgsToGlobal : public FunctionPass {
public:
  static char ID;

  AMDGPUPromotePointerKernArgsToGlobal() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
};

} // End anonymous namespace

char AMDGPUPromotePointerKernArgsToGlobal::ID = 0;

INITIALIZE_PASS(AMDGPUPromotePointerKernArgsToGlobal, DEBUG_TYPE,
                "Lower intrinsics", false, false)

bool AMDGPUPromotePointerKernArgsToGlobal::runOnFunction(Function &F) {
  // Skip non-entry function.
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return false;

  auto &Entry = F.getEntryBlock();
  IRBuilder<> IRB(&Entry, Entry.begin());

  bool Changed = false;
  for (auto &Arg : F.args()) {
    auto PtrTy = dyn_cast<PointerType>(Arg.getType());
    if (!PtrTy || PtrTy->getPointerAddressSpace() != AMDGPUAS::FLAT_ADDRESS)
      continue;

    auto GlobalPtr =
        IRB.CreateAddrSpaceCast(&Arg,
                                PointerType::get(PtrTy->getPointerElementType(),
                                                 AMDGPUAS::GLOBAL_ADDRESS),
                                Arg.getName());
    auto NewFlatPtr = IRB.CreateAddrSpaceCast(GlobalPtr, PtrTy, Arg.getName());
    Arg.replaceAllUsesWith(NewFlatPtr);
    // Fix the global pointer itself.
    cast<Instruction>(GlobalPtr)->setOperand(0, &Arg);
    Changed = true;
  }

  return Changed;
}

FunctionPass *llvm::createAMDGPUPromotePointerKernArgsToGlobalPass() {
  return new AMDGPUPromotePointerKernArgsToGlobal();
}
