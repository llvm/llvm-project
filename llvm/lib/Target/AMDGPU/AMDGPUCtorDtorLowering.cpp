//===-- AMDGPUCtorDtorLowering.cpp - Handle global ctors and dtors --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass creates a unified init and fini kernel with the required metadata
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-ctor-dtor"

namespace {
class AMDGPUCtorDtorLowering final : public ModulePass {
  bool runOnModule(Module &M) override;

public:
  Function *createInitOrFiniKernelFunction(Module &M, bool IsCtor) {
    StringRef InitOrFiniKernelName = "amdgcn.device.init";
    if (!IsCtor)
      InitOrFiniKernelName = "amdgcn.device.fini";

    Function *InitOrFiniKernel = Function::createWithDefaultAttr(
        FunctionType::get(Type::getVoidTy(M.getContext()), false),
        GlobalValue::ExternalLinkage, 0, InitOrFiniKernelName, &M);
    BasicBlock *InitOrFiniKernelBB =
        BasicBlock::Create(M.getContext(), "", InitOrFiniKernel);
    ReturnInst::Create(M.getContext(), InitOrFiniKernelBB);

    InitOrFiniKernel->setCallingConv(CallingConv::AMDGPU_KERNEL);
    if (IsCtor)
      InitOrFiniKernel->addFnAttr("device-init");
    else
      InitOrFiniKernel->addFnAttr("device-fini");
    return InitOrFiniKernel;
  }

  bool createInitOrFiniKernel(Module &M, GlobalVariable *GV, bool IsCtor) {
    if (!GV)
      return false;
    ConstantArray *GA = dyn_cast<ConstantArray>(GV->getInitializer());
    if (!GA || GA->getNumOperands() == 0)
      return false;
    Function *InitOrFiniKernel = createInitOrFiniKernelFunction(M, IsCtor);
    IRBuilder<> IRB(InitOrFiniKernel->getEntryBlock().getTerminator());
    for (Value *V : GA->operands()) {
      auto *CS = cast<ConstantStruct>(V);
      if (Function *F = dyn_cast<Function>(CS->getOperand(1))) {
        FunctionCallee Ctor =
            M.getOrInsertFunction(F->getName(), IRB.getVoidTy());
        IRB.CreateCall(Ctor);
      }
    }
    appendToUsed(M, {InitOrFiniKernel});
    return true;
  }

  static char ID;
  AMDGPUCtorDtorLowering() : ModulePass(ID) {}
};
} // End anonymous namespace

char AMDGPUCtorDtorLowering::ID = 0;
char &llvm::AMDGPUCtorDtorLoweringID = AMDGPUCtorDtorLowering::ID;
INITIALIZE_PASS(AMDGPUCtorDtorLowering, DEBUG_TYPE,
                "Lower ctors and dtors for AMDGPU", false, false)

ModulePass *llvm::createAMDGPUCtorDtorLoweringPass() {
  return new AMDGPUCtorDtorLowering();
}

bool AMDGPUCtorDtorLowering::runOnModule(Module &M) {
  bool Modified = false;
  Modified |=
      createInitOrFiniKernel(M, M.getGlobalVariable("llvm.global_ctors"),
                             /*IsCtor =*/true);
  Modified |=
      createInitOrFiniKernel(M, M.getGlobalVariable("llvm.global_dtors"),
                             /*IsCtor =*/false);
  return Modified;
}
