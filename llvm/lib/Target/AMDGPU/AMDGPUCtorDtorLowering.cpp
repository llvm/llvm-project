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

#include "AMDGPUCtorDtorLowering.h"
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

static Function *createInitOrFiniKernelFunction(Module &M, bool IsCtor) {
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

static bool createInitOrFiniKernel(Module &M, StringRef GlobalName,
                                   bool IsCtor) {
  GlobalVariable *GV = M.getGlobalVariable(GlobalName);
  if (!GV || !GV->hasInitializer())
    return false;
  ConstantArray *GA = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!GA || GA->getNumOperands() == 0)
    return false;

  Function *InitOrFiniKernel = createInitOrFiniKernelFunction(M, IsCtor);
  IRBuilder<> IRB(InitOrFiniKernel->getEntryBlock().getTerminator());

  FunctionType *ConstructorTy = InitOrFiniKernel->getFunctionType();

  for (Value *V : GA->operands()) {
    auto *CS = cast<ConstantStruct>(V);
    IRB.CreateCall(ConstructorTy, CS->getOperand(1));
  }

  appendToUsed(M, {InitOrFiniKernel});

  GV->eraseFromParent();
  return true;
}

static bool lowerCtorsAndDtors(Module &M) {
  bool Modified = false;
  Modified |= createInitOrFiniKernel(M, "llvm.global_ctors", /*IsCtor =*/true);
  Modified |= createInitOrFiniKernel(M, "llvm.global_dtors", /*IsCtor =*/false);
  return Modified;
}

class AMDGPUCtorDtorLoweringLegacy final : public ModulePass {
public:
  static char ID;
  AMDGPUCtorDtorLoweringLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    return lowerCtorsAndDtors(M);
  }
};

} // End anonymous namespace

PreservedAnalyses AMDGPUCtorDtorLoweringPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  lowerCtorsAndDtors(M);
  return PreservedAnalyses::all();
}

char AMDGPUCtorDtorLoweringLegacy::ID = 0;
char &llvm::AMDGPUCtorDtorLoweringLegacyPassID =
    AMDGPUCtorDtorLoweringLegacy::ID;
INITIALIZE_PASS(AMDGPUCtorDtorLoweringLegacy, DEBUG_TYPE,
                "Lower ctors and dtors for AMDGPU", false, false)

ModulePass *llvm::createAMDGPUCtorDtorLoweringLegacyPass() {
  return new AMDGPUCtorDtorLoweringLegacy();
}
