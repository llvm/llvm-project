//===- AMDGPUOpenCLEnqueuedBlockLowering.cpp - Lower enqueued block -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This post-linking pass replaces the function pointer of enqueued
// block kernel with a global variable (runtime handle) and adds
// "runtime-handle" attribute to the enqueued block kernel.
//
// In LLVM CodeGen the runtime-handle metadata will be translated to
// RuntimeHandle metadata in code object. Runtime allocates a global buffer
// for each kernel with RuntimeHandle metadata and saves the kernel address
// required for the AQL packet into the buffer. __enqueue_kernel function
// in device library knows that the invoke function pointer in the block
// literal is actually runtime handle and loads the kernel address from it
// and put it into AQL packet for dispatching.
//
// This cannot be done in FE since FE cannot create a unique global variable
// with external linkage across LLVM modules. The global variable with internal
// linkage does not work since optimization passes will try to replace loads
// of the global variable with its initialization value.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-lower-enqueued-block"

using namespace llvm;

namespace {

/// Lower enqueued blocks.
class AMDGPUOpenCLEnqueuedBlockLowering : public ModulePass {
public:
  static char ID;

  explicit AMDGPUOpenCLEnqueuedBlockLowering() : ModulePass(ID) {}

private:
  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

char AMDGPUOpenCLEnqueuedBlockLowering::ID = 0;

char &llvm::AMDGPUOpenCLEnqueuedBlockLoweringID =
    AMDGPUOpenCLEnqueuedBlockLowering::ID;

INITIALIZE_PASS(AMDGPUOpenCLEnqueuedBlockLowering, DEBUG_TYPE,
                "Lower OpenCL enqueued blocks", false, false)

ModulePass* llvm::createAMDGPUOpenCLEnqueuedBlockLoweringPass() {
  return new AMDGPUOpenCLEnqueuedBlockLowering();
}

bool AMDGPUOpenCLEnqueuedBlockLowering::runOnModule(Module &M) {
  auto &C = M.getContext();
  bool Changed = false;
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute("enqueued-block")) {
      if (!F.hasName()) {
        SmallString<64> Name;
        Mangler::getNameWithPrefix(Name, "__amdgpu_enqueued_kernel",
                                   M.getDataLayout());
        F.setName(Name);
      }
      LLVM_DEBUG(dbgs() << "found enqueued kernel: " << F.getName() << '\n');
      auto RuntimeHandle = (F.getName() + ".runtime_handle").str();
      auto T = ArrayType::get(Type::getInt64Ty(C), 2);
      auto *GV = new GlobalVariable(
          M, T,
          /*isConstant=*/false, GlobalValue::ExternalLinkage,
          /*Initializer=*/Constant::getNullValue(T), RuntimeHandle,
          /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
          AMDGPUAS::GLOBAL_ADDRESS,
          /*isExternallyInitialized=*/false);
      LLVM_DEBUG(dbgs() << "runtime handle created: " << *GV << '\n');

      F.replaceAllUsesWith(ConstantExpr::getAddrSpaceCast(GV, F.getType()));
      F.addFnAttr("runtime-handle", RuntimeHandle);
      F.setLinkage(GlobalValue::ExternalLinkage);
      Changed = true;
    }
  }

  return Changed;
}
