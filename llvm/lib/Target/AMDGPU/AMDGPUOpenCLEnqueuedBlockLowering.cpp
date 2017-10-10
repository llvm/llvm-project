//===- AMDGPUOpenCLEnqueuedBlockLowering.cpp - Lower enqueued block -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This post-linking pass replaces the function pointer of enqueued
// block kernel with a global variable (runtime handle) and adds
// "runtime-handle" attribute to the enqueued block kernel.
//
// In LLVM CodeGen the runtime-handle metadata will be translated to
// RuntimeHandle metadata in code object. Runtime allocates a global buffer
// for each kernel with RuntimeHandel metadata and saves the kernel address
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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-lower-enqueued-block"

using namespace llvm;

namespace {

/// \brief Lower enqueued blocks.
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
  auto AS = AMDGPU::getAMDGPUAS(M);
  bool Changed = false;
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute("enqueued-block")) {
      if (!F.hasOneUse() || !F.user_begin()->hasOneUse() ||
          !isa<ConstantExpr>(*F.user_begin()) ||
          !isa<ConstantExpr>(*F.user_begin()->user_begin())) {
        continue;
      }
      auto *BitCast = cast<ConstantExpr>(*F.user_begin());
      auto *AddrCast = cast<ConstantExpr>(*BitCast->user_begin());
      auto RuntimeHandle = (F.getName() + "_runtime_handle").str();
      auto *GV = new GlobalVariable(
          M, Type::getInt8Ty(C)->getPointerTo(AS.GLOBAL_ADDRESS),
          /*IsConstant=*/true, GlobalValue::ExternalLinkage,
          /*Initializer=*/nullptr, RuntimeHandle, /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal, AS.GLOBAL_ADDRESS,
          /*IsExternallyInitialized=*/true);
      DEBUG(dbgs() << "runtime handle created: " << *GV << '\n');
      auto *NewPtr = ConstantExpr::getPointerCast(GV, AddrCast->getType());
      AddrCast->replaceAllUsesWith(NewPtr);
      F.addFnAttr("runtime-handle", RuntimeHandle);
      F.setLinkage(GlobalValue::ExternalLinkage);
      Changed = true;
    }
  }
  return Changed;
}
