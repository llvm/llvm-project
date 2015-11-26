//===-- AMDGPUAnnotateKernelFeaturesPass.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This pass adds target attributes to functions which use intrinsics
/// which will impact calling convention lowering.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "amdgpu-annotate-kernel-features"

using namespace llvm;

namespace {

class AMDGPUAnnotateKernelFeatures : public ModulePass {
private:
  void addAttrToCallers(Function *Intrin, StringRef AttrName);
  bool addAttrsForIntrinsics(Module &M, ArrayRef<StringRef[2]>);

public:
  static char ID;

  AMDGPUAnnotateKernelFeatures() : ModulePass(ID) { }
  bool runOnModule(Module &M) override;
  const char *getPassName() const override {
    return "AMDGPU Annotate Kernel Features";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }
};

}

char AMDGPUAnnotateKernelFeatures::ID = 0;

char &llvm::AMDGPUAnnotateKernelFeaturesID = AMDGPUAnnotateKernelFeatures::ID;


INITIALIZE_PASS_BEGIN(AMDGPUAnnotateKernelFeatures, DEBUG_TYPE,
                      "Add AMDGPU function attributes", false, false)
INITIALIZE_PASS_END(AMDGPUAnnotateKernelFeatures, DEBUG_TYPE,
                    "Add AMDGPU function attributes", false, false)


void AMDGPUAnnotateKernelFeatures::addAttrToCallers(Function *Intrin,
                                                    StringRef AttrName) {
  SmallPtrSet<Function *, 4> SeenFuncs;

  for (User *U : Intrin->users()) {
    // CallInst is the only valid user for an intrinsic.
    CallInst *CI = cast<CallInst>(U);

    Function *CallingFunction = CI->getParent()->getParent();
    if (SeenFuncs.insert(CallingFunction).second)
      CallingFunction->addFnAttr(AttrName);
  }
}

bool AMDGPUAnnotateKernelFeatures::addAttrsForIntrinsics(
  Module &M,
  ArrayRef<StringRef[2]> IntrinsicToAttr) {
  bool Changed = false;

  for (const StringRef *Arr  : IntrinsicToAttr) {
    if (Function *Fn = M.getFunction(Arr[0])) {
      addAttrToCallers(Fn, Arr[1]);
      Changed = true;
    }
  }

  return Changed;
}

bool AMDGPUAnnotateKernelFeatures::runOnModule(Module &M) {
  Triple TT(M.getTargetTriple());

  static const StringRef IntrinsicToAttr[][2] = {
    // .x omitted
    { "llvm.r600.read.tgid.y", "amdgpu-work-group-id-y" },
    { "llvm.r600.read.tgid.z", "amdgpu-work-group-id-z" },

    // .x omitted
    { "llvm.r600.read.tidig.y", "amdgpu-work-item-id-y" },
    { "llvm.r600.read.tidig.z", "amdgpu-work-item-id-z" }

  };

  static const StringRef HSAIntrinsicToAttr[][2] = {
    { "llvm.r600.read.local.size.x", "amdgpu-dispatch-ptr" },
    { "llvm.r600.read.local.size.y", "amdgpu-dispatch-ptr" },
    { "llvm.r600.read.local.size.z", "amdgpu-dispatch-ptr" },

    { "llvm.r600.read.global.size.x", "amdgpu-dispatch-ptr" },
    { "llvm.r600.read.global.size.y", "amdgpu-dispatch-ptr" },
    { "llvm.r600.read.global.size.z", "amdgpu-dispatch-ptr" },
    { "llvm.amdgcn.dispatch.ptr",     "amdgpu-dispatch-ptr" }
  };

  // TODO: Intrinsics that require queue ptr.

  // We do not need to note the x workitem or workgroup id because they are
  // always initialized.

  bool Changed = addAttrsForIntrinsics(M, IntrinsicToAttr);
  if (TT.getOS() == Triple::AMDHSA)
    Changed |= addAttrsForIntrinsics(M, HSAIntrinsicToAttr);

  return Changed;
}

ModulePass *llvm::createAMDGPUAnnotateKernelFeaturesPass() {
  return new AMDGPUAnnotateKernelFeatures();
}
